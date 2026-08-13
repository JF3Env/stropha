"""Unit tests for the Phase-5 pytest gate (WIN-197).

Coverage targets:

- ``PytestGate`` happy path (mocked subprocess) and every failure path
  (timeout, OSError-on-spawn, missing uv binary, no-artifacts skip).
- ``GateResult`` serialisation.
- ``_tail`` byte / None / over-size edge cases.
- ``_stropha_project_root`` resolves to the repo root.
- ``ArtifactStore.write_gate_result`` writes the canonical
  ``gate.json`` side-car and accepts either a dict or any object with
  a ``to_dict()`` method.
- ``run_loop`` integration: a gate that returns ``passed=False``
  rejects the candidate before scoring (no ScoreCard, gate.json
  persisted, LoopError surfaced in the result.errors list).
- ``run_loop`` integration: a gate that returns ``passed=True`` lets
  the evaluator run as before. Same for the seed-init path.
- Calling ``run_loop`` with no ``gate`` keyword preserves the previous
  behaviour exactly (regression guard for existing callers).

All subprocess.run calls are mocked — we never actually spawn pytest.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from stropha.meta import (
    DEFAULT_PYTEST_ARGS,
    DEFAULT_TIMEOUT_S,
    ArtifactStore,
    ArtifactStoreError,
    GateError,
    GateResult,
    LoopResult,
    PytestGate,
    ScoreCard,
    run_loop,
)
from stropha.meta.gate import _stropha_project_root, _tail
from stropha.meta.sandbox import SandboxResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_candidate_with_artifact(store: ArtifactStore) -> Path:
    """Register a candidate that carries one code artifact + return its dir."""
    record = store.register_candidate(
        config={
            "walker": "git-ls-files",
            "reranker": "proposed",
            "__code_artifacts__": {
                "adapters/reranker/proposed.py": _valid_source(),
            },
        },
    )
    assert record.base_dir is not None
    return record.base_dir


def _valid_source() -> str:
    return (
        "from pydantic import BaseModel\n"
        "from stropha.models import SearchHit\n"
        "from stropha.pipeline.base import StageHealth\n"
        "from stropha.pipeline.registry import register_adapter\n"
        "class Config(BaseModel): pass\n"
        "@register_adapter(stage='reranker', name='proposed')\n"
        "class Proposed:\n"
        "    Config = Config\n"
        "    def __init__(self, config=None): self.config = config or Config()\n"
        "    stage_name = property(lambda self: 'reranker')\n"
        "    adapter_name = property(lambda self: 'proposed')\n"
        "    adapter_id = property(lambda self: 'reranker:proposed')\n"
        "    config_schema = property(lambda self: Config)\n"
        "    def health(self): return StageHealth(status='ready', message='ok')\n"
        "    def rerank(self, query: str, hits: list[SearchHit], *, top_k=10):\n"
        "        return hits[:top_k]\n"
    )


class _PassingSandbox:
    def run(self, candidate_dir: Path, *, job: str) -> SandboxResult:
        return SandboxResult(succeeded=True, returncode=0, payload={"ok": True})


def _write_candidate(tmp_path: Path) -> None:
    (tmp_path / "adapters" / "reranker").mkdir(parents=True)
    (tmp_path / "adapters" / "reranker" / "proposed.py").write_text(_valid_source())
    (tmp_path / "config.yaml").write_text("reranker: proposed\n")


def _mk_candidate_no_artifact(store: ArtifactStore) -> Path:
    record = store.register_candidate(config={"walker": "filesystem"})
    assert record.base_dir is not None
    return record.base_dir


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    """Build a fake ``subprocess.CompletedProcess`` for monkey-patching."""
    return subprocess.CompletedProcess(
        args=["uv", "run", "pytest"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


class TestTailHelper:
    def test_none_returns_empty(self) -> None:
        assert _tail(None) == ""

    def test_bytes_decoded(self) -> None:
        assert _tail(b"abc") == "abc"

    def test_string_under_limit_passthrough(self) -> None:
        assert _tail("short") == "short"

    def test_string_over_limit_truncated(self) -> None:
        body = "X" * 10_000
        out = _tail(body)
        # Last 4_000 bytes only.
        assert len(out) == 4_000
        assert out == "X" * 4_000

    def test_bytes_with_invalid_utf8(self) -> None:
        # Use lossy decoding so a bad byte never propagates.
        out = _tail(b"\xff\xfeok")
        # Replacement chars + the readable tail.
        assert "ok" in out


class TestProjectRoot:
    def test_resolves_to_repo_root(self) -> None:
        root = _stropha_project_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "src" / "stropha").is_dir()
        assert (root / "tests").is_dir()


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_to_dict_roundtrip(self) -> None:
        r = GateResult(
            passed=True,
            skipped=False,
            returncode=0,
            duration_s=1.234,
            command=["uv", "run", "pytest"],
            stdout_tail="hello",
            stderr_tail="",
        )
        d = r.to_dict()
        assert d["passed"] is True
        assert d["skipped"] is False
        assert d["returncode"] == 0
        assert d["duration_s"] == 1.234
        assert d["command"] == ["uv", "run", "pytest"]
        assert d["stdout_tail"] == "hello"
        assert d["stderr_tail"] == ""
        assert d["error"] is None

    def test_defaults_safe(self) -> None:
        r = GateResult(passed=False)
        d = r.to_dict()
        assert d["command"] == []
        assert d["duration_s"] == 0.0
        assert d["error"] is None


# ---------------------------------------------------------------------------
# PytestGate — primitive
# ---------------------------------------------------------------------------


class TestPytestGateSkipping:
    def test_no_adapters_dir_skips(self, tmp_path: Path) -> None:
        gate = PytestGate(uv_binary="/usr/bin/uv")
        result = gate(tmp_path / "nonexistent")
        assert result.passed is True
        assert result.skipped is True
        assert result.returncode == 0

    def test_empty_adapters_dir_skips(self, tmp_path: Path) -> None:
        (tmp_path / "adapters").mkdir()
        gate = PytestGate(uv_binary="/usr/bin/uv")
        result = gate(tmp_path)
        assert result.skipped is True

    def test_adapters_with_non_py_skips(self, tmp_path: Path) -> None:
        (tmp_path / "adapters").mkdir()
        (tmp_path / "adapters" / "README.md").write_text("hi")
        gate = PytestGate(uv_binary="/usr/bin/uv")
        result = gate(tmp_path)
        assert result.skipped is True


class TestPytestGateExecution:
    def test_missing_uv_raises_gate_error(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with patch("shutil.which", return_value=None):
            gate = PytestGate(uv_binary=None, sandbox=_PassingSandbox())  # type: ignore[arg-type]
            with pytest.raises(GateError, match="uv binary"):
                gate(tmp_path)

    def test_returncode_zero_passes(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with patch(
            "subprocess.run",
            return_value=_completed(0, stdout="all ok"),
        ) as run_mock:
            gate = PytestGate(
                uv_binary="/fake/uv", project_root=tmp_path, sandbox=_PassingSandbox()  # type: ignore[arg-type]
            )
            result = gate(tmp_path)
        assert result.passed is True
        assert result.skipped is False
        assert result.returncode == 0
        assert result.stdout_tail == "all ok"
        assert run_mock.called
        # Default args are -q + --no-header.
        called_cmd = run_mock.call_args.args[0]
        assert called_cmd[0] == "/fake/uv"
        assert called_cmd[1:3] == ["run", "pytest"]
        assert tuple(called_cmd[3:]) == DEFAULT_PYTEST_ARGS

    def test_returncode_nonzero_fails(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with patch(
            "subprocess.run",
            return_value=_completed(1, stdout="test failed"),
        ):
            gate = PytestGate(uv_binary="/fake/uv", sandbox=_PassingSandbox())  # type: ignore[arg-type]
            result = gate(tmp_path)
        assert result.passed is False
        assert result.returncode == 1
        assert result.stdout_tail == "test failed"

    def test_timeout_returns_failed_result(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)

        exc = subprocess.TimeoutExpired(
            cmd=["uv", "run", "pytest"],
            timeout=0.1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with patch("subprocess.run", side_effect=exc):
            gate = PytestGate(
                uv_binary="/fake/uv", timeout_s=0.1, sandbox=_PassingSandbox()  # type: ignore[arg-type]
            )
            result = gate(tmp_path)
        assert result.passed is False
        assert result.returncode == -1
        assert result.error is not None and "timeout" in result.error
        assert result.stdout_tail == "partial stdout"
        assert result.stderr_tail == "partial stderr"

    def test_os_error_raises_gate_error(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with patch("subprocess.run", side_effect=OSError("denied")):
            gate = PytestGate(uv_binary="/fake/uv", sandbox=_PassingSandbox())  # type: ignore[arg-type]
            with pytest.raises(GateError, match="spawn"):
                gate(tmp_path)

    def test_custom_pytest_args_passed_through(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with patch("subprocess.run", return_value=_completed(0)) as run_mock:
            gate = PytestGate(
                uv_binary="/fake/uv",
                pytest_args=("-x", "tests/unit/"),
                sandbox=_PassingSandbox(),  # type: ignore[arg-type]
            )
            gate(tmp_path)
        called_cmd = run_mock.call_args.args[0]
        assert tuple(called_cmd[3:]) == ("-x", "tests/unit/")

    def test_project_root_used_as_cwd(self, tmp_path: Path) -> None:
        cand = tmp_path / "candidate"
        cand.mkdir()
        _write_candidate(cand)
        with patch("subprocess.run", return_value=_completed(0)) as run_mock:
            gate = PytestGate(
                uv_binary="/fake/uv", project_root=tmp_path, sandbox=_PassingSandbox()  # type: ignore[arg-type]
            )
            gate(cand)
        assert run_mock.call_args.kwargs["cwd"] == str(tmp_path)

    def test_uv_binary_resolved_via_which(self, tmp_path: Path) -> None:
        _write_candidate(tmp_path)
        with (
            patch("shutil.which", return_value="/usr/local/bin/uv") as which,
            patch("subprocess.run", return_value=_completed(0)) as run_mock,
        ):
            gate = PytestGate(uv_binary=None, sandbox=_PassingSandbox())  # type: ignore[arg-type]
            gate(tmp_path)
        which.assert_called_with("uv")
        assert run_mock.call_args.args[0][0] == "/usr/local/bin/uv"

    def test_default_timeout_constant(self) -> None:
        gate = PytestGate()
        assert gate.timeout_s == DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# ArtifactStore.write_gate_result
# ---------------------------------------------------------------------------


class TestStoreWriteGateResult:
    def test_writes_json_from_gate_result(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(config={"walker": "filesystem"})
        gr = GateResult(passed=True, returncode=0, duration_s=1.0)
        store.write_gate_result(record.id, gr)
        gate_json = (
            tmp_path / "meta" / "candidates" / record.id / "gate.json"
        )
        data = json.loads(gate_json.read_text())
        assert data["passed"] is True
        assert data["duration_s"] == 1.0

    def test_accepts_dict(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(config={"walker": "filesystem"})
        payload = {"passed": False, "returncode": 2, "duration_s": 5.0}
        store.write_gate_result(record.id, payload)
        gate_json = (
            tmp_path / "meta" / "candidates" / record.id / "gate.json"
        )
        assert json.loads(gate_json.read_text())["returncode"] == 2

    def test_missing_candidate_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError, match="candidate"):
            store.write_gate_result(
                "nope", {"passed": True, "returncode": 0},
            )

    def test_unserialisable_payload_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(config={"walker": "filesystem"})
        with pytest.raises(ArtifactStoreError, match="gate"):
            store.write_gate_result(
                record.id, {"passed": True, "obj": object()},
            )


# ---------------------------------------------------------------------------
# run_loop integration
# ---------------------------------------------------------------------------


def _good_evaluator(config: dict[str, Any]) -> ScoreCard:
    return ScoreCard(quality=0.7, context_efficiency=0.5, latency_ms=100.0)


class _RaisingEvaluator:
    """Asserts it is never called (used when gate is expected to reject)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, config: dict[str, Any]) -> ScoreCard:
        self.calls += 1
        raise AssertionError(
            "evaluator must not run after a gate failure"
        )


class TestLoopWithGate:
    def test_failing_gate_blocks_scoring(self, tmp_path: Path) -> None:
        """Gate returns passed=False → no ScoreCard, candidate not in archive."""
        store = ArtifactStore(tmp_path / "meta")
        eval_spy = _RaisingEvaluator()

        def always_proposed(archive: Any) -> dict[str, Any]:
            return {"walker": "filesystem"}

        def failing_gate(candidate_dir: Path) -> GateResult:
            return GateResult(passed=False, returncode=1, duration_s=0.1)

        result = run_loop(
            proposer=always_proposed,
            evaluator=eval_spy,
            store=store,
            max_iterations=2,
            gate=failing_gate,
        )
        assert eval_spy.calls == 0
        assert result.archive == []
        assert any("rejected by pytest gate" in e for e in result.errors)
        # Every candidate must have a gate.json on disk.
        for cand_dir in (tmp_path / "meta" / "candidates").iterdir():
            assert (cand_dir / "gate.json").is_file()
            assert (cand_dir / "scores.json").read_text().strip() == "{}"

    def test_passing_gate_lets_evaluator_run(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")

        def propose(archive: Any) -> dict[str, Any]:
            return {"walker": "git-ls-files"}

        def passing_gate(candidate_dir: Path) -> GateResult:
            return GateResult(passed=True, skipped=True)

        result: LoopResult = run_loop(
            proposer=propose,
            evaluator=_good_evaluator,
            store=store,
            max_iterations=1,
            gate=passing_gate,
        )
        assert len(result.archive) == 1
        record, card = result.archive[0]
        assert card.quality == 0.7
        assert (record.base_dir / "gate.json").is_file()  # type: ignore[union-attr]

    def test_seed_init_respects_gate(self, tmp_path: Path) -> None:
        """A gate failure during seed init produces an error entry."""
        store = ArtifactStore(tmp_path / "meta")

        calls = {"n": 0}

        def gate(candidate_dir: Path) -> GateResult:
            calls["n"] += 1
            return GateResult(passed=False)

        result = run_loop(
            proposer=lambda a: {"walker": "filesystem"},
            evaluator=_good_evaluator,
            store=store,
            seeds=[{"walker": "git-ls-files"}],
            max_iterations=1,  # seed counts toward the budget
            gate=gate,
        )
        assert calls["n"] == 1
        assert result.archive == []
        assert any("seed 0:" in e for e in result.errors)

    def test_no_gate_keyword_preserves_old_behaviour(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        result = run_loop(
            proposer=lambda a: {"walker": "filesystem"},
            evaluator=_good_evaluator,
            store=store,
            max_iterations=1,
        )
        assert len(result.archive) == 1
        # No gate.json since the gate never ran.
        record, _ = result.archive[0]
        assert not (record.base_dir / "gate.json").exists()  # type: ignore[union-attr]

    def test_gate_raising_surfaces_as_loop_error(self, tmp_path: Path) -> None:
        """When the gate ITSELF blows up (not just returns False), the
        loop records the error and skips the candidate without scoring."""
        store = ArtifactStore(tmp_path / "meta")

        def boom(candidate_dir: Path) -> GateResult:
            raise GateError("uv missing")

        result = run_loop(
            proposer=lambda a: {"walker": "filesystem"},
            evaluator=_good_evaluator,
            store=store,
            max_iterations=1,
            gate=boom,
        )
        assert result.archive == []
        assert any("gate raised" in e for e in result.errors)

    def test_gate_persist_failure_does_not_block_decision(
        self, tmp_path: Path,
    ) -> None:
        """If writing gate.json fails, the gate's verdict still applies."""
        store = ArtifactStore(tmp_path / "meta")

        def passing(candidate_dir: Path) -> GateResult:
            return GateResult(passed=True, skipped=True)

        # Sabotage write_gate_result so persistence fails. The decision
        # (passed=True) must still propagate; the loop logs a warning
        # but keeps going.
        with patch.object(
            ArtifactStore,
            "write_gate_result",
            side_effect=ArtifactStoreError("disk full"),
        ):
            result = run_loop(
                proposer=lambda a: {"walker": "filesystem"},
                evaluator=_good_evaluator,
                store=store,
                max_iterations=1,
                gate=passing,
            )
        assert len(result.archive) == 1  # evaluator still ran


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_meta_package_reexports(self) -> None:
        import stropha.meta as m
        for name in (
            "PytestGate",
            "GateResult",
            "GateError",
            "GateFn",
            "DEFAULT_PYTEST_ARGS",
            "DEFAULT_TIMEOUT_S",
        ):
            assert hasattr(m, name), f"missing re-export: {name}"
