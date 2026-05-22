"""Unit tests for the Phase-5 subprocess sandbox (WIN-198).

Coverage targets:

- Module helpers: ``_tail_str`` (None / bytes / passthrough / truncation),
  ``_stropha_project_root``, ``_parse_payload`` (valid dict / non-dict /
  malformed / empty).
- ``SubprocessSandbox.run``: no-adapters skip; subprocess returncode
  0 → succeeded + payload parsed; non-zero → not succeeded; negative
  returncode → ``crashed=True``; ``TimeoutExpired`` → ``timed_out=True``;
  ``OSError`` on spawn → ``SandboxError``; PYTHONPATH prepended when
  set / from scratch when not; extra_env merged; non-JSON stdout
  falls back to ``{"raw": ...}``; non-dict JSON → same fallback.
- ``SandboxResult`` to_dict roundtrip + safe defaults.
- ``_sandbox_runner.main``: missing args → exit 2; unknown job → exit 2;
  ``import_check`` happy path → exit 0 + ok=True + imported list;
  ``import_check`` syntax error → exit 1 + failed list populated;
  ``import_check`` import-time exception → captured in failed list;
  ``import_check`` no-adapters-dir → reason populated, exit 0;
  ``import_check`` SystemExit at module level → captured (BaseException
  catch path).
- End-to-end: ``SubprocessSandbox`` driving the real runner in a real
  subprocess imports a candidate file cleanly.
- End-to-end: the parent process never imports the candidate's
  module — its ``sys.modules`` is untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from stropha.meta import (
    RUNNER_MODULE,
    SANDBOX_DEFAULT_TIMEOUT_S,
    SandboxError,
    SandboxResult,
    SubprocessSandbox,
    _sandbox_runner,
)
from stropha.meta.sandbox import (
    _parse_payload,
    _stropha_project_root,
    _tail_str,
)

VALID_ADAPTER_SRC = (
    "from stropha.pipeline.registry import register_adapter\n"
    "@register_adapter(stage='enricher', name='sandboxed')\n"
    "class Sandboxed:\n"
    "    class Config: ...\n"
)

BROKEN_ADAPTER_SRC = "this is not valid python def\n"

CRASHING_ADAPTER_SRC = "raise RuntimeError('boom at import time')\n"

SYSEXIT_ADAPTER_SRC = "import sys\nsys.exit(7)\n"


def _make_candidate_with(tmp_path: Path, *artifacts: tuple[str, str]) -> Path:
    """Create a fake candidate dir with the given (rel_path, source) pairs."""
    cand = tmp_path / "candidate"
    (cand / "adapters").mkdir(parents=True)
    for rel, src in artifacts:
        target = cand / "adapters" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return cand


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(
        args=["python", "-m", RUNNER_MODULE],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


class TestTailStr:
    def test_none_returns_empty(self) -> None:
        assert _tail_str(None, 10) == ""

    def test_bytes_decoded(self) -> None:
        assert _tail_str(b"hello", 10) == "hello"

    def test_passthrough_under_limit(self) -> None:
        assert _tail_str("short", 100) == "short"

    def test_truncates_over_limit(self) -> None:
        out = _tail_str("X" * 200, 50)
        assert out == "X" * 50

    def test_lossy_decode_for_bad_utf8(self) -> None:
        assert "ok" in _tail_str(b"\xff\xfeok", 100)


class TestParsePayload:
    def test_empty(self) -> None:
        assert _parse_payload("") == {}
        assert _parse_payload(None) == {}

    def test_valid_dict(self) -> None:
        assert _parse_payload('{"a": 1}') == {"a": 1}

    def test_non_dict_json_falls_back(self) -> None:
        out = _parse_payload("[1, 2, 3]")
        assert out == {"raw": "[1, 2, 3]"}

    def test_malformed_falls_back(self) -> None:
        out = _parse_payload("not json")
        assert out == {"raw": "not json"}


class TestProjectRoot:
    def test_resolves_to_repo_root(self) -> None:
        root = _stropha_project_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "src" / "stropha" / "meta" / "sandbox.py").is_file()


# ---------------------------------------------------------------------------
# SandboxResult
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_to_dict_roundtrip(self) -> None:
        r = SandboxResult(
            succeeded=True,
            returncode=0,
            duration_s=0.5,
            command=["python", "-m", "x"],
            stdout="hello",
            stderr_tail="warn",
            payload={"ok": True},
        )
        d = r.to_dict()
        assert d["succeeded"] is True
        assert d["returncode"] == 0
        assert d["duration_s"] == 0.5
        assert d["command"] == ["python", "-m", "x"]
        assert d["stdout"] == "hello"
        assert d["stderr_tail"] == "warn"
        assert d["payload"] == {"ok": True}
        assert d["timed_out"] is False
        assert d["crashed"] is False
        assert d["error"] is None

    def test_defaults_safe(self) -> None:
        r = SandboxResult(succeeded=False, returncode=1)
        d = r.to_dict()
        assert d["command"] == []
        assert d["payload"] == {}
        assert d["error"] is None


# ---------------------------------------------------------------------------
# SubprocessSandbox (mocked subprocess)
# ---------------------------------------------------------------------------


class TestSandboxSkipping:
    def test_no_adapters_dir(self, tmp_path: Path) -> None:
        sandbox = SubprocessSandbox()
        result = sandbox.run(tmp_path / "no_such_candidate")
        assert result.succeeded is True
        assert result.payload["skipped"] is True
        assert result.payload["reason"] == "no_adapters_dir"


class TestSandboxExecution:
    def test_returncode_zero_succeeds(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch(
            "subprocess.run",
            return_value=_completed(0, stdout='{"ok": true, "imported": ["foo.py"]}'),
        ) as run_mock:
            sandbox = SubprocessSandbox()
            result = sandbox.run(cand)
        assert result.succeeded is True
        assert result.returncode == 0
        assert result.payload == {"ok": True, "imported": ["foo.py"]}
        # Command shape — python -m runner job candidate_dir
        cmd = run_mock.call_args.args[0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", RUNNER_MODULE]
        assert cmd[3] == "import_check"
        assert cmd[4] == str(cand)

    def test_returncode_nonzero_fails(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", BROKEN_ADAPTER_SRC))
        with patch(
            "subprocess.run",
            return_value=_completed(1, stdout='{"ok": false, "failed": ["foo.py"]}'),
        ):
            sandbox = SubprocessSandbox()
            result = sandbox.run(cand)
        assert result.succeeded is False
        assert result.returncode == 1
        assert result.crashed is False
        assert result.payload["ok"] is False

    def test_crashed_when_negative_returncode(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch(
            "subprocess.run",
            return_value=_completed(-11, stdout="", stderr="Segmentation fault"),
        ):
            sandbox = SubprocessSandbox()
            result = sandbox.run(cand)
        assert result.succeeded is False
        assert result.crashed is True
        assert result.returncode == -11
        assert "Segmentation" in result.stderr_tail

    def test_timeout(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        exc = subprocess.TimeoutExpired(
            cmd=["python", "-m", RUNNER_MODULE],
            timeout=0.1,
            output=b"part-stdout",
            stderr=b"part-stderr",
        )
        with patch("subprocess.run", side_effect=exc):
            sandbox = SubprocessSandbox(timeout_s=0.1)
            result = sandbox.run(cand)
        assert result.timed_out is True
        assert result.succeeded is False
        assert result.returncode == -1
        assert "timeout" in (result.error or "")
        assert result.stdout == "part-stdout"
        assert result.stderr_tail == "part-stderr"

    def test_os_error_raises_sandbox_error(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", side_effect=OSError("denied")):
            sandbox = SubprocessSandbox()
            with pytest.raises(SandboxError, match="spawn"):
                sandbox.run(cand)

    def test_pythonpath_prepended_to_existing(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", return_value=_completed(0, stdout="{}")) as rm:
            sandbox = SubprocessSandbox(extra_env={"PYTHONPATH": "/preexisting"})
            sandbox.run(cand)
        env = rm.call_args.kwargs["env"]
        # The adapters dir is the first PATH segment.
        adapters_dir = str(cand / "adapters")
        assert env["PYTHONPATH"].startswith(adapters_dir)
        assert "/preexisting" in env["PYTHONPATH"]

    def test_pythonpath_set_when_absent(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", return_value=_completed(0, stdout="{}")) as rm:
            sandbox = SubprocessSandbox()
            sandbox.run(cand)
        env = rm.call_args.kwargs["env"]
        assert env["PYTHONPATH"] == str(cand / "adapters")

    def test_extra_env_merged(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", return_value=_completed(0, stdout="{}")) as rm:
            sandbox = SubprocessSandbox(extra_env={"STROPHA_SANDBOX_FLAG": "yes"})
            sandbox.run(cand)
        env = rm.call_args.kwargs["env"]
        assert env["STROPHA_SANDBOX_FLAG"] == "yes"
        # Parent env is preserved (PATH is always present in CI).
        assert "PATH" in env

    def test_stdout_not_json_falls_back(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch(
            "subprocess.run",
            return_value=_completed(0, stdout="hello world"),
        ):
            sandbox = SubprocessSandbox()
            result = sandbox.run(cand)
        assert result.succeeded is True
        assert result.payload == {"raw": "hello world"}

    def test_stdout_json_list_falls_back(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch(
            "subprocess.run",
            return_value=_completed(0, stdout="[1, 2, 3]"),
        ):
            sandbox = SubprocessSandbox()
            result = sandbox.run(cand)
        assert result.payload == {"raw": "[1, 2, 3]"}

    def test_explicit_python_binary(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", return_value=_completed(0, stdout="{}")) as rm:
            sandbox = SubprocessSandbox(python_binary="/opt/python")
            sandbox.run(cand)
        assert rm.call_args.args[0][0] == "/opt/python"

    def test_explicit_project_root(self, tmp_path: Path) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", VALID_ADAPTER_SRC))
        with patch("subprocess.run", return_value=_completed(0, stdout="{}")) as rm:
            sandbox = SubprocessSandbox(project_root=tmp_path)
            sandbox.run(cand)
        assert rm.call_args.kwargs["cwd"] == str(tmp_path)

    def test_default_timeout_constant(self) -> None:
        assert SubprocessSandbox().timeout_s == SANDBOX_DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# _sandbox_runner.main (in-process invocation; not the real subprocess)
# ---------------------------------------------------------------------------


class TestRunnerArgvHandling:
    def test_no_args_exits_2(self, capsys: Any) -> None:
        rc = _sandbox_runner.main([])
        assert rc == 2
        err = json.loads(capsys.readouterr().err)
        assert err["ok"] is False
        assert "usage" in err["error"]

    def test_one_arg_exits_2(self, capsys: Any) -> None:
        rc = _sandbox_runner.main(["import_check"])
        assert rc == 2

    def test_unknown_job_exits_2(self, capsys: Any) -> None:
        rc = _sandbox_runner.main(["bogus", "/tmp"])
        assert rc == 2
        err = json.loads(capsys.readouterr().err)
        assert "unknown job" in err["error"]


class TestRunnerImportCheck:
    def test_happy_path(self, tmp_path: Path, capsys: Any) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", "X = 1\n"))
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is True
        assert "foo.py" in out["imported"]
        assert out["failed"] == []

    def test_syntax_error_fails(self, tmp_path: Path, capsys: Any) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", BROKEN_ADAPTER_SRC))
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is False
        assert len(out["failed"]) == 1
        assert "SyntaxError" in out["failed"][0]["error"]

    def test_import_time_exception_captured(self, tmp_path: Path, capsys: Any) -> None:
        cand = _make_candidate_with(tmp_path, ("foo.py", CRASHING_ADAPTER_SRC))
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is False
        assert "RuntimeError" in out["failed"][0]["error"]
        assert "boom at import time" in out["failed"][0]["error"]
        assert "traceback" in out["failed"][0]

    def test_systemexit_captured(self, tmp_path: Path, capsys: Any) -> None:
        """A candidate that calls ``sys.exit()`` at import time MUST NOT
        tear down the runner mid-loop. The runner uses ``BaseException``
        in its handler precisely so SystemExit is captured."""
        cand = _make_candidate_with(tmp_path, ("foo.py", SYSEXIT_ADAPTER_SRC))
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert any(
            "SystemExit" in f["error"] for f in out["failed"]
        )

    def test_no_adapters_dir(self, tmp_path: Path, capsys: Any) -> None:
        # The candidate dir exists but has no adapters/ subdir.
        cand = tmp_path / "cand"
        cand.mkdir()
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is True
        assert out["reason"] == "no_adapters_dir"

    def test_partial_failure_marks_overall_failure(
        self, tmp_path: Path, capsys: Any,
    ) -> None:
        cand = _make_candidate_with(
            tmp_path,
            ("good.py", "X = 1\n"),
            ("bad.py", BROKEN_ADAPTER_SRC),
        )
        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is False
        assert "good.py" in out["imported"]
        assert any(f["path"] == "bad.py" for f in out["failed"])

    def test_spec_from_file_location_none_treated_as_failure(
        self, tmp_path: Path, capsys: Any, monkeypatch: Any,
    ) -> None:
        """Defensive: when ``importlib.util.spec_from_file_location``
        returns None (impossible with real .py files but cheap to
        guard), the runner records it as a failed import rather than
        crashing the subprocess.
        """
        cand = _make_candidate_with(tmp_path, ("foo.py", "X = 1\n"))

        # Patch ``spec_from_file_location`` to return None for the
        # candidate file. We use monkeypatch.setattr against the
        # ``importlib.util`` namespace the runner imports from.
        import importlib.util as _iu

        def _spec_none(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(_iu, "spec_from_file_location", _spec_none)

        rc = _sandbox_runner.main(["import_check", str(cand)])
        assert rc == 1
        out = json.loads(capsys.readouterr().out)
        assert out["ok"] is False
        # Either "ImportError: no module spec" or the more general
        # AttributeError if monkey patching missed something — both
        # end up captured by the BaseException handler.
        assert any(f["path"] == "foo.py" for f in out["failed"])


# ---------------------------------------------------------------------------
# ``python -m`` entry-point guard
# ---------------------------------------------------------------------------


class TestModuleEntryPoint:
    def test_module_runs_as_main(self, tmp_path: Path) -> None:
        """Spawning the runner via ``python -m`` exercises the
        ``if __name__ == "__main__":`` guard at the bottom of the
        module. We verify the exit code propagates correctly through
        ``sys.exit(main())``.
        """
        proc = subprocess.run(
            [sys.executable, "-m", RUNNER_MODULE],
            capture_output=True,
            text=True,
            check=False,
        )
        # No args → usage error → exit 2.
        assert proc.returncode == 2
        err = json.loads(proc.stderr)
        assert err["ok"] is False
        assert "usage" in err["error"]


# ---------------------------------------------------------------------------
# End-to-end: real subprocess (no mocking)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_real_subprocess_import_check_happy(self, tmp_path: Path) -> None:
        """Spawn the actual runner module in a real subprocess.

        Verifies the wiring (PYTHONPATH, argv, JSON stdout) works
        against the real Python interpreter, not just a mock.
        """
        cand = _make_candidate_with(tmp_path, ("foo.py", "X = 1\n"))
        sandbox = SubprocessSandbox(timeout_s=30.0)
        result = sandbox.run(cand)
        assert result.succeeded is True, result.stderr_tail
        assert result.returncode == 0
        assert result.payload["ok"] is True
        assert "foo.py" in result.payload["imported"]

    def test_real_subprocess_isolation_on_broken_candidate(
        self, tmp_path: Path,
    ) -> None:
        """A candidate that crashes at import time does NOT crash the
        parent. The sandbox returns succeeded=False; the parent's
        ``sys.modules`` is unchanged."""
        cand = _make_candidate_with(tmp_path, ("crash.py", CRASHING_ADAPTER_SRC))
        before_modules = set(sys.modules)
        sandbox = SubprocessSandbox(timeout_s=30.0)
        result = sandbox.run(cand)
        assert result.succeeded is False
        assert result.returncode == 1
        assert result.payload["ok"] is False
        # And the parent never imported the candidate module.
        after_modules = set(sys.modules)
        assert before_modules == after_modules

    def test_real_subprocess_no_adapters_skipped(self, tmp_path: Path) -> None:
        cand = tmp_path / "empty"
        cand.mkdir()
        # No adapters/ subdir → sandbox skip path (no subprocess spawned).
        sandbox = SubprocessSandbox()
        result = sandbox.run(cand)
        assert result.succeeded is True
        assert result.payload["skipped"] is True


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_meta_reexports(self) -> None:
        import stropha.meta as m
        for name in (
            "SubprocessSandbox",
            "SandboxResult",
            "SandboxError",
            "RUNNER_MODULE",
            "SANDBOX_DEFAULT_TIMEOUT_S",
        ):
            assert hasattr(m, name), f"missing re-export: {name}"
