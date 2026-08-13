"""System-level safety tests for Phase-5 code search (WIN-200).

The Phase-5 contract (paper §5, RFC §Phase-5 risk table): production
``src/stropha/adapters/`` MUST NEVER be touched while the Meta-Harness
explores code-space. Every previous ticket in EPIC 1.6 contributes one
layer of that guarantee:

- WIN-196 ships path-validated artifact paths.
- WIN-197 ships the pytest gate that catches breakage pre-scoring.
- WIN-198 ships the subprocess sandbox that runs candidate code in a
  fresh interpreter.
- WIN-199 ships the namespaced registry so candidates can register
  without colliding with the global production state.

This test file is the cross-cutting *system* suite that ties all four
together. Each test snapshots production state (file tree + mtime +
content hash + registry contents) before exercising one or more
Phase-5 components, then asserts the snapshot is identical afterwards.

The suite is exhaustive about the attack surface:

- Path traversal (``../``).
- Absolute paths (``/etc/passwd``).
- Windows-style drive prefixes (``C:foo.py``).
- Backslash separators.
- Symlinks planted under the candidate's ``adapters/`` tree.
- Candidates that crash at import time.
- Candidates that ``sys.exit()`` at module level.
- Two candidates registering same-named adapters concurrently.
- A full ``run_loop`` invocation with a malicious proposer mixed in.

When this file goes red, the Phase-5 safety story is compromised.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from stropha.meta import (
    ArtifactStore,
    ScoreCard,
    SubprocessSandbox,
    run_loop,
)
from stropha.meta.proposer.code import LocalCodeProposer
from stropha.meta.store import CODE_ARTIFACTS_KEY, ArtifactStoreError
from stropha.pipeline import registry as reg

# ---------------------------------------------------------------------------
# Constants + fixtures
# ---------------------------------------------------------------------------

PRODUCTION_ADAPTERS = (
    Path(__file__).resolve().parents[2] / "src" / "stropha" / "adapters"
)

VALID_ADAPTER_SRC = (
    "from pydantic import BaseModel\n"
    "from stropha.models import SearchHit\n"
    "from stropha.pipeline.base import StageHealth\n"
    "from stropha.pipeline.registry import register_adapter\n"
    "class Config(BaseModel): pass\n"
    "@register_adapter(stage='reranker', name='candidate-reranker')\n"
    "class CandidateReranker:\n"
    "    Config = Config\n"
    "    def __init__(self, config=None): self.config = config or Config()\n"
    "    stage_name = property(lambda self: 'reranker')\n"
    "    adapter_name = property(lambda self: 'candidate-reranker')\n"
    "    adapter_id = property(lambda self: 'reranker:candidate-reranker')\n"
    "    config_schema = property(lambda self: Config)\n"
    "    def health(self): return StageHealth(status='ready', message='ok')\n"
    "    def rerank(self, query: str, hits: list[SearchHit], *, top_k=10):\n"
    "        return hits[:top_k]\n"
)

CRASHING_SRC = (
    "from stropha.pipeline.registry import register_adapter\n"
    "raise RuntimeError('boom at import time')\n"
)

SYSEXIT_SRC = (
    "import sys\n"
    "sys.exit(7)\n"
)


def _snapshot_production() -> dict[Path, tuple[int, int, str]]:
    """Record ``(mtime_ns, size_bytes, sha256)`` for every .py under prod adapters."""
    out: dict[Path, tuple[int, int, str]] = {}
    for path in PRODUCTION_ADAPTERS.rglob("*.py"):
        data = path.read_bytes()
        out[path] = (
            path.stat().st_mtime_ns,
            len(data),
            hashlib.sha256(data).hexdigest(),
        )
    return out


def _snapshot_global_registry() -> dict[tuple[str, str], str]:
    """Capture ``{(stage, name): module_path}`` for the production registry."""
    return {
        key: f"{cls.__module__}.{cls.__qualname__}"
        for key, cls in reg._GLOBAL_REGISTRY.items()
    }


@pytest.fixture
def production_snapshot() -> dict[Path, tuple[int, int, str]]:
    """Capture the production adapters' on-disk state before a test runs."""
    return _snapshot_production()


@pytest.fixture
def registry_snapshot() -> dict[tuple[str, str], str]:
    """Capture the global registry's state before a test runs."""
    return _snapshot_global_registry()


def _assert_production_unchanged(
    before: dict[Path, tuple[int, int, str]],
) -> None:
    """The strongest invariant in this file."""
    after = _snapshot_production()
    assert set(before) == set(after), "files appeared / disappeared under production adapters/"
    for path, before_tuple in before.items():
        assert after[path] == before_tuple, (
            f"production file {path} changed during test "
            f"(before={before_tuple}, after={after[path]})"
        )


def _assert_registry_unchanged(
    before: dict[tuple[str, str], str],
) -> None:
    after = _snapshot_global_registry()
    assert before == after, "global registry mutated during test"


def _stub_space():
    """Search space shared across the loop-driven tests."""
    from stropha.meta import Dimension, SearchSpace

    return SearchSpace(
        dimensions=[
            Dimension(
                name="walker",
                kind="stage",
                indexing=True,
                type="choice",
                choices=("git-ls-files", "filesystem"),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1-2. Baseline + propose-only
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_baseline_snapshot_is_stable(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
    ) -> None:
        """Sanity check the snapshot helper is deterministic."""
        again = _snapshot_production()
        assert production_snapshot == again

    def test_propose_only_does_not_touch_production(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        registry_snapshot: dict[tuple[str, str], str],
    ) -> None:
        """Just running the proposer — no store, no sandbox — is safe."""
        class _Stub(LocalCodeProposer):
            def _call_llm(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "config": {"walker": "filesystem"},
                        "code_artifacts": {
                            "adapters/enricher/safe.py": VALID_ADAPTER_SRC,
                        },
                    }
                )

        prop = _Stub(
            space=_stub_space(),
            skill="<test>",
            backend_name="fake",
            env={},
            fallback_seed=0,
            prefer_backend=None,
        )
        prop([])
        _assert_production_unchanged(production_snapshot)
        _assert_registry_unchanged(registry_snapshot)


# ---------------------------------------------------------------------------
# 3-6. Store / gate / sandbox / namespace — each in isolation
# ---------------------------------------------------------------------------


class TestStoreLayer:
    def test_register_with_artifacts_does_not_touch_production(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        store = ArtifactStore(tmp_path / "meta")
        store.register_candidate(
            config={
                "walker": "git-ls-files",
                "reranker": "candidate-reranker",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/candidate-reranker.py": VALID_ADAPTER_SRC,
                },
            },
        )
        _assert_production_unchanged(production_snapshot)


class TestSandboxLayer:
    def test_sandbox_real_subprocess_does_not_touch_production(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        registry_snapshot: dict[tuple[str, str], str],
        tmp_path: Path,
    ) -> None:
        cand = tmp_path / "candidate"
        (cand / "adapters").mkdir(parents=True)
        (cand / "adapters" / "foo.py").write_text(VALID_ADAPTER_SRC)

        sandbox = SubprocessSandbox(timeout_s=30.0)
        result = sandbox.run(cand)
        assert result.succeeded is True
        _assert_production_unchanged(production_snapshot)
        _assert_registry_unchanged(registry_snapshot)


class TestNamespaceLayer:
    def test_namespace_scope_does_not_pollute_global(
        self,
        registry_snapshot: dict[tuple[str, str], str],
    ) -> None:
        from pydantic import BaseModel

        class _Config(BaseModel): ...

        class _CandidateAdapter:
            Config = _Config
            __module__ = "test_production_isolation"

        with reg.namespace_scope("isolation-test"):
            reg.register_adapter(
                stage="enricher", name="iso-test",
            )(_CandidateAdapter)
            # Inside the scope: visible.
            assert reg.lookup_adapter("enricher", "iso-test") is _CandidateAdapter
        # Outside: global must not have it.
        assert ("enricher", "iso-test") not in reg._GLOBAL_REGISTRY
        _assert_registry_unchanged(registry_snapshot)
        # Cleanup so this test doesn't leak.
        reg.drop_namespace("isolation-test")


# ---------------------------------------------------------------------------
# 7-10. Attack-surface coverage
# ---------------------------------------------------------------------------


class TestTraversalAttacks:
    def test_parent_traversal_blocked(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        """A path like ``../../src/stropha/adapters/owned.py`` MUST NOT land
        anywhere under production. Two safety nets must fire — the
        proposer's filter drops the artifact, and even if it somehow
        survived, the store's resolved-path check would catch it."""
        # Direct store-level test (bypasses the proposer's filter).
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "../../src/stropha/adapters/owned.py": VALID_ADAPTER_SRC,
                    },
                },
            )
        _assert_production_unchanged(production_snapshot)
        assert not (PRODUCTION_ADAPTERS / "owned.py").exists()

    def test_absolute_path_blocked(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "/tmp/abs_attack.py": VALID_ADAPTER_SRC,
                    },
                },
            )
        _assert_production_unchanged(production_snapshot)
        assert not Path("/tmp/abs_attack.py").exists()

    def test_drive_prefix_blocked(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "C:windrive.py": VALID_ADAPTER_SRC,
                    },
                },
            )
        _assert_production_unchanged(production_snapshot)

    def test_backslash_blocked(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "adapters\\foo.py": VALID_ADAPTER_SRC,
                    },
                },
            )
        _assert_production_unchanged(production_snapshot)

    def test_symlink_escape_blocked(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        """Even with a planted symlink under <candidate>/adapters/, the
        store's resolved-path check rejects writes that escape."""
        store = ArtifactStore(tmp_path / "meta")
        cand_id = "symlink-attack"
        cand_dir = store.candidates_dir / cand_id
        adapters = cand_dir / "adapters"
        adapters.mkdir(parents=True)

        # Plant a symlink that points at production. If the store's
        # resolved-path check failed, writing through this symlink would
        # corrupt production.
        (adapters / "escape").symlink_to(PRODUCTION_ADAPTERS)

        with pytest.raises(ArtifactStoreError, match="escape"):
            store._write_code_artifacts(
                cand_id,
                {"adapters/escape/owned.py": VALID_ADAPTER_SRC},
            )
        _assert_production_unchanged(production_snapshot)
        assert not (PRODUCTION_ADAPTERS / "owned.py").exists()


# ---------------------------------------------------------------------------
# 11-12. Crashing candidates do not corrupt production
# ---------------------------------------------------------------------------


class TestCandidateCrashIsolation:
    def test_crashing_candidate_isolated(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        registry_snapshot: dict[tuple[str, str], str],
        tmp_path: Path,
    ) -> None:
        cand = tmp_path / "cand"
        (cand / "adapters").mkdir(parents=True)
        (cand / "adapters" / "boom.py").write_text(CRASHING_SRC)

        sandbox = SubprocessSandbox(timeout_s=30.0)
        result = sandbox.run(cand)
        # The candidate's import RAISED; the subprocess exited 1.
        # The parent process is alive (we're still here), production
        # is untouched, and the global registry is unchanged.
        assert result.succeeded is False
        assert result.returncode == 1
        _assert_production_unchanged(production_snapshot)
        _assert_registry_unchanged(registry_snapshot)

    def test_sysexit_candidate_isolated(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        registry_snapshot: dict[tuple[str, str], str],
        tmp_path: Path,
    ) -> None:
        cand = tmp_path / "cand"
        (cand / "adapters").mkdir(parents=True)
        (cand / "adapters" / "exiter.py").write_text(SYSEXIT_SRC)

        sandbox = SubprocessSandbox(timeout_s=30.0)
        result = sandbox.run(cand)
        # SystemExit inside the candidate's module body is caught by the
        # runner's BaseException handler; the runner exits 1.
        assert result.succeeded is False
        assert result.returncode == 1
        _assert_production_unchanged(production_snapshot)
        _assert_registry_unchanged(registry_snapshot)


# ---------------------------------------------------------------------------
# 13-14. Concurrent candidates + drop_namespace cleanup
# ---------------------------------------------------------------------------


class TestConcurrentCandidates:
    def test_two_candidates_same_adapter_no_collision(
        self,
        registry_snapshot: dict[tuple[str, str], str],
    ) -> None:
        from pydantic import BaseModel

        class _Cfg(BaseModel): ...

        class _A:
            Config = _Cfg
            __module__ = "test_iso.a"

        class _B:
            Config = _Cfg
            __module__ = "test_iso.b"

        with reg.namespace_scope("cand-A"):
            reg.register_adapter(stage="enricher", name="shared")(_A)
        with reg.namespace_scope("cand-B"):
            reg.register_adapter(stage="enricher", name="shared")(_B)

        # Both namespaces have their own version.
        assert reg.get_registry("cand-A")[("enricher", "shared")] is _A
        assert reg.get_registry("cand-B")[("enricher", "shared")] is _B
        # Global is untouched.
        assert ("enricher", "shared") not in reg._GLOBAL_REGISTRY
        _assert_registry_unchanged(registry_snapshot)
        # Cleanup.
        reg.drop_namespace("cand-A")
        reg.drop_namespace("cand-B")

    def test_drop_namespace_after_candidate_done(self) -> None:
        from pydantic import BaseModel

        class _Cfg(BaseModel): ...

        class _Tmp:
            Config = _Cfg
            __module__ = "test_iso.tmp"

        with reg.namespace_scope("tmp"):
            reg.register_adapter(stage="enricher", name="t")(_Tmp)
        assert "tmp" in reg._NAMESPACED_REGISTRIES
        reg.drop_namespace("tmp")
        assert "tmp" not in reg._NAMESPACED_REGISTRIES


# ---------------------------------------------------------------------------
# 15-16. End-to-end run_loop with a code proposer
# ---------------------------------------------------------------------------


class TestLoopEndToEnd:
    def test_loop_e2e_no_production_mutation(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        registry_snapshot: dict[tuple[str, str], str],
        tmp_path: Path,
    ) -> None:
        """A full run_loop iteration with a code proposer + store +
        passing gate leaves production untouched. The proposer emits
        artifacts that land under the candidate dir; the loop registers
        the candidate, the gate clears (we mock it as a no-op pass),
        the evaluator scores, the loop archives the result — and
        production never sees a thing."""
        store = ArtifactStore(tmp_path / "meta")

        class _CodeProposer(LocalCodeProposer):
            def _call_llm(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "config": {"walker": "filesystem"},
                        "code_artifacts": {
                            "adapters/enricher/c1.py": VALID_ADAPTER_SRC,
                        },
                    }
                )

        prop = _CodeProposer(
            space=_stub_space(),
            skill="<test>",
            backend_name="fake",
            env={},
            fallback_seed=0,
            prefer_backend=None,
        )

        def good_evaluator(config: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.8, context_efficiency=0.5)

        def passing_gate(candidate_dir: Path) -> Any:
            from stropha.meta import GateResult
            return GateResult(passed=True, skipped=True)

        result = run_loop(
            proposer=prop,
            evaluator=good_evaluator,
            store=store,
            max_iterations=2,
            gate=passing_gate,
        )
        assert len(result.archive) == 2
        # Every candidate dropped its artifacts under the store.
        for cand_dir in (tmp_path / "meta" / "candidates").iterdir():
            if cand_dir.is_dir() and (cand_dir / "adapters").is_dir():
                files = list((cand_dir / "adapters").rglob("*.py"))
                assert len(files) >= 1
                # And none of them live under production.
                for f in files:
                    assert PRODUCTION_ADAPTERS not in f.parents
        _assert_production_unchanged(production_snapshot)
        _assert_registry_unchanged(registry_snapshot)

    def test_loop_e2e_with_malicious_proposer(
        self,
        production_snapshot: dict[Path, tuple[int, int, str]],
        tmp_path: Path,
    ) -> None:
        """A proposer that consistently emits invalid artifacts MUST NOT
        corrupt production OR the store. Bad artifacts are silently
        dropped by the proposer-side filter; the loop still archives
        the (artifact-free) candidate."""
        store = ArtifactStore(tmp_path / "meta")

        class _BadProposer(LocalCodeProposer):
            def _call_llm(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "config": {"walker": "git-ls-files"},
                        "code_artifacts": {
                            "../../src/stropha/adapters/owned.py": VALID_ADAPTER_SRC,
                            "/etc/passwd": VALID_ADAPTER_SRC,
                            "adapters/bad\\path.py": VALID_ADAPTER_SRC,
                        },
                    }
                )

        prop = _BadProposer(
            space=_stub_space(),
            skill="<test>",
            backend_name="fake",
            env={},
            fallback_seed=0,
            prefer_backend=None,
        )

        result = run_loop(
            proposer=prop,
            evaluator=lambda c: ScoreCard(quality=0.5),
            store=store,
            max_iterations=1,
        )
        # Archive landed normally (bad artifacts dropped, config stays).
        assert len(result.archive) == 1
        # And the candidate dir has no adapters/ subdir (every artifact
        # was filtered out by the proposer).
        record, _ = result.archive[0]
        assert not (record.base_dir / "adapters").exists()  # type: ignore[union-attr]
        _assert_production_unchanged(production_snapshot)


# ---------------------------------------------------------------------------
# 17-18. Multi-candidate isolation + post-loop registry cleanliness
# ---------------------------------------------------------------------------


class TestCrossCandidateIsolation:
    def test_concurrent_candidate_dirs_isolated(self, tmp_path: Path) -> None:
        """Two candidates registered back-to-back never share files on disk."""
        store = ArtifactStore(tmp_path / "meta")
        rec_a = store.register_candidate(
            config={
                "reranker": "candidate-reranker",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/candidate-reranker.py": VALID_ADAPTER_SRC,
                },
            },
            label="A",
        )
        rec_b = store.register_candidate(
            config={
                "reranker": "candidate-reranker",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/candidate-reranker.py": VALID_ADAPTER_SRC,
                },
            },
            label="B",
        )
        a_file = rec_a.base_dir / "adapters" / "reranker" / "candidate-reranker.py"  # type: ignore[union-attr]
        b_file = rec_b.base_dir / "adapters" / "reranker" / "candidate-reranker.py"  # type: ignore[union-attr]
        assert a_file.is_file()
        assert b_file.is_file()
        assert a_file != b_file  # different on-disk paths.


class TestPostLoopRegistry:
    def test_global_registry_pristine_after_loop(
        self,
        registry_snapshot: dict[tuple[str, str], str],
        tmp_path: Path,
    ) -> None:
        """Even after a full loop run that proposed code artifacts, the
        global registry is byte-for-byte identical."""
        store = ArtifactStore(tmp_path / "meta")

        class _P(LocalCodeProposer):
            def _call_llm(self, prompt: str) -> str:
                return json.dumps(
                    {
                        "config": {"walker": "filesystem"},
                        "code_artifacts": {
                            "adapters/enricher/c.py": VALID_ADAPTER_SRC,
                        },
                    }
                )

        prop = _P(
            space=_stub_space(),
            skill="<t>",
            backend_name="fake",
            env={},
            fallback_seed=0,
            prefer_backend=None,
        )
        run_loop(
            proposer=prop,
            evaluator=lambda c: ScoreCard(quality=0.5),
            store=store,
            max_iterations=3,
        )
        _assert_registry_unchanged(registry_snapshot)
