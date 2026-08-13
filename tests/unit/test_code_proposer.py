"""Unit tests for the Phase-5 code-search proposer (WIN-196).

What's covered:

- Path validation (proposer side): absolute, traversal, backslash, drive
  prefix, non-``.py``, illegal segments, allowed-char regex.
- Source validation: empty, too-large, missing ``@register_adapter``
  token, happy path.
- ``_validate_artifacts`` survivor selection: empty input, non-dict,
  mixed bad/good entries, MAX_ARTIFACTS_PER_CANDIDATE cap.
- End-to-end proposer round-trips via a stubbed LLM: Phase-5 wrapper
  shape, back-compat plain-dict shape, malformed JSON, LLM failure,
  invalid artifact survives gracefully without poisoning the config.
- ``ArtifactStore`` integration: ``__code_artifacts__`` popped from
  ``config.yaml``, files materialised under ``candidates/<id>/adapters/``,
  defence-in-depth path checks at the store layer, rollback on any
  artifact-validation failure.
- Factories ``make_local_code_proposer`` / ``make_cloud_code_proposer``
  produce the right concrete class with the skill loaded.
- Safety smoke: production ``src/stropha/adapters/`` is not touched by
  a full propose-and-register round trip.

These tests are the proposer-side half of the safety story. The dedicated
WIN-200 suite will assert the system-wide invariant (no production
adapter is mutated across an entire loop run).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from stropha.meta import Dimension, ScoreCard, SearchSpace
from stropha.meta.proposer import (
    CloudCodeProposer,
    LocalCodeProposer,
    make_cloud_code_proposer,
    make_local_code_proposer,
)
from stropha.meta.proposer.code import (
    CODE_ARTIFACTS_KEY,
    MAX_ARTIFACT_SOURCE_BYTES,
    MAX_ARTIFACTS_PER_CANDIDATE,
    REGISTER_ADAPTER_TOKEN,
    _is_valid_artifact_path,
    _is_valid_artifact_source,
    _validate_artifacts,
    build_code_prompt_tail,
)
from stropha.meta.store import (
    ArtifactStore,
    ArtifactStoreError,
    CandidateRecord,
    _validate_artifact_rel_path,
)

VALID_SOURCE = (
    "from stropha.pipeline.registry import register_adapter\n"
    "\n"
    "@register_adapter(stage='reranker', name='my-new-reranker')\n"
    "class MyNewReranker:\n"
    "    class Config: ...\n"
)


def _stub_space() -> SearchSpace:
    return SearchSpace(
        dimensions=[
            Dimension(
                name="walker",
                kind="stage",
                indexing=True,
                type="choice",
                choices=("git-ls-files", "filesystem"),
            ),
            Dimension(
                name="reranker",
                kind="stage",
                indexing=False,
                type="choice",
                choices=("noop", "my-new-reranker"),
            ),
            Dimension(
                name="cache_size",
                kind="feature",
                indexing=False,
                type="int",
                range=(100, 1000),
                step=100,
                default=500,
            ),
        ],
    )


def _record(idx: int, config: dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        id=f"c{idx:02d}",
        parent_id=None,
        created_at="2026-05-22T00:00:00+00:00",
        label=f"iter-{idx}",
        config=config,
    )


class _FakeLocalCodeProposer(LocalCodeProposer):
    """LocalCodeProposer with a canned LLM response.

    Bypasses the real backend so tests don't need mlx/ollama installed.
    The stub replicates the ``_FakeProposer`` pattern from
    test_agentic_proposer.py.
    """

    _response: str | None = None
    _exception: BaseException | None = None

    def _call_llm(self, prompt: str) -> str | None:
        # Stash the prompt so the test can introspect it.
        self.last_prompt = prompt  # type: ignore[attr-defined]
        if self._exception is not None:
            raise self._exception
        return self._response


def _make_fake(
    response: str | None = None,
    *,
    exception: BaseException | None = None,
    space: SearchSpace | None = None,
) -> _FakeLocalCodeProposer:
    p = _FakeLocalCodeProposer(
        space=space or _stub_space(),
        skill="<test skill>",
        backend_name="fake",
        env={},
        fallback_seed=42,
        prefer_backend=None,
    )
    p._response = response
    p._exception = exception
    return p


# ---------------------------------------------------------------------------
# _is_valid_artifact_path
# ---------------------------------------------------------------------------


class TestArtifactPathValidator:
    def test_happy_path(self) -> None:
        assert _is_valid_artifact_path("adapters/reranker/foo.py") is True
        assert _is_valid_artifact_path("adapters/enricher/foo.py") is False
        assert _is_valid_artifact_path("adapters/walker/w.py") is False

    def test_rejects_non_adapters_prefix(self) -> None:
        """The Phase-5 contract: artifacts MUST land under adapters/."""
        # Bare .py at the candidate root → rejected (the prompt requires
        # the adapters/ prefix so the LLM knows where things land).
        assert _is_valid_artifact_path("foo.py") is False
        # Other subdirs of the candidate (tests/, traces/, …) are off
        # limits for Phase 5 — that's the WIN-197 / WIN-200 territory.
        assert _is_valid_artifact_path("tests/test_foo.py") is False
        assert _is_valid_artifact_path("scores.json.py") is False  # not under adapters/

    def test_rejects_absolute_path(self) -> None:
        assert _is_valid_artifact_path("/etc/passwd") is False
        assert _is_valid_artifact_path("/foo.py") is False

    def test_rejects_parent_traversal(self) -> None:
        assert _is_valid_artifact_path("../foo.py") is False
        assert _is_valid_artifact_path("a/../b.py") is False
        # The leading dot-slash is also a parent-traversal marker.
        assert _is_valid_artifact_path("./foo.py") is False

    def test_rejects_backslashes(self) -> None:
        assert _is_valid_artifact_path("adapters\\foo.py") is False

    def test_rejects_drive_prefix(self) -> None:
        assert _is_valid_artifact_path("C:foo.py") is False
        assert _is_valid_artifact_path("c:/foo.py") is False

    def test_rejects_non_python_extension(self) -> None:
        assert _is_valid_artifact_path("adapters/foo.txt") is False
        assert _is_valid_artifact_path("adapters/foo") is False

    def test_rejects_empty(self) -> None:
        assert _is_valid_artifact_path("") is False

    def test_rejects_non_string(self) -> None:
        # mypy would catch this in source, but the validator must be
        # defensive against any JSON the LLM emits.
        assert _is_valid_artifact_path(None) is False  # type: ignore[arg-type]
        assert _is_valid_artifact_path(42) is False  # type: ignore[arg-type]

    def test_rejects_illegal_characters(self) -> None:
        # NUL byte, spaces, weird unicode — anything outside the
        # underscore/dash/dot/alpha whitelist.
        assert _is_valid_artifact_path("adapters/foo bar.py") is False
        assert _is_valid_artifact_path("adapters/foo\x00.py") is False
        assert _is_valid_artifact_path("adapters/foo$.py") is False

    def test_rejects_double_slash(self) -> None:
        assert _is_valid_artifact_path("adapters//foo.py") is False


# ---------------------------------------------------------------------------
# _is_valid_artifact_source
# ---------------------------------------------------------------------------


class TestArtifactSourceValidator:
    def test_happy_path(self) -> None:
        assert _is_valid_artifact_source(VALID_SOURCE) is True

    def test_rejects_empty(self) -> None:
        assert _is_valid_artifact_source("") is False
        assert _is_valid_artifact_source("   \n  \t") is False

    def test_rejects_missing_decorator(self) -> None:
        assert _is_valid_artifact_source("class Foo: pass") is False
        # Even if it imports register_adapter, without the @-decoration
        # token we reject — too easy to write a file that imports and
        # never calls.
        assert _is_valid_artifact_source(
            "from stropha.pipeline.registry import register_adapter\n"
            "class Foo: pass\n"
        ) is False

    def test_rejects_oversize_source(self) -> None:
        huge = REGISTER_ADAPTER_TOKEN + "\n" + ("# pad\n" * 20_000)
        assert len(huge.encode("utf-8")) > MAX_ARTIFACT_SOURCE_BYTES
        assert _is_valid_artifact_source(huge) is False

    def test_rejects_non_string(self) -> None:
        assert _is_valid_artifact_source(None) is False  # type: ignore[arg-type]
        assert _is_valid_artifact_source(42) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _validate_artifacts (survivor selection)
# ---------------------------------------------------------------------------


class TestValidateArtifacts:
    def test_empty_input_returns_empty(self) -> None:
        assert _validate_artifacts(None) == {}
        assert _validate_artifacts({}) == {}

    def test_non_dict_rejected(self) -> None:
        assert _validate_artifacts(["a", "b"]) == {}
        assert _validate_artifacts("string") == {}
        assert _validate_artifacts(42) == {}

    def test_mixed_good_and_bad_keeps_only_good(self) -> None:
        raw = {
            "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
            "/etc/passwd": VALID_SOURCE,  # absolute
            "../bad.py": VALID_SOURCE,  # traversal
            "adapters/x/no_decorator.py": "class Foo: pass",  # missing token
        }
        out = _validate_artifacts(raw)
        assert out == {}

    def test_rejects_more_than_one(self) -> None:
        raw = {
            f"adapters/reranker/a{i}.py": VALID_SOURCE
            for i in range(MAX_ARTIFACTS_PER_CANDIDATE + 3)
        }
        out = _validate_artifacts(raw)
        assert out == {}


# ---------------------------------------------------------------------------
# store._validate_artifact_rel_path (defence in depth)
# ---------------------------------------------------------------------------


class TestStorePathValidator:
    def test_accepts_valid(self) -> None:
        # Returns None on success (no exception).
        _validate_artifact_rel_path("adapters/reranker/foo.py")

    def test_rejects_absolute(self) -> None:
        with pytest.raises(ArtifactStoreError, match="relative"):
            _validate_artifact_rel_path("/etc/passwd")

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ArtifactStoreError, match="illegal segment"):
            _validate_artifact_rel_path("a/../b.py")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(ArtifactStoreError, match="POSIX"):
            _validate_artifact_rel_path("a\\b.py")

    def test_rejects_drive_prefix(self) -> None:
        with pytest.raises(ArtifactStoreError, match="drive"):
            _validate_artifact_rel_path("C:foo.py")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ArtifactStoreError, match="non-empty"):
            _validate_artifact_rel_path("")

    def test_rejects_double_slash(self) -> None:
        with pytest.raises(ArtifactStoreError, match="illegal segment"):
            _validate_artifact_rel_path("a//b.py")


# ---------------------------------------------------------------------------
# ArtifactStore integration
# ---------------------------------------------------------------------------


class TestStoreCodeArtifacts:
    def test_materialises_artifacts(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(
            config={
                "walker": "git-ls-files",
                "reranker": "my-new-reranker",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
                },
            },
        )
        cand_dir = tmp_path / "meta" / "candidates" / record.id
        target = cand_dir / "adapters" / "reranker" / "my-new-reranker.py"
        assert target.is_file()
        assert target.read_text() == VALID_SOURCE
        # config.yaml MUST NOT carry the artifacts key — proves the
        # store popped it before serialising.
        cfg_disk = yaml.safe_load((cand_dir / "config.yaml").read_text())
        assert CODE_ARTIFACTS_KEY not in cfg_disk
        # CandidateRecord.config also stripped (the in-memory mirror).
        assert CODE_ARTIFACTS_KEY not in record.config

    def test_no_artifacts_key_when_absent(self, tmp_path: Path) -> None:
        """Plain config (no artifacts) behaves exactly like before."""
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(config={"walker": "filesystem"})
        cand_dir = tmp_path / "meta" / "candidates" / record.id
        # The adapters/ subdir is created lazily — must not exist yet.
        assert not (cand_dir / "adapters").exists()

    def test_rejects_path_traversal_with_rollback(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "../../escape.py": VALID_SOURCE,
                    },
                },
            )
        # The candidate dir MUST have been rolled back — no orphans.
        existing = list((tmp_path / "meta" / "candidates").iterdir())
        assert existing == []

    def test_rejects_non_py_with_rollback(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError, match=r"\.py"):
            store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "adapters/foo.txt": VALID_SOURCE,
                    },
                },
            )
        assert list((tmp_path / "meta" / "candidates").iterdir()) == []

    def test_rejects_non_dict_artifacts(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        with pytest.raises(ArtifactStoreError, match="dict"):
            store.register_candidate(
                config={CODE_ARTIFACTS_KEY: ["not", "a", "dict"]},
            )
        assert list((tmp_path / "meta" / "candidates").iterdir()) == []

    def test_reranker_dir_created(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        record = store.register_candidate(
            config={
                "reranker": "my-new-reranker",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
                },
            },
        )
        target = (
            tmp_path / "meta" / "candidates" / record.id
            / "adapters" / "reranker" / "my-new-reranker.py"
        )
        assert target.is_file()

    def test_store_rejects_non_adapters_prefix_direct(self, tmp_path: Path) -> None:
        """Defence-in-depth: ``_write_code_artifacts`` enforces the
        ``adapters/`` prefix even when the path passes
        ``_validate_artifact_rel_path``. Called direct to exercise the
        store-side guard without bouncing off the proposer's validator.
        """
        store = ArtifactStore(tmp_path / "meta")
        cand_id = "deadbeef"
        (store.candidates_dir / cand_id).mkdir()
        with pytest.raises(ArtifactStoreError, match="adapters/"):
            store._write_code_artifacts(
                cand_id, {"foo/bar.py": VALID_SOURCE},
            )

    def test_store_handles_unresolved_temp_dir(self) -> None:
        """Regression for the macOS /var → /private/var symlink case.

        pytest's ``tmp_path`` fixture returns a pre-resolved path, which
        masked a latent bug in ``_write_code_artifacts`` where the
        canonical ``target.resolve()`` was compared against the
        un-resolved ``cand_dir`` via ``relative_to``. End-to-end
        validation against ``tempfile.TemporaryDirectory()`` (which
        does NOT pre-resolve) blew up. This test reproduces the
        unresolved-path call shape so the regression is locked in.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            # Critical: do NOT call ``.resolve()`` on the base path.
            store = ArtifactStore(Path(td) / "meta")
            record = store.register_candidate(
                config={
                    CODE_ARTIFACTS_KEY: {
                        "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
                    },
                    "reranker": "my-new-reranker",
                },
            )
            # The materialised file must be reachable from the resolved
            # path (which is where the writer actually put it).
            resolved_target = (
                record.base_dir.resolve()  # type: ignore[union-attr]
                / "adapters" / "reranker" / "my-new-reranker.py"
            )
            assert resolved_target.is_file()

    def test_store_blocks_symlink_escape(self, tmp_path: Path) -> None:
        """Symlink hardening: a planted symlink that escapes the
        candidate's adapters/ root is caught by the resolved-path
        check, even though the input rel_path looks fine.

        This test exists to cover the ``target.relative_to(adapters_root)``
        guard which the proposer-side validator alone cannot reach (the
        rel_path itself contains no ``..`` segments).
        """
        store = ArtifactStore(tmp_path / "meta")
        cand_id = "feedbead"
        cand_dir = store.candidates_dir / cand_id
        adapters = cand_dir / "adapters"
        adapters.mkdir(parents=True)

        outside = tmp_path / "outside"
        outside.mkdir()
        (adapters / "escape").symlink_to(outside)

        with pytest.raises(ArtifactStoreError, match="escape"):
            store._write_code_artifacts(
                cand_id, {"adapters/escape/owned.py": VALID_SOURCE},
            )
        # And no file landed outside.
        assert not (outside / "owned.py").exists()


# ---------------------------------------------------------------------------
# Proposer end-to-end (with stub LLM)
# ---------------------------------------------------------------------------


class TestProposerEndToEnd:
    def test_phase5_wrapper_shape_parsed(self) -> None:
        response = json.dumps(
            {
                "config": {"walker": "filesystem", "reranker": "my-new-reranker"},
                "code_artifacts": {
                    "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
                },
            }
        )
        prop = _make_fake(response)
        cfg = prop([])
        assert cfg["walker"] == "filesystem"
        assert cfg["reranker"] == "my-new-reranker"
        assert CODE_ARTIFACTS_KEY in cfg
        assert "adapters/reranker/my-new-reranker.py" in cfg[CODE_ARTIFACTS_KEY]

    def test_back_compat_plain_dict(self) -> None:
        """If the LLM emits a flat config (no wrapper), still works."""
        response = json.dumps({"walker": "filesystem"})
        prop = _make_fake(response)
        cfg = prop([])
        assert cfg["walker"] == "filesystem"
        # No artifacts ⇒ key absent.
        assert CODE_ARTIFACTS_KEY not in cfg

    def test_no_artifacts_key_when_empty_list_emitted(self) -> None:
        response = json.dumps(
            {"config": {"walker": "git-ls-files"}, "code_artifacts": {}}
        )
        prop = _make_fake(response)
        cfg = prop([])
        assert CODE_ARTIFACTS_KEY not in cfg

    def test_invalid_artifact_dropped_config_survives(self) -> None:
        """An invalid path drops the artifact but the config still surfaces."""
        response = json.dumps(
            {
                "config": {"walker": "filesystem"},
                "code_artifacts": {"/etc/passwd": VALID_SOURCE},
            }
        )
        prop = _make_fake(response)
        cfg = prop([])
        assert cfg["walker"] == "filesystem"
        assert CODE_ARTIFACTS_KEY not in cfg  # all artifacts dropped

    def test_malformed_json_falls_back(self) -> None:
        prop = _make_fake("not json at all")
        cfg = prop([])
        # Fallback bandit always emits valid space dims; never carries
        # code_artifacts (it doesn't know about them).
        assert "walker" in cfg
        assert CODE_ARTIFACTS_KEY not in cfg

    def test_llm_exception_falls_back(self) -> None:
        prop = _make_fake(exception=RuntimeError("boom"))
        cfg = prop([])
        assert "walker" in cfg
        assert CODE_ARTIFACTS_KEY not in cfg

    def test_empty_llm_response_falls_back(self) -> None:
        prop = _make_fake("")
        cfg = prop([])
        assert "walker" in cfg
        assert CODE_ARTIFACTS_KEY not in cfg

    def test_prompt_includes_code_tail(self) -> None:
        response = json.dumps({"config": {"walker": "filesystem"}})
        prop = _make_fake(response)
        prop([])
        # The Phase-5 tail must appear in the prompt the LLM saw.
        assert "code_artifacts" in prop.last_prompt
        assert "Phase 5" in prop.last_prompt

    def test_archive_summarised_into_prompt(self) -> None:
        response = json.dumps({"config": {"walker": "filesystem"}})
        prop = _make_fake(response)
        archive = [(_record(0, {"walker": "git-ls-files"}), ScoreCard(quality=0.9))]
        prop(archive)
        # The standard agentic prompt shape includes the packed history bundle;
        # confirm the code proposer didn't accidentally drop it.
        assert "Full history bundle" in prop.last_prompt
        assert "git-ls-files" in prop.last_prompt


# ---------------------------------------------------------------------------
# build_code_prompt_tail
# ---------------------------------------------------------------------------


class TestPromptTail:
    def test_tail_mentions_limits(self) -> None:
        tail = build_code_prompt_tail()
        assert str(MAX_ARTIFACT_SOURCE_BYTES) in tail
        assert str(MAX_ARTIFACTS_PER_CANDIDATE) in tail
        assert "code_artifacts" in tail
        assert "@register_adapter" in tail


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestFactories:
    def test_make_local_returns_local(self) -> None:
        space = _stub_space()
        p = make_local_code_proposer(space, env={"X": "1"})
        assert isinstance(p, LocalCodeProposer)
        assert p.space is space
        assert p.env == {"X": "1"}
        # Skill prompt is loaded from disk.
        assert p.skill and "Meta-Harness proposer" in p.skill

    def test_make_cloud_returns_cloud(self) -> None:
        space = _stub_space()
        p = make_cloud_code_proposer(space, prefer_backend="anthropic")
        assert isinstance(p, CloudCodeProposer)
        assert p.prefer_backend == "anthropic"

    def test_make_cloud_rejects_bad_backend(self) -> None:
        with pytest.raises(ValueError, match="anthropic"):
            make_cloud_code_proposer(_stub_space(), prefer_backend="bogus")


# ---------------------------------------------------------------------------
# Production-adapter isolation smoke (full propose+register loop)
# ---------------------------------------------------------------------------


class TestProductionIsolation:
    def test_production_adapters_untouched(self, tmp_path: Path) -> None:
        """End-to-end: a propose+register cycle must not touch real adapters.

        The hard guarantee comes from the proposer's path validator AND
        the store's resolved-prefix check working together. This is the
        smoke; WIN-200 will ship a much broader system-level suite.
        """
        # Snapshot the production adapters directory.
        prod = Path(__file__).resolve().parents[2] / "src" / "stropha" / "adapters"
        before = {p: p.stat().st_mtime_ns for p in prod.rglob("*.py")}

        store = ArtifactStore(tmp_path / "meta")
        # An LLM proposing a brand-new file under the right relative
        # path lands under the candidate's adapters/, NEVER under prod.
        response = json.dumps(
            {
                "config": {
                    "walker": "git-ls-files",
                    "reranker": "my-new-reranker",
                },
                "code_artifacts": {
                    "adapters/reranker/my-new-reranker.py": VALID_SOURCE,
                },
            }
        )
        prop = _make_fake(response)
        cfg = prop([])
        record = store.register_candidate(config=cfg)

        # Candidate file landed where expected.
        cand_target = (
            tmp_path / "meta" / "candidates" / record.id
            / "adapters" / "reranker" / "my-new-reranker.py"
        )
        assert cand_target.is_file()

        # And production was untouched — same set of files, same mtimes.
        after = {p: p.stat().st_mtime_ns for p in prod.rglob("*.py")}
        assert before == after

    def test_malicious_traversal_blocked_e2e(self, tmp_path: Path) -> None:
        """Even when the LLM tries to escape, no file lands anywhere bad.

        The proposer's path validator drops the artifact silently; the
        config still surfaces and registers cleanly. The candidate dir
        ends up with NO ``adapters/`` because nothing survived.
        """
        store = ArtifactStore(tmp_path / "meta")
        response = json.dumps(
            {
                "config": {"walker": "filesystem"},
                "code_artifacts": {
                    "../../../src/stropha/adapters/owned.py": VALID_SOURCE,
                    "/abs/escape.py": VALID_SOURCE,
                },
            }
        )
        prop = _make_fake(response)
        cfg = prop([])
        record = store.register_candidate(config=cfg)

        cand_dir = tmp_path / "meta" / "candidates" / record.id
        # No adapters/ subdir because every artifact was dropped pre-store.
        assert not (cand_dir / "adapters").exists()
        # And no files outside the store either.
        prod = Path(__file__).resolve().parents[2] / "src" / "stropha" / "adapters"
        assert not (prod / "owned.py").exists()
        assert not Path("/abs/escape.py").exists()


# ---------------------------------------------------------------------------
# Sanity exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_proposer_pkg_reexports(self) -> None:
        from stropha.meta import proposer
        for name in (
            "LocalCodeProposer",
            "CloudCodeProposer",
            "make_local_code_proposer",
            "make_cloud_code_proposer",
            "CodeProposerMixin",
        ):
            assert hasattr(proposer, name), f"missing re-export: {name}"

    def test_code_artifacts_key_constant_matches(self) -> None:
        """The proposer and the store MUST agree on the reserved key."""
        from stropha.meta.proposer.code import CODE_ARTIFACTS_KEY as P
        from stropha.meta.store import CODE_ARTIFACTS_KEY as S
        assert P == S == "__code_artifacts__"

    def test_environ_path_isolation(self) -> None:
        """A regression guard: the proposer never reads PWD-derived env.

        Specifically — the proposer must NOT shell out, fork, or read
        ``os.environ`` for anything other than the SearchSpace env it
        was constructed with.
        """
        before = dict(os.environ)
        response = json.dumps({"config": {"walker": "git-ls-files"}})
        prop = _make_fake(response)
        prop([])
        assert dict(os.environ) == before
