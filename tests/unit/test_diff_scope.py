"""Unit tests for ``stropha.retrieval.diff_scope`` (WIN-211).

Covers:

- ``diff_paths`` — shells out to git, parses the output, returns the
  changed-paths set. Tests use a real temp ``git init`` repo so the
  invocations are end-to-end (no mocking the shell call).
- ``filter_hits_to_diff`` — pure filter, exercised with constructed
  ``SearchHit`` instances + bare stubs.
- Error paths — non-git directory, bad ref, missing repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from stropha.models import SearchHit
from stropha.retrieval.diff_scope import (
    DiffScopeError,
    diff_paths,
    filter_hits_to_diff,
)


def _git(repo: Path, *args: str) -> None:
    """Run a git subcommand against ``repo`` and assert success."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", *args], cwd=repo, check=True, env=env,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialise a real git repo with a baseline commit."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "test@test.com")
    _git(r, "config", "user.name", "Test")
    (r / "a.py").write_text("a = 1\n", encoding="utf-8")
    (r / "b.py").write_text("b = 2\n", encoding="utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "baseline")
    return r


def _hit(rel_path: str) -> SearchHit:
    return SearchHit(
        chunk_id=f"c-{rel_path}",
        rel_path=rel_path,
        start_line=1,
        end_line=10,
        score=0.9,
        rank=0,
        snippet="x",
        language="python",
        kind="function",
        symbol="f",
    )


# ---------------------------------------------------------------------------
# diff_paths — happy path
# ---------------------------------------------------------------------------


class TestDiffPathsWorkingTree:
    def test_empty_when_clean_tree(self, repo: Path) -> None:
        assert diff_paths(repo) == set()

    def test_unstaged_modification_visible(self, repo: Path) -> None:
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        assert "a.py" in diff_paths(repo)

    def test_staged_modification_visible_by_default(
        self, repo: Path
    ) -> None:
        """``diff HEAD`` covers both staged and unstaged → the default
        include_staged=True picks up index changes."""
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        _git(repo, "add", "-A")
        paths = diff_paths(repo, include_staged=True)
        assert "a.py" in paths

    def test_staged_only_excluded_when_include_staged_false(
        self, repo: Path
    ) -> None:
        """include_staged=False uses ``git diff`` alone (unstaged
        only). After staging, the file is no longer in the working
        tree diff."""
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")
        _git(repo, "add", "-A")
        paths = diff_paths(repo, include_staged=False)
        # No unstaged changes → empty.
        assert paths == set()

    def test_new_untracked_file_not_in_diff(
        self, repo: Path
    ) -> None:
        """``git diff`` doesn't include untracked files. The agent
        should ``git add`` first to bring them into scope."""
        (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
        paths = diff_paths(repo)
        assert "new.py" not in paths

    def test_multiple_files_in_diff(self, repo: Path) -> None:
        (repo / "a.py").write_text("a = 9\n", encoding="utf-8")
        (repo / "b.py").write_text("b = 9\n", encoding="utf-8")
        paths = diff_paths(repo)
        assert paths == {"a.py", "b.py"}


# ---------------------------------------------------------------------------
# diff_paths — ref-to-ref
# ---------------------------------------------------------------------------


class TestDiffPathsRefToRef:
    def test_ref_to_ref_diff(self, repo: Path) -> None:
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        (repo / "a.py").write_text("a = 5\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "second")
        paths = diff_paths(repo, ref_a=base_sha, ref_b="HEAD")
        assert paths == {"a.py"}

    def test_ref_a_only_defaults_to_head(self, repo: Path) -> None:
        """When only ``ref_a`` is given, ``ref_b`` defaults to HEAD."""
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        (repo / "b.py").write_text("b = 5\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "second")
        paths = diff_paths(repo, ref_a=base_sha)
        assert paths == {"b.py"}


# ---------------------------------------------------------------------------
# diff_paths — error paths
# ---------------------------------------------------------------------------


class TestDiffPathsErrors:
    def test_non_git_dir_raises(self, tmp_path: Path) -> None:
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        with pytest.raises(DiffScopeError, match="not a git repo"):
            diff_paths(not_a_repo)

    def test_bad_ref_raises(self, repo: Path) -> None:
        with pytest.raises(DiffScopeError):
            diff_paths(repo, ref_a="nonexistent-ref")

    def test_missing_repo_path_raises(self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost"
        with pytest.raises(DiffScopeError):
            diff_paths(ghost)

    def test_subprocess_timeout_wraps_in_diffscope_error(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout / OSError from ``subprocess.run`` is wrapped in
        ``DiffScopeError`` so callers have a single exception class
        to handle."""
        import subprocess as _subprocess

        def boom(*_args, **_kwargs):
            raise _subprocess.TimeoutExpired(cmd="git diff", timeout=1)

        monkeypatch.setattr(
            "stropha.retrieval.diff_scope.subprocess.run", boom,
        )
        with pytest.raises(DiffScopeError, match="git diff failed"):
            diff_paths(repo)


# ---------------------------------------------------------------------------
# filter_hits_to_diff
# ---------------------------------------------------------------------------


class TestFilterHitsToDiff:
    def test_keeps_only_hits_in_diff(self) -> None:
        hits = [_hit("a.py"), _hit("b.py"), _hit("c.py")]
        out = filter_hits_to_diff(hits, {"a.py", "c.py"})
        assert [h.rel_path for h in out] == ["a.py", "c.py"]

    def test_empty_diff_returns_empty(self) -> None:
        hits = [_hit("a.py")]
        assert filter_hits_to_diff(hits, set()) == []

    def test_empty_hits_returns_empty(self) -> None:
        assert filter_hits_to_diff([], {"a.py"}) == []

    def test_order_preserved(self) -> None:
        hits = [_hit("z.py"), _hit("a.py"), _hit("m.py")]
        out = filter_hits_to_diff(hits, {"z.py", "a.py", "m.py"})
        assert [h.rel_path for h in out] == ["z.py", "a.py", "m.py"]

    def test_hit_without_rel_path_dropped(self) -> None:
        """Defensive: a malformed mock object missing rel_path is
        silently dropped, not raised on."""

        class Stub:
            pass

        s = Stub()
        out = filter_hits_to_diff([s], {"a.py"})
        assert out == []


# ---------------------------------------------------------------------------
# Integration: SearchEngine + diff_scope via assemble_context
# ---------------------------------------------------------------------------


class TestAssembleContextDiffScope:
    """assemble_context wires diff_scope through end-to-end. We mock
    the SearchEngine to return a controlled hit list; the real git
    diff drives the filtering."""

    def _engine(self, hits: list[SearchHit]):
        from unittest.mock import MagicMock

        eng = MagicMock()
        eng.search.return_value = hits
        return eng

    def test_diff_scope_filters_to_changed_files(
        self, repo: Path
    ) -> None:
        from stropha.retrieval.assemble import assemble_context

        # Modify a.py so the diff includes only that file.
        (repo / "a.py").write_text("a = 99\n", encoding="utf-8")

        eng = self._engine([
            _hit("a.py"), _hit("b.py"), _hit("c.py"),
        ])
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
            diff_scope=True,
            diff_repo=repo,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        # Only a.py made it through the filter.
        assert [h.rel_path for h in primary.hits] == ["a.py"]

    def test_diff_scope_without_repo_falls_back(self) -> None:
        """``diff_scope=True`` with ``diff_repo=None`` is logged but
        does not crash — the bundle still ships with the primary
        hits unfiltered."""
        from stropha.retrieval.assemble import assemble_context

        hits = [_hit("a.py"), _hit("b.py")]
        eng = self._engine(hits)
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
            diff_scope=True,
            diff_repo=None,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        # All hits survive — no filter applied.
        assert len(primary.hits) == 2

    def test_diff_scope_git_failure_falls_back(
        self, tmp_path: Path
    ) -> None:
        """When the git invocation fails (not a repo), the bundle
        falls back to unfiltered primary results — logged as a
        warning, not raised."""
        from stropha.retrieval.assemble import assemble_context

        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        hits = [_hit("a.py"), _hit("b.py")]
        eng = self._engine(hits)
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
            diff_scope=True,
            diff_repo=not_a_repo,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 2

    def test_diff_scope_oversamples_to_avoid_empty_after_filter(
        self, repo: Path
    ) -> None:
        """When diff_scope is on, the engine is called with a larger
        top_k so the filter has more candidates to keep. Verifies the
        engine receives at least top_k * 5 (default oversampling).
        """
        from stropha.retrieval.assemble import assemble_context

        (repo / "a.py").write_text("a = 9\n", encoding="utf-8")

        hits = [_hit(f"file_{i}.py") for i in range(50)]
        eng = self._engine(hits)
        assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
            top_k=5,
            diff_scope=True,
            diff_repo=repo,
        )
        # The engine should have been called with top_k >= 25 (5 × 5).
        call = eng.search.call_args
        assert call.kwargs["top_k"] >= 25


# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------


class TestPublicSymbols:
    def test_imports_resolve(self) -> None:
        from stropha.retrieval.diff_scope import (  # noqa: F401
            DiffScopeError,
            diff_paths,
            filter_hits_to_diff,
        )
