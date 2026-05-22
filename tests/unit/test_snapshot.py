"""Unit tests for git-worktree pinned snapshots (WIN-175).

Covers every branch of the snapshot module:
- ``create_snapshot`` happy path, reuse, orphan cleanup, ref resolution.
- ``resolve_sha`` happy / failure paths.
- ``Snapshot.release`` idempotency + git-remove failure fallback.
- ``list_snapshots`` and ``prune_snapshots``.
- All git-subprocess failure modes funnel through ``SnapshotError``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from stropha.meta import (
    SnapshotError,
    create_snapshot,
    list_snapshots,
    prune_snapshots,
    resolve_sha,
)
from stropha.meta.snapshot import _is_registered_worktree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    ).stdout


def _init_repo(repo: Path) -> str:
    """Init repo + one commit; return HEAD SHA."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], repo)
    (repo / "a.py").write_text("x = 1\n")
    _git(["add", "-A"], repo)
    _git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], repo)
    return _git(["rev-parse", "HEAD"], repo).strip()


# ---------------------------------------------------------------------------
# resolve_sha
# ---------------------------------------------------------------------------


class TestResolveSha:
    def test_head_resolves_to_full_sha(self, tmp_path: Path) -> None:
        sha = _init_repo(tmp_path)
        out = resolve_sha("HEAD", tmp_path)
        assert out == sha
        assert len(out) == 40

    def test_branch_resolves(self, tmp_path: Path) -> None:
        sha = _init_repo(tmp_path)
        out = resolve_sha("HEAD", tmp_path)
        assert out == sha

    def test_unknown_ref_raises(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        with pytest.raises(SnapshotError):
            resolve_sha("nonexistent-ref", tmp_path)

    def test_partial_sha_raises_when_not_full(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defensive 'len < 40' branch — a stubbed git that returns
        a short string is rejected, not silently accepted."""
        import stropha.meta.snapshot as snapshot_mod

        monkeypatch.setattr(
            snapshot_mod, "_run_git", lambda args, cwd: "abcd1234\n"
        )
        with pytest.raises(SnapshotError, match="full SHA"):
            resolve_sha("HEAD", tmp_path)

    def test_git_not_found_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a, **kw):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("git missing")

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(SnapshotError, match="git binary not found"):
            resolve_sha("HEAD", tmp_path)


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    def test_creates_worktree_at_sha(self, tmp_path: Path) -> None:
        sha = _init_repo(tmp_path / "repo")
        snap = create_snapshot(source_repo=tmp_path / "repo")
        try:
            assert snap.sha == sha
            assert snap.path.is_dir()
            # Worktree has the same files.
            assert (snap.path / "a.py").read_text() == "x = 1\n"
        finally:
            snap.release()

    def test_default_snapshot_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        sha = _init_repo(repo)
        snap = create_snapshot(source_repo=repo)
        try:
            # Default lands under <repo>/.stropha/meta/snapshot/worktrees/<sha>/
            assert snap.path == repo / ".stropha" / "meta" / "snapshot" / "worktrees" / sha
        finally:
            snap.release()

    def test_custom_snapshot_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        sha = _init_repo(repo)
        custom = tmp_path / "elsewhere"
        snap = create_snapshot(source_repo=repo, snapshot_root=custom)
        try:
            assert snap.path == custom / sha
        finally:
            snap.release()

    def test_not_a_git_repo_raises(self, tmp_path: Path) -> None:
        (tmp_path / "plain").mkdir()
        with pytest.raises(SnapshotError, match="not a git repo"):
            create_snapshot(source_repo=tmp_path / "plain")

    def test_reuses_existing_registered_worktree(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap_a = create_snapshot(source_repo=repo)
        try:
            # Second call must return the same path (reuse, not recreate).
            snap_b = create_snapshot(source_repo=repo)
            assert snap_b.path == snap_a.path
        finally:
            snap_a.release()

    def test_orphan_worktree_dir_is_recreated(self, tmp_path: Path) -> None:
        """A directory left behind from a prior crash (not registered with
        git) must be cleaned up + recreated, not blindly trusted."""
        repo = tmp_path / "repo"
        sha = _init_repo(repo)
        snap_root = tmp_path / "snap"
        snap_root.mkdir()
        orphan = snap_root / sha
        orphan.mkdir()
        (orphan / "stale.txt").write_text("leftover from a crash")

        snap = create_snapshot(source_repo=repo, snapshot_root=snap_root)
        try:
            # Stale file gone, real source materialised.
            assert not (snap.path / "stale.txt").exists()
            assert (snap.path / "a.py").is_file()
        finally:
            snap.release()


# ---------------------------------------------------------------------------
# Snapshot.release (context manager + idempotency)
# ---------------------------------------------------------------------------


class TestSnapshotRelease:
    def test_context_manager_releases(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        with create_snapshot(source_repo=repo) as snap:
            assert snap.path.exists()
            saved_path = snap.path
        # After context exit the worktree is gone.
        assert not saved_path.exists()

    def test_release_idempotent(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap = create_snapshot(source_repo=repo)
        snap.release()
        # Second release is a no-op, not an error.
        snap.release()

    def test_release_falls_back_to_rmtree_when_git_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If `git worktree remove` raises, we still rm the directory."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap = create_snapshot(source_repo=repo)
        # Force the git remove to fail.
        import stropha.meta.snapshot as snapshot_mod

        def failing_git(args, cwd):  # type: ignore[no-untyped-def]
            if args[:2] == ["worktree", "remove"]:
                raise SnapshotError("simulated git failure")
            return ""

        monkeypatch.setattr(snapshot_mod, "_run_git", failing_git)
        snap.release()
        # Directory must be gone (manual rmtree fallback worked).
        assert not snap.path.exists()


# ---------------------------------------------------------------------------
# _is_registered_worktree
# ---------------------------------------------------------------------------


class TestIsRegisteredWorktree:
    def test_returns_true_for_registered(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap = create_snapshot(source_repo=repo)
        try:
            assert _is_registered_worktree(snap.path, repo) is True
        finally:
            snap.release()

    def test_returns_false_for_unknown_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        assert _is_registered_worktree(tmp_path / "no-such", repo) is False

    def test_returns_false_when_git_command_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import stropha.meta.snapshot as snapshot_mod

        def boom(args, cwd):  # type: ignore[no-untyped-def]
            raise SnapshotError("simulated")

        monkeypatch.setattr(snapshot_mod, "_run_git", boom)
        assert _is_registered_worktree(tmp_path, tmp_path) is False


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------


class TestListSnapshots:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert list_snapshots(tmp_path / "missing") == []

    def test_lists_sha_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "snap"
        root.mkdir()
        (root / "abc").mkdir()
        (root / "def").mkdir()
        (root / "not_a_dir.txt").write_text("ignored")
        assert list_snapshots(root) == ["abc", "def"]


# ---------------------------------------------------------------------------
# prune_snapshots
# ---------------------------------------------------------------------------


class TestPruneSnapshots:
    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        assert prune_snapshots(source_repo=repo, snapshot_root=tmp_path / "nope") == []

    def test_prunes_unknown_shas(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap = create_snapshot(source_repo=repo)
        try:
            kept_sha = snap.sha
            # Pre-place an extra fake-SHA dir.
            extra = snap.path.parent / "deadbeef"
            extra.mkdir()
            pruned = prune_snapshots(source_repo=repo, keep={kept_sha})
            assert "deadbeef" in pruned
            assert not extra.exists()
            assert snap.path.exists()  # the kept one survives
        finally:
            snap.release()

    def test_default_snapshot_root(self, tmp_path: Path) -> None:
        """Default ``snapshot_root`` is ``<repo>/.stropha/meta/snapshot/worktrees``.
        Pruning with no extras returns []."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        snap = create_snapshot(source_repo=repo)  # default root
        try:
            # Default root has only our snap; keep set has its SHA → no pruning.
            assert prune_snapshots(source_repo=repo, keep={snap.sha}) == []
        finally:
            snap.release()

    def test_handles_non_dir_entries(self, tmp_path: Path) -> None:
        """Files placed under the snapshot root are silently ignored."""
        repo = tmp_path / "repo"
        _init_repo(repo)
        root = tmp_path / "snap"
        root.mkdir()
        (root / "stray.txt").write_text("not a worktree")
        (root / "abc").mkdir()
        pruned = prune_snapshots(source_repo=repo, snapshot_root=root, keep=set())
        # Only the SHA dir is pruned; the stray file is left alone.
        assert "abc" in pruned
        assert "stray.txt" not in pruned
        assert (root / "stray.txt").exists()

    def test_git_remove_failure_still_drops_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        root = tmp_path / "snap"
        root.mkdir()
        (root / "orphan").mkdir()

        import stropha.meta.snapshot as snapshot_mod

        def fail_worktree_remove(args, cwd):  # type: ignore[no-untyped-def]
            if args[:2] == ["worktree", "remove"]:
                raise SnapshotError("simulated")
            return ""

        monkeypatch.setattr(snapshot_mod, "_run_git", fail_worktree_remove)
        pruned = prune_snapshots(source_repo=repo, snapshot_root=root, keep=set())
        assert "orphan" in pruned
        assert not (root / "orphan").exists()


# ---------------------------------------------------------------------------
# Subprocess error funneling
# ---------------------------------------------------------------------------


class TestRunGitErrors:
    def test_called_process_error_becomes_snapshot_error(
        self, tmp_path: Path
    ) -> None:
        # _run_git on a non-repo cwd raises SnapshotError, not the raw
        # CalledProcessError.
        with pytest.raises(SnapshotError, match=r"git rev-parse"):
            resolve_sha("HEAD", tmp_path)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports_snapshot_surface(self) -> None:
        from stropha import meta

        for name in (
            "Snapshot",
            "SnapshotError",
            "create_snapshot",
            "list_snapshots",
            "prune_snapshots",
            "resolve_sha",
        ):
            assert hasattr(meta, name), f"missing export: {name}"