"""Pinned source snapshots via ``git worktree --detach`` (WIN-175).

Paper §4.1: "evaluation is the main bottleneck". Two candidates that
score against different HEADs are not comparable — the diff between
their fitness mixes "the proposer's change" with "the working tree
moved". The Meta-Harness loop fixes this by pinning every candidate
to an explicit SHA via a detached git worktree.

::

    .stropha/meta/snapshot/
      worktrees/<sha>/        # `git worktree add --detach`
        … source files …

The worktree lives until ``release()`` (or context-manager exit) so a
single process can spin up multiple isolated worktrees without bleeding
state. The directory tree is removed AND the worktree is unregistered
from git via ``git worktree remove`` so ``git worktree list`` stays
clean.

Why this module instead of just ``git checkout``: ``git checkout`` is
destructive — it mutates the user's working tree. A worktree is the
only git primitive that lets us materialise a tree at a SHA without
touching ``HEAD`` or the index.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


class SnapshotError(StrophaError):
    """Raised when a snapshot operation cannot be satisfied."""


@dataclass
class Snapshot:
    """One pinned worktree. Use as a context manager so it auto-cleans."""

    sha: str
    path: Path
    source_repo: Path

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        """Tear down the worktree. Idempotent — safe to call twice."""
        if not self.path.exists():
            return
        try:
            _run_git(
                ["worktree", "remove", "--force", str(self.path)],
                cwd=self.source_repo,
            )
        except SnapshotError as exc:
            # `git worktree remove` can fail when the path was already
            # removed out of band. Fall back to manual rm so the
            # operator never has a half-clean state.
            log.debug("snapshot.git_remove_failed", error=str(exc))
            shutil.rmtree(self.path, ignore_errors=True)
        log.info("snapshot.released", sha=self.sha, path=str(self.path))


def _run_git(args: list[str], cwd: Path) -> str:
    """Run ``git`` with shared error handling. Returns stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SnapshotError("git binary not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SnapshotError(
            f"git {' '.join(args)} failed (rc={exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    return result.stdout


def resolve_sha(ref: str, source_repo: Path) -> str:
    """Resolve ``ref`` (HEAD, branch, tag, partial SHA) to a full 40-char SHA."""
    out = _run_git(["rev-parse", ref], cwd=source_repo).strip()
    if not out or len(out) < 40:
        raise SnapshotError(f"could not resolve {ref!r} to a full SHA (got {out!r})")
    return out


def create_snapshot(
    *,
    source_repo: Path,
    ref: str = "HEAD",
    snapshot_root: Path | None = None,
) -> Snapshot:
    """Materialise ``source_repo`` @ ``ref`` into a detached worktree.

    Args:
        source_repo: the repo to clone from. Must be a git working tree.
        ref: any ref ``git rev-parse`` accepts (branch, tag, partial SHA).
            Defaults to HEAD.
        snapshot_root: where to put the worktree. Defaults to
            ``<source_repo>/.stropha/meta/snapshot/worktrees/``. The
            worktree itself lands at ``<snapshot_root>/<sha>/``.

    Returns the :class:`Snapshot` (a context manager that auto-cleans).
    Raises :class:`SnapshotError` for any git failure.
    """
    source_repo = Path(source_repo).resolve()
    try:
        inside_worktree = _run_git(
            ["rev-parse", "--is-inside-work-tree"], cwd=source_repo
        ).strip()
    except SnapshotError as exc:
        raise SnapshotError(f"not a git repo: {source_repo}") from exc
    if inside_worktree != "true":
        raise SnapshotError(f"not a git repo: {source_repo}")

    sha = resolve_sha(ref, source_repo)
    if snapshot_root is None:
        snapshot_root = source_repo / ".stropha" / "meta" / "snapshot" / "worktrees"
    snapshot_root = snapshot_root.expanduser().resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    target = snapshot_root / sha

    if target.exists():
        registered_head = _registered_worktree_head(target, source_repo)
        if registered_head is not None:
            try:
                actual_head = resolve_sha("HEAD", target)
            except SnapshotError:
                actual_head = None
            if registered_head == sha and actual_head == sha:
                log.info("snapshot.reused", sha=sha, path=str(target))
                return Snapshot(sha=sha, path=target, source_repo=source_repo)
            _run_git(
                ["worktree", "remove", "--force", str(target)],
                cwd=source_repo,
            )
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    _run_git(
        ["worktree", "add", "--detach", str(target), sha],
        cwd=source_repo,
    )
    log.info("snapshot.created", sha=sha, path=str(target))
    return Snapshot(sha=sha, path=target, source_repo=source_repo)


def _is_registered_worktree(path: Path, source_repo: Path) -> bool:
    """``True`` when git knows about ``path`` as a worktree of ``source_repo``."""
    return _registered_worktree_head(path, source_repo) is not None


def _registered_worktree_head(path: Path, source_repo: Path) -> str | None:
    """Return the registered worktree's HEAD, or ``None`` when unknown."""
    try:
        out = _run_git(["worktree", "list", "--porcelain"], cwd=source_repo)
    except SnapshotError:
        return None
    target = path.resolve()
    worktree: Path | None = None
    head: str | None = None
    for line in [*out.splitlines(), ""]:
        if not line:
            if worktree == target:
                return head
            worktree = None
            head = None
        elif line.startswith("worktree "):
            worktree = Path(line[len("worktree "):]).resolve()
        elif line.startswith("HEAD "):
            head = line[len("HEAD "):]
    return None


def list_snapshots(snapshot_root: Path) -> list[str]:
    """Enumerate SHA folders currently under ``snapshot_root``."""
    if not snapshot_root.is_dir():
        return []
    return sorted(p.name for p in snapshot_root.iterdir() if p.is_dir())


def prune_snapshots(
    *,
    source_repo: Path,
    snapshot_root: Path | None = None,
    keep: set[str] | None = None,
) -> list[str]:
    """Remove worktrees whose SHA is NOT in ``keep``.

    Returns the list of pruned SHAs. Safe to call when ``snapshot_root``
    does not exist (returns an empty list).
    """
    source_repo = Path(source_repo).resolve()
    if snapshot_root is None:
        snapshot_root = source_repo / ".stropha" / "meta" / "snapshot" / "worktrees"
    snapshot_root = Path(snapshot_root)
    if not snapshot_root.is_dir():
        return []
    keep_set = keep or set()
    pruned: list[str] = []
    for sha_dir in snapshot_root.iterdir():
        if not sha_dir.is_dir():
            continue
        if sha_dir.name in keep_set:
            continue
        with contextlib.suppress(SnapshotError):
            _run_git(
                ["worktree", "remove", "--force", str(sha_dir)],
                cwd=source_repo,
            )
        # If git left it behind (or never knew about it) — rm the dir.
        if sha_dir.exists():
            shutil.rmtree(sha_dir, ignore_errors=True)
        pruned.append(sha_dir.name)
    if pruned:
        log.info("snapshot.pruned", pruned=pruned, keep=sorted(keep_set))
    return pruned


__all__ = [
    "Snapshot",
    "SnapshotError",
    "create_snapshot",
    "list_snapshots",
    "prune_snapshots",
    "resolve_sha",
]
