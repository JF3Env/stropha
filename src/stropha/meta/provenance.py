"""Memory provenance + drift detection (WIN-228).

Every project_memory entry and every episode now records the git
HEAD SHA at the moment it was written (the v11 ``provenance_commit``
column). At recall time the agent can ask: "are these memories
stale?". The answer is the diff between the recorded SHA and the
current HEAD — staleness is whether files / symbols the memory
references have changed since.

This module owns:

- :func:`current_head_commit` — fetch the current HEAD SHA via
  ``git rev-parse``.
- :func:`changed_files_since` — list paths touched between two SHAs.
- :func:`staleness_signal` — compose the above into a structured
  result the MCP tool can return.

No LLM, no embeddings — pure git invocations bounded by a short
timeout. Failures (no git, bad SHA, detached HEAD) degrade
gracefully to ``unknown`` rather than raising — staleness is a hint,
not a contract.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger(__name__)


_GIT_TIMEOUT_S = 5


@dataclass(frozen=True)
class StalenessSignal:
    """Result of a staleness check.

    ``stale`` is a tri-state: ``True`` when changed files intersect
    the memory's anchor, ``False`` when the diff is empty, and
    ``None`` when we couldn't decide (no git, bad SHA, no
    provenance_commit recorded). Callers should treat ``None`` as
    "unknown — fall back to assuming fresh".
    """

    stale: bool | None
    since_commit: str | None
    current_commit: str | None
    changed_files: list[str] = field(default_factory=list)
    """All files that changed between ``since_commit`` and
    ``current_commit``. Empty when ``stale`` is False or None.
    The agent can intersect this with its own context to decide
    if the memory still applies."""
    reason: str | None = None
    """When ``stale`` is None, the textual reason for the unknown
    verdict (no provenance, git failure, etc.)."""


def current_head_commit(repo: Path) -> str | None:
    """Return the current HEAD SHA via ``git rev-parse``.

    Returns ``None`` on any failure (no git binary, not a repo,
    detached HEAD with no commits, timeout). Never raises.
    """
    repo = repo.expanduser()
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.decode("utf-8", errors="replace").strip()
    return sha or None


def changed_files_since(
    repo: Path, since_sha: str,
) -> list[str] | None:
    """Return the set of paths changed between ``since_sha`` and HEAD.

    ``None`` is returned on any git failure (unknown SHA, no git,
    timeout, …) — the caller should treat ``None`` as "can't
    decide" rather than "no changes".
    """
    if not since_sha:
        return None
    repo = repo.expanduser()
    if not (repo / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{since_sha}..HEAD"],
            cwd=repo, capture_output=True, check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", errors="replace")
    return [line for line in text.splitlines() if line.strip()]


def staleness_signal(
    repo: Path,
    *,
    since_commit: str | None,
    anchor_files: list[str] | None = None,
    anchor_symbols: list[str] | None = None,
) -> StalenessSignal:
    """Compose a :class:`StalenessSignal` for one memory entry.

    Args:
        repo: the repo root (used by ``git rev-parse`` + ``git diff``).
        since_commit: the SHA recorded on the memory when it was
            written. ``None`` short-circuits to ``stale=None``
            (no anchor to diff against).
        anchor_files: when set, ``stale`` only flips True if one of
            these paths appears in the diff. Lets callers test
            relevance ("did the change touch what this memory was
            about?") rather than "did anything change at all".
        anchor_symbols: currently advisory — we don't have a
            symbol-level diff, so we surface the parameter for
            future tightening but do not use it for the stale flag.
    """
    if not since_commit:
        return StalenessSignal(
            stale=None,
            since_commit=None,
            current_commit=None,
            reason="no provenance_commit recorded",
        )
    head = current_head_commit(repo)
    if head is None:
        return StalenessSignal(
            stale=None,
            since_commit=since_commit,
            current_commit=None,
            reason="git rev-parse HEAD failed",
        )
    if head == since_commit:
        # Same SHA → trivially fresh.
        return StalenessSignal(
            stale=False,
            since_commit=since_commit,
            current_commit=head,
        )
    changed = changed_files_since(repo, since_commit)
    if changed is None:
        return StalenessSignal(
            stale=None,
            since_commit=since_commit,
            current_commit=head,
            reason="git diff failed (unknown SHA?)",
        )
    if not changed:
        return StalenessSignal(
            stale=False,
            since_commit=since_commit,
            current_commit=head,
        )
    if anchor_files:
        # Relevance filter — stale only when changed ∩ anchor is
        # non-empty.
        anchor_set = set(anchor_files)
        if not any(c in anchor_set for c in changed):
            return StalenessSignal(
                stale=False,
                since_commit=since_commit,
                current_commit=head,
                changed_files=changed,
                reason="files changed but none touch the memory anchor",
            )
    # Generic "something changed" case (no anchor filter or the
    # filter matched).
    return StalenessSignal(
        stale=True,
        since_commit=since_commit,
        current_commit=head,
        changed_files=changed,
    )


__all__ = [
    "StalenessSignal",
    "changed_files_since",
    "current_head_commit",
    "staleness_signal",
]
