"""Diff-scoped retrieval (WIN-211).

When an agent is doing PR review, the universe of relevant code is
small: the changed lines + their immediate neighbourhood. Returning
the global top-K for the query wastes context window on files the
reviewer doesn't need to think about.

This module provides:

- :func:`diff_paths` — list of relative paths touched between two
  git refs (or in the working tree / staging area).
- :func:`filter_hits_to_diff` — keep only hits whose ``rel_path``
  is in the diff set.

The intent is composition: ``SearchEngine.search()`` (or
``assemble_context``) does the usual hybrid retrieval, then the
caller filters the result list against ``diff_paths(...)`` before
packing. Search itself stays cheap (no per-query git invocation when
the caller doesn't need diff-scoping); the filter is a pure
post-step.

Why not bake it into the walker
-------------------------------

The ``GitDiffWalker`` (WIN-Phase-B incremental indexing) is for the
*indexing* path — it tells the pipeline "only re-chunk these files".
Diff-scoping at *query time* needs different semantics: we want to
filter results regardless of whether the chunks were freshly
indexed, and we want to support uncommitted changes (working tree
diff) which the indexer can't index until they're committed.

A separate query-time helper keeps the two concerns clean: the
indexer always sees the committed state; the agent's query-time
filter sees whatever the operator's working tree currently has.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


_GIT_TIMEOUT_S = 10
"""Hard cap on every ``git diff`` invocation. We never run against
arbitrary remote refs, but a corrupted repo can hang the porcelain;
the timeout protects the agent's call."""


class DiffScopeError(StrophaError):
    """Raised when the git invocation backing ``diff_paths`` fails."""


def diff_paths(
    repo: Path,
    *,
    ref_a: str | None = None,
    ref_b: str | None = None,
    include_staged: bool = True,
) -> set[str]:
    """Return relative paths touched in the diff at ``repo``.

    Args:
        repo: path to the git repository (must contain ``.git/``).
        ref_a: starting ref. ``None`` = "no ref" — the function
            returns the working-tree diff (uncommitted + optionally
            staged). When both ``ref_a`` and ``ref_b`` are set, the
            function returns ``ref_a..ref_b`` (a ref-to-ref diff).
        ref_b: ending ref. When ``ref_a`` is set and ``ref_b`` is
            None, defaults to ``HEAD``.
        include_staged: when ``ref_a`` is None, controls whether the
            staging area is included. ``True`` (default) means the
            diff covers `git diff HEAD` (committed→working tree
            including staged changes). ``False`` means
            `git diff --staged` (committed→staged only) is NOT
            added; we use `git diff` alone (unstaged only).

    Returns:
        Set of repo-relative POSIX paths. May be empty.

    Raises:
        :class:`DiffScopeError` if git is missing, the repo isn't
        recognized, or the command fails for any other reason. The
        caller should treat a raised error as "diff-scope is
        unavailable" and decide whether to fall back to the global
        result set or abort.
    """
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise DiffScopeError(f"not a git repo: {repo}")

    if ref_a is None:
        # Working-tree diff. ``git diff HEAD`` includes both staged
        # and unstaged changes against HEAD; ``git diff`` alone is
        # unstaged-only. Toggle via include_staged.
        cmd = ["git", "diff", "--name-only"]
        if include_staged:
            cmd.append("HEAD")
    else:
        # Ref-to-ref diff. Default ``ref_b`` to HEAD so common
        # "show me changes since the merge base" patterns work.
        end_ref = ref_b or "HEAD"
        cmd = ["git", "diff", "--name-only", f"{ref_a}..{end_ref}"]

    try:
        result = subprocess.run(
            cmd, cwd=repo, capture_output=True, check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise DiffScopeError(f"git diff failed: {exc}") from exc
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise DiffScopeError(
            f"git diff exited {result.returncode}: {err or '(no stderr)'}"
        )

    paths = {
        # ``git diff --name-only`` already emits POSIX-style paths.
        line.strip()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    }
    log.info(
        "diff_scope.resolved",
        repo=str(repo),
        ref_a=ref_a, ref_b=ref_b,
        include_staged=include_staged,
        path_count=len(paths),
    )
    return paths


def filter_hits_to_diff(hits: Iterable, diff_files: set[str]) -> list:
    """Return hits whose ``rel_path`` is in ``diff_files``.

    Empty ``diff_files`` returns an empty list — explicit signal that
    "no files in the diff matched" rather than "fall back to global".
    Caller is responsible for deciding what to do with the empty
    result (probably: fetch a larger candidate pool and try again,
    or report to the user that the diff covers no chunks).

    Hits without a ``rel_path`` attribute (e.g. legacy mocks) are
    silently dropped — the function never raises on shape.
    """
    matched: list = []
    for hit in hits:
        rel_path = getattr(hit, "rel_path", None)
        if not rel_path:
            continue
        if rel_path in diff_files:
            matched.append(hit)
    return matched


__all__ = [
    "DiffScopeError",
    "diff_paths",
    "filter_hits_to_diff",
]
