"""Snippet-freshness assertion (WIN-247, EPIC 1.11 #85).

LLM agents cache snippets between calls — a class signature from
``search_code`` two turns ago, a function body from ``expand_chunk``
five turns ago. By the time the agent quotes that snippet in a
code edit, the file on disk may have moved, the line numbers may
have shifted, or the implementation may have changed entirely. The
agent silently quotes stale text and the PR review burns on a
broken reference.

WIN-247 ships the primitive that vets a cached snippet against the
current index state. Given a ``chunk_id`` + the snippet the agent
remembers, ``assert_snippet_freshness`` returns:

- ``fresh=True`` — the chunk still exists and its content matches.
- ``fresh=False`` — the chunk vanished or its content drifted. The
  verdict carries the current snippet so the agent can re-quote
  without another search.

Comparison is whitespace-normalised by default (trailing newlines
and per-line whitespace usually don't matter for the agent's
purpose). Strict mode disables normalisation for callers that
care about every byte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FreshnessVerdict:
    """Outcome of :func:`assert_snippet_freshness`.

    ``fresh`` is tri-state: ``True`` (chunk present + content
    matches), ``False`` (chunk vanished or content drifted),
    ``None`` (lookup raised — undetermined).
    """

    chunk_id: str
    fresh: bool | None
    reason: str = ""
    current_snippet: str | None = None
    expected_snippet: str = ""
    drift_details: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    """Strip per-line trailing whitespace + collapse blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Drop leading + trailing blank lines.
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _diff_lines(expected: str, current: str, *, limit: int = 5) -> list[str]:
    """Return up to ``limit`` short diff lines for the verdict.

    Not a full diff — just the first few lines that differ, so the
    agent can spot the drift without re-fetching the whole chunk.
    """
    exp_lines = expected.splitlines()
    cur_lines = current.splitlines()
    out: list[str] = []
    for i in range(max(len(exp_lines), len(cur_lines))):
        if i >= len(exp_lines):
            out.append(f"+L{i + 1}: {cur_lines[i]!r}")
        elif i >= len(cur_lines):
            out.append(f"-L{i + 1}: {exp_lines[i]!r}")
        elif exp_lines[i] != cur_lines[i]:
            out.append(
                f"~L{i + 1}: expected {exp_lines[i]!r} got {cur_lines[i]!r}"
            )
        if len(out) >= limit:
            break
    return out


def _source_from_hit(hit: Any) -> str | None:
    """Read the chunk's current source slice from disk, or ``None``.

    Uses the hit's ``rel_path`` + ``start_line``/``end_line`` + the repo's
    local path (``local_path`` or ``root_path``). Returns ``None`` when any
    piece is missing or the file can't be read — callers then fall back to the
    stored snippet/card. This is what lets freshness verify against real source
    rather than a structural card.
    """
    from pathlib import Path

    rel = getattr(hit, "rel_path", None)
    start = getattr(hit, "start_line", None)
    end = getattr(hit, "end_line", None)
    repo = getattr(hit, "repo", None)
    local = None
    if repo is not None:
        local = getattr(repo, "local_path", None) or getattr(repo, "root_path", None)
    if not (rel and local and isinstance(start, int) and isinstance(end, int) and start >= 1):
        return None
    try:
        text = Path(local).joinpath(rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if start > len(lines):
        return None
    return "\n".join(lines[start - 1 : end])


def assert_snippet_freshness(
    chunk_id: str,
    expected_snippet: str,
    *,
    lookup_fn: Callable[[str], Any | None],
    strict: bool = False,
) -> FreshnessVerdict:
    """Verify the cached snippet against the live index.

    Args:
        chunk_id: opaque id of the chunk to check.
        expected_snippet: the snippet text the agent remembers.
        lookup_fn: callable returning the current hit (must expose
            a ``snippet`` attribute) or ``None`` when the chunk is
            gone. Typically wraps ``Storage.lookup_by_chunk_id``.
        strict: when False (default), compares snippets after
            whitespace normalisation. When True, compares the raw
            text.

    Returns:
        :class:`FreshnessVerdict` whose ``fresh`` field is tri-state.
    """
    try:
        hit = lookup_fn(chunk_id)
    except Exception as exc:
        return FreshnessVerdict(
            chunk_id=chunk_id,
            fresh=None,
            reason=f"lookup raised: {exc!r}",
            expected_snippet=expected_snippet,
        )

    if hit is None:
        return FreshnessVerdict(
            chunk_id=chunk_id,
            fresh=False,
            reason=(
                f"chunk_id {chunk_id!r} no longer in the index "
                "(re-indexed / file deleted / id rotated)."
            ),
            expected_snippet=expected_snippet,
        )

    current_card = str(getattr(hit, "snippet", "") or "")
    current_source = _source_from_hit(hit)

    def _matches(cur: str) -> bool:
        if not cur:
            return False
        if strict:
            return cur == expected_snippet
        exp_n, cur_n = _normalise(expected_snippet), _normalise(cur)
        # Equality OR containment: a cached snippet (leaf body, sub-span, or
        # the structural card itself) is still fresh as long as it appears
        # verbatim in the current source or card.
        return bool(exp_n) and (exp_n == cur_n or exp_n in cur_n)

    # Compare against BOTH the real source slice (ground truth, read from disk)
    # and the stored snippet/card. Whatever representation the agent cached,
    # it's fresh if it still matches either. This fixes the container-chunk
    # false-drift: a class/file chunk's stored snippet is an outline CARD, not
    # source, so comparing only against it reported drift on unchanged files.
    if _matches(current_source or "") or _matches(current_card):
        return FreshnessVerdict(
            chunk_id=chunk_id,
            fresh=True,
            reason="Snippet still present in current source/index.",
            current_snippet=current_source or current_card,
            expected_snippet=expected_snippet,
        )

    if not (current_source or current_card):
        # Nothing to compare against — undetermined, not drift.
        return FreshnessVerdict(
            chunk_id=chunk_id,
            fresh=None,
            reason="chunk has no source or stored snippet to compare against.",
            expected_snippet=expected_snippet,
        )

    canonical = current_source or current_card
    return FreshnessVerdict(
        chunk_id=chunk_id,
        fresh=False,
        reason=(
            f"Snippet for {chunk_id!r} has drifted "
            f"({'strict' if strict else 'normalised/containment'} comparison): "
            "not found in current source or card."
        ),
        current_snippet=canonical,
        expected_snippet=expected_snippet,
        drift_details=_diff_lines(expected_snippet, canonical),
    )


__all__ = [
    "FreshnessVerdict",
    "assert_snippet_freshness",
]
