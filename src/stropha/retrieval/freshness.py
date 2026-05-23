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

    current = str(getattr(hit, "snippet", "") or "")

    if strict:
        match = current == expected_snippet
    else:
        match = _normalise(current) == _normalise(expected_snippet)

    if match:
        return FreshnessVerdict(
            chunk_id=chunk_id,
            fresh=True,
            reason="Snippet matches current index content.",
            current_snippet=current,
            expected_snippet=expected_snippet,
        )

    diff = _diff_lines(expected_snippet, current)
    return FreshnessVerdict(
        chunk_id=chunk_id,
        fresh=False,
        reason=(
            f"Snippet for {chunk_id!r} has drifted "
            f"({'strict' if strict else 'normalised'} comparison)."
        ),
        current_snippet=current,
        expected_snippet=expected_snippet,
        drift_details=diff,
    )


__all__ = [
    "FreshnessVerdict",
    "assert_snippet_freshness",
]
