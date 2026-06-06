"""Hierarchical result cards (WIN-212).

A SearchHit currently returns the whole chunk's content in its
``snippet`` field. That works when the agent needs every line, but
it's wasteful when the agent is scanning a result set to decide
which spans to dig into. Returning a 60-line method body for the
agent to skim is overkill — a 1-line "what is this?" summary lets
the agent triage cheaply and request the full content only for the
spans that actually matter.

This module offers three rendering levels:

- **``line``** — single line: ``rel_path:start_line-end_line  symbol  [kind]``.
  The smallest possible card. Fits 10-20 cards in 100 tokens.
- **``paragraph``** — signature + docstring (or first comment block).
  Roughly 3-10 lines per hit. Lets the agent see what the function
  promises in its declaration without paying for the body.
- **``full``** — the chunk's full content. The classic ``snippet``
  shape. Use when the agent has narrowed down to a span it really
  needs to read.

Progressive disclosure is the contract: render at the smallest level
that lets the agent make a decision, then escalate. The expansion
tool (``get_symbol`` / ``get_file_outline``) handles the actual
deep read once the agent has picked its target.

Composition with WIN-210 (compression)
--------------------------------------

The ``paragraph`` card uses the same signature-detection patterns as
the extractive compressor — keep_signature logic factored out into
a shared helper. The full-content path is also where compression
applies; if the caller wants a compressed full card they wire
``compress=True`` on the underlying search, then call
``render_card(hit, "full")`` and get the compressed snippet.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Literal

from ..models import SearchHit
from .compression import _is_pure_comment, _is_signature

CardLevel = Literal["line", "paragraph", "full"]
ALLOWED_LEVELS: tuple[CardLevel, ...] = ("line", "paragraph", "full")


@dataclass(frozen=True)
class ResultCard:
    """A single hit rendered at one of the three disclosure levels.

    ``hit`` is the originating SearchHit kept by reference so callers
    can pivot to the full content via ``hit.chunk_id`` /
    ``hit.rel_path`` without re-running the search.
    """

    hit: SearchHit
    level: CardLevel
    text: str


def _line_card(hit: SearchHit) -> str:
    """One-line summary: ``path:lines  symbol  [kind]``.

    Designed to be scannable in a list view. Symbol is shown verbatim
    (no language-style prefix) so the agent can match against query
    tokens cheaply. Kind goes in brackets to be visually distinct.
    """
    location = f"{hit.rel_path}:{hit.start_line}-{hit.end_line}"
    sym = hit.symbol or "—"
    return f"{location}  {sym}  [{hit.kind}]"


def _paragraph_card(hit: SearchHit) -> str:
    """Signature + docstring/comment block.

    Strategy:
      1. Walk the snippet lines looking for the first signature-shaped
         line (Python def/class, Java method, JS function, etc.).
      2. Keep that line + any consecutive comment-shaped lines
         immediately under it (docstring or block comment).
      3. Multi-line triple-quoted Python docstrings are followed
         through to the closing triple-quote so the body of the
         docstring is included.
      4. Falls back to the first 5 lines when no signature is
         detected (e.g. a file-level chunk with no declaration).

    Always returns at least the first line of the snippet so the
    card is never empty when the snippet isn't.
    """
    if not hit.snippet:
        return ""
    lines = hit.snippet.splitlines()
    # ``splitlines()`` on a non-empty string always yields at least
    # one element so we don't need a second guard here.

    # 1. Find signature.
    sig_idx: int | None = None
    for idx, line in enumerate(lines):
        if _is_signature(line):
            sig_idx = idx
            break

    if sig_idx is None:
        # Fallback: first ~5 lines.
        return "\n".join(lines[:5])

    out: list[str] = [lines[sig_idx]]
    idx = sig_idx + 1
    in_docstring: str | None = None
    while idx < len(lines):
        line = lines[idx]
        stripped = line.lstrip()
        if in_docstring is not None:
            out.append(line)
            if in_docstring in stripped:
                in_docstring = None
                break  # closed → docstring done, stop here
            idx += 1
            continue
        if _is_pure_comment(line):
            out.append(line)
            for triple in ('"""', "'''"):
                if stripped.startswith(triple) and stripped.count(triple) < 2:
                    in_docstring = triple
                    break
            idx += 1
            continue
        # First non-comment, non-docstring line below the signature →
        # stop. The body is the agent's business via "full".
        break
    return "\n".join(out)


def _full_card(hit: SearchHit) -> str:
    """The chunk's full content — same shape as today's ``snippet``."""
    return hit.snippet or ""


def render_card(hit: SearchHit, level: CardLevel = "full") -> str:
    """Render ``hit`` at the requested disclosure level."""
    if level == "line":
        return _line_card(hit)
    if level == "paragraph":
        return _paragraph_card(hit)
    if level == "full":
        return _full_card(hit)
    raise ValueError(
        f"unknown card level {level!r}; expected one of {ALLOWED_LEVELS}"
    )


def make_cards(
    hits: list[SearchHit], level: CardLevel = "full",
) -> list[ResultCard]:
    """Render every hit at the given level. Convenience for the MCP
    layer which wants a parallel list of cards plus the original hits."""
    return [
        ResultCard(hit=hit, level=level, text=render_card(hit, level))
        for hit in hits
    ]


def apply_level_to_snippets(
    hits: list[SearchHit], level: CardLevel,
) -> list[SearchHit]:
    """Return a copy of ``hits`` with ``snippet`` replaced by the
    requested card-level rendering.

    Pydantic ``SearchHit`` instances are immutable; ``model_copy``
    is used to swap the field. ``level="full"`` is a no-op pass-
    through — saves work in the common case.
    """
    if level == "full":
        return list(hits)
    out: list[SearchHit] = []
    for hit in hits:
        new_snippet = render_card(hit, level)
        if hasattr(hit, "model_copy"):
            out.append(hit.model_copy(update={"snippet": new_snippet}))
        else:
            with contextlib.suppress(AttributeError):
                hit.snippet = new_snippet
            out.append(hit)
    return out


@dataclass(frozen=True)
class DecaySchedule:
    """Per-rank card-level schedule for WIN-213 relevance decay.

    The schedule answers: "given a rank, what level should the hit
    render at?". Defaults assume top_k≤10 and target the typical
    triage workflow:

    - Ranks ``0 .. full_through-1``      → ``full`` (the agent will
      almost certainly read these).
    - Ranks ``full_through .. paragraph_through-1`` → ``paragraph``
      (likely scanned for signature + docstring).
    - Ranks ``paragraph_through ..``     → ``line`` (the agent only
      needs to know the span exists).

    A monotonic boundary keeps the schedule predictable: higher rank
    can never get MORE detail than a lower rank.
    """

    full_through: int = 1
    """Number of top hits rendered at ``full``. Default 1 = top hit only."""
    paragraph_through: int = 3
    """Cumulative number of top hits rendered at ``paragraph`` or better.
    Default 3 = ranks 0, 1, 2 get full+paragraph; rank 3+ get line."""

    def level_for(self, rank: int) -> CardLevel:
        """Return the card level for the given rank."""
        if rank < self.full_through:
            return "full"
        if rank < self.paragraph_through:
            return "paragraph"
        return "line"


def apply_relevance_decay(
    hits: list[SearchHit],
    schedule: DecaySchedule | None = None,
) -> list[SearchHit]:
    """Apply WIN-213 relevance decay: a per-rank card-level schedule.

    Replaces each hit's ``snippet`` with the card text at the level
    chosen by ``schedule.level_for(hit.rank)``. Pydantic ``SearchHit``
    is immutable; ``model_copy`` swaps the snippet.

    The decay schedule defaults to ``DecaySchedule()`` (full=1,
    paragraph=3). Callers can pass a custom schedule for tighter or
    looser triage windows.
    """
    if not hits:
        return []
    sched = schedule or DecaySchedule()
    out: list[SearchHit] = []
    for hit in hits:
        # ``rank`` on SearchHit is 0-indexed; the schedule reads
        # ``0`` as "the top hit".
        level = sched.level_for(int(hit.rank))
        new_snippet = render_card(hit, level)
        if hasattr(hit, "model_copy"):
            out.append(hit.model_copy(update={"snippet": new_snippet}))
        else:
            with contextlib.suppress(AttributeError):
                hit.snippet = new_snippet
            out.append(hit)
    return out


__all__ = [
    "ALLOWED_LEVELS",
    "CardLevel",
    "DecaySchedule",
    "ResultCard",
    "apply_level_to_snippets",
    "apply_relevance_decay",
    "make_cards",
    "render_card",
]
