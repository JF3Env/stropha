"""Working-set assembler (WIN-253, EPIC 1.12 #90).

The "working set" is the chunks the agent is actively reasoning
about: every pinned chunk (WIN-218) plus the most-recent N
non-pinned chunks the agent has seen this session (WIN-217). Today
the agent re-derives it from ``seen_chunk_ids`` / ``pinned_chunk_ids``
on every turn. WIN-253 exposes it as a first-class artefact —
returned by an MCP tool and rendered as a markdown bundle that
can be plugged straight into the next LLM prompt.

Resource vs. tool
-----------------

MCP gives us two shapes:

- **Tool** — explicit request/response. Useful when the agent
  wants to query a specific session.
- **Resource** — URI-addressable read. Lower per-call overhead;
  the agent can subscribe to ``stropha://working-set`` and watch
  for updates.

WIN-253 ships both. The primitive in this module is the assembler
both shapes call into; ``server.py`` registers the wrapper tool +
the resource.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkingSetEntry:
    """One chunk in the working set."""

    chunk_id: str
    path: str = ""
    symbol: str = ""
    kind: str = ""
    start_line: int = 0
    end_line: int = 0
    snippet: str = ""
    role: str = "recent"  # "pinned" | "recent"
    tokens: int = 0


@dataclass(frozen=True)
class WorkingSet:
    """Assembled working set with rendered markdown."""

    session_id: str = ""
    entries: list[WorkingSetEntry] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    pinned_count: int = 0
    recent_count: int = 0
    dropped_count: int = 0
    markdown: str = ""
    notes: list[str] = field(default_factory=list)


LookupFn = Callable[[str], Any | None]
EstimatorFn = Callable[[str], int]


def _default_estimator(text: str) -> int:
    """Rough 4-chars-per-token proxy when the caller doesn't supply
    a real tokeniser."""
    return max(1, len(text) // 4)


def _to_entry(hit: Any, *, role: str, estimator: EstimatorFn) -> WorkingSetEntry:
    snippet = str(getattr(hit, "snippet", "") or "")
    return WorkingSetEntry(
        chunk_id=str(getattr(hit, "chunk_id", "") or ""),
        path=str(getattr(hit, "rel_path", "") or ""),
        symbol=str(getattr(hit, "symbol", "") or ""),
        kind=str(getattr(hit, "kind", "") or ""),
        start_line=int(getattr(hit, "start_line", 0) or 0),
        end_line=int(getattr(hit, "end_line", 0) or 0),
        snippet=snippet,
        role=role,
        tokens=estimator(snippet),
    )


def _render_markdown(
    entries: list[WorkingSetEntry], *, session_id: str,
) -> str:
    """Render entries as a markdown bundle.

    Pinned chunks appear in a ``## Pinned`` section first, then
    recent ones in ``## Recent``. Each entry shows the path + line
    range as a code-block header followed by the snippet.
    """
    lines: list[str] = []
    if session_id:
        lines.append(f"# Working set — session `{session_id}`")
    else:
        lines.append("# Working set")
    lines.append("")

    pinned = [e for e in entries if e.role == "pinned"]
    recent = [e for e in entries if e.role == "recent"]

    if pinned:
        lines.append("## Pinned")
        lines.append("")
        for entry in pinned:
            lines.extend(_render_entry(entry))
    if recent:
        if pinned:
            lines.append("")
        lines.append("## Recent")
        lines.append("")
        for entry in recent:
            lines.extend(_render_entry(entry))
    if not pinned and not recent:
        lines.append("_no working-set entries — pin a chunk or run a search._")

    return "\n".join(lines) + "\n"


def _render_entry(entry: WorkingSetEntry) -> list[str]:
    """Render a single working-set entry as markdown lines."""
    header = f"### {entry.path or '?'}:{entry.start_line}-{entry.end_line}"
    if entry.symbol:
        header += f" — `{entry.symbol}`"
    if entry.kind:
        header += f" ({entry.kind})"
    lines = [header, "", f"_chunk_: `{entry.chunk_id}`", ""]
    if entry.snippet:
        lines.append("```")
        lines.append(entry.snippet.rstrip())
        lines.append("```")
        lines.append("")
    return lines


def build_working_set(
    pinned_chunk_ids: list[str],
    recent_chunk_ids: list[str],
    *,
    lookup_fn: LookupFn,
    session_id: str = "",
    budget_tokens: int = 4000,
    estimator: EstimatorFn = _default_estimator,
    max_recent: int = 20,
) -> WorkingSet:
    """Assemble the working set for a session.

    Args:
        pinned_chunk_ids: chunks the agent has explicitly pinned
            (WIN-218). Always included; never dropped by budget.
        recent_chunk_ids: chunks the agent has seen recently
            (WIN-217). Truncated to ``max_recent`` then packed
            under the remaining budget after pinned chunks consume
            their share.
        lookup_fn: callable returning a hit-like object (or
            ``None``) for a given chunk_id. Typically
            ``Storage.lookup_by_chunk_id``.
        session_id: optional — surfaces in the rendered markdown
            header.
        budget_tokens: total token budget for the assembled set.
            Pinned chunks ALWAYS consume their share first; the
            remainder is split across the most-recent entries.
        estimator: per-text token estimator. Default is a cheap
            4-chars-per-token proxy.
        max_recent: ceiling on the recent list length BEFORE
            budgeting. Defaults to 20.

    Returns:
        :class:`WorkingSet` with deduplicated entries, total token
        spend, dropped count, and a rendered markdown bundle.
    """
    notes: list[str] = []
    seen: set[str] = set()
    entries: list[WorkingSetEntry] = []
    total = 0
    dropped = 0
    pinned_count = 0

    def _resolve(cid: str, role: str) -> WorkingSetEntry | None:
        try:
            hit = lookup_fn(cid)
        except Exception as exc:
            notes.append(f"lookup raised for {cid!r}: {exc!r}")
            return None
        if hit is None:
            return None
        return _to_entry(hit, role=role, estimator=estimator)

    # Pinned chunks: always included, deduplicated.
    for cid in pinned_chunk_ids:
        cid_clean = (cid or "").strip()
        if not cid_clean or cid_clean in seen:
            continue
        seen.add(cid_clean)
        entry = _resolve(cid_clean, "pinned")
        if entry is None:
            continue
        entries.append(entry)
        total += entry.tokens
        pinned_count += 1

    # Recent chunks: truncate to max_recent, then pack under the
    # remaining budget. The cap is checked BEFORE the append so
    # ``max_recent=0`` actually yields zero recent entries.
    recent_pool: list[str] = []
    cap = max(0, int(max_recent))
    for cid in reversed(recent_chunk_ids):  # most-recent first
        cid_clean = (cid or "").strip()
        if not cid_clean or cid_clean in seen:
            continue
        if len(recent_pool) >= cap:
            break
        seen.add(cid_clean)
        recent_pool.append(cid_clean)

    recent_count = 0
    for cid in recent_pool:
        entry = _resolve(cid, "recent")
        if entry is None:
            continue
        if budget_tokens > 0 and total + entry.tokens > budget_tokens:
            dropped += 1
            continue
        entries.append(entry)
        total += entry.tokens
        recent_count += 1

    if budget_tokens > 0 and total > budget_tokens:
        notes.append(
            f"pinned chunks alone exceed budget ({total} > {budget_tokens})"
        )

    markdown = _render_markdown(entries, session_id=session_id)

    return WorkingSet(
        session_id=session_id,
        entries=entries,
        total_tokens=total,
        budget_tokens=budget_tokens,
        pinned_count=pinned_count,
        recent_count=recent_count,
        dropped_count=dropped,
        markdown=markdown,
        notes=notes,
    )


__all__ = [
    "WorkingSet",
    "WorkingSetEntry",
    "build_working_set",
]
