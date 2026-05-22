"""Tiered outcome labeling for Meta-Harness candidates (WIN-162).

The Meta-Harness fitness function must NOT punish retrievals that were
read-only-but-correct. The naive "edit ≠ use" trap is what the paper's
§4.13 warns about. We label every chunk a retrieval surfaced with one
of seven tiers, computed *post-hoc* from three signals:

1. The commit diff (paths + symbols that actually changed).
2. The graphify symbol graph (call / definition relationships).
3. The trace stream (same-session engagement around the chunk).

Tier table::

    T1  chunk file in commit diff                        Strong +
    T2  chunk defines a symbol that appears in the diff  Strong +
    T3  1-2 graph hops from any edited symbol            Weak   +
    T4  same session triggered a follow-up MCP call      Medium +
    T5  same session re-queried (looks like the chunk
        didn't answer the question)                      Weak   -
    T6  agent called report_bad_result on this chunk     Strong -
    T7  no signal either way                             Neutral (0)

Critical: T7 is **neutral, not negative**. Many of stropha's correct
retrievals are read-only — the agent consults a definition and writes
code that does not edit that file. Penalising those would train Phase 4
to deprioritise the very retrievals that make the tool worth shipping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..logging import get_logger

log = get_logger(__name__)


class Tier(StrEnum):
    """One of the seven labels assignable to a single retrieval hit."""

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"


# Calibrated scores. The magnitudes match the qualitative strength column
# in the tier table; flip the sign for T5/T6 (negative signals).
# Tunable as needed in Phase 1 — kept on a module constant so the fitness
# function can override per-experiment if a candidate config demands it.
TIER_SCORE: dict[Tier, float] = {
    Tier.T1: 1.0,
    Tier.T2: 1.0,
    Tier.T3: 0.3,
    Tier.T4: 0.6,
    Tier.T5: -0.2,
    Tier.T6: -1.0,
    Tier.T7: 0.0,
}


class TierLabel(BaseModel):
    """The decision the labeler made about one chunk.

    ``evidence`` carries the raw signals so a human (or a downstream
    debugger) can audit *why* the tier was chosen. ``score`` is the
    numeric value Phase 1's fitness aggregator consumes.
    """

    chunk_id: str
    rel_path: str | None = None
    symbol: str | None = None
    tier: Tier
    score: float
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class GraphAdapter(Protocol):
    """Minimal slice of the graph surface the labeler depends on.

    Production callers wire this to :func:`stropha.retrieval.graph.find_related`
    (or a similar storage-backed BFS). Tests pass a plain dict-backed stub.
    The labeler only needs to know "which symbols are within N hops of
    these edited symbols?" — nothing more.
    """

    def neighbors(self, symbols: Iterable[str], depth: int) -> set[str]:
        """Return symbols within ``depth`` graph hops of any of ``symbols``."""


@dataclass
class LabelingContext:
    """Everything the labeler needs that is not on the hit itself.

    ``diff_paths`` and ``diff_symbols`` come from the post-hoc git diff.
    ``session_traces`` is the slice of the trace stream sharing this
    candidate's session. ``bad_results`` is the set of chunk_ids the
    agent flagged via ``report_bad_result`` (WIN-163). ``graph`` is the
    optional symbol-graph adapter — if omitted, T3 cannot fire.
    """

    diff_paths: set[str] = field(default_factory=set)
    diff_symbols: set[str] = field(default_factory=set)
    session_traces: list[dict[str, Any]] = field(default_factory=list)
    bad_results: set[str] = field(default_factory=set)
    graph: GraphAdapter | None = None
    requery_window_s: float = 60.0
    """Two same-session queries closer together than this count as T5."""

    graph_hops: int = 2
    """How many graph hops count as T3."""


def label_hit(hit: dict[str, Any], ctx: LabelingContext) -> TierLabel:
    """Assign a tier to a single retrieval hit.

    Resolution order (first match wins) is the same as the docstring:
    T6 → T1 → T2 → T3 → T4 → T5 → T7.

    Each branch records its evidence under the same key in
    ``label.evidence`` so the audit trail is uniform.
    """
    chunk_id = str(hit.get("chunk_id", ""))
    rel_path = hit.get("rel_path")
    symbol = hit.get("symbol")

    # T6 — explicit negative signal beats everything else. The agent
    # actively told us this chunk was wrong.
    if chunk_id in ctx.bad_results:
        return _label(
            chunk_id, rel_path, symbol,
            tier=Tier.T6,
            rationale="agent called report_bad_result on this chunk",
            evidence={"bad_results": True},
        )

    # T1 — the file actually changed in the commit.
    if rel_path and rel_path in ctx.diff_paths:
        return _label(
            chunk_id, rel_path, symbol,
            tier=Tier.T1,
            rationale=f"chunk file {rel_path} appears in commit diff",
            evidence={"diff_path_match": rel_path},
        )

    # T2 — the chunk DEFINES a symbol whose name shows up in the diff.
    # This catches "agent edited line X by reading definition Y" exactly.
    if symbol and symbol in ctx.diff_symbols:
        return _label(
            chunk_id, rel_path, symbol,
            tier=Tier.T2,
            rationale=f"chunk symbol {symbol!r} appears in changed code",
            evidence={"diff_symbol_match": symbol},
        )

    # T3 — graph proximity. The chunk's symbol is within N hops of an
    # edited symbol. Only fires when a graph adapter is wired AND the
    # chunk has a symbol to anchor on.
    if symbol and ctx.graph is not None and ctx.diff_symbols:
        try:
            reachable = ctx.graph.neighbors(ctx.diff_symbols, ctx.graph_hops)
        except Exception as exc:
            # Defensive: graph errors must not break labeling.
            log.warning("outcome.graph_neighbors_failed", error=str(exc))
            reachable = set()
        if symbol in reachable:
            return _label(
                chunk_id, rel_path, symbol,
                tier=Tier.T3,
                rationale=(
                    f"chunk symbol {symbol!r} is within {ctx.graph_hops} "
                    f"graph hops of an edited symbol"
                ),
                evidence={"graph_hops": ctx.graph_hops, "anchor_symbols": sorted(ctx.diff_symbols)},
            )

    # T4 / T5 — engagement signals from the trace stream.
    follow_up = _has_follow_up(chunk_id, symbol, ctx.session_traces)
    if follow_up:
        return _label(
            chunk_id, rel_path, symbol,
            tier=Tier.T4,
            rationale="follow-up MCP call in same session referenced this chunk/symbol",
            evidence={"follow_up": follow_up},
        )

    requery = _looks_like_requery(hit, ctx.session_traces, ctx.requery_window_s)
    if requery:
        return _label(
            chunk_id, rel_path, symbol,
            tier=Tier.T5,
            rationale=(
                f"same-session re-query within {ctx.requery_window_s:.0f}s suggests "
                f"the chunk did not satisfy the agent"
            ),
            evidence={"requery_within_s": ctx.requery_window_s},
        )

    # T7 — no signal at all. Neutral by design.
    return _label(
        chunk_id, rel_path, symbol,
        tier=Tier.T7,
        rationale="no diff / graph / engagement signal — neutral (read-only OK)",
        evidence={},
    )


def label_hits(
    hits: Iterable[dict[str, Any]],
    ctx: LabelingContext,
) -> list[TierLabel]:
    """Vector form of :func:`label_hit`."""
    return [label_hit(h, ctx) for h in hits]


def aggregate_score(labels: Iterable[TierLabel]) -> float:
    """Mean tier score across a candidate's hits.

    Returns 0.0 (neutral) for an empty input. The Phase 1 fitness
    function uses this as one of its terms; that fitness fn may also
    weight it relative to recall@K, latency, etc.
    """
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(label.score for label in labels) / len(labels)


def tier_summary(labels: Iterable[TierLabel]) -> dict[Tier, int]:
    """Count of each tier in ``labels`` — useful for cost / observability."""
    counts: dict[Tier, int] = {t: 0 for t in Tier}
    for label in labels:
        counts[label.tier] += 1
    return counts


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _label(
    chunk_id: str,
    rel_path: str | None,
    symbol: str | None,
    *,
    tier: Tier,
    rationale: str,
    evidence: dict[str, Any],
) -> TierLabel:
    return TierLabel(
        chunk_id=chunk_id,
        rel_path=rel_path,
        symbol=symbol,
        tier=tier,
        score=TIER_SCORE[tier],
        rationale=rationale,
        evidence=evidence,
    )


def _has_follow_up(
    chunk_id: str,
    symbol: str | None,
    traces: list[dict[str, Any]],
) -> bool:
    """True if a later trace in the same session referenced this chunk / symbol.

    "Referenced" means either:
    - the chunk_id appears in a subsequent trace's ``results`` list, OR
    - the chunk's symbol appears in a subsequent trace's ``query`` / ``params``.
    """
    found_self = False
    for trace in traces:
        # Skip until we've found the trace that produced this hit.
        if not found_self:
            for result in trace.get("results", []) or []:
                if result.get("chunk_id") == chunk_id:
                    found_self = True
                    break
            continue
        # We're past the producing call — does anything later mention it?
        for result in trace.get("results", []) or []:
            if result.get("chunk_id") == chunk_id:
                return True
        query = trace.get("query") or ""
        if symbol and symbol in query:
            return True
        params = trace.get("params") or {}
        for value in params.values():
            if isinstance(value, str) and symbol and symbol in value:
                return True
    return False


def _looks_like_requery(
    hit: dict[str, Any],
    traces: list[dict[str, Any]],
    window_s: float,
) -> bool:
    """True if the agent issued the same / similar query again soon after.

    A weak negative signal: the chunk we returned didn't seem to settle
    the question. Determined by finding the trace that produced this
    chunk and checking whether the very next trace has a similar query
    issued within ``window_s`` seconds.
    """
    if not traces:
        return False
    chunk_id = str(hit.get("chunk_id", ""))
    for i, trace in enumerate(traces):
        for result in trace.get("results", []) or []:
            if result.get("chunk_id") != chunk_id:
                continue
            # Found the trace that produced this hit. Is the next call a re-query?
            if i + 1 >= len(traces):
                return False
            this_query = (trace.get("query") or "").strip().lower()
            next_trace = traces[i + 1]
            next_query = (next_trace.get("query") or "").strip().lower()
            if not this_query or not next_query:
                return False
            if not _similar_queries(this_query, next_query):
                return False
            delta = _ts_delta(trace.get("ts"), next_trace.get("ts"))
            if delta is None:
                # No timestamps — be conservative and require the queries
                # to be truly similar. Already checked above; treat as re-query.
                return True
            return delta <= window_s
    return False


def _similar_queries(a: str, b: str) -> bool:
    """Token-overlap heuristic — 'similar enough to count as a re-query'.

    Threshold tuned against the WIN-162 acceptance scenarios:
    - "fsrs" vs "fsrs"               → Jaccard 1.0 (identical)
    - "how does fsrs work"
      vs "how fsrs works exactly"    → Jaccard 0.33 (paraphrase, T5)
    - "apple" vs "banana cherry"     → Jaccard 0.0 (not a re-query)

    A 0.3 threshold catches paraphrases while still excluding unrelated
    queries. Loose by design; the time window (``requery_window_s``)
    does most of the gating.
    """
    if a == b:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return (len(overlap) / len(union)) >= 0.3


def _ts_delta(a: str | None, b: str | None) -> float | None:
    """Seconds between two ISO 8601 timestamps; None when either is missing."""
    if not a or not b:
        return None
    from datetime import datetime

    try:
        ta = datetime.fromisoformat(a)
        tb = datetime.fromisoformat(b)
    except ValueError:
        return None
    return abs((tb - ta).total_seconds())
