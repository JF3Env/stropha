"""Multi-objective fitness for Meta-Harness candidates (WIN-169).

A candidate's ScoreCard is a *vector*, not a scalar. Collapsing to a
single number is the trap Lee et al. §4.3 (Alg.1 line 13) warns about
— two candidates with identical fitness can be on very different
trade-off frontiers and a scalarised proposer would learn to pick the
short-term winner instead of preserving diversity.

Three dimensions, each with an explicit sign:

| Dim                   | Sign | Source                              |
|-----------------------|------|-------------------------------------|
| ``quality``           |  +   | Recall@K + NDCG@K + MRR mean        |
| ``context_efficiency``|  +   | (1 - returned_tokens / baseline)     |
| ``latency_ms``        |  -   | p95 wall-clock per query             |
| ``outcome_score``     |  +   | Aggregate T1-T7 score (WIN-162)      |

``outcome_score`` is the most important one to keep distinct. The
critical guard in this module: ``outcome_score`` MUST NOT collapse to
``T1`` overlap. T1 is "file in diff" — easy to game by retrieving
every changed file. The full T1-T7 distribution (which T7 = neutral
prevents from being negative) is what we actually want the proposer
to optimise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .outcome import TierLabel, aggregate_score, tier_summary

# Sign per dimension. +1 = maximise, -1 = minimise. The ArtifactStore's
# ``pareto_frontier`` consumes this exact dict.
DIMENSION_SIGNS: dict[str, int] = {
    "quality": +1,
    "context_efficiency": +1,
    "outcome_score": +1,
    "latency_ms": -1,
}


@dataclass
class ScoreCard:
    """Multi-objective fitness vector for one candidate.

    Fields are populated by :func:`build_scorecard`. ``to_dict`` returns a
    JSON-safe representation suitable for ``scores.json`` (WIN-161
    ArtifactStore.write_scores).
    """

    quality: float = 0.0
    context_efficiency: float = 0.0
    latency_ms: float = 0.0
    outcome_score: float = 0.0
    # Provenance — these are not optimisation dimensions, they are
    # context that lets the proposer reason about the score.
    n_queries: int = 0
    n_hits: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict ready for json.dump."""
        return {
            "quality": round(self.quality, 6),
            "context_efficiency": round(self.context_efficiency, 6),
            "latency_ms": round(self.latency_ms, 3),
            "outcome_score": round(self.outcome_score, 6),
            "n_queries": self.n_queries,
            "n_hits": self.n_hits,
            "tier_counts": dict(self.tier_counts),
        }

    def dominates(self, other: ScoreCard) -> bool:
        """True when ``self`` Pareto-dominates ``other`` across DIMENSION_SIGNS.

        Dominance: ``self`` is >= ``other`` on every dimension and > on
        at least one — after the sign flip so 'larger is always better'.
        """
        strict = False
        for dim, sign in DIMENSION_SIGNS.items():
            s = getattr(self, dim) * sign
            o = getattr(other, dim) * sign
            if s < o:
                return False
            if s > o:
                strict = True
        return strict


def _quality(
    *,
    recall_at_k: float | None = None,
    ndcg_at_k: float | None = None,
    mrr: float | None = None,
) -> float:
    """Mean of available retrieval-quality metrics in [0, 1].

    Skips ``None`` values so callers can pass whatever EvalReport
    surfaces without having to default-fill the missing ones.
    """
    parts = [m for m in (recall_at_k, ndcg_at_k, mrr) if m is not None]
    if not parts:
        return 0.0
    return sum(parts) / len(parts)


def _context_efficiency(returned_tokens: int, baseline_tokens: int) -> float:
    """``1 - returned / baseline`` clamped to ``[0, 1]``.

    Higher = fewer tokens shipped per query relative to the naive baseline
    (matches ``retrieval/savings.py``). Returns 0 when baseline is missing
    or smaller than returned (edge case where retrieval shipped more
    tokens than the naive read — possible with verbose enrichment).
    """
    if baseline_tokens <= 0:
        return 0.0
    ratio = returned_tokens / baseline_tokens
    return max(0.0, min(1.0, 1.0 - ratio))


def build_scorecard(
    *,
    recall_at_k: float | None = None,
    ndcg_at_k: float | None = None,
    mrr: float | None = None,
    returned_tokens: int = 0,
    baseline_tokens: int = 0,
    latency_ms: float = 0.0,
    labels: Iterable[TierLabel] | None = None,
    n_queries: int = 0,
    n_hits: int = 0,
) -> ScoreCard:
    """Assemble a :class:`ScoreCard` from per-dimension inputs.

    The caller is responsible for the inputs — this is the
    composition root for the fitness vector but not the evaluator (that
    lives in :mod:`stropha.meta.evaluator`, WIN-171).
    """
    labels_list = list(labels or [])
    outcome = aggregate_score(labels_list) if labels_list else 0.0
    tier_counts: dict[str, int] = {}
    if labels_list:
        tier_counts = {t.value: c for t, c in tier_summary(labels_list).items()}
    return ScoreCard(
        quality=_quality(recall_at_k=recall_at_k, ndcg_at_k=ndcg_at_k, mrr=mrr),
        context_efficiency=_context_efficiency(returned_tokens, baseline_tokens),
        latency_ms=latency_ms,
        outcome_score=outcome,
        n_queries=n_queries,
        n_hits=n_hits,
        tier_counts=tier_counts,
    )


def pareto_frontier(cards: Iterable[ScoreCard]) -> list[ScoreCard]:
    """Return the Pareto-non-dominated subset of ``cards``.

    Same contract as :meth:`stropha.meta.store.ArtifactStore.pareto_frontier`
    but operating on already-built ScoreCards instead of candidate
    records. Useful for in-memory experiments where the cards are not
    yet persisted.
    """
    cards_list = list(cards)
    frontier: list[ScoreCard] = []
    for cand in cards_list:
        dominated = any(
            other is not cand and other.dominates(cand) for other in cards_list
        )
        if not dominated:
            frontier.append(cand)
    return frontier


def outcome_score_decomposition(labels: Iterable[TierLabel]) -> dict[str, Any]:
    """Decomposition of the outcome score by tier — the WIN-169 guard.

    Returns a dict with per-tier counts AND the contribution each tier
    made to the aggregate. The proposer (or an operator) can verify the
    score did NOT collapse to T1 overlap (i.e. T1 alone < 100% of the
    positive contribution).
    """
    labels_list = list(labels)
    if not labels_list:
        return {"aggregate": 0.0, "by_tier": {}, "t1_share": 0.0}

    by_tier_count = tier_summary(labels_list)
    by_tier_contribution: dict[str, float] = {}
    for label in labels_list:
        by_tier_contribution[label.tier.value] = (
            by_tier_contribution.get(label.tier.value, 0.0) + label.score
        )

    positive_total = sum(c for c in by_tier_contribution.values() if c > 0)
    t1_contribution = max(0.0, by_tier_contribution.get("T1", 0.0))
    t1_share = (t1_contribution / positive_total) if positive_total > 0 else 0.0

    return {
        "aggregate": aggregate_score(labels_list),
        "by_tier": {
            t.value: {
                "count": by_tier_count[t],
                "contribution": round(by_tier_contribution.get(t.value, 0.0), 6),
            }
            for t in by_tier_count
        },
        "t1_share": round(t1_share, 4),
    }


__all__ = [
    "DIMENSION_SIGNS",
    "ScoreCard",
    "build_scorecard",
    "outcome_score_decomposition",
    "pareto_frontier",
]
