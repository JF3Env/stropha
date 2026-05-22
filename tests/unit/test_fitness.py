"""Unit tests for the multi-objective ScoreCard (WIN-169).

Covers:
- The four DIMENSION_SIGNS (quality + context_efficiency + outcome_score
  maximised; latency_ms minimised).
- ScoreCard.dominates implements Pareto strictly.
- build_scorecard composes correctly from per-dim inputs.
- pareto_frontier matches the dominance contract.
- outcome_score_decomposition prevents the T1-overlap collapse the
  ticket explicitly warns about.
"""

from __future__ import annotations

import pytest

from stropha.meta.fitness import (
    DIMENSION_SIGNS,
    ScoreCard,
    _context_efficiency,
    _quality,
    build_scorecard,
    outcome_score_decomposition,
    pareto_frontier,
)
from stropha.meta.outcome import Tier, TierLabel

# ---------------------------------------------------------------------------
# DIMENSION_SIGNS
# ---------------------------------------------------------------------------


class TestDimensionSigns:
    def test_quality_is_maximised(self) -> None:
        assert DIMENSION_SIGNS["quality"] == +1

    def test_context_efficiency_is_maximised(self) -> None:
        assert DIMENSION_SIGNS["context_efficiency"] == +1

    def test_outcome_score_is_maximised(self) -> None:
        assert DIMENSION_SIGNS["outcome_score"] == +1

    def test_latency_is_minimised(self) -> None:
        assert DIMENSION_SIGNS["latency_ms"] == -1

    def test_exactly_four_dimensions(self) -> None:
        assert set(DIMENSION_SIGNS) == {
            "quality", "context_efficiency", "outcome_score", "latency_ms",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestQuality:
    def test_mean_of_provided_metrics(self) -> None:
        assert _quality(recall_at_k=0.8, ndcg_at_k=0.7, mrr=0.6) == pytest.approx(0.7)

    def test_drops_none_values(self) -> None:
        # Only recall provided.
        assert _quality(recall_at_k=0.9) == pytest.approx(0.9)

    def test_all_none_returns_zero(self) -> None:
        assert _quality() == 0.0


class TestContextEfficiency:
    def test_basic_savings(self) -> None:
        # 100 returned of 1000 baseline = 90% savings.
        assert _context_efficiency(100, 1000) == pytest.approx(0.9)

    def test_zero_baseline_returns_zero(self) -> None:
        assert _context_efficiency(50, 0) == 0.0

    def test_returned_exceeds_baseline_clamps_to_zero(self) -> None:
        """Verbose enrichment may ship more tokens than the naive baseline.
        We never report a negative efficiency."""
        assert _context_efficiency(2000, 1000) == 0.0

    def test_zero_returned_is_full_savings(self) -> None:
        assert _context_efficiency(0, 1000) == 1.0


# ---------------------------------------------------------------------------
# ScoreCard
# ---------------------------------------------------------------------------


class TestScoreCardDominance:
    def test_strict_better_dominates(self) -> None:
        a = ScoreCard(quality=0.9, context_efficiency=0.5, latency_ms=10, outcome_score=0.5)
        b = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=10, outcome_score=0.5)
        assert a.dominates(b) is True
        assert b.dominates(a) is False

    def test_equal_does_not_dominate(self) -> None:
        a = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=10, outcome_score=0.5)
        b = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=10, outcome_score=0.5)
        assert a.dominates(b) is False
        assert b.dominates(a) is False

    def test_lower_latency_wins_when_other_dims_equal(self) -> None:
        fast = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=5, outcome_score=0.5)
        slow = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=50, outcome_score=0.5)
        assert fast.dominates(slow) is True
        assert slow.dominates(fast) is False

    def test_tradeoff_neither_dominates(self) -> None:
        accurate = ScoreCard(quality=0.95, latency_ms=200)
        fast = ScoreCard(quality=0.6, latency_ms=10)
        assert accurate.dominates(fast) is False
        assert fast.dominates(accurate) is False


class TestScoreCardToDict:
    def test_round_trip_keys(self) -> None:
        card = ScoreCard(
            quality=0.81234,
            context_efficiency=0.5,
            latency_ms=12.34567,
            outcome_score=0.5,
            n_queries=10,
            n_hits=42,
            tier_counts={"T1": 5, "T7": 5},
        )
        d = card.to_dict()
        for key in (
            "quality", "context_efficiency", "latency_ms", "outcome_score",
            "n_queries", "n_hits", "tier_counts",
        ):
            assert key in d
        # Rounding applied.
        assert d["quality"] == 0.81234
        assert d["latency_ms"] == 12.346


# ---------------------------------------------------------------------------
# build_scorecard
# ---------------------------------------------------------------------------


class TestBuildScorecard:
    def test_composes_all_dimensions(self) -> None:
        labels = [
            TierLabel(chunk_id="c1", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="c2", tier=Tier.T7, score=0.0, rationale=""),
        ]
        card = build_scorecard(
            recall_at_k=0.8,
            ndcg_at_k=0.7,
            mrr=0.6,
            returned_tokens=100,
            baseline_tokens=1000,
            latency_ms=42.0,
            labels=labels,
            n_queries=30,
            n_hits=50,
        )
        assert card.quality == pytest.approx(0.7)
        assert card.context_efficiency == pytest.approx(0.9)
        assert card.latency_ms == 42.0
        assert card.outcome_score == pytest.approx(0.5)  # (1 + 0) / 2
        assert card.n_queries == 30
        assert card.n_hits == 50
        assert card.tier_counts["T1"] == 1
        assert card.tier_counts["T7"] == 1

    def test_no_labels_zero_outcome(self) -> None:
        card = build_scorecard(recall_at_k=0.9)
        assert card.outcome_score == 0.0
        assert card.tier_counts == {}

    def test_no_inputs_returns_zero_card(self) -> None:
        card = build_scorecard()
        assert card.quality == 0.0
        assert card.context_efficiency == 0.0
        assert card.latency_ms == 0.0
        assert card.outcome_score == 0.0


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    def test_dominated_excluded(self) -> None:
        a = ScoreCard(quality=0.9, context_efficiency=0.9, latency_ms=10, outcome_score=0.9)
        b = ScoreCard(quality=0.5, context_efficiency=0.5, latency_ms=50, outcome_score=0.5)
        front = pareto_frontier([a, b])
        assert front == [a]

    def test_tradeoff_keeps_both(self) -> None:
        accurate = ScoreCard(quality=0.95, latency_ms=200)
        fast = ScoreCard(quality=0.6, latency_ms=10)
        front = pareto_frontier([accurate, fast])
        assert {id(c) for c in front} == {id(accurate), id(fast)}

    def test_single_returns_self(self) -> None:
        only = ScoreCard(quality=0.5)
        assert pareto_frontier([only]) == [only]

    def test_empty_returns_empty(self) -> None:
        assert pareto_frontier([]) == []


# ---------------------------------------------------------------------------
# WIN-169 — outcome_score_decomposition guard
# ---------------------------------------------------------------------------


class TestOutcomeDecomposition:
    """The critical invariant from the ticket: ``outcome_score must NOT
    collapse to T1 overlap``. The decomposition gives the proposer a way
    to verify that."""

    def test_t1_only_session_has_t1_share_100pct(self) -> None:
        labels = [
            TierLabel(chunk_id=str(i), tier=Tier.T1, score=1.0, rationale="")
            for i in range(5)
        ]
        d = outcome_score_decomposition(labels)
        assert d["t1_share"] == 1.0
        assert d["by_tier"]["T1"]["count"] == 5

    def test_mixed_session_has_t1_share_below_one(self) -> None:
        labels = [
            TierLabel(chunk_id="a", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="b", tier=Tier.T2, score=1.0, rationale=""),
            TierLabel(chunk_id="c", tier=Tier.T3, score=0.3, rationale=""),
            TierLabel(chunk_id="d", tier=Tier.T7, score=0.0, rationale=""),
        ]
        d = outcome_score_decomposition(labels)
        # T1 = 1.0; positive total = 1 + 1 + 0.3 = 2.3; share = 1/2.3 ≈ 0.43.
        assert d["t1_share"] < 0.5

    def test_empty_session(self) -> None:
        d = outcome_score_decomposition([])
        assert d == {"aggregate": 0.0, "by_tier": {}, "t1_share": 0.0}

    def test_aggregate_matches_outcome_module(self) -> None:
        from stropha.meta.outcome import aggregate_score

        labels = [
            TierLabel(chunk_id="a", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="b", tier=Tier.T6, score=-1.0, rationale=""),
        ]
        d = outcome_score_decomposition(labels)
        assert d["aggregate"] == pytest.approx(aggregate_score(labels))

    def test_negative_contributions_not_counted_in_t1_share(self) -> None:
        """T6 contributions are negative — they should NOT inflate the
        denominator of t1_share."""
        labels = [
            TierLabel(chunk_id="a", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="b", tier=Tier.T6, score=-1.0, rationale=""),
        ]
        d = outcome_score_decomposition(labels)
        # positive_total = 1.0 (only T1 positive); t1_share = 1/1 = 1.0.
        assert d["t1_share"] == 1.0


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports_fitness_surface(self) -> None:
        from stropha import meta

        assert hasattr(meta, "ScoreCard")
        assert hasattr(meta, "build_scorecard")
        assert hasattr(meta, "DIMENSION_SIGNS")
        assert hasattr(meta, "scorecard_pareto_frontier")
        assert hasattr(meta, "outcome_score_decomposition")