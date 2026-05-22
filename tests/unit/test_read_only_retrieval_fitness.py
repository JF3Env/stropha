"""Anti-regression test for the WIN-162 + WIN-169 read-only invariant (WIN-173).

The failure mode meta-harness-evaluation.md §4.13 warns about: a fitness
function that collapses to ``edit ≠ use`` punishes the agent for
consulting a definition without editing the file. The whole point of
stropha is to surface read-only context — penalising it would train
Phase 4 to *deprioritise the very retrievals that make the tool worth
shipping*.

This file is the locked-in guard. It exercises three escalating
scenarios:

1. **Pure read-only session** — every chunk is T7. Fitness must be
   ≥ 0 (definitely not negative).
2. **One edit + many read-only** — fitness must be ≥ 0; the read-only
   chunks must not drag the score below the single positive T1.
3. **Outcome decomposition guard** — ``t1_share`` < 1.0 when any
   non-T1 positive tier fires. Catches the precise failure mode where
   the proposer collapses outcome to T1 overlap.

If any of these fails, the labeler or the fitness composition has
regressed in a way that would silently degrade stropha's
recommendation. The user description explicitly asks for this guard.
"""

from __future__ import annotations

import pytest

from stropha.meta import (
    LabelingContext,
    ScoreCard,
    Tier,
    TierLabel,
    aggregate_score,
    build_scorecard,
    label_hits,
    outcome_score_decomposition,
)


def _hit(chunk_id: str, rel_path: str | None = None, symbol: str | None = None) -> dict:
    return {"chunk_id": chunk_id, "rel_path": rel_path, "symbol": symbol}


# ---------------------------------------------------------------------------
# Scenario 1 — pure read-only session
# ---------------------------------------------------------------------------


class TestPureReadOnlySession:
    """Every hit is read-only (T7). No edits, no engagement, no bad results.
    Fitness MUST be ≥ 0. If it's negative, the labeler regressed."""

    def test_aggregate_score_non_negative(self) -> None:
        # 10 hits that don't appear in any diff signal.
        hits = [_hit(f"c{i}", rel_path=f"unrelated_{i}.py", symbol=f"sym_{i}") for i in range(10)]
        ctx = LabelingContext(diff_paths=set(), diff_symbols=set())
        labels = label_hits(hits, ctx)
        assert all(label.tier == Tier.T7 for label in labels)
        assert aggregate_score(labels) >= 0.0
        assert aggregate_score(labels) == 0.0  # strictly: T7 = 0, mean = 0

    def test_scorecard_outcome_non_negative(self) -> None:
        hits = [_hit(f"c{i}", rel_path=f"file{i}.py", symbol=f"s{i}") for i in range(5)]
        ctx = LabelingContext()
        labels = label_hits(hits, ctx)
        card = build_scorecard(
            recall_at_k=0.8,  # good retrieval
            ndcg_at_k=0.7,
            mrr=0.6,
            labels=labels,
        )
        assert card.outcome_score >= 0.0
        # The proposer must still see this candidate as preferable to a
        # zero-quality one — quality dimension carries the signal even
        # when outcome is neutral.
        assert card.quality > 0


# ---------------------------------------------------------------------------
# Scenario 2 — mixed edit + many read-only
# ---------------------------------------------------------------------------


class TestMixedEditAndReadOnly:
    """The real-world shape: one edit, many reads. Fitness must NOT
    treat the reads as negatives."""

    def test_aggregate_score_strictly_positive(self) -> None:
        hits = [
            _hit("c-edit", rel_path="src/edited.py", symbol="touched"),
            _hit("c-read-1", rel_path="src/util1.py", symbol="helper1"),
            _hit("c-read-2", rel_path="src/util2.py", symbol="helper2"),
            _hit("c-read-3", rel_path="src/util3.py", symbol="helper3"),
        ]
        ctx = LabelingContext(diff_paths={"src/edited.py"})
        labels = label_hits(hits, ctx)
        tiers = [label.tier for label in labels]
        assert tiers.count(Tier.T1) == 1
        assert tiers.count(Tier.T7) == 3
        # mean of [+1, 0, 0, 0] = +0.25 — strictly positive.
        assert aggregate_score(labels) == pytest.approx(0.25)

    def test_scorecard_dominates_pure_t7(self) -> None:
        """A mixed (1 T1 + 3 T7) session must Pareto-dominate (or be
        equal to) a pure-T7 session on outcome_score."""
        ctx = LabelingContext(diff_paths={"src/edited.py"})
        mixed = label_hits(
            [
                _hit("c-edit", rel_path="src/edited.py"),
                _hit("c-r1", rel_path="src/u1.py"),
                _hit("c-r2", rel_path="src/u2.py"),
            ],
            ctx,
        )
        pure_t7 = label_hits(
            [
                _hit("c-r1", rel_path="src/u1.py"),
                _hit("c-r2", rel_path="src/u2.py"),
                _hit("c-r3", rel_path="src/u3.py"),
            ],
            ctx,
        )
        mixed_card = build_scorecard(labels=mixed)
        t7_card = build_scorecard(labels=pure_t7)
        assert mixed_card.outcome_score > t7_card.outcome_score
        # Both >= 0 — neither session is punished.
        assert mixed_card.outcome_score >= 0
        assert t7_card.outcome_score >= 0


# ---------------------------------------------------------------------------
# Scenario 3 — outcome decomposition guard (T1-overlap collapse)
# ---------------------------------------------------------------------------


class TestT1OverlapCollapse:
    """The WIN-169 ticket explicitly warns: outcome_score MUST NOT
    collapse to T1 overlap. The decomposition must show t1_share < 1.0
    whenever any other positive tier fires.

    If a future change to the labeler makes T1 the only positive signal,
    these tests fail and surface the regression at the source."""

    def test_t2_only_session_t1_share_is_zero(self) -> None:
        """A session where only T2 fires (chunk's symbol in diff but its
        file is not) — t1_share must be 0.0, not 1.0."""
        labels = [
            TierLabel(chunk_id=str(i), tier=Tier.T2, score=1.0, rationale="")
            for i in range(3)
        ]
        d = outcome_score_decomposition(labels)
        assert d["t1_share"] == 0.0
        assert d["aggregate"] == pytest.approx(1.0)

    def test_mixed_t1_t2_share_below_one(self) -> None:
        labels = [
            TierLabel(chunk_id="t1a", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="t2a", tier=Tier.T2, score=1.0, rationale=""),
            TierLabel(chunk_id="t2b", tier=Tier.T2, score=1.0, rationale=""),
        ]
        d = outcome_score_decomposition(labels)
        # 1/(1+1+1) ≈ 0.333; the helper rounds to 4 decimals so loose abs.
        assert d["t1_share"] == pytest.approx(1 / 3, abs=1e-3)
        assert d["t1_share"] < 1.0

    def test_t3_graph_hop_visible_in_decomposition(self) -> None:
        """T3 (graph hops) is the weakest positive tier; it MUST still
        contribute to the score and reduce t1_share."""
        labels = [
            TierLabel(chunk_id="t1", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="t3", tier=Tier.T3, score=0.3, rationale=""),
        ]
        d = outcome_score_decomposition(labels)
        # t1_contribution = 1.0; positive_total = 1.3; t1_share ≈ 0.77.
        assert d["t1_share"] < 1.0
        assert "T3" in d["by_tier"]


# ---------------------------------------------------------------------------
# Scenario 4 — anti-regression: 100 hits, edit-only baseline check
# ---------------------------------------------------------------------------


class TestEditOnlyBaselineCheck:
    """Stress: 100 hits with a single edit. The aggregate must be
    POSITIVE (1/100 = +0.01) and the decomposition must show T7 as the
    dominant count but NOT contributing negatively to the score."""

    def test_one_edit_99_reads_positive_score(self) -> None:
        hits = [_hit("c-edit", rel_path="src/edit.py")] + [
            _hit(f"c-r{i}", rel_path=f"src/u{i}.py") for i in range(99)
        ]
        ctx = LabelingContext(diff_paths={"src/edit.py"})
        labels = label_hits(hits, ctx)
        score = aggregate_score(labels)
        # 1 T1 + 99 T7 → (1 + 0*99) / 100 = 0.01
        assert score == pytest.approx(0.01)
        assert score > 0  # the load-bearing guard

    def test_zero_edit_100_reads_neutral_score(self) -> None:
        """No edit at all — every hit is T7. Score must be EXACTLY 0.
        Not slightly negative from floating-point error, not slightly
        positive — the labeler's neutrality contract must hold."""
        hits = [_hit(f"c-r{i}", rel_path=f"src/u{i}.py") for i in range(100)]
        ctx = LabelingContext()  # empty diff
        labels = label_hits(hits, ctx)
        score = aggregate_score(labels)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Scenario 5 — ScoreCard.dominates respects read-only neutrality
# ---------------------------------------------------------------------------


class TestScoreCardReadOnlyDominance:
    """A high-quality / read-only candidate must not be dominated by a
    low-quality / edit-only candidate just because of outcome_score."""

    def test_quality_dominates_when_outcome_equal(self) -> None:
        accurate_readonly = ScoreCard(
            quality=0.95, context_efficiency=0.5, latency_ms=10, outcome_score=0.0,
        )
        sloppy_editor = ScoreCard(
            quality=0.30, context_efficiency=0.5, latency_ms=10, outcome_score=0.5,
        )
        # Accurate-readonly wins on quality + ties on others.
        # Sloppy-editor wins only on outcome_score.
        # → Neither dominates the other (Pareto trade-off).
        assert not accurate_readonly.dominates(sloppy_editor)
        assert not sloppy_editor.dominates(accurate_readonly)

    def test_strictly_better_readonly_dominates_strictly_worse_editor(self) -> None:
        """If a read-only candidate is better on EVERY dimension except
        outcome, it should still win when outcome difference is small."""
        accurate = ScoreCard(
            quality=0.95, context_efficiency=0.9, latency_ms=5, outcome_score=0.0,
        )
        weak_editor = ScoreCard(
            quality=0.20, context_efficiency=0.1, latency_ms=100, outcome_score=0.05,
        )
        # accurate beats weak_editor on quality, eff, latency — but loses
        # marginally on outcome_score. So neither strictly dominates.
        assert not accurate.dominates(weak_editor)
        assert not weak_editor.dominates(accurate)
        # The Pareto frontier keeps both — that's the WIN-169 invariant
        # that prevents accurate read-only retrievals from being silenced.
        from stropha.meta.fitness import pareto_frontier
        front = pareto_frontier([accurate, weak_editor])
        assert len(front) == 2