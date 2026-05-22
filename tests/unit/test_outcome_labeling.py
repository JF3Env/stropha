"""Unit tests for the tiered outcome labeler (WIN-162).

The acceptance criteria are explicit:
- 30+ unit tests across the seven tiers
- T7 is neutral (score 0), NOT negative
- E2E backfill against a realistic scenario produces ≥1 T1, ≥1 T2, ≥1 T7
- aggregate_score is what the Phase 1 fitness function will consume

Tests are organised by tier so a regression in any single tier shows up
in one class. The labeler is pure-Python (no storage required) — the
graph dependency is injected via a Protocol so tests use a dict-backed
stub. Concrete graph wiring (find_related → neighbors) is tested in the
retrieval tests, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

from stropha.meta.outcome import (
    TIER_SCORE,
    GraphAdapter,
    LabelingContext,
    Tier,
    TierLabel,
    _similar_queries,
    _ts_delta,
    aggregate_score,
    label_hit,
    label_hits,
    tier_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(
    chunk_id: str = "c1",
    rel_path: str | None = "src/foo.py",
    symbol: str | None = "foo",
    **extra: Any,
) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "rel_path": rel_path, "symbol": symbol, **extra}


class _StubGraph:
    """Dict-backed graph adapter for tier-3 tests.

    BFS expansion: at each depth, visit every unvisited neighbour and keep
    accumulating into ``out``. ``visited`` guards against cycles.
    """

    def __init__(self, adjacency: dict[str, set[str]] | None = None) -> None:
        self.adjacency = adjacency or {}

    def neighbors(self, symbols, depth):  # type: ignore[no-untyped-def]
        out: set[str] = set()
        visited: set[str] = set(symbols)
        frontier: set[str] = set(symbols)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for s in frontier:
                for n in self.adjacency.get(s, set()):
                    if n not in visited:
                        next_frontier.add(n)
                        visited.add(n)
            out |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return out


# ---------------------------------------------------------------------------
# TIER_SCORE constants
# ---------------------------------------------------------------------------


class TestTierScoreTable:
    def test_strong_positive_tiers(self) -> None:
        assert TIER_SCORE[Tier.T1] == pytest.approx(1.0)
        assert TIER_SCORE[Tier.T2] == pytest.approx(1.0)

    def test_medium_positive(self) -> None:
        assert TIER_SCORE[Tier.T4] == pytest.approx(0.6)

    def test_weak_positive(self) -> None:
        assert TIER_SCORE[Tier.T3] == pytest.approx(0.3)

    def test_weak_negative(self) -> None:
        assert TIER_SCORE[Tier.T5] == pytest.approx(-0.2)

    def test_strong_negative(self) -> None:
        assert TIER_SCORE[Tier.T6] == pytest.approx(-1.0)

    def test_neutral_is_zero(self) -> None:
        """The critical invariant — T7 must NEVER be negative."""
        assert TIER_SCORE[Tier.T7] == 0.0


# ---------------------------------------------------------------------------
# T1 — file in commit diff
# ---------------------------------------------------------------------------


class TestT1FileInDiff:
    def test_exact_path_match(self) -> None:
        ctx = LabelingContext(diff_paths={"src/foo.py"})
        label = label_hit(_hit(rel_path="src/foo.py"), ctx)
        assert label.tier == Tier.T1
        assert label.score == 1.0
        assert label.evidence["diff_path_match"] == "src/foo.py"

    def test_path_not_in_diff(self) -> None:
        ctx = LabelingContext(diff_paths={"src/other.py"})
        label = label_hit(_hit(rel_path="src/foo.py"), ctx)
        assert label.tier == Tier.T7

    def test_missing_rel_path_falls_through(self) -> None:
        ctx = LabelingContext(diff_paths={"src/foo.py"})
        label = label_hit(_hit(rel_path=None), ctx)
        assert label.tier == Tier.T7

    def test_label_carries_rationale(self) -> None:
        ctx = LabelingContext(diff_paths={"src/foo.py"})
        label = label_hit(_hit(rel_path="src/foo.py"), ctx)
        assert "commit diff" in label.rationale


# ---------------------------------------------------------------------------
# T2 — chunk's symbol appears in diff
# ---------------------------------------------------------------------------


class TestT2SymbolInDiff:
    def test_symbol_match_wins(self) -> None:
        ctx = LabelingContext(diff_symbols={"calculate_fsrs"})
        label = label_hit(_hit(symbol="calculate_fsrs", rel_path="x.py"), ctx)
        assert label.tier == Tier.T2
        assert label.score == 1.0
        assert label.evidence["diff_symbol_match"] == "calculate_fsrs"

    def test_symbol_no_match(self) -> None:
        ctx = LabelingContext(diff_symbols={"other"})
        label = label_hit(_hit(symbol="calculate_fsrs"), ctx)
        assert label.tier == Tier.T7

    def test_missing_symbol_falls_through(self) -> None:
        ctx = LabelingContext(diff_symbols={"foo"})
        label = label_hit(_hit(symbol=None), ctx)
        assert label.tier == Tier.T7

    def test_t1_takes_precedence_over_t2(self) -> None:
        """A hit that matches BOTH diff_path and diff_symbol is T1 (stronger)."""
        ctx = LabelingContext(
            diff_paths={"src/foo.py"},
            diff_symbols={"foo"},
        )
        label = label_hit(_hit(rel_path="src/foo.py", symbol="foo"), ctx)
        assert label.tier == Tier.T1


# ---------------------------------------------------------------------------
# T3 — graph hops
# ---------------------------------------------------------------------------


class TestT3GraphHops:
    def test_one_hop_match(self) -> None:
        graph = _StubGraph({"diff_sym": {"chunk_sym"}})
        ctx = LabelingContext(
            diff_symbols={"diff_sym"},
            graph=graph,
            graph_hops=1,
        )
        label = label_hit(_hit(symbol="chunk_sym"), ctx)
        assert label.tier == Tier.T3
        assert label.score == 0.3

    def test_two_hop_match(self) -> None:
        graph = _StubGraph(
            {
                "diff_sym": {"middle"},
                "middle": {"chunk_sym"},
            }
        )
        ctx = LabelingContext(
            diff_symbols={"diff_sym"},
            graph=graph,
            graph_hops=2,
        )
        label = label_hit(_hit(symbol="chunk_sym"), ctx)
        assert label.tier == Tier.T3

    def test_beyond_hop_limit_no_match(self) -> None:
        graph = _StubGraph(
            {
                "diff_sym": {"middle"},
                "middle": {"chunk_sym"},
            }
        )
        ctx = LabelingContext(
            diff_symbols={"diff_sym"},
            graph=graph,
            graph_hops=1,
        )
        label = label_hit(_hit(symbol="chunk_sym"), ctx)
        assert label.tier == Tier.T7

    def test_no_graph_adapter_skips_t3(self) -> None:
        ctx = LabelingContext(diff_symbols={"diff_sym"}, graph=None)
        label = label_hit(_hit(symbol="other"), ctx)
        assert label.tier == Tier.T7

    def test_no_diff_symbols_skips_t3(self) -> None:
        graph = _StubGraph({"x": {"y"}})
        ctx = LabelingContext(diff_symbols=set(), graph=graph)
        label = label_hit(_hit(symbol="y"), ctx)
        assert label.tier == Tier.T7

    def test_graph_raises_falls_back_to_lower_tier(self) -> None:
        class Boom:
            def neighbors(self, symbols, depth):  # type: ignore[no-untyped-def]
                raise RuntimeError("simulated graph error")

        ctx = LabelingContext(diff_symbols={"x"}, graph=Boom())
        label = label_hit(_hit(symbol="y"), ctx)
        assert label.tier == Tier.T7

    def test_t2_takes_precedence_over_t3(self) -> None:
        graph = _StubGraph({"x": {"y"}})
        ctx = LabelingContext(diff_symbols={"y"}, graph=graph)
        # 'y' is both in diff AND reachable from 'x' — T2 wins.
        label = label_hit(_hit(symbol="y"), ctx)
        assert label.tier == Tier.T2


# ---------------------------------------------------------------------------
# T4 — follow-up MCP call in same session
# ---------------------------------------------------------------------------


class TestT4FollowUp:
    def test_chunk_reappears_in_later_trace_results(self) -> None:
        traces = [
            {"tool": "search_code", "query": "fsrs", "results": [{"chunk_id": "c1"}]},
            {"tool": "search_code", "query": "more", "results": [{"chunk_id": "c1"}]},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T4
        assert label.score == 0.6

    def test_symbol_appears_in_later_query(self) -> None:
        traces = [
            {"tool": "search_code", "query": "fsrs", "results": [{"chunk_id": "c1"}]},
            {"tool": "get_symbol", "query": "calculate_fsrs"},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1", symbol="calculate_fsrs"), ctx)
        assert label.tier == Tier.T4

    def test_symbol_appears_in_later_params(self) -> None:
        traces = [
            {"tool": "search_code", "results": [{"chunk_id": "c1"}]},
            {"tool": "find_callers", "params": {"symbol": "calculate_fsrs"}},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1", symbol="calculate_fsrs"), ctx)
        assert label.tier == Tier.T4

    def test_no_follow_up_falls_through(self) -> None:
        traces = [
            {"tool": "search_code", "results": [{"chunk_id": "c1"}]},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1", symbol="something_else"), ctx)
        assert label.tier == Tier.T7

    def test_traces_with_no_results_key(self) -> None:
        """``results`` key absent / None must not crash the lookup."""
        traces = [{"tool": "search_code"}, {"tool": "list_repos"}]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T7

    def test_unrelated_chunks_in_followup_dont_count(self) -> None:
        traces = [
            {"tool": "search_code", "results": [{"chunk_id": "c1"}]},
            {"tool": "search_code", "results": [{"chunk_id": "c999"}]},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1", symbol="something_else"), ctx)
        assert label.tier == Tier.T7


# ---------------------------------------------------------------------------
# T5 — same-session re-query (weak negative)
# ---------------------------------------------------------------------------


class TestT5Requery:
    def test_identical_query_within_window(self) -> None:
        traces = [
            {
                "tool": "search_code",
                "query": "how does fsrs work",
                "ts": "2026-05-22T16:00:00+00:00",
                "results": [{"chunk_id": "c1"}],
            },
            {
                "tool": "search_code",
                "query": "how does fsrs work",
                "ts": "2026-05-22T16:00:10+00:00",
            },
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T5
        assert label.score == -0.2

    def test_paraphrased_query_within_window(self) -> None:
        traces = [
            {
                "tool": "search_code",
                "query": "how does fsrs work",
                "ts": "2026-05-22T16:00:00+00:00",
                "results": [{"chunk_id": "c1"}],
            },
            {
                "tool": "search_code",
                "query": "how fsrs works exactly",
                "ts": "2026-05-22T16:00:30+00:00",
            },
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T5

    def test_requery_outside_window_is_t7(self) -> None:
        traces = [
            {
                "tool": "search_code",
                "query": "how does fsrs work",
                "ts": "2026-05-22T16:00:00+00:00",
                "results": [{"chunk_id": "c1"}],
            },
            {
                "tool": "search_code",
                "query": "how does fsrs work",
                "ts": "2026-05-22T16:30:00+00:00",  # 30 minutes later
            },
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T7

    def test_dissimilar_next_query_is_t7(self) -> None:
        traces = [
            {"tool": "search_code", "query": "fsrs algorithm", "results": [{"chunk_id": "c1"}],
             "ts": "2026-05-22T16:00:00+00:00"},
            {"tool": "search_code", "query": "completely different topic",
             "ts": "2026-05-22T16:00:05+00:00"},
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T7

    def test_no_followup_trace_after_producer(self) -> None:
        traces = [
            {"tool": "search_code", "query": "fsrs", "results": [{"chunk_id": "c1"}]},
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1", symbol="other"), ctx)
        assert label.tier == Tier.T7

    def test_skips_non_matching_results_in_trace(self) -> None:
        """A trace with multiple results — only the matching chunk triggers
        the re-query lookahead, others are skipped (branch coverage)."""
        traces = [
            {
                "tool": "search_code",
                "query": "fsrs",
                "ts": "2026-05-22T16:00:00+00:00",
                "results": [
                    {"chunk_id": "other-1"},   # not our chunk → skipped
                    {"chunk_id": "other-2"},   # not our chunk → skipped
                    {"chunk_id": "c1"},        # OUR chunk → triggers lookahead
                ],
            },
            {
                "tool": "search_code",
                "query": "fsrs",
                "ts": "2026-05-22T16:00:10+00:00",
            },
        ]
        ctx = LabelingContext(session_traces=traces, requery_window_s=60.0)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T5

    def test_missing_timestamps_uses_query_similarity_only(self) -> None:
        """When traces lack timestamps but the queries are similar, treat
        as a re-query (conservative)."""
        traces = [
            {"tool": "search_code", "query": "fsrs", "results": [{"chunk_id": "c1"}]},
            {"tool": "search_code", "query": "fsrs"},
        ]
        ctx = LabelingContext(session_traces=traces)
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T5


# ---------------------------------------------------------------------------
# T6 — report_bad_result
# ---------------------------------------------------------------------------


class TestT6BadResult:
    def test_chunk_in_bad_set(self) -> None:
        ctx = LabelingContext(bad_results={"c-bad"})
        label = label_hit(_hit(chunk_id="c-bad"), ctx)
        assert label.tier == Tier.T6
        assert label.score == -1.0

    def test_t6_beats_t1(self) -> None:
        """Explicit negative signal must override even a strong positive."""
        ctx = LabelingContext(
            diff_paths={"src/foo.py"},
            bad_results={"c1"},
        )
        label = label_hit(_hit(chunk_id="c1", rel_path="src/foo.py"), ctx)
        assert label.tier == Tier.T6

    def test_unrelated_bad_results_dont_match(self) -> None:
        ctx = LabelingContext(bad_results={"c-other"})
        label = label_hit(_hit(chunk_id="c1"), ctx)
        assert label.tier == Tier.T7


# ---------------------------------------------------------------------------
# T7 — neutral (the critical invariant)
# ---------------------------------------------------------------------------


class TestT7Neutral:
    def test_no_signals_returns_t7(self) -> None:
        label = label_hit(_hit(), LabelingContext())
        assert label.tier == Tier.T7
        assert label.score == 0.0

    def test_t7_score_is_exactly_zero(self) -> None:
        """T7 must be ZERO, not 0.001 or -epsilon — the tuner depends on it."""
        label = label_hit(_hit(), LabelingContext())
        assert label.score == 0.0

    def test_t7_carries_explaining_rationale(self) -> None:
        label = label_hit(_hit(), LabelingContext())
        assert "neutral" in label.rationale.lower()
        assert "read-only" in label.rationale.lower()


# ---------------------------------------------------------------------------
# label_hits + aggregate_score
# ---------------------------------------------------------------------------


class TestVectorisedAndAggregation:
    def test_label_hits_returns_one_per_input(self) -> None:
        ctx = LabelingContext(diff_paths={"a.py"})
        labels = label_hits(
            [
                _hit(chunk_id="c1", rel_path="a.py"),
                _hit(chunk_id="c2", rel_path="b.py"),
            ],
            ctx,
        )
        assert len(labels) == 2
        assert labels[0].tier == Tier.T1
        assert labels[1].tier == Tier.T7

    def test_aggregate_score_mean(self) -> None:
        labels = [
            TierLabel(chunk_id="a", tier=Tier.T1, score=1.0, rationale=""),
            TierLabel(chunk_id="b", tier=Tier.T7, score=0.0, rationale=""),
        ]
        assert aggregate_score(labels) == pytest.approx(0.5)

    def test_aggregate_score_empty(self) -> None:
        assert aggregate_score([]) == 0.0

    def test_aggregate_score_with_negatives(self) -> None:
        labels = [
            TierLabel(chunk_id="a", tier=Tier.T6, score=-1.0, rationale=""),
            TierLabel(chunk_id="b", tier=Tier.T1, score=1.0, rationale=""),
        ]
        assert aggregate_score(labels) == 0.0

    def test_tier_summary_counts_each_tier(self) -> None:
        labels = [
            TierLabel(chunk_id=str(i), tier=t, score=TIER_SCORE[t], rationale="")
            for i, t in enumerate(
                [Tier.T1, Tier.T1, Tier.T7, Tier.T7, Tier.T7, Tier.T3]
            )
        ]
        summary = tier_summary(labels)
        assert summary[Tier.T1] == 2
        assert summary[Tier.T7] == 3
        assert summary[Tier.T3] == 1
        assert summary[Tier.T2] == 0  # zeroes are still present

    def test_tier_summary_empty(self) -> None:
        summary = tier_summary([])
        assert all(c == 0 for c in summary.values())
        assert set(summary.keys()) == set(Tier)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_similar_queries_identical(self) -> None:
        assert _similar_queries("how does fsrs work", "how does fsrs work") is True

    def test_similar_queries_jaccard_above_threshold(self) -> None:
        assert _similar_queries("how does fsrs work", "how fsrs works") is True

    def test_similar_queries_below_threshold(self) -> None:
        assert _similar_queries("apple", "banana cherry date elderberry") is False

    def test_similar_queries_empty(self) -> None:
        assert _similar_queries("", "anything") is False
        assert _similar_queries("anything", "") is False

    def test_ts_delta_valid(self) -> None:
        d = _ts_delta(
            "2026-05-22T16:00:00+00:00", "2026-05-22T16:00:30+00:00"
        )
        assert d == pytest.approx(30.0)

    def test_ts_delta_missing(self) -> None:
        assert _ts_delta(None, "x") is None
        assert _ts_delta("x", None) is None

    def test_ts_delta_malformed(self) -> None:
        assert _ts_delta("not-iso", "2026-05-22T16:00:00+00:00") is None


# ---------------------------------------------------------------------------
# E2E backfill scenario (per ticket acceptance)
# ---------------------------------------------------------------------------


class TestEndToEndBackfill:
    def test_realistic_commit_produces_t1_t2_and_t7(self) -> None:
        """Simulate a real commit + session: at least one T1, one T2, one T7."""
        ctx = LabelingContext(
            diff_paths={"src/services/study.py"},
            diff_symbols={"StudyService.submitAnswer"},
            session_traces=[],  # no engagement signals in this scenario
            bad_results=set(),
        )

        hits = [
            # T1 — edited file.
            _hit(chunk_id="c-edit", rel_path="src/services/study.py", symbol="StudyService.review"),
            # T2 — defining symbol referenced in diff.
            _hit(chunk_id="c-def",  rel_path="src/services/util.py",  symbol="StudyService.submitAnswer"),
            # T7 — read-only chunk that turned out useful but not edited.
            _hit(chunk_id="c-read", rel_path="src/util/calendar.py", symbol="today"),
        ]
        labels = label_hits(hits, ctx)
        tiers = [lbl.tier for lbl in labels]
        assert Tier.T1 in tiers
        assert Tier.T2 in tiers
        assert Tier.T7 in tiers
        # Aggregate is positive overall: T1 + T2 + T7 = (1 + 1 + 0) / 3 ≈ 0.67.
        assert aggregate_score(labels) > 0.5

    def test_t7_does_not_drag_score_negative(self) -> None:
        """The most important behavioural test: a session of read-only
        retrievals that produced a correct edit must score >= 0."""
        ctx = LabelingContext(diff_paths={"src/foo.py"})

        hits = [
            _hit(chunk_id="edit", rel_path="src/foo.py"),
            _hit(chunk_id="read-1", rel_path="src/other1.py", symbol="helper1"),
            _hit(chunk_id="read-2", rel_path="src/other2.py", symbol="helper2"),
            _hit(chunk_id="read-3", rel_path="src/other3.py", symbol="helper3"),
        ]
        labels = label_hits(hits, ctx)
        score = aggregate_score(labels)
        # One T1 (+1) + three T7 (0) -> 0.25, positive.
        assert score > 0


# ---------------------------------------------------------------------------
# GraphAdapter Protocol — structural check
# ---------------------------------------------------------------------------


class TestGraphAdapterProtocol:
    def test_protocol_signature(self) -> None:
        """A class with the right shape satisfies the Protocol."""
        class _OK:
            def neighbors(self, symbols, depth):  # type: ignore[no-untyped-def]
                return set()

        # Should be assignable to GraphAdapter without runtime error.
        adapter: GraphAdapter = _OK()
        assert adapter.neighbors([], 1) == set()