"""Unit tests for the Pareto frontier tracker + held-out report (WIN-192)."""

from __future__ import annotations

from typing import Any

import pytest

from stropha.meta import (
    FrontierEntry,
    FrontierReport,
    FrontierTracker,
    ScoreCard,
    evaluate_frontier_on_holdout,
)
from stropha.meta.store import CandidateRecord


def _record(idx: int, config: dict[str, Any] | None = None) -> CandidateRecord:
    return CandidateRecord(
        id=f"c{idx:02d}",
        parent_id=None,
        created_at="2026-05-22T00:00:00+00:00",
        label=f"iter-{idx}",
        config=config or {"v": idx},
    )


# ---------------------------------------------------------------------------
# FrontierEntry / FrontierReport
# ---------------------------------------------------------------------------


class TestFrontierEntry:
    def test_to_dict_with_holdout(self) -> None:
        entry = FrontierEntry(
            record=_record(1),
            search=ScoreCard(quality=0.9),
            holdout=ScoreCard(quality=0.7),
            overfit_gap=0.2,
        )
        d = entry.to_dict()
        assert d["id"] == "c01"
        assert d["search"]["quality"] == 0.9
        assert d["holdout"]["quality"] == 0.7
        assert d["overfit_gap"] == 0.2

    def test_to_dict_without_holdout(self) -> None:
        entry = FrontierEntry(record=_record(1), search=ScoreCard(quality=0.9))
        d = entry.to_dict()
        assert d["holdout"] is None
        assert d["overfit_gap"] == 0.0


class TestFrontierReportToDict:
    def test_round_trip(self) -> None:
        entry = FrontierEntry(record=_record(1), search=ScoreCard(quality=0.5))
        report = FrontierReport(
            entries=[entry],
            archive_size=10,
            mean_overfit_gap=0.05,
            worst_overfit_id="c01",
            worst_overfit_gap=0.1,
        )
        d = report.to_dict()
        for key in (
            "archive_size", "frontier_size", "mean_overfit_gap",
            "worst_overfit_id", "worst_overfit_gap", "entries",
        ):
            assert key in d
        assert d["frontier_size"] == 1


# ---------------------------------------------------------------------------
# evaluate_frontier_on_holdout
# ---------------------------------------------------------------------------


class TestEvaluateFrontierOnHoldout:
    def test_empty_archive_returns_empty_report(self) -> None:
        report = evaluate_frontier_on_holdout(
            archive=[],
            holdout_evaluator=lambda _cfg: ScoreCard(quality=0.5),
        )
        assert report.entries == []
        assert report.archive_size == 0

    def test_every_frontier_candidate_scored(self) -> None:
        archive = [
            (_record(0), ScoreCard(quality=0.9, latency_ms=100)),
            (_record(1), ScoreCard(quality=0.6, latency_ms=10)),
            (_record(2), ScoreCard(quality=0.3, latency_ms=200)),
        ]
        # Holdout evaluator drops every candidate's quality by 0.1.
        def holdout_eval(cfg: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.5, latency_ms=50)
        report = evaluate_frontier_on_holdout(archive, holdout_eval)
        # Two frontier candidates (c00 + c01); c02 is dominated.
        assert len(report.entries) == 2
        assert {e.record.id for e in report.entries} == {"c00", "c01"}
        # Each entry has a holdout ScoreCard.
        assert all(e.holdout is not None for e in report.entries)

    def test_overfit_gap_computed(self) -> None:
        archive = [
            (_record(0), ScoreCard(quality=0.9)),
        ]
        def holdout_eval(_cfg: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.6)
        report = evaluate_frontier_on_holdout(archive, holdout_eval)
        entry = report.entries[0]
        # search 0.9 - holdout 0.6 = +0.3 overfit gap.
        assert entry.overfit_gap == pytest.approx(0.3)
        assert report.worst_overfit_id == "c00"
        assert report.worst_overfit_gap == pytest.approx(0.3)
        assert report.mean_overfit_gap == pytest.approx(0.3)

    def test_negative_gap_means_generalises_better(self) -> None:
        archive = [
            (_record(0), ScoreCard(quality=0.5)),
        ]
        def holdout_eval(_cfg: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.8)  # better on holdout!
        report = evaluate_frontier_on_holdout(archive, holdout_eval)
        assert report.entries[0].overfit_gap == pytest.approx(-0.3)

    def test_holdout_evaluator_failure_records_entry(self) -> None:
        # Both candidates must be on the Pareto frontier — give them
        # a true trade-off (one wins on quality, the other on latency).
        archive = [
            (_record(0), ScoreCard(quality=0.9, latency_ms=100)),
            (_record(1), ScoreCard(quality=0.5, latency_ms=10)),
        ]

        def flaky_eval(cfg: dict[str, Any]) -> ScoreCard:
            if cfg.get("v") == 0:
                raise RuntimeError("simulated holdout failure")
            return ScoreCard(quality=0.5)

        report = evaluate_frontier_on_holdout(archive, flaky_eval)
        # Both frontier entries are recorded; the failing one has holdout=None.
        by_id = {e.record.id: e for e in report.entries}
        assert by_id["c00"].holdout is None
        assert by_id["c00"].overfit_gap == 0.0
        assert by_id["c01"].holdout is not None

    def test_aggregate_metrics_ignore_failed_holdouts(self) -> None:
        archive = [
            (_record(0), ScoreCard(quality=0.9, latency_ms=100)),
            (_record(1), ScoreCard(quality=0.5, latency_ms=5)),
        ]

        def flaky_eval(cfg: dict[str, Any]) -> ScoreCard:
            if cfg.get("v") == 0:
                raise RuntimeError("nope")
            return ScoreCard(quality=0.3)

        report = evaluate_frontier_on_holdout(archive, flaky_eval)
        # Only c01 has a holdout score → 0.5 - 0.3 = 0.2.
        assert report.mean_overfit_gap == pytest.approx(0.2)
        assert report.worst_overfit_id == "c01"

    def test_all_holdouts_failed_zero_aggregates(self) -> None:
        archive = [(_record(0), ScoreCard(quality=0.9))]

        def always_fails(_cfg: dict[str, Any]) -> ScoreCard:
            raise RuntimeError("simulated")

        report = evaluate_frontier_on_holdout(archive, always_fails)
        assert report.mean_overfit_gap == 0.0
        assert report.worst_overfit_id is None
        assert report.worst_overfit_gap == 0.0


# ---------------------------------------------------------------------------
# FrontierTracker
# ---------------------------------------------------------------------------


class TestFrontierTracker:
    def test_observe_appends_snapshot(self) -> None:
        tracker = FrontierTracker()
        archive = [(_record(0), ScoreCard(quality=0.5))]
        tracker.observe(archive)
        assert len(tracker.snapshots) == 1
        assert len(tracker.snapshots[0]) == 1
        assert tracker.snapshots[0][0].record.id == "c00"

    def test_frontier_growth_over_time(self) -> None:
        tracker = FrontierTracker()
        # Iteration 1: one candidate.
        tracker.observe([(_record(0), ScoreCard(quality=0.5, latency_ms=10))])
        # Iteration 2: add a tradeoff candidate → both on frontier.
        tracker.observe([
            (_record(0), ScoreCard(quality=0.5, latency_ms=10)),
            (_record(1), ScoreCard(quality=0.9, latency_ms=100)),
        ])
        # Iteration 3: add a dominated candidate → frontier unchanged.
        tracker.observe([
            (_record(0), ScoreCard(quality=0.5, latency_ms=10)),
            (_record(1), ScoreCard(quality=0.9, latency_ms=100)),
            (_record(2), ScoreCard(quality=0.3, latency_ms=200)),
        ])
        assert tracker.frontier_size_history() == [1, 2, 2]

    def test_empty_archive_observe(self) -> None:
        tracker = FrontierTracker()
        tracker.observe([])
        assert tracker.frontier_size_history() == [0]


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        for name in (
            "FrontierEntry", "FrontierReport", "FrontierTracker",
            "HoldoutEvaluatorFn", "evaluate_frontier_on_holdout",
        ):
            assert hasattr(meta, name)