"""Pareto frontier tracker + held-out report (WIN-192).

Mirrors Lee et al. Figure 3: the held-out score is computed for EVERY
candidate on the frontier, not just the running best. A frontier-wide
held-out report exposes overfit across the trade-off curve — a config
that wins on the search split but loses on holdout is the textbook
overfit case the WIN-168 split was designed to catch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from .fitness import ScoreCard, pareto_frontier
from .store import CandidateRecord

log = get_logger(__name__)


# Re-evaluator signature: takes a candidate's config and returns its
# held-out ScoreCard. The loop's standard evaluator scores the SEARCH
# split; this one scores HOLDOUT.
HoldoutEvaluatorFn = Callable[[dict[str, Any]], ScoreCard]


@dataclass
class FrontierEntry:
    """One frontier candidate with its search + held-out scores."""

    record: CandidateRecord
    search: ScoreCard
    holdout: ScoreCard | None = None
    overfit_gap: float = 0.0
    """Difference search.quality - holdout.quality. Positive = overfit;
    negative = generalises better than expected; zero = no holdout
    evaluation yet."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "label": self.record.label,
            "search": self.search.to_dict(),
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "overfit_gap": round(self.overfit_gap, 4),
        }


@dataclass
class FrontierReport:
    """A full frontier evaluation snapshot."""

    entries: list[FrontierEntry] = field(default_factory=list)
    archive_size: int = 0
    mean_overfit_gap: float = 0.0
    worst_overfit_id: str | None = None
    worst_overfit_gap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_size": self.archive_size,
            "frontier_size": len(self.entries),
            "mean_overfit_gap": round(self.mean_overfit_gap, 4),
            "worst_overfit_id": self.worst_overfit_id,
            "worst_overfit_gap": round(self.worst_overfit_gap, 4),
            "entries": [e.to_dict() for e in self.entries],
        }


def evaluate_frontier_on_holdout(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    holdout_evaluator: HoldoutEvaluatorFn,
) -> FrontierReport:
    """Score every Pareto-frontier candidate on the holdout split.

    Args:
        archive: full archive of (record, search_scorecard) tuples.
        holdout_evaluator: callable returning a ScoreCard for a given
            config when evaluated against the holdout split.

    Returns the :class:`FrontierReport` with one :class:`FrontierEntry`
    per frontier candidate. Holdout-evaluation failures are recorded
    with ``holdout=None`` and contribute ``overfit_gap=0.0``.
    """
    if not archive:
        return FrontierReport(entries=[], archive_size=0)

    # Compute the frontier on the search-time ScoreCards.
    cards = [card for _rec, card in archive]
    frontier_card_ids = {id(c) for c in pareto_frontier(cards)}
    frontier = [
        (rec, card) for rec, card in archive if id(card) in frontier_card_ids
    ]

    entries: list[FrontierEntry] = []
    for record, search_card in frontier:
        try:
            holdout_card = holdout_evaluator(record.config)
        except Exception as exc:
            log.warning(
                "frontier.holdout_failed",
                candidate=record.id, error=str(exc),
            )
            entries.append(FrontierEntry(record=record, search=search_card))
            continue
        gap = round(search_card.quality - holdout_card.quality, 6)
        entries.append(FrontierEntry(
            record=record,
            search=search_card,
            holdout=holdout_card,
            overfit_gap=gap,
        ))

    gaps = [e.overfit_gap for e in entries if e.holdout is not None]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    worst_entry = max(
        (e for e in entries if e.holdout is not None),
        key=lambda e: e.overfit_gap,
        default=None,
    )
    return FrontierReport(
        entries=entries,
        archive_size=len(archive),
        mean_overfit_gap=mean_gap,
        worst_overfit_id=worst_entry.record.id if worst_entry else None,
        worst_overfit_gap=worst_entry.overfit_gap if worst_entry else 0.0,
    )


@dataclass
class FrontierTracker:
    """Maintains the running Pareto frontier across loop iterations.

    The loop's :class:`LoopResult` already exposes the final frontier;
    this tracker exposes ITERATIVE frontiers — one snapshot per
    iteration — useful for the CLI's ``stropha meta status`` live view
    and for the post-hoc overfit chart (Lee et al. Figure 3).
    """

    snapshots: list[list[FrontierEntry]] = field(default_factory=list)
    """One entry per iteration: the frontier as it stood at that step.
    Holdout scores are NOT computed here; call
    :func:`evaluate_frontier_on_holdout` after the loop finishes."""

    def observe(self, archive: list[tuple[CandidateRecord, ScoreCard]]) -> None:
        """Record the current frontier as a new snapshot."""
        cards = [card for _rec, card in archive]
        frontier_ids = {id(c) for c in pareto_frontier(cards)}
        snapshot = [
            FrontierEntry(record=rec, search=card)
            for rec, card in archive
            if id(card) in frontier_ids
        ]
        self.snapshots.append(snapshot)

    def frontier_size_history(self) -> list[int]:
        """One int per iteration: how many candidates were on the frontier.

        A monotonically growing series is the healthy signal. A series
        that stops growing means the loop has converged (no new
        non-dominated entries) — time to flip an indexing dimension
        per the WIN-188 skill's advice.
        """
        return [len(s) for s in self.snapshots]


__all__ = [
    "FrontierEntry",
    "FrontierReport",
    "FrontierTracker",
    "HoldoutEvaluatorFn",
    "evaluate_frontier_on_holdout",
]
