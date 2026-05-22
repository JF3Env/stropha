"""Meta-Harness outer loop — Algorithm 1 (WIN-187).

Per Lee et al. §3.2 (Algorithm 1), the outer loop is::

    1. Initialise an archive with seed candidates.
    2. Repeat until budget exhausted:
        a. proposer.propose(archive) -> next candidate config.
        b. evaluator.evaluate(config) -> ScoreCard.
        c. archive.add(candidate, scorecard).
        d. update Pareto frontier.

This module is the skeleton. The proposer (WIN-188/189/190/191) and
the evaluator (WIN-171) are injected — the loop doesn't know how
candidates are generated or scored, only that the proposer is a
callable returning a config dict and the evaluator is a callable
returning a ScoreCard.

Two budget knobs control termination:
- ``wall_clock_s``: stop when the wall-clock budget is exhausted.
- ``max_iterations``: hard cap on iterations (default: unbounded when
  wall_clock_s is set, else 50).

Both can be set simultaneously — termination fires on whichever hits
first. A KeyboardInterrupt during ``proposer`` / ``evaluator`` is
treated as a graceful stop signal — the loop returns the archive built
so far so the user can inspect partial progress.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import StrophaError
from ..logging import get_logger
from .fitness import ScoreCard, pareto_frontier
from .store import ArtifactStore, CandidateRecord

log = get_logger(__name__)


class LoopError(StrophaError):
    """Raised when the outer loop cannot satisfy a request."""


# Proposer signature: takes the current archive of (record, scorecard)
# tuples and returns the next candidate config dict to evaluate.
ProposerFn = Callable[
    [list[tuple[CandidateRecord, ScoreCard]]],
    dict[str, Any],
]

# Evaluator signature: takes a config dict and returns a ScoreCard.
EvaluatorFn = Callable[[dict[str, Any]], ScoreCard]


@dataclass
class LoopResult:
    """The output of one ``run_loop`` invocation."""

    iterations: int = 0
    elapsed_s: float = 0.0
    archive: list[tuple[CandidateRecord, ScoreCard]] = field(default_factory=list)
    frontier: list[tuple[CandidateRecord, ScoreCard]] = field(default_factory=list)
    stopped_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "elapsed_s": round(self.elapsed_s, 3),
            "archive_size": len(self.archive),
            "frontier_size": len(self.frontier),
            "stopped_reason": self.stopped_reason,
            "errors": list(self.errors),
            "frontier": [
                {"id": rec.id, "scores": card.to_dict()}
                for rec, card in self.frontier
            ],
        }


def run_loop(
    *,
    proposer: ProposerFn,
    evaluator: EvaluatorFn,
    store: ArtifactStore,
    seeds: Iterable[dict[str, Any]] | None = None,
    wall_clock_s: float | None = None,
    max_iterations: int | None = None,
    on_iteration: Callable[[int, CandidateRecord, ScoreCard], None] | None = None,
) -> LoopResult:
    """Run Algorithm 1 against ``proposer`` + ``evaluator``.

    Args:
        proposer: callable returning the next candidate config.
        evaluator: callable returning a ScoreCard for one config.
        store: ArtifactStore where every candidate is persisted.
        seeds: optional iterable of seed configs evaluated first
            (Lee et al. §3.2 — seed init). When None and the archive
            is empty, the proposer is called immediately.
        wall_clock_s: wall-clock budget. None = no time limit.
        max_iterations: hard cap on iterations. None = 50 (when
            wall_clock_s is also None) or unbounded (when wall_clock_s
            is set).
        on_iteration: optional hook called after every successful
            evaluation. Used by the CLI ``stropha meta run`` to update
            a live progress display.

    Returns the :class:`LoopResult` with the final archive + Pareto
    frontier + the reason iteration stopped.
    """
    if wall_clock_s is None and max_iterations is None:
        max_iterations = 50
    start = time.monotonic()
    archive: list[tuple[CandidateRecord, ScoreCard]] = []
    errors: list[str] = []
    # ``attempts`` counts every proposer + evaluator round trip, success
    # or failure. Without this, a loop where every iteration fails would
    # never increment ``len(archive)`` and the budget check would never
    # fire — the loop would hang forever.
    attempts = 0

    # 1. Seed init — evaluate every provided seed config before the
    # proposer takes over. Seeds are run sequentially in declaration
    # order so the archive's earliest entries are reproducible.
    for seed_idx, seed_cfg in enumerate(list(seeds or [])):
        if _budget_exhausted(start, wall_clock_s, attempts, max_iterations):
            return _result(archive, start, "budget exhausted during seed init", errors)
        attempts += 1
        try:
            record, card = _evaluate_and_persist(
                config=seed_cfg, store=store,
                evaluator=evaluator, parent=None,
                label=f"seed-{seed_idx}",
            )
        except LoopError as exc:
            errors.append(f"seed {seed_idx}: {exc}")
            continue
        archive.append((record, card))
        if on_iteration is not None:
            on_iteration(len(archive), record, card)
        log.info(
            "loop.seed_done",
            seed=seed_idx,
            candidate=record.id,
            quality=round(card.quality, 4),
        )

    # 2. Main loop — proposer + evaluator until budget exhausted.
    while True:
        if _budget_exhausted(start, wall_clock_s, attempts, max_iterations):
            return _result(archive, start, "budget exhausted", errors)
        attempts += 1
        try:
            cfg = proposer(list(archive))
        except KeyboardInterrupt:
            return _result(archive, start, "interrupted", errors)
        except Exception as exc:
            errors.append(f"propose: {exc}")
            log.warning("loop.propose_failed", error=str(exc))
            continue
        try:
            record, card = _evaluate_and_persist(
                config=cfg, store=store,
                evaluator=evaluator,
                parent=archive[-1][0].id if archive else None,
                label=f"iter-{len(archive)}",
            )
        except KeyboardInterrupt:
            return _result(archive, start, "interrupted", errors)
        except LoopError as exc:
            errors.append(f"iter {len(archive)}: {exc}")
            log.warning("loop.evaluate_failed", error=str(exc))
            continue
        archive.append((record, card))
        if on_iteration is not None:
            on_iteration(len(archive), record, card)
        log.info(
            "loop.iter_done",
            iter=len(archive),
            candidate=record.id,
            quality=round(card.quality, 4),
            outcome=round(card.outcome_score, 4),
        )


def _budget_exhausted(
    start: float,
    wall_clock_s: float | None,
    iterations: int,
    max_iterations: int | None,
) -> bool:
    if wall_clock_s is not None and (time.monotonic() - start) >= wall_clock_s:
        return True
    return max_iterations is not None and iterations >= max_iterations


def _evaluate_and_persist(
    *,
    config: dict[str, Any],
    store: ArtifactStore,
    evaluator: EvaluatorFn,
    parent: str | None,
    label: str,
) -> tuple[CandidateRecord, ScoreCard]:
    """Register ``config`` in the store, evaluate it, write the scores back."""
    try:
        record = store.register_candidate(
            config=config, parent=parent, label=label,
        )
    except Exception as exc:
        raise LoopError(f"failed to register candidate: {exc}") from exc
    try:
        card = evaluator(config)
    except Exception as exc:
        raise LoopError(f"evaluator raised for candidate {record.id}: {exc}") from exc
    try:
        store.write_scores(record.id, card.to_dict())
    except Exception as exc:
        raise LoopError(
            f"failed to write scores for candidate {record.id}: {exc}"
        ) from exc
    record.scores = {
        k: v for k, v in card.to_dict().items() if isinstance(v, (int, float))
    }
    return record, card


def _result(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    start: float,
    reason: str,
    errors: list[str],
) -> LoopResult:
    cards = [card for _rec, card in archive]
    front_cards = set(id(c) for c in pareto_frontier(cards))
    frontier = [
        (rec, card) for rec, card in archive if id(card) in front_cards
    ]
    log.info(
        "loop.done",
        iterations=len(archive),
        elapsed_s=round(time.monotonic() - start, 3),
        frontier=len(frontier),
        reason=reason,
        errors=len(errors),
    )
    return LoopResult(
        iterations=len(archive),
        elapsed_s=time.monotonic() - start,
        archive=archive,
        frontier=frontier,
        stopped_reason=reason,
        errors=errors,
    )


__all__ = [
    "EvaluatorFn",
    "LoopError",
    "LoopResult",
    "ProposerFn",
    "run_loop",
]


# Re-export Path so the CLI can use the same Path symbol the loop
# returns in its LoopResult dataclass without an extra import.
__path_re_export__ = Path
