"""Meta-Harness outer loop — Algorithm 1 (WIN-187).

Per Lee et al. §3.2 (Algorithm 1), the outer loop is::

    1. Initialise an archive with seed candidates.
    2. Repeat until budget exhausted:
        a. proposer.propose(archive) -> next candidate config.
        b. evaluator.evaluate(config) -> ScoreCard.
        c. archive.add(candidate, scorecard).
        d. update Pareto frontier.

The proposer (WIN-188/189/190/191) and evaluator (WIN-171) are injected.
Proposers may return a legacy config dict or a typed :class:`Proposal`.
Each iteration can register a deterministic batch before evaluating its
candidates sequentially or with a bounded thread pool.

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

import json
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import StrophaError
from ..logging import get_logger
from .fitness import ScoreCard, pareto_frontier
from .proposer._agentic_base import Hypothesis, Proposal
from .store import ArtifactStore, CandidateRecord

log = get_logger(__name__)


class LoopError(StrophaError):
    """Raised when the outer loop cannot satisfy a request."""


class CandidateCheckpoint(BaseModel):
    """Durable state of one seed or proposal attempt."""

    candidate_id: str | None = None
    phase: Literal["seed", "proposal"]
    iteration: int
    slot: int
    status: Literal["deduplicated", "failed", "registered", "scored", "timed_out"]
    error: str | None = None
    duplicate_of: str | None = None


class LoopCheckpoint(BaseModel):
    """Stable loop boundary that callers can persist and later resume from."""

    version: int = 1
    attempts: int = 0
    next_seed_index: int = 0
    next_iteration: int = 0
    next_slot: int = 0
    archive_ids: list[str] = Field(default_factory=list)
    frontier_ids: list[str] = Field(default_factory=list)
    candidates: list[CandidateCheckpoint] = Field(default_factory=list)
    elapsed_s: float = 0.0
    proposer_state: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LoopCheckpoint:
        return cls.model_validate(value)


# Proposer signature: takes the current archive of (record, scorecard)
# tuples and returns either the legacy config dict or a typed proposal.
ProposerFn = Callable[
    [list[tuple[CandidateRecord, ScoreCard]]],
    dict[str, Any] | Proposal,
]

# Evaluator signature: takes a config dict and returns a ScoreCard.
EvaluatorFn = Callable[[dict[str, Any]], ScoreCard]

# Registered evaluator signature: also receives the durable candidate record.
RegisteredEvaluatorFn = Callable[[dict[str, Any], CandidateRecord], ScoreCard]

# Gate signature (WIN-197). Takes the candidate's on-disk directory and
# returns an object that exposes ``passed: bool`` and optionally a
# ``to_dict()`` method for persistence. The loop never imports the gate
# implementation directly — duck-typed so a future alternative gate
# (e.g. ``mypy`` gate, ``ruff`` gate) plugs in without a code change here.
GateFn = Callable[[Path], Any]
CheckpointCallback = Callable[[LoopCheckpoint], None]


@dataclass
class _PreparedCandidate:
    record: CandidateRecord
    config: dict[str, Any]
    iteration: int
    slot: int
    state_index: int


@dataclass
class LoopResult:
    """The output of one ``run_loop`` invocation."""

    iterations: int = 0
    elapsed_s: float = 0.0
    archive: list[tuple[CandidateRecord, ScoreCard]] = field(default_factory=list)
    frontier: list[tuple[CandidateRecord, ScoreCard]] = field(default_factory=list)
    stopped_reason: str = ""
    errors: list[str] = field(default_factory=list)
    checkpoint: LoopCheckpoint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "elapsed_s": round(self.elapsed_s, 3),
            "archive_size": len(self.archive),
            "frontier_size": len(self.frontier),
            "stopped_reason": self.stopped_reason,
            "errors": list(self.errors),
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint is not None else None,
            "frontier": [
                {"id": rec.id, "scores": card.to_dict()}
                for rec, card in self.frontier
            ],
        }


def run_loop(
    *,
    proposer: ProposerFn,
    evaluator: EvaluatorFn | None = None,
    store: ArtifactStore,
    seeds: Iterable[dict[str, Any] | Proposal] | None = None,
    wall_clock_s: float | None = None,
    max_iterations: int | None = None,
    max_rounds: int | None = None,
    on_iteration: Callable[[int, CandidateRecord, ScoreCard], None] | None = None,
    gate: GateFn | None = None,
    candidates_per_iteration: int = 1,
    candidates_per_iter: int | None = None,
    max_parallel: int = 1,
    evaluation_timeout_s: float | None = None,
    on_checkpoint: CheckpointCallback | None = None,
    checkpoint: LoopCheckpoint | Mapping[str, Any] | None = None,
    registered_evaluator: RegisteredEvaluatorFn | None = None,
    run_id: str | None = None,
) -> LoopResult:
    """Run Algorithm 1 against ``proposer`` + ``evaluator``.

    Args:
        proposer: callable returning the next candidate config.
        evaluator: legacy callable returning a ScoreCard for one config. Optional
            when ``registered_evaluator`` is provided.
        store: ArtifactStore where every candidate is persisted.
        seeds: optional iterable of seed configs evaluated first
            (Lee et al. §3.2 — seed init). When None and the archive
            is empty, the proposer is called immediately.
        wall_clock_s: wall-clock budget. None = no time limit.
        max_iterations: hard cap on iterations. None = 50 (when
            wall_clock_s is also None) or unbounded (when wall_clock_s
            is set).
        max_rounds: hard cap on proposal batches, excluding seed evaluation.
        on_iteration: optional hook called after every successful
            evaluation. Used by the CLI ``stropha meta run`` to update
            a live progress display.
        gate: optional pre-scoring gate (WIN-197). When provided, the
            loop calls ``gate(candidate_dir)`` after
            ``store.register_candidate`` and before the evaluator. If
            the returned object's ``.passed`` attribute is False, the
            candidate is rejected (no ScoreCard, no archive entry) and
            the loop continues with the next iteration. The gate
            result is persisted under ``<candidate>/gate.json``. The
            ``stropha.meta.gate.PytestGate`` is the canonical
            implementation but any callable matching :data:`GateFn`
            works.
        candidates_per_iteration: number of proposals generated against
            the same pre-batch archive. Defaults to one for legacy behavior.
        candidates_per_iter: concise alias for ``candidates_per_iteration``.
        max_parallel: maximum concurrent evaluator calls. Proposal generation,
            registration, archive insertion, and callbacks remain slot ordered.
        evaluation_timeout_s: optional soft batch wait timeout. Timed-out
            evaluator threads are ignored, but Python cannot forcibly stop them;
            they may finish and atomically replace ``scores.json`` later.
        on_checkpoint: callback receiving a complete checkpoint after each seed,
            before and after proposal-batch evaluation, and before every terminal
            return. The callback may persist it atomically.
        checkpoint: prior checkpoint data. Scored candidates are loaded from the
            store and are not evaluated again; the loop continues at its next
            stable seed/batch boundary.
        registered_evaluator: optional evaluator receiving ``(config, record)``.
            When provided, it is used instead of ``evaluator`` so evaluation can
            use the registered candidate id, directory, and metadata.
        run_id: optional value added to every seed and proposal record's metadata.

    Returns the :class:`LoopResult` with the final archive + Pareto
    frontier + the reason iteration stopped.
    """
    if candidates_per_iter is not None:
        if candidates_per_iteration != 1 and candidates_per_iteration != candidates_per_iter:
            raise LoopError("candidate count aliases must agree")
        candidates_per_iteration = candidates_per_iter
    if candidates_per_iteration < 1:
        raise LoopError("candidates_per_iteration must be at least 1")
    if max_parallel < 1:
        raise LoopError("max_parallel must be at least 1")
    if evaluation_timeout_s is not None and evaluation_timeout_s <= 0:
        raise LoopError("evaluation_timeout_s must be greater than 0")
    if max_rounds is not None and max_rounds < 1:
        raise LoopError("max_rounds must be at least 1")
    if evaluator is None and registered_evaluator is None:
        raise LoopError("evaluator or registered_evaluator is required")
    if wall_clock_s is None and max_iterations is None and max_rounds is None:
        max_iterations = 50
    state = _coerce_checkpoint(checkpoint)
    start = time.monotonic() - state.elapsed_s
    archive = _restore_archive(state, store)
    errors: list[str] = []
    # ``attempts`` counts every proposer + evaluator round trip, success
    # or failure. Without this, a loop where every iteration fails would
    # never increment ``len(archive)`` and the budget check would never
    # fire — the loop would hang forever.
    attempts = state.attempts
    seen_configs = {_config_key(record.config): record.id for record, _card in archive}
    checkpoint_states = list(state.candidates)
    seed_values = list(seeds or [])
    seed_count = max(len(seed_values), state.next_seed_index)

    # 1. Seed init — evaluate every provided seed config before the
    # proposer takes over. Seeds are run sequentially in declaration
    # order so the archive's earliest entries are reproducible.
    for seed_idx, seed_cfg in enumerate(seed_values[state.next_seed_index :], state.next_seed_index):
        if _budget_exhausted(start, wall_clock_s, attempts, max_iterations):
            return _result(
                archive, start, "budget exhausted during seed init", errors,
                _checkpoint(
                    archive, attempts, seed_idx, state.next_iteration, 0,
                    checkpoint_states, start,
                ),
                on_checkpoint,
            )
        attempts += 1
        proposal = _coerce_proposal(seed_cfg)
        config = _proposal_config(proposal)
        config_key = _config_key(config)
        duplicate_of = seen_configs.get(config_key)
        if duplicate_of is not None:
            checkpoint_states.append(CandidateCheckpoint(
                phase="seed", iteration=seed_idx, slot=0,
                status="deduplicated", duplicate_of=duplicate_of,
            ))
            _emit_checkpoint(on_checkpoint, _checkpoint(
                archive, attempts, seed_idx + 1, state.next_iteration, 0,
                checkpoint_states, start,
            ))
            continue
        try:
            record, card = _evaluate_and_persist(
                config=config, store=store,
                evaluator=evaluator, parent=None,
                label=f"seed-{seed_idx}",
                gate=gate,
                meta=_proposal_metadata(
                    proposal, phase="seed", iteration=seed_idx, slot=0,
                    run_id=run_id,
                ),
                registered_evaluator=registered_evaluator,
            )
        except LoopError as exc:
            errors.append(f"seed {seed_idx}: {exc}")
            checkpoint_states.append(CandidateCheckpoint(
                phase="seed", iteration=seed_idx, slot=0,
                status="failed", error=str(exc),
            ))
            _emit_checkpoint(on_checkpoint, _checkpoint(
                archive, attempts, seed_idx + 1, state.next_iteration, 0,
                checkpoint_states, start,
            ))
            continue
        archive.append((record, card))
        seen_configs[config_key] = record.id
        checkpoint_states.append(CandidateCheckpoint(
            candidate_id=record.id, phase="seed", iteration=seed_idx,
            slot=0, status="scored",
        ))
        if on_iteration is not None:
            on_iteration(len(archive), record, card)
        log.info(
            "loop.seed_done",
            seed=seed_idx,
            candidate=record.id,
            quality=round(card.quality, 4),
        )
        _emit_checkpoint(on_checkpoint, _checkpoint(
            archive, attempts, seed_idx + 1, state.next_iteration, 0,
            checkpoint_states, start,
        ))

    # 2. Main loop — proposer + evaluator until budget exhausted.
    iteration = state.next_iteration
    while True:
        archive_snapshot = list(archive)
        prepared, already_scored = _restore_registered_candidates(
            checkpoint_states, iteration=iteration, store=store,
        )
        if not prepared and not already_scored and max_rounds is not None and iteration >= max_rounds:
            final_checkpoint = _checkpoint(
                archive, attempts, seed_count, iteration, 0,
                checkpoint_states, start,
            )
            return _result(
                archive, start, "round budget exhausted", errors, final_checkpoint,
                on_checkpoint,
            )
        if not prepared and not already_scored and _budget_exhausted(
            start, wall_clock_s, attempts, max_iterations
        ):
            final_checkpoint = _checkpoint(
                archive, attempts, seed_count, iteration, 0,
                checkpoint_states, start,
            )
            return _result(
                archive, start, "budget exhausted", errors, final_checkpoint,
                on_checkpoint,
            )

        interrupted = False
        next_slot = state.next_slot if iteration == state.next_iteration else 0
        for slot in range(next_slot, candidates_per_iteration):
            if _budget_exhausted(start, wall_clock_s, attempts, max_iterations):
                break
            attempts += 1
            try:
                proposal = _coerce_proposal(proposer(list(archive_snapshot)))
            except KeyboardInterrupt:
                interrupted = True
                next_slot = slot
                break
            except Exception as exc:
                errors.append(f"propose iter {iteration} slot {slot}: {exc}")
                checkpoint_states.append(CandidateCheckpoint(
                    phase="proposal", iteration=iteration, slot=slot,
                    status="failed", error=f"propose: {exc}",
                ))
                log.warning(
                    "loop.propose_failed", iteration=iteration, slot=slot,
                    error=str(exc),
                )
                next_slot = slot + 1
                continue

            config = _proposal_config(proposal)
            try:
                record = _register_candidate(
                    config=config,
                    store=store,
                    parent=archive_snapshot[-1][0].id if archive_snapshot else None,
                    label=(
                        f"iter-{len(archive)}" if candidates_per_iteration == 1
                        else f"iter-{iteration}-slot-{slot}"
                    ),
                    meta=_proposal_metadata(
                        proposal, phase="proposal", iteration=iteration, slot=slot,
                        run_id=run_id,
                    ),
                )
            except LoopError as exc:
                errors.append(f"iter {iteration} slot {slot}: {exc}")
                checkpoint_states.append(CandidateCheckpoint(
                    phase="proposal", iteration=iteration, slot=slot,
                    status="failed", error=str(exc),
                ))
                next_slot = slot + 1
                continue
            checkpoint_states.append(CandidateCheckpoint(
                candidate_id=record.id, phase="proposal", iteration=iteration,
                slot=slot, status="registered",
            ))
            prepared.append(_PreparedCandidate(
                record=record, config=config, iteration=iteration, slot=slot,
                state_index=len(checkpoint_states) - 1,
            ))
            next_slot = slot + 1

        if interrupted:
            final_checkpoint = _checkpoint(
                archive, attempts, seed_count, iteration, next_slot,
                checkpoint_states, start,
            )
            return _result(
                archive, start, "interrupted", errors, final_checkpoint,
                on_checkpoint,
            )

        if prepared:
            _emit_checkpoint(on_checkpoint, _checkpoint(
                archive, attempts, seed_count, iteration, next_slot,
                checkpoint_states, start,
            ))

        batch_results, batch_interrupted = _evaluate_batch(
            prepared=prepared,
            store=store,
            evaluator=evaluator,
            registered_evaluator=registered_evaluator,
            gate=gate,
            max_parallel=max_parallel,
            evaluation_timeout_s=evaluation_timeout_s,
        )
        outcomes = list(zip(prepared, batch_results, strict=True)) + already_scored
        for candidate, outcome in sorted(outcomes, key=lambda item: item[0].slot):
            state_entry = checkpoint_states[candidate.state_index]
            if isinstance(outcome, ScoreCard):
                archive.append((candidate.record, outcome))
                state_entry.status = "scored"
                if on_iteration is not None:
                    on_iteration(len(archive), candidate.record, outcome)
                log.info(
                    "loop.iter_done",
                    iter=iteration,
                    slot=candidate.slot,
                    candidate=candidate.record.id,
                    quality=round(outcome.quality, 4),
                    outcome=round(outcome.outcome_score, 4),
                )
            else:
                if outcome == "interrupted":
                    state_entry.status = "registered"
                    state_entry.error = None
                else:
                    state_entry.status = "timed_out" if outcome == "timed out" else "failed"
                    state_entry.error = outcome
                    errors.append(
                        f"iter {iteration} slot {candidate.slot}: {outcome}"
                    )
                log.warning(
                    "loop.evaluate_failed", iteration=iteration,
                    slot=candidate.slot, error=outcome,
                )

        if batch_interrupted:
            interrupted_checkpoint = _checkpoint(
                archive, attempts, seed_count, iteration, candidates_per_iteration,
                checkpoint_states, start,
            )
            return _result(
                archive, start, "interrupted", errors, interrupted_checkpoint,
                on_checkpoint,
            )

        iteration += 1
        batch_checkpoint = _checkpoint(
            archive, attempts, seed_count, iteration, 0,
            checkpoint_states, start,
        )
        _emit_checkpoint(on_checkpoint, batch_checkpoint)


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
    evaluator: EvaluatorFn | None,
    parent: str | None,
    label: str,
    gate: GateFn | None = None,
    meta: dict[str, Any] | None = None,
    registered_evaluator: RegisteredEvaluatorFn | None = None,
) -> tuple[CandidateRecord, ScoreCard]:
    """Register ``config`` in the store, optionally gate, evaluate, persist.

    The gate (WIN-197) runs after registration so any ``code_artifacts``
    on the candidate's disk are visible to the pytest subprocess, but
    before scoring so an evaluator that loads the candidate's adapters
    never sees a broken module. A gate failure raises :class:`LoopError`
    and writes ``gate.json`` for inspection; ``scores.json`` stays empty
    so the candidate sorts to the bottom of any fitness ranking.
    """
    record = _register_candidate(
        config=config, store=store, parent=parent, label=label, meta=meta,
    )

    card = _evaluate_registered(
        record=record, config=config, store=store, evaluator=evaluator,
        registered_evaluator=registered_evaluator, gate=gate,
    )
    return record, card


def _register_candidate(
    *,
    config: dict[str, Any],
    store: ArtifactStore,
    parent: str | None,
    label: str,
    meta: dict[str, Any] | None = None,
) -> CandidateRecord:
    try:
        return store.register_candidate(
            config=config, parent=parent, label=label, meta=meta,
        )
    except Exception as exc:
        raise LoopError(f"failed to register candidate: {exc}") from exc


def _evaluate_registered(
    *,
    record: CandidateRecord,
    config: dict[str, Any],
    store: ArtifactStore,
    evaluator: EvaluatorFn | None,
    registered_evaluator: RegisteredEvaluatorFn | None = None,
    gate: GateFn | None = None,
) -> ScoreCard:

    if gate is not None and record.base_dir is not None:
        try:
            gate_result = gate(record.base_dir)
        except Exception as exc:
            raise LoopError(
                f"gate raised for candidate {record.id}: {exc}"
            ) from exc
        try:
            store.write_gate_result(record.id, gate_result)
        except Exception as exc:
            # A failure to persist the gate result is logged but not
            # fatal — the gate's verdict is already in memory and the
            # loop should still honour it.
            log.warning(
                "loop.gate_persist_failed",
                id=record.id,
                error=str(exc),
            )
        if not getattr(gate_result, "passed", False):
            raise LoopError(
                f"candidate {record.id} rejected by pytest gate"
            )

    try:
        if registered_evaluator is not None:
            card = registered_evaluator(config, record)
        elif evaluator is not None:
            card = evaluator(config)
        else:  # guarded by run_loop; retained for direct private calls
            raise LoopError("evaluator or registered_evaluator is required")
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
    return card


def _evaluate_batch(
    *,
    prepared: list[_PreparedCandidate],
    store: ArtifactStore,
    evaluator: EvaluatorFn | None,
    registered_evaluator: RegisteredEvaluatorFn | None,
    gate: GateFn | None,
    max_parallel: int,
    evaluation_timeout_s: float | None,
) -> tuple[list[ScoreCard | str], bool]:
    if not prepared:
        return [], False
    if max_parallel == 1 and evaluation_timeout_s is None:
        sequential_results: list[ScoreCard | str] = []
        for candidate in prepared:
            try:
                sequential_results.append(_evaluate_registered(
                    record=candidate.record,
                    config=candidate.config,
                    store=store,
                    evaluator=evaluator,
                    registered_evaluator=registered_evaluator,
                    gate=gate,
                ))
            except KeyboardInterrupt:
                sequential_results.append("interrupted")
                sequential_results.extend(
                    "interrupted" for _ in prepared[len(sequential_results) :]
                )
                return sequential_results, True
            except LoopError as exc:
                sequential_results.append(str(exc))
        return sequential_results, False

    executor = ThreadPoolExecutor(max_workers=min(max_parallel, len(prepared)))
    futures: list[Future[ScoreCard]] = [
        executor.submit(
            _evaluate_registered,
            record=candidate.record,
            config=candidate.config,
            store=store,
            evaluator=evaluator,
            registered_evaluator=registered_evaluator,
            gate=gate,
        )
        for candidate in prepared
    ]
    try:
        done, _pending = wait(futures, timeout=evaluation_timeout_s)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        return ["interrupted" for _candidate in prepared], True

    results: list[ScoreCard | str] = []
    for future in futures:
        if future not in done:
            future.cancel()
            results.append("timed out")
            continue
        try:
            results.append(future.result())
        except KeyboardInterrupt:
            results.append("interrupted")
        except LoopError as exc:
            results.append(str(exc))
        except Exception as exc:  # defensive: worker failures must remain per-slot
            results.append(f"evaluation worker raised: {exc}")
    executor.shutdown(wait=evaluation_timeout_s is None, cancel_futures=True)
    return results, any(result == "interrupted" for result in results)


def _coerce_proposal(value: dict[str, Any] | Proposal) -> Proposal:
    if isinstance(value, Proposal):
        return value
    if isinstance(value, BaseModel):
        try:
            return Proposal.model_validate(value.model_dump())
        except Exception as exc:
            raise LoopError(f"invalid typed proposal: {exc}") from exc
    if not isinstance(value, dict):
        raise LoopError(
            f"proposer returned {type(value).__name__}; expected dict or Proposal"
        )
    return Proposal(config=dict(value))


def _proposal_config(proposal: Proposal) -> dict[str, Any]:
    config = dict(proposal.config)
    if proposal.code_artifacts:
        from .store import CODE_ARTIFACTS_KEY

        config[CODE_ARTIFACTS_KEY] = dict(proposal.code_artifacts)
    return config


def _proposal_metadata(
    proposal: Proposal,
    *,
    phase: Literal["seed", "proposal"],
    iteration: int,
    slot: int,
    run_id: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if proposal.hypothesis is not None:
        metadata["hypothesis"] = proposal.hypothesis.model_dump(
            mode="json", exclude_none=True
        )
    if proposal.usage:
        metadata["usage"] = dict(proposal.usage)
    metadata.update({"phase": phase, "iteration": iteration, "slot": slot})
    if run_id is not None:
        metadata["run_id"] = run_id
    return metadata


def _config_key(config: dict[str, Any]) -> str:
    try:
        return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LoopError(f"candidate config is not JSON-serializable: {exc}") from exc


def _coerce_checkpoint(
    checkpoint: LoopCheckpoint | Mapping[str, Any] | None,
) -> LoopCheckpoint:
    if checkpoint is None:
        return LoopCheckpoint()
    if isinstance(checkpoint, LoopCheckpoint):
        return checkpoint.model_copy(deep=True)
    return LoopCheckpoint.from_dict(checkpoint)


def _restore_archive(
    checkpoint: LoopCheckpoint,
    store: ArtifactStore,
) -> list[tuple[CandidateRecord, ScoreCard]]:
    archive: list[tuple[CandidateRecord, ScoreCard]] = []
    for candidate_id in checkpoint.archive_ids:
        try:
            record = store.read_candidate(candidate_id)
            card = _scorecard_from_record(record)
            if card is None:
                raise LoopError("candidate has no valid scores")
            archive.append((record, card))
        except Exception as exc:
            raise LoopError(
                f"failed to restore checkpoint candidate {candidate_id!r}: {exc}"
            ) from exc
    return archive


def _restore_registered_candidates(
    candidates: list[CandidateCheckpoint],
    *,
    iteration: int,
    store: ArtifactStore,
) -> tuple[
    list[_PreparedCandidate],
    list[tuple[_PreparedCandidate, ScoreCard]],
]:
    prepared: list[_PreparedCandidate] = []
    already_scored: list[tuple[_PreparedCandidate, ScoreCard]] = []
    for state_index, candidate in enumerate(candidates):
        if (
            candidate.phase != "proposal"
            or candidate.iteration != iteration
            or candidate.status != "registered"
            or candidate.candidate_id is None
        ):
            continue
        try:
            record = store.read_candidate(candidate.candidate_id)
        except Exception as exc:
            raise LoopError(
                f"failed to restore registered candidate {candidate.candidate_id!r}: {exc}"
            ) from exc
        restored = _PreparedCandidate(
            record=record,
            config=record.config,
            iteration=iteration,
            slot=candidate.slot,
            state_index=state_index,
        )
        card = _scorecard_from_record(record)
        if card is None:
            prepared.append(restored)
        else:
            already_scored.append((restored, card))
    return (
        sorted(prepared, key=lambda item: item.slot),
        sorted(already_scored, key=lambda item: item[0].slot),
    )


def _scorecard_from_record(record: CandidateRecord) -> ScoreCard | None:
    if not record.scores:
        return None
    score_fields = {
        key: value
        for key, value in record.scores.items()
        if key in ScoreCard.__dataclass_fields__
    }
    if not score_fields:
        return None
    try:
        card = ScoreCard(**score_fields)
        card.to_dict()
    except (TypeError, ValueError):
        return None
    return card


def _checkpoint(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    attempts: int,
    next_seed_index: int,
    next_iteration: int,
    next_slot: int,
    candidates: list[CandidateCheckpoint],
    start: float,
) -> LoopCheckpoint:
    cards = [card for _record, card in archive]
    frontier_cards = {id(card) for card in pareto_frontier(cards)}
    return LoopCheckpoint(
        attempts=attempts,
        next_seed_index=next_seed_index,
        next_iteration=next_iteration,
        next_slot=next_slot,
        archive_ids=[record.id for record, _card in archive],
        frontier_ids=[
            record.id for record, card in archive if id(card) in frontier_cards
        ],
        candidates=[candidate.model_copy(deep=True) for candidate in candidates],
        elapsed_s=time.monotonic() - start,
    )


def _emit_checkpoint(
    callback: CheckpointCallback | None,
    checkpoint: LoopCheckpoint,
) -> None:
    if callback is not None:
        callback(checkpoint.model_copy(deep=True))


def _result(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    start: float,
    reason: str,
    errors: list[str],
    checkpoint: LoopCheckpoint | None = None,
    on_checkpoint: CheckpointCallback | None = None,
) -> LoopResult:
    if checkpoint is not None:
        _emit_checkpoint(on_checkpoint, checkpoint)
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
        checkpoint=checkpoint,
    )


__all__ = [
    "CandidateCheckpoint",
    "CheckpointCallback",
    "EvaluatorFn",
    "GateFn",
    "Hypothesis",
    "LoopCheckpoint",
    "LoopError",
    "LoopResult",
    "Proposal",
    "ProposerFn",
    "RegisteredEvaluatorFn",
    "run_loop",
]


# Re-export Path so the CLI can use the same Path symbol the loop
# returns in its LoopResult dataclass without an extra import.
__path_re_export__ = Path
