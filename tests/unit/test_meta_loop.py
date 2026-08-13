"""Unit tests for the Meta-Harness Algorithm 1 outer loop (WIN-187)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stropha.meta import (
    ArtifactStore,
    Hypothesis,
    LoopCheckpoint,
    Proposal,
    ScoreCard,
    run_loop,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "meta")


def _evaluator_returning(card: ScoreCard):
    def _eval(_cfg: dict[str, Any]) -> ScoreCard:
        return card
    return _eval


def _proposer_returning(configs: list[dict[str, Any]]):
    """Proposer that yields configs in order; raises StopIteration after."""
    it = iter(configs)

    def _propose(_archive: list) -> dict[str, Any]:
        return next(it)
    return _propose


# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------


class TestDefaultBudget:
    def test_default_max_iterations_is_50(self, tmp_path: Path) -> None:
        """With no budget knobs, the loop stops at 50 iterations."""
        store = _store(tmp_path)
        proposer = lambda _a: {"v": 1}  # never-ending  # noqa: E731
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        result = run_loop(proposer=proposer, evaluator=evaluator, store=store)
        assert result.iterations == 50
        assert "budget exhausted" in result.stopped_reason

    def test_max_iterations_caps(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        proposer = lambda _a: {"v": 1}  # noqa: E731
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        result = run_loop(
            proposer=proposer, evaluator=evaluator, store=store,
            max_iterations=3,
        )
        assert result.iterations == 3

    def test_wall_clock_stops_loop(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        proposer = lambda _a: {"v": 1}  # noqa: E731

        def slow_eval(_cfg: dict[str, Any]) -> ScoreCard:
            time.sleep(0.02)
            return ScoreCard(quality=0.5)

        result = run_loop(
            proposer=proposer, evaluator=slow_eval, store=store,
            wall_clock_s=0.05,
        )
        # Should fit ~2 iterations in 50ms. Hard cap: ≤10 to allow CI slop.
        assert 1 <= result.iterations <= 10
        assert "budget exhausted" in result.stopped_reason


# ---------------------------------------------------------------------------
# Seed init
# ---------------------------------------------------------------------------


class TestSeedInit:
    def test_seeds_evaluated_first_in_order(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        proposer = lambda _a: {"v": "from-proposer"}  # noqa: E731
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        result = run_loop(
            proposer=proposer, evaluator=evaluator, store=store,
            seeds=[{"v": "seed-1"}, {"v": "seed-2"}],
            max_iterations=4,  # 2 seeds + 2 proposer
        )
        assert result.iterations == 4
        labels = [rec.label for rec, _ in result.archive]
        assert labels[0] == "seed-0"
        assert labels[1] == "seed-1"

    def test_seed_failure_records_error_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store(tmp_path)
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))

        # Force the first seed registration to raise.
        original_register = store.register_candidate
        calls = {"n": 0}

        def flaky_register(*a, **kw):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated store error")
            return original_register(*a, **kw)

        monkeypatch.setattr(store, "register_candidate", flaky_register)
        result = run_loop(
            proposer=_proposer_returning([{"v": "p1"}]),
            evaluator=evaluator, store=store,
            seeds=[{"v": "seed-bad"}, {"v": "seed-ok"}],
            max_iterations=3,  # 2 seeds (1 fails) + 1 proposer
        )
        # The bad seed registered but didn't reach the archive.
        assert result.iterations == 2  # seed-ok + 1 proposer
        assert any("seed 0" in e for e in result.errors)

    def test_seed_budget_exhausted_short_circuits(self, tmp_path: Path) -> None:
        """When budget is exhausted during seed init, stopped_reason
        surfaces the right message."""
        store = _store(tmp_path)
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        proposer = lambda _a: {"v": 1}  # noqa: E731
        result = run_loop(
            proposer=proposer, evaluator=evaluator, store=store,
            seeds=[{"v": f"s-{i}"} for i in range(10)],
            max_iterations=3,
        )
        assert result.iterations == 3
        assert "seed init" in result.stopped_reason

    def test_duplicate_seeds_are_skipped_in_input_order(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            calls.append(config)
            return ScoreCard(quality=0.5)

        result = run_loop(
            proposer=lambda _archive: {"v": "proposal"},
            evaluator=evaluator,
            store=_store(tmp_path),
            seeds=[{"v": 1}, {"v": 1}, {"v": 2}],
            max_iterations=3,
        )

        assert calls == [{"v": 1}, {"v": 2}]
        assert result.iterations == 2
        assert result.checkpoint is not None
        duplicate = result.checkpoint.candidates[1]
        assert duplicate.status == "deduplicated"
        assert duplicate.duplicate_of == result.archive[0][0].id


class TestTypedProposal:
    def test_typed_proposal_metadata_is_persisted(self, tmp_path: Path) -> None:
        proposal = Proposal(
            config={"v": 2},
            hypothesis=Hypothesis(
                statement="Improve quality",
                expected_objective_deltas={"quality": 0.1},
                confidence=0.8,
            ),
            usage={"input_tokens": 12},
        )

        result = run_loop(
            proposer=lambda _archive: proposal,
            evaluator=_evaluator_returning(ScoreCard(quality=0.7)),
            store=_store(tmp_path),
            max_iterations=1,
        )

        record = result.archive[0][0]
        assert record.config == {"v": 2}
        assert record.meta["phase"] == "proposal"
        assert record.meta["iteration"] == 0
        assert record.meta["slot"] == 0
        assert record.meta["hypothesis"]["statement"] == "Improve quality"
        assert record.meta["hypothesis"]["expected_objective_deltas"] == {
            "quality": 0.1,
        }
        assert record.meta["usage"] == {"input_tokens": 12}

    def test_hypothesis_is_typed(self) -> None:
        proposal = Proposal(
            config={}, hypothesis=Hypothesis(statement="Try a smaller context")
        )
        assert proposal.hypothesis is not None
        assert proposal.hypothesis.statement == "Try a smaller context"

    def test_agentic_proposal_model_is_the_public_type(self, tmp_path: Path) -> None:
        result = run_loop(
            proposer=lambda _archive: Proposal(config={"v": 4}),
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=_store(tmp_path),
            max_iterations=1,
        )
        assert result.archive[0][0].config == {"v": 4}

    def test_registered_evaluator_receives_record_and_run_id(
        self, tmp_path: Path
    ) -> None:
        records = []

        def evaluator(config: dict[str, Any], record) -> ScoreCard:
            assert record.base_dir is not None
            assert record.base_dir.is_dir()
            assert record.config == config
            records.append(record)
            return ScoreCard(quality=0.5)

        result = run_loop(
            proposer=lambda _archive: {"v": "proposal"},
            registered_evaluator=evaluator,
            store=_store(tmp_path),
            seeds=[{"v": "seed"}],
            run_id="run-123",
            max_iterations=2,
        )

        assert records == [record for record, _card in result.archive]
        assert [record.meta["run_id"] for record in records] == ["run-123", "run-123"]

    def test_legacy_evaluator_remains_single_argument(self, tmp_path: Path) -> None:
        seen: list[dict[str, Any]] = []

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            seen.append(config)
            return ScoreCard(quality=0.5)

        result = run_loop(
            proposer=lambda _archive: {"v": 1},
            evaluator=evaluator,
            store=_store(tmp_path),
            max_iterations=1,
        )

        assert seen == [{"v": 1}]
        assert "run_id" not in result.archive[0][0].meta


class TestCandidateBatches:
    def test_batch_proposals_share_archive_snapshot_and_finish_in_slot_order(
        self, tmp_path: Path
    ) -> None:
        proposals = iter([{"slot": 0}, {"slot": 1}, {"slot": 2}])
        archive_sizes: list[int] = []
        hooks: list[int] = []

        def proposer(archive: list) -> dict[str, Any]:
            archive_sizes.append(len(archive))
            return next(proposals)

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            time.sleep((2 - config["slot"]) * 0.01)
            return ScoreCard(quality=float(config["slot"]))

        result = run_loop(
            proposer=proposer,
            evaluator=evaluator,
            store=_store(tmp_path),
            max_iterations=3,
            candidates_per_iteration=3,
            max_parallel=3,
            on_iteration=lambda _idx, record, _card: hooks.append(record.config["slot"]),
        )

        assert archive_sizes == [0, 0, 0]
        assert [record.config["slot"] for record, _card in result.archive] == [0, 1, 2]
        assert hooks == [0, 1, 2]
        assert [record.meta["slot"] for record, _card in result.archive] == [0, 1, 2]

    def test_max_parallel_bounds_evaluator_concurrency(self, tmp_path: Path) -> None:
        lock = threading.Lock()
        active = 0
        peak = 0
        next_slot = iter(range(4))

        def evaluator(_config: dict[str, Any]) -> ScoreCard:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return ScoreCard(quality=0.5)

        run_loop(
            proposer=lambda _archive: {"slot": next(next_slot)},
            evaluator=evaluator,
            store=_store(tmp_path),
            max_iterations=4,
            candidates_per_iteration=4,
            max_parallel=2,
        )

        assert peak == 2

    def test_candidates_per_iter_alias(self, tmp_path: Path) -> None:
        slots = iter(range(2))
        result = run_loop(
            proposer=lambda _archive: {"slot": next(slots)},
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=_store(tmp_path),
            max_iterations=2,
            candidates_per_iter=2,
        )
        assert result.iterations == 2

    def test_soft_timeout_marks_candidate_without_waiting(self, tmp_path: Path) -> None:
        def slow_evaluator(_config: dict[str, Any]) -> ScoreCard:
            time.sleep(0.15)
            return ScoreCard()

        started = time.monotonic()
        result = run_loop(
            proposer=lambda _archive: {"v": 1},
            evaluator=slow_evaluator,
            store=_store(tmp_path),
            max_iterations=1,
            evaluation_timeout_s=0.01,
        )

        assert time.monotonic() - started < 0.1
        assert result.archive == []
        assert result.checkpoint is not None
        assert result.checkpoint.candidates[0].status == "timed_out"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_proposer_exception_logged_and_continues(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        calls = {"n": 0}

        def flaky_propose(_a: list) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated propose failure")
            return {"v": calls["n"]}

        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        # Budget counts every attempt (success or failure) — so for
        # 1 expected archive entry plus 1 failed attempt we need
        # max_iterations=2.
        result = run_loop(
            proposer=flaky_propose, evaluator=evaluator, store=store,
            max_iterations=2,
        )
        assert result.iterations == 1  # one successful entry in the archive
        assert any("propose" in e for e in result.errors)

    def test_evaluator_exception_recorded(self, tmp_path: Path) -> None:
        store = _store(tmp_path)

        def boom(_cfg: dict[str, Any]) -> ScoreCard:
            raise RuntimeError("simulated evaluator failure")

        # Failure-only loops MUST still terminate per the budget — the
        # loop counts attempts (success or failure), not just archive
        # adds. Otherwise an always-failing evaluator would hang the
        # process. We use max_iterations=3; archive stays empty,
        # errors list has 3 entries.
        result = run_loop(
            proposer=lambda _a: {"v": 1},
            evaluator=boom,
            store=store,
            max_iterations=3,
        )
        assert result.iterations == 0  # archive empty (all attempts failed)
        assert len(result.errors) == 3  # but 3 attempts were made
        assert all("evaluator raised" in e for e in result.errors)

    def test_max_iterations_zero_skips_loop_entirely(self, tmp_path: Path) -> None:
        """With max_iterations=0 the loop body never enters and the
        result reflects empty archive."""
        store = _store(tmp_path)
        proposer = lambda _a: {"v": 1}  # noqa: E731
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        result = run_loop(
            proposer=proposer, evaluator=evaluator, store=store,
            max_iterations=0,
        )
        assert result.iterations == 0
        assert result.errors == []

    def test_keyboard_interrupt_in_proposer_returns_archive(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)

        def kbd_propose(_a: list) -> dict[str, Any]:
            raise KeyboardInterrupt

        result = run_loop(
            proposer=kbd_propose,
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=store, max_iterations=10,
        )
        assert result.stopped_reason == "interrupted"
        assert result.archive == []

    def test_keyboard_interrupt_in_evaluator_returns_archive(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)

        def kbd_eval(_cfg: dict[str, Any]) -> ScoreCard:
            raise KeyboardInterrupt

        result = run_loop(
            proposer=lambda _a: {"v": 1},
            evaluator=kbd_eval,
            store=store, max_iterations=10,
        )
        assert result.stopped_reason == "interrupted"

    def test_store_write_scores_failure_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store(tmp_path)

        def boom_write(*_a, **_kw):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated scores write failure")

        monkeypatch.setattr(store, "write_scores", boom_write)
        evaluator = _evaluator_returning(ScoreCard(quality=0.5))
        result = run_loop(
            proposer=lambda _a: {"v": 1},
            evaluator=evaluator, store=store,
            max_iterations=2,
        )
        # Errors logged; archive empty because writes failed.
        assert result.archive == []
        assert any("write scores" in e for e in result.errors)


# ---------------------------------------------------------------------------
# on_iteration hook + Pareto frontier
# ---------------------------------------------------------------------------


class TestHookAndFrontier:
    def test_on_iteration_called_each_step(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        calls: list[int] = []

        def hook(idx: int, _rec, _card) -> None:
            calls.append(idx)

        run_loop(
            proposer=lambda _a: {"v": 1},
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=store, max_iterations=3, on_iteration=hook,
        )
        assert calls == [1, 2, 3]

    def test_on_iteration_called_for_seed_init_too(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        calls: list[int] = []

        def hook(idx: int, _rec, _card) -> None:
            calls.append(idx)

        run_loop(
            proposer=lambda _a: {"v": "from-proposer"},
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=store,
            seeds=[{"v": "seed-A"}, {"v": "seed-B"}],
            max_iterations=3,  # 2 seeds + 1 proposer
            on_iteration=hook,
        )
        # Hook fires for both seeds and the proposer iteration.
        assert calls == [1, 2, 3]

    def test_frontier_contains_only_non_dominated(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        cards = [
            ScoreCard(quality=0.9, latency_ms=100),  # accurate, slow
            ScoreCard(quality=0.6, latency_ms=10),   # less accurate, fast
            ScoreCard(quality=0.5, latency_ms=200),  # dominated
        ]
        idx = {"n": 0}

        def cycling_eval(_cfg: dict[str, Any]) -> ScoreCard:
            c = cards[idx["n"] % len(cards)]
            idx["n"] += 1
            return c

        result = run_loop(
            proposer=lambda _a: {"v": idx["n"]},
            evaluator=cycling_eval, store=store,
            max_iterations=3,
        )
        # All three entered the archive. The dominated one is excluded
        # from the frontier; the other two stay (Pareto trade-off).
        assert len(result.archive) == 3
        assert len(result.frontier) == 2


class TestCheckpoints:
    def test_callbacks_are_emitted_at_stable_boundaries(self, tmp_path: Path) -> None:
        checkpoints: list[LoopCheckpoint] = []
        slots = iter(range(2))

        result = run_loop(
            proposer=lambda _archive: {"slot": next(slots)},
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=_store(tmp_path),
            seeds=[{"seed": 1}],
            max_iterations=3,
            candidates_per_iteration=2,
            on_checkpoint=checkpoints.append,
        )

        assert len(checkpoints) == 4
        assert checkpoints[0].next_seed_index == 1
        assert checkpoints[0].archive_ids == [result.archive[0][0].id]
        assert [candidate.status for candidate in checkpoints[1].candidates[-2:]] == [
            "registered", "registered",
        ]
        assert checkpoints[2].next_iteration == 1
        assert checkpoints[2].archive_ids == [record.id for record, _ in result.archive]
        assert checkpoints[3].archive_ids == checkpoints[2].archive_ids
        assert LoopCheckpoint.from_dict(checkpoints[3].to_dict()) == checkpoints[3]

    def test_terminal_seed_budget_emits_checkpoint(self, tmp_path: Path) -> None:
        checkpoints: list[LoopCheckpoint] = []

        result = run_loop(
            proposer=lambda _archive: {"unexpected": True},
            evaluator=_evaluator_returning(ScoreCard()),
            store=_store(tmp_path),
            seeds=[{"v": 1}],
            max_iterations=0,
            on_checkpoint=checkpoints.append,
        )

        assert checkpoints == [result.checkpoint]

    def test_resume_restores_scores_without_rerunning_candidates(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        first = run_loop(
            proposer=lambda _archive: {"v": 1},
            evaluator=_evaluator_returning(ScoreCard(quality=0.4)),
            store=store,
            max_iterations=1,
        )
        assert first.checkpoint is not None
        evaluated: list[int] = []

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            evaluated.append(config["v"])
            return ScoreCard(quality=0.8)

        resumed = run_loop(
            proposer=lambda _archive: {"v": 2},
            evaluator=evaluator,
            store=store,
            max_iterations=2,
            checkpoint=first.checkpoint.to_dict(),
        )

        assert evaluated == [2]
        assert [record.config["v"] for record, _card in resumed.archive] == [1, 2]
        assert resumed.archive[0][1].quality == 0.4

    def test_resume_recovers_registered_candidate_once(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        record = store.register_candidate(
            {"v": 7},
            label="iter-0-slot-0",
            meta={"phase": "proposal", "iteration": 0, "slot": 0},
        )
        checkpoint = LoopCheckpoint.model_validate({
            "attempts": 1,
            "next_iteration": 0,
            "next_slot": 1,
            "candidates": [{
                "candidate_id": record.id,
                "phase": "proposal",
                "iteration": 0,
                "slot": 0,
                "status": "registered",
            }],
        })
        evaluated: list[int] = []

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            evaluated.append(config["v"])
            return ScoreCard(quality=0.9)

        result = run_loop(
            proposer=lambda _archive: {"unexpected": True},
            evaluator=evaluator,
            store=store,
            max_iterations=1,
            checkpoint=checkpoint,
        )

        assert evaluated == [7]
        assert [candidate.id for candidate, _card in result.archive] == [record.id]
        assert result.checkpoint is not None
        assert result.checkpoint.candidates[0].status == "scored"

    def test_resume_reconciles_registered_candidate_with_existing_scores(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        record = store.register_candidate(
            {"v": 9},
            label="iter-0-slot-0",
            meta={"phase": "proposal", "iteration": 0, "slot": 0},
        )
        store.write_scores(record.id, ScoreCard(quality=0.75).to_dict())
        checkpoint = LoopCheckpoint.model_validate({
            "attempts": 1,
            "next_iteration": 0,
            "next_slot": 1,
            "candidates": [{
                "candidate_id": record.id,
                "phase": "proposal",
                "iteration": 0,
                "slot": 0,
                "status": "registered",
            }],
        })

        def unexpected_evaluator(_config: dict[str, Any]) -> ScoreCard:
            pytest.fail("scored candidate was evaluated again")

        result = run_loop(
            proposer=lambda _archive: {"unexpected": True},
            evaluator=unexpected_evaluator,
            store=store,
            max_iterations=1,
            checkpoint=checkpoint,
        )

        assert [(candidate.id, card.quality) for candidate, card in result.archive] == [
            (record.id, 0.75),
        ]
        assert result.checkpoint is not None
        assert result.checkpoint.candidates[0].status == "scored"

    def test_resume_reruns_registered_candidate_with_invalid_scores(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        record = store.register_candidate(
            {"v": 10},
            label="iter-0-slot-0",
            meta={"phase": "proposal", "iteration": 0, "slot": 0},
        )
        store.write_scores(record.id, {"unrelated": 1})
        checkpoint = LoopCheckpoint.model_validate({
            "attempts": 1,
            "next_iteration": 0,
            "next_slot": 1,
            "candidates": [{
                "candidate_id": record.id,
                "phase": "proposal",
                "iteration": 0,
                "slot": 0,
                "status": "registered",
            }],
        })
        evaluated: list[int] = []

        def evaluator(config: dict[str, Any]) -> ScoreCard:
            evaluated.append(config["v"])
            return ScoreCard(quality=0.8)

        result = run_loop(
            proposer=lambda _archive: {"unexpected": True},
            evaluator=evaluator,
            store=store,
            max_iterations=1,
            checkpoint=checkpoint,
        )

        assert evaluated == [10]
        assert result.archive[0][1].quality == 0.8


# ---------------------------------------------------------------------------
# LoopResult
# ---------------------------------------------------------------------------


class TestLoopResultToDict:
    def test_to_dict_shape(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        result = run_loop(
            proposer=lambda _a: {"v": 1},
            evaluator=_evaluator_returning(ScoreCard(quality=0.5)),
            store=store, max_iterations=2,
        )
        d = result.to_dict()
        for key in (
            "iterations", "elapsed_s", "archive_size", "frontier_size",
            "stopped_reason", "errors", "frontier",
        ):
            assert key in d
        assert d["iterations"] == 2
        assert isinstance(d["frontier"], list)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        for name in (
            "run_loop", "LoopResult", "LoopError", "ProposerFn", "EvaluatorFn",
            "Proposal", "Hypothesis", "LoopCheckpoint", "CandidateCheckpoint",
            "CheckpointCallback",
        ):
            assert hasattr(meta, name)
