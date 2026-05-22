"""Unit tests for the Meta-Harness Algorithm 1 outer loop (WIN-187)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from stropha.meta import (
    ArtifactStore,
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
        for name in ("run_loop", "LoopResult", "LoopError", "ProposerFn", "EvaluatorFn"):
            assert hasattr(meta, name)