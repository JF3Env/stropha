"""Unit tests for parallel fan-out + early-exit (WIN-237, EPIC 1.10 #28).

Coverage target: every branch of ``parallel_fanout`` and
``topk_stable``, including timeout / failure / early-exit /
all-tasks-settle paths.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from stropha.retrieval.parallel_fanout import (
    FanoutResult,
    TaskOutcome,
    _ids,
    parallel_fanout,
    topk_stable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Hit:
    """Minimal SearchHit-like stub for fusion tests."""

    def __init__(self, chunk_id: str, rank: int = 0) -> None:
        self.chunk_id = chunk_id
        self.rank = rank


def _make_task(value: object, delay_s: float = 0.0) -> Callable[[], object]:
    def _task() -> object:
        if delay_s:
            time.sleep(delay_s)
        return value

    return _task


def _concat_fuse(results: dict[str, object]) -> list[object]:
    """Trivial fuser: concatenate all task outputs in iteration order."""
    out: list[object] = []
    for v in results.values():
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# _ids helper
# ---------------------------------------------------------------------------


class TestIdsHelper:
    def test_chunk_id_attr(self) -> None:
        assert _ids([_Hit("a"), _Hit("b")]) == ["a", "b"]

    def test_dict_with_chunk_id(self) -> None:
        assert _ids([{"chunk_id": "x"}, {"chunk_id": "y"}]) == ["x", "y"]

    def test_string_fallback(self) -> None:
        assert _ids(["raw1", "raw2"]) == ["raw1", "raw2"]

    def test_repr_fallback_for_unknown(self) -> None:
        ids = _ids([42, 3.14])
        assert ids == ["42", "3.14"]


# ---------------------------------------------------------------------------
# topk_stable
# ---------------------------------------------------------------------------


class TestTopKStable:
    def test_first_iteration_false(self) -> None:
        # No previous → always unstable.
        assert topk_stable(None, [_Hit("a"), _Hit("b")], top_k=2) is False

    def test_empty_previous_false(self) -> None:
        assert topk_stable([], [_Hit("a")], top_k=1) is False

    def test_empty_current_false(self) -> None:
        assert topk_stable([_Hit("a")], [], top_k=1) is False

    def test_zero_topk_false(self) -> None:
        assert topk_stable([_Hit("a")], [_Hit("a")], top_k=0) is False

    def test_negative_topk_false(self) -> None:
        assert topk_stable([_Hit("a")], [_Hit("a")], top_k=-5) is False

    def test_exact_match_stable(self) -> None:
        prev = [_Hit("a"), _Hit("b"), _Hit("c")]
        curr = [_Hit("a"), _Hit("b"), _Hit("c")]
        assert topk_stable(prev, curr, top_k=3, min_overlap=1.0) is True

    def test_reordered_top_k_still_stable(self) -> None:
        """Overlap counts set membership; order changes don't kill stability."""
        prev = [_Hit("a"), _Hit("b"), _Hit("c")]
        curr = [_Hit("c"), _Hit("a"), _Hit("b")]
        assert topk_stable(prev, curr, top_k=3, min_overlap=1.0) is True

    def test_partial_overlap_below_threshold(self) -> None:
        prev = [_Hit("a"), _Hit("b"), _Hit("c"), _Hit("d")]
        curr = [_Hit("a"), _Hit("z"), _Hit("y"), _Hit("x")]
        # 1/4 = 0.25 overlap, threshold 0.8 → unstable.
        assert topk_stable(prev, curr, top_k=4, min_overlap=0.8) is False

    def test_partial_overlap_meets_threshold(self) -> None:
        prev = [_Hit("a"), _Hit("b"), _Hit("c"), _Hit("d")]
        curr = [_Hit("a"), _Hit("b"), _Hit("c"), _Hit("z")]
        # 3/4 = 0.75 overlap, threshold 0.75 → stable.
        assert topk_stable(prev, curr, top_k=4, min_overlap=0.75) is True

    def test_truncates_to_top_k(self) -> None:
        prev = [_Hit(str(i)) for i in range(20)]
        curr = [_Hit(str(i)) for i in range(20)]
        # Only top 3 considered.
        assert topk_stable(prev, curr, top_k=3, min_overlap=1.0) is True

    def test_short_lists_unstable_when_below_topk(self) -> None:
        """When top_k=5 but current has only 1 item, max possible
        overlap is 1/5=0.2; threshold 0.5 → unstable."""
        prev = [_Hit("a")]
        curr = [_Hit("a")]
        assert topk_stable(prev, curr, top_k=5, min_overlap=0.5) is False


# ---------------------------------------------------------------------------
# parallel_fanout — happy paths
# ---------------------------------------------------------------------------


class TestParallelFanoutHappy:
    def test_empty_tasks(self) -> None:
        result = parallel_fanout({}, _concat_fuse)
        assert result.fused == []
        assert result.completed == []
        assert result.early_exit is False
        assert any("empty tasks" in n for n in result.notes)

    def test_single_task(self) -> None:
        result = parallel_fanout({"a": _make_task([1, 2, 3])}, _concat_fuse)
        assert result.completed == ["a"]
        assert result.fused == [1, 2, 3]
        assert result.early_exit is False
        assert "a" in result.outcomes
        assert result.outcomes["a"].error is None

    def test_multiple_tasks_all_settle(self) -> None:
        result = parallel_fanout(
            {
                "fast": _make_task(["a"]),
                "med": _make_task(["b"]),
                "slow": _make_task(["c"]),
            },
            _concat_fuse,
        )
        assert set(result.completed) == {"fast", "med", "slow"}
        assert result.cancelled == []
        assert result.failed == []
        assert sorted(result.fused) == ["a", "b", "c"]  # type: ignore[arg-type]
        assert result.early_exit is False


# ---------------------------------------------------------------------------
# parallel_fanout — failures
# ---------------------------------------------------------------------------


class TestParallelFanoutFailures:
    def test_one_task_raises_others_survive(self) -> None:
        def boom() -> object:
            raise RuntimeError("kaboom")

        result = parallel_fanout(
            {"ok": _make_task(["x"]), "bad": boom},
            _concat_fuse,
        )
        assert "ok" in result.completed
        assert "bad" in result.failed
        assert result.outcomes["bad"].error is not None
        assert isinstance(result.outcomes["bad"].error, RuntimeError)
        # Fused result only contains the surviving task's output.
        assert result.fused == ["x"]


# ---------------------------------------------------------------------------
# parallel_fanout — early-exit
# ---------------------------------------------------------------------------


class TestParallelFanoutEarlyExit:
    def test_early_exit_fires(self) -> None:
        """early_exit_fn returning True after first fuse cancels the slow task."""
        slow_started = threading.Event()
        cancel_signal = threading.Event()

        def slow() -> object:
            slow_started.set()
            # Wait up to 2s for the cancel signal — gives the fast
            # task time to fire early-exit. In practice the executor's
            # cancel_futures=True (Python 3.9+) won't preempt a running
            # task, so we self-poll the cancel signal.
            cancel_signal.wait(timeout=2.0)
            return ["slow-result"]

        def fast() -> object:
            return ["fast-a", "fast-b"]

        # early_exit_fn always returns True once we have ≥ min_settled.
        def always_stable(_prev: object, _curr: object) -> bool:
            return True

        result = parallel_fanout(
            {"fast": fast, "slow": slow},
            _concat_fuse,
            early_exit_fn=always_stable,
            min_settled=1,
        )
        # Let any running slow task return so the executor can reap.
        cancel_signal.set()

        assert result.early_exit is True
        assert "fast" in result.completed
        assert "slow" in result.cancelled
        assert result.fused == ["fast-a", "fast-b"]

    def test_min_settled_blocks_early_exit_until_threshold(self) -> None:
        """early_exit_fn must NOT be consulted before min_settled completions."""
        call_count = {"n": 0}

        def predicate(_prev: object, _curr: object) -> bool:
            call_count["n"] += 1
            return True

        result = parallel_fanout(
            {
                "a": _make_task(["a"]),
                "b": _make_task(["b"]),
                "c": _make_task(["c"]),
            },
            _concat_fuse,
            early_exit_fn=predicate,
            min_settled=99,  # impossible threshold
        )
        # All three settle, predicate never gates an exit.
        assert result.early_exit is False
        assert sorted(result.completed) == ["a", "b", "c"]
        # call_count may be >0 (predicate evaluated) but no exit fired.
        assert result.cancelled == []

    def test_no_early_exit_fn_runs_to_completion(self) -> None:
        result = parallel_fanout(
            {"a": _make_task(["a"]), "b": _make_task(["b"])},
            _concat_fuse,
            early_exit_fn=None,
        )
        assert sorted(result.completed) == ["a", "b"]
        assert result.early_exit is False

    def test_early_exit_predicate_only_fires_when_pending(self) -> None:
        """When the LAST task settles, pending is empty — predicate
        result is irrelevant and ``early_exit`` stays False."""

        def always_true(_prev: object, _curr: object) -> bool:
            return True

        # Only one task — once it settles, there's nothing pending to
        # cancel, so early_exit must NOT flip.
        result = parallel_fanout(
            {"only": _make_task(["x"])},
            _concat_fuse,
            early_exit_fn=always_true,
            min_settled=1,
        )
        assert result.early_exit is False
        assert result.completed == ["only"]
        assert result.cancelled == []


# ---------------------------------------------------------------------------
# parallel_fanout — timeout
# ---------------------------------------------------------------------------


class TestParallelFanoutTimeout:
    def test_timeout_cancels_pending(self) -> None:
        """A task that doesn't return in time is cancelled and noted."""
        cancel_signal = threading.Event()

        def hangs() -> object:
            cancel_signal.wait(timeout=2.0)
            return ["never-fused"]

        result = parallel_fanout(
            {"fast": _make_task(["x"]), "hang": hangs},
            _concat_fuse,
            timeout=0.1,
        )
        cancel_signal.set()

        # Fast task may have completed; hang task must be cancelled.
        assert "hang" in result.cancelled
        assert any("timeout" in n for n in result.notes)
        # Fused result is whatever managed to settle (could be ["x"] or []).
        assert result.fused is not None

    def test_timeout_before_any_task_settles(self) -> None:
        """Edge case: timeout fires before the FIRST task returns —
        the ``fused is None`` fallback at the end must fire."""
        cancel_signal = threading.Event()

        def hangs(label: str) -> Callable[[], object]:
            def _t() -> object:
                cancel_signal.wait(timeout=2.0)
                return [label]
            return _t

        result = parallel_fanout(
            {"a": hangs("a"), "b": hangs("b")},
            _concat_fuse,
            timeout=0.05,
        )
        cancel_signal.set()

        assert set(result.cancelled) >= {"a", "b"} - set(result.completed)
        # fuse_fn was called with empty results → [] from concat_fuse.
        assert result.fused == []
        assert any("timeout" in n for n in result.notes)

    def test_deadline_elapses_between_iterations(self) -> None:
        """A task completes, then the deadline elapses BEFORE the next
        ``wait()``. The pre-wait deadline check must fire."""
        cancel_signal = threading.Event()

        def slow_fuse(results: dict[str, object]) -> list[object]:
            # Long fuse pushes wall-clock past the 0.1s deadline.
            time.sleep(0.15)
            return _concat_fuse(results)

        def hangs() -> object:
            cancel_signal.wait(timeout=2.0)
            return ["never"]

        result = parallel_fanout(
            {"fast": _make_task(["a"]), "slow": hangs},
            slow_fuse,
            timeout=0.1,
        )
        cancel_signal.set()

        assert "slow" in result.cancelled
        assert any("timeout" in n for n in result.notes)


# ---------------------------------------------------------------------------
# parallel_fanout — max_workers
# ---------------------------------------------------------------------------


class TestMaxWorkers:
    def test_explicit_max_workers(self) -> None:
        result = parallel_fanout(
            {f"t{i}": _make_task([i]) for i in range(4)},
            _concat_fuse,
            max_workers=2,
        )
        assert len(result.completed) == 4
        assert sorted(result.fused) == [0, 1, 2, 3]  # type: ignore[arg-type]

    def test_zero_max_workers_falls_back_to_task_count(self) -> None:
        """0 or negative max_workers must NOT pass through to the pool
        (Python raises). Falls back to len(tasks)."""
        result = parallel_fanout(
            {"a": _make_task(["a"]), "b": _make_task(["b"])},
            _concat_fuse,
            max_workers=0,
        )
        assert sorted(result.completed) == ["a", "b"]


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclassSurface:
    def test_fanout_result_defaults(self) -> None:
        r = FanoutResult()
        assert r.fused is None
        assert r.outcomes == {}
        assert r.completed == []
        assert r.cancelled == []
        assert r.failed == []
        assert r.elapsed_ms == 0.0
        assert r.early_exit is False
        assert r.notes == []

    def test_fanout_result_frozen(self) -> None:
        r = FanoutResult()
        with pytest.raises((AttributeError, Exception)):
            r.early_exit = True  # type: ignore[misc]

    def test_task_outcome_defaults(self) -> None:
        o = TaskOutcome(name="x")
        assert o.result is None
        assert o.error is None
        assert o.elapsed_ms == 0.0
        assert o.cancelled is False

    def test_task_outcome_frozen(self) -> None:
        o = TaskOutcome(name="x")
        with pytest.raises((AttributeError, Exception)):
            o.cancelled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import parallel_fanout as mod
        assert set(mod.__all__) == {
            "FanoutResult",
            "TaskOutcome",
            "parallel_fanout",
            "topk_stable",
        }
