"""Unit tests for streaming tool results (WIN-256, EPIC 1.12 #93)."""

from __future__ import annotations

import pytest

from stropha.retrieval.streaming import StreamBatch, stream_batches

# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_defaults(self) -> None:
        b = StreamBatch()
        assert b.items == []
        assert b.batch_index == 0
        assert b.cumulative == 0
        assert b.done is False
        assert b.early_exit is False
        assert b.note == ""

    def test_frozen(self) -> None:
        b = StreamBatch()
        with pytest.raises((AttributeError, Exception)):
            b.done = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# stream_batches — basic
# ---------------------------------------------------------------------------


class TestStreamBatchesBasic:
    def test_empty_iterable_emits_terminator(self) -> None:
        batches = list(stream_batches([]))
        assert len(batches) == 1
        assert batches[0].items == []
        assert batches[0].done is True
        assert "empty" in batches[0].note

    def test_single_batch_smaller_than_size(self) -> None:
        batches = list(stream_batches([1, 2, 3], batch_size=10))
        assert len(batches) == 1
        assert batches[0].items == [1, 2, 3]
        assert batches[0].done is True
        assert batches[0].cumulative == 3

    def test_exact_multiple_of_batch_size(self) -> None:
        batches = list(stream_batches([1, 2, 3, 4], batch_size=2))
        # 2 + 2 + terminator (done=True).
        assert len(batches) == 3
        assert batches[0].items == [1, 2]
        assert batches[0].done is False
        assert batches[1].items == [3, 4]
        assert batches[1].done is False
        assert batches[2].items == []
        assert batches[2].done is True

    def test_uneven_split_emits_partial_last_batch(self) -> None:
        batches = list(stream_batches([1, 2, 3, 4, 5], batch_size=2))
        # 2 + 2 + 1 (partial, done=True).
        assert len(batches) == 3
        assert batches[-1].items == [5]
        assert batches[-1].done is True

    def test_cumulative_count_accumulates(self) -> None:
        batches = list(stream_batches([1, 2, 3, 4], batch_size=2))
        cumulatives = [b.cumulative for b in batches]
        assert cumulatives == [2, 4, 4]

    def test_batch_index_monotonic(self) -> None:
        batches = list(stream_batches([1, 2, 3], batch_size=1))
        indices = [b.batch_index for b in batches]
        assert indices == list(range(len(batches)))


# ---------------------------------------------------------------------------
# stream_batches — caps + degenerate inputs
# ---------------------------------------------------------------------------


class TestStreamBatchesEdgeCases:
    def test_negative_batch_size_normalised(self) -> None:
        batches = list(stream_batches([1, 2, 3], batch_size=-5))
        # Normalised to 1 → 3 batches of 1 + terminator.
        assert len([b for b in batches if b.items]) == 3

    def test_zero_batch_size_normalised(self) -> None:
        batches = list(stream_batches([1, 2], batch_size=0))
        # Normalised to 1.
        assert len([b for b in batches if b.items]) == 2

    def test_max_items_cap(self) -> None:
        batches = list(stream_batches(
            [1, 2, 3, 4, 5], batch_size=2, max_items=3,
        ))
        # First batch of 2 (cumulative 2), then 1 more would bring
        # us to 3 — that fits within the cap; but the 4th would
        # push to 4 > 3, so we flush + stop.
        cumulatives = [b.cumulative for b in batches]
        assert cumulatives[-1] <= 3
        assert any("max_items" in b.note for b in batches)
        assert batches[-1].done is True

    def test_max_items_zero_returns_immediate_done(self) -> None:
        batches = list(stream_batches([1, 2, 3], max_items=0))
        # The first iteration sees cumulative=0 + len(batch)=0 >= 0
        # → flush empty + stop.
        assert batches[-1].done is True
        assert any("max_items" in b.note for b in batches)


# ---------------------------------------------------------------------------
# early_exit_fn
# ---------------------------------------------------------------------------


class TestEarlyExit:
    def test_early_exit_triggers_terminator(self) -> None:
        batches = list(stream_batches(
            [1, 2, 3, 4, 5, 6], batch_size=2,
            early_exit_fn=lambda emitted, _: len(emitted) >= 4,
        ))
        # 2 + 2 (predicate returns True after 4 emitted) + terminator.
        assert any(b.early_exit for b in batches)
        assert batches[-1].done is True
        assert batches[-1].early_exit is True

    def test_early_exit_never_triggered(self) -> None:
        batches = list(stream_batches(
            [1, 2, 3], batch_size=1,
            early_exit_fn=lambda _emitted, _cum: False,
        ))
        # All 3 items + terminator; no early_exit flag.
        assert all(not b.early_exit for b in batches)

    def test_no_predicate_runs_to_completion(self) -> None:
        batches = list(stream_batches([1, 2, 3], batch_size=1))
        assert all(not b.early_exit for b in batches)


# ---------------------------------------------------------------------------
# Generator input
# ---------------------------------------------------------------------------


class TestGeneratorInput:
    def test_generator_consumed_lazily(self) -> None:
        peek = {"emitted": 0}

        def producer():
            for i in range(5):
                peek["emitted"] = i + 1
                yield i

        # Pull one batch (size 2) and verify only 2 items have been
        # consumed from the producer.
        gen = stream_batches(producer(), batch_size=2)
        first = next(gen)
        assert first.items == [0, 1]
        assert peek["emitted"] == 2

    def test_iterator_exhausted_terminator(self) -> None:
        batches = list(stream_batches(iter([1, 2, 3]), batch_size=10))
        assert batches[-1].done is True


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import streaming as mod
        assert set(mod.__all__) == {
            "StreamBatch",
            "stream_batches",
        }
