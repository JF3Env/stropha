"""Streaming tool results (WIN-256, EPIC 1.12 #93).

Today stropha's tools return when the whole pipeline is done —
the agent waits for fuse + rerank + recursive-merge to finish
before it sees any hits. For long-running queries this wastes
agent latency on the wait. MCP supports streaming responses,
and the agent host can start processing earlier when the server
streams partial results as they arrive.

WIN-256 ships the primitive that exposes a ranked iterable as a
batched generator. Each batch carries:

- ``items`` — the slice of hits in this batch.
- ``cumulative`` — running total of items emitted so far.
- ``done`` — True on the final batch.

Callers compose this with their own emit path (FastMCP currently
returns full responses; a future protocol upgrade exposes
``stream_progress`` / chunked responses). The primitive is
intentionally minimal so it composes with any emitter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StreamBatch:
    """One emitted batch from a streaming generator."""

    items: list[Any] = field(default_factory=list)
    batch_index: int = 0
    cumulative: int = 0
    done: bool = False
    early_exit: bool = False
    note: str = ""


def stream_batches(
    items: Iterable[T],
    *,
    batch_size: int = 5,
    early_exit_fn: Callable[[list[T], int], bool] | None = None,
    max_items: int | None = None,
) -> Iterator[StreamBatch]:
    """Yield ``items`` in batches of ``batch_size``.

    Args:
        items: any iterable; consumed lazily so a generator caller
            doesn't have to materialise the full list up-front.
        batch_size: items per emitted batch. Values ``<= 0`` are
            normalised to 1 to avoid an infinite-empty-batch loop.
        early_exit_fn: optional predicate called as
            ``early_exit_fn(emitted_so_far, cumulative_count)``
            after each batch. Returning True ends the stream with
            ``early_exit=True`` on the final batch.
        max_items: optional hard cap on emitted item count. When
            reached, the stream ends with ``done=True``.

    Yields:
        :class:`StreamBatch` records. The last batch carries
        ``done=True``; one extra empty-batch sentinel with
        ``done=True`` is yielded when the iterable is empty so the
        consumer always sees a terminator.
    """
    bs = max(1, int(batch_size))
    cap = max_items
    batch: list[T] = []
    emitted: list[T] = []
    cumulative = 0
    batch_index = 0

    def _flush(*, done: bool, early_exit: bool = False, note: str = ""):
        nonlocal cumulative, batch_index, batch
        cumulative += len(batch)
        result = StreamBatch(
            items=list(batch),
            batch_index=batch_index,
            cumulative=cumulative,
            done=done,
            early_exit=early_exit,
            note=note,
        )
        batch_index += 1
        batch = []
        return result

    for item in items:
        if cap is not None and cumulative + len(batch) >= cap:
            # The current batch + the item we're about to add would
            # exceed the cap — flush what we have, mark done.
            yield _flush(done=True, note="max_items reached")
            return
        batch.append(item)
        emitted.append(item)
        if len(batch) >= bs:
            yield _flush(done=False)
            if early_exit_fn is not None and early_exit_fn(
                list(emitted), cumulative,
            ):
                # Empty terminator batch so the consumer's
                # done-check fires.
                yield StreamBatch(
                    items=[], batch_index=batch_index,
                    cumulative=cumulative, done=True,
                    early_exit=True,
                    note="early_exit_fn returned True",
                )
                return

    # Final flush — emit whatever's left even if it's smaller than
    # batch_size, then a terminator if the iterable was empty.
    if batch:
        yield _flush(done=True)
    elif batch_index == 0:
        # Iterable was empty — emit a sentinel so consumers always
        # see a done=True batch.
        yield StreamBatch(
            items=[], batch_index=0, cumulative=0, done=True,
            note="empty iterable",
        )
    else:
        # We've already emitted at least one batch but no leftover —
        # yield a terminator with done=True for symmetry.
        yield StreamBatch(
            items=[], batch_index=batch_index, cumulative=cumulative,
            done=True,
        )


__all__ = [
    "StreamBatch",
    "stream_batches",
]
