"""Parallel fan-out with early-exit (WIN-237, EPIC 1.10 #28).

Hybrid retrieval runs multiple streams in parallel (dense, sparse,
symbol, graph-vec). Today the streams execute *sequentially* in
``HybridRrfRetrieval._search_single_query`` — the slowest stream
dictates the total latency. When the agent only needs the top-K and
the fast streams already agree on the top-K, paying for the slow
stream buys very little.

WIN-237 adds the primitive that makes a different choice possible:
``parallel_fanout`` runs N named tasks concurrently on a thread
pool, fuses results as each task completes, and cancels the still-
running tasks the moment the fused top-K is "stable enough" (or a
caller-supplied predicate returns True). When no early-exit fires,
behaviour reduces to "wait for everything" — i.e. the same answer
the sequential code produced, just typically faster.

This module is a **standalone primitive**, matching the WIN-230 /
WIN-236 pattern. The consuming stories (hybrid retrieval, multi-
hop fan-out, query decomposition fan-out) opt in when they're
ready — no automatic wiring into ``HybridRrfRetrieval`` yet.

Stability predicate
-------------------

``topk_stable(previous, current, top_k, min_overlap)`` returns True
when the top-K of the new fusion overlaps with the top-K of the
previous fusion by at least ``min_overlap`` (fraction in ``[0, 1]``).
"Overlap" is the Jaccard-style intersection size divided by ``top_k``
(so adding a stream that introduces one or two minor reorderings in
positions 4..10 doesn't kill stability). The first call returns
``False`` by definition — we need at least two fusions to compare.

Design choices
--------------

- **No raises across the boundary.** A task that raises is recorded
  as ``failed`` with the exception captured on the result. The
  fan-out completes and fuses the surviving tasks. Same defensive
  shape as the WIN-235 escalating router.
- **Thread pool, not async.** The existing retrieval stack is sync
  (``Storage`` / ``Embedder`` calls block on I/O / native code).
  ``concurrent.futures.ThreadPoolExecutor`` keeps the call site sync
  and avoids leaking an event loop into the MCP server.
- **Cancellation is best-effort.** Python ``Future.cancel()`` only
  works for tasks that haven't started yet. Running tasks finish; we
  just discard their results. This is correct for short retrievals
  (<200 ms each) — for longer tasks the caller should pass a
  ``timeout`` so the pool tears down deterministically.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class TaskOutcome:
    """Per-task outcome captured during fan-out.

    ``error`` is set when the task raised; ``result`` is then ``None``.
    ``elapsed_ms`` is the wall-clock time the task spent inside the
    pool.
    """

    name: str
    result: object | None = None
    error: BaseException | None = None
    elapsed_ms: float = 0.0
    cancelled: bool = False


@dataclass(frozen=True)
class FanoutResult:
    """Aggregate outcome of a :func:`parallel_fanout` invocation.

    ``fused`` is whatever ``fuse_fn`` returned at the moment fan-out
    ended (either early-exit fired or every task settled).
    ``completed`` / ``cancelled`` / ``failed`` are parallel ``name``
    lists. ``elapsed_ms`` is total wall-clock from submit to return.
    ``early_exit`` is True when the loop terminated before every task
    settled.
    """

    fused: object | None = None
    outcomes: dict[str, TaskOutcome] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    early_exit: bool = False
    notes: list[str] = field(default_factory=list)


def _ids(items: Iterable[object]) -> list[str]:
    """Pull stable identity from ranked items.

    Accepts ``SearchHit``-like objects with a ``chunk_id`` attribute,
    plain strings, or dicts with a ``chunk_id`` key. Anything else
    falls back to ``repr(item)`` so the stability check still works
    against arbitrary scalars in unit tests.
    """
    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
            continue
        cid = getattr(item, "chunk_id", None)
        if cid is None and isinstance(item, dict):
            cid = item.get("chunk_id")
        if cid is None:
            cid = repr(item)
        ids.append(str(cid))
    return ids


def topk_stable(
    previous: list[object] | None,
    current: list[object],
    *,
    top_k: int,
    min_overlap: float = 0.8,
) -> bool:
    """Return True when the fused top-K is stable across two iterations.

    The first iteration (when ``previous`` is ``None`` or empty)
    always returns False — we need a prior snapshot to compare.

    ``min_overlap`` is the fraction in ``[0, 1]`` of top-K
    ``chunk_id``s that must appear in both the previous and current
    fusions. ``min_overlap=1.0`` requires an exact match;
    ``min_overlap=0.0`` makes the check always pass once we have a
    prior snapshot (rarely useful).
    """
    if top_k <= 0:
        return False
    if not previous:
        return False
    prev_top = set(_ids(previous)[:top_k])
    curr_top = set(_ids(current)[:top_k])
    if not prev_top or not curr_top:
        return False
    intersection = len(prev_top & curr_top)
    overlap = intersection / float(top_k)
    return overlap >= min_overlap


def parallel_fanout(
    tasks: dict[str, Callable[[], object]],
    fuse_fn: Callable[[dict[str, object]], object],
    *,
    early_exit_fn: Callable[[object | None, object], bool] | None = None,
    min_settled: int = 1,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> FanoutResult:
    """Run ``tasks`` concurrently; abandon stragglers on stable fusion.

    Args:
        tasks: dict mapping task name → zero-argument callable. The
            callable's return value is opaque; ``fuse_fn`` is what
            interprets it.
        fuse_fn: called with the *cumulative* dict of completed task
            results after every task settles. Returns the fused
            artefact (typically a ranked list of ``SearchHit``).
        early_exit_fn: called as
            ``early_exit_fn(previous_fused, current_fused)`` after
            every fuse. Return True to cancel pending tasks and
            return immediately. ``None`` disables early-exit; the
            loop only ends when every task settles or ``timeout``
            elapses.
        min_settled: don't even consult ``early_exit_fn`` until at
            least this many tasks have settled. Defaults to 1
            (consult after every settle). Set to 2+ when the
            stability check needs at least two streams to be
            meaningful.
        max_workers: thread pool size. ``None`` → ``len(tasks)`` so
            every task runs concurrently.
        timeout: wall-clock cap in seconds. ``None`` → no cap.

    Returns:
        :class:`FanoutResult` whose ``early_exit`` flag tells the
        caller why the loop ended.
    """
    started_at = time.monotonic()
    if not tasks:
        return FanoutResult(
            fused=fuse_fn({}),
            elapsed_ms=0.0,
            notes=["empty tasks — nothing to fan out"],
        )

    pool_size = max_workers if max_workers and max_workers > 0 else len(tasks)
    results: dict[str, object] = {}
    outcomes: dict[str, TaskOutcome] = {}
    completed: list[str] = []
    failed: list[str] = []
    cancelled: list[str] = []
    notes: list[str] = []
    fused: object | None = None
    previous_fused: object | None = None
    early_exit = False

    executor = ThreadPoolExecutor(max_workers=pool_size)
    submit_times: dict[Future[object], float] = {}
    future_to_name: dict[Future[object], str] = {}
    try:
        for name, fn in tasks.items():
            future = executor.submit(_run_task, fn)
            future_to_name[future] = name
            submit_times[future] = time.monotonic()

        pending = set(future_to_name.keys())
        deadline = (started_at + timeout) if timeout else None

        while pending:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0:
                    notes.append(
                        f"timeout={timeout!r}s elapsed — cancelling pending"
                    )
                    break
            done, pending = wait(
                pending,
                timeout=remaining,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                # ``wait`` returned because the timeout fired.
                notes.append(
                    f"timeout={timeout!r}s elapsed — cancelling pending"
                )
                break
            for future in done:
                name = future_to_name[future]
                elapsed = (time.monotonic() - submit_times[future]) * 1000.0
                try:
                    value = future.result()
                except BaseException as exc:  # pragma: no cover - re-raised below
                    outcomes[name] = TaskOutcome(
                        name=name, error=exc, elapsed_ms=elapsed,
                    )
                    failed.append(name)
                    continue
                outcomes[name] = TaskOutcome(
                    name=name, result=value, elapsed_ms=elapsed,
                )
                results[name] = value
                completed.append(name)

            fused = fuse_fn(dict(results))
            if (
                early_exit_fn is not None
                and len(results) >= min_settled
                and pending
                and early_exit_fn(previous_fused, fused)
            ):
                early_exit = True
                break
            previous_fused = fused

        # Cancel anything still pending (timeout / early-exit branches).
        for future in pending:
            name = future_to_name[future]
            if future.cancel():
                cancelled.append(name)
                outcomes[name] = TaskOutcome(name=name, cancelled=True)
            else:
                # Running future — best effort, the result is discarded.
                cancelled.append(name)
                outcomes[name] = TaskOutcome(name=name, cancelled=True)
    finally:
        # Don't wait for cancelled futures — let the pool reap them.
        executor.shutdown(wait=False, cancel_futures=True)

    if fused is None:
        # No task ever settled (timed out before any completed).
        fused = fuse_fn(dict(results))

    elapsed_ms = (time.monotonic() - started_at) * 1000.0
    return FanoutResult(
        fused=fused,
        outcomes=outcomes,
        completed=completed,
        cancelled=cancelled,
        failed=failed,
        elapsed_ms=elapsed_ms,
        early_exit=early_exit,
        notes=notes,
    )


def _run_task(fn: Callable[[], object]) -> object:
    """Thin wrapper so the executor's ``submit`` payload is uniform."""
    return fn()


__all__ = [
    "FanoutResult",
    "TaskOutcome",
    "parallel_fanout",
    "topk_stable",
]
