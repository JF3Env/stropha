"""Tests for parallel enrichment in Pipeline._enrich_chunks.

The parallel path is exercised when ``Pipeline(enricher_concurrency=N)``
with ``N > 1`` and the file has 2+ chunks pending enrichment. The serial
fallback covers concurrency=1 and the single-chunk-pending edge case.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from stropha.models import Chunk
from stropha.pipeline.base import StageContext, StageHealth
from stropha.pipeline.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SleepyEnricher:
    """Enricher that sleeps to simulate network latency.

    Tracks call count and thread id of each call so tests can verify
    that the parallel path actually overlaps execution.
    """

    adapter_id = "sleepy:v1"
    stage_name = "enricher"
    adapter_name = "sleepy"

    def __init__(self, latency_s: float = 0.1) -> None:
        self._latency_s = latency_s
        self.calls: list[tuple[str, int]] = []  # (chunk_id, thread_id)

    @property
    def config_schema(self):  # pragma: no cover  not exercised
        from pydantic import BaseModel
        class _C(BaseModel):
            pass
        return _C

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="sleepy ok")

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        import threading
        time.sleep(self._latency_s)
        self.calls.append((chunk.chunk_id, threading.get_ident()))
        return f"enriched({chunk.chunk_id})"


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path="a.py",
        start_line=1,
        end_line=2,
        content=f"code-{chunk_id}",
        content_hash=f"hash-{chunk_id}",
        language="python",
    )


def _make_pipeline_shell(concurrency: int, latency_s: float = 0.1) -> tuple[Pipeline, _SleepyEnricher]:
    """Build a Pipeline instance bypassing __init__ to avoid storage/embedder setup.

    Only ``_enricher`` and ``_enricher_concurrency`` are needed for
    ``_enrich_chunks`` tests — the rest of the Pipeline is irrelevant.
    """
    enricher = _SleepyEnricher(latency_s=latency_s)
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = enricher
    pipe._enricher_concurrency = concurrency
    return pipe, enricher


# ---------------------------------------------------------------------------
# Parallel path: speedup + order preservation
# ---------------------------------------------------------------------------


def test_parallel_enrichment_speedup_is_real():
    """8 chunks × 100ms with concurrency=8 must complete in ~100ms, not 800ms."""
    pipe, enricher = _make_pipeline_shell(concurrency=8, latency_s=0.1)
    chunks = [_make_chunk(f"c{i}") for i in range(8)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    t0 = time.time()
    results = pipe._enrich_chunks(chunks, contexts)
    elapsed = time.time() - t0

    # Speedup tolerance: should be ~100ms not 800ms. We accept up to 300ms
    # to absorb thread spin-up overhead on slow CI runners.
    assert elapsed < 0.3, f"Parallel enrichment took {elapsed:.2f}s — too slow (expected <0.3s)"
    assert len(results) == 8


def test_parallel_enrichment_preserves_order():
    """Even though workers complete out of order, results MUST come back in input order."""
    pipe, enricher = _make_pipeline_shell(concurrency=8, latency_s=0.05)
    # Reverse-ordered chunk IDs to make any unintended reordering obvious.
    chunks = [_make_chunk(f"c{i}") for i in [7, 3, 5, 1, 0, 6, 2, 4]]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    # _enrich_chunks now returns list[tuple[str, bool]] — legacy enrichers
    # without enrich_with_status default to success=True.
    expected = [(f"enriched({c.chunk_id})", True) for c in chunks]
    assert results == expected


def test_parallel_uses_multiple_threads():
    """Verify the workers actually run on distinct OS threads."""
    pipe, enricher = _make_pipeline_shell(concurrency=4, latency_s=0.1)
    chunks = [_make_chunk(f"c{i}") for i in range(4)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    pipe._enrich_chunks(chunks, contexts)

    thread_ids = {tid for _, tid in enricher.calls}
    assert len(thread_ids) >= 2, (
        f"Expected calls to spread across threads, got {len(thread_ids)} unique tid(s)"
    )


# ---------------------------------------------------------------------------
# Serial fallback
# ---------------------------------------------------------------------------


def test_concurrency_one_uses_serial_path():
    """concurrency=1 means no thread pool — single-thread execution."""
    pipe, enricher = _make_pipeline_shell(concurrency=1, latency_s=0.05)
    chunks = [_make_chunk(f"c{i}") for i in range(4)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    pipe._enrich_chunks(chunks, contexts)

    thread_ids = {tid for _, tid in enricher.calls}
    assert len(thread_ids) == 1, f"Serial path must use one thread, saw {thread_ids}"

    # Serial timing: 4 × 50ms ≈ 200ms+, never under 150ms.
    pipe2, enricher2 = _make_pipeline_shell(concurrency=1, latency_s=0.05)
    t0 = time.time()
    pipe2._enrich_chunks(chunks, contexts)
    elapsed = time.time() - t0
    assert elapsed >= 0.15, (
        f"Serial 4×50ms should take ≥150ms, got {elapsed:.3f}s — accidentally parallelized?"
    )


def test_single_chunk_skips_thread_pool():
    """1 pending chunk → don't pay the thread pool spin-up cost."""
    pipe, enricher = _make_pipeline_shell(concurrency=8, latency_s=0.05)
    chunks = [_make_chunk("only")]
    contexts = [StageContext(repo_key="test")]

    results = pipe._enrich_chunks(chunks, contexts)

    assert results == [("enriched(only)", True)]
    # The one call should be on the main thread (no pool spawned).
    import threading
    main_tid = threading.get_ident()
    assert enricher.calls[0][1] == main_tid


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list():
    pipe, enricher = _make_pipeline_shell(concurrency=8)
    assert pipe._enrich_chunks([], []) == []
    assert enricher.calls == []


def test_mismatched_lengths_raise_value_error():
    pipe, _ = _make_pipeline_shell(concurrency=8)
    with pytest.raises(ValueError, match="must have the same length"):
        pipe._enrich_chunks(
            [_make_chunk("a"), _make_chunk("b")],
            [StageContext(repo_key="test")],  # only 1 context for 2 chunks
        )


def test_invalid_concurrency_rejected_at_pipeline_init():
    """concurrency < 1 is a config error caught at Pipeline.__init__."""
    from stropha.pipeline.pipeline import Pipeline
    from unittest.mock import MagicMock

    storage = MagicMock()
    embedder = MagicMock(batch_size=64, dim=512, model_name="m")
    enricher = MagicMock()

    with pytest.raises(ValueError, match="enricher_concurrency must be >= 1"):
        Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=enricher,
            repos=[__import__("pathlib").Path("/tmp")],
            enricher_concurrency=0,
        )


def test_enricher_exceptions_propagate():
    """If the enricher raises (not just returns None), the parallel map
    surfaces the exception so the caller can handle/log it. Currently the
    contextual enricher returns None on failure rather than raising, but
    if a custom enricher raises, we should not swallow it silently."""
    pipe, _ = _make_pipeline_shell(concurrency=4, latency_s=0.0)

    class RaisingEnricher(_SleepyEnricher):
        def enrich(self, chunk, ctx):
            raise RuntimeError(f"boom on {chunk.chunk_id}")

    pipe._enricher = RaisingEnricher()
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    with pytest.raises(RuntimeError, match="boom on"):
        pipe._enrich_chunks(chunks, contexts)


# ---------------------------------------------------------------------------
# Fix 2: enrich_with_status — failed enrichments must NOT poison the cache
# ---------------------------------------------------------------------------


class _FailingEnricher:
    """Enricher whose ``enrich_with_status`` reports failure for chunks
    whose id contains ``'fail'``. Models the contextual enricher under
    rate limit: returns raw content + success=False so the pipeline knows
    not to cache the result."""

    adapter_id = "failing:v1"
    stage_name = "enricher"
    adapter_name = "failing"

    @property
    def config_schema(self):  # pragma: no cover
        from pydantic import BaseModel
        class _C(BaseModel):
            pass
        return _C

    def health(self):
        return StageHealth(status="ready", message="ok")

    def enrich(self, chunk, ctx):
        return self.enrich_with_status(chunk, ctx)[0]

    def enrich_with_status(self, chunk, ctx):
        if "fail" in chunk.chunk_id:
            # Simulate LLM call returning None → fallback to raw content.
            return chunk.content, False
        return f"enriched({chunk.chunk_id})", True


def test_enrich_with_status_surfaces_failure_bit():
    """The serial path returns (text, success) tuples — success must reflect
    the enricher's signal, not always True."""
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = _FailingEnricher()
    pipe._enricher_concurrency = 1

    chunks = [_make_chunk("ok-1"), _make_chunk("fail-2"), _make_chunk("ok-3")]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    assert results == [
        ("enriched(ok-1)", True),
        ("code-fail-2", False),  # raw content, success=False
        ("enriched(ok-3)", True),
    ]


def test_enrich_with_status_parallel_path_preserves_status():
    """The ThreadPool path must also propagate the success bit per chunk."""
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = _FailingEnricher()
    pipe._enricher_concurrency = 4

    chunks = [_make_chunk(f"{'fail' if i % 2 else 'ok'}-{i}") for i in range(6)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    statuses = [success for _, success in results]
    assert statuses == [True, False, True, False, True, False]


def test_legacy_enricher_without_enrich_with_status_defaults_to_true():
    """Enrichers that only implement ``enrich`` (noop, hierarchical) MUST
    keep working — the helper defaults success=True for them."""
    pipe, _ = _make_pipeline_shell(concurrency=1, latency_s=0.0)
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    assert results == [("enriched(c1)", True), ("enriched(c2)", True)]


# ---------------------------------------------------------------------------
# Fix 3: batched dispatch
# ---------------------------------------------------------------------------


class _BatchingEnricher:
    """Enricher that implements ``enrich_batch`` and records call sizes
    so tests can verify the pipeline split chunks into the right batches."""

    adapter_id = "batching:v1"
    stage_name = "enricher"
    adapter_name = "batching"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.total_calls = 0

    @property
    def config_schema(self):  # pragma: no cover
        from pydantic import BaseModel
        class _C(BaseModel):
            pass
        return _C

    def health(self):
        return StageHealth(status="ready", message="ok")

    def enrich(self, chunk, ctx):
        return f"single({chunk.chunk_id})"

    def enrich_with_status(self, chunk, ctx):
        return f"single({chunk.chunk_id})", True

    def enrich_batch(self, chunks, contexts):
        self.batch_sizes.append(len(chunks))
        self.total_calls += 1
        return [(f"batched({c.chunk_id})", True) for c in chunks]


def test_pipeline_uses_enrich_batch_when_batch_size_gt_one():
    """batch_size=10 + 10 chunks → 1 call to enrich_batch with all 10."""
    enricher = _BatchingEnricher()
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = enricher
    pipe._enricher_concurrency = 1
    pipe._enricher_batch_size = 10

    chunks = [_make_chunk(f"c{i}") for i in range(10)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    assert enricher.total_calls == 1
    assert enricher.batch_sizes == [10]
    assert all(text.startswith("batched(") for text, _ in results)
    assert len(results) == 10


def test_pipeline_splits_into_batches_of_size_n():
    """25 chunks with batch_size=10 → 3 calls (10 + 10 + 5)."""
    enricher = _BatchingEnricher()
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = enricher
    pipe._enricher_concurrency = 1
    pipe._enricher_batch_size = 10

    chunks = [_make_chunk(f"c{i}") for i in range(25)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    assert enricher.batch_sizes == [10, 10, 5]
    assert len(results) == 25
    # Order preservation across batches.
    assert results[0][0] == "batched(c0)"
    assert results[12][0] == "batched(c12)"
    assert results[24][0] == "batched(c24)"


def test_pipeline_batch_path_respects_concurrency():
    """3 batches × concurrency=2 → at least 2 batches run in parallel."""
    import threading

    enricher = _BatchingEnricher()

    # Patch enrich_batch to record thread ids so we can verify parallelism.
    seen_threads: set[int] = set()
    lock = threading.Lock()

    def slow_enrich_batch(chunks, contexts):
        import time as _t
        _t.sleep(0.05)
        with lock:
            seen_threads.add(threading.get_ident())
        return [(f"batched({c.chunk_id})", True) for c in chunks]

    enricher.enrich_batch = slow_enrich_batch  # type: ignore[method-assign]

    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = enricher
    pipe._enricher_concurrency = 2
    pipe._enricher_batch_size = 5

    chunks = [_make_chunk(f"c{i}") for i in range(15)]  # 3 batches of 5
    contexts = [StageContext(repo_key="test") for _ in chunks]

    pipe._enrich_chunks(chunks, contexts)

    assert len(seen_threads) >= 2


def test_pipeline_batch_size_one_uses_per_chunk_path():
    """batch_size=1 must NOT call enrich_batch — legacy per-chunk path."""
    enricher = _BatchingEnricher()
    pipe = Pipeline.__new__(Pipeline)
    pipe._enricher = enricher
    pipe._enricher_concurrency = 1
    pipe._enricher_batch_size = 1

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    assert enricher.total_calls == 0  # enrich_batch never called
    assert all(text.startswith("single(") for text, _ in results)


def test_pipeline_adapter_without_enrich_batch_falls_back():
    """Even with batch_size=10, an enricher without enrich_batch goes per-chunk."""
    pipe, sleepy = _make_pipeline_shell(concurrency=1, latency_s=0.0)
    pipe._enricher_batch_size = 10  # configured but adapter doesn't support

    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    contexts = [StageContext(repo_key="test") for _ in chunks]

    results = pipe._enrich_chunks(chunks, contexts)

    # Falls back to the per-chunk dispatch — one call per chunk.
    assert len(sleepy.calls) == 5
    assert results == [(f"enriched(c{i})", True) for i in range(5)]


def test_pipeline_init_rejects_invalid_batch_size():
    """batch_size < 1 is a config error caught at Pipeline.__init__."""
    from unittest.mock import MagicMock

    storage = MagicMock()
    embedder = MagicMock(batch_size=64, dim=512, model_name="m")
    enricher = MagicMock()

    with pytest.raises(ValueError, match="enricher_batch_size must be >= 1"):
        Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=enricher,
            repos=[__import__("pathlib").Path("/tmp")],
            enricher_batch_size=0,
        )
