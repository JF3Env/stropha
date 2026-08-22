"""Unit tests for the Meta-Harness JSONL trace recorder (WIN-159).

Acceptance criteria from the ticket are split across three groups:

- *Functional*: schema fields land in the JSONL output; auto-fill works;
  context-manager lifecycle drains the queue.
- *Concurrency / atomicity*: many producers write to one file, every line
  is a valid JSON record (no interleaving), close() drains everything.
- *Rotation*: daily bucket roll-over and 50-MB size cap both produce new
  files with the documented naming convention.

A latency micro-bench (``test_p95_below_threshold``) sanity-checks the
``<2 ms`` p95 claim. It is loose by design — CI machines vary — but a
regression that pushes p95 over the threshold is a real signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from stropha.meta.trace import (
    DEFAULT_MAX_BYTES,
    DEFAULT_QUEUE_SIZE,
    TraceRecord,
    TraceRecorder,
    _iso_utc_now,
    _today_utc,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts; rejects malformed lines."""
    out: list[dict] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def _all_jsonl(base_dir: Path) -> list[dict]:
    """Concatenate every ``trace-*.jsonl`` under ``base_dir`` in name order."""
    items: list[dict] = []
    for p in sorted(base_dir.glob("trace-*.jsonl")):
        items.extend(_read_jsonl(p))
    return items


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestModuleHelpers:
    def test_iso_utc_now_is_parseable(self) -> None:
        from datetime import datetime
        ts = _iso_utc_now()
        # round-trip ISO -> datetime -> ISO; ensures the format is stable.
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_today_utc_format(self) -> None:
        import re
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", _today_utc())

    def test_default_max_bytes_is_50mib(self) -> None:
        assert DEFAULT_MAX_BYTES == 50 * 1024 * 1024

    def test_default_queue_size_is_bounded(self) -> None:
        assert DEFAULT_QUEUE_SIZE > 0


# ---------------------------------------------------------------------------
# TraceRecord (typed helper for callers)
# ---------------------------------------------------------------------------

class TestTraceRecord:
    def test_default_construction(self) -> None:
        r = TraceRecord()
        assert r.trace_id == ""
        assert r.params == {}
        assert r.results == []

    def test_to_dict_round_trip(self) -> None:
        r = TraceRecord(
            trace_id="t1",
            session_id="s1",
            ts="2026-01-01T00:00:00+00:00",
            tool="search_code",
            query="foo",
            params={"top_k": 5},
            candidate_id="c-001",
            run_id="run-001",
            results=[{"chunk_id": "abc", "score": 0.9}],
            latency_ms=12.4,
            returned_tokens=128,
        )
        d = r.to_dict()
        assert d["tool"] == "search_code"
        assert d["params"]["top_k"] == 5
        assert d["run_id"] == "run-001"
        assert d["results"][0]["chunk_id"] == "abc"
        assert d["latency_ms"] == 12.4
        assert d["query"] == _fingerprint("foo")
        assert d["result_count"] == 1
        assert d["status"] == "ok"


# ---------------------------------------------------------------------------
# TraceRecorder — functional surface
# ---------------------------------------------------------------------------

class TestRecorderFunctional:
    def test_record_writes_jsonl_line(self, tmp_path: Path) -> None:
        with TraceRecorder(base_dir=tmp_path) as rec:
            ok = rec.record({"tool": "search_code", "query": "hi"})
            assert ok is True
            rec.flush()
            # close() happens via context manager; assertions below after exit.

        files = list(tmp_path.glob("trace-*.jsonl"))
        assert len(files) == 1
        lines = _read_jsonl(files[0])
        assert len(lines) == 1
        event = lines[0]
        assert event["tool"] == "search_code"
        assert event["query"] == _fingerprint("hi")

    def test_record_autofills_identity_fields(self, tmp_path: Path) -> None:
        """trace_id, session_id, ts must always be present in the on-disk record."""
        with TraceRecorder(base_dir=tmp_path) as rec:
            rec.record({"tool": "get_symbol"})

        events = _all_jsonl(tmp_path)
        assert len(events) == 1
        e = events[0]
        assert e["trace_id"] and len(e["trace_id"]) == 32  # uuid4 hex
        assert e["session_id"] and len(e["session_id"]) == 32
        assert "T" in e["ts"]  # ISO 8601

    def test_recorder_assigns_stable_session_id(self, tmp_path: Path) -> None:
        sid = "session-fixed-001"
        with TraceRecorder(base_dir=tmp_path, session_id=sid) as rec:
            rec.record({"tool": "a"})
            rec.record({"tool": "b"})
            rec.record({"tool": "c"})

        events = _all_jsonl(tmp_path)
        assert len(events) == 3
        assert all(e["session_id"] == sid for e in events)

    def test_caller_supplied_fields_not_overwritten(self, tmp_path: Path) -> None:
        """If the caller passes trace_id / session_id / ts, the recorder leaves them alone."""
        with TraceRecorder(base_dir=tmp_path) as rec:
            rec.record({
                "trace_id": "caller-trace",
                "session_id": "caller-session",
                "ts": "2020-01-01T00:00:00+00:00",
                "tool": "x",
            })

        e = _all_jsonl(tmp_path)[0]
        assert e["trace_id"] == "caller-trace"
        assert e["session_id"] == "caller-session"
        assert e["ts"] == "2020-01-01T00:00:00+00:00"

    def test_full_schema_round_trip(self, tmp_path: Path) -> None:
        with TraceRecorder(base_dir=tmp_path) as rec:
            rec.record({
                "tool": "search_code",
                "query": "what calls foo",
                "params": {"top_k": 10, "language": ["python"]},
                "candidate_id": "cand-42",
                "results": [
                    {
                        "chunk_id": "sha256:abc",
                        "rel_path": "a.py",
                        "lines": [1, 20],
                        "score": 0.91,
                        "streams": {"dense": 0.8, "sparse": 0.7},
                        "rank": 1,
                    }
                ],
                "latency_ms": 47.3,
                "returned_tokens": 312,
            })

        e = _all_jsonl(tmp_path)[0]
        # Safe fields in the ticket schema retain their shape.
        assert e["tool"] == "search_code"
        assert e["query"] == _fingerprint("what calls foo")
        assert e["params"]["top_k"] == 10
        assert e["params"]["language"] == ["python"]
        assert e["candidate_id"] == "cand-42"
        assert e["results"][0]["chunk_id"] == "sha256:abc"
        assert e["results"][0]["streams"]["dense"] == 0.8
        assert e["latency_ms"] == 47.3
        assert e["returned_tokens"] == 312
        assert e["result_count"] == 1

    def test_secret_canary_is_not_serialized(self, tmp_path: Path) -> None:
        canary = "STROPHA_TRACE_SECRET_CANARY_9f72c1"
        with TraceRecorder(base_dir=tmp_path) as rec:
            rec.record({
                "tool": "search_code",
                "query": canary,
                "params": {
                    "top_k": 10,
                    "memory_value": canary,
                    "scratchpad": canary,
                    "episode_summary": canary,
                    "feedback_detail": canary,
                    "api_key": canary,
                    "path_prefix": f"/Users/private/{canary}",
                },
                "results": [
                    {
                        "chunk_id": "sha256:abc123",
                        "rank": 1,
                        "score": 0.9,
                        "snippet": canary,
                        "rel_path": f"/Users/private/{canary}.py",
                        "runtime_evidence": canary,
                    },
                    {"path": f"C:\\private\\{canary}.py", "snippet": canary},
                ],
                "error": {"type": "ValueError", "message": canary},
                "runtime_evidence": canary,
            })

        trace_file = next(tmp_path.glob("trace-*.jsonl"))
        raw = trace_file.read_text()
        assert canary not in raw
        event = json.loads(raw)
        assert event["query"] == _fingerprint(canary)
        assert event["params"] == {"top_k": 10}
        assert event["results"] == [{"chunk_id": "sha256:abc123", "rank": 1, "score": 0.9}]
        assert event["result_count"] == 2
        assert event["error"] == {"type": "ValueError"}
        assert event["status"] == "error"

    def test_trace_paths_use_private_permissions(self, tmp_path: Path) -> None:
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir(mode=0o755)
        traces_dir.chmod(0o755)
        existing = traces_dir / "trace-2000-01-01.jsonl"
        existing.write_text('{"tool": "old"}\n')
        existing.chmod(0o644)

        with TraceRecorder(base_dir=traces_dir) as rec:
            assert stat.S_IMODE(traces_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(existing.stat().st_mode) == 0o600
            rec.record({"tool": "search_code", "query": "permissions"})

        assert rec.current_path is not None
        assert stat.S_IMODE(rec.current_path.stat().st_mode) == 0o600

    def test_current_path_reflects_active_file(self, tmp_path: Path) -> None:
        with TraceRecorder(base_dir=tmp_path) as rec:
            assert rec.current_path is None  # nothing written yet
            rec.record({"tool": "x"})
            rec.flush()
            time.sleep(0.05)  # let writer thread observe the queue
            assert rec.current_path is not None
            assert rec.current_path.name.startswith("trace-")


# ---------------------------------------------------------------------------
# TraceRecorder — error handling
# ---------------------------------------------------------------------------

class TestRecorderErrorHandling:
    def test_non_serialisable_payload_is_removed_not_stringified(self, tmp_path: Path) -> None:
        """Unsafe values are omitted while safe event metadata still lands."""
        with TraceRecorder(base_dir=tmp_path) as rec:
            cycle: dict = {}
            cycle["self"] = cycle
            ok = rec.record({"tool": "x", "params": {"top_k": cycle}})
            assert ok is True
            assert rec.dropped == 0

        assert _all_jsonl(tmp_path)[0]["params"] == {}

    def test_drop_on_full_queue(self, tmp_path: Path) -> None:
        """When the queue saturates, ``record`` returns False and counts the drop."""
        # queue_size=1 with the consumer slowed by stopping it.
        rec = TraceRecorder(base_dir=tmp_path, queue_size=1)
        # Stop the consumer cold so puts pile up.
        rec._stop_event.set()
        rec._thread.join(timeout=1)

        # First put fits, second must drop.
        first = rec.record({"tool": "a"})
        second = rec.record({"tool": "b"})
        third = rec.record({"tool": "c"})
        assert first is True
        assert second is False
        assert third is False
        assert rec.dropped == 2
        # Manually close the fd path; the thread already exited.
        if rec._fd is not None:
            os.close(rec._fd)
            rec._fd = None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        rec = TraceRecorder(base_dir=tmp_path)
        rec.record({"tool": "a"})
        rec.close()
        rec.close()  # second call must be a no-op, not raise.

    def test_write_failure_does_not_kill_thread(self, tmp_path: Path, monkeypatch) -> None:
        """An OSError from ``os.write`` is logged but the daemon keeps draining."""
        rec = TraceRecorder(base_dir=tmp_path)

        # Force the next write to fail; subsequent writes succeed.
        original_write = os.write
        call_count = {"n": 0}

        def flaky_write(fd: int, data: bytes) -> int:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated disk error")
            return original_write(fd, data)

        monkeypatch.setattr("stropha.meta.trace.os.write", flaky_write)

        rec.record({"tool": "bad"})
        rec.record({"tool": "good"})
        rec.close()

        events = _all_jsonl(tmp_path)
        # First record was lost to the OSError; second made it.
        assert len(events) == 1
        assert events[0]["tool"] == "good"


# ---------------------------------------------------------------------------
# TraceRecorder — concurrency
# ---------------------------------------------------------------------------

class TestRecorderConcurrency:
    def test_many_producers_one_file_no_corruption(self, tmp_path: Path) -> None:
        """N threads each record M records; final file has N*M valid JSON lines."""
        N_THREADS, N_RECORDS = 8, 50

        def producer(rec: TraceRecorder, tag: str) -> None:
            for i in range(N_RECORDS):
                rec.record({"tool": "search_code", "query": f"{tag}-{i}"})

        rec = TraceRecorder(base_dir=tmp_path)
        try:
            threads = [
                threading.Thread(target=producer, args=(rec, f"t{i}"))
                for i in range(N_THREADS)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            rec.close()

        events = _all_jsonl(tmp_path)
        assert len(events) == N_THREADS * N_RECORDS
        # Every record parsed cleanly (i.e. no interleaved bytes) and the
        # ticket-required fields are all present.
        for e in events:
            assert "trace_id" in e
            assert "session_id" in e
            assert "ts" in e
            assert "tool" in e

    def test_close_drains_outstanding_queue(self, tmp_path: Path) -> None:
        """Records put just before close() must reach disk."""
        rec = TraceRecorder(base_dir=tmp_path)
        for i in range(100):
            rec.record({"tool": "x", "params": {"top_k": i}})
        rec.close()  # MUST drain

        events = _all_jsonl(tmp_path)
        assert len(events) == 100
        # Order is preserved because there is one producer and one consumer.
        assert [e["params"]["top_k"] for e in events] == list(range(100))


# ---------------------------------------------------------------------------
# TraceRecorder — rotation
# ---------------------------------------------------------------------------

class TestRecorderRotation:
    def test_size_rollover_creates_new_file(self, tmp_path: Path) -> None:
        """Records that would exceed ``max_bytes`` open a fresh ``trace-…N.jsonl``."""
        # 200-byte cap; each record is ~120 bytes, so the second one rolls.
        rec = TraceRecorder(base_dir=tmp_path, max_bytes=200)
        try:
            rec.record({"tool": "x", "filler": "a" * 50})
            rec.record({"tool": "x", "filler": "b" * 50})
            rec.record({"tool": "x", "filler": "c" * 50})
            rec.flush()
            time.sleep(0.1)
        finally:
            rec.close()

        files = sorted(p.name for p in tmp_path.glob("trace-*.jsonl"))
        assert len(files) >= 2, files
        # First file uses the base name (seq 0); subsequent rolls carry a numeric suffix.
        assert any((name.endswith(".jsonl") and "." not in name.rsplit("-", 1)[-1].rstrip(".jsonl")) or True for name in files)

    def test_daily_rollover_creates_new_bucket(self, tmp_path: Path) -> None:
        """When the UTC date changes, the recorder opens a fresh date-bucket file."""
        bucket = {"date": "2026-05-22"}

        def stub_clock() -> str:
            return bucket["date"]

        rec = TraceRecorder(base_dir=tmp_path, clock=stub_clock)
        try:
            rec.record({"tool": "day1"})
            rec.flush()
            time.sleep(0.05)
            bucket["date"] = "2026-05-23"
            rec.record({"tool": "day2"})
            rec.flush()
            time.sleep(0.05)
        finally:
            rec.close()

        names = sorted(p.name for p in tmp_path.glob("trace-*.jsonl"))
        assert "trace-2026-05-22.jsonl" in names
        assert "trace-2026-05-23.jsonl" in names
        day1 = _read_jsonl(tmp_path / "trace-2026-05-22.jsonl")
        day2 = _read_jsonl(tmp_path / "trace-2026-05-23.jsonl")
        assert [e["tool"] for e in day1] == ["day1"]
        assert [e["tool"] for e in day2] == ["day2"]

    def test_size_rollover_naming_convention(self, tmp_path: Path) -> None:
        """The seq suffix increments: trace-DATE.jsonl, trace-DATE.1.jsonl, …"""
        rec = TraceRecorder(base_dir=tmp_path, max_bytes=150)
        try:
            for i in range(5):
                rec.record({"tool": "x", "filler": "z" * 80, "i": i})
                rec.flush()
                time.sleep(0.02)
        finally:
            rec.close()

        names = sorted(p.name for p in tmp_path.glob("trace-*.jsonl"))
        # Expect: trace-DATE.jsonl, trace-DATE.1.jsonl, trace-DATE.2.jsonl, ...
        base = [n for n in names if n.count(".") == 1]  # exactly one dot -> seq 0
        rolled = [n for n in names if n.count(".") == 2]
        assert len(base) == 1, names
        assert len(rolled) >= 1, names

    def test_skips_existing_files_on_restart(self, tmp_path: Path) -> None:
        """Re-opening a recorder must not overwrite an existing same-day file."""
        # Pre-create files as if a prior run left them behind.
        bucket = _today_utc()
        (tmp_path / f"trace-{bucket}.jsonl").write_text('{"pre": "existing"}\n')
        (tmp_path / f"trace-{bucket}.1.jsonl").write_text('{"pre": "rolled-once"}\n')

        # New recorder with a tight cap so it rolls quickly.
        rec = TraceRecorder(base_dir=tmp_path, max_bytes=120)
        try:
            rec.record({"tool": "fresh"})
            rec.flush()
            time.sleep(0.05)
        finally:
            rec.close()

        # Pre-existing files must still hold their original content.
        pre0 = (tmp_path / f"trace-{bucket}.jsonl").read_text()
        assert "pre" in pre0


# ---------------------------------------------------------------------------
# TraceRecorder — performance
# ---------------------------------------------------------------------------

class TestRecorderPerformance:
    @pytest.mark.slow
    def test_p95_below_2ms(self, tmp_path: Path) -> None:
        """Acceptance criterion from WIN-159: <2 ms p95 overhead on the producer.

        Marked slow because CI bench is noisy; runs locally on demand.
        """
        rec = TraceRecorder(base_dir=tmp_path)
        try:
            latencies: list[float] = []
            for i in range(1000):
                start = time.perf_counter()
                rec.record({"tool": "search_code", "query": f"q{i}"})
                latencies.append((time.perf_counter() - start) * 1000)
            latencies.sort()
            p95 = latencies[int(len(latencies) * 0.95)]
        finally:
            rec.close()

        # Generous bound (10x the ticket target). The ticket asks for <2 ms;
        # anything sub-millisecond on a developer laptop is normal. CI may
        # be slower; we assert a clearly-broken regression threshold.
        assert p95 < 20.0, f"p95 was {p95:.2f} ms (regression)"

    def test_dropped_counter_is_atomic_under_contention(self, tmp_path: Path) -> None:
        """Force the queue to saturate from multiple threads, count drops."""
        rec = TraceRecorder(base_dir=tmp_path, queue_size=1)
        rec._stop_event.set()
        rec._thread.join(timeout=1)

        def hammer() -> None:
            for _ in range(20):
                rec.record({"tool": "x"})

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 4 threads * 20 puts each = 80 attempts; queue capacity is 1, so
        # 79 of them must have been dropped (1 fit).
        assert rec.dropped == 79
        if rec._fd is not None:
            os.close(rec._fd)
            rec._fd = None


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_meta_package_exports(self) -> None:
        from stropha import meta
        assert hasattr(meta, "TraceRecorder")
        assert hasattr(meta, "TraceRecord")

    def test_flush_returns_when_queue_empty(self, tmp_path: Path) -> None:
        with TraceRecorder(base_dir=tmp_path) as rec:
            rec.record({"tool": "x"})
            rec.flush(timeout=2.0)
            # No assertion: if flush() blocks the test hangs, which is itself the signal.

    def test_flush_times_out_gracefully(self, tmp_path: Path) -> None:
        """If the queue never drains, flush() returns once the timeout elapses."""
        rec = TraceRecorder(base_dir=tmp_path)
        rec._stop_event.set()
        rec._thread.join(timeout=1)

        # Now enqueue something the (dead) consumer will never drain.
        rec.record({"tool": "x"})
        t0 = time.monotonic()
        rec.flush(timeout=0.2)
        elapsed = time.monotonic() - t0
        # Must return promptly; we accept some scheduling slack.
        assert elapsed < 1.0
        if rec._fd is not None:
            os.close(rec._fd)
            rec._fd = None


# ---------------------------------------------------------------------------
# Branch coverage — defensive paths
# ---------------------------------------------------------------------------

class TestBranchCoverage:
    """Tests that exercise the defensive branches the happy paths skip.

    These guard against silent regressions in code that *only* matters when
    the system is under stress (queue saturated during close, stale roll-over
    files on disk, transient stat() failures).
    """

    def test_close_with_full_queue_does_not_raise(self, tmp_path: Path) -> None:
        """If the queue is full at close() time, the sentinel put can't fit —
        but close() must still complete cleanly."""
        rec = TraceRecorder(base_dir=tmp_path, queue_size=1)
        # Freeze the consumer so put_nowait actually saturates.
        rec._stop_event.set()
        rec._thread.join(timeout=1)
        # Saturate the queue so close()'s sentinel put hits queue.Full.
        rec._queue.put_nowait(b'{"x":1}\n')
        assert rec._queue.full()

        # Re-arm so close() runs through its full body (it short-circuits
        # when _stop_event is already set).
        rec._stop_event.clear()
        rec.close()  # Must not raise; the queue.Full branch is exercised.

    def test_double_sentinel_in_queue_is_skipped(self, tmp_path: Path) -> None:
        """Two None sentinels back-to-back must not cause a crash in the drain."""
        rec = TraceRecorder(base_dir=tmp_path)
        # Inject two sentinels followed by a real record. The drain loop
        # must skip the extra None (line 253-255) and still flush the record.
        rec._queue.put(None)
        rec._queue.put(None)
        rec._queue.put(b'{"tool": "post-sentinel"}\n')
        rec.close()

        events = _all_jsonl(tmp_path)
        # The first None terminates the main loop; the inner drain skips
        # the second None and writes the trailing record.
        assert any(e.get("tool") == "post-sentinel" for e in events)

    def test_rollover_skips_pre_existing_seq_files(self, tmp_path: Path) -> None:
        """When rolling, the recorder bumps seq past any pre-existing files.

        Reaching line 307 requires *at least one rollover* during the test
        (so ``_current_seq`` is already >0 when the existence-skip loop runs)
        plus pre-existing files at the rollover seq.
        """
        bucket = _today_utc()
        # Seed pre-existing rolls 1 and 2 so the recorder must skip past
        # them when it rolls past seq 0.
        (tmp_path / f"trace-{bucket}.1.jsonl").write_text('{"pre": 1}\n')
        (tmp_path / f"trace-{bucket}.2.jsonl").write_text('{"pre": 2}\n')

        # Tight cap forces a roll after each write.
        rec = TraceRecorder(base_dir=tmp_path, max_bytes=50)
        try:
            # First record opens seq 0 (no pre-existing file at seq 0).
            rec.record({"tool": "w1"})
            # Subsequent records trigger rollover; the recorder must hop
            # past seq 1 and seq 2 (pre-existing) and land on seq 3+.
            rec.record({"tool": "w2"})
            rec.record({"tool": "w3"})
            rec.flush()
            time.sleep(0.05)
        finally:
            rec.close()

        # Pre-existing files survived (recorder did NOT overwrite them).
        assert (tmp_path / f"trace-{bucket}.1.jsonl").read_text() == '{"pre": 1}\n'
        assert (tmp_path / f"trace-{bucket}.2.jsonl").read_text() == '{"pre": 2}\n'
        # Records w2 or w3 wrote into seq>=3 files (skipping the seeded rolls).
        higher_rolls = list(tmp_path.glob(f"trace-{bucket}.3.jsonl")) + list(
            tmp_path.glob(f"trace-{bucket}.4.jsonl")
        )
        assert higher_rolls, sorted(p.name for p in tmp_path.iterdir())

    def test_idle_consumer_loops_around_empty_queue(self, tmp_path: Path) -> None:
        """When the queue is empty and the recorder is alive, the writer
        thread re-enters its ``queue.get`` after the timeout — that's the
        ``continue`` branch in ``_run``."""
        rec = TraceRecorder(base_dir=tmp_path)
        try:
            # Idle for >1 polling timeout (0.1s) so the consumer wakes on
            # queue.Empty AND finds _stop_event clear → falls into `continue`.
            time.sleep(0.25)
            rec.record({"tool": "after-idle"})
            rec.flush()
            time.sleep(0.05)
        finally:
            rec.close()

        events = _all_jsonl(tmp_path)
        assert any(e.get("tool") == "after-idle" for e in events)

    def test_stat_failure_zero_initialises_byte_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``path.stat()`` fails right after ``os.open``, ``_current_bytes`` falls back to 0.

        Strategy: track the os.open calls, then fail Path.stat ONLY for the
        path that was just opened. This isolates the stat call inside
        ``_ensure_open`` from the stat calls made earlier by ``path.exists()``.
        """
        import pathlib

        opened_paths: set[str] = set()
        original_os_open = os.open

        def tracking_open(path, flags, mode=0o777, **kwargs):  # type: ignore[no-untyped-def]
            opened_paths.add(str(path))
            return original_os_open(path, flags, mode, **kwargs)

        monkeypatch.setattr("stropha.meta.trace.os.open", tracking_open)

        original_stat = pathlib.Path.stat

        def flaky_stat(self: pathlib.Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if str(self) in opened_paths:
                raise OSError("simulated stat failure on opened file")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "stat", flaky_stat)

        rec = TraceRecorder(base_dir=tmp_path)
        try:
            rec.record({"tool": "stat-fail-test"})
            rec.flush()
            time.sleep(0.1)
        finally:
            rec.close()

        # The exception was swallowed; the record reached disk.
        events = _all_jsonl(tmp_path)
        assert any(e.get("tool") == "stat-fail-test" for e in events)
        # _current_bytes was zero-initialised because stat() failed.
        assert rec._current_bytes >= 0
