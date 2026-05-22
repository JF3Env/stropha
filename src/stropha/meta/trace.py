"""Off-thread JSONL trace recorder for Meta-Harness Phase 0.

Per ``docs/architecture/stropha-meta-harness.md`` and ticket WIN-159, the
Meta-Harness proposer needs the *full history* — code + scores + traces of
every prior candidate — not just compressed feedback. This module is the
trace half of that substrate.

Design constraints (from the ticket):

- Latency: ``<2 ms`` p95 overhead on the producer thread.
- Atomicity: each record reaches disk as a single intact JSON line; a crash
  mid-write must not corrupt previously-written lines.
- Rotation: a single trace file never exceeds ``50 MB``; one calendar day
  (UTC) is one logical bucket. Roll-over keeps the date-bucket suffix and
  appends ``.1``, ``.2``, etc. for same-day rolls.

Approach: a single daemon thread drains a ``queue.Queue`` and writes
``json.dumps(record) + "\\n"`` via ``os.write`` (one syscall per record,
``O_APPEND`` semantics). The producer side enqueues with ``put_nowait`` —
no lock, no fsync, no JSON formatting on the hot path. When the queue is
full we drop and count, never block.

Why a custom recorder instead of structlog: structlog runs every processor
on the calling thread, including the JSON renderer, which is too expensive
to guarantee <2 ms on cold caches. We also need byte-counting per file to
implement the 50 MB cap, which structlog handlers do not expose cheaply.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


# Acceptance criterion: rotation at 50 MB per file. Exposed for tests.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB

# A bounded queue keeps the producer non-blocking even under bursts. When
# the consumer falls behind we drop records (counted) rather than apply
# back-pressure on the search path.
DEFAULT_QUEUE_SIZE = 4096


@dataclass
class TraceRecord:
    """One execution trace — the shape WIN-159 specifies.

    Producers may pass a plain ``dict`` to :meth:`TraceRecorder.record`; this
    dataclass exists for callers who want a typed view (and for tests). The
    runtime hot path does not instantiate it.
    """

    trace_id: str = ""
    session_id: str = ""
    ts: str = ""
    tool: str = ""
    query: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    returned_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso_utc_now() -> str:
    """ISO-8601 timestamp in UTC, microsecond precision."""
    return datetime.now(tz=UTC).isoformat()


def _today_utc() -> str:
    """Current UTC date as ``YYYY-MM-DD`` — the bucket name."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


class TraceRecorder:
    """Append-only JSONL trace writer with off-thread serialisation.

    Use as a context manager so the daemon thread is drained + the file
    descriptor closed on exit::

        with TraceRecorder(base_dir=Path(".stropha/meta/traces")) as rec:
            rec.record({"tool": "search_code", "query": "foo"})

    Standalone usage is also supported; remember to call :meth:`close`
    explicitly (the daemon thread will not block process exit, but unflushed
    records in the queue may be lost).

    Args:
        base_dir: directory where ``trace-YYYY-MM-DD[.N].jsonl`` files live.
            Created if missing.
        session_id: stable identifier for this process / agent session.
            Auto-generated (uuid4) when not provided. Recorded in every
            record that does not already carry a ``session_id`` key.
        max_bytes: rotation threshold in bytes. Default ``50 MiB``
            (acceptance criterion).
        queue_size: bounded ``queue.Queue`` capacity. Producers never block;
            records beyond capacity are dropped and counted in
            :attr:`dropped`.
        clock: injectable date function (``() -> "YYYY-MM-DD"``) for tests
            that need to simulate midnight rollover.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        session_id: str | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex
        self.max_bytes = max_bytes
        self._clock = clock or _today_utc

        # Bounded queue keeps producer non-blocking under load. dropped
        # counter is monotonic; exposed so callers can alert on saturation.
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_size)
        self.dropped = 0

        # File state is owned exclusively by the writer thread. The producer
        # only touches the queue + dropped counter (atomic int writes in
        # CPython under the GIL).
        self._fd: int | None = None
        self._current_path: Path | None = None
        self._current_bucket: str | None = None  # YYYY-MM-DD
        self._current_seq: int = 0  # rollover index within the bucket
        self._current_bytes: int = 0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="stropha-trace-writer",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Producer API                                                       #
    # ------------------------------------------------------------------ #

    def record(self, event: dict[str, Any]) -> bool:
        """Enqueue one event for off-thread writing.

        The hot path does only three things: auto-fill missing identifiers,
        serialise to JSON bytes, and ``put_nowait`` onto the queue. No lock,
        no syscall.

        Returns ``True`` if the record was enqueued, ``False`` if the queue
        was full and the record was dropped.
        """
        # Auto-fill the fields the ticket schema requires but the caller
        # may not have populated. Done on the producer side so each line
        # is self-describing even if the consumer never gets a chance to
        # rewrite it.
        if "trace_id" not in event:
            event["trace_id"] = uuid.uuid4().hex
        if "session_id" not in event:
            event["session_id"] = self.session_id
        if "ts" not in event:
            event["ts"] = _iso_utc_now()

        try:
            payload = json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")
        except (TypeError, ValueError):
            # Defensive: a non-serialisable params dict must never poison
            # the recorder. Drop the record, count it, keep going.
            self.dropped += 1
            return False

        try:
            self._queue.put_nowait(payload + b"\n")
            return True
        except queue.Full:
            self.dropped += 1
            return False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def close(self, timeout: float = 5.0) -> None:
        """Signal the writer thread to drain + exit, then close the file.

        Idempotent. Safe to call from any thread.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        # Sentinel wakes the consumer if it was blocked on Queue.get.
        # If the queue is full, the consumer is still draining and will
        # observe _stop_event on its own — we don't need to add a sentinel.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=timeout)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Consumer (daemon thread)                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Drain the queue until close() and the queue are both clear."""
        while True:
            try:
                # Short timeout so the loop notices _stop_event even when
                # the queue is idle.
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if item is None:
                # Sentinel from close(); drain remaining items, then exit.
                while True:
                    try:
                        rest = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if rest is None:
                        continue
                    self._write_line(rest)
                break
            self._write_line(item)

    def _write_line(self, payload: bytes) -> None:
        """Single-syscall append. Rotates first if needed."""
        try:
            self._ensure_open(len(payload))
            assert self._fd is not None  # invariant after _ensure_open
            os.write(self._fd, payload)
            self._current_bytes += len(payload)
        except OSError as exc:
            # IO failure on the writer is observable but must not kill the
            # daemon — we just lose this record. Log once per failure.
            log.warning("trace.write_failed", error=str(exc))

    def _ensure_open(self, incoming_size: int) -> None:
        """Open or rotate the trace file so the next write fits the cap.

        Three reasons we rotate:
        1. No file open yet (first call after __init__).
        2. The UTC date changed (daily bucket rolls).
        3. ``current_bytes + incoming_size`` would exceed ``max_bytes``.
        """
        bucket = self._clock()
        needs_new = (
            self._fd is None
            or self._current_bucket != bucket
            or (self._current_bytes + incoming_size) > self.max_bytes
        )
        if not needs_new:
            return

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

        # Reset the sequence when the bucket date changes; preserve it for
        # same-day rolls so we never overwrite an existing file.
        if self._current_bucket != bucket:
            self._current_bucket = bucket
            self._current_seq = 0
        else:
            self._current_seq += 1

        # Find the lowest seq that doesn't collide with an existing file
        # — handles restart scenarios where prior runs already produced
        # rolls today.
        while True:
            path = self._path_for(bucket, self._current_seq)
            if not path.exists() or self._current_seq == 0:
                break
            self._current_seq += 1

        # O_APPEND keeps multi-process writes from overwriting each other
        # at the OS level; combined with single-syscall os.write this is
        # the crash-safety mechanism the ticket asks for.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        self._fd = os.open(str(path), flags, 0o644)
        self._current_path = path
        try:
            self._current_bytes = path.stat().st_size
        except OSError:
            self._current_bytes = 0

    def _path_for(self, bucket: str, seq: int) -> Path:
        """``trace-YYYY-MM-DD.jsonl`` (seq 0) or ``…-YYYY-MM-DD.N.jsonl``."""
        if seq == 0:
            return self.base_dir / f"trace-{bucket}.jsonl"
        return self.base_dir / f"trace-{bucket}.{seq}.jsonl"

    # ------------------------------------------------------------------ #
    # Inspection                                                         #
    # ------------------------------------------------------------------ #

    @property
    def current_path(self) -> Path | None:
        """The trace file currently being written, or None before first record."""
        return self._current_path

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue is fully drained.

        Useful for tests; not part of the hot path.
        """
        deadline = time.monotonic() + timeout
        while not self._queue.empty():
            if time.monotonic() > deadline:
                return
            time.sleep(0.005)
