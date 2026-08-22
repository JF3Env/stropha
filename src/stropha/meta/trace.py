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
import hashlib
import json
import math
import os
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
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


# Closed policy: nested fields are retained only when named and validated below.
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+<>-]{0,255}\Z")
_QUERY_FINGERPRINT_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_BOOL_PARAMS = frozenset(
    {
        "compress",
        "conversation_aware",
        "dedup",
        "diff_scope",
        "exclude_tests",
        "expired_only",
        "in_files",
        "in_symbols",
        "include_callers",
        "include_expired",
        "include_facets",
        "include_other_repos",
        "include_rationale",
        "include_tests",
        "recursive",
        "refresh",
        "relevance_decay",
        "strict",
        "use_llm_router",
    }
)
_SAFE_NUMBER_PARAMS = frozenset(
    {
        "budget_tokens",
        "context_budget",
        "depth",
        "limit",
        "max_depth",
        "max_paths",
        "max_recent",
        "max_suggestions",
        "repo_id",
        "top_k",
        "transitive_depth",
    }
)
_SAFE_ENUM_PARAMS: dict[str, frozenset[str]] = {
    "card_level": frozenset({"full", "line", "paragraph"}),
    "kind": frozenset({"class", "function", "interface", "method", "module", "section"}),
    "language": frozenset(
        {
            "c",
            "cpp",
            "csharp",
            "go",
            "java",
            "javascript",
            "kotlin",
            "python",
            "ruby",
            "rust",
            "typescript",
        }
    ),
    "reason": frozenset(
        {
            "duplicate",
            "irrelevant",
            "other",
            "stale",
            "too_verbose",
            "wrong_symbol",
        }
    ),
    "relations": frozenset(
        {
            "calls",
            "contains",
            "extends",
            "implements",
            "imports",
            "rationale_for",
            "references",
            "tests",
        }
    ),
    "runner": frozenset({"go test", "npm test", "pytest"}),
}
_SAFE_ID_PARAMS = frozenset({"chunk_id", "session_id", "trace_id"})
_SAFE_ID_LIST_PARAMS = frozenset({"chunk_ids"})
_SAFE_RESULT_NUMBERS = frozenset(
    {
        "end_line",
        "rank",
        "returned_tokens",
        "score",
        "start_line",
    }
)
_SAFE_STREAMS = frozenset({"dense", "graph_vec", "model2vec", "sparse", "symbol"})


def _safe_identifier(value: Any) -> str | None:
    if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
        return None
    return value


def _safe_number(value: Any, bounds: tuple[float, float]) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or not bounds[0] <= value <= bounds[1]:
        return None
    return value


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 512:
        return None
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith(("/", "~"))
        or re.match(r"[A-Za-z]:", normalized)
        or any(ord(char) < 32 for char in normalized)
    ):
        return None
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        return None
    return path.as_posix()


def _query_fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if _QUERY_FINGERPRINT_RE.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sha256:{digest}"


def _sanitize_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SAFE_BOOL_PARAMS and isinstance(item, bool):
            safe[key] = item
        elif key in _SAFE_NUMBER_PARAMS:
            if item is None:
                safe[key] = None
                continue
            number = _safe_number(item, (0, 10_000_000))
            if number is not None:
                safe[key] = number
        elif key in _SAFE_ENUM_PARAMS:
            allowed = _SAFE_ENUM_PARAMS[key]
            if isinstance(item, str) and item in allowed:
                safe[key] = item
            elif isinstance(item, (list, tuple)):
                selected = [
                    entry for entry in item[:100] if isinstance(entry, str) and entry in allowed
                ]
                if selected:
                    safe[key] = selected
        elif key in _SAFE_ID_PARAMS:
            identifier = _safe_identifier(item)
            if identifier is not None:
                safe[key] = identifier
        elif key in _SAFE_ID_LIST_PARAMS and isinstance(item, (list, tuple)):
            identifiers = [
                identifier
                for entry in item[:100]
                if (identifier := _safe_identifier(entry)) is not None
            ]
            if identifiers:
                safe[key] = identifiers
    return safe


def _sanitize_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    results: list[dict[str, Any]] = []
    for item in value[:1000]:
        if not isinstance(item, dict):
            continue
        safe: dict[str, Any] = {}
        chunk_id = _safe_identifier(item.get("chunk_id"))
        if chunk_id is not None:
            safe["chunk_id"] = chunk_id
        symbol = _safe_identifier(item.get("symbol"))
        if symbol is not None:
            safe["symbol"] = symbol
        for key in ("rel_path", "path"):
            relative_path = _safe_relative_path(item.get(key))
            if relative_path is not None:
                safe[key] = relative_path
        for key in _SAFE_RESULT_NUMBERS:
            number = _safe_number(item.get(key), (-1_000_000, 1_000_000_000_000))
            if number is not None:
                safe[key] = number
        lines = item.get("lines")
        if isinstance(lines, (list, tuple)):
            safe_lines = [
                number
                for entry in lines[:2]
                if (number := _safe_number(entry, (0, 100_000_000))) is not None
            ]
            if safe_lines:
                safe["lines"] = safe_lines
        for key in ("kind", "language"):
            enum_value = item.get(key)
            if isinstance(enum_value, str) and enum_value in _SAFE_ENUM_PARAMS[key]:
                safe[key] = enum_value
        streams = item.get("streams")
        if isinstance(streams, dict):
            safe_streams = {
                key: number
                for key, raw in streams.items()
                if key in _SAFE_STREAMS
                and (number := _safe_number(raw, (-1_000_000, 1_000_000))) is not None
            }
            if safe_streams:
                safe["streams"] = safe_streams
        if safe:
            results.append(safe)
    return results


def _sanitize_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    """Project an event onto the non-sensitive trace schema."""
    safe: dict[str, Any] = {
        "tool": _safe_identifier(event.get("tool")) or "unknown",
    }
    for key in ("trace_id", "session_id", "candidate_id", "run_id"):
        identifier = _safe_identifier(event.get(key))
        if identifier is not None:
            safe[key] = identifier

    timestamp = event.get("ts")
    if isinstance(timestamp, str) and len(timestamp) <= 64:
        with contextlib.suppress(ValueError):
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            safe["ts"] = timestamp

    if "query" in event:
        safe["query"] = _query_fingerprint(event.get("query"))
    if "params" in event:
        safe["params"] = _sanitize_params(event.get("params"))

    raw_results = event.get("results")
    if "results" in event:
        safe["results"] = _sanitize_results(raw_results)
    explicit_count = _safe_number(event.get("result_count"), (0, 1_000_000_000))
    if isinstance(raw_results, list):
        safe["result_count"] = max(
            min(len(raw_results), 1_000_000_000),
            int(explicit_count or 0),
        )
    elif explicit_count is not None:
        safe["result_count"] = int(explicit_count)
    else:
        safe["result_count"] = 0

    for key, bounds in {
        "latency_ms": (0, 1_000_000_000_000),
        "returned_tokens": (0, 1_000_000_000_000),
    }.items():
        number = _safe_number(event.get(key), bounds)
        if number is not None:
            safe[key] = number

    raw_error = event.get("error")
    error_type = raw_error.get("type") if isinstance(raw_error, dict) else None
    if error_type is None:
        error_type = event.get("error_type")
    safe_error_type = _safe_identifier(error_type)
    if safe_error_type is not None:
        safe["error"] = {"type": safe_error_type}
    status = event.get("status")
    if safe_error_type is not None:
        safe["status"] = "error"
    else:
        safe["status"] = status if isinstance(status, str) and status in {"error", "ok"} else "ok"
    return safe


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
    run_id: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    returned_tokens: int = 0
    result_count: int = 0
    status: str = "ok"
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_trace_event(asdict(self))


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
        self.base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.base_dir.chmod(0o700)
        self._harden_existing_files()
        self.session_id = _safe_identifier(session_id) or uuid.uuid4().hex
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

        The hot path projects onto the safe schema, auto-fills identifiers,
        serialises to JSON bytes, and calls ``put_nowait``. No lock or syscall.

        Returns ``True`` if the record was enqueued, ``False`` if the queue
        was full and the record was dropped.
        """
        try:
            safe_event = _sanitize_trace_event(event)
            safe_event.setdefault("trace_id", uuid.uuid4().hex)
            safe_event.setdefault("session_id", self.session_id)
            safe_event.setdefault("ts", _iso_utc_now())
            payload = json.dumps(safe_event, ensure_ascii=False).encode("utf-8")
        except (RecursionError, TypeError, UnicodeError, ValueError):
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

    def _harden_existing_files(self) -> None:
        """Restrict regular trace files left by older recorder versions."""
        for path in self.base_dir.glob("trace-*.jsonl"):
            try:
                if not path.is_symlink() and path.is_file():
                    path.chmod(0o600)
            except OSError as exc:
                log.warning(
                    "trace.permissions_failed",
                    error_type=type(exc).__qualname__,
                )

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
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(str(path), flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            os.close(fd)
            raise
        self._fd = fd
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
