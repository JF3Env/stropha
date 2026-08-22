"""Watch a saved working tree and persistently reconcile file changes.

Per spec §8 / Phase 3: "Soft index para working tree (file watcher)."
The post-commit hook covers the committed state; this command covers
the *uncommitted* state while you iterate, so retrieval queries see
your latest edits within seconds of saving.

Design:
- Standard library only (``watchdog`` not pulled in to keep deps light).
  We poll full source hashes on a low-frequency interval (default 1 s),
  using the configured walker when the MCP runtime provides it.
- Debounced: a wave of writes from an LSP/format/save-on-blur cycle
  collapses into a single re-index pass after the debounce window
  (default 2 s of quiet).
- Respects `.gitignore` / `.strophaignore` via the standard ``Walker``
  filters, so build artefacts and caches don't trigger storms.
- Re-indexes through the pipeline's hash-based reconcile mode, so changed
  files are refreshed and unchanged files pay only the freshness skip.
- Doesn't reload the graphify graph or re-embed graph nodes by default
  — those are heavy and the hook handles them on commit. Override via
  ``--full-refresh`` if you want a complete pass.

MCP Integration:
- When ``mcp.watch: true`` in ``~/.stropha/config.yaml`` (the default), the MCP
  server starts the watcher in a background thread during lifespan. This ensures queries
  always see recent edits without requiring a separate ``stropha watch``.
- The watcher can be stopped via ``WatchController.stop()`` or by
  setting the threading.Event passed to ``watch_repo_async()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import signal
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_config
from .errors import PipelineError, StrophaError
from .ingest.walker import Walker
from .logging import get_logger
from .pipeline import Pipeline, build_stages, build_walker, load_pipeline_config
from .stages.walker import WalkerStage

log = get_logger(__name__)

_DEFAULT_INTERVAL_S = 1.0
_DEFAULT_DEBOUNCE_S = 2.0
_RETRY_INITIAL_S = 1.0
_RETRY_MAX_S = 30.0


@dataclass
class _Snapshot:
    """Full-hash snapshot for saved-source change detection."""

    files: dict[Path, tuple[int, int, str]] = field(default_factory=dict)

    def diff(self, fresh: _Snapshot) -> tuple[set[Path], set[Path]]:
        """Return ``(changed_or_added, removed)`` since the previous snapshot."""
        changed: set[Path] = set()
        for path, stamp in fresh.files.items():
            previous_stamp = self.files.get(path)
            if previous_stamp is None or previous_stamp != stamp:
                changed.add(path)
        removed = set(self.files) - set(fresh.files)
        return changed, removed


def _snapshot(
    repo: Path,
    *,
    max_file_bytes: int = 524_288,
    walker: WalkerStage | None = None,
) -> _Snapshot:
    """Walk indexable files and capture stable full-byte hashes."""
    snap = _Snapshot()
    source_files = (
        walker.discover(repo)
        if walker is not None
        else Walker(repo, max_file_bytes=max_file_bytes).discover()
    )
    for sf in source_files:
        try:
            # Snapshot identity follows the walker's repository-relative path,
            # not a potentially canonicalized physical SourceFile.path.
            logical_path = (repo / sf.rel_path).absolute()
            read_path = (
                sf.path if sf.path.is_absolute() else repo / sf.rel_path
            ).resolve()
            before = read_path.stat()
            raw = read_path.read_bytes()
            after = read_path.stat()
            resolved_after = logical_path.resolve()
        except OSError:
            continue
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or len(raw) != after.st_size
            or resolved_after != read_path
        ):
            continue
        snap.files[logical_path] = (
            after.st_mtime_ns,
            after.st_size,
            hashlib.sha256(raw).hexdigest(),
        )
    return snap


@dataclass
class _WatchState:
    """Pending reconciliation and retry state shared by both watcher entry points."""

    previous: _Snapshot
    pending_since: float | None
    retry_at: float | None = None
    failures: int = 0
    next_retry_s: float = field(default=_RETRY_INITIAL_S, repr=False)

    @classmethod
    def starting(cls, previous: _Snapshot, *, now: float) -> _WatchState:
        # Reconcile edits made while no watcher was running.
        return cls(previous=previous, pending_since=now)

    def observe(self, fresh: _Snapshot, *, now: float) -> tuple[set[Path], set[Path]]:
        """Merge a fresh filesystem observation into pending work."""
        changed, removed = self.previous.diff(fresh)
        if changed or removed:
            self.previous = fresh
            self.pending_since = now
        return changed, removed

    def is_ready(self, *, now: float, debounce_s: float) -> bool:
        """Return whether pending work is quiet and outside retry backoff."""
        if self.pending_since is None or now < self.pending_since + debounce_s:
            return False
        return self.retry_at is None or now >= self.retry_at

    def mark_succeeded(self) -> None:
        """Clear pending work only after a complete refresh."""
        self.pending_since = None
        self.retry_at = None
        self.failures = 0
        self.next_retry_s = _RETRY_INITIAL_S

    def mark_failed(self, *, now: float) -> float:
        """Keep work pending and schedule the next bounded retry."""
        delay = self.next_retry_s
        self.retry_at = now + delay
        self.failures += 1
        self.next_retry_s = min(delay * 2, _RETRY_MAX_S)
        return delay


def _format_changes(
    changed: Iterable[Path],
    removed: Iterable[Path],
    *,
    root: Path,
) -> str:
    parts: list[str] = []
    changed_list = sorted(changed)
    removed_list = sorted(removed)
    if changed_list:
        rels = [
            str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in changed_list[:5]
        ]
        more = ""
        if len(changed_list) > 5:
            more = f" (+{len(changed_list) - 5} more)"
        parts.append(f"{len(changed_list)} changed: {', '.join(rels)}{more}")
    if removed_list:
        parts.append(f"{len(removed_list)} removed")
    return " · ".join(parts) if parts else "no changes"


def _run_watch_loop(
    repo: Path,
    *,
    interval_s: float,
    debounce_s: float,
    full_refresh: bool,
    wait: Callable[[float], bool],
    max_file_bytes: int = 524_288,
    walker: WalkerStage | None = None,
    once: bool = False,
    now_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Poll, debounce, and retry until ``wait`` reports a stop request."""
    try:
        initial = _snapshot(repo, max_file_bytes=max_file_bytes, walker=walker)
    except Exception as exc:
        log.warning("watch.snapshot_failed", error=str(exc))
        initial = _Snapshot()
    state = _WatchState.starting(initial, now=now_fn())

    while not wait(interval_s):
        try:
            fresh = _snapshot(repo, max_file_bytes=max_file_bytes, walker=walker)
        except Exception as exc:
            log.warning("watch.snapshot_failed", error=str(exc))
            if once:
                break
            continue

        now = now_fn()
        changed, removed = state.observe(fresh, now=now)
        if changed or removed:
            log.info(
                "watch.changes_seen",
                summary=_format_changes(changed, removed, root=repo),
            )

        if state.is_ready(now=now, debounce_s=debounce_s):
            log.info("watch.reindex_start")
            try:
                _reindex(repo, full_refresh=full_refresh)
            except Exception as exc:
                retry_in_s = state.mark_failed(now=now_fn())
                log.warning(
                    "watch.reindex_failed",
                    error=str(exc),
                    failures=state.failures,
                    retry_in_s=retry_in_s,
                )
            else:
                state.mark_succeeded()
                log.info("watch.reindex_done")

        if once:
            break


def watch_repo(
    repo: Path,
    *,
    interval_s: float = _DEFAULT_INTERVAL_S,
    debounce_s: float = _DEFAULT_DEBOUNCE_S,
    full_refresh: bool = False,
    once: bool = False,
) -> None:
    """Watch ``repo`` and re-index changed files after the debounce window.

    Args:
        repo: The repository root to watch.
        interval_s: Polling interval in seconds. Lower = snappier, higher
            CPU. 1.0 is comfortable for most editors.
        debounce_s: Wait this many seconds of *no further changes* before
            triggering a re-index. Protects against save bursts.
        full_refresh: When True, every re-index runs a full ``Pipeline.run()``
            (including graphify graph reload and graph-vec embedding).
            Default False — only the changed files go through the pipeline.
        once: Run one detection cycle and return. Used by tests.
    """
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists() and not repo.is_dir():
        raise StrophaError(f"Not a directory: {repo}")
    log.info(
        "watch.start",
        repo=str(repo),
        interval_s=interval_s,
        debounce_s=debounce_s,
        full_refresh=full_refresh,
    )

    stop_event = threading.Event()

    def _on_sigint(_sig: int, _frame: object) -> None:
        stop_event.set()
        log.info("watch.stop_requested")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
        signal.signal(signal.SIGTERM, _on_sigint)
    except (ValueError, OSError):
        # signal() can only be called from main thread; in test runners
        # the watcher may run elsewhere — skip silently.
        pass

    resolved = load_pipeline_config()
    _run_watch_loop(
        repo,
        interval_s=interval_s,
        debounce_s=debounce_s,
        full_refresh=full_refresh,
        wait=stop_event.wait,
        max_file_bytes=get_config().max_file_bytes,
        walker=build_walker(resolved),
        once=once,
    )


def _reindex(repo: Path, *, full_refresh: bool) -> None:
    """Run a fresh ``Pipeline`` pass against ``repo``."""
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    inference = get_config().inference
    try:
        with built.storage:  # type: ignore[union-attr]
            pipeline = Pipeline(
                storage=built.storage,
                embedder=built.embedder,
                secondary_embedder=built.secondary_embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=[repo],
                enricher_concurrency=inference.enricher_concurrency,
                enricher_batch_size=inference.effective_batch_size(),
                graphify=full_refresh,
            )
            stats = pipeline.run()
        log.info(
            "watch.reindex_summary",
            files=stats.files_visited,
            chunks=stats.chunks_seen,
            embedded=stats.chunks_embedded,
            reused=stats.chunks_skipped_fresh,
            failed=stats.chunks_enriched_failed,
        )
        if stats.chunks_enriched_failed:
            raise PipelineError(
                f"Re-index incomplete: {stats.chunks_enriched_failed} chunks failed enrichment"
            )
    finally:
        if built.storage is not None:
            with contextlib.suppress(Exception):
                built.storage.close()


# ---------------------------------------------------------------------------
# Async/threaded watch for MCP server integration
# ---------------------------------------------------------------------------


@dataclass
class WatchController:
    """Controller for a background watch thread.

    Use this to start/stop the watcher from the MCP server lifespan.
    """

    repo: Path
    interval_s: float = _DEFAULT_INTERVAL_S
    debounce_s: float = _DEFAULT_DEBOUNCE_S
    full_refresh: bool = False
    max_file_bytes: int = 524_288
    walker: WalkerStage | None = None

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        """Start the watch loop in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("watch.already_running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="stropha-watch",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "watch.thread_started",
            repo=str(self.repo),
            interval_s=self.interval_s,
            debounce_s=self.debounce_s,
        )

    def stop(self, timeout: float | None = None) -> None:
        """Signal the watch loop to stop and wait for thread to finish."""
        if self._thread is None or not self._thread.is_alive():
            return
        log.info("watch.thread_stopping")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning("watch.thread_did_not_stop", timeout=timeout)
        else:
            log.info("watch.thread_stopped")

    def _run_loop(self) -> None:
        """The actual watch loop, runs in a background thread."""
        repo = self.repo.expanduser().resolve()
        if not repo.is_dir():
            log.error("watch.not_a_directory", path=str(repo))
            return

        _run_watch_loop(
            repo,
            interval_s=self.interval_s,
            debounce_s=self.debounce_s,
            full_refresh=self.full_refresh,
            wait=self._stop_event.wait,
            max_file_bytes=self.max_file_bytes,
            walker=self.walker,
        )
