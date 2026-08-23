"""Persist saved working-tree changes through native filesystem notifications.

The post-commit hook covers committed state. This watcher covers uncommitted
saves without repeatedly walking and hashing the repository while idle.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watchfiles import Change, watch

from .config import get_config
from .errors import FullReconcileRequiredError, PipelineError, StrophaError
from .logging import get_logger
from .pipeline import BuiltStages, Pipeline, PipelineStats, build_stages, load_pipeline_config
from .pipeline.builder import build_enricher, build_storage
from .stages.storage import StorageStage

log = get_logger(__name__)

_DEFAULT_INTERVAL_S = 1.0
_DEFAULT_DEBOUNCE_S = 2.0
_DEFAULT_RECONCILE_INTERVAL_S = 300.0
_RETRY_INITIAL_S = 1.0
_RETRY_MAX_S = 30.0
_STARTUP_TIMEOUT_S = 5.0
_ADMISSION_RULE_FILES = frozenset({".gitignore", ".strophaignore", ".ragignore"})
_GIT_ADMISSION_PATHS = frozenset({("config",), ("index",), ("info", "exclude")})

FileChange = tuple[Change, str]
WatchFn = Callable[..., Iterator[set[FileChange]]]


@dataclass
class _PendingRefresh:
    """Coalesced native events plus bounded retry state."""

    changed: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    full_reconcile: bool = False
    retry_at: float | None = None
    failures: int = 0
    next_retry_s: float = field(default=_RETRY_INITIAL_S, repr=False)

    @property
    def has_work(self) -> bool:
        return self.full_reconcile or bool(self.changed or self.deleted)

    def merge(
        self,
        changed: Iterable[str],
        deleted: Iterable[str],
        *,
        full_reconcile: bool = False,
    ) -> None:
        if full_reconcile:
            self.full_reconcile = True
            self.changed.clear()
            self.deleted.clear()
            return
        if self.full_reconcile:
            return
        changed_set = set(changed)
        deleted_set = set(deleted)
        self.deleted.difference_update(changed_set)
        self.changed.update(changed_set)
        self.changed.difference_update(deleted_set)
        self.deleted.update(deleted_set)

    def is_ready(self, *, now: float) -> bool:
        return self.has_work and (self.retry_at is None or now >= self.retry_at)

    def mark_succeeded(self) -> None:
        self.changed.clear()
        self.deleted.clear()
        self.full_reconcile = False
        self.retry_at = None
        self.failures = 0
        self.next_retry_s = _RETRY_INITIAL_S

    def mark_failed(self, *, now: float) -> float:
        delay = self.next_retry_s
        self.retry_at = now + delay
        self.failures += 1
        self.next_retry_s = min(delay * 2, _RETRY_MAX_S)
        return delay


def _classify_changes(repo: Path, changes: Iterable[FileChange]) -> tuple[set[str], set[str], bool]:
    """Resolve event actions against final filesystem state."""
    changed: set[str] = set()
    deleted: set[str] = set()
    full_reconcile = False
    root = repo.expanduser().resolve()

    touched: set[str] = set()
    for _change, raw_path in changes:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.absolute().relative_to(root)
        except ValueError:
            continue
        if ".git" in relative.parts:
            if _is_git_admission_path(relative):
                full_reconcile = True
            continue
        rel_path = relative.as_posix()
        if rel_path in ("", "."):
            full_reconcile = True
            continue
        touched.add(rel_path)

    for rel_path in touched:
        path = root / rel_path
        if Path(rel_path).name in _ADMISSION_RULE_FILES or path.is_dir():
            full_reconcile = True
        elif path.is_file():
            changed.add(rel_path)
        else:
            deleted.add(rel_path)

    if full_reconcile:
        return set(), set(), True
    return changed, deleted, False


def _is_git_admission_path(relative: Path) -> bool:
    """Return whether Git metadata can change configured walker admission."""
    for index, part in enumerate(relative.parts):
        if part != ".git":
            continue
        remainder = relative.parts[index + 1 :]
        if not remainder or remainder in _GIT_ADMISSION_PATHS:
            return True
    return False


def _watch_filter(repo: Path) -> Callable[[Change, str], bool]:
    """Keep source and Git-admission events while discarding other Git internals."""
    root = repo.expanduser().resolve()

    def include(_change: Change, raw_path: str) -> bool:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.absolute().relative_to(root)
        except ValueError:
            return False
        return ".git" not in relative.parts or _is_git_admission_path(relative)

    return include


def _format_changes(changed: Iterable[str], deleted: Iterable[str]) -> str:
    changed_list = sorted(changed)
    deleted_list = sorted(deleted)
    parts: list[str] = []
    if changed_list:
        examples = ", ".join(changed_list[:5])
        more = f" (+{len(changed_list) - 5} more)" if len(changed_list) > 5 else ""
        parts.append(f"{len(changed_list)} changed: {examples}{more}")
    if deleted_list:
        parts.append(f"{len(deleted_list)} removed")
    return " · ".join(parts) if parts else "full reconcile"


def _run_watch_loop(
    repo: Path,
    *,
    interval_s: float,
    debounce_s: float,
    full_refresh: bool,
    stop_event: threading.Event,
    refresh: Callable[..., None],
    once: bool = False,
    watch_fn: WatchFn | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    reconcile_interval_s: float = _DEFAULT_RECONCILE_INTERVAL_S,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Consume native event batches, periodically reconcile, and retry failures."""
    watch_fn = watch if watch_fn is None else watch_fn
    interval_ms = max(1, int(interval_s * 1000))
    debounce_ms = max(1, int(debounce_s * 1000))
    # Keep backend batches responsive and apply the user-facing quiet period below.
    batch_window_ms = min(debounce_ms, 50)
    timeout_ms = max(50, min(interval_ms, debounce_ms, 1000))
    event_batches = iter(
        watch_fn(
            repo,
            watch_filter=_watch_filter(repo),
            debounce=batch_window_ms,
            step=batch_window_ms,
            stop_event=stop_event,
            rust_timeout=timeout_ms,
            yield_on_timeout=True,
            raise_interrupt=False,
            force_polling=None,
        )
    )

    first_batch = next(event_batches)
    if on_ready is not None:
        on_ready()

    pending = _PendingRefresh(full_reconcile=True)
    settle_at: float | None = None
    changed, deleted, requires_full = _classify_changes(repo, first_batch)
    pending.merge(changed, deleted, full_reconcile=requires_full)
    last_reconcile = now_fn()

    def apply_pending(now: float) -> bool:
        nonlocal last_reconcile, settle_at
        if settle_at is not None and now < settle_at:
            return False
        if not pending.is_ready(now=now):
            return False
        settle_at = None
        requested_full = pending.full_reconcile or full_refresh
        log.info(
            "watch.refresh_start",
            summary=_format_changes(pending.changed, pending.deleted),
            full_reconcile=requested_full,
        )
        try:
            refresh(
                set(pending.changed),
                set(pending.deleted),
                full_reconcile=requested_full,
            )
        except Exception as exc:
            retry_in_s = pending.mark_failed(now=now_fn())
            log.warning(
                "watch.refresh_failed",
                error=str(exc),
                failures=pending.failures,
                retry_in_s=retry_in_s,
            )
            return False
        if requested_full:
            last_reconcile = now_fn()
        pending.mark_succeeded()
        log.info("watch.refresh_done")
        return True

    initial_succeeded = apply_pending(now_fn())
    if once and initial_succeeded:
        return

    for batch in event_batches:
        if stop_event.is_set():
            break
        now = now_fn()
        changed, deleted, requires_full = _classify_changes(repo, batch)
        if changed or deleted or requires_full:
            pending.merge(changed, deleted, full_reconcile=requires_full)
            settle_at = now + max(0.0, debounce_s)
            log.info(
                "watch.changes_seen",
                summary=_format_changes(changed, deleted),
            )
        if now - last_reconcile >= reconcile_interval_s:
            pending.merge(set(), set(), full_reconcile=True)
            settle_at = None
        succeeded = apply_pending(now)
        if once and succeeded:
            return


def _run_resilient_watch_loop(
    repo: Path,
    *,
    interval_s: float,
    debounce_s: float,
    full_refresh: bool,
    stop_event: threading.Event,
    refresh: Callable[..., None],
    once: bool = False,
    watch_fn: WatchFn | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    reconcile_interval_s: float = _DEFAULT_RECONCILE_INTERVAL_S,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Restart an initialized native backend with bounded backoff."""
    ready = False
    backend_retry_s = _RETRY_INITIAL_S

    def mark_ready() -> None:
        nonlocal ready
        ready = True
        if on_ready is not None:
            on_ready()

    while not stop_event.is_set():
        try:
            _run_watch_loop(
                repo,
                interval_s=interval_s,
                debounce_s=debounce_s,
                full_refresh=full_refresh,
                stop_event=stop_event,
                refresh=refresh,
                once=once,
                watch_fn=watch_fn,
                now_fn=now_fn,
                reconcile_interval_s=reconcile_interval_s,
                on_ready=mark_ready,
            )
            if once or stop_event.is_set():
                return
            raise StrophaError("Native watcher stopped unexpectedly")
        except Exception as exc:
            if not ready:
                raise
            if stop_event.is_set():
                return
            log.warning(
                "watch.backend_failed",
                error=str(exc),
                retry_in_s=backend_retry_s,
            )
            if stop_event.wait(backend_retry_s):
                return
            backend_retry_s = min(backend_retry_s * 2, _RETRY_MAX_S)


def _pipeline_from_stages(
    repo: Path,
    built: BuiltStages,
    *,
    storage: StorageStage | None = None,
    enricher: Any = None,
    graphify: bool,
    enricher_concurrency: int,
    enricher_batch_size: int,
) -> Pipeline:
    return Pipeline(
        storage=storage or built.storage,
        embedder=built.embedder,
        secondary_embedder=built.secondary_embedder,
        enricher=enricher or built.enricher,
        walker=built.walker,
        chunker=built.chunker,
        repos=[repo],
        enricher_concurrency=enricher_concurrency,
        enricher_batch_size=enricher_batch_size,
        graphify=graphify,
    )


def _refresh_pipeline(
    pipeline: Pipeline,
    repo: Path,
    changed: set[str],
    deleted: set[str],
    *,
    full_reconcile: bool,
) -> None:
    if full_reconcile:
        stats = pipeline.run(mode="reconcile")
    else:
        try:
            stats = pipeline.refresh_paths(
                repo=repo,
                changed_or_added=changed,
                deleted=deleted,
            )
        except FullReconcileRequiredError as exc:
            log.info("watch.reconcile_required", reason=str(exc))
            stats = pipeline.run(mode="reconcile")
    _log_refresh_stats(stats)


def _log_refresh_stats(stats: PipelineStats) -> None:
    log.info(
        "watch.refresh_summary",
        files=stats.files_visited,
        evicted=stats.files_evicted,
        chunks=stats.chunks_seen,
        embedded=stats.chunks_embedded,
        reused=stats.chunks_skipped_fresh,
        failed=stats.chunks_enriched_failed,
    )
    if stats.chunks_enriched_failed:
        raise PipelineError(
            f"Re-index incomplete: {stats.chunks_enriched_failed} chunks failed enrichment"
        )


def watch_repo(
    repo: Path,
    *,
    interval_s: float = _DEFAULT_INTERVAL_S,
    debounce_s: float = _DEFAULT_DEBOUNCE_S,
    full_refresh: bool = False,
    once: bool = False,
) -> None:
    """Watch ``repo`` and persist debounced saved-file changes."""
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise StrophaError(f"Not a directory: {repo}")
    log.info(
        "watch.start",
        repo=str(repo),
        interval_s=interval_s,
        debounce_s=debounce_s,
        full_refresh=full_refresh,
        backend="watchfiles",
    )

    stop_event = threading.Event()

    def _on_signal(_sig: int, _frame: object) -> None:
        stop_event.set()
        log.info("watch.stop_requested")

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except (ValueError, OSError):
        pass

    built = build_stages(load_pipeline_config())
    inference = get_config().inference
    pipeline = _pipeline_from_stages(
        repo,
        built,
        graphify=full_refresh,
        enricher_concurrency=inference.enricher_concurrency,
        enricher_batch_size=inference.effective_batch_size(),
    )
    try:
        _run_resilient_watch_loop(
            repo,
            interval_s=interval_s,
            debounce_s=debounce_s,
            full_refresh=full_refresh,
            stop_event=stop_event,
            refresh=lambda changed, deleted, *, full_reconcile: _refresh_pipeline(
                pipeline,
                repo,
                changed,
                deleted,
                full_reconcile=full_reconcile,
            ),
            once=once,
        )
    finally:
        with contextlib.suppress(Exception):
            built.storage.close()


def _reindex(repo: Path, *, full_refresh: bool) -> None:
    """Run one independently-owned reconcile pass against ``repo``."""
    built = build_stages(load_pipeline_config())
    inference = get_config().inference
    try:
        pipeline = _pipeline_from_stages(
            repo,
            built,
            graphify=full_refresh,
            enricher_concurrency=inference.enricher_concurrency,
            enricher_batch_size=inference.effective_batch_size(),
        )
        _log_refresh_stats(pipeline.run(mode="reconcile"))
    finally:
        if built.storage is not None:
            with contextlib.suppress(Exception):
                built.storage.close()


@dataclass
class WatchController:
    """Own a native watch thread and its long-lived writer connection."""

    repo: Path
    interval_s: float = _DEFAULT_INTERVAL_S
    debounce_s: float = _DEFAULT_DEBOUNCE_S
    full_refresh: bool = False
    stages: BuiltStages | None = None
    enricher_concurrency: int = 1
    enricher_batch_size: int = 1
    reconcile_interval_s: float = _DEFAULT_RECONCILE_INTERVAL_S

    _stop_event: threading.Event = field(default_factory=threading.Event)
    _ready_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, init=False)
    _startup_error: BaseException | None = field(default=None, init=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            log.warning("watch.already_running")
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="stropha-watch",
            daemon=True,
        )
        self._thread.start()
        startup_timeout_s = _STARTUP_TIMEOUT_S
        if not self._ready_event.wait(startup_timeout_s):
            self._stop_event.set()
            raise StrophaError(
                f"Native watcher did not become ready within {startup_timeout_s:g} seconds"
            )
        if self._startup_error is not None:
            raise StrophaError(f"Native watcher failed to start: {self._startup_error}")
        log.info(
            "watch.thread_started",
            repo=str(self.repo),
            interval_s=self.interval_s,
            debounce_s=self.debounce_s,
            backend="watchfiles",
        )

    def stop(self, timeout: float | None = None) -> None:
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
        repo = self.repo.expanduser().resolve()
        writer_storage: StorageStage | None = None
        ready = False
        try:
            if not repo.is_dir():
                raise StrophaError(f"Not a directory: {repo}")

            refresh_fn: Callable[..., None]
            if self.stages is None:

                def cold_refresh(
                    _changed: set[str],
                    _deleted: set[str],
                    *,
                    full_reconcile: bool,
                ) -> None:
                    _reindex(repo, full_refresh=self.full_refresh)

                refresh_fn = cold_refresh
            else:
                writer_storage = build_storage(
                    self.stages.resolved,
                    embedding_dim=self.stages.embedder.dim,
                )
                enricher = self.stages.enricher
                if getattr(enricher, "_storage", None) is not None:
                    enricher = build_enricher(self.stages.resolved, storage=writer_storage)
                pipeline = _pipeline_from_stages(
                    repo,
                    self.stages,
                    storage=writer_storage,
                    enricher=enricher,
                    graphify=self.full_refresh,
                    enricher_concurrency=self.enricher_concurrency,
                    enricher_batch_size=self.enricher_batch_size,
                )

                def warm_refresh(
                    changed: set[str],
                    deleted: set[str],
                    *,
                    full_reconcile: bool,
                ) -> None:
                    _refresh_pipeline(
                        pipeline,
                        repo,
                        changed,
                        deleted,
                        full_reconcile=full_reconcile,
                    )

                refresh_fn = warm_refresh

            def mark_ready() -> None:
                nonlocal ready
                ready = True
                self._ready_event.set()

            _run_resilient_watch_loop(
                repo,
                interval_s=self.interval_s,
                debounce_s=self.debounce_s,
                full_refresh=self.full_refresh,
                stop_event=self._stop_event,
                refresh=refresh_fn,
                reconcile_interval_s=self.reconcile_interval_s,
                on_ready=mark_ready,
            )
        except BaseException as exc:
            if not ready:
                self._startup_error = exc
                self._ready_event.set()
            elif not self._stop_event.is_set():
                log.error("watch.thread_failed", error=str(exc))
        finally:
            if writer_storage is not None:
                with contextlib.suppress(Exception):
                    writer_storage.close()
