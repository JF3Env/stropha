"""Runtime/context helpers extracted from ``stropha.server``."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .config import Config
from .embeddings.base import Embedder
from .logging import get_logger
from .pipeline import BuiltStages
from .stages.retrieval import RetrievalStage
from .storage import Storage

# Lightweight type aliases for FastMCP types that depend on an optional
# install of the ``mcp`` package. Keeping them as ``Any`` lets the runtime
# stay import-cheap without forcing every consumer of this module to pull
# the FastMCP stub graph.
Context = Any
FastMCP = Any

log = get_logger(__name__)

_SHARED_RUNTIME_ATTR = "_stropha_shared_runtime"
_SHARED_RUNTIME_LOCK_ATTR = "_stropha_shared_runtime_lock"


@dataclass
class SharedRuntime:
    """Process-level dependencies reused by every MCP session."""

    config: Config
    built: BuiltStages
    watcher: Any = None
    retired: list[BuiltStages] = field(default_factory=list)
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.watcher is not None:
            try:
                self.watcher.stop()
            except Exception as exc:
                log.warning("mcp.watcher.stop_failed", error=str(exc))
        seen: set[int] = set()
        for built in [self.built, *self.retired]:
            storage = built.storage
            if id(storage) in seen:
                continue
            seen.add(id(storage))
            try:
                storage.close()
            except Exception as exc:
                log.warning("mcp.storage.close_failed", error=str(exc))


@dataclass
class AppContext:
    """Shared dependencies plus state isolated to one MCP session."""

    config: Config
    storage: Storage
    embedder: Embedder
    # Retains the historical field name for tool helpers; production injects
    # the configured RetrievalStage, not the legacy SearchEngine.
    search_engine: RetrievalStage
    built: BuiltStages
    session_id: str = ""
    trace_recorder: object = None
    bad_result_store: object = None
    seen_chunk_ids: set[str] = field(default_factory=set)
    pinned_chunk_ids: set[str] = field(default_factory=set)
    client_id: str | None = None
    recent_queries: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    scratchpad: str = ""


def chunk_belongs_to_repo(storage: Any, chunk_id: str, repo_id: int | None) -> bool:
    """Return whether an exact chunk exists within the requested repository scope."""
    try:
        chunk = storage.get_chunk_by_id(chunk_id)
    except Exception:
        return False
    return chunk is not None and (repo_id is None or chunk.get("repo_id") == repo_id)


def ctx_from_mcp(mcp_ctx: Context) -> AppContext:
    """Pull the AppContext out of an MCP Context and bind session_id to logs."""
    app: AppContext = mcp_ctx.request_context.lifespan_context
    # Self-heal a dropped DB connection (e.g. the Postgres container bounced by
    # a Colima/Docker VM restart on laptop sleep) before the tool touches
    # storage. Best-effort + cheap (one ``SELECT 1``); a no-op for backends
    # (e.g. sqlite) that don't expose ``ensure_connected``.
    ensure_connected = getattr(app.storage, "ensure_connected", None)
    if callable(ensure_connected):
        ensure_connected()
    if app.session_id:
        import structlog

        structlog.contextvars.bind_contextvars(session_id=app.session_id)
    return app


def persist_session_state(app: AppContext) -> None:
    """Write-through the current AppContext session state to storage."""
    if not app.client_id:
        return
    try:
        app.storage.upsert_session(
            session_id=app.session_id,
            client_id=app.client_id,
            seen_chunk_ids=app.seen_chunk_ids,
            pinned_chunk_ids=app.pinned_chunk_ids,
            scratchpad=app.scratchpad,
        )
    except Exception as exc:
        log.warning(
            "mcp.sessions.persist_failed",
            session_id=app.session_id,
            error=str(exc),
        )


async def _shared_runtime(
    app: FastMCP,
    *,
    version: str,
    get_config_fn: Callable[[], Config],
    configure_logging_fn: Callable[[str], None],
    load_pipeline_config_fn: Callable[[], Any],
    build_stages_fn: Callable[[Any], BuiltStages],
    watch_controller_cls: type[Any],
) -> SharedRuntime:
    """Build the heavy pipeline once on the process-wide FastMCP instance."""
    lock = getattr(app, _SHARED_RUNTIME_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(app, _SHARED_RUNTIME_LOCK_ATTR, lock)

    async with lock:
        existing = getattr(app, _SHARED_RUNTIME_ATTR, None)
        if existing is not None and not existing.closed:
            return existing

        cfg = get_config_fn()
        configure_logging_fn(cfg.log_level)
        log.info(
            "mcp.startup",
            version=version,
            index_path=str(cfg.resolve_index_path()),
            target=str(cfg.target_repo),
            mcp_watch=cfg.mcp_watch,
        )
        built = build_stages_fn(load_pipeline_config_fn())

        watcher: Any = None
        try:
            if cfg.mcp_watch:
                watcher = watch_controller_cls(
                    repo=cfg.target_repo,
                    interval_s=cfg.mcp_watch_interval,
                    debounce_s=cfg.mcp_watch_debounce,
                    full_refresh=False,
                )
                watcher.start()
        except Exception:
            SharedRuntime(config=cfg, built=built, watcher=watcher).close()
            raise

        runtime = SharedRuntime(config=cfg, built=built, watcher=watcher)
        setattr(app, _SHARED_RUNTIME_ATTR, runtime)
        return runtime


async def get_shared_runtime(
    app: FastMCP,
    *,
    version: str,
    get_config_fn: Callable[[], Config],
    configure_logging_fn: Callable[[str], None],
    load_pipeline_config_fn: Callable[[], Any],
    build_stages_fn: Callable[[Any], BuiltStages],
    watch_controller_cls: type[Any],
) -> SharedRuntime:
    """Return the process runtime, building it when no MCP session has yet."""
    return await _shared_runtime(
        app,
        version=version,
        get_config_fn=get_config_fn,
        configure_logging_fn=configure_logging_fn,
        load_pipeline_config_fn=load_pipeline_config_fn,
        build_stages_fn=build_stages_fn,
        watch_controller_cls=watch_controller_cls,
    )


async def reload_shared_runtime(
    app: FastMCP,
    *,
    get_config_fn: Callable[[], Config],
    configure_logging_fn: Callable[[str], None],
    load_pipeline_config_fn: Callable[[], Any],
    build_stages_fn: Callable[[Any], BuiltStages],
    watch_controller_cls: type[Any],
) -> SharedRuntime:
    """Build and atomically install a replacement process-level runtime.

    Existing MCP sessions retain references to their prior adapters. Their
    storage handles are therefore retired, not closed immediately; shutdown
    closes both the current and retired handles. New sessions and frontend
    requests observe the replacement as soon as it is installed.
    """
    lock = getattr(app, _SHARED_RUNTIME_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(app, _SHARED_RUNTIME_LOCK_ATTR, lock)

    async with lock:
        cfg = get_config_fn()
        pipeline_config = load_pipeline_config_fn()
        # Adapter construction may download and materialize multi-gigabyte
        # models. Keep the ASGI event loop responsive while that work runs.
        built = await asyncio.to_thread(build_stages_fn, pipeline_config)
        watcher: Any = None
        try:
            if cfg.mcp_watch:
                watcher = watch_controller_cls(
                    repo=cfg.target_repo,
                    interval_s=cfg.mcp_watch_interval,
                    debounce_s=cfg.mcp_watch_debounce,
                    full_refresh=False,
                )
                watcher.start()
        except Exception:
            SharedRuntime(config=cfg, built=built, watcher=watcher).close()
            raise

        try:
            configure_logging_fn(cfg.log_level)
        except Exception:
            SharedRuntime(config=cfg, built=built, watcher=watcher).close()
            raise

        previous = getattr(app, _SHARED_RUNTIME_ATTR, None)
        retired: list[BuiltStages] = []
        if previous is not None and not previous.closed:
            if previous.watcher is not None:
                try:
                    previous.watcher.stop()
                except Exception as exc:
                    log.warning("mcp.watcher.stop_failed", error=str(exc))
                previous.watcher = None
            retired = [previous.built, *previous.retired]
            previous.closed = True

        runtime = SharedRuntime(
            config=cfg,
            built=built,
            watcher=watcher,
            retired=retired,
        )
        setattr(app, _SHARED_RUNTIME_ATTR, runtime)
        log.info(
            "mcp.runtime_reloaded",
            retired_runtimes=len(retired),
            storage=getattr(built.storage, "adapter_name", type(built.storage).__name__),
        )
        return runtime


def close_shared_runtime(app: FastMCP) -> None:
    """Close and detach process-level resources during ASGI shutdown."""
    runtime = getattr(app, _SHARED_RUNTIME_ATTR, None)
    if runtime is not None:
        runtime.close()
        delattr(app, _SHARED_RUNTIME_ATTR)
    if hasattr(app, _SHARED_RUNTIME_LOCK_ATTR):
        delattr(app, _SHARED_RUNTIME_LOCK_ATTR)
    if runtime is not None:
        log.info("mcp.shutdown")


@asynccontextmanager
async def lifespan(
    app: FastMCP,
    *,
    version: str,
    get_config_fn: Callable[[], Config],
    configure_logging_fn: Callable[[str], None],
    load_pipeline_config_fn: Callable[[], Any],
    build_stages_fn: Callable[[Any], Any],
    watch_controller_cls: type[Any],
) -> AsyncIterator[AppContext]:
    from .meta import TraceRecorder
    from .meta.feedback import BadResultStore

    shared = await _shared_runtime(
        app,
        version=version,
        get_config_fn=get_config_fn,
        configure_logging_fn=configure_logging_fn,
        load_pipeline_config_fn=load_pipeline_config_fn,
        build_stages_fn=build_stages_fn,
        watch_controller_cls=watch_controller_cls,
    )
    cfg = shared.config
    built = shared.built
    embedder = built.embedder
    storage = cast(Storage, built.storage)
    engine = built.retrieval

    session_id = uuid.uuid4().hex
    seen_chunk_ids: set[str] = set()
    pinned_chunk_ids: set[str] = set()
    client_id = os.environ.get("STROPHA_CLIENT_ID")
    if client_id:
        try:
            prior_sessions = storage.list_sessions(client_id=client_id, limit=1)
        except Exception as exc:
            log.warning("mcp.sessions.list_failed", error=str(exc))
            prior_sessions = []
        if prior_sessions:
            prior = prior_sessions[0]
            seen_chunk_ids = set(prior.seen_chunk_ids)
            pinned_chunk_ids = set(prior.pinned_chunk_ids)
            prior_scratchpad = prior.scratchpad or ""
            session_id = prior.session_id
            log.info(
                "mcp.sessions.resumed",
                session_id=session_id,
                client_id=client_id,
                seen=len(seen_chunk_ids),
                pinned=len(pinned_chunk_ids),
                scratchpad_chars=len(prior_scratchpad),
            )
        else:
            prior_scratchpad = ""
    else:
        prior_scratchpad = ""

    traces_dir = (
        Path(traces_dir_env)
        if (traces_dir_env := os.environ.get("STROPHA_TRACES_DIR"))
        else Path(cfg.resolve_index_path()).parent / "meta" / "traces"
    )
    try:
        trace_recorder: object = TraceRecorder(base_dir=traces_dir, session_id=session_id)
    except OSError as exc:
        log.warning("mcp.trace_recorder.disabled", error=str(exc))
        trace_recorder = None

    try:
        bad_result_store: object = BadResultStore(base_dir=traces_dir.parent)
    except OSError as exc:
        log.warning("mcp.bad_result_store.disabled", error=str(exc))
        bad_result_store = None

    app_ctx = AppContext(
        config=cfg,
        storage=storage,
        embedder=embedder,
        search_engine=engine,
        built=built,
        session_id=session_id,
        trace_recorder=trace_recorder,
        bad_result_store=bad_result_store,
        seen_chunk_ids=seen_chunk_ids,
        pinned_chunk_ids=pinned_chunk_ids,
        client_id=client_id,
        scratchpad=prior_scratchpad,
    )
    log.info("mcp.session.startup", session_id=session_id, client_id=client_id)
    try:
        yield app_ctx
    finally:
        if trace_recorder is not None:
            try:
                trace_recorder.close()  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("mcp.trace_recorder.close_failed", error=str(exc))
        try:
            storage.upsert_session(
                session_id=session_id,
                client_id=client_id,
                seen_chunk_ids=app_ctx.seen_chunk_ids,
                pinned_chunk_ids=app_ctx.pinned_chunk_ids,
                scratchpad=app_ctx.scratchpad,
            )
        except Exception as exc:
            log.warning("mcp.sessions.persist_failed", error=str(exc))
        log.info("mcp.session.shutdown", session_id=session_id, client_id=client_id)
