"""Process-level MCP runtime lifecycle tests (WIN-428)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _config(tmp_path):
    return SimpleNamespace(
        log_level="INFO",
        target_repo=tmp_path,
        mcp_watch=False,
        mcp_watch_interval=1.0,
        mcp_watch_debounce=1.0,
        resolve_index_path=lambda: tmp_path / "index.db",
    )


@pytest.mark.asyncio
async def test_lifespan_reuses_heavy_runtime_and_isolates_session_state(tmp_path) -> None:
    from stropha.server_runtime import close_shared_runtime, lifespan

    owner = SimpleNamespace()
    storage = MagicMock()
    storage.list_sessions.return_value = []
    built = SimpleNamespace(
        embedder=MagicMock(),
        storage=storage,
        retrieval=MagicMock(),
    )
    build_stages = MagicMock(return_value=built)
    get_config = MagicMock(return_value=_config(tmp_path))
    configure_logging = MagicMock()
    load_pipeline_config = MagicMock(return_value={})

    kwargs = {
        "version": "test",
        "get_config_fn": get_config,
        "configure_logging_fn": configure_logging,
        "load_pipeline_config_fn": load_pipeline_config,
        "build_stages_fn": build_stages,
        "watch_controller_cls": MagicMock,
    }
    async with lifespan(owner, **kwargs) as first:
        first.seen_chunk_ids.add("first-only")
        first.pinned_chunk_ids.add("pin-first")
        first.scratchpad = "first"
        first.recent_queries.append("query-first")

        async with lifespan(owner, **kwargs) as second:
            assert first.storage is second.storage is storage
            assert first.embedder is second.embedder is built.embedder
            assert first.search_engine is second.search_engine is built.retrieval
            assert second.seen_chunk_ids == set()
            assert second.pinned_chunk_ids == set()
            assert second.scratchpad == ""
            assert list(second.recent_queries) == []
            second.seen_chunk_ids.add("second-only")

        assert first.seen_chunk_ids == {"first-only"}

    build_stages.assert_called_once_with({})
    get_config.assert_called_once_with()
    configure_logging.assert_called_once_with("INFO")
    load_pipeline_config.assert_called_once_with()
    storage.close.assert_not_called()

    close_shared_runtime(owner)
    storage.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_shared_runtime_allows_clean_rebuild(tmp_path) -> None:
    from stropha.server_runtime import close_shared_runtime, lifespan

    owner = SimpleNamespace()
    first_storage = MagicMock()
    first_storage.list_sessions.return_value = []
    second_storage = MagicMock()
    second_storage.list_sessions.return_value = []
    builds = [
        SimpleNamespace(embedder=MagicMock(), storage=first_storage, retrieval=MagicMock()),
        SimpleNamespace(embedder=MagicMock(), storage=second_storage, retrieval=MagicMock()),
    ]
    build_stages = MagicMock(side_effect=builds)
    kwargs = {
        "version": "test",
        "get_config_fn": lambda: _config(tmp_path),
        "configure_logging_fn": lambda _level: None,
        "load_pipeline_config_fn": lambda: {},
        "build_stages_fn": build_stages,
        "watch_controller_cls": MagicMock,
    }

    async with lifespan(owner, **kwargs):
        pass
    close_shared_runtime(owner)
    async with lifespan(owner, **kwargs):
        pass
    close_shared_runtime(owner)

    assert build_stages.call_count == 2
    first_storage.close.assert_called_once_with()
    second_storage.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_concurrent_sessions_build_and_start_watcher_once(tmp_path) -> None:
    from stropha.server_runtime import close_shared_runtime, lifespan

    owner = SimpleNamespace()
    storage = MagicMock()
    storage.list_sessions.return_value = []
    built = SimpleNamespace(
        embedder=MagicMock(),
        storage=storage,
        retrieval=MagicMock(),
    )
    build_stages = MagicMock(return_value=built)
    watcher = MagicMock()
    watch_controller = MagicMock(return_value=watcher)
    config = _config(tmp_path)
    config.mcp_watch = True
    both_entered = asyncio.Event()
    entered = 0

    async def open_session() -> None:
        nonlocal entered
        async with lifespan(
            owner,
            version="test",
            get_config_fn=lambda: config,
            configure_logging_fn=lambda _level: None,
            load_pipeline_config_fn=lambda: {},
            build_stages_fn=build_stages,
            watch_controller_cls=watch_controller,
        ):
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()

    await asyncio.gather(open_session(), open_session())

    build_stages.assert_called_once_with({})
    watch_controller.assert_called_once_with(
        repo=tmp_path,
        interval_s=1.0,
        debounce_s=1.0,
        full_refresh=False,
    )
    watcher.start.assert_called_once_with()
    watcher.stop.assert_not_called()
    storage.close.assert_not_called()

    close_shared_runtime(owner)
    watcher.stop.assert_called_once_with()
    storage.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_failed_watcher_start_cleans_partial_runtime(tmp_path) -> None:
    from stropha.server_runtime import close_shared_runtime, lifespan

    owner = SimpleNamespace()
    storage = MagicMock()
    built = SimpleNamespace(
        embedder=MagicMock(),
        storage=storage,
        retrieval=MagicMock(),
    )
    watcher = MagicMock()
    watcher.start.side_effect = RuntimeError("watch failed")
    config = _config(tmp_path)
    config.mcp_watch = True

    with pytest.raises(RuntimeError, match="watch failed"):
        async with lifespan(
            owner,
            version="test",
            get_config_fn=lambda: config,
            configure_logging_fn=lambda _level: None,
            load_pipeline_config_fn=lambda: {},
            build_stages_fn=lambda _resolved: built,
            watch_controller_cls=MagicMock(return_value=watcher),
        ):
            pass

    watcher.stop.assert_called_once_with()
    storage.close.assert_called_once_with()
    close_shared_runtime(owner)
    assert not hasattr(owner, "_stropha_shared_runtime_lock")
