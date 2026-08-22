"""Process-level MCP runtime lifecycle tests (WIN-428)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
        first.selected_repository = "github.com/example/first"
        first.agent_state = MagicMock(name="first-agent-state")
        first.runtime_artifacts["first-artifact"] = MagicMock(name="first-artifact")
        first.runtime_artifact_ids_seen.add("first-artifact")

        async with lifespan(owner, **kwargs) as second:
            assert first.storage is second.storage is storage
            assert first.embedder is second.embedder is built.embedder
            assert first.search_engine is second.search_engine is built.retrieval
            assert second.seen_chunk_ids == set()
            assert second.pinned_chunk_ids == set()
            assert second.scratchpad == ""
            assert list(second.recent_queries) == []
            assert second.selected_repository is None
            assert second.agent_state is None
            assert second.runtime_artifacts == {}
            assert second.runtime_artifact_ids_seen == set()
            second.seen_chunk_ids.add("second-only")

        assert first.seen_chunk_ids == {"first-only"}
        assert first.selected_repository == "github.com/example/first"
        assert first.agent_state is not None
        assert set(first.runtime_artifacts) == {"first-artifact"}
        assert first.runtime_artifact_ids_seen == {"first-artifact"}

    assert first.selected_repository is None
    assert first.agent_state is None
    assert first.runtime_artifacts == {}
    assert first.runtime_artifact_ids_seen == set()
    assert all(
        "agent_state" not in call_.kwargs
        and "selected_repository" not in call_.kwargs
        and "runtime_artifacts" not in call_.kwargs
        and "runtime_artifact_ids_seen" not in call_.kwargs
        for call_ in storage.upsert_session.call_args_list
    )

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


@pytest.mark.asyncio
async def test_reload_shared_runtime_swaps_without_closing_active_sessions(tmp_path) -> None:
    from stropha.server_runtime import (
        close_shared_runtime,
        get_shared_runtime,
        reload_shared_runtime,
    )

    owner = SimpleNamespace()
    first_storage = MagicMock(adapter_name="first")
    second_storage = MagicMock(adapter_name="second")
    first = SimpleNamespace(
        embedder=MagicMock(),
        storage=first_storage,
        retrieval=MagicMock(),
    )
    second = SimpleNamespace(
        embedder=MagicMock(),
        storage=second_storage,
        retrieval=MagicMock(),
    )
    builds = MagicMock(side_effect=[first, second])
    first_config = _config(tmp_path)
    second_config = _config(tmp_path)
    second_config.log_level = "DEBUG"
    configs = iter([first_config, second_config])
    configure_logging = MagicMock()
    kwargs = {
        "get_config_fn": lambda: next(configs),
        "configure_logging_fn": configure_logging,
        "load_pipeline_config_fn": lambda: {},
        "build_stages_fn": builds,
        "watch_controller_cls": MagicMock,
    }

    active = await get_shared_runtime(owner, version="test", **kwargs)
    replacement = await reload_shared_runtime(owner, **kwargs)

    assert active.built is first
    assert replacement.built is second
    assert replacement.retired == [first]
    first_storage.close.assert_not_called()
    second_storage.close.assert_not_called()
    configure_logging.assert_has_calls([call("INFO"), call("DEBUG")])

    close_shared_runtime(owner)
    first_storage.close.assert_called_once_with()
    second_storage.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_failed_reload_restarts_previous_watcher(tmp_path) -> None:
    from stropha.server_runtime import (
        close_shared_runtime,
        get_shared_runtime,
        reload_shared_runtime,
    )

    owner = SimpleNamespace()
    first_storage = MagicMock(adapter_name="first")
    second_storage = MagicMock(adapter_name="second")
    first_built = SimpleNamespace(
        embedder=MagicMock(), storage=first_storage, retrieval=MagicMock()
    )
    second_built = SimpleNamespace(
        embedder=MagicMock(), storage=second_storage, retrieval=MagicMock()
    )
    first_watcher = MagicMock()
    second_watcher = MagicMock()
    second_watcher.start.side_effect = RuntimeError("replacement watch failed")
    first_config = _config(tmp_path)
    first_config.mcp_watch = True
    second_config = _config(tmp_path)
    second_config.mcp_watch = True
    configs = iter([first_config, second_config])
    builds = MagicMock(side_effect=[first_built, second_built])
    watcher_cls = MagicMock(side_effect=[first_watcher, second_watcher])
    kwargs = {
        "get_config_fn": lambda: next(configs),
        "configure_logging_fn": lambda _level: None,
        "load_pipeline_config_fn": lambda: {},
        "build_stages_fn": builds,
        "watch_controller_cls": watcher_cls,
    }

    active = await get_shared_runtime(owner, version="test", **kwargs)
    with pytest.raises(RuntimeError, match="replacement watch failed"):
        await reload_shared_runtime(owner, **kwargs)

    assert owner._stropha_shared_runtime is active
    assert active.closed is False
    assert active.watcher is first_watcher
    assert first_watcher.start.call_count == 2
    first_watcher.stop.assert_called_once_with()
    second_watcher.start.assert_called_once_with()
    second_watcher.stop.assert_called_once_with()
    first_storage.close.assert_not_called()
    second_storage.close.assert_called_once_with()

    close_shared_runtime(owner)


@pytest.mark.asyncio
async def test_cancelled_reload_restores_previous_watcher(tmp_path) -> None:
    from stropha.server_runtime import (
        close_shared_runtime,
        get_shared_runtime,
        reload_shared_runtime,
    )

    owner = SimpleNamespace()
    first_storage = MagicMock(adapter_name="first")
    second_storage = MagicMock(adapter_name="second")
    first_built = SimpleNamespace(
        embedder=MagicMock(), storage=first_storage, retrieval=MagicMock()
    )
    second_built = SimpleNamespace(
        embedder=MagicMock(), storage=second_storage, retrieval=MagicMock()
    )
    stop_entered = threading.Event()
    allow_stop = threading.Event()
    first_watcher = MagicMock()

    def blocking_stop() -> None:
        stop_entered.set()
        allow_stop.wait(timeout=2)

    first_watcher.stop.side_effect = blocking_stop
    second_watcher = MagicMock()
    first_config = _config(tmp_path)
    first_config.mcp_watch = True
    second_config = _config(tmp_path)
    second_config.mcp_watch = True
    configs = iter([first_config, second_config])
    builds = MagicMock(side_effect=[first_built, second_built])
    watcher_cls = MagicMock(side_effect=[first_watcher, second_watcher])
    kwargs = {
        "get_config_fn": lambda: next(configs),
        "configure_logging_fn": lambda _level: None,
        "load_pipeline_config_fn": lambda: {},
        "build_stages_fn": builds,
        "watch_controller_cls": watcher_cls,
    }

    active = await get_shared_runtime(owner, version="test", **kwargs)
    reload_task = asyncio.create_task(reload_shared_runtime(owner, **kwargs))
    assert await asyncio.to_thread(stop_entered.wait, 1)
    reload_task.cancel()
    allow_stop.set()

    with pytest.raises(asyncio.CancelledError):
        await reload_task

    assert owner._stropha_shared_runtime is active
    assert active.closed is False
    assert active.watcher is first_watcher
    assert first_watcher.start.call_count == 2
    second_watcher.start.assert_not_called()
    second_storage.close.assert_called_once_with()

    close_shared_runtime(owner)


@pytest.mark.asyncio
async def test_reload_shared_runtime_keeps_event_loop_responsive(tmp_path) -> None:
    from stropha.server_runtime import reload_shared_runtime

    owner = SimpleNamespace()
    storage = MagicMock(adapter_name="test")
    built = SimpleNamespace(
        embedder=MagicMock(),
        storage=storage,
        retrieval=MagicMock(),
    )

    def slow_build(_resolved):
        time.sleep(0.1)
        return built

    task = asyncio.create_task(
        reload_shared_runtime(
            owner,
            get_config_fn=lambda: _config(tmp_path),
            configure_logging_fn=lambda _level: None,
            load_pipeline_config_fn=lambda: {},
            build_stages_fn=slow_build,
            watch_controller_cls=MagicMock,
        )
    )

    await asyncio.sleep(0.02)
    assert not task.done()
    assert await task == owner._stropha_shared_runtime
