"""Tests for native saved-worktree watching and BGE-M3 registration."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from watchfiles import Change

from stropha.adapters.embedder.bge_m3 import BgeM3Config, BgeM3Embedder
from stropha.errors import PipelineError, StrophaError
from stropha.pipeline.registry import lookup_adapter
from stropha.stages.embedder import SynchronizedEmbedder
from stropha.watch import (
    WatchController,
    _classify_changes,
    _PendingRefresh,
    _reindex,
    _run_resilient_watch_loop,
    _run_watch_loop,
    _watch_filter,
)


def test_classify_changes_uses_final_filesystem_state(tmp_path: Path) -> None:
    changed = tmp_path / "changed.py"
    changed.write_text("x = 1\n", encoding="utf-8")
    deleted = tmp_path / "deleted.py"

    dirty, removed, reconcile = _classify_changes(
        tmp_path,
        {
            (Change.deleted, str(changed)),
            (Change.modified, str(deleted)),
        },
    )

    assert dirty == {"changed.py"}
    assert removed == {"deleted.py"}
    assert reconcile is False


@pytest.mark.parametrize("name", [".gitignore", ".strophaignore", ".ragignore"])
def test_classify_changes_reconciles_admission_rule_edits(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / name
    path.write_text("generated/\n", encoding="utf-8")

    _changed, _deleted, reconcile = _classify_changes(tmp_path, {(Change.modified, str(path))})

    assert reconcile is True


def test_classify_changes_reconciles_current_directories(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()

    changed, deleted, reconcile = _classify_changes(tmp_path, {(Change.added, str(package))})

    assert changed == set()
    assert deleted == set()
    assert reconcile is True


def test_classify_changes_ignores_unrelated_git_internal_and_outside_paths(
    tmp_path: Path,
) -> None:
    git_object = tmp_path / ".git" / "objects" / "pack" / "data"
    outside = tmp_path.parent / "outside.py"

    changed, deleted, reconcile = _classify_changes(
        tmp_path,
        {
            (Change.modified, str(git_object)),
            (Change.modified, str(outside)),
        },
    )

    assert changed == set()
    assert deleted == set()
    assert reconcile is False


@pytest.mark.parametrize(
    "rel_path",
    [
        ".git/index",
        ".git/config",
        ".git/info/exclude",
        "nested/.git",
        "nested/.git/index",
    ],
)
def test_git_admission_events_require_full_reconcile(tmp_path: Path, rel_path: str) -> None:
    path = tmp_path / rel_path

    changed, deleted, reconcile = _classify_changes(
        tmp_path,
        {(Change.modified, str(path))},
    )

    assert changed == set()
    assert deleted == set()
    assert reconcile is True
    assert _watch_filter(tmp_path)(Change.modified, str(path)) is True


def test_watch_loop_uses_native_backend_and_reconciles_on_startup(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    refresh = MagicMock()

    def fake_watch(*paths, **kwargs):
        captured["paths"] = paths
        captured.update(kwargs)
        yield set()

    _run_watch_loop(
        tmp_path,
        interval_s=0.25,
        debounce_s=1.5,
        full_refresh=False,
        stop_event=threading.Event(),
        refresh=refresh,
        once=True,
        watch_fn=fake_watch,
    )

    assert captured["paths"] == (tmp_path,)
    assert captured["force_polling"] is None
    assert captured["yield_on_timeout"] is True
    assert captured["stop_event"] is not None
    assert captured["debounce"] == 50
    assert captured["step"] == 50
    watch_filter = captured["watch_filter"]
    assert callable(watch_filter)
    assert watch_filter(Change.modified, str(tmp_path / ".venv" / "tracked.py"))
    assert watch_filter(Change.modified, str(tmp_path / ".git" / "index"))
    assert not watch_filter(Change.modified, str(tmp_path / ".git" / "objects" / "data"))
    refresh.assert_called_once_with(set(), set(), full_reconcile=True)


def test_watch_loop_emits_only_dirty_paths_after_startup(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")
    refresh = MagicMock()

    def fake_watch(*_paths, **_kwargs):
        yield set()
        yield {(Change.modified, str(source))}

    _run_watch_loop(
        tmp_path,
        interval_s=0.1,
        debounce_s=0,
        full_refresh=False,
        stop_event=threading.Event(),
        refresh=refresh,
        watch_fn=fake_watch,
    )

    assert refresh.call_args_list == [
        ((set(), set()), {"full_reconcile": True}),
        (({"module.py"}, set()), {"full_reconcile": False}),
    ]


def test_watch_loop_applies_configured_debounce_after_last_change(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")
    clock = 0.0
    refresh = MagicMock()

    def now() -> float:
        return clock

    def fake_watch(*_paths, **_kwargs):
        nonlocal clock
        yield set()
        clock = 0.1
        yield {(Change.modified, str(source))}
        clock = 1.0
        yield set()
        clock = 1.6
        yield set()

    _run_watch_loop(
        tmp_path,
        interval_s=0.1,
        debounce_s=1.5,
        full_refresh=False,
        stop_event=threading.Event(),
        refresh=refresh,
        watch_fn=fake_watch,
        now_fn=now,
    )

    assert refresh.call_args_list == [
        ((set(), set()), {"full_reconcile": True}),
        (({"module.py"}, set()), {"full_reconcile": False}),
    ]


def test_watch_loop_retries_without_another_filesystem_event(tmp_path: Path) -> None:
    clock = 0.0
    attempts: list[float] = []

    def now() -> float:
        return clock

    def refresh(
        _changed: set[str],
        _deleted: set[str],
        *,
        full_reconcile: bool,
    ) -> None:
        assert full_reconcile is True
        attempts.append(clock)
        if len(attempts) == 1:
            raise StrophaError("advisory lock held")

    def fake_watch(*_paths, **_kwargs):
        nonlocal clock
        yield set()
        clock = 0.5
        yield set()
        clock = 1.0
        yield set()

    _run_watch_loop(
        tmp_path,
        interval_s=0.1,
        debounce_s=0.1,
        full_refresh=False,
        stop_event=threading.Event(),
        refresh=refresh,
        once=True,
        watch_fn=fake_watch,
        now_fn=now,
    )

    assert attempts == [0.0, 1.0]


def test_pending_refresh_uses_bounded_exponential_backoff() -> None:
    pending = _PendingRefresh(full_reconcile=True)

    delays = [pending.mark_failed(now=0.0) for _ in range(7)]

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_pending_refresh_full_reconcile_supersedes_paths() -> None:
    pending = _PendingRefresh(changed={"a.py"}, deleted={"old.py"})

    pending.merge({"b.py"}, {"gone.py"}, full_reconcile=True)

    assert pending.full_reconcile is True
    assert pending.changed == set()
    assert pending.deleted == set()


def test_watch_controller_starts_and_stops(tmp_path: Path) -> None:
    def fake_watch(*_paths, **kwargs):
        stop_event = kwargs["stop_event"]
        yield set()
        while not stop_event.wait(0.01):
            yield set()

    with (
        patch("stropha.watch.watch", side_effect=fake_watch),
        patch("stropha.watch._reindex"),
    ):
        controller = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
        controller.start()
        assert controller._thread is not None
        assert controller._thread.is_alive()
        controller.stop(timeout=2.0)

    assert not controller._thread.is_alive()


def test_watch_controller_surfaces_startup_failure(tmp_path: Path) -> None:
    def broken_watch(*_paths, **_kwargs):
        raise OSError("native backend failed")
        yield set()

    with patch("stropha.watch.watch", side_effect=broken_watch):
        controller = WatchController(repo=tmp_path)
        with pytest.raises(StrophaError, match="native backend failed"):
            controller.start()


def test_watch_controller_startup_is_not_delayed_by_debounce(tmp_path: Path) -> None:
    with patch("stropha.watch._reindex"):
        controller = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=6.0)
        started_at = time.monotonic()
        controller.start()
        elapsed = time.monotonic() - started_at
        controller.stop(timeout=2.0)

    assert elapsed < 1.0


def test_resilient_watch_loop_restarts_backend_after_runtime_failure(tmp_path: Path) -> None:
    attempts = 0
    stop_event = threading.Event()

    def flaky_watch(*_paths, **_kwargs):
        nonlocal attempts
        attempts += 1
        yield set()
        if attempts == 1:
            raise OSError("backend disconnected")
        stop_event.set()
        yield set()

    with patch("stropha.watch._RETRY_INITIAL_S", 0):
        _run_resilient_watch_loop(
            tmp_path,
            interval_s=0.1,
            debounce_s=0.1,
            full_refresh=False,
            stop_event=stop_event,
            refresh=MagicMock(),
            watch_fn=flaky_watch,
        )

    assert attempts == 2


def test_watch_controller_restarts_backend_after_runtime_failure(tmp_path: Path) -> None:
    attempts = 0
    restarted = threading.Event()

    def flaky_watch(*_paths, **kwargs):
        nonlocal attempts
        attempts += 1
        stop_event = kwargs["stop_event"]
        yield set()
        if attempts == 1:
            raise OSError("backend disconnected")
        restarted.set()
        while not stop_event.wait(0.01):
            yield set()

    with (
        patch("stropha.watch.watch", side_effect=flaky_watch),
        patch("stropha.watch._reindex"),
        patch("stropha.watch._RETRY_INITIAL_S", 0.01),
        patch("stropha.watch._RETRY_MAX_S", 0.01),
    ):
        controller = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
        controller.start()
        assert restarted.wait(timeout=1)
        assert controller._thread is not None
        assert controller._thread.is_alive()
        controller.stop(timeout=1)

    assert attempts == 2


def test_watch_controller_owns_writer_storage_and_reuses_warm_stages(
    tmp_path: Path,
) -> None:
    stages = SimpleNamespace(
        resolved={"storage": {"adapter": "test"}},
        embedder=SimpleNamespace(dim=3),
        secondary_embedder=None,
        enricher=SimpleNamespace(),
    )
    writer_storage = MagicMock()
    pipeline = MagicMock()
    refreshed = threading.Event()
    worker_thread_ids: list[int] = []

    def build_writer(*_args, **_kwargs):
        worker_thread_ids.append(threading.get_ident())
        return writer_storage

    def run_loop(_repo, **kwargs):
        kwargs["on_ready"]()
        kwargs["refresh"]({"src/app.py"}, set(), full_reconcile=False)
        refreshed.set()
        kwargs["stop_event"].wait(timeout=1)

    with (
        patch("stropha.watch.build_storage", side_effect=build_writer) as build_storage,
        patch("stropha.watch._pipeline_from_stages", return_value=pipeline) as make_pipeline,
        patch("stropha.watch._refresh_pipeline") as refresh_pipeline,
        patch("stropha.watch._run_watch_loop", side_effect=run_loop),
    ):
        controller = WatchController(
            tmp_path,
            stages=stages,
            enricher_concurrency=2,
            enricher_batch_size=4,
        )
        controller.start()
        assert refreshed.wait(timeout=1)
        controller.stop(timeout=1)

    assert controller._thread is not None
    assert worker_thread_ids == [controller._thread.ident]
    assert worker_thread_ids[0] != threading.get_ident()
    build_storage.assert_called_once_with(stages.resolved, embedding_dim=3)
    make_pipeline.assert_called_once_with(
        tmp_path.resolve(),
        stages,
        storage=writer_storage,
        enricher=stages.enricher,
        graphify=False,
        enricher_concurrency=2,
        enricher_batch_size=4,
    )
    refresh_pipeline.assert_called_once_with(
        pipeline,
        tmp_path.resolve(),
        {"src/app.py"},
        set(),
        full_reconcile=False,
    )
    writer_storage.close.assert_called_once_with()


def test_synchronized_embedder_serializes_query_and_document_calls() -> None:
    delegate = MagicMock()
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    start = threading.Barrier(3)

    def embed(*_args):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with active_lock:
            active -= 1
        return [1.0]

    delegate.embed_query.side_effect = embed
    delegate.embed_documents.side_effect = embed
    embedder = SynchronizedEmbedder(delegate)

    def embed_query() -> None:
        start.wait()
        embedder.embed_query("query")

    def embed_documents() -> None:
        start.wait()
        embedder.embed_documents(["document"])

    threads = [
        threading.Thread(target=embed_query),
        threading.Thread(target=embed_documents),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=1)

    assert max_active == 1
    assert all(not thread.is_alive() for thread in threads)


def test_watch_controller_double_start_is_noop(tmp_path: Path) -> None:
    def fake_watch(*_paths, **kwargs):
        stop_event = kwargs["stop_event"]
        yield set()
        while not stop_event.wait(0.01):
            yield set()

    with (
        patch("stropha.watch.watch", side_effect=fake_watch),
        patch("stropha.watch._reindex"),
    ):
        controller = WatchController(repo=tmp_path)
        controller.start()
        first_thread = controller._thread
        controller.start()
        assert controller._thread is first_thread
        controller.stop(timeout=2.0)


def test_watch_controller_stop_without_start_is_noop(tmp_path: Path) -> None:
    WatchController(repo=tmp_path).stop()


@pytest.mark.parametrize("full_refresh", [False, True])
def test_reindex_only_runs_graph_lifecycle_for_full_refresh(
    tmp_path: Path,
    full_refresh: bool,
) -> None:
    storage = MagicMock()
    built = SimpleNamespace(
        storage=storage,
        embedder=MagicMock(),
        secondary_embedder=None,
        enricher=MagicMock(),
        walker=MagicMock(),
        chunker=MagicMock(),
    )
    inference = SimpleNamespace(
        enricher_concurrency=1,
        effective_batch_size=lambda: 1,
    )
    stats = SimpleNamespace(
        files_visited=1,
        files_evicted=0,
        chunks_seen=1,
        chunks_embedded=1,
        chunks_skipped_fresh=0,
        chunks_enriched_failed=0,
    )

    with (
        patch("stropha.watch.load_pipeline_config", return_value={}),
        patch("stropha.watch.build_stages", return_value=built),
        patch("stropha.watch.get_config", return_value=SimpleNamespace(inference=inference)),
        patch("stropha.watch.Pipeline") as pipeline_cls,
    ):
        pipeline_cls.return_value.run.return_value = stats
        _reindex(tmp_path, full_refresh=full_refresh)

    assert pipeline_cls.call_args.kwargs["graphify"] is full_refresh


def test_reindex_rejects_incomplete_pipeline_stats(tmp_path: Path) -> None:
    storage = MagicMock()
    built = SimpleNamespace(
        storage=storage,
        embedder=MagicMock(),
        secondary_embedder=None,
        enricher=MagicMock(),
        walker=MagicMock(),
        chunker=MagicMock(),
    )
    inference = SimpleNamespace(
        enricher_concurrency=1,
        effective_batch_size=lambda: 1,
    )
    stats = SimpleNamespace(
        files_visited=1,
        files_evicted=0,
        chunks_seen=1,
        chunks_embedded=0,
        chunks_skipped_fresh=0,
        chunks_enriched_failed=2,
    )

    with (
        patch("stropha.watch.load_pipeline_config", return_value={}),
        patch("stropha.watch.build_stages", return_value=built),
        patch("stropha.watch.get_config", return_value=SimpleNamespace(inference=inference)),
        patch("stropha.watch.Pipeline") as pipeline_cls,
        pytest.raises(PipelineError, match="2 chunks failed enrichment"),
    ):
        pipeline_cls.return_value.run.return_value = stats
        _reindex(tmp_path, full_refresh=False)


def test_bge_m3_registered_in_adapter_registry() -> None:
    cls = lookup_adapter("embedder", "bge-m3")
    assert cls is BgeM3Embedder


def test_bge_m3_default_model_is_bge_m3() -> None:
    cfg = BgeM3Config()
    assert cfg.model == "BAAI/bge-m3"


def test_bge_m3_adapter_id_prefix() -> None:
    import inspect

    src = inspect.getsource(BgeM3Embedder.adapter_name.fget)  # type: ignore[union-attr]
    assert '"bge-m3"' in src


def test_bge_m3_delegates_to_mlx_registry_alias() -> None:
    with (
        patch("stropha.adapters.embedder.bge_m3._is_mlx_available", return_value=True),
        patch("stropha.adapters.embedder.mlx.MlxEmbedder.__init__", return_value=None) as init,
    ):
        BgeM3Embedder(BgeM3Config(batch_size=12))

    config = init.call_args.args[0]
    assert config.model == "bge-m3"
    assert config.batch_size == 12


def test_bge_m3_explains_platform_dependency() -> None:
    from stropha.errors import EmbeddingError

    with (
        patch("stropha.adapters.embedder.bge_m3._is_mlx_available", return_value=False),
        pytest.raises(EmbeddingError, match="not supported by FastEmbed"),
    ):
        BgeM3Embedder()


def _fake_watch_factory(seen: list) -> Callable:
    def fake_watch(*paths, **kwargs):
        seen.append(tuple(str(p) for p in paths))
        stop_event = kwargs["stop_event"]
        yield set()
        while not stop_event.wait(0.01):
            yield set()

    return fake_watch


def test_watch_controller_observes_every_configured_root(tmp_path: Path) -> None:
    """`indexing.watch_all_repos` gives each checkout its own observer."""
    primary = tmp_path / "primary"
    second = tmp_path / "second"
    third = tmp_path / "third"
    for path in (primary, second, third):
        path.mkdir()
    seen: list = []

    with (
        patch("stropha.watch.watch", side_effect=_fake_watch_factory(seen)),
        patch("stropha.watch._reindex"),
    ):
        controller = WatchController(
            repo=primary,
            additional_repos=(second, third),
            interval_s=0.1,
            debounce_s=0.1,
        )
        controller.start()
        assert len(controller._extra_threads) == 2
        assert all(thread.is_alive() for thread in controller._extra_threads)
        # start() deliberately does not block on secondary roots, so wait for
        # them to reach the backend instead of assuming they already have.
        deadline = time.monotonic() + 5.0
        while len(seen) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        controller.stop(timeout=5.0)

    watched = {entry[0] for entry in seen}
    assert watched == {str(primary.resolve()), str(second.resolve()), str(third.resolve())}
    assert not any(thread.is_alive() for thread in controller._extra_threads)


def test_watch_controller_skips_missing_and_duplicate_roots(tmp_path: Path) -> None:
    """A repository can outlive its directory; listing it twice is harmless."""
    primary = tmp_path / "primary"
    primary.mkdir()
    gone = tmp_path / "deleted-checkout"

    with (
        patch("stropha.watch.watch", side_effect=_fake_watch_factory([])),
        patch("stropha.watch._reindex"),
    ):
        controller = WatchController(
            repo=primary,
            additional_repos=(primary, gone, primary / "."),
            interval_s=0.1,
            debounce_s=0.1,
        )
        controller.start()
        assert controller._extra_threads == []
        controller.stop(timeout=5.0)


def test_a_failing_secondary_root_does_not_break_the_primary(tmp_path: Path) -> None:
    """One unreadable checkout must not stop the daemon's own repository."""
    primary = tmp_path / "primary"
    second = tmp_path / "second"
    primary.mkdir()
    second.mkdir()

    def selective_watch(*paths, **kwargs):
        if str(paths[0]) == str(second.resolve()):
            raise OSError("native backend failed for the secondary root")
        stop_event = kwargs["stop_event"]
        yield set()
        while not stop_event.wait(0.01):
            yield set()

    with (
        patch("stropha.watch.watch", side_effect=selective_watch),
        patch("stropha.watch._reindex"),
    ):
        controller = WatchController(
            repo=primary,
            additional_repos=(second,),
            interval_s=0.1,
            debounce_s=0.1,
        )
        controller.start()  # must not raise
        assert controller._thread is not None and controller._thread.is_alive()
        controller.stop(timeout=5.0)


def test_watch_controller_serializes_writes_across_roots(tmp_path: Path) -> None:
    """The index admits one writer, so refreshes must never overlap."""
    primary = tmp_path / "primary"
    second = tmp_path / "second"
    primary.mkdir()
    second.mkdir()
    overlaps: list[int] = []
    active = 0
    guard = threading.Lock()

    def slow_reindex(*_args, **_kwargs):
        nonlocal active
        with guard:
            active += 1
            overlaps.append(active)
        time.sleep(0.05)
        with guard:
            active -= 1

    def busy_watch(*_paths, **kwargs):
        stop_event = kwargs["stop_event"]
        yield {(Change.modified, "file.py")}
        while not stop_event.wait(0.01):
            yield {(Change.modified, "file.py")}

    with (
        patch("stropha.watch.watch", side_effect=busy_watch),
        patch("stropha.watch._reindex", side_effect=slow_reindex),
    ):
        controller = WatchController(
            repo=primary,
            additional_repos=(second,),
            interval_s=0.01,
            debounce_s=0.01,
        )
        controller.start()
        time.sleep(0.4)
        controller.stop(timeout=5.0)

    assert overlaps, "expected at least one refresh"
    assert max(overlaps) == 1, f"writers overlapped: {overlaps}"
