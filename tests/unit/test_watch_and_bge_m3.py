"""Tests for `stropha watch` snapshot diffing and bge-m3 adapter registration.

The blocking loop is exercised with injected clocks and wait callbacks so
the reliability tests stay deterministic. The rest covers the snapshot
helpers and BGE-M3 adapter registration.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from stropha.adapters.embedder.bge_m3 import BgeM3Config, BgeM3Embedder
from stropha.errors import PipelineError, StrophaError
from stropha.models import SourceFile
from stropha.pipeline.registry import lookup_adapter
from stropha.watch import (
    WatchController,
    _format_changes,
    _reindex,
    _run_watch_loop,
    _Snapshot,
    _snapshot,
    _WatchState,
)

# --------------------------------------------------------------------------- snapshot diff


def test_snapshot_empty_diff() -> None:
    a = _Snapshot()
    b = _Snapshot()
    changed, removed = a.diff(b)
    assert changed == set()
    assert removed == set()


def test_snapshot_detects_new_files() -> None:
    a = _Snapshot()
    b = _Snapshot(files={Path("/x/a.py"): (1, 10, "a")})
    changed, _removed = a.diff(b)
    assert changed == {Path("/x/a.py")}


def test_snapshot_detects_changed_mtime_ns() -> None:
    a = _Snapshot(files={Path("/x/a.py"): (1, 10, "a")})
    b = _Snapshot(files={Path("/x/a.py"): (2, 10, "a")})
    changed, _removed = a.diff(b)
    assert changed == {Path("/x/a.py")}


def test_snapshot_detects_changed_size_with_same_mtime_ns() -> None:
    a = _Snapshot(files={Path("/x/a.py"): (1, 10, "a")})
    b = _Snapshot(files={Path("/x/a.py"): (1, 11, "a")})
    changed, _removed = a.diff(b)
    assert changed == {Path("/x/a.py")}


def test_snapshot_detects_removed() -> None:
    a = _Snapshot(
        files={Path("/x/a.py"): (1, 10, "a"), Path("/x/b.py"): (1, 10, "b")}
    )
    b = _Snapshot(files={Path("/x/a.py"): (1, 10, "a")})
    changed, removed = a.diff(b)
    assert changed == set()
    assert removed == {Path("/x/b.py")}


def test_snapshot_walks_real_directory(tmp_path: Path) -> None:
    """End-to-end: snapshot picks up real files via the Walker."""
    # The Walker requires a git repo for git-ls-files but falls back to
    # filesystem walk when .git is absent.
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("# hi", encoding="utf-8")
    snap = _snapshot(tmp_path)
    file_names = {p.name for p in snap.files}
    assert "a.py" in file_names
    assert "b.md" in file_names
    path = (tmp_path / "a.py").resolve()
    stat = path.stat()
    assert snap.files[path] == (
        stat.st_mtime_ns,
        stat.st_size,
        "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
    )


def test_snapshot_detects_same_size_edit_with_preserved_mtime(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    before = _snapshot(tmp_path)
    stat = path.stat()
    path.write_text("x = 2\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    changed, removed = before.diff(_snapshot(tmp_path))

    assert changed == {path.resolve()}
    assert removed == set()


def test_snapshot_keeps_distinct_logical_symlink_paths(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("x = 1\n", encoding="utf-8")
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.symlink_to(target)
    second.symlink_to(target)
    walker = MagicMock()
    walker.discover.return_value = [
        SourceFile(
            path=target,
            rel_path="first.py",
            language="python",
            size_bytes=target.stat().st_size,
        ),
        SourceFile(
            path=target,
            rel_path="second.py",
            language="python",
            size_bytes=target.stat().st_size,
        ),
    ]

    snap = _snapshot(tmp_path, walker=walker)

    assert set(snap.files) == {first.absolute(), second.absolute()}


# --------------------------------------------------------------------------- formatting


def test_format_changes_empty() -> None:
    out = _format_changes([], [], root=Path("/x"))
    assert out == "no changes"


def test_format_changes_only_changed() -> None:
    out = _format_changes(
        [Path("/x/a.py"), Path("/x/b.py")],
        [],
        root=Path("/x"),
    )
    assert "2 changed" in out
    assert "a.py" in out


def test_format_changes_only_removed() -> None:
    out = _format_changes(
        [],
        [Path("/x/old.py")],
        root=Path("/x"),
    )
    assert "1 removed" in out


def test_format_changes_caps_long_lists() -> None:
    many = [Path(f"/x/f{i}.py") for i in range(10)]
    out = _format_changes(many, [], root=Path("/x"))
    assert "10 changed" in out
    assert "more" in out  # the "(+N more)" elision


# --------------------------------------------------------------------------- WatchController


def test_watch_loop_reconciles_existing_work_on_startup(tmp_path: Path) -> None:
    snapshot = _Snapshot(files={(tmp_path / "a.py").resolve(): (1, 1, "a")})

    with (
        patch("stropha.watch._snapshot", return_value=snapshot),
        patch("stropha.watch._reindex") as reindex,
    ):
        _run_watch_loop(
            tmp_path,
            interval_s=1.0,
            debounce_s=0.0,
            full_refresh=False,
            wait=lambda _timeout: False,
            once=True,
            now_fn=lambda: 0.0,
        )

    reindex.assert_called_once_with(tmp_path, full_refresh=False)


def test_watch_loop_retries_after_initial_snapshot_failure(tmp_path: Path) -> None:
    snapshot = _Snapshot(files={(tmp_path / "a.py").absolute(): (1, 1, "a")})

    with (
        patch(
            "stropha.watch._snapshot",
            side_effect=[RuntimeError("walker failed"), snapshot],
        ),
        patch("stropha.watch._reindex") as reindex,
    ):
        _run_watch_loop(
            tmp_path,
            interval_s=1.0,
            debounce_s=0.0,
            full_refresh=False,
            wait=lambda _timeout: False,
            once=True,
            now_fn=lambda: 0.0,
        )

    reindex.assert_called_once_with(tmp_path, full_refresh=False)


def test_watch_loop_retries_lock_failure_without_new_event(tmp_path: Path) -> None:
    snapshot = _Snapshot(files={(tmp_path / "a.py").resolve(): (1, 1, "a")})
    clock = 0.0
    finished = False
    attempts: list[float] = []

    def wait(_timeout: float) -> bool:
        nonlocal clock
        if finished:
            return True
        clock += 0.25
        return False

    def reindex(_repo: Path, *, full_refresh: bool) -> None:
        nonlocal finished
        assert full_refresh is False
        attempts.append(clock)
        if len(attempts) == 1:
            raise StrophaError("advisory lock held")
        finished = True

    with (
        patch("stropha.watch._snapshot", return_value=snapshot),
        patch("stropha.watch._reindex", side_effect=reindex),
    ):
        _run_watch_loop(
            tmp_path,
            interval_s=0.25,
            debounce_s=0.0,
            full_refresh=False,
            wait=wait,
            now_fn=lambda: clock,
        )

    assert attempts == [0.25, 1.25]


def test_watch_state_uses_bounded_exponential_backoff() -> None:
    state = _WatchState.starting(_Snapshot(), now=0.0)

    delays = [state.mark_failed(now=0.0) for _ in range(7)]

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_watch_state_merges_events_while_retry_is_pending() -> None:
    path = Path("/x/a.py")
    state = _WatchState.starting(_Snapshot(files={path: (1, 1, "a")}), now=0.0)
    assert state.mark_failed(now=0.0) == 1.0

    changed, removed = state.observe(_Snapshot(files={path: (2, 2, "b")}), now=0.5)

    assert changed == {path}
    assert removed == set()
    assert state.pending_since == 0.5
    assert state.retry_at == 1.0
    assert state.is_ready(now=0.99, debounce_s=0.25) is False
    assert state.is_ready(now=1.0, debounce_s=0.25) is True


def test_watch_controller_starts_and_stops(tmp_path: Path) -> None:
    """WatchController can be started and stopped cleanly."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    ctrl = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
    ctrl.start()
    assert ctrl._thread is not None
    assert ctrl._thread.is_alive()
    ctrl.stop(timeout=2.0)
    assert not ctrl._thread.is_alive()


def test_watch_controller_double_start_is_noop(tmp_path: Path) -> None:
    """Starting twice doesn't spawn two threads."""
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    ctrl = WatchController(repo=tmp_path, interval_s=0.1, debounce_s=0.1)
    ctrl.start()
    first_thread = ctrl._thread
    ctrl.start()  # Should be no-op
    assert ctrl._thread is first_thread
    ctrl.stop(timeout=2.0)


def test_watch_controller_stop_without_start_is_noop(tmp_path: Path) -> None:
    """Stopping without starting doesn't raise."""
    ctrl = WatchController(repo=tmp_path)
    ctrl.stop()  # Should not raise


@pytest.mark.parametrize("full_refresh", [False, True])
def test_reindex_only_runs_graph_lifecycle_for_full_refresh(
    tmp_path: Path, full_refresh: bool
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
        chunks_seen=1,
        chunks_embedded=1,
        chunks_skipped_fresh=0,
        chunks_enriched_failed=0,
    )

    with (
        patch("stropha.watch.load_pipeline_config", return_value={}),
        patch("stropha.watch.build_stages", return_value=built),
        patch(
            "stropha.watch.get_config",
            return_value=SimpleNamespace(inference=inference),
        ),
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
        chunks_seen=1,
        chunks_embedded=0,
        chunks_skipped_fresh=0,
        chunks_enriched_failed=2,
    )

    with (
        patch("stropha.watch.load_pipeline_config", return_value={}),
        patch("stropha.watch.build_stages", return_value=built),
        patch(
            "stropha.watch.get_config",
            return_value=SimpleNamespace(inference=inference),
        ),
        patch("stropha.watch.Pipeline") as pipeline_cls,
        pytest.raises(PipelineError, match="2 chunks failed enrichment"),
    ):
        pipeline_cls.return_value.run.return_value = stats
        _reindex(tmp_path, full_refresh=False)


# --------------------------------------------------------------------------- bge-m3 registry


def test_bge_m3_registered_in_adapter_registry() -> None:
    cls = lookup_adapter("embedder", "bge-m3")
    assert cls is BgeM3Embedder


def test_bge_m3_default_model_is_bge_m3() -> None:
    cfg = BgeM3Config()
    assert cfg.model == "BAAI/bge-m3"


def test_bge_m3_adapter_id_prefix() -> None:
    """We don't load the real model in tests — just check the class
    declares the correct adapter_name without instantiating."""
    assert BgeM3Embedder.__name__ == "BgeM3Embedder"
    # adapter_name is a @property on instances; check via class-level
    # attribute access by reading the source contract.
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
