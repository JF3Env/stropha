"""Unit tests for the meta-dir pruning policy (WIN-166).

Covers:
- ``disk_usage`` totals + per-bucket breakdown.
- Age-based trace pruning (real vs dry-run).
- Candidate compaction delegation + dry-run preview parity.
- CLI ``stropha meta prune`` smoke (typer registration).
- ``aggregate(... , meta_dir=...)`` populates ``CostReport.meta_disk``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from stropha.meta import (
    DEFAULT_TRACE_MAX_AGE_DAYS,
    ArtifactStore,
    PrunePlan,
    disk_usage,
    prune,
)
from stropha.meta.prune import _compact_preview, _stale_trace_files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch_old(path: Path, days_ago: int) -> None:
    """Set both atime and mtime of ``path`` to ``days_ago`` ago."""
    target = time.time() - days_ago * 86400
    os.utime(path, (target, target))


def _make_meta(tmp_path: Path) -> Path:
    """Spin up a meta dir with a candidate, two traces, and an outcome."""
    meta = tmp_path / "meta"
    store = ArtifactStore(meta)
    rec = store.register_candidate({"v": 1})
    store.write_scores(rec.id, {"score": 0.5})
    traces = meta / "traces"
    traces.mkdir(exist_ok=True)
    (traces / "trace-old.jsonl").write_text(json.dumps({"tool": "x"}) + "\n")
    (traces / "trace-new.jsonl").write_text(json.dumps({"tool": "y"}) + "\n")
    outcomes = meta / "outcomes"
    outcomes.mkdir(exist_ok=True)
    (outcomes / "abc.json").write_text(json.dumps({"head": "abc"}))
    return meta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_age_is_90_days(self) -> None:
        assert DEFAULT_TRACE_MAX_AGE_DAYS == 90


# ---------------------------------------------------------------------------
# disk_usage
# ---------------------------------------------------------------------------


class TestDiskUsage:
    def test_missing_dir_returns_zeros(self, tmp_path: Path) -> None:
        out = disk_usage(tmp_path / "nope")
        assert out == {"total": 0, "traces": 0, "outcomes": 0, "candidates": 0}

    def test_breaks_down_by_subdir(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        u = disk_usage(meta)
        assert u["traces"] > 0
        assert u["outcomes"] > 0
        assert u["candidates"] > 0
        assert u["total"] >= u["traces"] + u["outcomes"] + u["candidates"]

    def test_skips_files_that_vanish_mid_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``stat()`` raises mid-walk, ``_dir_bytes`` keeps going.

        Targets stat() only on *file* entries with names ending in
        ``.jsonl`` (avoids breaking ``is_dir`` checks that also call stat).
        """
        meta = _make_meta(tmp_path)
        original_stat = Path.stat

        def flaky_stat(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.suffix == ".jsonl":
                raise OSError("simulated mid-walk vanish")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", flaky_stat)
        # Some files fail stat; the others succeed → walk completes without raising.
        out = disk_usage(meta)
        assert out["total"] >= 0  # didn't raise

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        out = disk_usage(str(meta))
        assert out["total"] > 0


# ---------------------------------------------------------------------------
# Stale-trace helper
# ---------------------------------------------------------------------------


class TestStaleTraceFiles:
    def test_no_traces_dir_returns_empty(self, tmp_path: Path) -> None:
        assert _stale_trace_files(tmp_path / "missing", 30) == []

    def test_only_stale_files_returned(self, tmp_path: Path) -> None:
        traces = tmp_path / "traces"
        traces.mkdir()
        old = traces / "old.jsonl"
        new = traces / "new.jsonl"
        old.write_text("x\n")
        new.write_text("y\n")
        _touch_old(old, days_ago=120)
        # Now: 90-day cutoff returns old, not new.
        stale = _stale_trace_files(traces, max_age_days=90)
        assert old in stale
        assert new not in stale

    def test_age_zero_returns_everything(self, tmp_path: Path) -> None:
        traces = tmp_path / "traces"
        traces.mkdir()
        for name in ("a.jsonl", "b.jsonl"):
            (traces / name).write_text("x\n")
        stale = _stale_trace_files(traces, max_age_days=0)
        assert len(stale) == 2

    def test_vanished_file_tolerated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "ghost.jsonl").write_text("x\n")

        original_stat = Path.stat

        def gone(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.name == "ghost.jsonl":
                raise OSError("vanished")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", gone)
        # Must not raise — vanished file is skipped.
        result = _stale_trace_files(traces, max_age_days=0)
        assert result == []


# ---------------------------------------------------------------------------
# prune — dry-run vs real
# ---------------------------------------------------------------------------


class TestPruneDryRun:
    def test_dry_run_does_not_touch_disk(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        old_trace = meta / "traces" / "trace-old.jsonl"
        _touch_old(old_trace, days_ago=120)

        plan = prune(meta, max_age_days=90, dry_run=True)
        # Plan reports what WOULD be removed.
        assert "traces/trace-old.jsonl" in plan.traces_to_remove
        # But the file is still there.
        assert old_trace.is_file()

    def test_dry_run_populates_byte_totals(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        plan = prune(meta, max_age_days=90, dry_run=True)
        assert plan.total_meta_bytes > 0
        assert plan.traces_dir_bytes > 0

    def test_dry_run_compaction_preview(self, tmp_path: Path) -> None:
        """The dry-run plan must list candidates that compact() would prune."""
        meta = tmp_path / "meta"
        store = ArtifactStore(meta)
        for i in range(5):
            rec = store.register_candidate({"v": i})
            store.write_scores(rec.id, {"score": float(i)})
        plan = prune(meta, keep_top=1, keep_bottom=0, keep_recent=0, dry_run=True)
        assert len(plan.candidates_to_prune) == 4  # top-1 keeps 1, prunes 4
        # No mutation yet — ledger still has all 5 entries live.
        live = store.list_candidates(include_pruned=False)
        assert len(live) == 5


class TestPruneReal:
    def test_removes_stale_trace_files(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        old = meta / "traces" / "trace-old.jsonl"
        _touch_old(old, days_ago=120)

        plan = prune(meta, max_age_days=90, dry_run=False)
        assert not old.exists(), "stale trace was not removed"
        assert plan.bytes_freed > 0

    def test_keeps_fresh_trace_files(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        fresh = meta / "traces" / "trace-new.jsonl"
        plan = prune(meta, max_age_days=90, dry_run=False)
        assert fresh.is_file(), "fresh trace was incorrectly removed"
        assert "traces/trace-new.jsonl" not in plan.traces_to_remove

    def test_prunes_candidates_per_keep_buckets(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta"
        store = ArtifactStore(meta)
        ids = []
        for i in range(5):
            rec = store.register_candidate({"v": i})
            store.write_scores(rec.id, {"score": float(i)})
            ids.append(rec.id)
        plan = prune(meta, keep_top=2, keep_bottom=1, keep_recent=1, dry_run=False)
        # Pruned 2 candidates (out of 5). Survivors are top-2, bottom-1, recent-1.
        assert len(plan.candidates_to_prune) == 2
        # The pruned candidate dirs are gone.
        for cid in plan.candidates_to_prune:
            assert not (meta / "candidates" / cid).exists()

    def test_default_max_age_constant(self, tmp_path: Path) -> None:
        """Without --max-age-days, defaults to the 90-day constant."""
        meta = _make_meta(tmp_path)
        plan = prune(meta, dry_run=True)
        assert plan.max_age_days == 90

    def test_missing_meta_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="meta dir"):
            prune(tmp_path / "missing")

    def test_unlink_failure_logged_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``Path.unlink`` raises OSError, prune logs + continues."""
        meta = _make_meta(tmp_path)
        old = meta / "traces" / "trace-old.jsonl"
        _touch_old(old, days_ago=120)

        original_unlink = Path.unlink

        def boom(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.name == "trace-old.jsonl":
                raise OSError("simulated permission denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", boom)
        # Must not raise — failure is logged at warn level.
        prune(meta, max_age_days=90, dry_run=False)

    def test_no_candidates_dir_skipped(self, tmp_path: Path) -> None:
        """A meta dir with only traces (no candidates/) doesn't crash."""
        meta = tmp_path / "meta"
        meta.mkdir()
        (meta / "traces").mkdir()
        plan = prune(meta, dry_run=True)
        assert plan.candidates_to_prune == []

    def test_stale_file_size_lookup_failure_tolerated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``path.stat()`` fails for a stale file we marked for removal,
        the loop records it with size=0 — never raise.

        Strategy: stub ``_stale_trace_files`` via ``sys.modules`` so the
        test controls the returned list directly, then install a
        stat-failure mock right before ``prune`` runs the loop.

        (``stropha.meta.prune`` resolves to the re-exported function, not
        the module — the package namespace collision means monkeypatch
        with dotted paths cannot reach the helper. ``sys.modules`` is the
        unambiguous handle.)
        """
        import sys

        meta = _make_meta(tmp_path)
        old = meta / "traces" / "trace-old.jsonl"
        prune_mod = sys.modules["stropha.meta.prune"]

        monkeypatch.setattr(
            prune_mod, "_stale_trace_files", lambda *_a, **_kw: [old]
        )
        original_stat = Path.stat

        def flaky_stat(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self == old:
                raise OSError("simulated vanish between iter and stat")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", flaky_stat)
        plan = prune_mod.prune(meta, max_age_days=90, dry_run=True)
        # Despite the stat failure, the file is recorded for removal
        # (with size=0 implicit in bytes_freed).
        assert "traces/trace-old.jsonl" in plan.traces_to_remove


# ---------------------------------------------------------------------------
# Compaction preview
# ---------------------------------------------------------------------------


class TestCompactPreview:
    def test_preview_matches_compact_output(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta"
        store_dry = ArtifactStore(meta)
        for i in range(5):
            rec = store_dry.register_candidate({"v": i})
            store_dry.write_scores(rec.id, {"score": float(i)})

        # Compute the dry preview.
        preview_ids = set(_compact_preview(store_dry, 2, 1, 1))

        # Compact for real on a fresh store with the same contents — but
        # since we already populated `meta`, just call compact() on the
        # SAME store. The two should agree.
        real_pruned = set(
            store_dry.compact(keep_top=2, keep_bottom=1, keep_recent=1)
        )
        assert preview_ids == real_pruned

    def test_preview_handles_empty_store(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        assert _compact_preview(store, 10, 10, 50) == []


# ---------------------------------------------------------------------------
# CostReport.meta_disk
# ---------------------------------------------------------------------------


class TestCostReportMetaDisk:
    def test_meta_disk_populated_when_meta_dir_passed(self, tmp_path: Path) -> None:
        from stropha.cost import aggregate

        meta = _make_meta(tmp_path)
        # No log files — just the meta-dir reporting path.
        report = aggregate(paths=[], meta_dir=meta)
        assert report.meta_disk.total_bytes > 0
        assert report.meta_disk.traces_bytes > 0
        assert report.meta_disk.meta_dir == str(meta)

    def test_meta_disk_zero_when_meta_dir_omitted(self, tmp_path: Path) -> None:
        from stropha.cost import aggregate

        report = aggregate(paths=[])
        assert report.meta_disk.total_bytes == 0
        assert report.meta_disk.meta_dir is None

    def test_meta_disk_zero_when_dir_missing(self, tmp_path: Path) -> None:
        from stropha.cost import aggregate

        # Pass a non-existent path; populate stays zero.
        report = aggregate(paths=[], meta_dir=tmp_path / "nope")
        assert report.meta_disk.total_bytes == 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_meta_prune_command_registered(self) -> None:
        from stropha.cli import app

        meta_app = None
        for group in app.registered_groups:
            if group.name == "meta":
                meta_app = group.typer_instance
                break
        assert meta_app is not None
        commands = {cmd.name for cmd in meta_app.registered_commands}
        assert "prune" in commands


# ---------------------------------------------------------------------------
# PrunePlan dataclass
# ---------------------------------------------------------------------------


class TestPrunePlan:
    def test_to_dict_contains_every_field(self) -> None:
        plan = PrunePlan(
            meta_dir="/x",
            dry_run=True,
            max_age_days=90,
            keep_top=10,
            keep_bottom=5,
            keep_recent=50,
            traces_to_remove=["traces/a.jsonl"],
            candidates_to_prune=["cand-1"],
            bytes_freed=1024,
            total_meta_bytes=2048,
            traces_dir_bytes=512,
            outcomes_dir_bytes=256,
            candidates_dir_bytes=1280,
        )
        d = plan.to_dict()
        for key in (
            "meta_dir", "dry_run", "max_age_days",
            "keep_top", "keep_bottom", "keep_recent",
            "traces_to_remove", "candidates_to_prune",
            "bytes_freed", "total_meta_bytes",
            "traces_dir_bytes", "outcomes_dir_bytes", "candidates_dir_bytes",
        ):
            assert key in d
        assert d["traces_to_remove"] == ["traces/a.jsonl"]
        assert d["bytes_freed"] == 1024


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        assert hasattr(meta, "PrunePlan")
        assert hasattr(meta, "prune")
        assert hasattr(meta, "disk_usage")
        assert hasattr(meta, "DEFAULT_TRACE_MAX_AGE_DAYS")