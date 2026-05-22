"""Disk-growth pruning for ``.stropha/meta/`` (WIN-166).

Three policies, applied independently and unioned:

1. **Age-based trace pruning** — trace JSONL files older than
   ``max_age_days`` are deleted. The trace bodies are the bulk of meta
   disk usage and grow unbounded otherwise. ``config.yaml`` /
   ``meta.json`` / ``scores.json`` are kept forever (they're tiny, and
   they're the durable record the proposer reads back).
2. **Bucket-based candidate pruning** — delegates to
   :meth:`stropha.meta.store.ArtifactStore.compact` so the keep-top /
   keep-bottom / keep-recent contract from WIN-161 stays the single
   source of truth.
3. **Dry-run preview** — every method has a ``dry_run=True`` flag that
   collects the would-be-removed paths into a :class:`PrunePlan` without
   touching disk. The CLI surfaces this as ``stropha meta prune --dry-run``.

The plan also reports disk-usage totals so ``stropha cost`` can publish
the per-run meta footprint without re-walking the tree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .store import ArtifactStore

log = get_logger(__name__)


DEFAULT_TRACE_MAX_AGE_DAYS = 90


@dataclass
class PrunePlan:
    """What pruning *would* do (dry-run) or *did* (real run).

    All sizes are in bytes; paths are repo-relative when possible.
    """

    meta_dir: str
    dry_run: bool
    max_age_days: int
    keep_top: int
    keep_bottom: int
    keep_recent: int
    traces_to_remove: list[str] = field(default_factory=list)
    candidates_to_prune: list[str] = field(default_factory=list)
    bytes_freed: int = 0
    total_meta_bytes: int = 0
    traces_dir_bytes: int = 0
    outcomes_dir_bytes: int = 0
    candidates_dir_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_dir": self.meta_dir,
            "dry_run": self.dry_run,
            "max_age_days": self.max_age_days,
            "keep_top": self.keep_top,
            "keep_bottom": self.keep_bottom,
            "keep_recent": self.keep_recent,
            "traces_to_remove": self.traces_to_remove,
            "candidates_to_prune": self.candidates_to_prune,
            "bytes_freed": self.bytes_freed,
            "total_meta_bytes": self.total_meta_bytes,
            "traces_dir_bytes": self.traces_dir_bytes,
            "outcomes_dir_bytes": self.outcomes_dir_bytes,
            "candidates_dir_bytes": self.candidates_dir_bytes,
        }


def _dir_bytes(path: Path) -> int:
    """Recursive ``du`` — sum of every regular file under ``path``.

    Missing directory returns 0. Tolerant of permission errors (skip the
    offending file silently — disk-usage reporting must never fail loudly).
    """
    total = 0
    if not path.is_dir():
        return 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            # File vanished mid-walk or permission denied — skip it.
            continue
    return total


def _stale_trace_files(traces_dir: Path, max_age_days: int) -> list[Path]:
    """Trace ``*.jsonl`` files whose mtime is older than the cutoff."""
    if not traces_dir.is_dir():
        return []
    cutoff = time.time() - max_age_days * 86400
    stale: list[Path] = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except OSError:
            # Vanished between iter and stat — treat as not-stale.
            continue
    return stale


def disk_usage(meta_dir: Path | str) -> dict[str, int]:
    """Public-ish helper for ``stropha cost``.

    Returns a ``{"total": …, "traces": …, "outcomes": …, "candidates": …}``
    dict in bytes. Always returns valid data even when ``meta_dir`` does
    not exist (everything zero) so callers don't need defensive code.
    """
    md = Path(meta_dir)
    if not md.is_dir():
        return {"total": 0, "traces": 0, "outcomes": 0, "candidates": 0}
    traces = _dir_bytes(md / "traces")
    outcomes = _dir_bytes(md / "outcomes")
    candidates = _dir_bytes(md / "candidates")
    total = _dir_bytes(md)
    return {
        "total": total,
        "traces": traces,
        "outcomes": outcomes,
        "candidates": candidates,
    }


def prune(
    meta_dir: Path | str,
    *,
    max_age_days: int = DEFAULT_TRACE_MAX_AGE_DAYS,
    keep_top: int = 10,
    keep_bottom: int = 10,
    keep_recent: int = 50,
    dry_run: bool = False,
) -> PrunePlan:
    """Apply all three policies and return what was (or would be) removed.

    Args:
        meta_dir: typically ``.stropha/meta/``. Must exist; raises ``FileNotFoundError`` otherwise.
        max_age_days: trace JSONL files older than this are removed.
            Set to 0 to remove every trace (mostly useful for tests).
        keep_top / keep_bottom / keep_recent: forwarded to
            :meth:`ArtifactStore.compact`.
        dry_run: when True, no files are removed; the plan still
            populates the would-remove lists and the disk-usage totals.

    Returns the :class:`PrunePlan` describing the run.
    """
    md = Path(meta_dir)
    if not md.is_dir():
        raise FileNotFoundError(f"meta dir does not exist: {md}")

    # --- Disk-usage snapshot (taken BEFORE any removal so the dry-run
    # output matches the live state the operator sees right now).
    usage = disk_usage(md)
    plan = PrunePlan(
        meta_dir=str(md),
        dry_run=dry_run,
        max_age_days=max_age_days,
        keep_top=keep_top,
        keep_bottom=keep_bottom,
        keep_recent=keep_recent,
        total_meta_bytes=usage["total"],
        traces_dir_bytes=usage["traces"],
        outcomes_dir_bytes=usage["outcomes"],
        candidates_dir_bytes=usage["candidates"],
    )

    # --- Policy 1: age-based trace pruning ------------------------------
    traces_dir = md / "traces"
    stale = _stale_trace_files(traces_dir, max_age_days)
    bytes_freed = 0
    for path in stale:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        plan.traces_to_remove.append(str(path.relative_to(md)))
        bytes_freed += size
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                log.warning("prune.unlink_failed", path=str(path), error=str(exc))

    # --- Policy 2: candidate compaction --------------------------------
    candidates_dir = md / "candidates"
    if candidates_dir.is_dir():
        store = ArtifactStore(md)
        if dry_run:
            # Compute what compact() WOULD prune without mutating.
            preview = _compact_preview(store, keep_top, keep_bottom, keep_recent)
            plan.candidates_to_prune = preview
        else:
            pruned = store.compact(
                keep_top=keep_top,
                keep_bottom=keep_bottom,
                keep_recent=keep_recent,
            )
            plan.candidates_to_prune = pruned

    plan.bytes_freed = bytes_freed
    log.info(
        "prune.done",
        dry_run=dry_run,
        traces_removed=len(plan.traces_to_remove),
        candidates_pruned=len(plan.candidates_to_prune),
        bytes_freed=plan.bytes_freed,
    )
    return plan


def _compact_preview(
    store: ArtifactStore,
    keep_top: int,
    keep_bottom: int,
    keep_recent: int,
) -> list[str]:
    """Dry-run twin of :meth:`ArtifactStore.compact`.

    Mirrors the bucket math without touching disk. Kept here (not on
    ArtifactStore) because compact's contract is "mutate" — splitting it
    would risk drift; we accept the duplicated logic to preserve clarity.
    """
    live = store.list_candidates(include_pruned=False)
    if not live:
        return []
    survivors: set[str] = set()
    by_fit = sorted(live, key=lambda r: r.fitness(), reverse=True)
    for r in by_fit[: max(0, keep_top)]:
        survivors.add(r.id)
    if keep_bottom > 0:
        for r in by_fit[-max(0, keep_bottom) :]:
            survivors.add(r.id)
    by_recent = sorted(live, key=lambda r: r.created_at, reverse=True)
    for r in by_recent[: max(0, keep_recent)]:
        survivors.add(r.id)
    return [r.id for r in live if r.id not in survivors]


__all__ = [
    "DEFAULT_TRACE_MAX_AGE_DAYS",
    "PrunePlan",
    "disk_usage",
    "prune",
]
