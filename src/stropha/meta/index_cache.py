"""Reusable index cache for the Meta-Harness loop (WIN-177).

Two candidates whose ``BuiltStages.indexing_hash()`` (WIN-176) match
produce identical index databases — the only differences are at query
time (storage / retrieval / reranker). Re-indexing from scratch per
candidate would be wasteful; the cache lets the second candidate
*hardlink-or-copy* the first's index instead of paying full re-embed.

Filesystem layout::

    .stropha/meta/snapshot/indexes/<hash>/
      index.db         # the SQLite-vec database
      meta.json        # provenance: indexing_hash, sha, created_at,
                       # adapter_ids, source_repo

The cache is a filesystem dictionary keyed by indexing_hash. Operations:

- ``get(hash) -> Path | None`` — return path to a cached index.db if
  present, else None.
- ``put(hash, index_path, meta)`` — materialise ``index_path`` into the
  cache under ``hash``. Idempotent on collision.
- ``list_entries() -> list[CacheEntry]`` — enumerate every cached
  index for ``stropha cost`` / pruning.
- ``prune(keep)`` — remove entries whose hash is NOT in ``keep``.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


class IndexCacheError(StrophaError):
    """Raised when an index cache operation cannot be satisfied."""


@dataclass
class CacheEntry:
    """One cache entry's on-disk metadata."""

    indexing_hash: str
    index_path: Path
    meta_path: Path
    size_bytes: int = 0
    created_at: str = ""
    sha: str | None = None
    adapter_ids: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexing_hash": self.indexing_hash,
            "index_path": str(self.index_path),
            "meta_path": str(self.meta_path),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "sha": self.sha,
            "adapter_ids": dict(self.adapter_ids),
        }


def _now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class IndexCache:
    """Filesystem-backed dictionary of indexing_hash → index.db.

    Args:
        base_dir: ``.stropha/meta/snapshot/indexes/`` by convention.
            Created on first use.
    """

    INDEX_FILENAME = "index.db"
    META_FILENAME = "meta.json"

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, indexing_hash: str) -> Path:
        return self.base_dir / indexing_hash

    # ------------------------------------------------------------------ #
    # Reads                                                              #
    # ------------------------------------------------------------------ #

    def get(self, indexing_hash: str) -> Path | None:
        """Return the cached index.db path for ``indexing_hash`` or None."""
        candidate = self._entry_dir(indexing_hash) / self.INDEX_FILENAME
        return candidate if candidate.is_file() else None

    def has(self, indexing_hash: str) -> bool:
        return self.get(indexing_hash) is not None

    def read_meta(self, indexing_hash: str) -> dict[str, Any]:
        """Load the meta.json for a cached entry.

        Raises ``IndexCacheError`` when the entry is missing or the
        meta.json is malformed.
        """
        meta_path = self._entry_dir(indexing_hash) / self.META_FILENAME
        if not meta_path.is_file():
            raise IndexCacheError(
                f"no meta for indexing_hash={indexing_hash!r} at {meta_path}"
            )
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCacheError(f"meta.json malformed for {indexing_hash!r}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Writes                                                             #
    # ------------------------------------------------------------------ #

    def put(
        self,
        *,
        indexing_hash: str,
        index_path: Path | str,
        adapter_ids: dict[str, str] | None = None,
        sha: str | None = None,
        copy_mode: str = "hardlink",
    ) -> CacheEntry:
        """Materialise ``index_path`` into the cache under ``indexing_hash``.

        Args:
            indexing_hash: 12-hex hash from
                :meth:`stropha.pipeline.BuiltStages.indexing_hash`.
            index_path: source ``index.db`` to copy in.
            adapter_ids: optional dict {stage: adapter_id} for the
                meta.json provenance.
            sha: optional source-tree SHA the index was built against
                (matches WIN-175 snapshots).
            copy_mode: ``"hardlink"`` (default, near-instant + dedup'd),
                ``"copy"`` (safe across filesystems), or ``"symlink"``.

        Idempotent: if an entry for ``indexing_hash`` already exists,
        the existing entry is returned without re-materialising.
        """
        src = Path(index_path).resolve()
        if not src.is_file():
            raise IndexCacheError(f"source index does not exist: {src}")

        entry_dir = self._entry_dir(indexing_hash)
        dst_index = entry_dir / self.INDEX_FILENAME
        dst_meta = entry_dir / self.META_FILENAME

        if dst_index.is_file():
            # Idempotent — re-using cached entry.
            log.info("index_cache.reuse", indexing_hash=indexing_hash)
            return self._inspect_entry(indexing_hash)

        entry_dir.mkdir(parents=True, exist_ok=True)
        try:
            _materialise(src, dst_index, copy_mode)
        except OSError as exc:
            # WIN-180 race-condition fix: if a concurrent ``put()`` for
            # the same hash already wrote ``dst_index``, our hardlink /
            # copy raises FileExistsError or similar. The entry is
            # actually valid (someone else won the race), so DO NOT
            # rmtree the dir — that would destroy the sibling's work.
            if dst_index.is_file():
                log.info(
                    "index_cache.race_idempotent",
                    indexing_hash=indexing_hash,
                )
            else:
                shutil.rmtree(entry_dir, ignore_errors=True)
                raise IndexCacheError(
                    f"failed to materialise index for {indexing_hash!r}: {exc}"
                ) from exc

        meta = {
            "indexing_hash": indexing_hash,
            "sha": sha,
            "adapter_ids": dict(adapter_ids or {}),
            "created_at": _now_utc_iso(),
            "source_path": str(src),
            "copy_mode": copy_mode,
        }
        dst_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        log.info(
            "index_cache.put",
            indexing_hash=indexing_hash,
            mode=copy_mode,
            size_bytes=dst_index.stat().st_size,
        )
        return self._inspect_entry(indexing_hash)

    # ------------------------------------------------------------------ #
    # Listing + pruning                                                  #
    # ------------------------------------------------------------------ #

    def list_entries(self) -> list[CacheEntry]:
        """Enumerate every cached index, sorted by indexing_hash."""
        if not self.base_dir.is_dir():
            return []
        out: list[CacheEntry] = []
        for sub in sorted(self.base_dir.iterdir()):
            if not sub.is_dir():
                continue
            if not (sub / self.INDEX_FILENAME).is_file():
                continue
            with contextlib.suppress(IndexCacheError):
                out.append(self._inspect_entry(sub.name))
        return out

    def prune(self, *, keep: set[str] | None = None) -> list[str]:
        """Drop every cached entry whose hash is NOT in ``keep``.

        Returns the list of pruned hashes.
        """
        keep_set = keep or set()
        pruned: list[str] = []
        for entry in self.list_entries():
            if entry.indexing_hash in keep_set:
                continue
            shutil.rmtree(self._entry_dir(entry.indexing_hash), ignore_errors=True)
            pruned.append(entry.indexing_hash)
        if pruned:
            log.info("index_cache.prune", pruned=pruned, keep=sorted(keep_set))
        return pruned

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _inspect_entry(self, indexing_hash: str) -> CacheEntry:
        entry_dir = self._entry_dir(indexing_hash)
        index_path = entry_dir / self.INDEX_FILENAME
        meta_path = entry_dir / self.META_FILENAME
        size = 0
        with contextlib.suppress(OSError):
            size = index_path.stat().st_size
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return CacheEntry(
            indexing_hash=indexing_hash,
            index_path=index_path,
            meta_path=meta_path,
            size_bytes=size,
            created_at=str(meta.get("created_at") or ""),
            sha=meta.get("sha"),
            adapter_ids=dict(meta.get("adapter_ids") or {}),
        )


def _materialise(src: Path, dst: Path, mode: str) -> None:
    """Place ``src`` at ``dst`` according to ``mode``.

    ``hardlink`` (default) tries to use ``Path.link_to`` / ``os.link``;
    falls back to copy when across-filesystem. ``copy`` always copies;
    ``symlink`` creates a symlink.
    """
    import os

    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        shutil.copyfile(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        return
    raise IndexCacheError(f"unknown copy_mode={mode!r}; expected hardlink|copy|symlink")


__all__ = [
    "CacheEntry",
    "IndexCache",
    "IndexCacheError",
]
