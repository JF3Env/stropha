"""Reusable index cache for the Meta-Harness loop (WIN-177).

Two candidates whose ``BuiltStages.indexing_hash()`` (WIN-176) match
produce identical index databases — the only differences are at query
time (storage / retrieval / reranker). Re-indexing from scratch per
candidate would be wasteful; the cache lets the second candidate copy
the first's index instead of paying full re-embed.

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
import os
import shutil
import tempfile
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
    source_repo: str | None = None
    adapter_ids: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexing_hash": self.indexing_hash,
            "index_path": str(self.index_path),
            "meta_path": str(self.meta_path),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "sha": self.sha,
            "source_repo": self.source_repo,
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

    def get(
        self,
        indexing_hash: str,
        *,
        source_repo: Path | str | None = None,
        sha: str | None = None,
    ) -> Path | None:
        """Return a complete cached index matching the requested identity."""
        candidate = self._entry_dir(indexing_hash) / self.INDEX_FILENAME
        if not candidate.is_file() or candidate.is_symlink():
            return None
        try:
            self.read_meta(indexing_hash, source_repo=source_repo, sha=sha)
        except IndexCacheError:
            return None
        return candidate

    def has(
        self,
        indexing_hash: str,
        *,
        source_repo: Path | str | None = None,
        sha: str | None = None,
    ) -> bool:
        return self.get(indexing_hash, source_repo=source_repo, sha=sha) is not None

    def read_meta(
        self,
        indexing_hash: str,
        *,
        source_repo: Path | str | None = None,
        sha: str | None = None,
    ) -> dict[str, Any]:
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
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCacheError(f"meta.json malformed for {indexing_hash!r}: {exc}") from exc
        if not isinstance(meta, dict):
            raise IndexCacheError(f"meta.json malformed for {indexing_hash!r}: expected object")
        expected_repo = _normalise_source_repo(source_repo)
        if meta.get("indexing_hash") != indexing_hash:
            raise IndexCacheError(f"cache identity mismatch for {indexing_hash!r}: indexing_hash")
        if meta.get("copy_mode") != "copy":
            raise IndexCacheError(f"cache identity mismatch for {indexing_hash!r}: copy_mode")
        if expected_repo is not None and meta.get("source_repo") != expected_repo:
            raise IndexCacheError(f"cache identity mismatch for {indexing_hash!r}: source_repo")
        if sha is not None and meta.get("sha") != sha:
            raise IndexCacheError(f"cache identity mismatch for {indexing_hash!r}: sha")
        return meta

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
        source_repo: Path | str | None = None,
        copy_mode: str = "copy",
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
            source_repo: optional source repository path. When provided,
                cache hits must match its normalized path.
            copy_mode: only ``"copy"`` is accepted. Cached data must not
                share writable storage with its source.

        Idempotent: if an entry for ``indexing_hash`` already exists,
        the existing entry is returned without re-materialising.
        """
        src = Path(index_path).resolve()
        if not src.is_file():
            raise IndexCacheError(f"source index does not exist: {src}")
        if copy_mode != "copy":
            raise IndexCacheError("copy_mode must be 'copy'; hardlinks and symlinks are unsafe")

        entry_dir = self._entry_dir(indexing_hash)
        normalized_repo = _normalise_source_repo(source_repo)

        existing = self.get(indexing_hash, source_repo=normalized_repo, sha=sha)
        if existing is not None:
            log.info("index_cache.reuse", indexing_hash=indexing_hash)
            return self._inspect_entry(indexing_hash)
        if entry_dir.exists():
            raise IndexCacheError(f"cache identity mismatch for {indexing_hash!r}")

        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{indexing_hash}.tmp-", dir=self.base_dir)
        )
        staging_index = staging_dir / self.INDEX_FILENAME
        staging_meta = staging_dir / self.META_FILENAME
        try:
            _materialise(src, staging_index, copy_mode)
            meta = {
                "indexing_hash": indexing_hash,
                "sha": sha,
                "source_repo": normalized_repo,
                "adapter_ids": dict(adapter_ids or {}),
                "created_at": _now_utc_iso(),
                "source_path": str(src),
                "copy_mode": copy_mode,
            }
            staging_meta.write_text(
                json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.rename(staging_dir, entry_dir)
        except OSError as exc:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if self.get(indexing_hash, source_repo=normalized_repo, sha=sha) is not None:
                log.info(
                    "index_cache.race_idempotent",
                    indexing_hash=indexing_hash,
                )
                return self._inspect_entry(indexing_hash)
            raise IndexCacheError(
                f"failed to materialise index for {indexing_hash!r}: {exc}"
            ) from exc
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        dst_index = entry_dir / self.INDEX_FILENAME
        log.info(
            "index_cache.put",
            indexing_hash=indexing_hash,
            mode=copy_mode,
            size_bytes=dst_index.stat().st_size,
        )
        return self._inspect_entry(indexing_hash)

    def clone_writable(
        self,
        *,
        indexing_hash: str,
        destination: Path | str,
        source_repo: Path | str | None = None,
        sha: str | None = None,
    ) -> Path:
        """Copy a cache hit to an independent, writable database file."""
        cached = self.get(indexing_hash, source_repo=source_repo, sha=sha)
        if cached is None:
            raise IndexCacheError(f"no matching cache entry for indexing_hash={indexing_hash!r}")

        dst = Path(destination).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.resolve() == cached.resolve():
            raise IndexCacheError("writable clone destination must not be the cached index")

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{dst.name}.tmp-", dir=dst.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(cached, temporary)
            os.replace(temporary, dst)
        except OSError as exc:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise IndexCacheError(
                f"failed to clone cached index {indexing_hash!r} to {dst}: {exc}"
            ) from exc
        return dst

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
            if sub.name.startswith(".") or self.get(sub.name) is None:
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
            source_repo=meta.get("source_repo"),
            adapter_ids=dict(meta.get("adapter_ids") or {}),
        )


def _materialise(src: Path, dst: Path, mode: str) -> None:
    """Copy ``src`` to ``dst`` without sharing writable storage."""
    if mode == "copy":
        shutil.copyfile(src, dst)
        return
    raise IndexCacheError(f"unknown copy_mode={mode!r}; expected copy")


def _normalise_source_repo(source_repo: Path | str | None) -> str | None:
    if source_repo is None:
        return None
    return str(Path(source_repo).expanduser().resolve())


__all__ = [
    "CacheEntry",
    "IndexCache",
    "IndexCacheError",
]
