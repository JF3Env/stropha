"""Unit tests for the Meta-Harness index cache (WIN-177)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stropha.meta import CacheEntry, IndexCache, IndexCacheError
from stropha.meta.index_cache import _materialise


def _fake_index(tmp_path: Path, content: bytes = b"sqlite:db") -> Path:
    p = tmp_path / "src.db"
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_base_dir(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        assert (tmp_path / "cache").is_dir()
        assert cache.base_dir == tmp_path / "cache"

    def test_accepts_string_base_dir(self, tmp_path: Path) -> None:
        cache = IndexCache(str(tmp_path / "cache"))
        assert cache.base_dir == tmp_path / "cache"

    def test_idempotent_on_existing(self, tmp_path: Path) -> None:
        IndexCache(tmp_path)
        IndexCache(tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# get / has
# ---------------------------------------------------------------------------


class TestGet:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path)
        assert cache.get("abcdef123456") is None
        assert cache.has("abcdef123456") is False

    def test_hit_after_put(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        assert cache.has("h1") is True
        cached = cache.get("h1")
        assert cached is not None
        assert cached.read_bytes() == b"sqlite:db"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


class TestPut:
    def test_writes_meta_json(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(
            indexing_hash="h1",
            index_path=src,
            adapter_ids={"walker": "git-ls-files", "chunker": "tree-sitter"},
            sha="deadbeef",
        )
        meta = json.loads((cache.base_dir / "h1" / "meta.json").read_text())
        assert meta["indexing_hash"] == "h1"
        assert meta["sha"] == "deadbeef"
        assert meta["adapter_ids"]["walker"] == "git-ls-files"
        assert "created_at" in meta

    def test_idempotent_when_already_present(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        first = cache.put(indexing_hash="h1", index_path=src)
        # Second put with a different src — must NOT overwrite.
        second_dir = tmp_path / "second"
        second_dir.mkdir(parents=True, exist_ok=True)
        src2 = second_dir / "src.db"
        src2.write_bytes(b"different")
        second = cache.put(indexing_hash="h1", index_path=src2)
        assert first.index_path == second.index_path
        # Content unchanged from the first put.
        assert (cache.base_dir / "h1" / "index.db").read_bytes() == b"sqlite:db"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        with pytest.raises(IndexCacheError, match="source index does not exist"):
            cache.put(indexing_hash="h1", index_path=tmp_path / "missing.db")

    def test_copy_mode_copy(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, copy_mode="copy")
        # Distinct inode — change source, cache unchanged.
        src.write_bytes(b"mutated")
        assert (cache.base_dir / "h1" / "index.db").read_bytes() == b"sqlite:db"

    def test_copy_mode_hardlink_dedupes(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, copy_mode="hardlink")
        dst = cache.base_dir / "h1" / "index.db"
        # Same inode → mutating one updates both. (Hardlink on same fs.)
        assert dst.stat().st_ino == src.stat().st_ino

    def test_copy_mode_symlink(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, copy_mode="symlink")
        dst = cache.base_dir / "h1" / "index.db"
        assert dst.is_symlink()

    def test_unknown_copy_mode_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        with pytest.raises(IndexCacheError, match="unknown copy_mode"):
            cache.put(indexing_hash="h1", index_path=src, copy_mode="bogus")

    def test_materialise_failure_rolls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)

        import stropha.meta.index_cache as mod

        def boom(*_a, **_kw):  # type: ignore[no-untyped-def]
            raise OSError("simulated disk full")

        monkeypatch.setattr(mod, "_materialise", boom)
        with pytest.raises(IndexCacheError, match="failed to materialise"):
            cache.put(indexing_hash="h1", index_path=src)
        # Entry dir cleaned up so a retry can re-create.
        assert not (cache.base_dir / "h1").exists()


# ---------------------------------------------------------------------------
# read_meta
# ---------------------------------------------------------------------------


class TestReadMeta:
    def test_missing_entry_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        with pytest.raises(IndexCacheError, match="no meta"):
            cache.read_meta("nope")

    def test_malformed_meta_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        # Corrupt the meta file.
        (cache.base_dir / "h1" / "meta.json").write_text("not json{{")
        with pytest.raises(IndexCacheError, match="malformed"):
            cache.read_meta("h1")

    def test_round_trip(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, sha="abc")
        meta = cache.read_meta("h1")
        assert meta["sha"] == "abc"


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------


class TestListEntries:
    def test_empty_when_no_entries(self, tmp_path: Path) -> None:
        assert IndexCache(tmp_path / "cache").list_entries() == []

    def test_lists_each_entry(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        cache.put(indexing_hash="h2", index_path=src)
        entries = cache.list_entries()
        assert {e.indexing_hash for e in entries} == {"h1", "h2"}

    def test_skips_dir_without_index_db(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        # Manually create a folder with no index.db.
        (cache.base_dir / "stray").mkdir()
        assert cache.list_entries() == []

    def test_skips_non_dir_entries(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        (cache.base_dir / "stray.txt").write_text("not a dir")
        assert cache.list_entries() == []

    def test_missing_base_dir_returns_empty(self, tmp_path: Path) -> None:
        # Construct then delete the dir.
        cache = IndexCache(tmp_path / "cache")
        os.rmdir(cache.base_dir)
        assert cache.list_entries() == []

    def test_entry_size_reflects_index(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path, content=b"x" * 1024)
        cache.put(indexing_hash="h1", index_path=src)
        [entry] = cache.list_entries()
        assert entry.size_bytes == 1024


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_empty_keep_drops_everything(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        cache.put(indexing_hash="h2", index_path=src)
        pruned = cache.prune(keep=set())
        assert sorted(pruned) == ["h1", "h2"]
        assert cache.list_entries() == []

    def test_keeps_specified(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="keep", index_path=src)
        cache.put(indexing_hash="drop", index_path=src)
        pruned = cache.prune(keep={"keep"})
        assert pruned == ["drop"]
        assert {e.indexing_hash for e in cache.list_entries()} == {"keep"}

    def test_no_entries_returns_empty(self, tmp_path: Path) -> None:
        assert IndexCache(tmp_path / "cache").prune() == []


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_to_dict_round_trip(self, tmp_path: Path) -> None:
        entry = CacheEntry(
            indexing_hash="h",
            index_path=tmp_path / "i.db",
            meta_path=tmp_path / "m.json",
            size_bytes=42,
            created_at="2026-05-22T12:00:00+00:00",
            sha="abc",
            adapter_ids={"walker": "git-ls-files"},
        )
        d = entry.to_dict()
        for key in (
            "indexing_hash", "index_path", "meta_path",
            "size_bytes", "created_at", "sha", "adapter_ids",
        ):
            assert key in d
        assert d["size_bytes"] == 42


# ---------------------------------------------------------------------------
# _materialise (direct)
# ---------------------------------------------------------------------------


class TestMaterialiseHelper:
    def test_hardlink_falls_back_to_copy_across_fs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force os.link to raise (simulates cross-filesystem) and verify
        the copy fallback path."""
        src = _fake_index(tmp_path)
        dst = tmp_path / "out.db"

        original_link = os.link

        def cross_fs(a, b):  # type: ignore[no-untyped-def]
            raise OSError("simulated EXDEV")

        monkeypatch.setattr(os, "link", cross_fs)
        _materialise(src, dst, "hardlink")
        # Restore for cleanup safety.
        monkeypatch.setattr(os, "link", original_link)
        assert dst.read_bytes() == src.read_bytes()
        # Different inodes (it was a copy, not a hardlink).
        assert dst.stat().st_ino != src.stat().st_ino


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        for name in ("IndexCache", "IndexCacheError", "CacheEntry"):
            assert hasattr(meta, name)