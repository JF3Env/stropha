"""Unit tests for the Meta-Harness index cache (WIN-177)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.meta import CacheEntry, IndexCache, IndexCacheError


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

    def test_requires_complete_entry(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        partial = cache.base_dir / "h1"
        partial.mkdir()
        (partial / "index.db").write_bytes(b"not published")
        assert cache.get("h1") is None

    def test_rejects_legacy_shared_storage_entry(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        entry = cache.base_dir / "h1"
        entry.mkdir()
        (entry / "index.db").write_bytes(b"legacy")
        (entry / "meta.json").write_text(
            json.dumps({"indexing_hash": "h1", "copy_mode": "hardlink"})
        )
        assert cache.get("h1") is None

    def test_validates_source_identity(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        repo = tmp_path / "repo"
        repo.mkdir()
        cache.put(indexing_hash="h1", index_path=src, source_repo=repo, sha="abc")

        assert cache.get("h1", source_repo=repo / ".", sha="abc") is not None
        assert cache.get("h1", source_repo=tmp_path / "other", sha="abc") is None
        assert cache.get("h1", source_repo=repo, sha="def") is None


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
            source_repo=tmp_path / "repo",
        )
        meta = json.loads((cache.base_dir / "h1" / "meta.json").read_text())
        assert meta["indexing_hash"] == "h1"
        assert meta["sha"] == "deadbeef"
        assert meta["source_repo"] == str((tmp_path / "repo").resolve())
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

    def test_existing_entry_with_different_identity_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, source_repo=tmp_path, sha="abc")

        with pytest.raises(IndexCacheError, match="identity mismatch"):
            cache.put(
                indexing_hash="h1",
                index_path=src,
                source_repo=tmp_path,
                sha="def",
            )

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        with pytest.raises(IndexCacheError, match="source index does not exist"):
            cache.put(indexing_hash="h1", index_path=tmp_path / "missing.db")

    def test_default_copy_does_not_share_storage(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        dst = cache.base_dir / "h1" / "index.db"
        assert not dst.is_symlink()
        assert dst.stat().st_ino != src.stat().st_ino
        src.write_bytes(b"mutated")
        assert dst.read_bytes() == b"sqlite:db"

    @pytest.mark.parametrize("copy_mode", ["hardlink", "symlink", "bogus"])
    def test_unsafe_copy_modes_raise(self, tmp_path: Path, copy_mode: str) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        with pytest.raises(IndexCacheError, match="hardlinks and symlinks are unsafe"):
            cache.put(indexing_hash="h1", index_path=src, copy_mode=copy_mode)

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
        assert not list(cache.base_dir.glob(".h1.tmp-*"))

    def test_publishes_complete_entry_with_one_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        destination = cache.base_dir / "h1"
        observed: dict[str, bool] = {}

        import stropha.meta.index_cache as mod

        real_rename = mod.os.rename

        def inspect_then_rename(staging: Path, target: Path) -> None:
            observed["hidden"] = not destination.exists()
            observed["index"] = (Path(staging) / "index.db").is_file()
            observed["meta"] = (Path(staging) / "meta.json").is_file()
            real_rename(staging, target)

        monkeypatch.setattr(mod.os, "rename", inspect_then_rename)
        cache.put(indexing_hash="h1", index_path=src)

        assert observed == {"hidden": True, "index": True, "meta": True}
        assert cache.get("h1") == destination / "index.db"


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

    def test_identity_mismatch_raises(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, source_repo=tmp_path, sha="abc")
        with pytest.raises(IndexCacheError, match="source_repo"):
            cache.read_meta("h1", source_repo=tmp_path / "other", sha="abc")


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
        cache.base_dir.rmdir()
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
            source_repo="/repo",
            adapter_ids={"walker": "git-ls-files"},
        )
        d = entry.to_dict()
        for key in (
            "indexing_hash", "index_path", "meta_path",
            "size_bytes", "created_at", "sha", "source_repo", "adapter_ids",
        ):
            assert key in d
        assert d["size_bytes"] == 42


# ---------------------------------------------------------------------------
# writable clones
# ---------------------------------------------------------------------------


class TestCloneWritable:
    def test_returns_independent_regular_file(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, source_repo=tmp_path, sha="abc")
        cached = cache.get("h1", source_repo=tmp_path, sha="abc")
        assert cached is not None

        clone = cache.clone_writable(
            indexing_hash="h1",
            destination=tmp_path / "candidate" / "index.db",
            source_repo=tmp_path,
            sha="abc",
        )

        assert clone.is_file()
        assert not clone.is_symlink()
        assert clone.stat().st_ino != cached.stat().st_ino
        clone.write_bytes(b"candidate mutation")
        assert cached.read_bytes() == b"sqlite:db"

    def test_identity_mismatch_is_not_cloned(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src, sha="abc")
        with pytest.raises(IndexCacheError, match="no matching cache entry"):
            cache.clone_writable(
                indexing_hash="h1", destination=tmp_path / "clone.db", sha="def"
            )

    def test_replaces_destination_atomically(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        destination = tmp_path / "clone.db"
        destination.write_bytes(b"old")

        cache.clone_writable(indexing_hash="h1", destination=destination)

        assert destination.read_bytes() == b"sqlite:db"
        assert not list(tmp_path.glob(".clone.db.tmp-*"))

    def test_rejects_destination_aliasing_cache(self, tmp_path: Path) -> None:
        cache = IndexCache(tmp_path / "cache")
        src = _fake_index(tmp_path)
        cache.put(indexing_hash="h1", index_path=src)
        alias = tmp_path / "alias"
        alias.symlink_to(cache.base_dir / "h1", target_is_directory=True)

        with pytest.raises(IndexCacheError, match="must not be the cached index"):
            cache.clone_writable(
                indexing_hash="h1", destination=alias / "index.db"
            )


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        for name in ("IndexCache", "IndexCacheError", "CacheEntry"):
            assert hasattr(meta, name)
