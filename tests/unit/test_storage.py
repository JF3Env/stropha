"""Tests for SQLite + sqlite-vec + FTS5 layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha.errors import StorageError
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk
from stropha.storage import Storage
from stropha.storage.sqlite import SCHEMA_VERSION


def _identity(key: str = "github.com/example/repo", url: str | None = None) -> RepoIdentity:
    return RepoIdentity(
        normalized_key=key,
        remote_url=url or f"https://{key}.git",
        root_path=Path("/tmp/example/repo"),
        default_branch="main",
        head_commit="abc123",
    )


def _make_chunk(
    chunk_id: str,
    rel_path: str,
    content: str,
    symbol: str | None = None,
    kind: str = "method",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        language="java",
        kind=kind,
        symbol=symbol,
        start_line=1,
        end_line=10,
        content=content,
        content_hash=chunk_id,  # use chunk_id as a stable proxy
    )


def _vec(value: float, dim: int = 4) -> list[float]:
    """Return a unit vector aligned to one axis."""
    v = [0.0] * dim
    v[int(value) % dim] = 1.0
    return v


def _secondary_vec(axis: int) -> list[float]:
    return _vec(float(axis), dim=256)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    db = tmp_path / "test.db"
    return Storage(db, embedding_dim=4)


def test_upsert_and_dense_search(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:a", "Foo.java", "class Foo {}"),
        _vec(0), "test-model", 4,
    )
    storage.upsert_chunk(
        _make_chunk("sha256:b", "Bar.java", "class Bar {}"),
        _vec(1), "test-model", 4,
    )
    storage.commit()

    hits = storage.search_dense(_vec(0), k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "sha256:a"  # exact match wins
    assert hits[0].score > hits[1].score


def test_bm25_search_returns_lexical_matches(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "FsrsCalculator.java",
            "public class FsrsCalculator { void calculateStability() {} }",
            symbol="FsrsCalculator",
        ),
        _vec(0), "test-model", 4,
    )
    storage.upsert_chunk(
        _make_chunk(
            "sha256:b", "Unrelated.java",
            "public class Other { void doSomething() {} }",
        ),
        _vec(1), "test-model", 4,
    )
    storage.commit()

    hits = storage.search_bm25("FsrsCalculator", k=5)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "sha256:a"


def test_bm25_camelcase_split(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "x.java",
            "the method submitAnswer is here",
        ),
        _vec(0), "test-model", 4,
    )
    storage.commit()
    # Query uses different casing — sanitizer should split CamelCase.
    hits = storage.search_bm25("submitAnswer", k=5)
    assert any(h.chunk_id == "sha256:a" for h in hits)


def test_chunk_is_fresh(storage: Storage) -> None:
    chunk = _make_chunk("sha256:a", "x.java", "...")
    storage.upsert_chunk(chunk, _vec(0), "test-model", 4)
    storage.commit()
    assert storage.chunk_is_fresh("sha256:a", chunk.content_hash, "test-model")
    assert not storage.chunk_is_fresh("sha256:a", "other-hash", "test-model")
    assert not storage.chunk_is_fresh("sha256:a", chunk.content_hash, "other-model")
    assert not storage.chunk_is_fresh("sha256:nope", chunk.content_hash, "test-model")


def test_delete_by_paths(storage: Storage) -> None:
    storage.upsert_chunk(_make_chunk("sha256:a", "x.java", "a"), _vec(0), "m", 4)
    storage.upsert_chunk(_make_chunk("sha256:b", "x.java", "b"), _vec(1), "m", 4)
    storage.upsert_chunk(_make_chunk("sha256:c", "y.java", "c"), _vec(2), "m", 4)
    storage.commit()
    deleted = storage.delete_by_paths(["x.java"])
    assert deleted == 2
    storage.commit()
    assert storage.stats()["chunks"] == 1


def test_lookup_by_symbol_prefers_exact(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk(
            "sha256:a", "x.java", "...", symbol="FsrsCalculator.calculateStability",
        ),
        _vec(0), "m", 4,
    )
    storage.upsert_chunk(
        _make_chunk(
            "sha256:b", "y.java", "...", symbol="FsrsCalculator",
        ),
        _vec(1), "m", 4,
    )
    storage.commit()
    hits = storage.lookup_by_symbol("FsrsCalculator")
    assert hits[0].chunk_id == "sha256:b"


def test_file_outline(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:a", "F.java", "...", symbol="F", kind="class"),
        _vec(0), "m", 4,
    )
    storage.upsert_chunk(
        _make_chunk("sha256:b", "F.java", "...", symbol="F.x", kind="method"),
        _vec(1), "m", 4,
    )
    storage.commit()
    outline = storage.file_outline("F.java")
    assert [o["symbol"] for o in outline] == ["F", "F.x"]


def test_file_outline_filters_duplicate_path_by_repo(storage: Storage) -> None:
    local_repo = storage.register_repo(_identity("github.com/example/local"))
    foreign_repo = storage.register_repo(_identity("github.com/example/foreign"))
    storage.upsert_chunk(
        _make_chunk("sha256:local-outline", "F.java", "...", symbol="Local"),
        _vec(0), "m", 4, repo_id=local_repo,
    )
    storage.upsert_chunk(
        _make_chunk("sha256:foreign-outline", "F.java", "...", symbol="Foreign"),
        _vec(1), "m", 4, repo_id=foreign_repo,
    )
    storage.commit()

    scoped = storage.file_outline("F.java", repo_id=local_repo)
    broad = storage.file_outline("F.java")

    assert [entry["symbol"] for entry in scoped] == ["Local"]
    assert {entry["symbol"] for entry in broad} == {"Local", "Foreign"}


def test_meta_roundtrip(storage: Storage) -> None:
    storage.set_meta("foo", "bar")
    storage.commit()
    assert storage.get_meta("foo") == "bar"
    storage.set_meta("foo", "baz")
    storage.commit()
    assert storage.get_meta("foo") == "baz"


def test_dimension_mismatch_rejected(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    s.upsert_chunk(_make_chunk("sha256:a", "x.java", "x"), _vec(0, 4), "m", 4)
    s.commit()
    s.close()
    with pytest.raises(StorageError):
        Storage(db, embedding_dim=8)


# ---- repos table & SearchHit.repo ----------------------------------------

def test_register_repo_is_idempotent(storage: Storage) -> None:
    rid1 = storage.register_repo(_identity("github.com/a/b"))
    rid2 = storage.register_repo(_identity("github.com/a/b"))
    storage.commit()
    assert rid1 == rid2
    repos = storage.list_repos()
    assert len(repos) == 1


def test_register_repo_distinct_keys_get_distinct_ids(storage: Storage) -> None:
    a = storage.register_repo(_identity("github.com/a/b"))
    b = storage.register_repo(_identity("github.com/c/d"))
    storage.commit()
    assert a != b
    assert {r.normalized_key for r in storage.list_repos()} == {
        "github.com/a/b",
        "github.com/c/d",
    }


def test_search_hit_carries_repo(storage: Storage) -> None:
    rid = storage.register_repo(_identity("github.com/foo/bar"))
    storage.upsert_chunk(
        _make_chunk("sha256:r", "Foo.java", "class Foo {}", symbol="Foo"),
        _vec(0), "m", 4, repo_id=rid,
    )
    storage.commit()

    hits = storage.search_dense(_vec(0), k=1)
    assert hits and hits[0].repo is not None
    assert hits[0].repo.normalized_key == "github.com/foo/bar"
    assert hits[0].repo.url == "https://github.com/foo/bar.git"
    assert hits[0].repo.default_branch == "main"

    # Same field surfaces through bm25 and symbol-lookup streams.
    bm = storage.search_bm25("Foo", k=1)
    assert bm and bm[0].repo and bm[0].repo.normalized_key == "github.com/foo/bar"

    sym = storage.lookup_by_symbol("Foo", limit=1)
    assert sym and sym[0].repo and sym[0].repo.normalized_key == "github.com/foo/bar"


def test_search_lanes_filter_repo_before_ranking_and_limit(storage: Storage) -> None:
    local_repo = storage.register_repo(_identity("github.com/example/local"))
    foreign_repo = storage.register_repo(_identity("github.com/example/foreign"))
    storage.upsert_chunk(
        _make_chunk(
            "sha256:foreign",
            "foreign.java",
            "Needle Needle Needle Needle",
            symbol="Needle",
        ),
        _vec(0),
        "m",
        4,
        repo_id=foreign_repo,
    )
    storage.upsert_chunk(
        _make_chunk(
            "sha256:local",
            "local.java",
            "Needle",
            symbol="LongerNeedleSymbol",
        ),
        _vec(1),
        "m",
        4,
        repo_id=local_repo,
    )
    storage.commit()

    assert storage.search_dense(_vec(0), k=1)[0].chunk_id == "sha256:foreign"
    assert storage.search_dense(_vec(0), k=1, repo_id=local_repo)[0].chunk_id == (
        "sha256:local"
    )
    assert storage.search_bm25("Needle", k=1, repo_id=local_repo)[0].chunk_id == (
        "sha256:local"
    )
    assert storage.search_symbol_tokens("Needle", k=1, repo_id=local_repo)[
        0
    ].chunk_id == "sha256:local"
    assert storage.lookup_by_symbol("Needle", limit=1, repo_id=local_repo)[
        0
    ].chunk_id == "sha256:local"


def test_secondary_search_filters_repo_before_limit(storage: Storage) -> None:
    local_repo = storage.register_repo(_identity("github.com/example/local"))
    foreign_repo = storage.register_repo(_identity("github.com/example/foreign"))
    for chunk_id, repo_id, axis in (
        ("sha256:foreign", foreign_repo, 0),
        ("sha256:local", local_repo, 1),
    ):
        chunk = _make_chunk(chunk_id, f"{chunk_id}.java", "Needle")
        storage.upsert_chunk(chunk, _vec(axis), "m", 4, repo_id=repo_id)
        storage.upsert_secondary_embeddings(
            [
                {
                    "chunk_id": chunk_id,
                    "content_hash": chunk.content_hash,
                    "embedding": _secondary_vec(axis),
                }
            ],
            "model2vec:v1",
        )
    storage.commit()

    hits = storage.search_secondary_embeddings(
        _secondary_vec(0),
        "model2vec:v1",
        k=1,
        repo_id=local_repo,
    )
    assert [hit.chunk_id for hit in hits] == ["sha256:local"]


def test_chunk_without_repo_id_yields_none_repo(storage: Storage) -> None:
    storage.upsert_chunk(
        _make_chunk("sha256:n", "Bare.java", "class Bare {}"),
        _vec(0), "m", 4, repo_id=None,
    )
    storage.commit()
    hits = storage.search_dense(_vec(0), k=1)
    assert hits and hits[0].repo is None


def test_clear_preserves_repos(storage: Storage) -> None:
    rid = storage.register_repo(_identity("github.com/keep/me"))
    storage.upsert_chunk(
        _make_chunk("sha256:k", "x.java", "x"),
        _vec(0), "m", 4, repo_id=rid,
    )
    storage.commit()
    storage.clear()
    # Chunks gone, repo preserved.
    assert storage.stats()["chunks"] == 0
    assert len(storage.list_repos()) == 1


def test_backfill_with_sanity_check(tmp_path: Path) -> None:
    """Auto-backfill assigns orphan chunks when target_repo paths match."""
    repo_root = tmp_path / "fake-repo"
    repo_root.mkdir()
    real_file = repo_root / "Hello.java"
    real_file.write_text("class Hello {}")

    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    # Insert orphan chunk pointing at a path that EXISTS under repo_root.
    s.upsert_chunk(
        _make_chunk("sha256:o", "Hello.java", "class Hello {}"),
        _vec(0), "m", 4, repo_id=None,
    )
    s.commit()

    assert s.count_chunks_without_repo() == 1
    rid = s.register_repo(_identity("github.com/q/r"))
    n = s.backfill_chunks_to_repo(rid, sample_root=repo_root)
    s.commit()
    assert n == 1
    assert s.count_chunks_without_repo() == 0


def test_backfill_refuses_when_paths_dont_match(tmp_path: Path) -> None:
    """Auto-backfill is a no-op when the target_repo changed since indexing."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    s.upsert_chunk(
        _make_chunk("sha256:o", "GhostFile.java", "x"),
        _vec(0), "m", 4, repo_id=None,
    )
    s.commit()
    rid = s.register_repo(_identity("github.com/x/y"))
    # tmp_path has no GhostFile.java → sanity check fails.
    n = s.backfill_chunks_to_repo(rid, sample_root=tmp_path)
    assert n == 0
    assert s.count_chunks_without_repo() == 1


# ---- independent secondary embeddings (schema v14 / WIN-300) ------------


def test_schema_v14_secondary_table_migrates_idempotently(tmp_path: Path) -> None:
    db = tmp_path / "v14.db"
    storage = Storage(db, embedding_dim=4)
    storage.set_meta("schema_version", "13")
    storage.commit()
    storage.close()

    upgraded = Storage(db, embedding_dim=4)
    assert SCHEMA_VERSION == 14
    assert upgraded.get_meta("schema_version") == "14"
    table = upgraded._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_chunks_secondary'"
    ).fetchone()
    assert table is not None and "float[256]" in table["sql"]
    upgraded.close()

    reopened = Storage(db, embedding_dim=4)
    assert reopened.get_meta("schema_version") == "14"


def test_secondary_embedding_batch_freshness_search_and_identity(
    storage: Storage,
) -> None:
    chunks = [
        _make_chunk("sha256:secondary-a", "A.java", "class A {}"),
        _make_chunk("sha256:secondary-b", "B.java", "class B {}"),
    ]
    for index, chunk in enumerate(chunks):
        storage.upsert_chunk(chunk, _vec(float(index)), "primary", 4)

    pending = storage.list_chunks_needing_secondary_embeddings("model2vec:v1")
    assert [item["chunk_id"] for item in pending] == [chunk.chunk_id for chunk in chunks]
    assert storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "embedding": _secondary_vec(index),
            }
            for index, chunk in enumerate(chunks)
        ],
        "model2vec:v1",
    ) == 2
    storage.commit()

    assert storage.list_chunks_needing_secondary_embeddings("model2vec:v1") == []
    assert storage.count_secondary_embeddings("model2vec:v1") == 2
    hits = storage.search_secondary_embeddings(
        _secondary_vec(0), "model2vec:v1", k=2
    )
    assert [hit.chunk_id for hit in hits] == [
        "sha256:secondary-a",
        "sha256:secondary-b",
    ]

    assert storage.count_secondary_embeddings("model2vec:v2") == 0
    assert storage.search_secondary_embeddings(_secondary_vec(0), "model2vec:v2") == []
    assert len(storage.list_chunks_needing_secondary_embeddings("model2vec:v2")) == 2


def test_secondary_embedding_rejects_stale_batch_item(storage: Storage) -> None:
    chunk = _make_chunk("sha256:secondary-stale", "Stale.java", "class Stale {}")
    storage.upsert_chunk(chunk, _vec(0), "primary", 4)

    assert storage.upsert_secondary_embeddings(
        [{"chunk_id": chunk.chunk_id, "content_hash": "old", "embedding": _secondary_vec(0)}],
        "model2vec:v1",
    ) == 0
    assert storage.count_secondary_embeddings("model2vec:v1") == 0

    assert storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "embedding": _secondary_vec(0),
            }
        ],
        "model2vec:v1",
    ) == 1
    storage._conn.execute(
        "UPDATE chunks SET content_hash = 'new-content' WHERE chunk_id = ?",
        (chunk.chunk_id,),
    )
    assert storage.count_secondary_embeddings("model2vec:v1") == 0
    assert storage.search_secondary_embeddings(_secondary_vec(0), "model2vec:v1") == []
    pending = storage.list_chunks_needing_secondary_embeddings("model2vec:v1")
    assert pending[0]["content_hash"] == "new-content"


def test_secondary_embedding_delete_clear_and_rename_lifecycle(storage: Storage) -> None:
    repo_id = storage.register_repo(_identity("github.com/secondary/lifecycle"))
    chunk = _make_chunk("sha256:secondary-life", "Old.java", "class Life {}")
    storage.upsert_chunk(chunk, _vec(0), "primary", 4, repo_id=repo_id)
    storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "embedding": _secondary_vec(0),
            }
        ],
        "model2vec:v1",
    )

    assert storage.rename_chunks(repo_id, "Old.java", "New.java") == 1
    assert storage.count_secondary_embeddings("model2vec:v1") == 1
    renamed_hit = storage.search_secondary_embeddings(
        _secondary_vec(0), "model2vec:v1", k=1
    )[0]
    assert renamed_hit.rel_path == "New.java"
    assert renamed_hit.chunk_id != chunk.chunk_id

    assert storage.delete_chunk(renamed_hit.chunk_id) is True
    assert storage.count_secondary_embeddings("model2vec:v1") == 0

    replacement = _make_chunk("sha256:secondary-clear", "Clear.java", "class Clear {}")
    storage.upsert_chunk(replacement, _vec(1), "primary", 4, repo_id=repo_id)
    storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": replacement.chunk_id,
                "content_hash": replacement.content_hash,
                "embedding": _secondary_vec(1),
            }
        ],
        "model2vec:v1",
    )
    storage.clear()
    assert storage.count_secondary_embeddings("model2vec:v1") == 0


# ---- faceted search tests ----


def _make_chunk_with_lang(
    chunk_id: str,
    rel_path: str,
    content: str,
    language: str = "java",
    kind: str = "method",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        language=language,
        kind=kind,
        symbol=None,
        start_line=1,
        end_line=10,
        content=content,
        content_hash=chunk_id,
    )


def test_compute_facets_full_index(tmp_path: Path) -> None:
    """compute_facets(None) returns counts for entire index."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    # Insert chunks with different languages and kinds
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:b", "Bar.java", "void bar()", "java", "method"),
        _vec(1), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:c", "baz.py", "def baz():", "python", "function"),
        _vec(2), "m", 4,
    )
    s.commit()

    facets = s.compute_facets(None)

    assert facets["language"] == {"java": 2, "python": 1}
    assert facets["kind"] == {"class": 1, "method": 1, "function": 1}
    assert "(local)" in facets["repo"]  # no repo assigned


def test_compute_facets_for_specific_chunks(tmp_path: Path) -> None:
    """compute_facets(chunk_ids) returns counts only for those chunks."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:b", "Bar.java", "void bar()", "java", "method"),
        _vec(1), "m", 4,
    )
    s.upsert_chunk(
        _make_chunk_with_lang("sha256:c", "baz.py", "def baz():", "python", "function"),
        _vec(2), "m", 4,
    )
    s.commit()

    # Only get facets for Java chunks
    facets = s.compute_facets(["sha256:a", "sha256:b"])

    assert facets["language"] == {"java": 2}
    assert facets["kind"] == {"class": 1, "method": 1}


def test_compute_facets_empty_list(tmp_path: Path) -> None:
    """compute_facets([]) returns empty dicts."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    s.upsert_chunk(
        _make_chunk_with_lang("sha256:a", "Foo.java", "class Foo", "java", "class"),
        _vec(0), "m", 4,
    )
    s.commit()

    facets = s.compute_facets([])

    assert facets == {"language": {}, "kind": {}, "repo": {}}


def test_vacuum_reclaims_space_after_clear(tmp_path: Path) -> None:
    """VACUUM shrinks the file after clear() marks pages as free.

    SQLite's DELETE FROM does not shrink the file — it marks pages as
    free for reuse. VACUUM rewrites the file to reclaim that space.

    We insert enough chunks with large content to force the SQLite file
    to grow across multiple pages; only then can VACUUM demonstrably
    shrink the file.

    Note: Storage uses WAL journal mode so data initially lives in the
    .db-wal file. We use ``wal_checkpoint(TRUNCATE)`` to flush before
    measuring file sizes, mirroring what VACUUM does internally.
    """
    db = tmp_path / "test.db"
    # Use a larger embedding dim so each chunk consumes more bytes.
    s = Storage(db, embedding_dim=384)

    # Insert many large chunks to force multi-page file growth.
    large_content = "class Foo { " + ("x" * 10_000) + " }"  # ~10 KB each
    embedding = [0.1] * 384
    for i in range(50):
        chunk = _make_chunk(
            f"sha256:{i:04d}",
            f"file_{i}.java",
            large_content,
        )
        s.upsert_chunk(chunk, embedding, "m", 384)
    s.commit()
    # Force WAL flush so the main .db file reflects committed data.
    s._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size_after_insert = db.stat().st_size
    # Sanity check: the DB should have grown beyond a single page.
    assert size_after_insert > 100_000, (
        f"Test setup failed: DB only {size_after_insert} bytes after inserts"
    )

    # Free pages count BEFORE clear (should be near zero).
    free_before_clear = s._conn.execute("PRAGMA freelist_count").fetchone()[0]

    # Clear all data — file size should not change (pages just freed).
    s.clear()
    s._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    size_after_clear = db.stat().st_size
    free_after_clear = s._conn.execute("PRAGMA freelist_count").fetchone()[0]

    # Allow tiny variation (WAL journal cleanup) but file should not have
    # significantly shrunk.
    assert size_after_clear >= size_after_insert * 0.95, (
        "clear() should not significantly shrink the file (pages are reused)"
    )
    assert free_after_clear > free_before_clear, (
        "clear() should produce free pages"
    )

    # VACUUM should actually reduce the file size now.
    size_before, size_after = s.vacuum()
    assert size_after < size_before, (
        f"VACUUM should reduce file size: before={size_before}, after={size_after}"
    )
    # Sanity check: actual file size matches reported size_after.
    assert db.stat().st_size == size_after
    # Free pages should be zero after VACUUM.
    free_after_vacuum = s._conn.execute("PRAGMA freelist_count").fetchone()[0]
    assert free_after_vacuum == 0


def test_vacuum_on_empty_database(tmp_path: Path) -> None:
    """VACUUM on an empty / fresh DB should succeed without error."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)

    size_before, size_after = s.vacuum()
    assert size_before >= 0
    assert size_after >= 0
    # An empty DB may shrink slightly or stay the same — both are fine.
    assert size_after <= size_before


def test_dim_mismatch_without_force_rebuild_raises(tmp_path: Path) -> None:
    """Opening an existing index at a different dim MUST raise unless rebuild.

    This is the contract the consistency check protects: a silent dim shift
    would put mismatched vectors into vec_chunks and corrupt search results.
    """
    db = tmp_path / "test.db"

    # First open: dim=1024 (e.g. bge-m3 era).
    s = Storage(db, embedding_dim=1024)
    repo_id = s.register_repo(_identity())
    s.upsert_chunk(
        _make_chunk("c1", "a.py", "x = 1"),
        _vec(0, dim=1024),
        "model-a",
        1024,
        repo_id=repo_id,
        embedding_text="x = 1",
        enricher_id="noop",
    )
    s.upsert_secondary_embeddings(
        [{"chunk_id": "c1", "content_hash": "c1", "embedding": _secondary_vec(0)}],
        "model2vec:v1",
    )
    s.commit()
    s.close()

    # Reopen at dim=512 (e.g. voyage-code-3 era) without force_rebuild.
    with pytest.raises(StorageError, match=r"dim=1024.*dim=512"):
        Storage(db, embedding_dim=512)


def test_dim_mismatch_with_force_rebuild_recreates_vec_chunks(tmp_path: Path) -> None:
    """Regression: --rebuild MUST resolve a dim shift in one shot.

    Before the fix, ``stropha index --rebuild`` would still raise the dim
    mismatch error during ``build_stages()`` (which constructs the storage
    BEFORE Pipeline.run gets a chance to call clear()), leaving the user
    stuck in a loop where the suggested fix (--rebuild) did not actually
    work. force_rebuild=True at Storage.__init__ now drops + recreates
    vec_chunks at the new dim BEFORE the consistency check fires.
    """
    db = tmp_path / "test.db"

    # First open: dim=1024.
    s = Storage(db, embedding_dim=1024)
    repo_id = s.register_repo(_identity())
    s.upsert_chunk(
        _make_chunk("c1", "a.py", "x = 1"),
        _vec(0, dim=1024),
        "model-a",
        1024,
        repo_id=repo_id,
        embedding_text="x = 1",
        enricher_id="noop",
    )
    s.upsert_secondary_embeddings(
        [{"chunk_id": "c1", "content_hash": "c1", "embedding": _secondary_vec(0)}],
        "model2vec:v1",
    )
    s.commit()
    chunks_before = s._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    assert chunks_before == 1
    s.close()

    # Reopen at dim=512 with force_rebuild=True. Should NOT raise; should
    # silently drop the dim-coupled tables and recreate vec_chunks at 512.
    s2 = Storage(db, embedding_dim=512, force_rebuild=True)
    assert s2.dim == 512

    # All dim-coupled data is now cleared.
    assert s2._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0
    assert s2.count_secondary_embeddings("model2vec:v1") == 0

    # The vec_chunks table is now keyed at the new dim — inserting a
    # 512-vector works, while a 1024-vector would have failed.
    repo_id_2 = s2.register_repo(_identity())
    s2.upsert_chunk(
        _make_chunk("c2", "b.py", "y = 2"),
        _vec(0, dim=512),
        "model-b",
        512,
        repo_id=repo_id_2,
        embedding_text="y = 2",
        enricher_id="noop",
    )
    s2.commit()
    assert s2._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 1


def test_force_rebuild_is_noop_when_dim_unchanged(tmp_path: Path) -> None:
    """force_rebuild=True with matching dim must not destroy existing data.

    Use case: user runs --rebuild without changing the embedder. The
    storage path should be ready to clear via Pipeline.clear() but must
    not silently nuke the chunks at construction time.
    """
    db = tmp_path / "test.db"

    s = Storage(db, embedding_dim=4)
    repo_id = s.register_repo(_identity())
    s.upsert_chunk(
        _make_chunk("c1", "a.py", "x = 1"),
        _vec(0),
        "model-a",
        4,
        repo_id=repo_id,
        embedding_text="x = 1",
        enricher_id="noop",
    )
    s.upsert_secondary_embeddings(
        [{"chunk_id": "c1", "content_hash": "c1", "embedding": _secondary_vec(0)}],
        "model2vec:v1",
    )
    s.commit()
    s.close()

    # Reopen at the SAME dim with force_rebuild=True — should preserve data.
    # (Pipeline.run will call clear() separately when rebuild=True.)
    s2 = Storage(db, embedding_dim=4, force_rebuild=True)
    assert s2._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 1
    assert s2.count_secondary_embeddings("model2vec:v1") == 1


def test_delete_repo_removes_identity_and_every_scoped_row(storage: Storage) -> None:
    """`delete_repo` is the teardown `clear_repo` deliberately is not."""
    kept_id = storage.register_repo(_identity("github.com/example/kept"))
    doomed_id = storage.register_repo(_identity("github.com/example/doomed"))
    for repo_id, prefix in ((kept_id, "kept"), (doomed_id, "doomed")):
        storage.upsert_chunk(
            _make_chunk(f"sha256:{prefix}", f"{prefix}/Foo.java", "class Foo {}"),
            _vec(0),
            "test-model",
            4,
            repo_id=repo_id,
        )
        storage.upsert_file_meta(
            repo_id=repo_id,
            rel_path=f"{prefix}/Foo.java",
            content_hash="hash",
            size_bytes=10,
            mtime=1.0,
            chunk_count=1,
        )
        storage._conn.execute(
            "INSERT INTO project_memory (repo_id, kind, key, value, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (repo_id, "note", f"{prefix}-key", "remembered", "2026-08-23", "2026-08-23"),
        )
    storage.commit()

    removed = storage.delete_repo(doomed_id)

    assert removed["chunks"] == 1
    assert removed["files"] == 1
    assert removed["project_memory"] == 1
    assert removed["repos"] == 1
    # The identity is gone, so a stale or duplicate entry stops being listed.
    assert [repo.normalized_key for repo in storage.list_repos()] == ["github.com/example/kept"]
    assert storage.lookup_repo_id("github.com/example/doomed") is None
    # Everything scoped to the surviving repository is untouched.
    assert storage.lookup_repo_id("github.com/example/kept") == kept_id
    survivors = storage._conn.execute(
        "SELECT count(*) AS n FROM project_memory WHERE repo_id = ?", (kept_id,)
    ).fetchone()["n"]
    assert survivors == 1
    assert storage._conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 1


def test_clear_repo_still_keeps_the_identity_for_rebuilds(storage: Storage) -> None:
    """The rebuild path must keep reusing the same repo id."""
    repo_id = storage.register_repo(_identity("github.com/example/repo"))
    storage.upsert_chunk(
        _make_chunk("sha256:a", "Foo.java", "class Foo {}"),
        _vec(0),
        "test-model",
        4,
        repo_id=repo_id,
    )
    storage.commit()

    storage.clear_repo(repo_id)

    assert storage.lookup_repo_id("github.com/example/repo") == repo_id
    assert storage._conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0


def test_repo_scoped_meta_prefixes_match_the_pipeline() -> None:
    """The delete list must not drift from the keys the pipeline writes."""
    from stropha.models import SOURCE_SNAPSHOT_META_PREFIX
    from stropha.pipeline import pipeline as pipeline_module
    from stropha.storage_schema import REPO_SCOPED_META_PREFIXES

    written = {
        "last_indexed_sha_",
        SOURCE_SNAPSHOT_META_PREFIX,
        pipeline_module._GRAPH_DIRTY_META_PREFIX,
        pipeline_module._CHUNKER_ID_META_PREFIX,
        pipeline_module._WALKER_ID_META_PREFIX,
    }
    assert set(REPO_SCOPED_META_PREFIXES) == written


def test_delete_repo_leaves_no_meta_rows_behind(storage: Storage) -> None:
    """Repo-scoped meta keys carry the id in the key, so they need naming."""
    from stropha.storage_schema import repo_scoped_meta_keys

    kept_id = storage.register_repo(_identity("github.com/example/kept"))
    doomed_id = storage.register_repo(_identity("github.com/example/doomed"))
    for repo_id in (kept_id, doomed_id):
        for key in repo_scoped_meta_keys(repo_id):
            storage.set_meta(key, "value")
    storage.commit()

    storage.delete_repo(doomed_id)

    assert all(storage.get_meta(key) is None for key in repo_scoped_meta_keys(doomed_id))
    assert all(storage.get_meta(key) == "value" for key in repo_scoped_meta_keys(kept_id))
