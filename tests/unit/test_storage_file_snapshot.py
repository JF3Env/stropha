"""Atomic file-snapshot contract tests for the SQLite storage adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from stropha.errors import StorageError
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk
from stropha.storage import Storage

if TYPE_CHECKING:
    from stropha.stages.storage import FileSnapshotChunkWrite

DIM = 4


def _identity(key: str) -> RepoIdentity:
    return RepoIdentity(
        normalized_key=key,
        remote_url=f"https://{key}.git",
        root_path=Path(f"/tmp/{key}"),
        default_branch="main",
        head_commit="abc123",
    )


def _chunk(
    chunk_id: str,
    rel_path: str,
    content: str,
    *,
    content_hash: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        rel_path=rel_path,
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=2,
        content=content,
        content_hash=content_hash or f"hash:{chunk_id}",
    )


def _embedding(axis: int, dim: int = DIM) -> list[float]:
    vector = [0.0] * dim
    vector[axis % dim] = 1.0
    return vector


def _write(chunk: Chunk, axis: int = 0) -> FileSnapshotChunkWrite:
    return {
        "chunk": chunk,
        "embedding": _embedding(axis),
        "embedding_text": f"enriched:{chunk.content}",
    }


def _replace(
    storage: Storage,
    *,
    repo_id: int,
    rel_path: str,
    desired: list[str],
    writes: list[FileSnapshotChunkWrite],
    file_hash: str = "file-hash",
    size: int = 100,
    mtime: float = 123.5,
    reused: list[Chunk] | None = None,
) -> None:
    storage.replace_file_snapshot(
        repo_id=repo_id,
        rel_path=rel_path,
        desired_chunk_ids=desired,
        chunk_writes=writes,
        embedding_model="primary:v1",
        embedding_dim=DIM,
        enricher_id="enricher:v1",
        file_content_hash=file_hash,
        size_bytes=size,
        mtime=mtime,
        reused_chunks=reused,
    )


def _path_chunk_ids(storage: Storage, repo_id: int, rel_path: str) -> list[str]:
    rows = storage._conn.execute(
        "SELECT chunk_id FROM chunks WHERE repo_id = ? AND rel_path = ? ORDER BY chunk_id",
        (repo_id, rel_path),
    ).fetchall()
    return [str(row["chunk_id"]) for row in rows]


def _satellite_count(storage: Storage, table: str, rowid: int) -> int:
    row = storage._conn.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE rowid = ?", (rowid,)
    ).fetchone()
    return int(row["n"])


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "snapshot.db", embedding_dim=DIM)


def test_replace_prunes_fewer_chunks_scopes_repo_and_retains_secondary(
    storage: Storage,
) -> None:
    local_repo = storage.register_repo(_identity("github.com/example/local-snapshot"))
    foreign_repo = storage.register_repo(_identity("github.com/example/foreign-snapshot"))
    rel_path = "shared.py"
    local = [
        _chunk("local-a", rel_path, "def retained(): pass"),
        _chunk("local-b", rel_path, "def old_token(): pass", content_hash="old-b"),
        _chunk("local-c", rel_path, "def stale(): pass"),
    ]
    foreign = _chunk("foreign-a", rel_path, "def foreign(): pass")
    _replace(
        storage,
        repo_id=local_repo,
        rel_path=rel_path,
        desired=[chunk.chunk_id for chunk in local],
        writes=[_write(chunk, index) for index, chunk in enumerate(local)],
        file_hash="local-v1",
    )
    _replace(
        storage,
        repo_id=foreign_repo,
        rel_path=rel_path,
        desired=[foreign.chunk_id],
        writes=[_write(foreign)],
        file_hash="foreign-v1",
    )
    storage.commit()

    rowids = {
        row["chunk_id"]: int(row["id"])
        for row in storage._conn.execute(
            "SELECT id, chunk_id FROM chunks WHERE repo_id = ?", (local_repo,)
        ).fetchall()
    }
    storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": local[0].chunk_id,
                "content_hash": local[0].content_hash,
                "embedding": _embedding(0, 256),
            },
            {
                "chunk_id": foreign.chunk_id,
                "content_hash": foreign.content_hash,
                "embedding": _embedding(1, 256),
            },
        ],
        "secondary:v1",
    )
    storage.commit()

    dirty_b = _chunk(
        "local-b",
        rel_path,
        "def replacement_token(): pass",
        content_hash="new-b",
    )
    _replace(
        storage,
        repo_id=local_repo,
        rel_path=rel_path,
        desired=[local[0].chunk_id, dirty_b.chunk_id],
        writes=[_write(dirty_b, 3)],
        file_hash="local-v2",
        size=222,
        mtime=456.25,
    )

    assert _path_chunk_ids(storage, local_repo, rel_path) == ["local-a", "local-b"]
    assert _path_chunk_ids(storage, foreign_repo, rel_path) == ["foreign-a"]
    retained = storage._conn.execute(
        "SELECT id FROM chunks WHERE chunk_id = 'local-a'"
    ).fetchone()
    assert retained is not None and int(retained["id"]) == rowids["local-a"]
    assert _satellite_count(storage, "vec_chunks_secondary", rowids["local-a"]) == 1
    assert storage.count_secondary_embeddings("secondary:v1") == 2
    assert [
        hit.chunk_id
        for hit in storage.search_bm25("replacement token", repo_id=local_repo)
    ] == ["local-b"]

    file_row = storage._conn.execute(
        """SELECT content_hash, size_bytes, mtime, last_chunk_count,
                  last_enricher_id, last_embedder_model
           FROM files WHERE repo_id = ? AND rel_path = ?""",
        (local_repo, rel_path),
    ).fetchone()
    assert file_row is not None
    assert dict(file_row) == {
        "content_hash": "local-v2",
        "size_bytes": 222,
        "mtime": 456.25,
        "last_chunk_count": 2,
        "last_enricher_id": "enricher:v1",
        "last_embedder_model": "primary:v1",
    }
    for table in ("vec_chunks", "vec_chunks_secondary", "fts_chunks"):
        assert _satellite_count(storage, table, rowids["local-c"]) == 0

    storage.commit()
    assert storage.delete_file_snapshot(repo_id=local_repo, rel_path=rel_path) == 2
    assert _path_chunk_ids(storage, local_repo, rel_path) == []
    assert _path_chunk_ids(storage, foreign_repo, rel_path) == ["foreign-a"]
    assert storage.count_secondary_embeddings("secondary:v1") == 1
    assert storage._conn.execute(
        "SELECT 1 FROM files WHERE repo_id = ? AND rel_path = ?",
        (local_repo, rel_path),
    ).fetchone() is None
    storage.commit()

    empty_source = _chunk("local-empty-source", rel_path, "pass")
    _replace(
        storage,
        repo_id=local_repo,
        rel_path=rel_path,
        desired=[empty_source.chunk_id],
        writes=[_write(empty_source)],
    )
    storage.commit()
    _replace(
        storage,
        repo_id=local_repo,
        rel_path=rel_path,
        desired=[],
        writes=[],
        file_hash="empty-file",
        size=0,
        mtime=999.0,
    )
    assert _path_chunk_ids(storage, local_repo, rel_path) == []
    empty_meta = storage._conn.execute(
        "SELECT content_hash, size_bytes, mtime, last_chunk_count FROM files "
        "WHERE repo_id = ? AND rel_path = ?",
        (local_repo, rel_path),
    ).fetchone()
    assert empty_meta is not None and dict(empty_meta) == {
        "content_hash": "empty-file",
        "size_bytes": 0,
        "mtime": 999.0,
        "last_chunk_count": 0,
    }
    assert storage.delete_file_snapshot(repo_id=local_repo, rel_path=rel_path) == 0
    assert storage._conn.execute(
        "SELECT 1 FROM files WHERE repo_id = ? AND rel_path = ?",
        (local_repo, rel_path),
    ).fetchone() is None


def test_reused_chunk_refreshes_metadata_and_fts_without_replacing_vectors(
    storage: Storage,
) -> None:
    repo_id = storage.register_repo(_identity("github.com/example/reused-metadata"))
    rel_path = "module.py"
    original = _chunk("stable", rel_path, "def body(): pass").model_copy(
        update={"symbol": "OldMetadata", "parent_chunk_id": "old-parent"}
    )
    _replace(
        storage,
        repo_id=repo_id,
        rel_path=rel_path,
        desired=[original.chunk_id],
        writes=[_write(original, 2)],
    )
    storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": original.chunk_id,
                "content_hash": original.content_hash,
                "embedding": _embedding(3, 256),
            }
        ],
        "secondary:v1",
    )
    storage.commit()
    row = storage._conn.execute(
        "SELECT id FROM chunks WHERE chunk_id = ?", (original.chunk_id,)
    ).fetchone()
    assert row is not None
    rowid = int(row["id"])
    primary_before = storage._conn.execute(
        "SELECT embedding FROM vec_chunks WHERE rowid = ?", (rowid,)
    ).fetchone()["embedding"]
    secondary_before = storage._conn.execute(
        "SELECT embedding FROM vec_chunks_secondary WHERE rowid = ?", (rowid,)
    ).fetchone()["embedding"]

    refreshed = original.model_copy(
        update={
            "kind": "method",
            "symbol": "RenamedMetadata",
            "parent_chunk_id": "new-parent",
            "start_line": 3,
            "end_line": 4,
        }
    )
    _replace(
        storage,
        repo_id=repo_id,
        rel_path=rel_path,
        desired=[refreshed.chunk_id],
        writes=[],
        reused=[refreshed],
    )

    updated = storage._conn.execute(
        """SELECT kind, symbol, parent_chunk_id, start_line, end_line
           FROM chunks WHERE id = ?""",
        (rowid,),
    ).fetchone()
    assert updated is not None
    assert dict(updated) == {
        "kind": "method",
        "symbol": "RenamedMetadata",
        "parent_chunk_id": "new-parent",
        "start_line": 3,
        "end_line": 4,
    }
    assert storage._conn.execute(
        "SELECT embedding FROM vec_chunks WHERE rowid = ?", (rowid,)
    ).fetchone()["embedding"] == primary_before
    assert storage._conn.execute(
        "SELECT embedding FROM vec_chunks_secondary WHERE rowid = ?", (rowid,)
    ).fetchone()["embedding"] == secondary_before
    assert [
        hit.chunk_id for hit in storage.search_bm25("RenamedMetadata", repo_id=repo_id)
    ] == [original.chunk_id]


def test_snapshot_methods_leave_commit_control_to_caller(tmp_path: Path) -> None:
    db_path = tmp_path / "commit-control.db"
    storage = Storage(db_path, embedding_dim=DIM)
    repo_id = storage.register_repo(_identity("github.com/example/commit-control"))
    storage.commit()
    chunk = _chunk("pending", "pending.py", "def pending(): pass")

    _replace(
        storage,
        repo_id=repo_id,
        rel_path=chunk.rel_path,
        desired=[chunk.chunk_id],
        writes=[_write(chunk)],
    )
    assert storage._conn.in_transaction is True
    with sqlite3.connect(db_path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0

    storage.commit()
    with sqlite3.connect(db_path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    assert storage.delete_file_snapshot(repo_id=repo_id, rel_path=chunk.rel_path) == 1
    with sqlite3.connect(db_path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    storage.commit()
    with sqlite3.connect(db_path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    storage.close()


def test_replace_validates_ids_scope_paths_and_dimensions_before_writes(
    storage: Storage,
) -> None:
    local_repo = storage.register_repo(_identity("github.com/example/validation-local"))
    foreign_repo = storage.register_repo(_identity("github.com/example/validation-foreign"))
    storage.commit()
    chunk = _chunk("candidate", "target.py", "pass")

    with pytest.raises(StorageError, match="must be unique"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=[chunk.chunk_id, chunk.chunk_id],
            writes=[_write(chunk)],
        )
    with pytest.raises(StorageError, match="not in desired_chunk_ids"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=[],
            writes=[_write(chunk)],
        )
    with pytest.raises(StorageError, match="Duplicate snapshot write"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=[chunk.chunk_id],
            writes=[_write(chunk), _write(chunk)],
        )
    with pytest.raises(StorageError, match=r"expected 'target\.py'"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=["wrong-path"],
            writes=[_write(_chunk("wrong-path", "other.py", "pass"))],
        )
    with pytest.raises(StorageError, match="Snapshot embedding dim"):
        storage.replace_file_snapshot(
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired_chunk_ids=[chunk.chunk_id],
            chunk_writes=[_write(chunk)],
            embedding_model="primary:v1",
            embedding_dim=DIM + 1,
            enricher_id="enricher:v1",
            file_content_hash="file",
            size_bytes=1,
            mtime=1.0,
        )
    bad_embedding = _write(chunk)
    bad_embedding["embedding"] = [1.0]
    with pytest.raises(StorageError, match="Embedding dim mismatch"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=[chunk.chunk_id],
            writes=[bad_embedding],
        )
    with pytest.raises(StorageError, match="no stored row or write"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=["missing"],
            writes=[],
        )

    foreign = _chunk("foreign", chunk.rel_path, "pass")
    _replace(
        storage,
        repo_id=foreign_repo,
        rel_path=chunk.rel_path,
        desired=[foreign.chunk_id],
        writes=[_write(foreign)],
    )
    storage.commit()
    with pytest.raises(StorageError, match="outside repo_id"):
        _replace(
            storage,
            repo_id=local_repo,
            rel_path=chunk.rel_path,
            desired=[foreign.chunk_id],
            writes=[],
        )

    assert _path_chunk_ids(storage, local_repo, chunk.rel_path) == []
    assert _path_chunk_ids(storage, foreign_repo, chunk.rel_path) == [foreign.chunk_id]
    assert storage._conn.execute(
        "SELECT 1 FROM files WHERE repo_id = ? AND rel_path = ?",
        (local_repo, chunk.rel_path),
    ).fetchone() is None


def test_replace_failure_rolls_back_only_snapshot_work(storage: Storage) -> None:
    repo_id = storage.register_repo(_identity("github.com/example/replace-rollback"))
    rel_path = "rollback.py"
    retained = _chunk("rollback-a", rel_path, "def a(): pass")
    stale = _chunk("rollback-b", rel_path, "def b(): pass")
    _replace(
        storage,
        repo_id=repo_id,
        rel_path=rel_path,
        desired=[retained.chunk_id, stale.chunk_id],
        writes=[_write(retained), _write(stale, 1)],
        file_hash="before",
    )
    storage.upsert_secondary_embeddings(
        [
            {
                "chunk_id": retained.chunk_id,
                "content_hash": retained.content_hash,
                "embedding": _embedding(0, 256),
            }
        ],
        "secondary:v1",
    )
    storage.commit()
    stale_rowid = int(
        storage._conn.execute(
            "SELECT id FROM chunks WHERE chunk_id = ?", (stale.chunk_id,)
        ).fetchone()["id"]
    )
    storage._conn.execute(
        """CREATE TRIGGER reject_snapshot_meta
           BEFORE UPDATE OF content_hash ON files
           WHEN NEW.content_hash = 'reject'
           BEGIN SELECT RAISE(ABORT, 'reject snapshot'); END;"""
    )
    storage.commit()

    pending = _chunk("pending-before-snapshot", "unrelated.py", "pass")
    storage.upsert_chunk(pending, _embedding(2), "primary:v1", DIM, repo_id=repo_id)
    replacement = _chunk("rollback-c", rel_path, "def c(): pass")
    with pytest.raises(sqlite3.IntegrityError, match="reject snapshot"):
        _replace(
            storage,
            repo_id=repo_id,
            rel_path=rel_path,
            desired=[retained.chunk_id, replacement.chunk_id],
            writes=[_write(replacement, 3)],
            file_hash="reject",
        )

    assert _path_chunk_ids(storage, repo_id, rel_path) == ["rollback-a", "rollback-b"]
    assert storage.chunk_exists(replacement.chunk_id) is False
    assert storage.chunk_exists(pending.chunk_id) is True
    assert _satellite_count(storage, "vec_chunks", stale_rowid) == 1
    assert _satellite_count(storage, "fts_chunks", stale_rowid) == 1
    assert storage.count_secondary_embeddings("secondary:v1") == 1
    assert storage._conn.execute(
        "SELECT content_hash FROM files WHERE repo_id = ? AND rel_path = ?",
        (repo_id, rel_path),
    ).fetchone()["content_hash"] == "before"
    assert storage._conn.execute("SELECT 1").fetchone()[0] == 1
    storage.commit()
    storage._conn.execute("DROP TRIGGER reject_snapshot_meta")
    storage.commit()


def test_delete_failure_restores_path_state_and_keeps_connection_usable(
    storage: Storage,
) -> None:
    repo_id = storage.register_repo(_identity("github.com/example/delete-rollback"))
    chunk = _chunk("delete-rollback", "delete.py", "pass")
    _replace(
        storage,
        repo_id=repo_id,
        rel_path=chunk.rel_path,
        desired=[chunk.chunk_id],
        writes=[_write(chunk)],
    )
    storage.commit()
    rowid = int(
        storage._conn.execute(
            "SELECT id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()["id"]
    )
    storage._conn.execute(
        """CREATE TRIGGER reject_snapshot_delete
           BEFORE DELETE ON files
           BEGIN SELECT RAISE(ABORT, 'reject delete'); END;"""
    )
    storage.commit()

    pending = _chunk("pending-before-delete", "unrelated.py", "pass")
    storage.upsert_chunk(pending, _embedding(1), "primary:v1", DIM, repo_id=repo_id)
    with pytest.raises(sqlite3.IntegrityError, match="reject delete"):
        storage.delete_file_snapshot(repo_id=repo_id, rel_path=chunk.rel_path)

    assert storage.chunk_exists(chunk.chunk_id) is True
    assert storage.chunk_exists(pending.chunk_id) is True
    assert _satellite_count(storage, "vec_chunks", rowid) == 1
    assert _satellite_count(storage, "fts_chunks", rowid) == 1
    assert storage._conn.execute(
        "SELECT 1 FROM files WHERE repo_id = ? AND rel_path = ?",
        (repo_id, chunk.rel_path),
    ).fetchone() is not None
    assert storage._conn.execute("SELECT 1").fetchone()[0] == 1
    storage.commit()
    storage._conn.execute("DROP TRIGGER reject_snapshot_delete")
    storage.commit()
