"""Integration tests for the PostgreSQL + pgvector storage adapter.

These run against a live Postgres (the local Docker instance started by
``STROPHA_USE_POSTGRES=1 bin/stropha-mcp-ensure``). When no Postgres is
reachable the whole module is skipped, so ``uv run pytest -q`` stays green on
machines without one.

Each run provisions a throwaway ``stropha_pgtest`` database and drops it after,
so it never touches real index data.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

psycopg = pytest.importorskip("psycopg")

from stropha.adapters.storage.postgres import (  # noqa: E402
    PostgresStorage,
    PostgresStorageConfig,
)
from stropha.ingest.git_meta import RepoIdentity  # noqa: E402
from stropha.models import Chunk  # noqa: E402

BASE_DSN = os.environ.get(
    "STROPHA_PG_DSN", "postgresql://stropha:stropha@127.0.0.1:54329/stropha"
)
TEST_DB = "stropha_pgtest"
DIM = 8


def _swap_db(dsn: str, db: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit((parts.scheme, parts.netloc, f"/{db}", parts.query, parts.fragment))


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _emb(seed: int) -> list[float]:
    return _norm([math.sin(seed + i) for i in range(DIM)])


def _chunk(i: int, path: str, symbol: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"sha256:t{i}",
        rel_path=path,
        language="python",
        kind="method",
        symbol=symbol,
        start_line=1,
        end_line=2,
        content=content,
        content_hash=f"h{i}",
    )


@pytest.fixture(scope="module")
def pg_dsn() -> Iterator[str]:
    try:
        admin = psycopg.connect(BASE_DSN, autocommit=True)
    except psycopg.Error:
        pytest.skip("Postgres not reachable at STROPHA_PG_DSN")
    try:
        with admin.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (TEST_DB,))
            if cur.fetchone():
                cur.execute(f'DROP DATABASE "{TEST_DB}";')
            cur.execute(f'CREATE DATABASE "{TEST_DB}";')
        yield _swap_db(BASE_DSN, TEST_DB)
    finally:
        with admin.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid();",
                (TEST_DB,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}";')
        admin.close()


@pytest.fixture
def store(pg_dsn: str) -> Iterator[PostgresStorage]:
    st = PostgresStorage(PostgresStorageConfig(dsn=pg_dsn), embedding_dim=DIM)
    st.clear()
    yield st
    st.close()


def _seed(st: PostgresStorage) -> int:
    repo_id = st.register_repo(
        RepoIdentity(
            normalized_key="local:/tmp/demo",
            remote_url=None,
            root_path=Path("/tmp/demo"),
            default_branch="main",
            head_commit="abc",
        )
    )
    rows = [
        ("auth.py", "validate_token", "def validate_token(t): return verify(t)"),
        ("auth.py", "login_user", "def login_user(u, p): return check(u, p)"),
        ("db.py", "ConnectionPool", "class ConnectionPool: ..."),
        ("search.py", "search_dense", "def search_dense(vec, k): return knn(vec, k)"),
    ]
    for i, (path, sym, content) in enumerate(rows):
        st.upsert_chunk(_chunk(i, path, sym, content), _emb(i), "demo", DIM,
                        repo_id=repo_id, enricher_id="noop")
    st.commit()
    return repo_id


def test_migrate_and_stats(store: PostgresStorage) -> None:
    _seed(store)
    s = store.stats()
    assert s["chunks"] == 4
    assert s["files"] == 3
    assert s["index_dim"] == DIM
    assert "***" in s["db_path"]  # password redacted


def test_search_dense_orders_by_similarity(store: PostgresStorage) -> None:
    _seed(store)
    hits = store.search_dense(_emb(0), k=3)
    assert hits, "expected dense hits"
    assert hits[0].symbol == "validate_token"  # exact-match vector
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert hits[0].repo is not None
    assert hits[0].repo.normalized_key == "local:/tmp/demo"


def test_search_bm25_finds_tokens(store: PostgresStorage) -> None:
    _seed(store)
    hits = store.search_bm25("validate token", k=5)
    assert any(h.symbol == "validate_token" for h in hits)


def test_search_symbol_tokens_camelcase(store: PostgresStorage) -> None:
    _seed(store)
    hits = store.search_symbol_tokens("ConnectionPool", k=5)
    assert hits[0].symbol == "ConnectionPool"
    assert hits[0].score >= 1.0


def test_chunk_freshness_drift(store: PostgresStorage) -> None:
    _seed(store)
    assert store.chunk_is_fresh("sha256:t0", "h0", "demo", "noop") is True
    assert store.chunk_is_fresh("sha256:t0", "DIFFERENT", "demo", "noop") is False
    assert store.chunk_is_fresh("sha256:t0", "h0", "OTHER-MODEL", "noop") is False


def test_clear_preserves_repos(store: PostgresStorage) -> None:
    _seed(store)
    store.clear()
    assert store.stats()["chunks"] == 0
    # repo identity survives a rebuild
    assert store.lookup_repo_id("local:/tmp/demo") is not None


def test_project_memory_roundtrip(store: PostgresStorage) -> None:
    repo_id = _seed(store)
    rec = store.remember(repo_id=repo_id, kind="decision", key="k1", value="v1")
    assert rec.id
    got = store.recall(repo_id=repo_id, kind="decision")
    assert got and got[0].value == "v1"
    # upsert preserves created_at, updates value
    store.remember(repo_id=repo_id, kind="decision", key="k1", value="v2")
    again = store.recall(repo_id=repo_id, key="k1")
    assert again[0].value == "v2"
    assert again[0].created_at == rec.created_at
    assert store.forget(repo_id=repo_id, kind="decision", key="k1") is True
    assert store.recall(repo_id=repo_id, key="k1") == []


def test_episodes_and_co_occurrence(store: PostgresStorage) -> None:
    repo_id = _seed(store)
    ep = store.record_episode(
        repo_id=repo_id, title="fix fsrs", summary="did the thing",
        touched_files=["a.py", "b.py"], touched_symbols=["Foo"],
    )
    assert ep.id
    found = store.recall_episodes(repo_id=repo_id, query="FSRS")  # ILIKE
    assert found and found[0].title == "fix fsrs"
    store.record_episode(repo_id=repo_id, title="more", touched_files=["a.py", "c.py"])
    co = store.episode_co_occurrence(repo_id=repo_id, key="a.py")
    assert "b.py" in co["files"] and "c.py" in co["files"]


def test_compute_facets(store: PostgresStorage) -> None:
    _seed(store)
    facets = store.compute_facets()
    assert facets["language"].get("python") == 4
    ids = [h.chunk_id for h in store.search_dense(_emb(0), k=2)]
    scoped = store.compute_facets(ids)
    assert sum(scoped["language"].values()) == len(ids)
    assert store.compute_facets([]) == {"language": {}, "kind": {}, "repo": {}}


def test_cross_session_cache_via_conn_shim(store: PostgresStorage) -> None:
    from stropha.retrieval import cross_session_cache as csc

    repo_id = _seed(store)
    csc.store(
        store, repo_id=repo_id, query_text="how does auth work",
        query_embedding=_emb(0), embedding_model="m", embedding_dim=DIM,
        results=[{"chunk_id": "x", "score": 1.0}],
    )
    hits = csc.find_similar(
        store, repo_id=repo_id, query_embedding=_emb(0),
        embedding_model="m", similarity_threshold=0.5,
    )
    assert hits and hits[0].results == [{"chunk_id": "x", "score": 1.0}]
    assert csc.stats(store, repo_id=repo_id)["total_entries"] == 1
    assert csc.prune(store, repo_id=repo_id) == 1


def test_graph_loaders_and_traversal_via_conn_shim(store: PostgresStorage) -> None:
    """Exercise the _conn shim with the graph subsystem's SQLite-dialect SQL.

    Inserts a 2-node 1-edge graph via `INSERT OR REPLACE` (translated to
    ON CONFLICT) and runs the graph.py traversal layer against Postgres.
    """
    from stropha.retrieval.graph import find_callers, graph_loaded, resolve_symbol_to_node

    repo_id = _seed(store)
    conn = store._conn  # the compatibility shim
    conn.execute(
        "INSERT OR REPLACE INTO graph_nodes "
        "(node_id, label, source_file, source_location, repo_id) VALUES (?, ?, ?, ?, ?)",
        ("n1", "auth.validate_token", "auth.py", "2:4", repo_id),
    )
    conn.execute(
        "INSERT OR REPLACE INTO graph_nodes "
        "(node_id, label, source_file, source_location, repo_id) VALUES (?, ?, ?, ?, ?)",
        ("n2", "api.handler", "api.py", "9:4", repo_id),
    )
    conn.execute(
        "INSERT OR REPLACE INTO graph_edges "
        "(source, target, relation, confidence, source_file, source_location, repo_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("n2", "n1", "calls", "EXTRACTED", "api.py", "9:4", repo_id),
    )
    conn.execute(
        "INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?, ?)",
        ("last_loaded_at", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    assert graph_loaded(store) is True
    assert resolve_symbol_to_node(store, "auth.validate_token") == "n1"
    res = find_callers(store, "auth.validate_token")
    assert res["resolved_node"] == "n1"
    assert len(res["edges"]) >= 1  # the n2 -> n1 'calls' edge was traversed

    # graph stats surface through the standard stats() dict
    gstats = store.stats()["graph"]
    assert gstats is not None and gstats["nodes"] == 2 and gstats["edges_total"] == 1


def test_sql_shim_translation() -> None:
    from stropha.adapters.storage.postgres import _translate_sqlite_sql

    assert _translate_sqlite_sql("SELECT * FROM t WHERE a = ?") == "SELECT * FROM t WHERE a = %s"
    assert _translate_sqlite_sql("BEGIN IMMEDIATE") is None
    assert _translate_sqlite_sql("PRAGMA table_info(x)") is None
    isnot = _translate_sqlite_sql("SELECT 1 FROM n WHERE embedding_model IS NOT ?")
    assert isnot is not None and "IS DISTINCT FROM %s" in isnot
    meta = _translate_sqlite_sql("INSERT OR REPLACE INTO graph_meta (key, value) VALUES (?, ?)")
    assert meta is not None and "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value" in meta
    nodes = _translate_sqlite_sql(
        "INSERT OR REPLACE INTO graph_nodes (node_id, label) VALUES (?, ?)"
    )
    assert nodes is not None and "ON CONFLICT (node_id) DO UPDATE SET label=EXCLUDED.label" in nodes


def test_reads_not_blocked_by_open_write_txn(pg_dsn: str) -> None:
    """The whole point: an open write transaction must not block readers.

    On SQLite this is exactly the scenario that raises 'database is locked';
    Postgres MVCC lets the reader proceed and hides the uncommitted row.
    """
    writer = PostgresStorage(PostgresStorageConfig(dsn=pg_dsn), embedding_dim=DIM)
    reader = PostgresStorage(PostgresStorageConfig(dsn=pg_dsn), embedding_dim=DIM)
    try:
        writer.clear()
        repo_id = _seed(writer)  # commits 4 chunks

        # Writer opens a transaction and inserts WITHOUT committing.
        writer.upsert_chunk(
            _chunk(99, "pending.py", "pending_symbol", "def pending(): ..."),
            _emb(99), "demo", DIM, repo_id=repo_id, enricher_id="noop",
        )

        # Reader (separate connection) is not blocked and does not see the
        # uncommitted row.
        hits = reader.search_dense(_emb(0), k=10)
        assert hits, "reader should return committed data while writer txn is open"
        assert all(h.chunk_id != "sha256:t99" for h in hits)
        assert reader.chunk_exists("sha256:t99") is False

        # After commit the row becomes visible to a fresh read.
        writer.commit()
        assert reader.chunk_exists("sha256:t99") is True
    finally:
        writer.close()
        reader.close()
