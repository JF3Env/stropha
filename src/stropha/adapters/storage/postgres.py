"""Storage adapter for PostgreSQL + pgvector + native full-text search.

This is the MVCC-backed sibling of the ``sqlite-vec`` adapter. It implements
the same :class:`stropha.stages.storage.StorageStage` protocol so the Pipeline,
retrieval streams and CLI depend only on the abstract surface — they neither
know nor care that the backend is Postgres.

Why this exists
---------------
SQLite serializes writers and takes a database-wide exclusive lock during
``VACUUM``/rebuild, so a long indexing run blocks the live MCP read path
("database is locked"). PostgreSQL uses MVCC: readers never block writers and
writers never block readers, so the index can be rebuilt while the server keeps
serving. See ``docs/architecture`` for the migration rationale.

Mapping from the sqlite-vec schema
----------------------------------
- ``vec_chunks`` (vec0 virtual table) -> ``chunks.embedding vector(N)`` (pgvector),
  KNN via the L2 operator ``<->``. Vectors are assumed normalized, so the same
  ``score = max(0, 1 - d²/2)`` transform reproduces cosine similarity.
- ``fts_chunks`` (FTS5 contentless) -> ``chunks.fts tsvector`` (GIN index),
  ranked with ``ts_rank_cd``. The indexed document is the SAME augmented text
  produced by :func:`stropha.storage.sqlite._fts_text` (CamelCase split, doubled
  file stem, path/dir tokens) so BM25-style ordering stays comparable.

Phase 1 scope: the full index + dense/bm25/symbol search vertical. Graph,
session, episode and project-memory tables are not yet ported (those tools fall
back to their graph-gated empty responses on this backend).

The heavy ``psycopg`` import is deferred to :meth:`__init__` so importing this
module (which the adapter autoloader does for *every* install) stays cheap and
never fails when the optional ``postgres`` extra is absent.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...errors import StorageError
from ...logging import get_logger
from ...models import (
    Chunk,
    EpisodeRecord,
    ProjectMemoryRecord,
    RepoStats,
    SearchHit,
    SessionRecord,
    StoredFileMeta,
)
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter

# Pure text helpers reused verbatim so FTS/symbol tokenization matches the
# sqlite-vec backend exactly. These pull in ``storage.sqlite`` (and therefore
# ``sqlite_vec``, a core dependency), which is cheap and always installed.
from ...storage.sqlite import (
    SCHEMA_VERSION,
    _fts_text,
    _identifier_tokens,
    _repo_from_row,
    _snippet,
)

if TYPE_CHECKING:
    from ...ingest.git_meta import RepoIdentity

log = get_logger(__name__)

# Shared SELECT list: chunk columns + repo_* aliases consumed by
# ``_repo_from_row``. Mirrors ``_REPO_JOIN_COLUMNS`` in storage/sqlite.py.
_REPO_COLS = """
    c.chunk_id, c.rel_path, c.language, c.kind, c.symbol,
    c.start_line, c.end_line, c.content,
    r.normalized_key AS repo_normalized_key,
    r.remote_url     AS repo_url,
    r.root_path      AS repo_local_path,
    r.default_branch AS repo_default_branch,
    r.head_commit    AS repo_head_commit
"""


def _vec_literal(vec: list[float]) -> str:
    """pgvector text input form: ``[1.0,2.0,3.0]`` (cast with ``::vector``)."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _pg_tsquery(query: str) -> str:
    """Turn free text into a safe ``to_tsquery('simple', ...)`` OR-expression.

    Mirrors ``_sanitize_fts_query`` (CamelCase split + drop ≤1-char tokens) but
    emits Postgres tsquery syntax (``tok1 | tok2``) instead of FTS5's
    ``"tok1" OR "tok2"``. Tokens are lowercased and stripped to word chars so
    the query never trips tsquery's grammar.
    """
    import re

    from ...storage.sqlite import _split_identifiers

    cleaned = re.sub(r"[^\w\s.]+", " ", query)
    cleaned = _split_identifiers(cleaned)
    tokens = []
    for tok in cleaned.split():
        norm = re.sub(r"\W", "", tok).lower()
        if len(norm) > 1:
            tokens.append(norm)
    # de-dup preserving order
    seen: set[str] = set()
    uniq = [t for t in tokens if not (t in seen or seen.add(t))]
    return " | ".join(uniq)


# --------------------------------------------------------------------------
# SQLite-dialect compatibility shim for the ``_conn`` escape hatch.
#
# Several modules (retrieval/graph.py, retrieval/cross_session_cache.py,
# ingest/graphify_loader.py, ingest/graph_vec_loader.py, ingest/glossary.py,
# the graph-vec stream and graph-aware enricher) reach into ``storage._conn``
# and run SQLite-dialect SQL. Rather than rewrite all of them, ``_conn`` on the
# Postgres backend returns this shim, which translates the SQLite-isms those
# modules use into Postgres on the fly:
#   * ``?``           -> ``%s``
#   * ``x IS NOT ?``  -> ``x IS DISTINCT FROM %s``
#   * ``INSERT OR REPLACE INTO t (...)`` -> ``... ON CONFLICT (<pk>) DO UPDATE``
#   * ``INSERT OR IGNORE INTO t (...)``  -> ``... ON CONFLICT DO NOTHING``
#   * ``BEGIN [IMMEDIATE]`` / ``PRAGMA ...`` -> swallowed (psycopg owns txns)
# The backend's own typed methods use the raw psycopg connection (``self._raw``)
# directly, so this translation never touches the index/search hot paths.
# --------------------------------------------------------------------------

# Conflict targets for the ``INSERT OR REPLACE`` statements issued via _conn.
_UPSERT_PK: dict[str, list[str]] = {
    "graph_nodes": ["node_id"],
    "graph_edges": ["source", "target", "relation", "source_file", "source_location"],
    "graph_meta": ["key"],
}

# Session advisory-lock key that serializes concurrent index writers so two
# `stropha index` runs can't deadlock on chunks/files. Arbitrary constant.
_INDEX_LOCK_KEY = 0x53545250  # "STRP"


def _translate_sqlite_sql(sql: str) -> str | None:
    """Translate a SQLite-dialect statement to Postgres. ``None`` = swallow."""
    import re

    stripped = sql.lstrip()
    upper = stripped[:7].upper()
    if upper.startswith("BEGIN") or upper.startswith("PRAGMA"):
        return None

    m = re.match(
        r"^\s*INSERT\s+OR\s+(REPLACE|IGNORE)\s+INTO\s+(\w+)\s*\(([^)]*)\)",
        sql,
        flags=re.IGNORECASE,
    )
    if m:
        mode = m.group(1).upper()
        table = m.group(2).lower()
        cols = [c.strip() for c in m.group(3).split(",")]
        base = re.sub(
            r"^\s*INSERT\s+OR\s+(REPLACE|IGNORE)\s+INTO",
            "INSERT INTO",
            sql,
            count=1,
            flags=re.IGNORECASE,
        )
        pk = _UPSERT_PK.get(table)
        if mode == "IGNORE" or pk is None:
            sql = base.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        else:
            non_pk = [c for c in cols if c not in pk]
            set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in non_pk)
            if not set_clause:
                set_clause = f"{pk[0]}=EXCLUDED.{pk[0]}"
            sql = (
                base.rstrip().rstrip(";")
                + f" ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {set_clause}"
            )

    # `x IS NOT ?` is valid SQLite but a syntax error in Postgres.
    sql = re.sub(r"\bIS\s+NOT\s+\?", "IS DISTINCT FROM ?", sql, flags=re.IGNORECASE)
    # qmark -> psycopg pyformat. The codebase never embeds a literal `?`.
    return sql.replace("?", "%s")


class _Row:
    """sqlite3.Row-like row: supports positional (``row[0]``), by-name
    (``row["col"]``), ``.get()``, ``.keys()``, iteration and ``dict(row)``.

    The graph/cache/glossary consumers mix all of these access styles
    (e.g. the graphify loader reads ``row[0]`` while graph.py reads
    ``row["col"]``), which psycopg's ``dict_row`` / ``tuple_row`` cannot
    satisfy alone.
    """

    __slots__ = ("_cols", "_map", "_vals")

    def __init__(self, cols: list[str], vals: Any) -> None:
        self._cols = cols
        self._vals = vals
        self._map = {c: vals[i] for i, c in enumerate(cols)}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._vals[key]
        return self._map[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._map.get(key, default)

    def keys(self) -> list[str]:
        return self._cols

    def __iter__(self) -> Any:
        return iter(self._vals)

    def __len__(self) -> int:
        return len(self._vals)

    def __contains__(self, key: Any) -> bool:
        return key in self._map


def _sqlite_like_row(cursor: Any) -> Any:
    def make(values: Any) -> _Row:
        cols = [d.name for d in (cursor.description or [])]
        return _Row(cols, values)

    return make


class _CompatCursor:
    """Cursor wrapper that translates SQLite-dialect SQL before executing."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    def execute(self, sql: str, params: Any = ()) -> _CompatCursor:
        translated = _translate_sqlite_sql(sql)
        if translated is not None:
            self._cur.execute(translated, params)
        return self

    def executemany(self, sql: str, seq: Any) -> _CompatCursor:
        translated = _translate_sqlite_sql(sql)
        if translated is not None:
            self._cur.executemany(translated, list(seq))
        return self

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> Any:
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def lastrowid(self) -> Any:
        return None

    def close(self) -> None:
        self._cur.close()

    def __iter__(self) -> Any:
        return iter(self._cur)

    def __enter__(self) -> _CompatCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _CompatConn:
    """``storage._conn`` shim presenting a sqlite3-like API over psycopg."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def cursor(self) -> _CompatCursor:
        # sqlite3.Row-like rows so external consumers can use both positional
        # (``row[0]`` — the graphify loader) and by-name (``row["col"]`` —
        # graph.py) access on the same result.
        return _CompatCursor(self._raw.cursor(row_factory=_sqlite_like_row))

    def execute(self, sql: str, params: Any = ()) -> _CompatCursor:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, seq: Any) -> _CompatCursor:
        cur = self.cursor()
        cur.executemany(sql, seq)
        return cur

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    @property
    def in_transaction(self) -> bool:
        import psycopg

        return self._raw.info.transaction_status != psycopg.pq.TransactionStatus.IDLE


class PostgresStorageConfig(BaseModel):
    dsn: str = Field(
        default_factory=lambda: os.environ.get(
            "STROPHA_PG_DSN",
            "postgresql://stropha:stropha@127.0.0.1:54329/stropha",
        ),
        description=(
            "libpq connection string for the Postgres+pgvector backend. "
            "Defaults to the local Docker instance started by "
            "`STROPHA_USE_POSTGRES=1 bin/stropha-mcp-ensure`. Override via the "
            "STROPHA_PG_DSN env var or YAML `storage.config.dsn`."
        ),
    )


@register_adapter(stage="storage", name="postgres")
class PostgresStorage:
    """StorageStage backed by PostgreSQL + pgvector + tsvector."""

    Config = PostgresStorageConfig

    # Graph subsystem is ported: graphify mirror load + find_* traversal run
    # against the Postgres graph tables (via the typed methods + the _conn
    # compatibility shim). Pipeline.run() loads the graph mirror for this
    # backend.
    graph_capable = True

    def __init__(
        self,
        config: PostgresStorageConfig | None = None,
        *,
        embedding_dim: int | None = None,
        force_rebuild: bool = False,
    ) -> None:
        if config is None:
            config = PostgresStorageConfig()
        if embedding_dim is None:
            raise ValueError(
                "PostgresStorage requires `embedding_dim=` (propagate from "
                "the active embedder)."
            )
        self._adapter_config = config
        self._dsn = config.dsn
        self._dim = embedding_dim
        self._index_lock_held = False

        try:
            import psycopg
            from psycopg.rows import dict_row
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise StorageError(
                "PostgresStorage needs psycopg. Install the extra: "
                "`uv pip install 'psycopg[binary]>=3.2'` (or `pip install "
                "'stropha[postgres]'`)."
            ) from exc

        try:
            self._raw = psycopg.connect(self._dsn, row_factory=dict_row, autocommit=False)
        except psycopg.Error as exc:
            raise StorageError(
                f"Could not connect to Postgres at {self._dsn!r}: {exc}. Is it "
                f"running? Try `STROPHA_USE_POSTGRES=1 bin/stropha-mcp-ensure`."
            ) from exc

        # ``_conn`` is the StorageStage escape hatch. External consumers
        # (graph traversal, cross_session_cache, glossary, graphify loaders)
        # issue SQLite-dialect SQL through it, so it is the compatibility shim.
        # The backend's own methods use ``self._raw`` (native psycopg) directly.
        self._conn = _CompatConn(self._raw)

        self._migrate()
        if force_rebuild:
            self._reset_dim_dependent_tables_if_needed()
        self._check_dimension_consistency()

    # ----- schema ---------------------------------------------------------

    def _migrate(self) -> None:
        with self._raw.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key   text PRIMARY KEY,
                    value text NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS repos (
                    id               bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    normalized_key   text NOT NULL UNIQUE,
                    remote_url       text,
                    root_path        text NOT NULL,
                    default_branch   text,
                    head_commit      text,
                    first_indexed_at text NOT NULL,
                    last_indexed_at  text NOT NULL
                );
                """
            )
            self._create_chunk_tables(cur)
            self._create_aux_tables(cur)
            cur.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', %s) "
                "ON CONFLICT (key) DO NOTHING;",
                (str(SCHEMA_VERSION),),
            )
            cur.execute(
                "INSERT INTO meta(key, value) VALUES ('embedding_dim', %s) "
                "ON CONFLICT (key) DO NOTHING;",
                (str(self._dim),),
            )
        self._raw.commit()

    def _create_chunk_tables(self, cur: Any) -> None:
        """Create chunks/files/enrichments + indexes. Idempotent."""
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id              bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                chunk_id        text NOT NULL UNIQUE,
                rel_path        text NOT NULL,
                language        text NOT NULL,
                kind            text NOT NULL,
                symbol          text,
                parent_chunk_id text,
                start_line      integer NOT NULL,
                end_line        integer NOT NULL,
                content         text NOT NULL,
                content_hash    text NOT NULL,
                embedding_model text NOT NULL,
                embedding_dim   integer NOT NULL,
                indexed_at      text NOT NULL,
                repo_id         bigint REFERENCES repos(id),
                embedding_text  text,
                enricher_id     text,
                embedding       vector({self._dim}),
                fts             tsvector
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_enricher ON chunks(enricher_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(fts);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks "
            "USING hnsw (embedding vector_l2_ops);"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id                  bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                repo_id             bigint NOT NULL REFERENCES repos(id),
                rel_path            text NOT NULL,
                content_hash        text NOT NULL,
                size_bytes          bigint NOT NULL,
                mtime               double precision NOT NULL,
                last_indexed_at     text NOT NULL,
                last_chunk_count    integer NOT NULL DEFAULT 0,
                last_enricher_id    text,
                last_embedder_model text,
                UNIQUE (repo_id, rel_path)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_repo_path ON files(repo_id, rel_path);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS enrichments (
                content_hash   text NOT NULL,
                enricher_id    text NOT NULL,
                embedding_text text NOT NULL,
                cached_at      text NOT NULL,
                PRIMARY KEY (content_hash, enricher_id)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_enrichments_enricher ON enrichments(enricher_id);"
        )

    def _create_aux_tables(self, cur: Any) -> None:
        """Non-dim-dependent tables. Created once; NOT dropped on dim reset.

        Currently just ``sessions`` (the MCP server persists per-session agent
        state here). Project-memory / episodes / graph tables are a later phase;
        their MCP tools are not yet wired to this backend.
        """
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id       text PRIMARY KEY,
                client_id        text,
                created_at       text NOT NULL,
                updated_at       text NOT NULL,
                seen_chunk_ids   text NOT NULL DEFAULT '[]',
                pinned_chunk_ids text NOT NULL DEFAULT '[]',
                meta             text NOT NULL DEFAULT '{}',
                scratchpad       text NOT NULL DEFAULT ''
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON sessions(client_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);")

        # ----- project memory (v9/v11) -----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS project_memory (
                id                bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                repo_id           bigint NOT NULL REFERENCES repos(id),
                kind              text NOT NULL,
                key               text NOT NULL,
                value             text NOT NULL,
                source            text,
                created_at        text NOT NULL,
                updated_at        text NOT NULL,
                expires_at        text,
                provenance_commit text,
                UNIQUE (repo_id, kind, key)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pm_repo ON project_memory(repo_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pm_repo_kind ON project_memory(repo_id, kind);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pm_expires ON project_memory(expires_at);")

        # ----- episodes (v10/v11) -----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id                bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                repo_id           bigint NOT NULL REFERENCES repos(id),
                session_id        text,
                title             text NOT NULL,
                summary           text NOT NULL DEFAULT '',
                touched_files     text NOT NULL DEFAULT '[]',
                touched_symbols   text NOT NULL DEFAULT '[]',
                created_at        text NOT NULL,
                updated_at        text NOT NULL,
                provenance_commit text
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_repo_created "
            "ON episodes(repo_id, created_at DESC);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);")

        # ----- cross_session_cache (v13) -----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cross_session_cache (
                id               bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                repo_id          bigint NOT NULL REFERENCES repos(id),
                query_text       text NOT NULL,
                query_embedding  bytea NOT NULL,
                embedding_model  text NOT NULL,
                embedding_dim    integer NOT NULL,
                results_json     text NOT NULL,
                session_id       text,
                created_at       text NOT NULL,
                last_hit_at      text,
                hit_count        integer NOT NULL DEFAULT 0,
                expires_at       text
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_csc_repo ON cross_session_cache(repo_id);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_csc_model "
            "ON cross_session_cache(embedding_model, embedding_dim);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_csc_expires ON cross_session_cache(expires_at);")

        # ----- graph subsystem (v4/v5/v7). Edge PK columns are NOT NULL DEFAULT
        # '' because Postgres forbids NULL in a PRIMARY KEY (SQLite allows it,
        # which is why the loader already coerces them to ''). Embeddings are
        # stored as the same little-endian float32 BLOB the loaders pack, so the
        # graph-vec stream's struct.unpack works unchanged. -----
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id          text PRIMARY KEY,
                label            text NOT NULL,
                file_type        text,
                source_file      text,
                source_location  text,
                community_id     integer,
                community_label  text,
                embedding        bytea,
                embedding_model  text,
                embedding_dim    integer,
                repo_id          bigint REFERENCES repos(id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_label ON graph_nodes(label);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_source_file ON graph_nodes(source_file);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_community ON graph_nodes(community_id);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo ON graph_nodes(repo_id);")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                source           text NOT NULL,
                target           text NOT NULL,
                relation         text NOT NULL,
                confidence       text NOT NULL,
                confidence_score real,
                context          text,
                source_file      text NOT NULL DEFAULT '',
                source_location  text NOT NULL DEFAULT '',
                weight           real,
                repo_id          bigint REFERENCES repos(id),
                PRIMARY KEY (source, target, relation, source_file, source_location)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_target_relation "
            "ON graph_edges(target, relation, confidence);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_source_relation "
            "ON graph_edges(source, relation, confidence);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_relation "
            "ON graph_edges(relation, confidence);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_repo ON graph_edges(repo_id);")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS graph_meta (key text PRIMARY KEY, value text NOT NULL);"
        )

    def _check_dimension_consistency(self) -> None:
        stored = self.get_meta("embedding_dim")
        if stored is not None and int(stored) != self._dim:
            raise StorageError(
                f"Index was built with embedding_dim={stored} but the active "
                f"embedder produces dim={self._dim}. Re-index with --rebuild to "
                f"switch embedding models."
            )
        if stored is None:
            self.set_meta("embedding_dim", str(self._dim))
            self._raw.commit()

    def _reset_dim_dependent_tables_if_needed(self) -> None:
        """On --rebuild with a changed dim, drop the vector-typed tables.

        pgvector ``vector(N)`` is fixed-width per column (like sqlite-vec's
        vec0), so a dim change requires recreating ``chunks``. Repos are
        preserved so ids stay stable across rebuilds.
        """
        stored = self.get_meta("embedding_dim")
        if stored is None or int(stored) == self._dim:
            return
        log.info("postgres.dim_change_reset", old=stored, new=self._dim)
        with self._raw.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS files CASCADE;")
            cur.execute("DROP TABLE IF EXISTS enrichments CASCADE;")
            self._create_chunk_tables(cur)
            cur.execute("DELETE FROM meta WHERE key = 'embedding_dim';")
        self._raw.commit()

    # ----- Stage contract -------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def stage_name(self) -> str:
        return "storage"

    @property
    def adapter_name(self) -> str:
        return "postgres"

    @property
    def adapter_id(self) -> str:
        return f"postgres:dim={self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return PostgresStorageConfig

    def health(self) -> StageHealth:
        try:
            stats = self.stats()
            return StageHealth(
                status="ready",
                message=f"postgres ready ({stats['chunks']} chunks)",
                detail={
                    "dsn": self._sanitized_dsn(),
                    "chunks": str(stats["chunks"]),
                    "dim": str(stats["index_dim"]),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            return StageHealth(status="error", message=f"postgres probe failed: {exc}")

    def _sanitized_dsn(self) -> str:
        """DSN with any password redacted, for logs/stats."""
        import re

        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", self._dsn)

    @property
    def storage_target(self) -> str:
        """Where this backend persists data: the (password-redacted) DSN.

        Surfaced by the ``get_config`` MCP tool (``pipeline.storage.path``)
        so callers see the real Postgres target instead of the unused
        SQLite ``index.db`` path.
        """
        return self._sanitized_dsn()

    # ----- Lifecycle ------------------------------------------------------

    def commit(self) -> None:
        self._raw.commit()

    def clear(self) -> None:
        """Wipe the index, preserving repo identities (mirrors sqlite clear())."""
        with self._raw.cursor() as cur:
            cur.execute("DELETE FROM chunks;")
            cur.execute("DELETE FROM files;")
            cur.execute("DELETE FROM enrichments;")
        self._raw.commit()

    def vacuum(self) -> tuple[int, int]:
        """Reclaim space. Postgres autovacuums online; this is a manual nudge.

        Returns ``(size_before, size_after)`` of the whole database in bytes.
        """
        size_before = self._db_size()
        self._raw.commit()
        prev = self._raw.autocommit
        self._raw.autocommit = True
        try:
            self._raw.execute("VACUUM;")
        finally:
            self._raw.autocommit = prev
        return (size_before, self._db_size())

    def _db_size(self) -> int:
        row = self._raw.execute("SELECT pg_database_size(current_database()) AS n;").fetchone()
        return int(row["n"]) if row else 0

    def try_acquire_index_lock(self) -> bool:
        """Serialize concurrent index writers via a Postgres session advisory
        lock. Returns ``False`` when another connection already holds it
        (another ``stropha index`` is running) — which prevents the
        chunks/files deadlock two concurrent indexers would otherwise hit.
        Readers/search are unaffected (advisory locks don't block queries).
        """
        row = self._raw.execute(
            "SELECT pg_try_advisory_lock(%s) AS ok;", (_INDEX_LOCK_KEY,)
        ).fetchone()
        self._raw.commit()
        self._index_lock_held = bool(row and row["ok"])
        return self._index_lock_held

    def release_index_lock(self) -> None:
        if not self._index_lock_held:
            return
        try:
            self._raw.execute("SELECT pg_advisory_unlock(%s);", (_INDEX_LOCK_KEY,))
            self._raw.commit()
        except Exception:  # pragma: no cover - defensive
            pass
        self._index_lock_held = False

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.release_index_lock()
        with contextlib.suppress(Exception):
            self._raw.close()

    # ----- Writes ---------------------------------------------------------

    def register_repo(self, identity: RepoIdentity) -> int:
        now = datetime.now(UTC).isoformat()
        row = self._raw.execute(
            """
            INSERT INTO repos (normalized_key, remote_url, root_path,
                               default_branch, head_commit,
                               first_indexed_at, last_indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (normalized_key) DO UPDATE SET
                remote_url      = EXCLUDED.remote_url,
                root_path       = EXCLUDED.root_path,
                default_branch  = EXCLUDED.default_branch,
                head_commit     = EXCLUDED.head_commit,
                last_indexed_at = EXCLUDED.last_indexed_at
            RETURNING id;
            """,
            (
                identity.normalized_key,
                identity.remote_url,
                str(identity.root_path),
                identity.default_branch,
                identity.head_commit,
                now,
                now,
            ),
        ).fetchone()
        self._raw.commit()
        return int(row["id"]) if row else 0

    def upsert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
        embedding_model: str,
        embedding_dim: int,
        repo_id: int | None = None,
        embedding_text: str | None = None,
        enricher_id: str | None = None,
    ) -> int:
        if len(embedding) != self._dim:
            raise StorageError(
                f"Embedding dim {len(embedding)} != index dim {self._dim} "
                f"for chunk {chunk.chunk_id}"
            )
        now = datetime.now(UTC).isoformat()
        effective_text = embedding_text if embedding_text is not None else chunk.content
        fts_doc = _fts_text(chunk.content, chunk.rel_path, chunk.symbol)
        row = self._raw.execute(
            """
            INSERT INTO chunks (
                chunk_id, rel_path, language, kind, symbol, parent_chunk_id,
                start_line, end_line, content, content_hash,
                embedding_model, embedding_dim, indexed_at, repo_id,
                embedding_text, enricher_id, embedding, fts
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s::vector, to_tsvector('simple', %s)
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                rel_path        = EXCLUDED.rel_path,
                language        = EXCLUDED.language,
                kind            = EXCLUDED.kind,
                symbol          = EXCLUDED.symbol,
                parent_chunk_id = EXCLUDED.parent_chunk_id,
                start_line      = EXCLUDED.start_line,
                end_line        = EXCLUDED.end_line,
                content         = EXCLUDED.content,
                content_hash    = EXCLUDED.content_hash,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim   = EXCLUDED.embedding_dim,
                indexed_at      = EXCLUDED.indexed_at,
                repo_id         = EXCLUDED.repo_id,
                embedding_text  = EXCLUDED.embedding_text,
                enricher_id     = EXCLUDED.enricher_id,
                embedding       = EXCLUDED.embedding,
                fts             = EXCLUDED.fts
            RETURNING id;
            """,
            (
                chunk.chunk_id, chunk.rel_path, chunk.language, chunk.kind,
                chunk.symbol, chunk.parent_chunk_id, chunk.start_line,
                chunk.end_line, chunk.content, chunk.content_hash,
                embedding_model, embedding_dim, now, repo_id,
                effective_text, enricher_id, _vec_literal(embedding), fts_doc,
            ),
        ).fetchone()
        return int(row["id"]) if row else 0

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
        enricher_id: str | None = None,
    ) -> bool:
        row = self._raw.execute(
            "SELECT content_hash, embedding_model, enricher_id FROM chunks WHERE chunk_id = %s;",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return False
        if row["content_hash"] != content_hash:
            return False
        if row["embedding_model"] != embedding_model:
            return False
        if enricher_id is not None:
            stored = row["enricher_id"] or "noop"
            if stored != enricher_id:
                return False
        return True

    def get_enrichment(self, content_hash: str, enricher_id: str) -> str | None:
        row = self._raw.execute(
            "SELECT embedding_text FROM enrichments WHERE content_hash = %s AND enricher_id = %s;",
            (content_hash, enricher_id),
        ).fetchone()
        return row["embedding_text"] if row else None

    def put_enrichment(self, content_hash: str, enricher_id: str, embedding_text: str) -> None:
        now = datetime.now(UTC).isoformat()
        self._raw.execute(
            """
            INSERT INTO enrichments (content_hash, enricher_id, embedding_text, cached_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (content_hash, enricher_id) DO UPDATE SET
                embedding_text = EXCLUDED.embedding_text,
                cached_at      = EXCLUDED.cached_at;
            """,
            (content_hash, enricher_id, embedding_text, now),
        )

    def count_chunks_without_repo(self) -> int:
        row = self._raw.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE repo_id IS NULL;"
        ).fetchone()
        return int(row["n"]) if row else 0

    def backfill_chunks_to_repo(self, repo_id: int, sample_root: Any) -> int:
        rows = self._raw.execute(
            "SELECT rel_path FROM chunks WHERE repo_id IS NULL LIMIT 5;"
        ).fetchall()
        if not rows:
            return 0
        root = Path(sample_root)
        if not any((root / r["rel_path"]).is_file() for r in rows):
            log.warning("storage.backfill_skipped_sanity", repo_id=repo_id)
            return 0
        cur = self._raw.execute(
            "UPDATE chunks SET repo_id = %s WHERE repo_id IS NULL;", (repo_id,)
        )
        return int(cur.rowcount or 0)

    def set_meta(self, key: str, value: str) -> None:
        self._raw.execute(
            "INSERT INTO meta(key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._raw.execute("SELECT value FROM meta WHERE key = %s;", (key,)).fetchone()
        return row["value"] if row else None

    # ----- File-level incremental ops -------------------------------------

    def file_is_fresh(
        self,
        repo_id: int,
        rel_path: str,
        *,
        mtime: float,
        size_bytes: int,
        enricher_id: str | None = None,
        embedder_model: str | None = None,
    ) -> bool:
        row = self._raw.execute(
            "SELECT mtime, size_bytes, last_enricher_id, last_embedder_model "
            "FROM files WHERE repo_id = %s AND rel_path = %s;",
            (repo_id, rel_path),
        ).fetchone()
        if row is None:
            return False
        if float(row["mtime"]) != float(mtime):
            return False
        if int(row["size_bytes"]) != int(size_bytes):
            return False
        if enricher_id is not None and row["last_enricher_id"] != enricher_id:
            return False
        return not (embedder_model is not None and row["last_embedder_model"] != embedder_model)

    def load_file_meta_map(self, repo_id: int) -> dict[str, StoredFileMeta]:
        """Bulk-load freshness columns for every ``files`` row of a repo.

        Postgres mirror of ``Storage.load_file_meta_map`` (WIN-425): one
        query for the whole repo so the reconcile path diffs the working
        tree in memory instead of one ``file_is_fresh`` SELECT per file.
        """
        rows = self._raw.execute(
            "SELECT rel_path, mtime, size_bytes, last_enricher_id, last_embedder_model "
            "FROM files WHERE repo_id = %s;",
            (repo_id,),
        ).fetchall()
        return {
            r["rel_path"]: StoredFileMeta(
                mtime=float(r["mtime"]),
                size_bytes=int(r["size_bytes"]),
                enricher_id=r["last_enricher_id"],
                embedder_model=r["last_embedder_model"],
            )
            for r in rows
        }

    def upsert_file_meta(
        self,
        *,
        repo_id: int,
        rel_path: str,
        content_hash: str,
        size_bytes: int,
        mtime: float,
        chunk_count: int,
        enricher_id: str | None = None,
        embedder_model: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self._raw.execute(
            """
            INSERT INTO files
              (repo_id, rel_path, content_hash, size_bytes, mtime,
               last_indexed_at, last_chunk_count, last_enricher_id, last_embedder_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_id, rel_path) DO UPDATE SET
                content_hash        = EXCLUDED.content_hash,
                size_bytes          = EXCLUDED.size_bytes,
                mtime               = EXCLUDED.mtime,
                last_indexed_at     = EXCLUDED.last_indexed_at,
                last_chunk_count    = EXCLUDED.last_chunk_count,
                last_enricher_id    = EXCLUDED.last_enricher_id,
                last_embedder_model = EXCLUDED.last_embedder_model;
            """,
            (repo_id, rel_path, content_hash, size_bytes, mtime, now,
             chunk_count, enricher_id, embedder_model),
        )

    def delete_file_meta(self, repo_id: int, rel_paths: list[str]) -> int:
        if not rel_paths:
            return 0
        cur = self._raw.execute(
            "DELETE FROM files WHERE repo_id = %s AND rel_path = ANY(%s);",
            (repo_id, list(rel_paths)),
        )
        return int(cur.rowcount or 0)

    def list_stale_files(self, repo_id: int, current_rel_paths: list[str]) -> list[str]:
        rows = self._raw.execute(
            "SELECT rel_path FROM files WHERE repo_id = %s;", (repo_id,)
        ).fetchall()
        current = set(current_rel_paths)
        return [r["rel_path"] for r in rows if r["rel_path"] not in current]

    def rename_chunks(self, repo_id: int, old_rel_path: str, new_rel_path: str) -> int:
        if old_rel_path == new_rel_path:
            return 0
        from ...ingest.chunkers.base import make_chunk_id

        rows = self._raw.execute(
            """
            SELECT c.id, c.start_line, c.end_line, c.content, c.symbol,
                   c.content_hash, r.normalized_key AS repo_key
            FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.repo_id = %s AND c.rel_path = %s;
            """,
            (repo_id, old_rel_path),
        ).fetchall()
        moved = 0
        for row in rows:
            new_id = make_chunk_id(
                new_rel_path,
                row["start_line"],
                row["end_line"],
                row["content_hash"],
                repo_key=row["repo_key"],
            )
            fts_doc = _fts_text(row["content"], new_rel_path, row["symbol"])
            self._raw.execute(
                "UPDATE chunks SET rel_path = %s, chunk_id = %s, "
                "fts = to_tsvector('simple', %s) WHERE id = %s;",
                (new_rel_path, new_id, fts_doc, row["id"]),
            )
            moved += 1
        self._patch_file_meta_rename(repo_id, old_rel_path, new_rel_path)
        return moved

    def _patch_file_meta_rename(self, repo_id: int, old: str, new: str) -> None:
        self._raw.execute(
            "UPDATE files SET rel_path = %s WHERE repo_id = %s AND rel_path = %s "
            "AND NOT EXISTS (SELECT 1 FROM files WHERE repo_id = %s AND rel_path = %s);",
            (new, repo_id, old, repo_id, new),
        )
        self._raw.execute(
            "DELETE FROM files WHERE repo_id = %s AND rel_path = %s;", (repo_id, old)
        )

    def delete_chunks_by_repo_paths(self, repo_id: int, rel_paths: list[str]) -> int:
        if not rel_paths:
            return 0
        cur = self._raw.execute(
            "DELETE FROM chunks WHERE repo_id = %s AND rel_path = ANY(%s);",
            (repo_id, list(rel_paths)),
        )
        return int(cur.rowcount or 0)

    def lookup_repo_id(self, normalized_key: str) -> int | None:
        row = self._raw.execute(
            "SELECT id FROM repos WHERE normalized_key = %s;", (normalized_key,)
        ).fetchone()
        return int(row["id"]) if row else None

    def chunk_exists(self, chunk_id: str) -> bool:
        row = self._raw.execute(
            "SELECT 1 AS x FROM chunks WHERE chunk_id = %s LIMIT 1;", (chunk_id,)
        ).fetchone()
        return row is not None

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._raw.execute(
            "SELECT * FROM chunks WHERE chunk_id = %s;", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_chunk(self, chunk_id: str) -> bool:
        cur = self._raw.execute("DELETE FROM chunks WHERE chunk_id = %s;", (chunk_id,))
        return bool(cur.rowcount)

    def lookup_by_chunk_id(self, chunk_id: str) -> SearchHit | None:
        row = self._raw.execute(
            f"""
            SELECT {_REPO_COLS}
            FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.chunk_id = %s;
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hit(row, rank=0, score=1.0)

    def augment_fts_with_graph(self) -> dict[str, int]:
        """Re-augment chunk tsvectors with graph node labels/communities.

        Mirrors the sqlite-vec ``augment_fts_with_graph`` but writes the
        ``chunks.fts`` tsvector directly (no FTS5 satellite table). No-op
        sentinel when the graph mirror is empty.
        """
        probe = self._raw.execute("SELECT 1 AS x FROM graph_nodes LIMIT 1;").fetchone()
        if probe is None:
            return {"augmented": 0, "unchanged": 0, "missing_graph": 1}
        rows = self._raw.execute(
            """
            SELECT c.id, c.rel_path, c.symbol, c.content,
                   n.label, n.community_label, n.source_location
            FROM chunks c
            JOIN graph_nodes n ON n.source_file = c.rel_path
            WHERE n.source_location IS NOT NULL;
            """
        ).fetchall()
        augmented = 0
        for r in rows:
            extras = " ".join(
                str(x) for x in (r["label"], r["community_label"], r["source_location"]) if x
            )
            doc = _fts_text(r["content"], r["rel_path"], r["symbol"]) + "\n" + extras
            self._raw.execute(
                "UPDATE chunks SET fts = to_tsvector('simple', %s) WHERE id = %s;",
                (doc, r["id"]),
            )
            augmented += 1
        self._raw.commit()
        return {"augmented": augmented, "unchanged": 0, "missing_graph": 0}

    def _graph_stats(self) -> dict[str, Any] | None:
        row = self._raw.execute(
            "SELECT value FROM graph_meta WHERE key = 'last_loaded_at';"
        ).fetchone()
        if row is None:
            return None
        n_nodes = self._raw.execute("SELECT COUNT(*) AS n FROM graph_nodes;").fetchone()["n"]
        n_edges = self._raw.execute("SELECT COUNT(*) AS n FROM graph_edges;").fetchone()["n"]
        by_conf = {
            r["confidence"]: int(r["n"])
            for r in self._raw.execute(
                "SELECT confidence, COUNT(*) AS n FROM graph_edges GROUP BY confidence;"
            ).fetchall()
        }
        n_comm = self._raw.execute(
            "SELECT COUNT(DISTINCT community_id) AS n FROM graph_nodes "
            "WHERE community_id IS NOT NULL;"
        ).fetchone()["n"]
        path_row = self._raw.execute(
            "SELECT value FROM graph_meta WHERE key = 'graph_path';"
        ).fetchone()
        conf_row = self._raw.execute(
            "SELECT value FROM graph_meta WHERE key = 'confidence_filter';"
        ).fetchone()
        return {
            "nodes": int(n_nodes),
            "edges_total": int(n_edges),
            "edges_by_confidence": by_conf,
            "communities": int(n_comm),
            "last_loaded_at": row["value"],
            "graph_path": path_row["value"] if path_row else None,
            "confidence_filter": (conf_row["value"] or "").split(",") if conf_row else None,
        }

    # ----- project memory (remember/recall/forget) ------------------------

    def remember(
        self,
        *,
        repo_id: int,
        kind: str,
        key: str,
        value: str,
        source: str | None = None,
        expires_at: str | None = None,
        provenance_commit: str | None = None,
    ) -> ProjectMemoryRecord:
        now = datetime.now(UTC).isoformat()
        row = self._raw.execute(
            """
            INSERT INTO project_memory
                (repo_id, kind, key, value, source, created_at, updated_at,
                 expires_at, provenance_commit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (repo_id, kind, key) DO UPDATE SET
                value = EXCLUDED.value,
                source = EXCLUDED.source,
                updated_at = EXCLUDED.updated_at,
                expires_at = EXCLUDED.expires_at,
                provenance_commit = EXCLUDED.provenance_commit
            RETURNING id, created_at;
            """,
            (repo_id, kind, key, value, source, now, now, expires_at, provenance_commit),
        ).fetchone()
        self._raw.commit()
        return ProjectMemoryRecord(
            id=int(row["id"]), repo_id=repo_id, kind=kind, key=key, value=value,
            source=source, created_at=row["created_at"], updated_at=now,
            expires_at=expires_at, provenance_commit=provenance_commit,
        )

    def recall(
        self,
        *,
        repo_id: int,
        kind: str | None = None,
        key: str | None = None,
        limit: int = 50,
        include_expired: bool = False,
    ) -> list[ProjectMemoryRecord]:
        clauses = ["repo_id = %s"]
        params: list[Any] = [repo_id]
        if kind is not None:
            clauses.append("kind = %s")
            params.append(kind)
        if key is not None:
            clauses.append("key = %s")
            params.append(key)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > %s)")
            params.append(datetime.now(UTC).isoformat())
        params.append(limit)
        rows = self._raw.execute(
            f"""
            SELECT id, repo_id, kind, key, value, source, created_at, updated_at,
                   expires_at, provenance_commit
            FROM project_memory WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC LIMIT %s;
            """,
            params,
        ).fetchall()
        return [
            ProjectMemoryRecord(
                id=int(r["id"]), repo_id=r["repo_id"], kind=r["kind"], key=r["key"],
                value=r["value"], source=r["source"], created_at=r["created_at"],
                updated_at=r["updated_at"], expires_at=r["expires_at"],
                provenance_commit=r.get("provenance_commit"),
            )
            for r in rows
        ]

    def forget(self, *, repo_id: int, kind: str, key: str) -> bool:
        cur = self._raw.execute(
            "DELETE FROM project_memory WHERE repo_id = %s AND kind = %s AND key = %s;",
            (repo_id, kind, key),
        )
        self._raw.commit()
        return bool(cur.rowcount)

    def prune_project_memory(
        self,
        *,
        repo_id: int | None = None,
        older_than_iso: str | None = None,
        expired_only: bool = False,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if repo_id is not None:
            clauses.append("repo_id = %s")
            params.append(repo_id)
        if expired_only:
            clauses.append("(expires_at IS NOT NULL AND expires_at <= %s)")
            params.append(datetime.now(UTC).isoformat())
        elif older_than_iso is not None:
            clauses.append("updated_at < %s")
            params.append(older_than_iso)
        if not clauses:
            return 0
        cur = self._raw.execute(
            f"DELETE FROM project_memory WHERE {' AND '.join(clauses)};", params
        )
        self._raw.commit()
        return int(cur.rowcount or 0)

    # ----- episodes -------------------------------------------------------

    def _row_to_episode(self, r: dict[str, Any]) -> EpisodeRecord:
        try:
            files = json.loads(r["touched_files"] or "[]")
            symbols = json.loads(r["touched_symbols"] or "[]")
        except (json.JSONDecodeError, TypeError):
            log.warning("episodes.corrupted_row", episode_id=r.get("id"))
            files, symbols = [], []
        return EpisodeRecord(
            id=r["id"], repo_id=r["repo_id"], session_id=r["session_id"],
            title=r["title"], summary=r["summary"], touched_files=files,
            touched_symbols=symbols, created_at=r["created_at"],
            updated_at=r["updated_at"], provenance_commit=r.get("provenance_commit"),
        )

    def record_episode(
        self,
        *,
        repo_id: int,
        title: str,
        summary: str = "",
        session_id: str | None = None,
        touched_files: list[str] | None = None,
        touched_symbols: list[str] | None = None,
        provenance_commit: str | None = None,
    ) -> EpisodeRecord:
        now = datetime.now(UTC).isoformat()
        files_norm = sorted(set(touched_files or []))
        symbols_norm = sorted(set(touched_symbols or []))
        row = self._raw.execute(
            """
            INSERT INTO episodes
                (repo_id, session_id, title, summary, touched_files,
                 touched_symbols, created_at, updated_at, provenance_commit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """,
            (repo_id, session_id, title, summary, json.dumps(files_norm),
             json.dumps(symbols_norm), now, now, provenance_commit),
        ).fetchone()
        self._raw.commit()
        return EpisodeRecord(
            id=int(row["id"]), repo_id=repo_id, session_id=session_id, title=title,
            summary=summary, touched_files=files_norm, touched_symbols=symbols_norm,
            created_at=now, updated_at=now, provenance_commit=provenance_commit,
        )

    def recall_episodes(
        self,
        *,
        repo_id: int,
        query: str | None = None,
        limit: int = 10,
        since_iso: str | None = None,
    ) -> list[EpisodeRecord]:
        clauses = ["repo_id = %s"]
        params: list[Any] = [repo_id]
        if query:
            clauses.append("(title ILIKE %s OR summary ILIKE %s)")
            params.extend([f"%{query}%", f"%{query}%"])
        if since_iso is not None:
            clauses.append("created_at >= %s")
            params.append(since_iso)
        params.append(max(1, min(limit, 200)))
        rows = self._raw.execute(
            f"""
            SELECT id, repo_id, session_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at, provenance_commit
            FROM episodes WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC LIMIT %s;
            """,
            params,
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def episode_co_occurrence(
        self,
        *,
        repo_id: int,
        key: str,
        in_files: bool = True,
        in_symbols: bool = True,
        limit: int = 10,
    ) -> dict[str, dict[str, int]]:
        from collections import Counter

        if not key:
            return {"files": {}, "symbols": {}}
        like = f"%{json.dumps(key)[1:-1]}%"
        clauses = ["repo_id = %s"]
        params: list[Any] = [repo_id]
        ors: list[str] = []
        if in_files:
            ors.append("touched_files ILIKE %s")
            params.append(like)
        if in_symbols:
            ors.append("touched_symbols ILIKE %s")
            params.append(like)
        if ors:
            clauses.append("(" + " OR ".join(ors) + ")")
        rows = self._raw.execute(
            f"SELECT touched_files, touched_symbols FROM episodes "
            f"WHERE {' AND '.join(clauses)};",
            params,
        ).fetchall()
        file_counter: Counter[str] = Counter()
        symbol_counter: Counter[str] = Counter()
        for r in rows:
            try:
                files = json.loads(r["touched_files"] or "[]")
                syms = json.loads(r["touched_symbols"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            if key not in files and key not in syms:
                continue
            for f in files:
                if f != key:
                    file_counter[f] += 1
            for s in syms:
                if s != key:
                    symbol_counter[s] += 1
        return {
            "files": dict(file_counter.most_common(max(1, limit))),
            "symbols": dict(symbol_counter.most_common(max(1, limit))),
        }

    def forget_episode(self, *, episode_id: int) -> bool:
        cur = self._raw.execute("DELETE FROM episodes WHERE id = %s;", (episode_id,))
        self._raw.commit()
        return bool(cur.rowcount)

    # ----- facets + chunk helpers -----------------------------------------

    def compute_facets(self, chunk_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
        if chunk_ids is None:
            lang = self._raw.execute(
                "SELECT language, COUNT(*) AS n FROM chunks GROUP BY language;"
            ).fetchall()
            kind = self._raw.execute(
                "SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind;"
            ).fetchall()
            repo = self._raw.execute(
                "SELECT COALESCE(r.normalized_key, '(local)') AS repo, COUNT(*) AS n "
                "FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id GROUP BY repo;"
            ).fetchall()
        elif not chunk_ids:
            return {"language": {}, "kind": {}, "repo": {}}
        else:
            ids = list(chunk_ids)
            lang = self._raw.execute(
                "SELECT language, COUNT(*) AS n FROM chunks WHERE chunk_id = ANY(%s) "
                "GROUP BY language;",
                (ids,),
            ).fetchall()
            kind = self._raw.execute(
                "SELECT kind, COUNT(*) AS n FROM chunks WHERE chunk_id = ANY(%s) GROUP BY kind;",
                (ids,),
            ).fetchall()
            repo = self._raw.execute(
                "SELECT COALESCE(r.normalized_key, '(local)') AS repo, COUNT(*) AS n "
                "FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id "
                "WHERE c.chunk_id = ANY(%s) GROUP BY repo;",
                (ids,),
            ).fetchall()
        return {
            "language": {r["language"]: int(r["n"]) for r in lang},
            "kind": {r["kind"]: int(r["n"]) for r in kind},
            "repo": {r["repo"]: int(r["n"]) for r in repo},
        }

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._raw.execute(
            "SELECT chunk_id, rel_path, language, kind, symbol, parent_chunk_id, "
            "start_line, end_line, content, embedding_model FROM chunks WHERE chunk_id = %s;",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_by_paths(self, rel_paths: list[str]) -> int:
        if not rel_paths:
            return 0
        cur = self._raw.execute(
            "DELETE FROM chunks WHERE rel_path = ANY(%s);", (list(rel_paths),)
        )
        return int(cur.rowcount or 0)

    # ----- sessions (server lifespan + working_set) -----------------------

    def _session_from_row(self, row: dict[str, Any]) -> SessionRecord:
        try:
            seen = set(json.loads(row["seen_chunk_ids"] or "[]"))
            pinned = set(json.loads(row["pinned_chunk_ids"] or "[]"))
            meta = json.loads(row["meta"] or "{}")
        except (json.JSONDecodeError, TypeError):
            log.warning("sessions.corrupted_row", session_id=row.get("session_id"))
            seen, pinned, meta = set(), set(), {}
        return SessionRecord(
            session_id=row["session_id"],
            client_id=row["client_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            seen_chunk_ids=seen,
            pinned_chunk_ids=pinned,
            meta=meta,
            scratchpad=row.get("scratchpad") or "",
        )

    _SESSION_COLS = (
        "session_id, client_id, created_at, updated_at, "
        "seen_chunk_ids, pinned_chunk_ids, meta, scratchpad"
    )

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._raw.execute(
            f"SELECT {self._SESSION_COLS} FROM sessions WHERE session_id = %s;",
            (session_id,),
        ).fetchone()
        return self._session_from_row(row) if row else None

    def get_latest_session(self) -> SessionRecord | None:
        row = self._raw.execute(
            "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 1;"
        ).fetchone()
        return self.get_session(row["session_id"]) if row else None

    def list_sessions(
        self, *, client_id: str | None = None, limit: int = 50
    ) -> list[SessionRecord]:
        if client_id is None:
            rows = self._raw.execute(
                f"SELECT {self._SESSION_COLS} FROM sessions "
                "ORDER BY updated_at DESC LIMIT %s;",
                (limit,),
            ).fetchall()
        else:
            rows = self._raw.execute(
                f"SELECT {self._SESSION_COLS} FROM sessions WHERE client_id = %s "
                "ORDER BY updated_at DESC LIMIT %s;",
                (client_id, limit),
            ).fetchall()
        return [self._session_from_row(r) for r in rows]

    def upsert_session(
        self,
        session_id: str,
        *,
        client_id: str | None = None,
        seen_chunk_ids: set[str] | None = None,
        pinned_chunk_ids: set[str] | None = None,
        meta: dict[str, Any] | None = None,
        scratchpad: str | None = None,
    ) -> SessionRecord:
        now = datetime.now(UTC).isoformat()
        existing = self._raw.execute(
            "SELECT created_at, client_id, scratchpad FROM sessions WHERE session_id = %s;",
            (session_id,),
        ).fetchone()
        seen_json = json.dumps(sorted(seen_chunk_ids or set()))
        pinned_json = json.dumps(sorted(pinned_chunk_ids or set()))
        meta_json = json.dumps(meta or {}, sort_keys=True)
        if existing is None:
            scratchpad_resolved = scratchpad or ""
            self._raw.execute(
                "INSERT INTO sessions (session_id, client_id, created_at, updated_at, "
                "seen_chunk_ids, pinned_chunk_ids, meta, scratchpad) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                (session_id, client_id, now, now, seen_json, pinned_json,
                 meta_json, scratchpad_resolved),
            )
            created_at = now
            resolved_client_id = client_id
        else:
            resolved_client_id = (
                client_id if client_id is not None else existing["client_id"]
            )
            scratchpad_resolved = (
                scratchpad if scratchpad is not None else (existing["scratchpad"] or "")
            )
            self._raw.execute(
                "UPDATE sessions SET client_id = %s, updated_at = %s, seen_chunk_ids = %s, "
                "pinned_chunk_ids = %s, meta = %s, scratchpad = %s WHERE session_id = %s;",
                (resolved_client_id, now, seen_json, pinned_json, meta_json,
                 scratchpad_resolved, session_id),
            )
            created_at = existing["created_at"]
        self._raw.commit()
        return SessionRecord(
            session_id=session_id,
            client_id=resolved_client_id,
            created_at=created_at,
            updated_at=now,
            seen_chunk_ids=set(seen_chunk_ids or set()),
            pinned_chunk_ids=set(pinned_chunk_ids or set()),
            meta=dict(meta or {}),
            scratchpad=scratchpad_resolved,
        )

    def delete_session(self, session_id: str) -> bool:
        cur = self._raw.execute("DELETE FROM sessions WHERE session_id = %s;", (session_id,))
        self._raw.commit()
        return bool(cur.rowcount)

    def prune_sessions(self, *, older_than_iso: str) -> int:
        cur = self._raw.execute(
            "DELETE FROM sessions WHERE updated_at < %s;", (older_than_iso,)
        )
        self._raw.commit()
        return int(cur.rowcount or 0)

    # ----- Reads ----------------------------------------------------------

    def search_dense(self, query_vec: list[float], k: int = 10) -> list[SearchHit]:
        if len(query_vec) != self._dim:
            raise StorageError(
                f"Query vector dim {len(query_vec)} != index dim {self._dim}"
            )
        vec = _vec_literal(query_vec)
        rows = self._raw.execute(
            f"""
            SELECT {_REPO_COLS}, c.embedding <-> %s::vector AS distance
            FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <-> %s::vector
            LIMIT %s;
            """,
            (vec, vec, k),
        ).fetchall()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row["distance"])
            score = max(0.0, 1.0 - (distance * distance) / 2.0)
            hits.append(self._hit(row, rank=rank, score=score))
        return hits

    def search_bm25(self, query: str, k: int = 50) -> list[SearchHit]:
        tsq = _pg_tsquery(query)
        if not tsq:
            return []
        try:
            rows = self._raw.execute(
                f"""
                SELECT {_REPO_COLS},
                       ts_rank_cd(c.fts, to_tsquery('simple', %s)) AS score
                FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
                WHERE c.fts @@ to_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s;
                """,
                (tsq, tsq, k),
            ).fetchall()
        except Exception as exc:
            log.warning("storage.fts_query_failed", error=str(exc))
            self._raw.rollback()
            return []
        return [
            self._hit(row, rank=rank, score=float(row["score"]))
            for rank, row in enumerate(rows, start=1)
        ]

    def search_symbol_tokens(self, query: str, k: int = 20) -> list[SearchHit]:
        tokens = _identifier_tokens(query)
        if not tokens:
            return []
        score_terms = " + ".join(
            "(CASE WHEN c.symbol ILIKE %s THEN 1 ELSE 0 END)" for _ in tokens
        )
        where_clause = " OR ".join("c.symbol ILIKE %s" for _ in tokens)
        like_params = [f"%{t}%" for t in tokens]
        params = [*like_params, *like_params, k]
        rows = self._raw.execute(
            f"""
            SELECT {_REPO_COLS}, ({score_terms}) AS match_count
            FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
            WHERE ({where_clause}) AND c.symbol IS NOT NULL
            ORDER BY match_count DESC, char_length(c.symbol) ASC
            LIMIT %s;
            """,
            params,
        ).fetchall()
        return [
            self._hit(row, rank=rank, score=float(row["match_count"]))
            for rank, row in enumerate(rows, start=1)
        ]

    def lookup_by_symbol(self, symbol: str, limit: int = 10) -> list[SearchHit]:
        rows = self._raw.execute(
            f"""
            SELECT {_REPO_COLS}
            FROM chunks c LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.symbol = %s
               OR c.symbol ILIKE %s
               OR c.symbol ILIKE %s
            ORDER BY
                CASE WHEN c.symbol = %s THEN 0
                     WHEN c.symbol ILIKE %s THEN 1
                     ELSE 2 END,
                char_length(c.symbol)
            LIMIT %s;
            """,
            (symbol, f"%.{symbol}", f"%{symbol}%", symbol, f"%.{symbol}", limit),
        ).fetchall()
        return [self._hit(row, rank=rank, score=1.0) for rank, row in enumerate(rows, start=1)]

    def file_outline(self, rel_path: str) -> list[dict[str, Any]]:
        rows = self._raw.execute(
            "SELECT chunk_id, kind, symbol, parent_chunk_id, start_line, end_line "
            "FROM chunks WHERE rel_path = %s ORDER BY start_line;",
            (rel_path,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_repos(self) -> list[RepoStats]:
        rows = self._raw.execute(
            """
            SELECT r.normalized_key, r.remote_url AS url, r.root_path AS local_path,
                   r.default_branch, r.head_commit, r.last_indexed_at,
                   COUNT(DISTINCT c.rel_path) AS files,
                   COUNT(c.id) AS chunks
            FROM repos r LEFT JOIN chunks c ON c.repo_id = r.id
            GROUP BY r.id
            ORDER BY chunks DESC;
            """
        ).fetchall()
        return [
            RepoStats(
                normalized_key=row["normalized_key"],
                url=row["url"],
                local_path=row["local_path"],
                default_branch=row["default_branch"],
                head_commit=row["head_commit"],
                files=int(row["files"]),
                chunks=int(row["chunks"]),
                last_indexed_at=row["last_indexed_at"],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        chunks = int(self._raw.execute("SELECT COUNT(*) AS n FROM chunks;").fetchone()["n"])
        files = int(
            self._raw.execute("SELECT COUNT(DISTINCT rel_path) AS n FROM chunks;").fetchone()["n"]
        )
        models = [
            dict(r)
            for r in self._raw.execute(
                "SELECT embedding_model, embedding_dim, COUNT(*) AS n "
                "FROM chunks GROUP BY embedding_model, embedding_dim;"
            ).fetchall()
        ]
        enrichers = [
            dict(r)
            for r in self._raw.execute(
                "SELECT COALESCE(enricher_id, '(none)') AS enricher_id, COUNT(*) AS n "
                "FROM chunks GROUP BY enricher_id;"
            ).fetchall()
        ]
        cache_size = int(
            self._raw.execute("SELECT COUNT(*) AS n FROM enrichments;").fetchone()["n"]
        )
        files_tracked = int(self._raw.execute("SELECT COUNT(*) AS n FROM files;").fetchone()["n"])
        return {
            "db_path": self._sanitized_dsn(),
            "size_bytes": self._db_size(),
            "chunks": chunks,
            "files": files,
            "files_tracked": files_tracked,
            "index_dim": self._dim,
            "models": models,
            "enrichers": enrichers,
            "enrichment_cache_size": cache_size,
            "repos": [r.model_dump() for r in self.list_repos()],
            "graph": self._graph_stats(),
        }

    # ----- helpers --------------------------------------------------------

    def _hit(self, row: dict[str, Any], *, rank: int, score: float) -> SearchHit:
        return SearchHit(
            rank=rank,
            score=score,
            rel_path=row["rel_path"],
            language=row["language"],
            kind=row["kind"],
            symbol=row["symbol"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            snippet=_snippet(row["content"]),
            chunk_id=row["chunk_id"],
            repo=_repo_from_row(row),
        )

    # ----- context manager ------------------------------------------------
    # ``self._raw`` is the native psycopg connection used by this backend's
    # own methods; ``self._conn`` is the SQLite-dialect shim exposed to
    # external graph/cache/glossary consumers (the StorageStage escape hatch).

    def __enter__(self) -> PostgresStorage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
