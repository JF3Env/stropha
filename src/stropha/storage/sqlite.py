"""sqlite-vec + FTS5 + metadata, all in one SQLite database.

Per spec §5: keep everything in a single `.db` file so it is trivially
backed up, synced, and shipped. Phase 0 uses dense search only; the FTS5
table is provisioned now so Phase 1 hybrid retrieval is a query change,
not a schema migration.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlite_vec  # type: ignore[import-untyped]

from ..errors import StorageError
from ..ingest.git_meta import RepoIdentity
from ..logging import get_logger
from ..models import Chunk, RepoRef, RepoStats, SearchHit, StoredFileMeta
from ..secondary_embeddings import SECONDARY_EMBEDDING_DIM

if TYPE_CHECKING:
    # These are imported inline at call sites to avoid pulling memory/session
    # models into the storage import graph at runtime. Listed here so type
    # annotations on the public storage methods resolve cleanly.
    from ..models import EpisodeRecord, ProjectMemoryRecord, SessionRecord
    from ..secondary_embeddings import SecondaryEmbeddingChunk, SecondaryEmbeddingVector

log = get_logger(__name__)

# v1: initial schema (chunks, vec_chunks, fts_chunks, meta).
# v2: adds `repos` table and `chunks.repo_id` column for multi-repo support.
# v3: pipeline-adapters Phase 1 — `chunks.embedding_text` (what was actually
#     embedded), `chunks.enricher_id` (which adapter produced it), and the
#     `enrichments` cache table. Enables drift detection: changing enricher
#     config triggers re-enrich + re-embed without `--rebuild`.
# v4: graphify integration (RFC §9 Fase 1.5a) — `graph_nodes`, `graph_edges`
#     and `graph_meta` tables backing `find_callers` / `find_related` /
#     `get_community` / `find_rationale` MCP tools. Conditional on
#     `graphify-out/graph.json` being present at index time.
# v5: Trilha A L3 — `graph_nodes.embedding` BLOB + `graph_nodes.embedding_model`
#     for the `graph-vec` retrieval stream. Brute-force cosine in Python:
#     graphs are typically <50K nodes so a single dense pass is sub-ms.
# v6: file-level dirty cache (Phase A incremental). `files` table keyed by
#     (repo_id, rel_path) with mtime + size + content_hash + chunk_count.
#     Lets `Pipeline._index_files()` skip chunking entirely when a file is
#     fresh — saves tree-sitter parse + chunk-fresh probes for typical
#     no-op `stropha index` invocations.
# v7: multi-repo graphify support. Adds `repo_id` to `graph_nodes` and
#     `graph_edges` tables to allow multiple repos with their own graphify
#     graphs to share a single stropha index. Node IDs are prefixed with
#     repo_id to avoid collisions across repos.
# v14: independent 256-d secondary chunk embeddings (WIN-300). A dedicated
#      sqlite-vec table carries content/embedder metadata for drift-safe search.
SCHEMA_VERSION = 14


def _serialize_vector(vec: list[float]) -> bytes:
    """Float32 little-endian, the format sqlite-vec expects for BLOB inputs."""
    return struct.pack(f"<{len(vec)}f", *vec)


# Characters that must be stripped from user queries before handing to FTS5.
_FTS_BAD = re.compile(r'[^\w\s\.]+', re.UNICODE)
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _split_identifiers(text: str) -> str:
    """Expand CamelCase + dotted identifiers into separate tokens.

    Used on BOTH indexed content and queries so `FsrsCalculator` finds
    `fsrs calculator` after FTS5's unicode61 tokenizer lowercases and splits
    on non-alphanumerics.
    """
    text = _CAMEL_BOUNDARY_1.sub(" ", text)
    text = _CAMEL_BOUNDARY_2.sub(" ", text)
    return text


def _sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (OR-joined tokens)."""
    cleaned = _FTS_BAD.sub(" ", query)
    cleaned = _split_identifiers(cleaned)
    tokens = [t for t in cleaned.split() if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _extract_file_stem(rel_path: str) -> str:
    """Extract the file stem (name without extension) from a path."""
    # Get the filename
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1] if parts else ""
    # Remove extension
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem


def _extract_directory_tokens(rel_path: str, max_depth: int = 3) -> str:
    """Extract directory component names from a path.
    
    Returns the last `max_depth` directory names, which are often meaningful
    (e.g., 'services', 'models', 'utils', 'api').
    """
    parts = rel_path.replace("\\", "/").split("/")
    # Get directory components (exclude filename)
    dirs = parts[:-1] if len(parts) > 1 else []
    # Take last N directories
    relevant_dirs = dirs[-max_depth:] if len(dirs) > max_depth else dirs
    return " ".join(relevant_dirs)


def _fts_text(content: str, rel_path: str, symbol: str | None) -> str:
    """Pre-process chunk content before indexing in FTS5.

    We assemble a single FTS document from multiple sources, separated by
    newlines so FTS5 treats them as a single token stream:

    1. Original content — preserves exact identifier matches.
    2. CamelCase / dotted split of content — recovers tokens like `Fsrs`,
       `Calculator` from `FsrsCalculator`.
    3. File path tokens — boosts results whose path mentions the query
       (e.g. `FsrsCalculator.java` is a strong signal for "fsrs calculator").
    4. File stem (repeated) — extra weight for the filename, which is often
       the primary identifier (e.g., `UserService` in `UserService.java`).
    5. Directory tokens — parent directories often indicate module/domain.
    6. Symbol (qualified name) — gives chunks a small boost when the user
       asks for them by name.
    """
    path_tokens = _split_identifiers(
        rel_path.replace("/", " ").replace("\\", " ").replace(".", " ")
    )
    
    # Extract and process file stem (repeat twice for stronger signal)
    file_stem = _extract_file_stem(rel_path)
    stem_tokens = _split_identifiers(file_stem)
    
    # Extract directory tokens
    dir_tokens = _extract_directory_tokens(rel_path)
    dir_tokens_split = _split_identifiers(dir_tokens)
    
    # Build FTS document with weighted components
    parts: list[str] = [
        content,                      # Original content
        _split_identifiers(content),  # CamelCase-split content
        path_tokens,                  # Full path tokens
        stem_tokens,                  # File stem (1st occurrence)
        stem_tokens,                  # File stem (2nd occurrence for boost)
        dir_tokens_split,             # Directory tokens
    ]
    if symbol:
        parts.append(_split_identifiers(symbol.replace(".", " ")))
    return "\n".join(parts)


def _snippet(content: str, limit: int = 480) -> str:
    return content[:limit].rstrip() + ("…" if len(content) > limit else "")


def _row_to_episode(row: Any) -> Any:
    """Map an ``episodes`` SQLite row to an ``EpisodeRecord``.

    Module-level helper colocated with the other row mappers. JSON
    columns fall back to empty lists when the stored value is
    corrupted — same defensive contract as ``sessions``.
    """
    from ..models import EpisodeRecord

    try:
        files = list(json.loads(row["touched_files"] or "[]"))
        symbols = list(json.loads(row["touched_symbols"] or "[]"))
    except (json.JSONDecodeError, TypeError):
        log.warning("episodes.corrupted_row", episode_id=row["id"])
        files, symbols = [], []
    return EpisodeRecord(
        id=int(row["id"]),
        repo_id=int(row["repo_id"]),
        session_id=row["session_id"],
        title=row["title"],
        summary=row["summary"] or "",
        touched_files=files,
        touched_symbols=symbols,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        provenance_commit=(
            row["provenance_commit"]
            if "provenance_commit" in row.keys() else None
        ),
    )


def _repo_from_row(row: Any) -> RepoRef | None:
    """Build a ``RepoRef`` from a row that LEFT JOINed ``repos``.

    Returns ``None`` for chunks without a repo (legacy data pre-schema-v2).

    Includes both the git origin URL and local filesystem path so callers
    can access the repository via either method. File paths (rel_path) are
    always relative to the git root.
    """
    key = row["repo_normalized_key"] if "repo_normalized_key" in row.keys() else None
    if not key:
        return None
    return RepoRef(
        normalized_key=key,
        url=row["repo_url"] if "repo_url" in row.keys() else None,
        local_path=(
            row["repo_local_path"] if "repo_local_path" in row.keys() else None
        ),
        default_branch=(
            row["repo_default_branch"] if "repo_default_branch" in row.keys() else None
        ),
        head_commit=(
            row["repo_head_commit"] if "repo_head_commit" in row.keys() else None
        ),
    )


# Common SELECT clause emitting both chunk and repo columns. Reused by all
# search_* paths. Aliases are stable so _repo_from_row can rely on them.
# Includes both git origin URL (repo_url) and local path (repo_local_path)
# so callers can access files via either method.
_REPO_JOIN_COLUMNS = """
    c.chunk_id, c.rel_path, c.language, c.kind, c.symbol,
    c.start_line, c.end_line, c.content,
    r.normalized_key  AS repo_normalized_key,
    r.remote_url      AS repo_url,
    r.root_path       AS repo_local_path,
    r.default_branch  AS repo_default_branch,
    r.head_commit     AS repo_head_commit
"""


# Common English stopwords + question words. Filtered out from identifier-token
# extraction so a free-text query like "where is the FSRS calculator" routes
# only the meaningful tokens [FSRS, calculator] to the symbol lookup stream.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had",
    "this", "that", "these", "those",
    "what", "where", "when", "why", "how", "who", "which",
    "show", "find", "look", "get",
    "with", "without", "by", "from", "as",
    "it", "its", "i", "you", "we",
    "code", "function", "class", "method", "file",
})


def _identifier_tokens(query: str) -> list[str]:
    """Extract identifier-like tokens from a free-text query.

    Drops stopwords and tokens of length < 3. CamelCase splits before
    filtering so `FsrsCalculator` produces both `Fsrs` and `Calculator`.
    """
    cleaned = _FTS_BAD.sub(" ", query)
    cleaned = _split_identifiers(cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for tok in cleaned.split():
        if len(tok) < 3:
            continue
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(tok)
    return out


class Storage:
    """Owns the SQLite connection. All schema mutations go through here."""

    def __init__(
        self,
        db_path: Path,
        embedding_dim: int,
        *,
        force_rebuild: bool = False,
    ) -> None:
        """Open the SQLite + sqlite-vec store.

        Args:
            db_path: Filesystem path to the SQLite file.
            embedding_dim: The dimensionality of vectors this store will
                hold. The first time the store is opened, this becomes
                the fixed dim of the ``vec_chunks`` virtual table.
            force_rebuild: When the caller knows a ``--rebuild`` is in
                effect, set this to ``True``. The constructor will then
                reset every dim-dependent table (vec_chunks must be DROP +
                recreated when the dim changes; you cannot ALTER a
                ``CREATE VIRTUAL TABLE ... USING vec0(embedding float[N])``).
                Without this flag, opening at a different dim raises a
                :class:`StorageError` so an unintentional dim shift cannot
                silently invalidate the index.
        """
        self._db_path = db_path
        self._dim = embedding_dim
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        try:
            self._conn.enable_load_extension(True)
        except (AttributeError, sqlite3.NotSupportedError) as exc:
            raise StorageError(
                "Your Python sqlite3 was built without extension loading. "
                "Reinstall Python via `uv python install 3.12` and retry."
            ) from exc
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.row_factory = sqlite3.Row

        self._migrate()
        if force_rebuild:
            # Resolve any dim mismatch BEFORE the consistency check would
            # raise. No-op when the stored dim already matches self._dim.
            self._reset_dim_dependent_tables_if_needed()
        self._check_dimension_consistency()

    @property
    def dim(self) -> int:
        return self._dim

    # ----- schema -----

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id         TEXT NOT NULL UNIQUE,
                rel_path         TEXT NOT NULL,
                language         TEXT NOT NULL,
                kind             TEXT NOT NULL,
                symbol           TEXT,
                parent_chunk_id  TEXT,
                start_line       INTEGER NOT NULL,
                end_line         INTEGER NOT NULL,
                content          TEXT NOT NULL,
                content_hash     TEXT NOT NULL,
                embedding_model  TEXT NOT NULL,
                embedding_dim    INTEGER NOT NULL,
                indexed_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_rel_path     ON chunks(rel_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_language     ON chunks(language);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{self._dim}]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                content,
                rel_path UNINDEXED,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61'
            );

            -- v2: repository identity (multi-repo support).
            CREATE TABLE IF NOT EXISTS repos (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_key     TEXT NOT NULL UNIQUE,
                remote_url         TEXT,
                root_path          TEXT NOT NULL,
                default_branch     TEXT,
                head_commit        TEXT,
                first_indexed_at   TEXT NOT NULL,
                last_indexed_at    TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_repos_normalized_key
                ON repos(normalized_key);
            """
        )
        # v2: add chunks.repo_id (ALTER TABLE is idempotent only when guarded).
        self._add_column_if_missing(
            "chunks", "repo_id", "INTEGER REFERENCES repos(id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);"
        )

        # v3: pipeline-adapters Phase 1.
        # - chunks.embedding_text  → exact text fed to the embedder
        # - chunks.enricher_id     → adapter id of the enricher that produced it
        # - enrichments cache     → reusable (content_hash, adapter_id) → text
        self._add_column_if_missing("chunks", "embedding_text", "TEXT")
        self._add_column_if_missing("chunks", "enricher_id", "TEXT")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_enricher ON chunks(enricher_id);"
        )
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS enrichments (
                content_hash   TEXT NOT NULL,
                enricher_id    TEXT NOT NULL,
                embedding_text TEXT NOT NULL,
                cached_at      TEXT NOT NULL,
                PRIMARY KEY (content_hash, enricher_id)
            );
            CREATE INDEX IF NOT EXISTS idx_enrichments_enricher
                ON enrichments(enricher_id);
            """
        )

        # v4: graphify integration tables. Optional — populated by
        # GraphifyLoader iff `graphify-out/graph.json` is present.
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id          TEXT PRIMARY KEY,
                label            TEXT NOT NULL,
                file_type        TEXT,
                source_file      TEXT,
                source_location  TEXT,
                community_id     INTEGER,
                community_label  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_label
                ON graph_nodes(label);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_source_file
                ON graph_nodes(source_file);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_community
                ON graph_nodes(community_id);

            CREATE TABLE IF NOT EXISTS graph_edges (
                source           TEXT NOT NULL,
                target           TEXT NOT NULL,
                relation         TEXT NOT NULL,
                confidence       TEXT NOT NULL,
                confidence_score REAL,
                context          TEXT,
                source_file      TEXT,
                source_location  TEXT,
                weight           REAL,
                PRIMARY KEY (source, target, relation, source_file, source_location)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target_relation
                ON graph_edges(target, relation, confidence);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source_relation
                ON graph_edges(source, relation, confidence);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_relation
                ON graph_edges(relation, confidence);

            CREATE TABLE IF NOT EXISTS graph_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        # v5: graph node embeddings for the `graph-vec` retrieval stream
        # (Trilha A L3). Optional — populated by `GraphVecLoader` when an
        # embedder is provided. Brute-force cosine at query time keeps the
        # implementation tiny; for the corpora we target (<50K graph nodes)
        # one full pass is sub-millisecond.
        self._add_column_if_missing("graph_nodes", "embedding", "BLOB")
        self._add_column_if_missing("graph_nodes", "embedding_model", "TEXT")
        self._add_column_if_missing("graph_nodes", "embedding_dim", "INTEGER")

        # v6: file-level dirty cache (Phase A incremental).
        # `mtime` + `size_bytes` are the fast-path keys; `content_hash` is
        # the authoritative key set when the chunker actually reads the
        # file. `last_chunk_count` is informational (sanity check + stats).
        # `last_enricher_id` + `last_embedder_model` are denormalised so
        # `file_is_fresh()` can drop the skip when the active pipeline
        # config has drifted away from the values used at last index
        # (otherwise file-level skip would silently override the chunk-level
        # drift detection of ADR-004).
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id             INTEGER NOT NULL REFERENCES repos(id),
                rel_path            TEXT NOT NULL,
                content_hash        TEXT NOT NULL,
                size_bytes          INTEGER NOT NULL,
                mtime               REAL NOT NULL,
                last_indexed_at     TEXT NOT NULL,
                last_chunk_count    INTEGER NOT NULL DEFAULT 0,
                last_enricher_id    TEXT,
                last_embedder_model TEXT,
                UNIQUE (repo_id, rel_path)
            );
            CREATE INDEX IF NOT EXISTS idx_files_repo_path
                ON files(repo_id, rel_path);
            CREATE INDEX IF NOT EXISTS idx_files_content_hash
                ON files(content_hash);
            """
        )
        # Upgrade path for installs that ran an earlier v6 build before
        # these columns existed.
        self._add_column_if_missing("files", "last_enricher_id", "TEXT")
        self._add_column_if_missing("files", "last_embedder_model", "TEXT")

        # v7: multi-repo graphify support. Add repo_id to graph_nodes and
        # graph_edges so multiple repos can share a single index without
        # node ID collisions. Foreign key to repos(id).
        self._add_column_if_missing(
            "graph_nodes", "repo_id", "INTEGER REFERENCES repos(id)"
        )
        self._add_column_if_missing(
            "graph_edges", "repo_id", "INTEGER REFERENCES repos(id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo ON graph_nodes(repo_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_edges_repo ON graph_edges(repo_id);"
        )

        # v10: episodes table — episodic task memory (WIN-225).
        # Captures "the last time I worked on FSRS I also touched the
        # migration" by logging an episode summary + the files/symbols
        # the agent touched. Sister of project_memory (v9):
        # - project_memory: stable facts ("FSRS is authoritative")
        # - episodes: timeline events ("on 2026-05-23 I added a
        #   late-chunking adapter and touched files X, Y, Z")
        # Both scoped to a repo via repo_id FK.
        #
        # ``touched_files`` / ``touched_symbols`` are JSON arrays so
        # the schema stays trivially exportable. Co-occurrence queries
        # at recall time scan the JSON via LIKE (small N — episodes
        # rarely exceed a few hundred per repo).
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id           INTEGER NOT NULL REFERENCES repos(id),
                session_id        TEXT,
                title             TEXT NOT NULL,
                summary           TEXT NOT NULL DEFAULT '',
                touched_files     TEXT NOT NULL DEFAULT '[]',
                touched_symbols   TEXT NOT NULL DEFAULT '[]',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_repo_created
                ON episodes(repo_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_session
                ON episodes(session_id);
            """
        )

        # v9: project_memory table — long-term per-repo notes (WIN-222).
        # Agents curate decisions / constraints / anchor facts here so
        # they don't have to re-explain "the FSRS scheduler is
        # authoritative" on every session. Distinct from the sessions
        # table because project memory is REPO-scoped (survives
        # sessions, even survives the agent identity) while sessions
        # scope to one MCP-server lifetime.
        #
        # ``kind`` is a free-form bucket the agent picks (decision /
        # constraint / fact / convention / glossary / TBD). Keeping it
        # free-form rather than an ENUM means future categories don't
        # need a schema bump.
        # ``key`` + ``kind`` + ``repo_id`` together form the natural
        # UPSERT key — the agent writes ``remember("decision", "fsrs-
        # authoritative", "...")`` and the same key in a follow-up
        # call updates the existing row.
        # ``source`` records which session_id curated the entry, for
        # provenance + future blame.
        # ``expires_at`` is the WIN-228 TTL hook — None = forever.
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS project_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id     INTEGER NOT NULL REFERENCES repos(id),
                kind        TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                source      TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                expires_at  TEXT,
                UNIQUE (repo_id, kind, key)
            );
            CREATE INDEX IF NOT EXISTS idx_project_memory_repo
                ON project_memory(repo_id);
            CREATE INDEX IF NOT EXISTS idx_project_memory_repo_kind
                ON project_memory(repo_id, kind);
            CREATE INDEX IF NOT EXISTS idx_project_memory_expires
                ON project_memory(expires_at);
            """
        )

        # v8: sessions table — KEYSTONE for EPIC 1.9 statefulness work
        # (WIN-221). Persists per-session state (seen_chunk_ids,
        # pinned_chunk_ids) across MCP-server restarts so the agent's
        # working set survives process boundaries. Future stories in the
        # epic will add columns for project memory (WIN-222), episodic
        # task memory (WIN-225), scratchpad (WIN-229), etc. — every new
        # piece is an _add_column_if_missing on this same table or a
        # foreign-key relation to it.
        #
        # ``session_id`` is the same uuid4 the MCP server already
        # generates per process (WIN-160), so cross-session correlation
        # via the trace recorder stays cheap.
        # ``client_id`` is optional — the client can pass a stable id
        # (hostname, IDE identity, project path) so the same workspace
        # rejoins its prior session across restarts. When None the row
        # is process-unique.
        # ``seen_chunk_ids`` / ``pinned_chunk_ids`` are JSON arrays.
        # SQLite has no native set type; JSON keeps the on-disk shape
        # human-readable and trivially exportable. Sets fit in the
        # 1 MB row default; if a session ever exceeds that, the
        # behaviour falls back to "lose history" which is degraded
        # but never crashes (WIN-228 TTL will handle the long-term
        # cleanup).
        # ``meta`` is a catchall JSON blob for forward-compat. Future
        # stories can lean on it before deciding whether a feature
        # deserves its own column.
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id        TEXT PRIMARY KEY,
                client_id         TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                seen_chunk_ids    TEXT NOT NULL DEFAULT '[]',
                pinned_chunk_ids  TEXT NOT NULL DEFAULT '[]',
                meta              TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_client_id
                ON sessions(client_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                ON sessions(updated_at);
            """
        )

        # v11: provenance_commit on memory tables (WIN-228). Records
        # the git HEAD SHA at the moment the entry was written so the
        # staleness check can diff against HEAD at recall time. Both
        # ``project_memory`` and ``episodes`` get the column —
        # forward-only ALTER, idempotent.
        self._add_column_if_missing(
            "project_memory", "provenance_commit", "TEXT"
        )
        self._add_column_if_missing(
            "episodes", "provenance_commit", "TEXT"
        )

        # v12: scratchpad column on sessions (WIN-229). Durable
        # free-form text the agent uses as working notes for the
        # current task. Survives MCP-server restarts via the same
        # write-through path as seen_chunk_ids / pinned_chunk_ids.
        # Default empty string so existing v8+ rows upgrade cleanly.
        self._add_column_if_missing(
            "sessions", "scratchpad", "TEXT NOT NULL DEFAULT ''"
        )

        # v13: cross_session_cache table — semantic cache (WIN-230).
        # Persists (query_embedding, results_json) tuples so a session
        # querying "how does mastery decay work" can hit a cached
        # entry written by an earlier session that asked "explain
        # mastery decay logic". Cross-session findings are reusable
        # via cosine-similarity lookup over the stored embeddings.
        #
        # Implementation note: brute-force cosine over the rows, same
        # pattern as the graph-vec retrieval stream (sub-ms for ≤50K
        # entries). embedding_model + embedding_dim are stored so the
        # lookup can filter to entries that came from the SAME
        # embedder — cross-embedder cosine is meaningless.
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS cross_session_cache (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id          INTEGER NOT NULL REFERENCES repos(id),
                query_text       TEXT NOT NULL,
                query_embedding  BLOB NOT NULL,
                embedding_model  TEXT NOT NULL,
                embedding_dim    INTEGER NOT NULL,
                results_json     TEXT NOT NULL,
                session_id       TEXT,
                created_at       TEXT NOT NULL,
                last_hit_at      TEXT,
                hit_count        INTEGER NOT NULL DEFAULT 0,
                expires_at       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_csc_repo
                ON cross_session_cache(repo_id);
            CREATE INDEX IF NOT EXISTS idx_csc_model
                ON cross_session_cache(embedding_model, embedding_dim);
            CREATE INDEX IF NOT EXISTS idx_csc_expires
                ON cross_session_cache(expires_at);
            """
        )

        # v14: independent Model2Vec secondary embedding space (WIN-300).
        # vec0 metadata columns let KNN constrain by embedder identity before
        # ranking; content_hash is checked against the current chunks row so a
        # stale vector can never surface after chunk content changes.
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks_secondary USING vec0(
                embedding float[{SECONDARY_EMBEDDING_DIM}],
                content_hash text,
                embedder_id text,
                embedding_dim integer
            );
            """
        )

        cur.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        cur.execute(
            "INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)",
            ("embedding_dim", str(self._dim)),
        )
        # Bump schema_version if this is an in-place upgrade.
        cur.execute(
            "UPDATE meta SET value=? WHERE key='schema_version' AND CAST(value AS INTEGER) < ?",
            (str(SCHEMA_VERSION), SCHEMA_VERSION),
        )
        self._conn.commit()

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        """Idempotent ALTER TABLE — SQLite has no IF NOT EXISTS for ADD COLUMN."""
        cols = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in cols):
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _check_dimension_consistency(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='embedding_dim'"
        ).fetchone()
        stored_dim = int(row["value"]) if row else self._dim
        if stored_dim != self._dim:
            raise StorageError(
                f"Index was built with dim={stored_dim} but current embedder uses dim={self._dim}. "
                f"Run `stropha index --rebuild` to rebuild."
            )

    def _reset_dim_dependent_tables_if_needed(self) -> None:
        """Drop and recreate dim-dependent tables when the embedder dim changed.

        Called from ``__init__`` ONLY when ``force_rebuild=True``. SQLite's
        sqlite-vec extension creates ``vec_chunks`` as a virtual table with
        the dimensionality baked into the schema:

            CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[N]);

        That ``N`` cannot be ALTERed — and ``DELETE FROM vec_chunks`` only
        clears rows, not the column type. So switching from a 1024-dim
        embedder (e.g. ``bge-m3``) to a 512-dim one (e.g. ``voyage-code-3``)
        requires a full table replacement.

        This method:

        1. Reads the previous dim from ``meta``.
        2. No-op if it already matches the new dim (or no previous dim).
        3. Otherwise: ``DROP TABLE vec_chunks`` + clear every dim-coupled
           data table (chunks, fts_chunks, files, enrichments).
        4. Deletes the ``meta.embedding_dim`` row so the next migrate
           writes the new value cleanly.
        5. Re-runs ``_migrate()`` which recreates ``vec_chunks`` at the
           current ``self._dim``.

        Idempotent — safe to call repeatedly.
        """
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='embedding_dim'"
        ).fetchone()
        stored_dim = int(row["value"]) if row else self._dim
        if stored_dim == self._dim:
            return  # No mismatch — nothing to reset.

        log.warning(
            "storage.dim_rebuild",
            old_dim=stored_dim,
            new_dim=self._dim,
            advice="vec_chunks is being dropped and recreated; all chunks cleared",
        )
        self._conn.executescript(
            """
            DROP TABLE IF EXISTS vec_chunks;
            DELETE FROM vec_chunks_secondary;
            DELETE FROM chunks;
            DELETE FROM fts_chunks;
            DELETE FROM files;
            DELETE FROM enrichments;
            DELETE FROM meta WHERE key='embedding_dim';
            """
        )
        self._conn.commit()
        # _migrate recreates vec_chunks at self._dim and writes the new
        # meta.embedding_dim row via _set_meta inside the migration.
        self._migrate()

    # ----- repos -----

    def register_repo(self, identity: RepoIdentity) -> int:
        """Insert or update a repo row keyed by ``normalized_key``. Returns its id.

        Idempotent: re-running ``index`` against the same repo updates
        ``last_indexed_at`` + ``head_commit`` + ``default_branch`` without
        creating duplicates.
        """
        now = datetime.now(UTC).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id FROM repos WHERE normalized_key = ?",
            (identity.normalized_key,),
        )
        row = cur.fetchone()
        if row is not None:
            repo_id = int(row["id"])
            cur.execute(
                """
                UPDATE repos
                   SET remote_url       = ?,
                       root_path        = ?,
                       default_branch   = ?,
                       head_commit      = ?,
                       last_indexed_at  = ?
                 WHERE id = ?
                """,
                (
                    identity.remote_url,
                    str(identity.root_path),
                    identity.default_branch,
                    identity.head_commit,
                    now,
                    repo_id,
                ),
            )
            return repo_id

        cur.execute(
            """
            INSERT INTO repos (
                normalized_key, remote_url, root_path,
                default_branch, head_commit,
                first_indexed_at, last_indexed_at
            ) VALUES (?,?,?,?,?,?,?)
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
        )
        return int(cur.lastrowid or 0)

    def count_chunks_without_repo(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE repo_id IS NULL"
        ).fetchone()
        return int(row["n"])

    def backfill_chunks_to_repo(self, repo_id: int, sample_root: Path) -> int:
        """Best-effort assignment of orphan chunks to ``repo_id``.

        Used on upgrade from schema v1 (no ``repo_id`` column). Performs a
        sanity check first: at least one of the orphan chunks' ``rel_path``
        must exist under ``sample_root``. If the check fails we refuse to
        backfill — the user likely changed ``STROPHA_TARGET_REPO`` between
        indexer runs.

        Returns the number of chunks updated (0 if sanity check failed).
        """
        rows = self._conn.execute(
            "SELECT rel_path FROM chunks WHERE repo_id IS NULL LIMIT 5"
        ).fetchall()
        if not rows:
            return 0
        if not any((sample_root / r["rel_path"]).is_file() for r in rows):
            log.warning(
                "storage.backfill_skipped_sanity",
                sample_root=str(sample_root),
                samples=[r["rel_path"] for r in rows],
            )
            return 0
        cur = self._conn.execute(
            "UPDATE chunks SET repo_id = ? WHERE repo_id IS NULL",
            (repo_id,),
        )
        return int(cur.rowcount or 0)

    def list_repos(self) -> list[RepoStats]:
        """Per-repo aggregate counters.

        Returns both the git origin URL and local filesystem path for each
        repo, so callers can access files via either method.
        """
        rows = self._conn.execute(
            """
            SELECT
                r.normalized_key,
                r.remote_url      AS url,
                r.root_path       AS local_path,
                r.default_branch,
                r.head_commit,
                r.last_indexed_at,
                COUNT(DISTINCT c.rel_path) AS files,
                COUNT(c.id)                AS chunks
            FROM repos r
            LEFT JOIN chunks c ON c.repo_id = r.id
            GROUP BY r.id
            ORDER BY chunks DESC
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

    # ----- writes -----

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
        enricher_id: str | None = None,
    ) -> bool:
        """True if the stored chunk matches every (hash, model, enricher).

        Per ADR-004 (pipeline-adapters): adapter drift triggers re-processing
        automatically. Pass ``enricher_id=None`` to preserve the v0.1.0
        behavior (only hash + model checked) — used by legacy code paths.

        Back-compat: a stored NULL ``enricher_id`` (chunks indexed before
        schema v3) is treated as equivalent to ``'noop'`` so upgrading from
        v0.1.0 with the default enricher does NOT cause a full re-embed.
        """
        row = self._conn.execute(
            """SELECT content_hash, embedding_model, enricher_id
               FROM chunks WHERE chunk_id = ?""",
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

    # ----- enrichment cache (pipeline-adapters Phase 1) -------------------

    def get_enrichment(self, content_hash: str, enricher_id: str) -> str | None:
        """Return cached embedding_text for ``(content_hash, enricher_id)`` or None."""
        row = self._conn.execute(
            """SELECT embedding_text FROM enrichments
               WHERE content_hash = ? AND enricher_id = ?""",
            (content_hash, enricher_id),
        ).fetchone()
        return row["embedding_text"] if row else None

    def put_enrichment(
        self, content_hash: str, enricher_id: str, embedding_text: str
    ) -> None:
        """Cache an enrichment result. Safe to call concurrently (UPSERT)."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO enrichments
               (content_hash, enricher_id, embedding_text, cached_at)
               VALUES (?, ?, ?, ?)""",
            (content_hash, enricher_id, embedding_text, now),
        )

    # ----- file-level dirty cache (Phase A incremental, schema v6) --------

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
        """True iff the stored ``files`` row for ``(repo_id, rel_path)``
        matches ``(mtime, size_bytes)`` AND was last indexed by the same
        ``(enricher_id, embedder_model)`` pair currently active.

        The pipeline config check honours the ADR-004 drift contract:
        swapping the enricher MUST force re-processing of every file even
        when the on-disk content is identical. Pass ``None`` for either
        parameter to skip that side of the comparison (used by tests).

        Returns ``False`` for unknown files (treated as "needs indexing").
        """
        row = self._conn.execute(
            """SELECT mtime, size_bytes, last_enricher_id, last_embedder_model
               FROM files
               WHERE repo_id = ? AND rel_path = ?""",
            (repo_id, rel_path),
        ).fetchone()
        if row is None:
            return False
        # mtime comes from filesystem (float seconds). Allow exact match;
        # rounding error is essentially impossible since we store what we
        # read with no transformation.
        if float(row["mtime"]) != float(mtime):
            return False
        if int(row["size_bytes"]) != int(size_bytes):
            return False
        if enricher_id is not None and row["last_enricher_id"] != enricher_id:
            return False
        return not (
            embedder_model is not None
            and row["last_embedder_model"] != embedder_model
        )

    def load_file_meta_map(self, repo_id: int) -> dict[str, StoredFileMeta]:
        """Bulk-load the freshness columns for every ``files`` row of a repo.

        One query for the whole repo, vs ``file_is_fresh`` which is one
        query per file. The hash-based reconcile path (WIN-425) uses this
        to diff the working tree against the index in memory — O(files)
        stats + a dict lookup, no per-file SQL round-trip.
        """
        rows = self._conn.execute(
            """SELECT rel_path, mtime, size_bytes,
                      last_enricher_id, last_embedder_model
               FROM files WHERE repo_id = ?""",
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
        """Record that ``rel_path`` was indexed at ``mtime`` / ``size_bytes``."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO files
                 (repo_id, rel_path, content_hash, size_bytes, mtime,
                  last_indexed_at, last_chunk_count,
                  last_enricher_id, last_embedder_model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(repo_id, rel_path) DO UPDATE SET
                  content_hash = excluded.content_hash,
                  size_bytes = excluded.size_bytes,
                  mtime = excluded.mtime,
                  last_indexed_at = excluded.last_indexed_at,
                  last_chunk_count = excluded.last_chunk_count,
                  last_enricher_id = excluded.last_enricher_id,
                  last_embedder_model = excluded.last_embedder_model""",
            (
                repo_id, rel_path, content_hash, size_bytes, mtime,
                now, chunk_count, enricher_id, embedder_model,
            ),
        )

    def delete_file_meta(self, repo_id: int, rel_paths: list[str]) -> int:
        """Drop the ``files`` rows for ``rel_paths`` under ``repo_id``."""
        if not rel_paths:
            return 0
        placeholders = ",".join("?" * len(rel_paths))
        cur = self._conn.execute(
            f"""DELETE FROM files
               WHERE repo_id = ? AND rel_path IN ({placeholders})""",
            (repo_id, *rel_paths),
        )
        return int(cur.rowcount or 0)

    def list_stale_files(
        self, repo_id: int, current_rel_paths: list[str],
    ) -> list[str]:
        """Return ``rel_paths`` of ``files`` rows for ``repo_id`` that are
        NOT present in ``current_rel_paths``. Caller uses this to identify
        files that have been removed from the working tree and need their
        chunks evicted.
        """
        current = set(current_rel_paths)
        rows = self._conn.execute(
            "SELECT rel_path FROM files WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        return [r["rel_path"] for r in rows if r["rel_path"] not in current]

    def delete_by_paths(self, rel_paths: list[str]) -> int:
        """Remove every chunk whose rel_path is in the list. Returns count."""
        if not rel_paths:
            return 0
        placeholders = ",".join("?" * len(rel_paths))
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT id FROM chunks WHERE rel_path IN ({placeholders})", rel_paths
        )
        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            return 0
        id_placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(
            f"DELETE FROM vec_chunks_secondary WHERE rowid IN ({id_placeholders})", ids
        )
        cur.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(f"DELETE FROM chunks WHERE id IN ({id_placeholders})", ids)
        return len(ids)

    def rename_chunks(
        self, repo_id: int, old_rel_path: str, new_rel_path: str,
    ) -> int:
        """Move every chunk's ``rel_path`` from ``old_rel_path`` to
        ``new_rel_path`` for the given repo, **recomputing chunk_id**
        and regenerating the FTS5 row so the chunk identity remains
        stable across renames.

        Phase C contract: the user-visible win is that subsequent
        `chunk_is_fresh()` calls find the renamed chunks under the new
        path, so they are skipped instead of re-embedded. Cost: one
        hash + one FTS5 DELETE+INSERT per chunk. For typical renames
        (≤ tens of chunks) this is sub-ms.

        Returns the number of chunk rows touched.
        """
        if old_rel_path == new_rel_path:
            return 0

        # Step 1: identify chunks under the old path and load enough
        # data to recompute their chunk_id.
        rows = self._conn.execute(
            """SELECT c.id, c.chunk_id, c.start_line, c.end_line,
                      c.content, c.symbol, c.content_hash,
                      r.normalized_key AS repo_key
               FROM chunks c
               LEFT JOIN repos r ON r.id = c.repo_id
               WHERE c.repo_id = ? AND c.rel_path = ?""",
            (repo_id, old_rel_path),
        ).fetchall()

        if not rows:
            # No chunks for this path — still update file_meta for
            # consistency with the walker's notion of rename.
            self._patch_file_meta_rename(repo_id, old_rel_path, new_rel_path)
            return 0

        # Lazy import to avoid a top-of-module dependency on a sibling
        # package from storage/.
        from ..ingest.chunkers.base import make_chunk_id

        cur = self._conn.cursor()
        moved = 0
        for row in rows:
            new_chunk_id = make_chunk_id(
                new_rel_path,
                int(row["start_line"]),
                int(row["end_line"]),
                row["content_hash"],
                repo_key=row["repo_key"],
            )
            cur.execute(
                """UPDATE chunks
                   SET rel_path = ?, chunk_id = ?
                   WHERE id = ?""",
                (new_rel_path, new_chunk_id, int(row["id"])),
            )
            # Regenerate FTS5 row so path tokens reflect the new path.
            rowid = int(row["id"])
            cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
            cur.execute(
                "INSERT INTO fts_chunks(rowid, content, rel_path) VALUES (?, ?, ?)",
                (
                    rowid,
                    _fts_text(row["content"], new_rel_path, row["symbol"]),
                    new_rel_path,
                ),
            )
            moved += 1

        self._patch_file_meta_rename(repo_id, old_rel_path, new_rel_path)
        return moved

    def _patch_file_meta_rename(
        self, repo_id: int, old_rel_path: str, new_rel_path: str,
    ) -> None:
        """Update the `files` (file_meta) row for a rename, handling the
        case where a row for the destination path already exists."""
        self._conn.execute(
            """UPDATE OR IGNORE files SET rel_path = ?
               WHERE repo_id = ? AND rel_path = ?""",
            (new_rel_path, repo_id, old_rel_path),
        )
        # If UPDATE OR IGNORE was a no-op due to UNIQUE collision, drop
        # the stranded old row.
        self._conn.execute(
            "DELETE FROM files WHERE repo_id = ? AND rel_path = ?",
            (repo_id, old_rel_path),
        )

    def delete_chunks_by_repo_paths(
        self, repo_id: int, rel_paths: list[str],
    ) -> int:
        """Repo-scoped variant of ``delete_by_paths``. Used by the
        incremental pipeline to evict chunks of files that no longer
        exist in a specific repo, without touching identically-named
        files in other indexed repos.
        """
        if not rel_paths:
            return 0
        placeholders = ",".join("?" * len(rel_paths))
        cur = self._conn.cursor()
        cur.execute(
            f"""SELECT id FROM chunks
               WHERE repo_id = ? AND rel_path IN ({placeholders})""",
            (repo_id, *rel_paths),
        )
        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            return 0
        id_placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(
            f"DELETE FROM vec_chunks_secondary WHERE rowid IN ({id_placeholders})", ids
        )
        cur.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({id_placeholders})", ids)
        cur.execute(f"DELETE FROM chunks WHERE id IN ({id_placeholders})", ids)
        return len(ids)

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk with the given ID exists."""
        row = self._conn.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ? LIMIT 1", (chunk_id,)
        ).fetchone()
        return row is not None

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Get a chunk by its ID.
        
        Returns:
            Dictionary with chunk fields, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by its ID.
        
        Returns:
            True if deleted, False if not found.
        """
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT id FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return False
        
        rowid = int(row["id"])
        cur.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
        cur.execute("DELETE FROM vec_chunks_secondary WHERE rowid = ?", (rowid,))
        cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
        cur.execute("DELETE FROM chunks WHERE id = ?", (rowid,))
        return True

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
        """Insert (or update) a chunk and its vector atomically. Returns rowid.

        ``repo_id`` carries the source-repository identity (see ``repos`` table).
        ``embedding_text`` / ``enricher_id`` (schema v3) record what was
        actually embedded and by which enricher adapter, enabling drift
        detection. Both default to None so legacy callers stay compatible —
        in that case ``embedding_text`` falls back to ``chunk.content``.
        """
        if len(embedding) != self._dim:
            raise StorageError(
                f"Embedding dim mismatch: got {len(embedding)}, expected {self._dim}"
            )
        now = datetime.now(UTC).isoformat()
        # Default to source content when caller (e.g. legacy IndexPipeline)
        # has not gone through an enricher.
        effective_text = embedding_text if embedding_text is not None else chunk.content
        cur = self._conn.cursor()
        # Try update first to preserve rowid (vec_chunks linkage).
        cur.execute(
            "SELECT id, content_hash FROM chunks WHERE chunk_id = ?",
            (chunk.chunk_id,),
        )
        existing = cur.fetchone()
        if existing is not None:
            rowid = int(existing["id"])
            if existing["content_hash"] != chunk.content_hash:
                cur.execute(
                    "DELETE FROM vec_chunks_secondary WHERE rowid = ?", (rowid,)
                )
            cur.execute(
                """
                UPDATE chunks SET
                    rel_path=?, language=?, kind=?, symbol=?, parent_chunk_id=?,
                    start_line=?, end_line=?, content=?, content_hash=?,
                    embedding_model=?, embedding_dim=?, indexed_at=?, repo_id=?,
                    embedding_text=?, enricher_id=?
                WHERE id = ?
                """,
                (
                    chunk.rel_path, chunk.language, chunk.kind, chunk.symbol,
                    chunk.parent_chunk_id, chunk.start_line, chunk.end_line,
                    chunk.content, chunk.content_hash, embedding_model,
                    embedding_dim, now, repo_id,
                    effective_text, enricher_id, rowid,
                ),
            )
            cur.execute("DELETE FROM vec_chunks WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (rowid,))
        else:
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, rel_path, language, kind, symbol, parent_chunk_id,
                    start_line, end_line, content, content_hash,
                    embedding_model, embedding_dim, indexed_at, repo_id,
                    embedding_text, enricher_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    chunk.chunk_id, chunk.rel_path, chunk.language, chunk.kind,
                    chunk.symbol, chunk.parent_chunk_id, chunk.start_line,
                    chunk.end_line, chunk.content, chunk.content_hash,
                    embedding_model, embedding_dim, now, repo_id,
                    effective_text, enricher_id,
                ),
            )
            rowid = int(cur.lastrowid or 0)

        cur.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_vector(embedding)),
        )
        cur.execute(
            "INSERT INTO fts_chunks(rowid, content, rel_path) VALUES (?, ?, ?)",
            (rowid, _fts_text(chunk.content, chunk.rel_path, chunk.symbol), chunk.rel_path),
        )
        return rowid

    # ----- independent secondary embeddings (WIN-300) --------------------

    def list_chunks_needing_secondary_embeddings(
        self,
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
        limit: int = 1000,
    ) -> list[SecondaryEmbeddingChunk]:
        """Return chunks missing a current vector for the secondary embedder."""
        if embedding_dim != SECONDARY_EMBEDDING_DIM:
            raise StorageError(
                f"Secondary embedding dim {embedding_dim} != provisioned dim "
                f"{SECONDARY_EMBEDDING_DIM}"
            )
        if limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, c.content_hash, c.content
            FROM chunks c
            LEFT JOIN vec_chunks_secondary v ON v.rowid = c.id
            WHERE v.rowid IS NULL
               OR v.content_hash IS NOT c.content_hash
               OR v.embedder_id IS NOT ?
               OR v.embedding_dim IS NOT ?
            ORDER BY c.id
            LIMIT ?
            """,
            (embedder_id, embedding_dim, limit),
        ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "content_hash": row["content_hash"],
                "content": row["content"],
            }
            for row in rows
        ]

    def upsert_secondary_embeddings(
        self,
        vectors: list[SecondaryEmbeddingVector],
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
    ) -> int:
        """Insert a batch of vectors that still match current chunk content."""
        if embedding_dim != SECONDARY_EMBEDDING_DIM:
            raise StorageError(
                f"Secondary embedding dim {embedding_dim} != provisioned dim "
                f"{SECONDARY_EMBEDDING_DIM}"
            )
        for item in vectors:
            if len(item["embedding"]) != embedding_dim:
                raise StorageError(
                    f"Secondary embedding dim mismatch for {item['chunk_id']}: "
                    f"got {len(item['embedding'])}, expected {embedding_dim}"
                )
        cur = self._conn.cursor()
        upserted = 0
        for item in vectors:
            embedding = item["embedding"]
            row = cur.execute(
                "SELECT id FROM chunks WHERE chunk_id = ? AND content_hash = ?",
                (item["chunk_id"], item["content_hash"]),
            ).fetchone()
            if row is None:
                continue
            rowid = int(row["id"])
            cur.execute("DELETE FROM vec_chunks_secondary WHERE rowid = ?", (rowid,))
            cur.execute(
                """INSERT INTO vec_chunks_secondary
                   (rowid, embedding, content_hash, embedder_id, embedding_dim)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    rowid,
                    _serialize_vector(embedding),
                    item["content_hash"],
                    embedder_id,
                    embedding_dim,
                ),
            )
            upserted += 1
        return upserted

    def count_secondary_embeddings(
        self,
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
    ) -> int:
        """Count current secondary vectors for one exact embedder identity."""
        if embedding_dim != SECONDARY_EMBEDDING_DIM:
            return 0
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM vec_chunks_secondary v
            JOIN chunks c ON c.id = v.rowid
            WHERE v.content_hash = c.content_hash
              AND v.embedder_id = ?
              AND v.embedding_dim = ?
            """,
            (embedder_id, embedding_dim),
        ).fetchone()
        return int(row["n"]) if row else 0

    def search_secondary_embeddings(
        self,
        query_vec: list[float],
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
        k: int = 10,
    ) -> list[SearchHit]:
        """Search only current vectors from one exact secondary embedder."""
        if len(query_vec) != embedding_dim:
            raise StorageError(
                f"Secondary query dim {len(query_vec)} != requested dim {embedding_dim}"
            )
        if embedding_dim != SECONDARY_EMBEDDING_DIM:
            raise StorageError(
                f"Secondary embedding dim {embedding_dim} != provisioned dim "
                f"{SECONDARY_EMBEDDING_DIM}"
            )
        if k <= 0:
            return []
        rows = self._conn.execute(
            f"""
            SELECT
                {_REPO_JOIN_COLUMNS},
                v.distance AS distance
            FROM vec_chunks_secondary v
            JOIN chunks c ON c.id = v.rowid
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE v.embedding MATCH ?
              AND v.embedder_id = ?
              AND v.embedding_dim = ?
              AND v.content_hash = c.content_hash
              AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(query_vec), embedder_id, embedding_dim, k),
        ).fetchall()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row["distance"])
            score = max(0.0, 1.0 - (distance * distance) / 2.0)
            hits.append(
                SearchHit(
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
            )
        return hits

    # ----- L2 retroactive FTS5 augmentation (Trilha A) --------------------

    def augment_fts_with_graph(self) -> dict[str, int]:
        """Rebuild FTS5 rows with graph-derived terms appended.

        For every ``chunks`` row whose ``(rel_path, line range)`` overlaps a
        ``graph_nodes`` entry, the FTS5 document is regenerated as the
        legacy ``_fts_text(...)`` *plus* the matching ``community_label`` and
        ``node label``. Queries that mention a community-level concept
        (e.g. "where is the hybrid retrieval pipeline?") then match these
        chunks even when their body lacks the literal term.

        Retroactive: no re-embedding, no re-walking. The augmentation
        works on the existing index after a graph load. Idempotent:
        running it twice produces the same FTS5 state.

        Returns ``{augmented, unchanged, missing_graph}`` counts.
        """
        cur = self._conn.cursor()
        # Check graph_nodes is populated before doing anything.
        try:
            has_graph = cur.execute(
                "SELECT 1 FROM graph_nodes LIMIT 1"
            ).fetchone() is not None
        except sqlite3.OperationalError:
            has_graph = False
        if not has_graph:
            return {"augmented": 0, "unchanged": 0, "missing_graph": 1}

        # Find every chunk that overlaps a graph node. We compute the
        # tightest match per chunk so multi-node files (most are) still
        # produce a focused augmentation per chunk.
        rows = cur.execute(
            """
            SELECT c.id, c.chunk_id, c.rel_path, c.symbol, c.content,
                   c.start_line, c.end_line,
                   n.label, n.community_label, n.source_location
            FROM chunks c
            JOIN graph_nodes n
              ON n.source_file = c.rel_path
            WHERE n.source_location IS NOT NULL
            """
        ).fetchall()

        # Group rows by chunk id, pick the tightest node per chunk.
        by_chunk: dict[int, dict] = {}
        for r in rows:
            cid = int(r["id"])
            loc = r["source_location"] or ""
            line: int | None = None
            stripped = loc.lstrip("L")
            try:
                line = int(stripped.split(":")[0])
            except (ValueError, TypeError):
                continue
            if not (r["start_line"] <= line <= r["end_line"]):
                continue
            best = by_chunk.get(cid)
            cand_score = -abs(line - r["start_line"])  # closer = better
            if best is None or best["_score"] < cand_score:
                by_chunk[cid] = {
                    "rowid": cid,
                    "rel_path": r["rel_path"],
                    "symbol": r["symbol"],
                    "content": r["content"],
                    "label": r["label"],
                    "community_label": r["community_label"],
                    "_score": cand_score,
                }

        augmented = 0
        for cid, info in by_chunk.items():
            extra_parts: list[str] = []
            if info["community_label"]:
                extra_parts.append(f"community: {info['community_label']}")
            if info["label"]:
                extra_parts.append(f"node: {info['label']}")
            if not extra_parts:
                continue
            base = _fts_text(info["content"], info["rel_path"], info["symbol"])
            new_doc = base + "\n" + "\n".join(extra_parts)
            cur.execute("DELETE FROM fts_chunks WHERE rowid = ?", (cid,))
            cur.execute(
                "INSERT INTO fts_chunks(rowid, content, rel_path) VALUES (?, ?, ?)",
                (cid, new_doc, info["rel_path"]),
            )
            augmented += 1

        # The remaining chunks (no graph match) are left as-is.
        total_chunks = int(
            cur.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        )
        unchanged = total_chunks - augmented
        self.commit()
        log.info(
            "fts_augment.done",
            augmented=augmented, unchanged=unchanged, total=total_chunks,
        )
        return {"augmented": augmented, "unchanged": unchanged, "missing_graph": 0}

    def commit(self) -> None:
        self._conn.commit()

    def clear(self) -> None:
        """Wipe the index. Used by `index --rebuild`.

        Preserves the ``repos`` table — repository identity survives rebuilds
        so the same id is reused, keeping FK semantics meaningful across runs.

        Also clears the ``files`` table so incremental indexing sees all files
        as dirty after a rebuild.

        Note: ``DELETE FROM`` marks pages as free for reuse but does not shrink
        the SQLite file on disk. Call :meth:`vacuum` to reclaim disk space.
        """
        cur = self._conn.cursor()
        cur.executescript(
            """
            DELETE FROM vec_chunks;
            DELETE FROM vec_chunks_secondary;
            DELETE FROM fts_chunks;
            DELETE FROM chunks;
            DELETE FROM files;
            DELETE FROM enrichments;
            """
        )
        self._conn.commit()

    def vacuum(self) -> tuple[int, int]:
        """Reclaim unused disk space by rebuilding the database file.

        SQLite ``DELETE FROM`` only marks pages as free for reuse — the file
        size on disk stays the same. ``VACUUM`` rewrites the entire database
        file, dropping unused pages and reclaiming disk space.

        VACUUM cannot run inside a transaction. We commit any pending work
        first, then issue VACUUM. The operation can be slow on large
        databases (it rewrites the whole file), so use sparingly.

        WAL mode caveat: in WAL journal mode (our default), VACUUM moves
        the freed pages to the WAL file. The main ``.db`` file only shrinks
        after a ``wal_checkpoint(TRUNCATE)``, which we issue here so the
        on-disk size visible to the user matches the reported size_after.

        Returns:
            Tuple of (size_before_bytes, size_after_bytes).
        """
        from pathlib import Path

        # Resolve the actual file path. ``:memory:`` databases cannot be
        # vacuumed in a meaningful way (no file to shrink).
        db_path_row = self._conn.execute(
            "PRAGMA database_list"
        ).fetchone()
        db_file = db_path_row["file"] if db_path_row else ""
        if not db_file:
            # In-memory database — VACUUM still works but file size is 0.
            self._conn.execute("VACUUM")
            return (0, 0)

        # Force the WAL to flush BEFORE measuring so size_before reflects
        # the actual disk footprint the user sees.
        self._conn.commit()
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size_before = Path(db_file).stat().st_size

        # VACUUM cannot run inside a transaction.
        self._conn.execute("VACUUM")
        self._conn.commit()
        # Truncate the WAL so the reclaimed pages actually leave disk.
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        size_after = Path(db_file).stat().st_size
        return (size_before, size_after)

    # ----- reads -----

    def search_dense(self, query_vec: list[float], k: int = 10) -> list[SearchHit]:
        """Top-k dense nearest neighbors."""
        if len(query_vec) != self._dim:
            raise StorageError(
                f"Query embedding dim {len(query_vec)} != index dim {self._dim}"
            )
        rows = self._conn.execute(
            f"""
            SELECT
                {_REPO_JOIN_COLUMNS},
                v.distance AS distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(query_vec), k),
        ).fetchall()
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row["distance"])
            # vec_distance is L2 on normalized vectors → score = 1 - d²/2
            score = max(0.0, 1.0 - (distance * distance) / 2.0)
            hits.append(
                SearchHit(
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
            )
        return hits

    def search_bm25(self, query: str, k: int = 50) -> list[SearchHit]:
        """Top-k via FTS5 BM25 ranking. Returns [] for empty/sanitized queries."""
        match_expr = _sanitize_fts_query(query)
        if not match_expr:
            return []
        try:
            rows = self._conn.execute(
                f"""
                SELECT
                    {_REPO_JOIN_COLUMNS},
                    bm25(fts_chunks) AS score
                FROM fts_chunks
                JOIN chunks c ON c.id = fts_chunks.rowid
                LEFT JOIN repos r ON r.id = c.repo_id
                WHERE fts_chunks MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("storage.fts_query_failed", query=query, error=str(exc))
            return []
        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            # FTS5 bm25() returns a non-positive number; closer to 0 is better.
            # Normalize to [0, 1) where 1 is most relevant.
            raw = float(row["score"])
            score = 1.0 / (1.0 + max(0.0, -raw))
            hits.append(
                SearchHit(
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
            )
        return hits

    def search_symbol_tokens(self, query: str, k: int = 20) -> list[SearchHit]:
        """Match chunks whose `symbol` column contains identifier tokens from query.

        Used as a third stream in hybrid search (spec §6.3.5 — query routing).
        Cheap because the symbol column is small. Falls back to empty if no
        useful tokens survive the stopword filter.
        """
        tokens = _identifier_tokens(query)
        if not tokens:
            return []
        # OR-join LIKE patterns; rank by token-match count + length (shorter
        # symbols win on ties).
        like_clauses = " OR ".join("symbol LIKE ? COLLATE NOCASE" for _ in tokens)
        params: list[Any] = [f"%{t}%" for t in tokens]
        # Score = number of tokens matched.
        score_terms = " + ".join(
            "(CASE WHEN symbol LIKE ? COLLATE NOCASE THEN 1 ELSE 0 END)"
            for _ in tokens
        )
        params = params + [f"%{t}%" for t in tokens] + [k]
        rows = self._conn.execute(
            f"""
            SELECT {_REPO_JOIN_COLUMNS},
                   ({score_terms}) AS match_count
            FROM chunks c
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE ({like_clauses.replace("symbol", "c.symbol")})
              AND c.symbol IS NOT NULL
            ORDER BY match_count DESC, LENGTH(c.symbol) ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            SearchHit(
                rank=rank,
                score=float(row["match_count"]),
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
            for rank, row in enumerate(rows, start=1)
        ]

    # ----- structured lookups (used by MCP tools) -----

    # ----- sessions (v8) -----

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Load a session row by id. Returns ``None`` when unknown.

        WIN-221 / EPIC 1.9: this is the read side of the sessions
        table. ``upsert_session`` is the write side; together they
        persist the WIN-217 seen-set + WIN-218 pin-set so the agent's
        working state survives MCP-server restarts.
        """
        from ..models import SessionRecord

        row = self._conn.execute(
            """SELECT session_id, client_id, created_at, updated_at,
                      seen_chunk_ids, pinned_chunk_ids, meta, scratchpad
               FROM sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            seen = set(json.loads(row["seen_chunk_ids"] or "[]"))
            pinned = set(json.loads(row["pinned_chunk_ids"] or "[]"))
            meta = json.loads(row["meta"] or "{}")
        except (json.JSONDecodeError, TypeError):
            # Corrupted row → log + treat as empty state instead of
            # crashing the MCP server. The next upsert will rewrite
            # the JSON with valid content.
            log.warning(
                "sessions.corrupted_row",
                session_id=session_id,
            )
            seen, pinned, meta = set(), set(), {}
        scratchpad = (
            row["scratchpad"] or ""
            if "scratchpad" in row.keys() else ""
        )
        return SessionRecord(
            session_id=row["session_id"],
            client_id=row["client_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            seen_chunk_ids=seen,
            pinned_chunk_ids=pinned,
            meta=meta,
            scratchpad=scratchpad,
        )

    def get_latest_session(self) -> SessionRecord | None:
        """Return the most-recently-updated session row, or ``None``
        when the table is empty.

        Backs WIN-253's ``stropha://working-set`` resource — that
        FastMCP resource can't access the lifespan AppContext, so
        we surface the freshest session row from the v8 ``sessions``
        table instead.
        """
        row = self._conn.execute(
            """SELECT session_id FROM sessions
               ORDER BY updated_at DESC LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        return self.get_session(row["session_id"])

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
        """Insert or update a session row. Idempotent.

        First call creates the row with ``created_at = updated_at =
        now``. Subsequent calls bump ``updated_at`` and overwrite the
        sets / meta dict. ``client_id`` once set is preserved across
        calls (passing ``None`` does NOT clear it — callers must pass
        an empty string explicitly if they want to wipe).

        Returns the canonical :class:`SessionRecord` after the write
        so callers can use the post-upsert timestamps directly.
        """
        from ..models import SessionRecord

        now = datetime.now(UTC).isoformat()
        cur = self._conn.cursor()
        existing = cur.execute(
            "SELECT created_at, client_id, scratchpad FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seen_json = json.dumps(sorted(seen_chunk_ids or set()))
        pinned_json = json.dumps(sorted(pinned_chunk_ids or set()))
        meta_json = json.dumps(meta or {}, sort_keys=True)
        # WIN-229: preserve scratchpad on update when caller passes
        # None (the common case — most upserts only touch the chunk
        # sets). Pass an empty string explicitly to clear.
        if existing is None:
            scratchpad_resolved = scratchpad or ""
            cur.execute(
                """INSERT INTO sessions (
                       session_id, client_id, created_at, updated_at,
                       seen_chunk_ids, pinned_chunk_ids, meta, scratchpad
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, client_id, now, now,
                    seen_json, pinned_json, meta_json, scratchpad_resolved,
                ),
            )
            created_at = now
            resolved_client_id = client_id
        else:
            # Preserve client_id when caller passes None (most common
            # case — the agent only updates the state sets).
            resolved_client_id = (
                client_id if client_id is not None else existing["client_id"]
            )
            scratchpad_resolved = (
                scratchpad if scratchpad is not None
                else (existing["scratchpad"] or "")
            )
            cur.execute(
                """UPDATE sessions
                       SET client_id = ?, updated_at = ?,
                           seen_chunk_ids = ?, pinned_chunk_ids = ?,
                           meta = ?, scratchpad = ?
                     WHERE session_id = ?""",
                (
                    resolved_client_id, now, seen_json, pinned_json,
                    meta_json, scratchpad_resolved, session_id,
                ),
            )
            created_at = existing["created_at"]
        self._conn.commit()
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
        """Remove a session row. Returns ``True`` on delete, ``False``
        when the row didn't exist (caller's no-op signal)."""
        cur = self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_sessions(
        self, *, client_id: str | None = None, limit: int = 50,
    ) -> list[SessionRecord]:
        """Enumerate sessions, optionally filtered by ``client_id``.

        Used by WIN-228 (TTL cleanup) + WIN-222 (cross-session memory).
        Sorted by ``updated_at`` descending so "the most recent session
        for this client" is row zero.
        """
        from ..models import SessionRecord

        if client_id is None:
            rows = self._conn.execute(
                """SELECT session_id, client_id, created_at, updated_at,
                          seen_chunk_ids, pinned_chunk_ids, meta, scratchpad
                   FROM sessions ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT session_id, client_id, created_at, updated_at,
                          seen_chunk_ids, pinned_chunk_ids, meta, scratchpad
                   FROM sessions
                   WHERE client_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (client_id, limit),
            ).fetchall()
        out: list[SessionRecord] = []
        for row in rows:
            try:
                seen = set(json.loads(row["seen_chunk_ids"] or "[]"))
                pinned = set(json.loads(row["pinned_chunk_ids"] or "[]"))
                meta = json.loads(row["meta"] or "{}")
            except (json.JSONDecodeError, TypeError):
                log.warning(
                    "sessions.corrupted_row",
                    session_id=row["session_id"],
                )
                seen, pinned, meta = set(), set(), {}
            scratchpad = (
                row["scratchpad"] or ""
                if "scratchpad" in row.keys() else ""
            )
            out.append(
                SessionRecord(
                    session_id=row["session_id"],
                    client_id=row["client_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    seen_chunk_ids=seen,
                    pinned_chunk_ids=pinned,
                    meta=meta,
                    scratchpad=scratchpad,
                )
            )
        return out

    def prune_sessions(self, *, older_than_iso: str) -> int:
        """Delete sessions whose ``updated_at`` is older than the cutoff.

        WIN-228 hooks here for TTL cleanup. Returns the number of rows
        removed.
        """
        cur = self._conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (older_than_iso,)
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    # ----- project memory (v9, WIN-222) -----

    def lookup_repo_id(self, normalized_key: str) -> int | None:
        """Resolve a ``normalized_key`` to its ``repos.id``.

        Used by WIN-223 (MCP remember/recall tools) to translate the
        configured ``target_repo`` into the FK the project_memory
        table requires. Returns ``None`` when the repo has never
        been indexed (no row yet) — caller should index first or
        use a different repo.
        """
        row = self._conn.execute(
            "SELECT id FROM repos WHERE normalized_key = ?",
            (normalized_key,),
        ).fetchone()
        return int(row["id"]) if row is not None else None

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
        """Upsert a project-memory entry. Idempotent.

        Args:
            repo_id: foreign key to ``repos.id``. Project memory is
                always scoped to a repo so cross-repo leakage is
                impossible.
            kind: free-form bucket (``"decision"`` / ``"constraint"``
                / ``"fact"`` / ``"convention"`` / agent's choice).
            key: stable short identifier within ``(repo_id, kind)``.
                Re-using the key updates the existing row.
            value: the actual note (markdown allowed). No length cap
                — SQLite TEXT handles arbitrary length, the agent's
                context window is the real limit.
            source: optional session_id provenance (WIN-160 / WIN-221).
            expires_at: optional ISO timestamp after which the entry
                is considered stale. ``None`` = perpetual.

        Returns:
            The :class:`ProjectMemoryRecord` after the upsert with the
            authoritative timestamps populated.
        """
        from ..models import ProjectMemoryRecord

        now = datetime.now(UTC).isoformat()
        cur = self._conn.cursor()
        existing = cur.execute(
            """SELECT id, created_at FROM project_memory
               WHERE repo_id = ? AND kind = ? AND key = ?""",
            (repo_id, kind, key),
        ).fetchone()
        if existing is None:
            cur.execute(
                """INSERT INTO project_memory (
                       repo_id, kind, key, value, source,
                       created_at, updated_at, expires_at,
                       provenance_commit
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    repo_id, kind, key, value, source, now, now,
                    expires_at, provenance_commit,
                ),
            )
            row_id = int(cur.lastrowid or 0)
            created_at = now
        else:
            cur.execute(
                """UPDATE project_memory
                       SET value = ?, source = ?, updated_at = ?,
                           expires_at = ?, provenance_commit = ?
                     WHERE id = ?""",
                (
                    value, source, now, expires_at,
                    provenance_commit, existing["id"],
                ),
            )
            row_id = int(existing["id"])
            created_at = existing["created_at"]
        self._conn.commit()
        return ProjectMemoryRecord(
            id=row_id, repo_id=repo_id, kind=kind, key=key, value=value,
            source=source, created_at=created_at, updated_at=now,
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
        """Fetch project-memory entries for a repo.

        Filter cascade: ``repo_id`` (required) → ``kind`` (optional) →
        ``key`` (optional). When both ``kind`` and ``key`` are set,
        the unique constraint returns at most one row.

        ``include_expired=True`` returns entries whose ``expires_at``
        is in the past — useful for debugging / audit; the default
        treats them as gone.

        Sorted by ``updated_at`` DESC so the freshest memory shows
        first.
        """
        from ..models import ProjectMemoryRecord

        clauses = ["repo_id = ?"]
        params: list[Any] = [repo_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now(UTC).isoformat())
        params.append(limit)
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"""SELECT id, repo_id, kind, key, value, source,
                       created_at, updated_at, expires_at,
                       provenance_commit
               FROM project_memory
               WHERE {where}
               ORDER BY updated_at DESC LIMIT ?""",
            tuple(params),
        ).fetchall()
        return [
            ProjectMemoryRecord(
                id=int(r["id"]),
                repo_id=int(r["repo_id"]),
                kind=r["kind"],
                key=r["key"],
                value=r["value"],
                source=r["source"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                expires_at=r["expires_at"],
                provenance_commit=(
                    r["provenance_commit"]
                    if "provenance_commit" in r.keys() else None
                ),
            )
            for r in rows
        ]

    def forget(self, *, repo_id: int, kind: str, key: str) -> bool:
        """Delete a single project-memory entry. Returns ``True`` on
        delete, ``False`` when the row didn't exist."""
        cur = self._conn.execute(
            """DELETE FROM project_memory
               WHERE repo_id = ? AND kind = ? AND key = ?""",
            (repo_id, kind, key),
        )
        self._conn.commit()
        return bool(cur.rowcount or 0)

    # ----- episodic memory (v10, WIN-225) -----

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
        """Insert one episodic-memory row. Always creates a new row;
        episodes are append-only by design (the timeline matters).

        ``touched_files`` / ``touched_symbols`` are deduplicated +
        sorted before persistence so the JSON shape is canonical.
        ``provenance_commit`` (WIN-228) records the git HEAD SHA at
        the moment the episode was logged so the staleness check
        can diff against the current HEAD.
        """
        from ..models import EpisodeRecord

        now = datetime.now(UTC).isoformat()
        files_norm = sorted(set(touched_files or []))
        symbols_norm = sorted(set(touched_symbols or []))
        cur = self._conn.execute(
            """INSERT INTO episodes (
                   repo_id, session_id, title, summary,
                   touched_files, touched_symbols,
                   created_at, updated_at, provenance_commit
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_id, session_id, title, summary,
                json.dumps(files_norm), json.dumps(symbols_norm),
                now, now, provenance_commit,
            ),
        )
        self._conn.commit()
        return EpisodeRecord(
            id=int(cur.lastrowid or 0),
            repo_id=repo_id,
            session_id=session_id,
            title=title,
            summary=summary,
            touched_files=files_norm,
            touched_symbols=symbols_norm,
            created_at=now,
            updated_at=now,
            provenance_commit=provenance_commit,
        )

    def recall_episodes(
        self,
        *,
        repo_id: int,
        query: str | None = None,
        limit: int = 10,
        since_iso: str | None = None,
    ) -> list[EpisodeRecord]:
        """Fetch episodes for a repo, optionally filtered.

        ``query`` is matched against ``title`` and ``summary`` via
        case-insensitive LIKE. None = return the most-recent episodes
        unfiltered.

        ``since_iso`` filters to episodes created on/after the given
        ISO timestamp — useful for "what did I do this week?".

        Sorted by ``created_at`` DESC so the freshest episode is row 0.
        """

        clauses = ["repo_id = ?"]
        params: list[Any] = [repo_id]
        if query:
            clauses.append("(title LIKE ? OR summary LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        if since_iso is not None:
            clauses.append("created_at >= ?")
            params.append(since_iso)
        params.append(max(1, min(limit, 200)))
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"""SELECT id, repo_id, session_id, title, summary,
                       touched_files, touched_symbols,
                       created_at, updated_at, provenance_commit
               FROM episodes
               WHERE {where}
               ORDER BY created_at DESC LIMIT ?""",
            tuple(params),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def episode_co_occurrence(
        self,
        *,
        repo_id: int,
        key: str,
        in_files: bool = True,
        in_symbols: bool = True,
        limit: int = 10,
    ) -> dict[str, dict[str, int]]:
        """Answer "what do I usually touch alongside ``key``?".

        Scans every episode that touched ``key`` (file path OR
        symbol, depending on the flags) and counts how often other
        files / symbols co-appeared. Returns:

            {
                "files":   {"path/a.py": 3, "path/b.py": 2, ...},
                "symbols": {"FooBar.do": 4, ...},
            }

        Counts are sorted descending; both dicts are capped at
        ``limit`` entries. The ``key`` itself is excluded from the
        result (the agent already knows about it).

        Implementation: LIKE-based scan over the JSON arrays. Fine
        for the corpora we target (≤500 episodes per repo); a future
        story can normalise into a join table if it grows.
        """
        if not key:
            return {"files": {}, "symbols": {}}
        like = f"%{json.dumps(key)[1:-1]}%"  # quote-escaped substring

        clauses = ["repo_id = ?"]
        params: list[Any] = [repo_id]
        match_clauses: list[str] = []
        if in_files:
            match_clauses.append("touched_files LIKE ?")
            params.append(like)
        if in_symbols:
            match_clauses.append("touched_symbols LIKE ?")
            params.append(like)
        if match_clauses:
            clauses.append(f"({' OR '.join(match_clauses)})")
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"""SELECT touched_files, touched_symbols
               FROM episodes
               WHERE {where}""",
            tuple(params),
        ).fetchall()
        from collections import Counter

        file_counter: Counter[str] = Counter()
        sym_counter: Counter[str] = Counter()
        for r in rows:
            try:
                files = json.loads(r["touched_files"] or "[]")
                syms = json.loads(r["touched_symbols"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            # Confirm the LIKE hit actually means key is present —
            # JSON escaping might cause false positives.
            if key not in files and key not in syms:
                continue
            for f in files:
                if f != key:
                    file_counter[f] += 1
            for s in syms:
                if s != key:
                    sym_counter[s] += 1
        return {
            "files": dict(file_counter.most_common(max(1, limit))),
            "symbols": dict(sym_counter.most_common(max(1, limit))),
        }

    def forget_episode(self, *, episode_id: int) -> bool:
        """Delete one episode by id. Returns True / False (existed)."""
        cur = self._conn.execute(
            "DELETE FROM episodes WHERE id = ?", (episode_id,)
        )
        self._conn.commit()
        return bool(cur.rowcount or 0)

    def prune_project_memory(
        self,
        *,
        repo_id: int | None = None,
        older_than_iso: str | None = None,
        expired_only: bool = False,
    ) -> int:
        """Delete project-memory entries by filter. Returns count
        removed.

        Args:
            repo_id: when set, only entries for this repo are touched.
            older_than_iso: when set, only entries whose ``updated_at``
                is older than this ISO timestamp are removed.
            expired_only: when True, only entries whose ``expires_at``
                is in the past are removed (overrides
                ``older_than_iso``). WIN-228 hooks here for TTL.

        All filters compose with AND. ``prune_project_memory()`` with
        no args is a no-op — explicit confirmation needed to wipe
        the table.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if repo_id is not None:
            clauses.append("repo_id = ?")
            params.append(repo_id)
        if expired_only:
            clauses.append(
                "(expires_at IS NOT NULL AND expires_at <= ?)"
            )
            params.append(datetime.now(UTC).isoformat())
        elif older_than_iso is not None:
            clauses.append("updated_at < ?")
            params.append(older_than_iso)
        if not clauses:
            # Refuse to wipe everything implicitly — explicit
            # repo_id=... is required.
            return 0
        where = " AND ".join(clauses)
        cur = self._conn.execute(
            f"DELETE FROM project_memory WHERE {where}", tuple(params)
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    def lookup_by_chunk_id(self, chunk_id: str) -> SearchHit | None:
        """Dereference a chunk_id to a fully-formed SearchHit.

        Used by ``expand_chunk`` / ``expand_chunks`` (WIN-216) to
        resolve a lightweight handle into a renderable hit. Returns
        ``None`` when the id is unknown (most likely cause: the
        index was rebuilt since the id was issued, so the agent
        should treat this as a stale-handle signal rather than a
        retrieval error).

        Same column projection as :func:`lookup_by_symbol` so the
        returned hit has the full ``repo``/``snippet``/etc. payload
        every other retrieval path produces.
        """
        row = self._conn.execute(
            f"""
            SELECT {_REPO_JOIN_COLUMNS}
            FROM chunks c
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return SearchHit(
            rank=0,
            score=1.0,
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

    def lookup_by_symbol(self, symbol: str, limit: int = 10) -> list[SearchHit]:
        """Exact + suffix match on symbol column. Used by `get_symbol`.

        Matches `Foo.bar`, `bar` (suffix), and case-insensitively as fallback.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_REPO_JOIN_COLUMNS}
            FROM chunks c
            LEFT JOIN repos r ON r.id = c.repo_id
            WHERE c.symbol = ?
               OR c.symbol LIKE ? COLLATE NOCASE
               OR c.symbol LIKE ? COLLATE NOCASE
            ORDER BY
                CASE WHEN c.symbol = ? THEN 0
                     WHEN c.symbol LIKE ? COLLATE NOCASE THEN 1
                     ELSE 2 END,
                LENGTH(c.symbol)
            LIMIT ?
            """,
            (symbol, f"%.{symbol}", f"%{symbol}%", symbol, f"%.{symbol}", limit),
        ).fetchall()
        return [
            SearchHit(
                rank=rank,
                score=1.0,
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
            for rank, row in enumerate(rows, start=1)
        ]

    def file_outline(self, rel_path: str) -> list[dict[str, Any]]:
        """Symbolic outline of a file (chunks sorted by start_line)."""
        rows = self._conn.execute(
            """
            SELECT chunk_id, kind, symbol, parent_chunk_id, start_line, end_line
            FROM chunks
            WHERE rel_path = ?
            ORDER BY start_line
            """,
            (rel_path,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT chunk_id, rel_path, language, kind, symbol, parent_chunk_id,
                      start_line, end_line, content, embedding_model
               FROM chunks WHERE chunk_id = ?""",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

    # ----- meta key/value (used by incremental reindex) -----

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
        )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def stats(self) -> dict[str, Any]:
        cur = self._conn.cursor()
        chunks = cur.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        files = cur.execute(
            "SELECT COUNT(DISTINCT rel_path) AS n FROM chunks"
        ).fetchone()["n"]
        models = [
            dict(row)
            for row in cur.execute(
                """SELECT embedding_model, embedding_dim, COUNT(*) AS n
                   FROM chunks GROUP BY embedding_model, embedding_dim"""
            ).fetchall()
        ]
        # Per-enricher chunk distribution (schema v3). NULL = legacy / noop.
        enrichers = [
            dict(row)
            for row in cur.execute(
                """SELECT COALESCE(enricher_id, '(none)') AS enricher_id,
                          COUNT(*) AS n
                   FROM chunks GROUP BY enricher_id"""
            ).fetchall()
        ]
        enrichment_cache = cur.execute(
            "SELECT COUNT(*) AS n FROM enrichments"
        ).fetchone()["n"]
        # Schema v6: file-level dirty cache size.
        try:
            files_tracked = cur.execute(
                "SELECT COUNT(*) AS n FROM files"
            ).fetchone()["n"]
        except sqlite3.OperationalError:
            files_tracked = 0
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
        repos = [r.model_dump() for r in self.list_repos()]
        # Schema v4: graphify mirror summary (None when never loaded).
        graph_stats = self._graph_stats()
        return {
            "db_path": str(self._db_path),
            "size_bytes": size_bytes,
            "chunks": int(chunks),
            "files": int(files),
            "files_tracked": int(files_tracked),
            "index_dim": self._dim,
            "models": models,
            "enrichers": enrichers,
            "enrichment_cache_size": int(enrichment_cache),
            "repos": repos,
            "graph": graph_stats,
        }

    def _graph_stats(self) -> dict[str, Any] | None:
        """Aggregate graphify mirror state. Returns None when never loaded."""
        cur = self._conn.cursor()
        # graph_meta is created at v4 but the row only exists after a load.
        # Multi-repo mode stores per-repo-suffixed keys
        # (``last_loaded_at:{repo_id}``); single-repo mode uses the bare key
        # (see ``GraphifyLoader`` in ingest/graphify_loader.py). Accept either
        # so a populated multi-repo mirror is NOT misreported as "no graph" —
        # this mirrors ``retrieval.graph.graph_loaded``. We surface the most
        # recent ``last_loaded_at`` (ISO strings sort chronologically) and
        # aggregate node/edge counts across every repo.
        try:
            row = cur.execute(
                "SELECT value FROM graph_meta WHERE key = ? OR key LIKE ? "
                "ORDER BY value DESC LIMIT 1",
                ("last_loaded_at", "last_loaded_at:%"),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        n_nodes = cur.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
        n_edges = cur.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
        by_conf = {
            r["confidence"]: int(r["n"])
            for r in cur.execute(
                "SELECT confidence, COUNT(*) AS n FROM graph_edges GROUP BY confidence"
            ).fetchall()
        }
        n_communities = cur.execute(
            "SELECT COUNT(DISTINCT community_id) AS n FROM graph_nodes WHERE community_id IS NOT NULL"
        ).fetchone()["n"]
        path_row = cur.execute(
            "SELECT value FROM graph_meta WHERE key = ? OR key LIKE ? LIMIT 1",
            ("graph_path", "graph_path:%"),
        ).fetchone()
        conf_row = cur.execute(
            "SELECT value FROM graph_meta WHERE key = ? OR key LIKE ? LIMIT 1",
            ("confidence_filter", "confidence_filter:%"),
        ).fetchone()
        return {
            "nodes": int(n_nodes),
            "edges_total": int(n_edges),
            "edges_by_confidence": by_conf,
            "communities": int(n_communities),
            "last_loaded_at": row["value"],
            "graph_path": path_row["value"] if path_row else None,
            "confidence_filter": (
                (conf_row["value"] or "").split(",") if conf_row else None
            ),
        }

    def compute_facets(
        self,
        chunk_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Compute facet counts for search results.

        Returns counts grouped by language, kind, and repo. When chunk_ids is
        provided, only counts for those chunks are returned. When None, counts
        for the entire index are returned.

        Args:
            chunk_ids: Optional list of chunk_ids to compute facets for.
                       If None, computes facets for all chunks.

        Returns:
            Dict with keys 'language', 'kind', 'repo', each mapping to
            {value: count} dicts.
        """
        cur = self._conn.cursor()

        if chunk_ids is None:
            # Full index facets
            language_rows = cur.execute(
                "SELECT language, COUNT(*) AS n FROM chunks GROUP BY language"
            ).fetchall()
            kind_rows = cur.execute(
                "SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind"
            ).fetchall()
            repo_rows = cur.execute(
                """SELECT COALESCE(r.normalized_key, '(local)') AS repo, COUNT(*) AS n
                   FROM chunks c
                   LEFT JOIN repos r ON r.id = c.repo_id
                   GROUP BY repo"""
            ).fetchall()
        else:
            if not chunk_ids:
                return {"language": {}, "kind": {}, "repo": {}}

            # Facets for specific chunks - use temp table for efficiency
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _facet_ids (chunk_id TEXT)")
            cur.execute("DELETE FROM _facet_ids")
            cur.executemany(
                "INSERT INTO _facet_ids (chunk_id) VALUES (?)",
                [(cid,) for cid in chunk_ids],
            )

            language_rows = cur.execute(
                """SELECT c.language, COUNT(*) AS n
                   FROM chunks c
                   JOIN _facet_ids f ON f.chunk_id = c.chunk_id
                   GROUP BY c.language"""
            ).fetchall()
            kind_rows = cur.execute(
                """SELECT c.kind, COUNT(*) AS n
                   FROM chunks c
                   JOIN _facet_ids f ON f.chunk_id = c.chunk_id
                   GROUP BY c.kind"""
            ).fetchall()
            repo_rows = cur.execute(
                """SELECT COALESCE(r.normalized_key, '(local)') AS repo, COUNT(*) AS n
                   FROM chunks c
                   JOIN _facet_ids f ON f.chunk_id = c.chunk_id
                   LEFT JOIN repos r ON r.id = c.repo_id
                   GROUP BY repo"""
            ).fetchall()

        return {
            "language": {row["language"]: int(row["n"]) for row in language_rows},
            "kind": {row["kind"]: int(row["n"]) for row in kind_rows},
            "repo": {row["repo"]: int(row["n"]) for row in repo_rows},
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def backup_to(self, dest_path: str | Path) -> None:
        """WIN-178: WAL-safe DB copy via ``sqlite3.Connection.backup``.

        The original RFC suggested ``cp`` for snapshotting the index
        before a candidate evaluation run. That's unsafe when the
        source DB has an active WAL (which stropha always does —
        ``PRAGMA journal_mode=WAL`` is set on open). ``cp`` copies the
        main file without the WAL segment so the resulting database is
        either truncated or inconsistent.

        ``sqlite3.Connection.backup`` is the official online-backup API:
        it takes pages under the WAL writer lock, copies them to the
        destination connection, then commits a clean snapshot — no
        coordination with the writer required, no torn state.

        Args:
            dest_path: where to write the snapshot. Parent directory
                created if missing. Overwrites any existing file at the
                path (single atomic backup, not an incremental).
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # If a stale file exists at ``dest`` (e.g. a corrupted prior
        # backup, or a placeholder the operator dropped there),
        # ``sqlite3.connect`` would later choke with 'file is not a
        # database' when ``backup()`` runs. Drop it first so the new
        # connection materialises a clean empty DB the backup can fill.
        if dest.exists():
            dest.unlink()
        # Open a fresh destination connection — backup() expects a
        # Connection target, not a path. Close it once the copy is done
        # so the file handle is released before we return.
        target = sqlite3.connect(str(dest))
        try:
            self._conn.backup(target)
        finally:
            target.close()
