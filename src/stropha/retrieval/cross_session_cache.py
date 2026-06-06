"""Cross-session semantic cache (WIN-230, EPIC 1.9 #22).

The existing :class:`stropha.retrieval.cache.SemanticCache` is an
in-memory LRU keyed by a quantized embedding fingerprint. Within a
process it's useful, but every session restart wipes the table —
the agent re-pays for queries other sessions have already answered.

WIN-230 adds a PERSISTENT layer backed by the Schema v13
``cross_session_cache`` table. Lookups do brute-force cosine
similarity over stored embeddings (same pattern as the graph-vec
retrieval stream) and surface cached results when similarity
exceeds a threshold.

Why a separate module
---------------------

The in-memory cache stays as the fast path — sub-microsecond lookup
for repeat-identical queries within a session. The cross-session
cache is the slower but durable layer: tens of milliseconds per
lookup at ~50K entries, but it carries findings across process
boundaries. The two can compose (in-memory L1, persistent L2) in
a future story; this ticket ships the L2 primitives only.

What we don't do
----------------

- No automatic invalidation on commit. WIN-228's
  ``provenance_commit`` + ``check_staleness`` are the staleness
  signals; a future story can hook them up to invalidate cached
  results that touch changed files. For the MVP the operator
  prunes manually (or via the TTL on ``expires_at``).
- No automatic integration into ``SearchEngine.search``. The
  module exposes the lookup / store / prune primitives; a future
  ticket can decide the policy (when to read from L2, when to
  write through).
- No quantization. The graph-vec module proved that brute-force
  float32 cosine is fast enough for the corpora we target;
  quantization would only matter past ~100K entries.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


# Default similarity threshold — high enough that cached hits are
# real matches, low enough that paraphrases hit the cache. Tuned by
# intuition; a future story can build a golden eval to refine.
DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass(frozen=True)
class CacheEntry:
    """One row of the cross_session_cache table.

    ``similarity`` is set by :func:`find_similar` for the returned
    matches — None when read via :func:`get_by_id`. ``results`` is
    the raw JSON-decoded list of result dicts; callers can
    deserialise to SearchHit if needed.
    """

    id: int
    repo_id: int
    query_text: str
    embedding_model: str
    embedding_dim: int
    results: list[dict[str, Any]]
    session_id: str | None
    created_at: str
    last_hit_at: str | None
    hit_count: int
    expires_at: str | None = None
    similarity: float | None = None
    """Cosine similarity vs the query at lookup time. None when the
    entry was fetched outside a similarity search."""


def _pack_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into the blob format used by graph_vec —
    a flat ``struct.pack`` of float32 values. Endianness is little
    (the graph_vec module's convention)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Same magnitude as graph_vec's
    inner cosine (built-in float math; numpy would be faster but adds
    a dep)."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def store(
    storage: Any,
    *,
    repo_id: int,
    query_text: str,
    query_embedding: list[float],
    embedding_model: str,
    embedding_dim: int,
    results: list[dict[str, Any]],
    session_id: str | None = None,
    expires_at: str | None = None,
) -> int:
    """Persist one (query, results) tuple. Returns the row id.

    ``results`` is a list of plain dicts (the caller is responsible
    for serialising SearchHit / etc. before passing). Storing dicts
    not pydantic models keeps the table schema-stable across model
    refactors.
    """
    blob = _pack_embedding(query_embedding)
    now = _now_utc_iso()
    cur = storage._conn.execute(
        """INSERT INTO cross_session_cache (
               repo_id, query_text, query_embedding, embedding_model,
               embedding_dim, results_json, session_id, created_at,
               hit_count, expires_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            repo_id, query_text, blob, embedding_model,
            embedding_dim, json.dumps(results), session_id, now,
            0, expires_at,
        ),
    )
    storage._conn.commit()
    return int(cur.lastrowid or 0)


def find_similar(
    storage: Any,
    *,
    repo_id: int,
    query_embedding: list[float],
    embedding_model: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    limit: int = 5,
    include_expired: bool = False,
) -> list[CacheEntry]:
    """Brute-force cosine over the repo's cached entries.

    Filters to entries matching ``embedding_model`` (cross-embedder
    cosine is meaningless) and within their TTL by default. Sorts
    matches above ``similarity_threshold`` descending; truncates
    to ``limit``.

    Side-effect: every returned match has its ``hit_count`` bumped
    and ``last_hit_at`` updated so the operator can see which
    cached queries are actually useful.
    """
    clauses = [
        "repo_id = ?",
        "embedding_model = ?",
    ]
    params: list[Any] = [repo_id, embedding_model]
    if not include_expired:
        clauses.append("(expires_at IS NULL OR expires_at > ?)")
        params.append(_now_utc_iso())
    where = " AND ".join(clauses)
    rows = storage._conn.execute(
        f"""SELECT id, repo_id, query_text, query_embedding,
                   embedding_model, embedding_dim, results_json,
                   session_id, created_at, last_hit_at,
                   hit_count, expires_at
           FROM cross_session_cache
           WHERE {where}""",
        tuple(params),
    ).fetchall()
    scored: list[tuple[float, CacheEntry]] = []
    for row in rows:
        try:
            stored_vec = _unpack_embedding(
                row["query_embedding"], int(row["embedding_dim"])
            )
            results = json.loads(row["results_json"])
        except (json.JSONDecodeError, struct.error, ValueError, TypeError):
            log.warning(
                "cross_session_cache.corrupted_row",
                row_id=row["id"],
            )
            continue
        sim = _cosine(query_embedding, stored_vec)
        if sim < similarity_threshold:
            continue
        entry = CacheEntry(
            id=int(row["id"]),
            repo_id=int(row["repo_id"]),
            query_text=row["query_text"],
            embedding_model=row["embedding_model"],
            embedding_dim=int(row["embedding_dim"]),
            results=results,
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_hit_at=row["last_hit_at"],
            hit_count=int(row["hit_count"] or 0),
            expires_at=row["expires_at"],
            similarity=sim,
        )
        scored.append((sim, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [e for _sim, e in scored[: max(1, limit)]]
    # Bump hit counters for the matches we're about to return.
    if top:
        now = _now_utc_iso()
        storage._conn.executemany(
            """UPDATE cross_session_cache
                  SET hit_count = hit_count + 1, last_hit_at = ?
                WHERE id = ?""",
            [(now, e.id) for e in top],
        )
        storage._conn.commit()
    return top


def prune(
    storage: Any,
    *,
    repo_id: int | None = None,
    expired_only: bool = False,
    older_than_iso: str | None = None,
) -> int:
    """Bulk-delete cache entries. Same defensive contract as
    ``Storage.prune_project_memory``: no-arg call is a NO-OP.

    ``expired_only=True`` wins over ``older_than_iso`` when both are
    set.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if repo_id is not None:
        clauses.append("repo_id = ?")
        params.append(repo_id)
    if expired_only:
        clauses.append("(expires_at IS NOT NULL AND expires_at <= ?)")
        params.append(_now_utc_iso())
    elif older_than_iso is not None:
        clauses.append("created_at < ?")
        params.append(older_than_iso)
    if not clauses:
        return 0
    where = " AND ".join(clauses)
    cur = storage._conn.execute(
        f"DELETE FROM cross_session_cache WHERE {where}",
        tuple(params),
    )
    storage._conn.commit()
    return int(cur.rowcount or 0)


def stats(storage: Any, *, repo_id: int | None = None) -> dict[str, int]:
    """Return basic counters for the cache.

    ``total_entries``, ``total_hits``, ``distinct_models``. Used by
    the MCP tool to surface "is the cache earning its keep?".
    """
    clauses = "repo_id = ?" if repo_id is not None else "1=1"
    params: tuple[Any, ...] = (repo_id,) if repo_id is not None else ()
    # ``SELECT COUNT(*) ...`` always returns one row in SQLite (even
    # for an empty table — the count is 0, not "no row"). No defensive
    # ``if rows is None`` branch is needed.
    rows = storage._conn.execute(
        f"""SELECT COUNT(*) AS n,
                   COALESCE(SUM(hit_count), 0) AS hits,
                   COUNT(DISTINCT embedding_model) AS models
           FROM cross_session_cache WHERE {clauses}""",
        params,
    ).fetchone()
    return {
        "total_entries": int(rows["n"] or 0),
        "total_hits": int(rows["hits"] or 0),
        "distinct_models": int(rows["models"] or 0),
    }


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "CacheEntry",
    "find_similar",
    "prune",
    "stats",
    "store",
]
