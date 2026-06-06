"""Unit tests for the cross-session semantic cache (WIN-230).

Covers:
- Schema v13 cross_session_cache table.
- ``store`` / ``find_similar`` / ``prune`` / ``stats`` library
  helpers.
- MCP ``cache_stats`` / ``cache_prune`` tools.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.retrieval.cross_session_cache import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CacheEntry,
    find_similar,
    prune,
    stats,
    store,
)
from stropha.storage.sqlite import SCHEMA_VERSION, Storage

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_thirteen(self) -> None:
        assert SCHEMA_VERSION >= 13

    def test_cross_session_cache_table_exists(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v13.db", embedding_dim=4) as s:
            row = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name='cross_session_cache'"""
            ).fetchone()
            assert row is not None

    def test_indexes_exist(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v13.db", embedding_dim=4) as s:
            rows = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='index' AND tbl_name='cross_session_cache'"""
            ).fetchall()
            names = {r["name"] for r in rows}
            for n in (
                "idx_csc_repo", "idx_csc_model", "idx_csc_expires",
            ):
                assert n in names


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "csc.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="k", remote_url=None,
            root_path=tmp_path, default_branch=None,
            head_commit=None,
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestStore:
    def test_basic_insert(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        row_id = store(
            s, repo_id=repo_id,
            query_text="how does mastery decay",
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            embedding_model="mxbai",
            embedding_dim=4,
            results=[{"chunk_id": "c1", "score": 0.9}],
            session_id="sess-1",
        )
        assert row_id > 0

    def test_expires_at_persisted(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        row_id = store(
            s, repo_id=repo_id,
            query_text="q",
            query_embedding=[0.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[],
            expires_at=future,
        )
        row = s._conn.execute(
            "SELECT expires_at FROM cross_session_cache WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert row["expires_at"] == future


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    def test_high_similarity_match_returned(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        # Stored embedding [1,0,0,0]; query [1,0,0,0] → similarity 1.0.
        store(
            s, repo_id=repo_id, query_text="q1",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[{"chunk_id": "c"}],
        )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
        )
        assert len(matches) == 1
        assert matches[0].similarity == pytest.approx(1.0)
        assert matches[0].query_text == "q1"

    def test_low_similarity_filtered(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="orthogonal",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[],
        )
        # Orthogonal query → cosine 0.0; below default threshold.
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[0.0, 1.0, 0.0, 0.0],
            embedding_model="m",
        )
        assert matches == []

    def test_model_filter(self, storage_with_repo) -> None:
        """Lookup filters to entries with the same embedding_model —
        cross-embedder cosine is meaningless."""
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="q1",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m1", embedding_dim=4,
            results=[],
        )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m2",  # different model
        )
        assert matches == []

    def test_expired_entry_skipped(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[], expires_at=past,
        )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
        )
        assert matches == []

    def test_include_expired_true(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[], expires_at=past,
        )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
            include_expired=True,
        )
        assert len(matches) == 1

    def test_hit_count_bumps(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        row_id = store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4,
            results=[],
        )
        find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
        )
        find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
        )
        row = s._conn.execute(
            "SELECT hit_count FROM cross_session_cache WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert row["hit_count"] == 2

    def test_sorted_by_similarity_desc(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="closer",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        store(
            s, repo_id=repo_id, query_text="farther",
            query_embedding=[0.95, 0.1, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
            similarity_threshold=0.5,
        )
        # closer first, then farther.
        assert matches[0].query_text == "closer"
        assert matches[1].query_text == "farther"

    def test_threshold_parameter_respected(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            embedding_model="m", embedding_dim=4, results=[],
        )
        # Similarity to [1,0,0,0] is 0.5; threshold 0.9 → no match.
        no_match = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
            similarity_threshold=0.9,
        )
        assert no_match == []
        # Lower the threshold → match.
        matched = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
            similarity_threshold=0.4,
        )
        assert len(matched) == 1

    def test_corrupted_row_skipped(self, storage_with_repo) -> None:
        """A row with malformed embedding/results JSON is logged and
        skipped, not raised."""
        s, repo_id = storage_with_repo
        # Plant a row with garbage values.
        now = datetime.now(UTC).isoformat()
        s._conn.execute(
            """INSERT INTO cross_session_cache (
                   repo_id, query_text, query_embedding, embedding_model,
                   embedding_dim, results_json, created_at, hit_count
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "bad", b"\x00", "m", 4, "{{not json", now, 0),
        )
        s._conn.commit()
        # No raise; empty result.
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
        )
        assert matches == []

    def test_limit_caps_returned(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        for i in range(10):
            store(
                s, repo_id=repo_id, query_text=f"q{i}",
                query_embedding=[1.0, 0.0, 0.0, 0.0],
                embedding_model="m", embedding_dim=4, results=[],
            )
        matches = find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m",
            limit=3,
        )
        assert len(matches) == 3


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_no_args_is_noop(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        assert prune(s) == 0
        # Row preserved.
        row = s._conn.execute(
            "SELECT COUNT(*) AS n FROM cross_session_cache"
        ).fetchone()
        assert row["n"] == 1

    def test_prune_by_repo(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        for i in range(3):
            store(
                s, repo_id=repo_id, query_text=f"q{i}",
                query_embedding=[1.0, 0.0, 0.0, 0.0],
                embedding_model="m", embedding_dim=4, results=[],
            )
        removed = prune(s, repo_id=repo_id)
        assert removed == 3

    def test_expired_only(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        store(
            s, repo_id=repo_id, query_text="stale",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
            expires_at=past,
        )
        store(
            s, repo_id=repo_id, query_text="live",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        removed = prune(s, repo_id=repo_id, expired_only=True)
        assert removed == 1
        survivors = s._conn.execute(
            "SELECT query_text FROM cross_session_cache"
        ).fetchall()
        assert {r["query_text"] for r in survivors} == {"live"}

    def test_older_than_iso(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        # Plant an old row directly.
        old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        s._conn.execute(
            """INSERT INTO cross_session_cache (
                   repo_id, query_text, query_embedding, embedding_model,
                   embedding_dim, results_json, created_at, hit_count
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "old", b"\x00", "m", 4, "[]", old, 0),
        )
        s._conn.commit()
        store(
            s, repo_id=repo_id, query_text="new",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        removed = prune(
            s, repo_id=repo_id, older_than_iso=cutoff,
        )
        assert removed == 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        counters = stats(s)
        assert counters == {
            "total_entries": 0, "total_hits": 0, "distinct_models": 0,
        }

    def test_counts_entries_hits_models(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        store(
            s, repo_id=repo_id, query_text="q1",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m1", embedding_dim=4, results=[],
        )
        store(
            s, repo_id=repo_id, query_text="q2",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m2", embedding_dim=4, results=[],
        )
        find_similar(
            s, repo_id=repo_id,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m1",
        )
        counters = stats(s, repo_id=repo_id)
        assert counters["total_entries"] == 2
        assert counters["total_hits"] == 1
        assert counters["distinct_models"] == 2


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo_path(tmp_path: Path) -> Path:
    p = tmp_path / "r"
    p.mkdir()
    return p


@pytest.fixture
def storage_with_target(tmp_path: Path, fake_repo_path: Path):
    from stropha.ingest.git_meta import detect

    identity = detect(fake_repo_path)
    s = Storage(tmp_path / "mcp.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(identity)
    yield s, repo_id, identity
    s.__exit__(None, None, None)


def _ctx(storage, repo_path):
    ctx = MagicMock()
    app = MagicMock()
    app.storage = storage
    app.session_id = "sess"
    app.config = MagicMock()
    app.config.target_repo = repo_path
    ctx.request_context.lifespan_context = app
    return ctx


class TestMCPCacheStats:
    def test_empty_returns_zeros(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_stats

        s, _, _ = storage_with_target
        result = cache_stats(ctx=_ctx(s, fake_repo_path))
        assert result.total_entries == 0
        assert result.total_hits == 0
        assert result.distinct_models == 0

    def test_returns_counters(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_stats

        s, repo_id, _ = storage_with_target
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        result = cache_stats(ctx=_ctx(s, fake_repo_path))
        assert result.total_entries == 1

    def test_unindexed_repo_returns_zeros(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_stats

        s = Storage(tmp_path / "fresh.db", embedding_dim=4)
        s.__enter__()
        try:
            result = cache_stats(ctx=_ctx(s, fake_repo_path))
            assert result.total_entries == 0
        finally:
            s.__exit__(None, None, None)


class TestMCPCachePrune:
    def test_no_filters_returns_zero(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_prune

        s, repo_id, _ = storage_with_target
        store(
            s, repo_id=repo_id, query_text="q",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        result = cache_prune(ctx=_ctx(s, fake_repo_path))
        # The MCP layer always passes repo_id, so prune deletes for
        # this repo by default (matches the user expectation that
        # cache_prune scopes to target_repo).
        assert result.removed == 1

    def test_expired_only(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_prune

        s, repo_id, _ = storage_with_target
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        store(
            s, repo_id=repo_id, query_text="stale",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
            expires_at=past,
        )
        store(
            s, repo_id=repo_id, query_text="live",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            embedding_model="m", embedding_dim=4, results=[],
        )
        result = cache_prune(ctx=_ctx(s, fake_repo_path), expired_only=True)
        assert result.removed == 1

    def test_unindexed_repo_returns_zero(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import cache_prune

        s = Storage(tmp_path / "fresh.db", embedding_dim=4)
        s.__enter__()
        try:
            result = cache_prune(ctx=_ctx(s, fake_repo_path))
            assert result.removed == 0
        finally:
            s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_cache_stats_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "cache_stats" in names

    def test_cache_prune_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "cache_prune" in names


# ---------------------------------------------------------------------------
# CacheEntry + constants
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_default_threshold_constant(self) -> None:
        assert pytest.approx(0.92) == DEFAULT_SIMILARITY_THRESHOLD

    def test_cache_entry_immutable(self) -> None:
        entry = CacheEntry(
            id=1, repo_id=1, query_text="q",
            embedding_model="m", embedding_dim=4, results=[],
            session_id=None, created_at="x", last_hit_at=None,
            hit_count=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.id = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


class TestCosineDefensivePaths:
    def test_different_length_returns_zero(self) -> None:
        from stropha.retrieval.cross_session_cache import _cosine

        assert _cosine([1.0, 0.0], [1.0]) == 0.0

    def test_zero_norm_returns_zero(self) -> None:
        from stropha.retrieval.cross_session_cache import _cosine

        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
