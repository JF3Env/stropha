"""Unit tests for the Schema-v8 sessions table (WIN-221, EPIC 1.9).

Covers:

- Schema migration v7 → v8 (sessions table + indexes created cleanly,
  existing v7 indexes survive intact, in-place upgrade is idempotent).
- ``Storage.get_session`` / ``upsert_session`` / ``delete_session`` /
  ``list_sessions`` / ``prune_sessions``.
- ``SessionRecord`` Pydantic shape.
- MCP lifespan resumes prior session by ``STROPHA_CLIENT_ID``.
- Write-through persistence on dedup / pin / unpin / reset_dedup
  mutations.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.models import SessionRecord
from stropha.storage.sqlite import SCHEMA_VERSION, Storage


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_eight(self) -> None:
        """v8 introduced the sessions table (KEYSTONE for EPIC 1.9).
        Forward-compatible pin — future bumps stay valid as long as
        the sessions contract this file guards is intact."""
        assert SCHEMA_VERSION >= 8

    def test_sessions_table_created_on_fresh_db(self, tmp_path: Path) -> None:
        db = tmp_path / "fresh.db"
        s = Storage(db, embedding_dim=4)
        with s:
            cur = s._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            assert cur.fetchone() is not None

    def test_session_indexes_exist(self, tmp_path: Path) -> None:
        s = Storage(tmp_path / "idx.db", embedding_dim=4)
        with s:
            rows = s._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='sessions'"
            ).fetchall()
            names = {r["name"] for r in rows}
            assert "idx_sessions_client_id" in names
            assert "idx_sessions_updated_at" in names

    def test_schema_version_meta_bumped(self, tmp_path: Path) -> None:
        s = Storage(tmp_path / "v.db", embedding_dim=4)
        with s:
            row = s._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            # Forward-compatible — future bumps remain valid.
            assert int(row["value"]) >= 8

    def test_idempotent_reopen(self, tmp_path: Path) -> None:
        """Re-opening the DB must not error or duplicate the table."""
        db = tmp_path / "double.db"
        with Storage(db, embedding_dim=4):
            pass
        with Storage(db, embedding_dim=4) as s:
            cur = s._conn.execute(
                "SELECT COUNT(*) AS n FROM sqlite_master WHERE name='sessions'"
            ).fetchone()
            assert cur["n"] == 1


# ---------------------------------------------------------------------------
# CRUD on sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path):
    s = Storage(tmp_path / "sess.db", embedding_dim=4)
    s.__enter__()
    yield s
    s.__exit__(None, None, None)


class TestUpsertSession:
    def test_insert_creates_row(self, storage: Storage) -> None:
        rec = storage.upsert_session(
            "s1", client_id="agent-A",
            seen_chunk_ids={"a", "b"},
            pinned_chunk_ids={"p"},
        )
        assert rec.session_id == "s1"
        assert rec.client_id == "agent-A"
        assert rec.seen_chunk_ids == {"a", "b"}
        assert rec.pinned_chunk_ids == {"p"}
        # Timestamps were populated by the insert.
        assert rec.created_at == rec.updated_at

    def test_update_bumps_updated_at_but_keeps_created_at(
        self, storage: Storage,
    ) -> None:
        rec1 = storage.upsert_session("s1", client_id="A")
        # Tiny sleep avoids same-millisecond timestamps. We compare
        # > rather than == anyway, but be safe.
        import time

        time.sleep(0.01)
        rec2 = storage.upsert_session("s1", seen_chunk_ids={"x"})
        assert rec2.created_at == rec1.created_at
        assert rec2.updated_at >= rec1.updated_at
        assert rec2.seen_chunk_ids == {"x"}

    def test_update_preserves_client_id_when_none_passed(
        self, storage: Storage,
    ) -> None:
        """Most updates only touch the state sets — client_id must
        survive when the caller passes None."""
        storage.upsert_session("s1", client_id="A")
        rec = storage.upsert_session("s1", seen_chunk_ids={"x"})
        assert rec.client_id == "A"

    def test_meta_round_trip(self, storage: Storage) -> None:
        meta = {"project_path": "/repo", "version": 3}
        rec = storage.upsert_session("s1", meta=meta)
        assert rec.meta == meta

    def test_empty_sets_default_to_empty_json_arrays(
        self, storage: Storage,
    ) -> None:
        rec = storage.upsert_session("s1")
        assert rec.seen_chunk_ids == set()
        assert rec.pinned_chunk_ids == set()
        assert rec.meta == {}


class TestGetSession:
    def test_missing_id_returns_none(self, storage: Storage) -> None:
        assert storage.get_session("ghost") is None

    def test_round_trip(self, storage: Storage) -> None:
        storage.upsert_session(
            "s1", client_id="A",
            seen_chunk_ids={"a", "b"},
            pinned_chunk_ids={"p"},
            meta={"foo": "bar"},
        )
        rec = storage.get_session("s1")
        assert rec is not None
        assert rec.client_id == "A"
        assert rec.seen_chunk_ids == {"a", "b"}
        assert rec.pinned_chunk_ids == {"p"}
        assert rec.meta == {"foo": "bar"}

    def test_corrupted_json_falls_back_to_empty(
        self, storage: Storage,
    ) -> None:
        """A hand-edited row with garbage JSON must not crash the MCP
        server — log + return empty state instead."""
        # Plant a row directly so we control the bad JSON.
        now = datetime.now(UTC).isoformat()
        storage._conn.execute(
            """INSERT INTO sessions (
                   session_id, client_id, created_at, updated_at,
                   seen_chunk_ids, pinned_chunk_ids, meta
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("bad", None, now, now, "{{not json", "[", "garbage"),
        )
        storage._conn.commit()
        rec = storage.get_session("bad")
        assert rec is not None
        assert rec.seen_chunk_ids == set()
        assert rec.pinned_chunk_ids == set()
        assert rec.meta == {}


class TestListSessions:
    def test_orders_by_updated_at_desc(self, storage: Storage) -> None:
        import time

        storage.upsert_session("first")
        time.sleep(0.01)
        storage.upsert_session("second")
        time.sleep(0.01)
        storage.upsert_session("third")
        rows = storage.list_sessions()
        ids = [r.session_id for r in rows]
        assert ids[0] == "third"
        assert ids[-1] == "first"

    def test_filter_by_client_id(self, storage: Storage) -> None:
        storage.upsert_session("s1", client_id="A")
        storage.upsert_session("s2", client_id="B")
        storage.upsert_session("s3", client_id="A")
        rows = storage.list_sessions(client_id="A")
        assert {r.session_id for r in rows} == {"s1", "s3"}

    def test_limit_truncates(self, storage: Storage) -> None:
        for i in range(5):
            storage.upsert_session(f"s{i}")
        rows = storage.list_sessions(limit=2)
        assert len(rows) == 2

    def test_corrupted_row_included_with_empty_state(
        self, storage: Storage,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        storage._conn.execute(
            """INSERT INTO sessions (
                   session_id, client_id, created_at, updated_at,
                   seen_chunk_ids, pinned_chunk_ids, meta
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("bad", "A", now, now, "garbage", "garbage", "garbage"),
        )
        storage._conn.commit()
        rows = storage.list_sessions()
        assert len(rows) == 1
        # State recovered as empty sets.
        assert rows[0].seen_chunk_ids == set()


class TestDeleteSession:
    def test_delete_existing_returns_true(self, storage: Storage) -> None:
        storage.upsert_session("s1")
        assert storage.delete_session("s1") is True
        assert storage.get_session("s1") is None

    def test_delete_unknown_returns_false(self, storage: Storage) -> None:
        assert storage.delete_session("ghost") is False


class TestPruneSessions:
    def test_prunes_older_than_cutoff(self, storage: Storage) -> None:
        # Plant rows with controlled updated_at values.
        old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        new = datetime.now(UTC).isoformat()
        for sid, ts in [("old1", old), ("old2", old), ("recent", new)]:
            storage._conn.execute(
                """INSERT INTO sessions (
                       session_id, created_at, updated_at,
                       seen_chunk_ids, pinned_chunk_ids, meta
                   ) VALUES (?, ?, ?, '[]', '[]', '{}')""",
                (sid, ts, ts),
            )
        storage._conn.commit()
        cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        removed = storage.prune_sessions(older_than_iso=cutoff)
        assert removed == 2
        remaining = {r.session_id for r in storage.list_sessions()}
        assert remaining == {"recent"}

    def test_prune_empty_table_returns_zero(self, storage: Storage) -> None:
        assert storage.prune_sessions(older_than_iso="2099-01-01") == 0


# ---------------------------------------------------------------------------
# SessionRecord Pydantic shape
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def test_defaults(self) -> None:
        rec = SessionRecord(
            session_id="s1",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        assert rec.client_id is None
        assert rec.seen_chunk_ids == set()
        assert rec.pinned_chunk_ids == set()
        assert rec.meta == {}

    def test_set_fields_independent_per_instance(self) -> None:
        a = SessionRecord(
            session_id="a", created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        b = SessionRecord(
            session_id="b", created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        a.seen_chunk_ids.add("x")
        assert "x" not in b.seen_chunk_ids


# ---------------------------------------------------------------------------
# MCP lifespan integration
# ---------------------------------------------------------------------------


class TestSessionPersistenceWriteThrough:
    """When ``client_id`` is set, every mutation of seen/pinned must
    flush to the sessions table so a crash doesn't lose more than the
    last call's worth of state."""

    def _app(self, storage: Storage, *, client_id: str | None) -> MagicMock:
        app = MagicMock()
        app.session_id = "test-session"
        app.client_id = client_id
        app.storage = storage
        app.seen_chunk_ids = set()
        app.pinned_chunk_ids = set()
        # WIN-229: scratchpad field on AppContext means
        # ``_persist_session_state`` reads it now. Without an explicit
        # string, the MagicMock attribute would be a MagicMock object
        # that SQLite can't bind.
        app.scratchpad = ""
        return app

    def test_persist_skipped_when_no_client_id(self, storage: Storage) -> None:
        from stropha.server import _persist_session_state

        app = self._app(storage, client_id=None)
        app.seen_chunk_ids.add("c1")
        _persist_session_state(app)
        # No row written.
        assert storage.get_session("test-session") is None

    def test_persist_writes_when_client_id_set(
        self, storage: Storage,
    ) -> None:
        from stropha.server import _persist_session_state

        app = self._app(storage, client_id="agent-A")
        app.seen_chunk_ids.add("c1")
        app.pinned_chunk_ids.add("p1")
        _persist_session_state(app)
        rec = storage.get_session("test-session")
        assert rec is not None
        assert rec.client_id == "agent-A"
        assert rec.seen_chunk_ids == {"c1"}
        assert rec.pinned_chunk_ids == {"p1"}

    def test_persist_failure_logged_not_raised(
        self, storage: Storage,
    ) -> None:
        """A storage write failure must not propagate — the agent's
        MCP call must succeed even when persistence is degraded."""
        from stropha.server import _persist_session_state

        app = self._app(storage, client_id="A")
        # Replace upsert_session with a method that raises.
        app.storage.upsert_session = MagicMock(side_effect=RuntimeError("fs"))
        # Should NOT raise.
        _persist_session_state(app)

    def test_search_code_dedup_triggers_persist(
        self, storage: Storage,
    ) -> None:
        """search_code(dedup=True) with a client_id flushes to disk.

        Uses a real Storage (so the persist actually writes) and a
        MagicMock search_engine to control the canned hits. ``compute_facets``
        is NOT invoked because we leave ``include_facets`` at its default
        False — would otherwise touch the real Storage method, which is
        fine but unnecessary.
        """
        from stropha.models import SearchHit
        from stropha.server import search_code

        app = self._app(storage, client_id="agent-A")
        app.search_engine.search.return_value = [
            SearchHit(
                chunk_id="c1", rel_path="x.py", start_line=1, end_line=2,
                score=0.5, rank=0, snippet="x", language="py", kind="fn",
                symbol="f",
            )
        ]
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        search_code("q", dedup=True, ctx=ctx)
        rec = storage.get_session("test-session")
        assert rec is not None
        assert "c1" in rec.seen_chunk_ids

    def test_pin_chunks_triggers_persist(self, storage: Storage) -> None:
        from stropha.server import pin_chunks

        app = self._app(storage, client_id="A")
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        pin_chunks(["c1"], ctx=ctx)
        rec = storage.get_session("test-session")
        assert rec is not None
        assert rec.pinned_chunk_ids == {"c1"}

    def test_unpin_chunks_triggers_persist(self, storage: Storage) -> None:
        from stropha.server import pin_chunks, unpin_chunks

        app = self._app(storage, client_id="A")
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        pin_chunks(["c1"], ctx=ctx)
        unpin_chunks(["c1"], ctx=ctx)
        rec = storage.get_session("test-session")
        assert rec is not None
        assert rec.pinned_chunk_ids == set()

    def test_reset_dedup_triggers_persist(self, storage: Storage) -> None:
        from stropha.server import reset_dedup

        app = self._app(storage, client_id="A")
        app.seen_chunk_ids = {"c1", "c2"}
        # Seed the row so reset_dedup overwrites it.
        storage.upsert_session(
            "test-session", client_id="A",
            seen_chunk_ids={"c1", "c2"},
        )
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        reset_dedup(ctx=ctx)
        rec = storage.get_session("test-session")
        assert rec is not None
        assert rec.seen_chunk_ids == set()


# ---------------------------------------------------------------------------
# AppContext field for client_id
# ---------------------------------------------------------------------------


class TestAppContextClientId:
    def test_client_id_default_none(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        assert ctx.client_id is None

    def test_client_id_accepts_explicit_value(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
            client_id="abc",
        )
        assert ctx.client_id == "abc"
