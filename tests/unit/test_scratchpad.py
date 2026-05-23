"""Unit tests for the writable scratchpad (WIN-229).

Covers:
- Schema v12 ``sessions.scratchpad`` column.
- ``Storage.upsert_session`` + ``get_session`` round-trip.
- ``AppContext.scratchpad`` field default + per-instance isolation.
- MCP tools: scratchpad_read / scratchpad_write / scratchpad_append
  / scratchpad_clear.
- Write-through persistence to the v12 column.
- Lifespan resume restores the prior scratchpad.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.models import SessionRecord
from stropha.storage.sqlite import SCHEMA_VERSION, Storage


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_twelve(self) -> None:
        assert SCHEMA_VERSION >= 12

    def test_sessions_has_scratchpad_column(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v12.db", embedding_dim=4) as s:
            cols = s._conn.execute(
                "PRAGMA table_info(sessions)"
            ).fetchall()
            names = {c["name"] for c in cols}
            assert "scratchpad" in names


# ---------------------------------------------------------------------------
# Storage CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path):
    s = Storage(tmp_path / "s.db", embedding_dim=4)
    s.__enter__()
    yield s
    s.__exit__(None, None, None)


class TestUpsertSessionScratchpad:
    def test_default_empty(self, storage: Storage) -> None:
        rec = storage.upsert_session("s1")
        assert rec.scratchpad == ""

    def test_persist_on_insert(self, storage: Storage) -> None:
        rec = storage.upsert_session("s1", scratchpad="initial notes")
        assert rec.scratchpad == "initial notes"
        # Round-trip via get_session.
        loaded = storage.get_session("s1")
        assert loaded is not None
        assert loaded.scratchpad == "initial notes"

    def test_persist_on_update(self, storage: Storage) -> None:
        storage.upsert_session("s1", scratchpad="first")
        storage.upsert_session("s1", scratchpad="second")
        loaded = storage.get_session("s1")
        assert loaded is not None
        assert loaded.scratchpad == "second"

    def test_none_preserves_existing(self, storage: Storage) -> None:
        """Passing scratchpad=None on update preserves the existing
        value — same pattern as client_id."""
        storage.upsert_session("s1", scratchpad="kept")
        storage.upsert_session("s1", seen_chunk_ids={"x"})  # no scratchpad arg
        loaded = storage.get_session("s1")
        assert loaded is not None
        assert loaded.scratchpad == "kept"

    def test_explicit_empty_string_clears(self, storage: Storage) -> None:
        storage.upsert_session("s1", scratchpad="content")
        storage.upsert_session("s1", scratchpad="")
        loaded = storage.get_session("s1")
        assert loaded is not None
        assert loaded.scratchpad == ""

    def test_list_sessions_returns_scratchpad(
        self, storage: Storage,
    ) -> None:
        storage.upsert_session("s1", scratchpad="note one")
        rows = storage.list_sessions()
        assert any(r.scratchpad == "note one" for r in rows)


# ---------------------------------------------------------------------------
# SessionRecord defaults
# ---------------------------------------------------------------------------


class TestSessionRecordScratchpad:
    def test_default_empty(self) -> None:
        rec = SessionRecord(
            session_id="s",
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        assert rec.scratchpad == ""


# ---------------------------------------------------------------------------
# AppContext field
# ---------------------------------------------------------------------------


class TestAppContextScratchpad:
    def test_default_empty(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        assert ctx.scratchpad == ""

    def test_per_instance_isolation(self) -> None:
        from stropha.server import AppContext

        a = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
            scratchpad="A's note",
        )
        b = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        assert a.scratchpad == "A's note"
        assert b.scratchpad == ""


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


def _make_app(storage: Storage, *, client_id: str | None = None) -> MagicMock:
    app = MagicMock()
    app.storage = storage
    app.session_id = "test-session"
    app.client_id = client_id
    app.seen_chunk_ids = set()
    app.pinned_chunk_ids = set()
    app.scratchpad = ""
    return app


def _ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestScratchpadMCPTools:
    def test_read_default_empty(self, storage: Storage) -> None:
        from stropha.server import scratchpad_read

        app = _make_app(storage)
        result = scratchpad_read(ctx=_ctx(app))
        assert result.content == ""
        assert result.char_count == 0

    def test_write_replaces(self, storage: Storage) -> None:
        from stropha.server import scratchpad_read, scratchpad_write

        app = _make_app(storage)
        scratchpad_write("hello", ctx=_ctx(app))
        scratchpad_write("world", ctx=_ctx(app))
        result = scratchpad_read(ctx=_ctx(app))
        assert result.content == "world"

    def test_append_adds_newline_separator(self, storage: Storage) -> None:
        from stropha.server import scratchpad_append, scratchpad_write

        app = _make_app(storage)
        scratchpad_write("line1", ctx=_ctx(app))
        result = scratchpad_append("line2", ctx=_ctx(app))
        # Existing didn't end with newline → separator inserted.
        assert result.content == "line1\nline2"

    def test_append_respects_existing_newline(self, storage: Storage) -> None:
        from stropha.server import scratchpad_append, scratchpad_write

        app = _make_app(storage)
        scratchpad_write("line1\n", ctx=_ctx(app))
        result = scratchpad_append("line2", ctx=_ctx(app))
        # Existing already ended with newline → no extra inserted.
        assert result.content == "line1\nline2"

    def test_append_empty_input_is_noop(self, storage: Storage) -> None:
        from stropha.server import scratchpad_append, scratchpad_write

        app = _make_app(storage)
        scratchpad_write("base", ctx=_ctx(app))
        result = scratchpad_append("", ctx=_ctx(app))
        assert result.content == "base"

    def test_clear_empties(self, storage: Storage) -> None:
        from stropha.server import scratchpad_clear, scratchpad_write

        app = _make_app(storage)
        scratchpad_write("some content", ctx=_ctx(app))
        result = scratchpad_clear(ctx=_ctx(app))
        assert result.content == ""
        assert result.char_count == 0

    def test_char_count_matches_content_length(
        self, storage: Storage,
    ) -> None:
        from stropha.server import scratchpad_write

        app = _make_app(storage)
        result = scratchpad_write("hello 🚀", ctx=_ctx(app))
        # Char count is Python str length (not byte count).
        assert result.char_count == len("hello 🚀")


# ---------------------------------------------------------------------------
# Write-through persistence
# ---------------------------------------------------------------------------


class TestWriteThrough:
    def test_write_persists_when_client_id_set(
        self, storage: Storage,
    ) -> None:
        from stropha.server import scratchpad_write

        app = _make_app(storage, client_id="agent-A")
        scratchpad_write("durable note", ctx=_ctx(app))
        # Read back via Storage directly.
        loaded = storage.get_session("test-session")
        assert loaded is not None
        assert loaded.scratchpad == "durable note"

    def test_write_skipped_without_client_id(
        self, storage: Storage,
    ) -> None:
        """Without a stable client_id, persistence is no-op (same
        contract as WIN-217 / WIN-218)."""
        from stropha.server import scratchpad_write

        app = _make_app(storage, client_id=None)
        scratchpad_write("note", ctx=_ctx(app))
        # No row written.
        assert storage.get_session("test-session") is None

    def test_clear_persists(self, storage: Storage) -> None:
        from stropha.server import scratchpad_clear, scratchpad_write

        app = _make_app(storage, client_id="agent-A")
        scratchpad_write("seed", ctx=_ctx(app))
        scratchpad_clear(ctx=_ctx(app))
        loaded = storage.get_session("test-session")
        assert loaded is not None
        assert loaded.scratchpad == ""

    def test_append_persists(self, storage: Storage) -> None:
        from stropha.server import scratchpad_append

        app = _make_app(storage, client_id="agent-A")
        app.scratchpad = "line1"
        scratchpad_append("line2", ctx=_ctx(app))
        loaded = storage.get_session("test-session")
        assert loaded is not None
        assert loaded.scratchpad == "line1\nline2"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_scratchpad_read_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "scratchpad_read" in names

    def test_scratchpad_write_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "scratchpad_write" in names

    def test_scratchpad_append_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "scratchpad_append" in names

    def test_scratchpad_clear_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "scratchpad_clear" in names


# ---------------------------------------------------------------------------
# Lifespan resume — integration smoke
# ---------------------------------------------------------------------------


class TestLifespanResume:
    def test_resume_loads_prior_scratchpad(
        self, storage: Storage,
    ) -> None:
        """Simulate the resume path: a prior session exists for the
        same client_id. The lifespan would call list_sessions() and
        rehydrate ``prior_scratchpad`` from row 0."""
        storage.upsert_session(
            "prior-session",
            client_id="agent-X",
            scratchpad="resumed content",
        )
        prior = storage.list_sessions(client_id="agent-X", limit=1)
        assert len(prior) == 1
        assert prior[0].scratchpad == "resumed content"
