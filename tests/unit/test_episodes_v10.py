"""Unit tests for Schema v10 episodic memory (WIN-225).

Covers:
- Schema migration v9 → v10 (episodes table + indexes).
- Storage.record_episode / recall_episodes /
  episode_co_occurrence / forget_episode.
- MCP record_episode / recall_episodes / co_occurrence tools.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.models import EpisodeRecord
from stropha.storage.sqlite import SCHEMA_VERSION, Storage

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_ten(self) -> None:
        """v10 introduced the episodes table. Forward-compatible pin
        so future schema bumps don't churn this guard."""
        assert SCHEMA_VERSION >= 10

    def test_episodes_table_created(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v10.db", embedding_dim=4) as s:
            row = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name='episodes'"""
            ).fetchone()
            assert row is not None

    def test_episodes_indexes_exist(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
            rows = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='index' AND tbl_name='episodes'"""
            ).fetchall()
            names = {r["name"] for r in rows}
            assert "idx_episodes_repo_created" in names
            assert "idx_episodes_session" in names


# ---------------------------------------------------------------------------
# Storage CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "ep.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="github.com/test/repo",
            remote_url=None,
            root_path=Path("/tmp/test"),
            default_branch="main",
            head_commit="abc",
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


class TestRecordEpisode:
    def test_basic_insert(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(
            repo_id=repo_id,
            title="Refactored FSRS calculator",
            summary="Added decay-rate guard",
            touched_files=["src/fsrs/calc.py", "tests/test_fsrs.py"],
            touched_symbols=["FsrsCalculator.update"],
            session_id="sess-1",
        )
        assert rec.id is not None
        assert rec.repo_id == repo_id
        assert rec.title == "Refactored FSRS calculator"
        assert rec.session_id == "sess-1"
        assert rec.touched_files == ["src/fsrs/calc.py", "tests/test_fsrs.py"]
        assert rec.touched_symbols == ["FsrsCalculator.update"]

    def test_touched_files_deduplicated_and_sorted(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(
            repo_id=repo_id, title="t",
            touched_files=["b.py", "a.py", "a.py", "b.py", "c.py"],
        )
        assert rec.touched_files == ["a.py", "b.py", "c.py"]

    def test_empty_touched_fields_default(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(repo_id=repo_id, title="t")
        assert rec.touched_files == []
        assert rec.touched_symbols == []
        assert rec.summary == ""

    def test_append_only_creates_distinct_rows(
        self, storage_with_repo,
    ) -> None:
        """Re-recording an episode with the same title does NOT
        upsert — episodes are timeline events, duplicates are
        legitimate (the agent worked on the same area twice)."""
        s, repo_id = storage_with_repo
        rec1 = s.record_episode(repo_id=repo_id, title="Same topic")
        rec2 = s.record_episode(repo_id=repo_id, title="Same topic")
        assert rec1.id != rec2.id


class TestRecallEpisodes:
    def test_unknown_repo_returns_empty(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        assert s.recall_episodes(repo_id=99999) == []

    def test_default_returns_most_recent_first(
        self, storage_with_repo,
    ) -> None:
        import time

        s, repo_id = storage_with_repo
        s.record_episode(repo_id=repo_id, title="first")
        time.sleep(0.01)
        s.record_episode(repo_id=repo_id, title="second")
        time.sleep(0.01)
        s.record_episode(repo_id=repo_id, title="third")
        rows = s.recall_episodes(repo_id=repo_id)
        titles = [r.title for r in rows]
        assert titles == ["third", "second", "first"]

    def test_query_filters_title(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(repo_id=repo_id, title="Refactored FSRS calc")
        s.record_episode(repo_id=repo_id, title="Added unit test")
        rows = s.recall_episodes(repo_id=repo_id, query="FSRS")
        assert len(rows) == 1
        assert "FSRS" in rows[0].title

    def test_query_filters_summary(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(
            repo_id=repo_id, title="A",
            summary="Fixed the auth bug in middleware",
        )
        s.record_episode(repo_id=repo_id, title="B", summary="Unrelated")
        rows = s.recall_episodes(repo_id=repo_id, query="auth")
        assert len(rows) == 1
        assert rows[0].title == "A"

    def test_query_case_insensitive(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(repo_id=repo_id, title="FSRS scheduler")
        rows = s.recall_episodes(repo_id=repo_id, query="fsrs")
        assert len(rows) == 1

    def test_since_iso_filter(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        new = datetime.now(UTC).isoformat()
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "old", "", "[]", "[]", old, old),
        )
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "new", "", "[]", "[]", new, new),
        )
        s._conn.commit()
        cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        rows = s.recall_episodes(repo_id=repo_id, since_iso=cutoff)
        assert {r.title for r in rows} == {"new"}

    def test_limit_clamped_to_200(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        for i in range(5):
            s.record_episode(repo_id=repo_id, title=f"t{i}")
        rows = s.recall_episodes(repo_id=repo_id, limit=999_999)
        assert len(rows) == 5  # all returned, no crash on big limit

    def test_corrupted_json_falls_back_to_empty_lists(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        now = datetime.now(UTC).isoformat()
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, touched_files, touched_symbols,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (repo_id, "bad", "garbage", "{{not json", now, now),
        )
        s._conn.commit()
        rows = s.recall_episodes(repo_id=repo_id)
        assert len(rows) == 1
        assert rows[0].touched_files == []
        assert rows[0].touched_symbols == []


class TestCoOccurrence:
    def test_files_co_touched_with_key(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(
            repo_id=repo_id, title="t1",
            touched_files=["src/fsrs.py", "tests/test_fsrs.py", "docs/fsrs.md"],
        )
        s.record_episode(
            repo_id=repo_id, title="t2",
            touched_files=["src/fsrs.py", "tests/test_fsrs.py"],
        )
        s.record_episode(
            repo_id=repo_id, title="t3",
            touched_files=["src/fsrs.py", "migrations/v3.sql"],
        )
        result = s.episode_co_occurrence(repo_id=repo_id, key="src/fsrs.py")
        # tests/test_fsrs.py co-touched 2x, others 1x.
        assert result["files"].get("tests/test_fsrs.py") == 2
        assert result["files"].get("migrations/v3.sql") == 1
        # key itself NOT included.
        assert "src/fsrs.py" not in result["files"]

    def test_symbols_co_touched_with_key(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(
            repo_id=repo_id, title="t",
            touched_symbols=["FsrsCalc.update", "Migration.apply"],
        )
        s.record_episode(
            repo_id=repo_id, title="t",
            touched_symbols=["FsrsCalc.update", "Migration.apply"],
        )
        result = s.episode_co_occurrence(
            repo_id=repo_id, key="FsrsCalc.update",
        )
        assert result["symbols"].get("Migration.apply") == 2

    def test_key_not_found_returns_empty(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.record_episode(repo_id=repo_id, title="t", touched_files=["a.py"])
        result = s.episode_co_occurrence(
            repo_id=repo_id, key="never-touched.py",
        )
        assert result == {"files": {}, "symbols": {}}

    def test_empty_key_returns_empty_dicts(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        assert s.episode_co_occurrence(repo_id=repo_id, key="") == {
            "files": {}, "symbols": {},
        }

    def test_in_files_false_skips_file_search(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        # An episode touching the key in files but not symbols.
        s.record_episode(
            repo_id=repo_id, title="t",
            touched_files=["x.py"], touched_symbols=["unrelated"],
        )
        result = s.episode_co_occurrence(
            repo_id=repo_id, key="x.py", in_files=False, in_symbols=True,
        )
        # Disabled the files lookup → no co-occurrence found via the
        # symbols lookup either (key not in symbols of this episode).
        assert result == {"files": {}, "symbols": {}}

    def test_skip_episodes_where_key_appears_only_in_json_substring(
        self, storage_with_repo,
    ) -> None:
        """LIKE-based scan can hit JSON substring false positives
        (e.g. key "py" matches "x.py"). The post-filter that confirms
        ``key in files or key in symbols`` rejects those — verified
        by planting an episode that LIKE would hit but the literal
        check rejects."""
        s, repo_id = storage_with_repo
        s.record_episode(
            repo_id=repo_id, title="t",
            touched_files=["x.py", "y.py"], touched_symbols=[],
        )
        # "py" appears as a substring of every filename. LIKE would
        # match; but the literal check `"py" in files` is False
        # because none of the files equal "py".
        result = s.episode_co_occurrence(repo_id=repo_id, key="py")
        assert result == {"files": {}, "symbols": {}}

    def test_results_sorted_descending(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        for _ in range(3):
            s.record_episode(
                repo_id=repo_id, title="t",
                touched_files=["k.py", "common.py"],
            )
        s.record_episode(
            repo_id=repo_id, title="t",
            touched_files=["k.py", "rare.py"],
        )
        result = s.episode_co_occurrence(repo_id=repo_id, key="k.py")
        # Common.py wins (3x); rare.py at 1x.
        items = list(result["files"].items())
        assert items[0][0] == "common.py"
        assert items[0][1] == 3


class TestForgetEpisode:
    def test_existing_returns_true(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(repo_id=repo_id, title="t")
        assert s.forget_episode(episode_id=rec.id or 0) is True
        assert s.recall_episodes(repo_id=repo_id) == []

    def test_unknown_returns_false(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        assert s.forget_episode(episode_id=999_999) is False


# ---------------------------------------------------------------------------
# EpisodeRecord model
# ---------------------------------------------------------------------------


class TestEpisodeRecord:
    def test_defaults(self) -> None:
        rec = EpisodeRecord(
            repo_id=1, title="t",
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        assert rec.id is None
        assert rec.session_id is None
        assert rec.summary == ""
        assert rec.touched_files == []
        assert rec.touched_symbols == []


# ---------------------------------------------------------------------------
# MCP tools — record_episode / recall_episodes / co_occurrence
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo_path(tmp_path: Path) -> Path:
    p = tmp_path / "ep-repo"
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


def _make_app(storage: Storage, repo_path: Path) -> MagicMock:
    app = MagicMock()
    app.storage = storage
    app.session_id = "test-session"
    app.config = MagicMock()
    app.config.target_repo = repo_path
    return app


def _ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestMCPRecordEpisode:
    def test_writes_when_target_repo_indexed(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import record_episode

        s, _repo_id, _ = storage_with_target
        app = _make_app(s, fake_repo_path)
        result = record_episode(
            "Refactored FSRS",
            summary="cleaned up the calc",
            touched_files=["src/fsrs.py"],
            touched_symbols=["FsrsCalc.update"],
            ctx=_ctx(app),
        )
        assert result.ok is True
        assert result.entry is not None
        assert result.entry.title == "Refactored FSRS"
        assert "src/fsrs.py" in result.entry.touched_files

    def test_session_id_captured(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import record_episode

        s, _, _ = storage_with_target
        app = _make_app(s, fake_repo_path)
        result = record_episode("t", ctx=_ctx(app))
        assert result.entry is not None
        assert result.entry.session_id == "test-session"

    def test_unindexed_repo_returns_error(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import record_episode

        s = Storage(tmp_path / "fresh.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app(s, fake_repo_path)
            result = record_episode("t", ctx=_ctx(app))
            assert result.ok is False
            assert "no indexed repo" in (result.error or "")
        finally:
            s.__exit__(None, None, None)


class TestMCPRecallEpisodes:
    def test_returns_episodes_for_target(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall_episodes, record_episode

        s, _, _ = storage_with_target
        app = _make_app(s, fake_repo_path)
        record_episode("first", ctx=_ctx(app))
        record_episode("second", ctx=_ctx(app))
        result = recall_episodes(ctx=_ctx(app))
        titles = [e.title for e in result.entries]
        assert set(titles) == {"first", "second"}

    def test_query_filters(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall_episodes, record_episode

        s, _, _ = storage_with_target
        app = _make_app(s, fake_repo_path)
        record_episode("FSRS work", ctx=_ctx(app))
        record_episode("Auth bugfix", ctx=_ctx(app))
        result = recall_episodes(query="FSRS", ctx=_ctx(app))
        assert len(result.entries) == 1
        assert "FSRS" in result.entries[0].title

    def test_no_repo_returns_empty(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall_episodes

        s = Storage(tmp_path / "empty.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app(s, fake_repo_path)
            result = recall_episodes(ctx=_ctx(app))
            assert result.entries == []
            assert result.repo is None
        finally:
            s.__exit__(None, None, None)


class TestMCPCoOccurrence:
    def test_returns_co_touched_files(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import co_occurrence, record_episode

        s, _, _ = storage_with_target
        app = _make_app(s, fake_repo_path)
        record_episode(
            "ep1", touched_files=["src/fsrs.py", "test_fsrs.py"],
            ctx=_ctx(app),
        )
        record_episode(
            "ep2", touched_files=["src/fsrs.py", "test_fsrs.py"],
            ctx=_ctx(app),
        )
        result = co_occurrence("src/fsrs.py", ctx=_ctx(app))
        assert result.key == "src/fsrs.py"
        assert result.files.get("test_fsrs.py") == 2

    def test_no_repo_returns_empty(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import co_occurrence

        s = Storage(tmp_path / "empty.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app(s, fake_repo_path)
            result = co_occurrence("anything", ctx=_ctx(app))
            assert result.files == {}
            assert result.symbols == {}
        finally:
            s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_record_episode_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "record_episode" in names

    def test_recall_episodes_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "recall_episodes" in names

    def test_co_occurrence_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "co_occurrence" in names
