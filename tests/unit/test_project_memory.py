"""Unit tests for project_memory (Schema v9, WIN-222).

Per-repo long-term notes survive sessions — the agent curates
decisions, constraints, and anchor facts without re-explaining them
every time. Backed by the ``project_memory`` table; exposed via the
MCP ``remember`` / ``recall`` tools landing in WIN-223.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.models import ProjectMemoryRecord
from stropha.storage.sqlite import SCHEMA_VERSION, Storage


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_nine(self) -> None:
        """v9 added project_memory; future bumps preserve the
        contract this file guards."""
        assert SCHEMA_VERSION >= 9

    def test_project_memory_table_created(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v9.db", embedding_dim=4) as s:
            row = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name='project_memory'"""
            ).fetchone()
            assert row is not None

    def test_project_memory_indexes_exist(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
            rows = s._conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='index' AND tbl_name='project_memory'"""
            ).fetchall()
            names = {r["name"] for r in rows}
            assert "idx_project_memory_repo" in names
            assert "idx_project_memory_repo_kind" in names
            assert "idx_project_memory_expires" in names

    def test_unique_constraint_repo_kind_key(self, tmp_path: Path) -> None:
        """The UNIQUE (repo_id, kind, key) constraint is what makes
        ``remember`` an UPSERT — the same key can't be inserted twice."""
        with Storage(tmp_path / "u.db", embedding_dim=4) as s:
            repo_id = s.register_repo(
                RepoIdentity(
                    normalized_key="k", remote_url=None,
                    root_path=Path("/r"), default_branch=None,
                    head_commit=None,
                )
            )
            now = datetime.now(UTC).isoformat()
            s._conn.execute(
                """INSERT INTO project_memory (
                       repo_id, kind, key, value, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (repo_id, "decision", "k1", "v1", now, now),
            )
            s._conn.commit()
            import sqlite3

            with pytest.raises(sqlite3.IntegrityError):
                s._conn.execute(
                    """INSERT INTO project_memory (
                           repo_id, kind, key, value, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (repo_id, "decision", "k1", "v2", now, now),
                )


# ---------------------------------------------------------------------------
# Storage CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "pm.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="github.com/test/repo",
            remote_url="https://github.com/test/repo.git",
            root_path=Path("/tmp/test"),
            default_branch="main",
            head_commit="abc",
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


class TestRemember:
    def test_insert_creates_row(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.remember(
            repo_id=repo_id, kind="decision", key="fsrs-authoritative",
            value="The FSRS scheduler is the source of truth for review intervals.",
        )
        assert rec.id is not None
        assert rec.repo_id == repo_id
        assert rec.kind == "decision"
        assert rec.key == "fsrs-authoritative"
        assert rec.value.startswith("The FSRS")
        assert rec.created_at == rec.updated_at

    def test_upsert_same_key_updates_in_place(self, storage_with_repo) -> None:
        import time

        s, repo_id = storage_with_repo
        first = s.remember(
            repo_id=repo_id, kind="fact", key="lang", value="Python",
        )
        time.sleep(0.01)
        second = s.remember(
            repo_id=repo_id, kind="fact", key="lang", value="Python 3.12",
        )
        # Same id (UPSERT, not INSERT).
        assert second.id == first.id
        # updated_at bumped.
        assert second.updated_at >= first.updated_at
        # created_at preserved.
        assert second.created_at == first.created_at
        # Value reflects the latest write.
        assert second.value == "Python 3.12"

    def test_source_provenance_captured(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            source="session-abc",
        )
        assert rec.source == "session-abc"

    def test_expires_at_round_trip(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        expiry = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        rec = s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            expires_at=expiry,
        )
        assert rec.expires_at == expiry

    def test_different_kind_same_key_does_not_collide(
        self, storage_with_repo,
    ) -> None:
        """The UNIQUE constraint is (repo_id, kind, key) — same key
        under different kinds coexist."""
        s, repo_id = storage_with_repo
        s.remember(
            repo_id=repo_id, kind="decision", key="x", value="a",
        )
        s.remember(
            repo_id=repo_id, kind="fact", key="x", value="b",
        )
        rows = s.recall(repo_id=repo_id)
        assert len(rows) == 2

    def test_different_repos_same_key_do_not_collide(
        self, tmp_path: Path,
    ) -> None:
        s = Storage(tmp_path / "two.db", embedding_dim=4)
        s.__enter__()
        repo_a = s.register_repo(
            RepoIdentity(
                normalized_key="a", remote_url=None,
                root_path=Path("/a"), default_branch=None,
                head_commit=None,
            )
        )
        repo_b = s.register_repo(
            RepoIdentity(
                normalized_key="b", remote_url=None,
                root_path=Path("/b"), default_branch=None,
                head_commit=None,
            )
        )
        s.remember(
            repo_id=repo_a, kind="fact", key="x", value="A",
        )
        s.remember(
            repo_id=repo_b, kind="fact", key="x", value="B",
        )
        # Each repo sees only its own entry.
        a_rows = s.recall(repo_id=repo_a)
        b_rows = s.recall(repo_id=repo_b)
        assert len(a_rows) == 1
        assert len(b_rows) == 1
        assert a_rows[0].value == "A"
        assert b_rows[0].value == "B"
        s.__exit__(None, None, None)


class TestRecall:
    def test_unknown_repo_returns_empty(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        assert s.recall(repo_id=99999) == []

    def test_filter_by_kind(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="decision", key="d1", value="x")
        s.remember(repo_id=repo_id, kind="fact", key="f1", value="y")
        decisions = s.recall(repo_id=repo_id, kind="decision")
        facts = s.recall(repo_id=repo_id, kind="fact")
        assert len(decisions) == 1
        assert len(facts) == 1
        assert decisions[0].kind == "decision"
        assert facts[0].kind == "fact"

    def test_filter_by_key_with_kind(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="fact", key="x", value="A")
        s.remember(repo_id=repo_id, kind="fact", key="y", value="B")
        rows = s.recall(repo_id=repo_id, kind="fact", key="x")
        assert len(rows) == 1
        assert rows[0].value == "A"

    def test_sorted_by_updated_at_desc(self, storage_with_repo) -> None:
        import time

        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="x", key="old", value="1")
        time.sleep(0.01)
        s.remember(repo_id=repo_id, kind="x", key="newer", value="2")
        time.sleep(0.01)
        # Touch "old" — its updated_at should now beat "newer".
        s.remember(repo_id=repo_id, kind="x", key="old", value="1!")
        rows = s.recall(repo_id=repo_id, kind="x")
        assert rows[0].key == "old"  # most-recently updated

    def test_expired_entries_hidden_by_default(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        s.remember(
            repo_id=repo_id, kind="fact", key="stale", value="x",
            expires_at=past,
        )
        s.remember(repo_id=repo_id, kind="fact", key="live", value="y")
        rows = s.recall(repo_id=repo_id)
        keys = {r.key for r in rows}
        assert keys == {"live"}

    def test_include_expired_true(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        s.remember(
            repo_id=repo_id, kind="fact", key="stale", value="x",
            expires_at=past,
        )
        rows = s.recall(repo_id=repo_id, include_expired=True)
        assert len(rows) == 1

    def test_limit_truncates(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        for i in range(10):
            s.remember(
                repo_id=repo_id, kind="x", key=f"k{i}", value=str(i),
            )
        rows = s.recall(repo_id=repo_id, limit=3)
        assert len(rows) == 3


class TestForget:
    def test_existing_entry_returns_true(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="fact", key="x", value="y")
        assert s.forget(repo_id=repo_id, kind="fact", key="x") is True
        assert s.recall(repo_id=repo_id) == []

    def test_unknown_entry_returns_false(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        assert s.forget(repo_id=repo_id, kind="fact", key="ghost") is False

    def test_forget_scoped_to_kind(self, storage_with_repo) -> None:
        """Forgetting (fact, x) does not remove (decision, x)."""
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="fact", key="x", value="A")
        s.remember(repo_id=repo_id, kind="decision", key="x", value="B")
        s.forget(repo_id=repo_id, kind="fact", key="x")
        rows = s.recall(repo_id=repo_id)
        assert len(rows) == 1
        assert rows[0].kind == "decision"


class TestPruneProjectMemory:
    def test_no_args_is_noop(self, storage_with_repo) -> None:
        """Refusing to wipe everything implicitly — explicit filter
        required."""
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="fact", key="x", value="y")
        removed = s.prune_project_memory()
        assert removed == 0
        assert len(s.recall(repo_id=repo_id)) == 1

    def test_prune_by_repo(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        s.remember(repo_id=repo_id, kind="x", key="a", value="1")
        s.remember(repo_id=repo_id, kind="x", key="b", value="2")
        removed = s.prune_project_memory(repo_id=repo_id)
        assert removed == 2
        assert s.recall(repo_id=repo_id) == []

    def test_prune_older_than_iso(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        # Plant rows with controlled timestamps.
        old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        new = datetime.now(UTC).isoformat()
        s._conn.execute(
            """INSERT INTO project_memory (
                   repo_id, kind, key, value, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (repo_id, "fact", "old", "x", old, old),
        )
        s._conn.execute(
            """INSERT INTO project_memory (
                   repo_id, kind, key, value, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (repo_id, "fact", "new", "y", new, new),
        )
        s._conn.commit()
        cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        removed = s.prune_project_memory(
            repo_id=repo_id, older_than_iso=cutoff,
        )
        assert removed == 1
        survivors = {r.key for r in s.recall(repo_id=repo_id)}
        assert survivors == {"new"}

    def test_prune_expired_only(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        s.remember(
            repo_id=repo_id, kind="fact", key="stale", value="x",
            expires_at=past,
        )
        s.remember(repo_id=repo_id, kind="fact", key="live", value="y")
        removed = s.prune_project_memory(
            repo_id=repo_id, expired_only=True,
        )
        assert removed == 1
        survivors = {r.key for r in s.recall(repo_id=repo_id)}
        assert survivors == {"live"}

    def test_expired_only_overrides_older_than_iso(
        self, storage_with_repo,
    ) -> None:
        """When both filters are passed, expired_only wins (it's the
        explicit TTL signal)."""
        s, repo_id = storage_with_repo
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        s.remember(
            repo_id=repo_id, kind="fact", key="stale-expired",
            value="x", expires_at=past,
        )
        s.remember(
            repo_id=repo_id, kind="fact", key="old-untimed-but-no-expiry",
            value="y",
            # Note: this row's updated_at is fresh, but it has no
            # expiry → won't be touched by expired_only.
        )
        removed = s.prune_project_memory(
            repo_id=repo_id,
            older_than_iso=future,  # would normally remove both
            expired_only=True,       # but this wins
        )
        # Only the actually-expired row is touched.
        assert removed == 1
        survivors = {r.key for r in s.recall(repo_id=repo_id)}
        assert survivors == {"old-untimed-but-no-expiry"}


# ---------------------------------------------------------------------------
# ProjectMemoryRecord
# ---------------------------------------------------------------------------


class TestProjectMemoryRecord:
    def test_defaults(self) -> None:
        rec = ProjectMemoryRecord(
            repo_id=1, kind="fact", key="x", value="y",
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        assert rec.id is None
        assert rec.source is None
        assert rec.expires_at is None
