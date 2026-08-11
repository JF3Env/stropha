"""Unit tests for once-per-process schema migration (lock-wedge fix).

The MCP daemon builds a fresh PostgresStorage per transport session. Before
the ``_MIGRATED_DSNS`` memo, every session re-ran the migration DDL, and
``CREATE INDEX IF NOT EXISTS`` takes a SHARE lock on its table even when the
index already exists — queueing behind any in-flight ``stropha index`` write
transaction and wedging MCP session creation until the indexer committed.

These tests bypass ``__init__`` (no real Postgres needed) and drive
``_migrate`` directly: the memo must short-circuit repeat runs, and a lock
timeout must retry with bounded attempts instead of hanging or memoizing a
failed migration.
"""

from __future__ import annotations

import psycopg
import pytest

from stropha.adapters.storage import postgres as pg_mod
from stropha.adapters.storage.postgres import PostgresStorage, _CompatConn
from stropha.errors import StorageError

_TEST_DSN = "postgresql://u:p@127.0.0.1:1/migrate-once-test"


class _PoisonedRaw:
    """Connection stand-in that fails the test if the memo path touches it."""

    def cursor(self) -> None:
        raise AssertionError("memoized DSN must not touch the connection")

    def rollback(self) -> None:
        self.rolled_back = getattr(self, "rolled_back", 0) + 1


class _RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> None:
        self.calls.append((sql, params))


class _RecordingRaw:
    def __init__(self) -> None:
        self.recording_cursor = _RecordingCursor()
        self.committed = False

    def cursor(self) -> _RecordingCursor:
        return self.recording_cursor

    def commit(self) -> None:
        self.committed = True


class _LockingCursor(_RecordingCursor):
    def execute(self, sql: str, params: object = None) -> None:
        super().execute(sql, params)
        if sql.startswith("DROP TABLE"):
            raise psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")


class _LockingRaw(_RecordingRaw):
    def __init__(self) -> None:
        super().__init__()
        self.recording_cursor = _LockingCursor()
        self.rolled_back = False

    def rollback(self) -> None:
        self.rolled_back = True


def _bare_storage(dsn: str = _TEST_DSN) -> PostgresStorage:
    st = object.__new__(PostgresStorage)
    st._dsn = dsn
    st._index_lock_held = False
    st._raw = _PoisonedRaw()
    st._conn = _CompatConn(st._raw)
    return st


@pytest.fixture(autouse=True)
def _clean_memo():
    pg_mod._MIGRATED_DSNS.discard(_TEST_DSN)
    yield
    pg_mod._MIGRATED_DSNS.discard(_TEST_DSN)


def test_migrate_skipped_for_memoized_dsn() -> None:
    st = _bare_storage()
    pg_mod._MIGRATED_DSNS.add(_TEST_DSN)

    st._migrate()  # must return before any connection use


def test_migrate_runs_once_then_memoizes() -> None:
    st = _bare_storage()
    calls: list[int] = []
    st._run_migration_ddl = lambda: calls.append(1)  # type: ignore[method-assign]

    st._migrate()
    st._migrate()

    assert calls == [1]
    assert _TEST_DSN in pg_mod._MIGRATED_DSNS


def test_migrate_retries_lock_timeout_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _bare_storage()
    attempts: list[int] = []

    def _always_locked() -> None:
        attempts.append(1)
        raise psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")

    st._run_migration_ddl = _always_locked  # type: ignore[method-assign]
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with pytest.raises(StorageError, match="indexing"):
        st._migrate()

    assert len(attempts) == pg_mod._MIGRATE_LOCK_RETRIES
    assert st._raw.rolled_back == pg_mod._MIGRATE_LOCK_RETRIES
    # A failed migration must not be memoized — the next instance retries.
    assert _TEST_DSN not in pg_mod._MIGRATED_DSNS


def test_migrate_recovers_when_lock_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _bare_storage()
    attempts: list[int] = []

    def _locked_once() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise psycopg.errors.LockNotAvailable("canceling statement due to lock timeout")

    st._run_migration_ddl = _locked_once  # type: ignore[method-assign]
    monkeypatch.setattr("time.sleep", lambda _s: None)

    st._migrate()

    assert len(attempts) == 2
    assert _TEST_DSN in pg_mod._MIGRATED_DSNS


def test_migration_updates_existing_schema_version() -> None:
    st = _bare_storage()
    raw = _RecordingRaw()
    st._raw = raw
    st._dim = 4

    st._run_migration_ddl()

    schema_calls = [
        (sql, params)
        for sql, params in raw.recording_cursor.calls
        if "VALUES ('schema_version'" in sql
    ]
    assert len(schema_calls) == 1
    sql, params = schema_calls[0]
    assert "DO UPDATE SET value = EXCLUDED.value" in sql
    assert "CAST(meta.value AS integer) < CAST(EXCLUDED.value AS integer)" in sql
    assert params == ("14",)
    assert raw.committed is True


def test_dimension_reset_has_bounded_lock_wait() -> None:
    st = _bare_storage()
    raw = _LockingRaw()
    st._raw = raw
    st._dim = 2560
    st.get_meta = lambda _key: "1024"  # type: ignore[method-assign]

    with pytest.raises(StorageError, match="timed out waiting for active Postgres readers"):
        st._reset_dim_dependent_tables_if_needed()

    assert raw.recording_cursor.calls[0][0].startswith("SET LOCAL lock_timeout")
    assert raw.rolled_back is True
