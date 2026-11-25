"""Writer-lock tests for the PostgreSQL adapter.

These cover the ``pg_try_advisory_lock`` / ``pg_advisory_unlock`` pair that
serialises index writers. The live-Postgres tests in ``test_postgres_storage.py``
only exercise the happy path (acquire, release, re-acquire) and are skipped
whenever ``STROPHA_PG_DSN`` is unreachable, which left two real defects open:

* ``release_index_lock()`` returned early when the per-instance
  ``_index_lock_held`` flag was false, so a lock acquired before a runtime
  reload stayed held on the daemon's writer session indefinitely;
* a release path naming a function PostgreSQL does not have
  (``pg_try_advisory_unlock``) raises 42883, and the exception handler's
  reconnect abandoned the lock instead of freeing it.

Both are testable without a server by observing the SQL the adapter sends.
"""

from __future__ import annotations

from contextlib import contextmanager

from stropha.adapters.storage.postgres import (
    _INDEX_LOCK_KEY,
    PostgresStorage,
    PostgresStorageConfig,
)


class _RecordingWriter:
    """Minimal stand-in for the long-lived psycopg writer connection."""

    def __init__(self, *, unlock_result: bool = True, lock_result: bool = True) -> None:
        self.closed = False
        self.sent: list[tuple[str, tuple[object, ...] | None]] = []
        self.commits = 0
        self.rollbacks = 0
        self._unlock_result = unlock_result
        self._lock_result = lock_result
        self._last_kind = ""

    def close(self) -> None:
        self.closed = True

    def rollback(self) -> None:
        self.rollbacks += 1

    def commit(self) -> None:
        self.commits += 1

    @contextmanager
    def transaction(self):
        yield self

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> _RecordingWriter:
        if "unlock" in sql:
            self._last_kind = "unlock"
        elif "pg_locks" in sql:
            self._last_kind = "holder"
        elif "lock" in sql:
            self._last_kind = "lock"
        else:
            self._last_kind = "other"
        self.sent.append((sql, params))
        return self

    def fetchone(self) -> dict[str, object]:
        if self._last_kind == "unlock":
            return {"released": self._unlock_result}
        if self._last_kind == "holder":
            return {"pid": 4321, "state": "idle", "idle_s": 777}
        return {"ok": self._lock_result}

    @property
    def sql(self) -> list[str]:
        return [statement for statement, _ in self.sent]


def _storage_with(writer: _RecordingWriter, *, lock_held: bool) -> PostgresStorage:
    storage = object.__new__(PostgresStorage)
    storage._adapter_config = PostgresStorageConfig(
        dsn="postgresql://u:p@127.0.0.1:1/none",
        read_pool_size=1,
        read_pool_timeout=0.1,
    )
    storage._dsn = storage._adapter_config.dsn
    storage._index_lock_held = lock_held
    storage._raw = writer
    return storage


def test_release_unlocks_even_when_the_flag_was_lost() -> None:
    """A session advisory lock outlives the object that took it.

    The daemon keeps one writer session across runtime reloads, and each reload
    builds a fresh adapter whose flag starts False. Returning early in that
    state strands the lock on a session that never ends, so every later
    ``stropha index`` fails with "wait for it to finish" while nothing is left
    to wait for.
    """
    writer = _RecordingWriter(unlock_result=True)
    storage = _storage_with(writer, lock_held=False)

    storage.release_index_lock()

    unlock = [statement for statement in writer.sql if "pg_advisory_unlock" in statement]
    assert unlock, f"release never attempted the unlock: {writer.sql}"
    assert writer.sent[0][1] == (_INDEX_LOCK_KEY,)
    assert writer.commits == 1
    # The connection is healthy, so it must not be replaced: reconnecting would
    # leave whatever the session held behind.
    assert writer.closed is False
    assert storage._raw is writer


def test_release_never_names_a_function_postgres_lacks() -> None:
    """PostgreSQL has ``pg_advisory_unlock`` and no ``pg_try_advisory_unlock``.

    Naming the missing one raises 42883, the ``except`` handler reconnects, and
    the lock stays held by the abandoned session — the exact leak class this
    guards.
    """
    writer = _RecordingWriter(unlock_result=True)
    storage = _storage_with(writer, lock_held=True)

    storage.release_index_lock()

    joined = " ".join(writer.sql)
    assert "pg_advisory_unlock" in joined
    assert "pg_try_advisory_unlock" not in joined
    assert storage._index_lock_held is False


def test_release_tolerates_a_lock_another_session_already_took() -> None:
    """``pg_advisory_unlock`` returns false when this session holds nothing.

    That must stay a quiet outcome, not an error path: the adapter still clears
    its flag and keeps its connection.
    """
    writer = _RecordingWriter(unlock_result=False)
    storage = _storage_with(writer, lock_held=True)

    storage.release_index_lock()

    assert storage._index_lock_held is False
    assert writer.closed is False
    assert storage._raw is writer


def test_acquire_records_the_key_and_its_result() -> None:
    """Acquire must ask for the single bigint overload and report the outcome."""
    writer = _RecordingWriter(lock_result=True)
    storage = _storage_with(writer, lock_held=False)

    assert storage.try_acquire_index_lock() is True
    assert storage._index_lock_held is True
    assert any("pg_try_advisory_lock" in statement for statement in writer.sql)
    assert writer.sent[0][1] == (_INDEX_LOCK_KEY,)
    assert writer.commits == 1

    busy = _RecordingWriter(lock_result=False)
    busy_storage = _storage_with(busy, lock_held=False)
    assert busy_storage.try_acquire_index_lock() is False
    assert busy_storage._index_lock_held is False


def test_lock_holder_diagnostics_ends_its_transaction() -> None:
    """The busy-path diagnostic must not leave the session idle in transaction.

    An idle-in-transaction session keeps its relation locks for the life of the
    session, and that is what makes a later schema migration time out waiting
    for a table lock. It also has to name the holder, or the error text stays
    unactionable.
    """
    writer = _RecordingWriter()
    storage = _storage_with(writer, lock_held=False)

    holder = storage.describe_index_lock_holder()

    assert holder is not None
    assert "4321" in holder
    assert "pg_terminate_backend(4321)" in holder
    assert any("pg_locks" in statement for statement in writer.sql)
    # The read is closed out, so nothing is left open on the writer session.
    assert writer.rollbacks == 1
    assert writer.closed is False
