"""Unit tests for PostgresStorage connection self-healing (PG-bounce survival).

These do NOT need a real Postgres: we bypass ``__init__`` and drive
``ensure_connected`` / ``_reconnect`` against fake psycopg-like connections.

Context: the backend holds a single long-lived psycopg connection. When the
Postgres container is bounced out from under a running daemon (e.g. a
Colima/Docker VM restart on laptop sleep) the connection dies; without a heal,
every subsequent query fails on a dead socket until the daemon is restarted.
``ensure_connected`` (called at each MCP request boundary) detects the drop and
reconnects, rebinding the SQLite-dialect shim so all downstream consumers heal.
"""

from __future__ import annotations

import psycopg

from stropha.adapters.storage.postgres import PostgresStorage, _CompatConn


class _FakeRaw:
    """Minimal stand-in for a psycopg connection."""

    def __init__(self, *, closed: bool = False, ping_error: Exception | None = None) -> None:
        self.closed = closed
        self._ping_error = ping_error
        self.rolled_back = False
        self.close_called = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._ping_error)

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True
        self.close_called = True


class _FakeCursor:
    def __init__(self, ping_error: Exception | None) -> None:
        self._ping_error = ping_error

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = ()) -> None:
        if self._ping_error is not None:
            raise self._ping_error


def _bare_storage(raw: _FakeRaw) -> PostgresStorage:
    st = object.__new__(PostgresStorage)
    st._index_lock_held = False
    st._dsn = "postgresql://u:p@127.0.0.1:1/none"
    st._raw = raw
    st._conn = _CompatConn(raw)
    return st


def test_ensure_connected_noop_when_healthy() -> None:
    alive = _FakeRaw()
    st = _bare_storage(alive)

    def _must_not_reconnect() -> _FakeRaw:
        raise AssertionError("must not reconnect a healthy connection")

    st._connect = _must_not_reconnect  # type: ignore[method-assign]

    st.ensure_connected()

    assert st._raw is alive
    assert alive.rolled_back is True  # ping transaction was closed cleanly
    assert alive.close_called is False


def test_ensure_connected_reconnects_on_terminated_connection() -> None:
    dead = _FakeRaw(
        ping_error=psycopg.OperationalError("terminating connection due to administrator command")
    )
    fresh = _FakeRaw()
    st = _bare_storage(dead)
    st._connect = lambda: fresh  # type: ignore[method-assign]

    st.ensure_connected()

    assert st._raw is fresh
    assert st._conn._raw is fresh  # shim rebound -> downstream consumers heal
    assert dead.close_called is True


def test_ensure_connected_reconnects_when_closed_flag_set() -> None:
    dead = _FakeRaw(closed=True)
    fresh = _FakeRaw()
    st = _bare_storage(dead)
    st._connect = lambda: fresh  # type: ignore[method-assign]

    st.ensure_connected()

    assert st._raw is fresh
    assert st._conn._raw is fresh


def test_ensure_connected_swallows_reconnect_failure() -> None:
    dead = _FakeRaw(ping_error=psycopg.OperationalError("gone"))
    st = _bare_storage(dead)

    def _still_down() -> _FakeRaw:
        raise psycopg.OperationalError("still down")

    st._connect = _still_down  # type: ignore[method-assign]

    # Must NOT raise — a genuine outage surfaces later on the real query.
    st.ensure_connected()

    assert st._raw is dead  # unchanged; reconnect failed but no exception escaped


def test_ensure_connected_skips_while_index_lock_held() -> None:
    alive = _FakeRaw()
    st = _bare_storage(alive)
    st._index_lock_held = True

    def _must_not_touch() -> _FakeRaw:
        raise AssertionError("must not touch the connection during indexing")

    st._connect = _must_not_touch  # type: ignore[method-assign]

    st.ensure_connected()

    assert alive.rolled_back is False  # skipped entirely (no ping)


def test_release_index_lock_rolls_back_aborted_transaction_first() -> None:
    calls: list[str] = []

    class _UnlockResult:
        def fetchone(self) -> dict[str, bool]:
            return {"ok": True}

    class _AbortedRaw(_FakeRaw):
        def rollback(self) -> None:
            calls.append("rollback")

        def execute(self, _sql: str, _params: object) -> _UnlockResult:
            calls.append("unlock")
            return _UnlockResult()

        def commit(self) -> None:
            calls.append("commit")

    storage = _bare_storage(_AbortedRaw())
    storage._index_lock_held = True

    storage.release_index_lock()

    assert calls == ["rollback", "unlock", "commit"]
    assert storage._index_lock_held is False
