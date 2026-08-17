"""Unit tests for the PostgreSQL retrieval connection pool."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

psycopg_pool = pytest.importorskip("psycopg_pool")
PoolTimeout = psycopg_pool.PoolTimeout

from stropha.adapters.storage.postgres import (  # noqa: E402
    PostgresStorage,
    PostgresStorageConfig,
)
from stropha.errors import StorageError  # noqa: E402


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    @contextmanager
    def transaction(self):
        yield self


def _bare_storage() -> PostgresStorage:
    storage = object.__new__(PostgresStorage)
    storage._adapter_config = PostgresStorageConfig(
        dsn="postgresql://u:p@127.0.0.1:1/none",
        read_pool_size=3,
        read_pool_timeout=0.1,
    )
    storage._dsn = storage._adapter_config.dsn
    storage._index_lock_held = False
    storage._raw = _FakeWriter()
    storage._read_pool = MagicMock()
    storage._read_pool.closed = False
    return storage


def test_open_read_pool_uses_configured_bounds() -> None:
    storage = _bare_storage()
    with patch("psycopg_pool.ConnectionPool") as pool_type:
        pool_type.check_connection = MagicMock()
        pool = storage._open_read_pool()

    assert pool is pool_type.return_value
    assert pool_type.call_args.kwargs["min_size"] == 1
    assert pool_type.call_args.kwargs["max_size"] == 3
    assert pool_type.call_args.kwargs["timeout"] == 0.1
    pool.wait.assert_called_once_with(timeout=0.1)


def test_open_read_pool_closes_partial_resources_on_failure() -> None:
    storage = _bare_storage()
    with patch("psycopg_pool.ConnectionPool") as pool_type:
        pool_type.check_connection = MagicMock()
        pool_type.return_value.wait.side_effect = RuntimeError("offline")

        with pytest.raises(StorageError, match="initialize"):
            storage._open_read_pool()

    pool_type.return_value.close.assert_called_once_with()
    assert storage._raw.closed is True


def test_read_connection_uses_pool_outside_indexing() -> None:
    storage = _bare_storage()
    connection = object()
    storage._read_pool.connection.return_value.__enter__.return_value = connection

    with storage._read_connection() as actual:
        assert actual is connection

    storage._read_pool.connection.assert_called_once_with(timeout=0.1)


def test_read_connection_uses_writer_during_indexing() -> None:
    storage = _bare_storage()
    storage._index_lock_held = True

    with storage._read_connection() as actual:
        assert actual is storage._raw

    storage._read_pool.connection.assert_not_called()
    assert storage.supports_concurrent_reads is False


def test_read_connection_converts_pool_timeout() -> None:
    storage = _bare_storage()
    storage._read_pool.connection.side_effect = PoolTimeout("full")

    with pytest.raises(StorageError, match="Timed out"), storage._read_connection():
        pass


def test_concurrent_read_capability_tracks_pool_lifecycle() -> None:
    storage = _bare_storage()
    assert storage.supports_concurrent_reads is True

    storage._read_pool.closed = True
    assert storage.supports_concurrent_reads is False


def test_read_pool_stats_are_exposed() -> None:
    storage = _bare_storage()
    storage._read_pool.get_stats.return_value = {
        "pool_available": 2,
        "requests_waiting": 1,
        "requests_wait_ms": 12,
    }

    assert storage.read_pool_stats == {
        "pool_available": 2,
        "requests_waiting": 1,
        "requests_wait_ms": 12,
    }
