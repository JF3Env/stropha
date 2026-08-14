from __future__ import annotations

import json
import re
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from stropha.evaluation import postgres as postgres_module
from stropha.evaluation.postgres import (
    DatabaseRef,
    PostgresEvaluationDatabaseManager,
    PostgresEvaluationError,
    PostgresEvaluationSafetyError,
    redact_dsn,
)

ADMIN_DSN = "postgresql://eval_user:top-secret@db.internal:5544/eval_admin"
PREFIX = "eval"
OWNER = "eval_user"


@dataclass
class _FakeDatabase:
    owner: str
    comment: str | None = None
    allow_connections: bool = True
    is_template: bool = False
    extensions: tuple[tuple[str, str], ...] = ()


@dataclass
class _FakeInfo:
    host: str = "db.internal"
    port: int = 5544


class _FakeCluster:
    def __init__(self, *, can_create_database: bool = True) -> None:
        self.can_create_database = can_create_database
        self.databases = {
            "eval_admin": _FakeDatabase(owner=OWNER),
            "template1": _FakeDatabase(
                owner="postgres",
                is_template=True,
                extensions=(("plpgsql", "1.0"), ("vector", "0.8.1")),
            ),
        }
        self.active_connections: dict[str, int] = {}
        self.calls: list[tuple[str, object]] = []
        self.connect_calls: list[tuple[str, bool]] = []
        self.composable_ddl: list[str] = []
        self.drops: list[str] = []

    def connect(self, dsn: str, *, autocommit: bool = False) -> _FakeConnection:
        self.connect_calls.append((dsn, autocommit))
        database = str(conninfo_to_dict(dsn).get("dbname") or "eval_admin")
        return _FakeConnection(self, database)

    def marker(self, kind: str, cache_key: str) -> str:
        return json.dumps(
            {
                "stropha_evaluation_database": 1,
                "cache_key": cache_key,
                "created_at": "2026-08-14T00:00:00+00:00",
                "kind": kind,
                "prefix": PREFIX,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class _FakeConnection:
    def __init__(self, cluster: _FakeCluster, database: str) -> None:
        self.cluster = cluster
        self.database = database
        self.info = _FakeInfo()
        self.closed = False

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.cluster, self.database)

    def execute(self, query: object, params: object = None) -> _FakeCursor:
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor

    def close(self) -> None:
        self.closed = True


class _FakeCursor:
    def __init__(self, cluster: _FakeCluster, database: str) -> None:
        self.cluster = cluster
        self.database = database
        self._result: tuple[object, ...] | None = None
        self._results: list[tuple[object, ...]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: object, params: object = None) -> None:
        if isinstance(query, sql.Composable):
            statement = query.as_string()
            self.cluster.composable_ddl.append(statement)
        else:
            statement = str(query)
        statement = " ".join(statement.split())
        self.cluster.calls.append((statement, params))
        self._result = None
        self._results = []

        if statement.startswith("SELECT current_database()"):
            self._result = (
                self.database,
                OWNER,
                "17.2",
                self.cluster.can_create_database,
            )
            return
        if statement.startswith("SELECT pg_get_userbyid(datdba), datistemplate"):
            assert isinstance(params, tuple)
            database = self.cluster.databases.get(str(params[0]))
            if database is not None:
                self._result = (database.owner, database.is_template)
            return
        if statement.startswith("SELECT extname, extversion FROM pg_extension"):
            self._results = list(self.cluster.databases[self.database].extensions)
            return
        if "FROM pg_database AS d" in statement:
            assert isinstance(params, tuple)
            name = str(params[0])
            database = self.cluster.databases.get(name)
            if database is not None:
                self._result = (name, database.owner, database.comment)
            return
        if statement.startswith("SELECT count(*) FROM pg_stat_activity"):
            assert isinstance(params, tuple)
            self._result = (self.cluster.active_connections.get(str(params[0]), 0),)
            return
        if statement.startswith("SELECT pg_advisory_"):
            self._result = (True,)
            return

        create_match = re.fullmatch(r'CREATE DATABASE "([^"]+)"(?: TEMPLATE "([^"]+)")?', statement)
        if create_match:
            name, template = create_match.groups()
            if name in self.cluster.databases:
                raise psycopg.errors.DuplicateDatabase(name)
            if template is not None and template not in self.cluster.databases:
                raise psycopg.errors.InvalidCatalogName(template)
            self.cluster.databases[name] = _FakeDatabase(owner=OWNER)
            return

        comment_match = re.fullmatch(r"COMMENT ON DATABASE \"([^\"]+)\" IS '(.*)'", statement)
        if comment_match:
            assert params is None
            self.cluster.databases[comment_match.group(1)].comment = comment_match.group(2).replace(
                "''", "'"
            )
            return

        allow_match = re.fullmatch(
            r'ALTER DATABASE "([^"]+)" WITH ALLOW_CONNECTIONS false', statement
        )
        if allow_match:
            self.cluster.databases[allow_match.group(1)].allow_connections = False
            return

        rename_match = re.fullmatch(r'ALTER DATABASE "([^"]+)" RENAME TO "([^"]+)"', statement)
        if rename_match:
            source, target = rename_match.groups()
            self.cluster.databases[target] = self.cluster.databases.pop(source)
            return

        drop_match = re.fullmatch(r'DROP DATABASE "([^"]+)" WITH \(FORCE\)', statement)
        if drop_match:
            name = drop_match.group(1)
            self.cluster.databases.pop(name)
            self.cluster.active_connections.pop(name, None)
            self.cluster.drops.append(name)
            return

        raise AssertionError(f"Unexpected SQL: {statement}")

    def fetchone(self) -> tuple[object, ...] | None:
        return self._result

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._results


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> _FakeCluster:
    value = _FakeCluster()
    monkeypatch.setattr(postgres_module.psycopg, "connect", value.connect)
    return value


@pytest.fixture
def manager(cluster: _FakeCluster) -> PostgresEvaluationDatabaseManager:
    return PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
        production_database="stropha_production",
    )


@pytest.mark.parametrize("prefix", ["", "Eval", "_eval", "eval-test", "e" * 33])
def test_rejects_unsafe_database_prefix(prefix: str) -> None:
    with pytest.raises(PostgresEvaluationSafetyError):
        PostgresEvaluationDatabaseManager(
            admin_dsn=ADMIN_DSN,
            database_prefix=prefix,
        )


def test_rejects_non_dedicated_or_production_admin_database() -> None:
    with pytest.raises(PostgresEvaluationSafetyError, match="must begin"):
        PostgresEvaluationDatabaseManager(
            admin_dsn="postgresql://eval_user:secret@db.internal/postgres",
            database_prefix=PREFIX,
        )

    with pytest.raises(PostgresEvaluationSafetyError, match="production"):
        PostgresEvaluationDatabaseManager(
            admin_dsn=ADMIN_DSN,
            database_prefix=PREFIX,
            production_database="eval_admin",
        )


def test_redacts_dsn_and_password_safe_reprs() -> None:
    redacted = redact_dsn(ADMIN_DSN)
    database = DatabaseRef(name="eval_work_aabbccdd", dsn=ADMIN_DSN)
    manager = PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
    )

    assert "top-secret" not in redacted
    assert "password=***" in redacted
    assert "top-secret" not in repr(database)
    assert "top-secret" not in repr(manager)


def test_preflight_records_non_secret_identity(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    identity = manager.preflight()

    assert identity.host == "db.internal"
    assert identity.port == 5544
    assert identity.database == "eval_admin"
    assert identity.user == OWNER
    assert identity.server_version == "17.2"
    assert identity.template_database == "template1"
    assert identity.template_extensions == ("plpgsql=1.0", "vector=0.8.1")
    assert "top-secret" not in repr(identity)
    assert manager.cluster_identity is identity
    assert cluster.connect_calls == [
        (ADMIN_DSN, True),
        (
            "user=eval_user password=top-secret dbname=template1 host=db.internal port=5544",
            False,
        ),
    ]


def test_preflight_requires_createdb(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _FakeCluster(can_create_database=False)
    monkeypatch.setattr(postgres_module.psycopg, "connect", cluster.connect)
    manager = PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
    )

    with pytest.raises(PostgresEvaluationSafetyError, match="CREATEDB"):
        manager.preflight()


def test_preflight_requires_vector_in_template(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _FakeCluster()
    cluster.databases["template1"].extensions = (("plpgsql", "1.0"),)
    monkeypatch.setattr(postgres_module.psycopg, "connect", cluster.connect)
    manager = PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
    )

    with pytest.raises(PostgresEvaluationSafetyError, match="vector extension"):
        manager.preflight()


def test_connection_errors_do_not_leak_password(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_connect(_dsn: str, *, autocommit: bool = False) -> None:
        raise psycopg.OperationalError("top-secret")

    monkeypatch.setattr(postgres_module.psycopg, "connect", fail_connect)
    manager = PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
    )

    with pytest.raises(PostgresEvaluationError) as caught:
        manager.preflight()

    assert "top-secret" not in str(caught.value)


def test_cache_name_is_deterministic_and_within_postgres_limit(
    manager: PostgresEvaluationDatabaseManager,
) -> None:
    first = manager.cache_database_name("repo:sha:indexing")

    assert first == manager.cache_database_name("repo:sha:indexing")
    assert first != manager.cache_database_name("repo:sha:other")
    assert re.fullmatch(r"eval_cache_[0-9a-f]{24}", first)
    assert len(first.encode()) <= 63


def test_returns_existing_marked_canonical_cache_without_building(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    cache_key = "existing-cache"
    name = manager.cache_database_name(cache_key)
    cluster.databases[name] = _FakeDatabase(
        owner=OWNER,
        comment=cluster.marker("cache", cache_key),
        allow_connections=False,
    )

    result = manager.get_or_create_cache(
        cache_key,
        lambda _ref: pytest.fail("existing cache must not be rebuilt"),
    )

    assert result.name == name
    assert conninfo_to_dict(result.dsn)["dbname"] == name
    assert not any(call.startswith("CREATE DATABASE") for call, _ in cluster.calls)


def test_builds_staging_cache_and_atomically_publishes_it(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    built: list[DatabaseRef] = []

    result = manager.get_or_create_cache("new-cache", built.append)

    assert len(built) == 1
    staging = built[0]
    assert re.fullmatch(r"eval_staging_[0-9a-f]{16}", staging.name)
    assert len(staging.name.encode()) <= 63
    assert conninfo_to_dict(staging.dsn)["dbname"] == staging.name
    assert result.name == manager.cache_database_name("new-cache")
    assert staging.name not in cluster.databases
    canonical = cluster.databases[result.name]
    marker = json.loads(canonical.comment or "")
    assert marker["stropha_evaluation_database"] == 1
    assert marker["kind"] == "cache"
    assert marker["cache_key"] == "new-cache"
    assert marker["prefix"] == PREFIX
    assert marker["created_at"]
    assert canonical.allow_connections is False
    assert any(call.startswith("SELECT pg_advisory_lock") for call, _ in cluster.calls)
    assert any(call.startswith("SELECT pg_advisory_unlock") for call, _ in cluster.calls)
    staging_create = next(call for call, _ in cluster.calls if call.startswith("CREATE DATABASE"))
    assert 'TEMPLATE "template1"' in staging_create
    assert any(call.startswith("ALTER DATABASE") for call in cluster.composable_ddl)


def test_builder_failure_force_drops_marked_staging_without_leaking_error(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    built: list[DatabaseRef] = []

    def fail(ref: DatabaseRef) -> None:
        built.append(ref)
        raise RuntimeError("top-secret")

    with pytest.raises(PostgresEvaluationError) as caught:
        manager.get_or_create_cache("broken-cache", fail)

    assert "top-secret" not in str(caught.value)
    assert built[0].name not in cluster.databases
    assert built[0].name in cluster.drops
    assert any("WITH (FORCE)" in call for call in cluster.composable_ddl)


def test_builder_interrupt_force_drops_marked_staging(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    built: list[DatabaseRef] = []

    def interrupt(ref: DatabaseRef) -> None:
        built.append(ref)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        manager.get_or_create_cache("interrupted-cache", interrupt)

    assert built[0].name not in cluster.databases
    assert built[0].name in cluster.drops


def test_builder_postgres_error_preserves_safe_root_cause(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    def fail(_ref: DatabaseRef) -> None:
        raise psycopg.errors.InsufficientPrivilege("permission denied to create extension vector")

    with pytest.raises(PostgresEvaluationError, match="permission denied") as caught:
        manager.get_or_create_cache("postgres-error", fail)

    assert "top-secret" not in str(caught.value)
    assert any(name.startswith("eval_staging_") for name in cluster.drops)


def test_cache_builder_must_close_database_connections(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    built: list[DatabaseRef] = []

    def leave_connection_open(ref: DatabaseRef) -> None:
        built.append(ref)
        cluster.active_connections[ref.name] = 1

    with pytest.raises(PostgresEvaluationSafetyError, match="connections still open"):
        manager.get_or_create_cache("open-connection", leave_connection_open)

    assert built[0].name not in cluster.databases
    assert built[0].name in cluster.drops


def test_writable_clone_uses_cache_template_and_force_drops_in_finally(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    cache_key = "clone-source"
    cache_name = manager.cache_database_name(cache_key)
    cluster.databases[cache_name] = _FakeDatabase(
        owner=OWNER,
        comment=cluster.marker("cache", cache_key),
        allow_connections=False,
    )

    with (
        pytest.raises(RuntimeError, match="evaluation failed"),
        manager.writable_clone(cache_name) as work,
    ):
        assert re.fullmatch(r"eval_work_[0-9a-f]{16}", work.name)
        assert len(work.name.encode()) <= 63
        assert conninfo_to_dict(work.dsn)["dbname"] == work.name
        marker = json.loads(cluster.databases[work.name].comment or "")
        assert marker["kind"] == "work"
        assert marker["cache_key"] == cache_key
        raise RuntimeError("evaluation failed")

    assert any(
        call.startswith("CREATE DATABASE") and f'TEMPLATE "{cache_name}"' in call
        for call, _ in cluster.calls
    )
    work_drop = cluster.drops[-1]
    assert work_drop.startswith("eval_work_")
    assert work_drop not in cluster.databases


@pytest.mark.parametrize(
    ("name", "database"),
    [
        ("eval", _FakeDatabase(owner=OWNER)),
        ("eval_cache_aaaaaaaaaaaaaaaaaaaaaaaa", _FakeDatabase(owner=OWNER)),
        (
            "eval_cache_bbbbbbbbbbbbbbbbbbbbbbbb",
            _FakeDatabase(owner="different_owner"),
        ),
    ],
)
def test_drop_refuses_prefix_only_unmarked_or_unowned_database(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
    name: str,
    database: _FakeDatabase,
) -> None:
    cluster.databases[name] = database

    with pytest.raises(PostgresEvaluationSafetyError):
        manager.drop_database(name, expected_kind="cache")

    assert name in cluster.databases
    assert name not in cluster.drops


def test_drop_refuses_admin_and_wrong_marker_kind(
    manager: PostgresEvaluationDatabaseManager,
    cluster: _FakeCluster,
) -> None:
    cache_name = "eval_cache_cccccccccccccccccccccccc"
    cluster.databases[cache_name] = _FakeDatabase(
        owner=OWNER,
        comment=cluster.marker("work", "wrong-kind"),
    )

    with pytest.raises(PostgresEvaluationSafetyError, match="admin"):
        manager.drop_database("eval_admin", expected_kind="cache")
    with pytest.raises(PostgresEvaluationSafetyError, match="marker"):
        manager.drop_database(cache_name, expected_kind="cache")

    assert "eval_admin" in cluster.databases
    assert cache_name in cluster.databases


def test_drop_refuses_supplied_production_identity(
    cluster: _FakeCluster,
) -> None:
    production = "eval_cache_dddddddddddddddddddddddd"
    manager = PostgresEvaluationDatabaseManager(
        admin_dsn=ADMIN_DSN,
        database_prefix=PREFIX,
        production_database=production,
    )
    cluster.databases[production] = _FakeDatabase(
        owner=OWNER,
        comment=cluster.marker("cache", "production"),
    )

    with pytest.raises(PostgresEvaluationSafetyError, match="production"):
        manager.drop_database(production, expected_kind="cache")

    assert production in cluster.databases
