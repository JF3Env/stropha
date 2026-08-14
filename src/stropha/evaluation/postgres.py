"""Native PostgreSQL database isolation for evaluation workloads.

The manager owns only disposable evaluation databases. It deliberately does not
know how to build a Stropha index: callers provide a builder callback and receive
a database-specific DSN. Destructive operations require an exact generated name,
a valid database marker, and ownership by the preflighted admin role.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from ..errors import StrophaError

_MAX_DATABASE_NAME_BYTES = 63
_MAX_PREFIX_LENGTH = 32
_MARKER_VERSION = 1
_MARKER_VERSION_FIELD = "stropha_evaluation_database"
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")

DatabaseKind = Literal["staging", "cache", "work"]
DatabaseBuilder = Callable[["DatabaseRef"], None]


class PostgresEvaluationError(StrophaError):
    """Failure while managing native PostgreSQL evaluation databases."""


class PostgresEvaluationSafetyError(PostgresEvaluationError):
    """A database failed a destructive-operation safety check."""


def redact_dsn(dsn: str) -> str:
    """Return a libpq conninfo string with its password replaced."""
    try:
        params = conninfo_to_dict(dsn)
        if "password" in params:
            params["password"] = "***"
        return make_conninfo("", **params)
    except psycopg.Error:
        return "<invalid PostgreSQL DSN>"


@dataclass(frozen=True, repr=False)
class DatabaseRef:
    """Name and connection string for one managed evaluation database."""

    name: str
    dsn: str

    def __repr__(self) -> str:
        return f"DatabaseRef(name={self.name!r}, dsn={redact_dsn(self.dsn)!r})"


@dataclass(frozen=True)
class ClusterIdentity:
    """Non-secret identity recorded by :meth:`preflight`."""

    host: str
    port: int
    database: str
    user: str
    server_version: str
    template_database: str
    template_extensions: tuple[str, ...]


@dataclass(frozen=True)
class _Marker:
    kind: DatabaseKind
    cache_key: str
    created_at: str
    prefix: str


@dataclass(frozen=True)
class _DatabaseRecord:
    name: str
    owner: str
    comment: str | None


class _BuilderFailed(PostgresEvaluationError):
    """Internal password-safe wrapper for an arbitrary builder exception."""


class PostgresEvaluationDatabaseManager:
    """Create guarded cache databases and short-lived writable clones."""

    def __init__(
        self,
        *,
        admin_dsn: str,
        database_prefix: str,
        template_database: str = "template1",
        production_database: str | None = None,
    ) -> None:
        self._validate_prefix(database_prefix)
        try:
            admin_params = conninfo_to_dict(admin_dsn)
        except psycopg.Error:
            raise PostgresEvaluationError("Invalid PostgreSQL admin DSN") from None

        admin_database = str(admin_params.get("dbname", ""))
        if not admin_database:
            raise PostgresEvaluationSafetyError(
                "The admin DSN must name a dedicated evaluation database"
            )
        if not admin_database.startswith(f"{database_prefix}_"):
            raise PostgresEvaluationSafetyError(
                f"Admin database must begin with {database_prefix + '_'!r}"
            )
        if production_database is not None and admin_database == production_database:
            raise PostgresEvaluationSafetyError(
                "The admin database must not be the production database"
            )
        if not _PREFIX_RE.fullmatch(template_database) or len(template_database.encode()) > 63:
            raise PostgresEvaluationSafetyError(
                "Template database must be a lowercase PostgreSQL identifier"
            )
        if template_database in {admin_database, production_database}:
            raise PostgresEvaluationSafetyError(
                "The template database must not be the admin or production database"
            )
        if template_database not in {"template0", "template1"} and not template_database.startswith(
            f"{database_prefix}_"
        ):
            raise PostgresEvaluationSafetyError(
                f"Custom template database must begin with {database_prefix + '_'!r}"
            )

        self._admin_dsn = admin_dsn
        self._admin_database = admin_database
        self._database_prefix = database_prefix
        self._template_database = template_database
        self._production_database = production_database
        self._identity: ClusterIdentity | None = None
        escaped_prefix = re.escape(database_prefix)
        self._name_patterns = {
            "cache": re.compile(rf"^{escaped_prefix}_cache_[0-9a-f]{{24}}$"),
            "staging": re.compile(rf"^{escaped_prefix}_staging_[0-9a-f]{{16}}$"),
            "work": re.compile(rf"^{escaped_prefix}_work_[0-9a-f]{{16}}$"),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(admin_dsn={redact_dsn(self._admin_dsn)!r}, "
            f"database_prefix={self._database_prefix!r}, "
            f"template_database={self._template_database!r}, "
            f"production_database={self._production_database!r})"
        )

    @property
    def cluster_identity(self) -> ClusterIdentity:
        """Return the cached cluster identity, running preflight if needed."""
        return self.preflight()

    @staticmethod
    def _validate_prefix(prefix: str) -> None:
        if not _PREFIX_RE.fullmatch(prefix) or len(prefix) > _MAX_PREFIX_LENGTH:
            raise PostgresEvaluationSafetyError(
                "Database prefix must be a lowercase PostgreSQL identifier fragment "
                f"of at most {_MAX_PREFIX_LENGTH} characters"
            )

    @contextmanager
    def _admin_connection(self, operation: str) -> Iterator[Any]:
        connection: Any = None
        try:
            connection = psycopg.connect(self._admin_dsn, autocommit=True)
            yield connection
        except PostgresEvaluationError:
            raise
        except psycopg.Error:
            raise PostgresEvaluationError(
                f"PostgreSQL evaluation {operation} failed at {redact_dsn(self._admin_dsn)}"
            ) from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except psycopg.Error:
                    raise PostgresEvaluationError(
                        f"Could not close the PostgreSQL admin connection after {operation}"
                    ) from None

    def preflight(self) -> ClusterIdentity:
        """Verify cluster access and CREATEDB, then record a non-secret identity."""
        if self._identity is not None:
            return self._identity

        with self._admin_connection("preflight") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_database(), current_user,
                           current_setting('server_version'),
                           COALESCE((
                               SELECT rolcreatedb OR rolsuper
                               FROM pg_roles
                               WHERE rolname = current_user
                           ), false)
                    """
                )
                row = cursor.fetchone()
            if row is None:
                raise PostgresEvaluationError("PostgreSQL preflight returned no identity")

            database, user, server_version, can_create_database = row
            database = str(database)
            if database != self._admin_database:
                raise PostgresEvaluationSafetyError(
                    "The admin DSN resolved to an unexpected database"
                )
            if not bool(can_create_database):
                raise PostgresEvaluationSafetyError("The PostgreSQL admin role must have CREATEDB")

            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT pg_get_userbyid(datdba), datistemplate
                    FROM pg_database
                    WHERE datname = %s
                    """,
                    (self._template_database,),
                )
                template_row = cursor.fetchone()
            if template_row is None:
                raise PostgresEvaluationSafetyError(
                    f"PostgreSQL evaluation template database {self._template_database!r} "
                    "does not exist"
                )
            template_owner, is_template = template_row
            if str(template_owner) != str(user) and not bool(is_template):
                raise PostgresEvaluationSafetyError(
                    "The PostgreSQL admin role must own the evaluation template database"
                )

            info = connection.info
            host = str(info.host or "local")
            port = int(info.port or 5432)

        template_dsn = self._database_dsn(self._template_database)
        try:
            with psycopg.connect(template_dsn) as template_connection:
                rows = template_connection.execute(
                    "SELECT extname, extversion FROM pg_extension ORDER BY extname"
                ).fetchall()
        except psycopg.Error:
            raise PostgresEvaluationError(
                "Could not inspect the PostgreSQL evaluation template database"
            ) from None
        extensions = tuple(f"{name}={version}" for name, version in rows)
        if not any(extension.startswith("vector=") for extension in extensions):
            raise PostgresEvaluationSafetyError(
                "PostgreSQL evaluation template database must have the vector extension installed"
            )
        self._identity = ClusterIdentity(
            host=host,
            port=port,
            database=database,
            user=str(user),
            server_version=str(server_version),
            template_database=self._template_database,
            template_extensions=extensions,
        )
        return self._identity

    def cache_database_name(self, cache_key: str) -> str:
        """Return the deterministic canonical database name for ``cache_key``."""
        self._validate_cache_key(cache_key)
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        return f"{self._database_prefix}_cache_{digest}"

    def _temporary_name(self, kind: Literal["staging", "work"]) -> str:
        name = f"{self._database_prefix}_{kind}_{secrets.token_hex(8)}"
        if len(name.encode()) > _MAX_DATABASE_NAME_BYTES:  # guarded by prefix validation
            raise PostgresEvaluationSafetyError("Generated database name is too long")
        return name

    @staticmethod
    def _validate_cache_key(cache_key: str) -> None:
        if not cache_key:
            raise PostgresEvaluationSafetyError("Cache key must not be empty")

    def _database_dsn(self, name: str) -> str:
        return make_conninfo(self._admin_dsn, dbname=name)

    def _ensure_not_protected(self, name: str) -> None:
        if name == self._admin_database:
            raise PostgresEvaluationSafetyError("Refusing to manage the admin database")
        if self._production_database is not None and name == self._production_database:
            raise PostgresEvaluationSafetyError("Refusing to manage the production database")

    def _name_kind(self, name: str) -> DatabaseKind | None:
        for kind, pattern in self._name_patterns.items():
            if pattern.fullmatch(name):
                return kind  # type: ignore[return-value]
        return None

    def _marker_comment(self, kind: DatabaseKind, cache_key: str) -> str:
        marker = {
            _MARKER_VERSION_FIELD: _MARKER_VERSION,
            "cache_key": cache_key,
            "created_at": datetime.now(UTC).isoformat(),
            "kind": kind,
            "prefix": self._database_prefix,
        }
        return json.dumps(marker, sort_keys=True, separators=(",", ":"))

    def _safe_builder_error(self, exc: BaseException) -> str | None:
        if not isinstance(exc, (StrophaError, psycopg.Error)):
            return None
        detail = str(exc)
        with_password = conninfo_to_dict(self._admin_dsn).get("password")
        if with_password:
            detail = detail.replace(str(with_password), "***")
        return detail[:500]

    @staticmethod
    def _decode_marker(comment: str | None) -> _Marker | None:
        if comment is None:
            return None
        try:
            value = json.loads(comment)
            if not isinstance(value, dict) or value.get(_MARKER_VERSION_FIELD) != _MARKER_VERSION:
                return None
            kind = value.get("kind")
            if kind not in {"staging", "cache", "work"}:
                return None
            cache_key = value.get("cache_key")
            created_at = value.get("created_at")
            prefix = value.get("prefix")
            if (
                not isinstance(cache_key, str)
                or not isinstance(created_at, str)
                or not isinstance(prefix, str)
            ):
                return None
            return _Marker(
                kind=kind,
                cache_key=cache_key,
                created_at=created_at,
                prefix=prefix,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _database_record(cursor: Any, name: str) -> _DatabaseRecord | None:
        cursor.execute(
            """
            SELECT d.datname, pg_get_userbyid(d.datdba),
                   shobj_description(d.oid, 'pg_database')
            FROM pg_database AS d
            WHERE d.datname = %s
            """,
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _DatabaseRecord(name=str(row[0]), owner=str(row[1]), comment=row[2])

    def _validate_database(
        self,
        cursor: Any,
        name: str,
        *,
        expected_name_kind: DatabaseKind,
        allowed_marker_kinds: set[DatabaseKind],
        expected_cache_key: str | None = None,
    ) -> _Marker:
        self._ensure_not_protected(name)
        if self._name_kind(name) != expected_name_kind:
            raise PostgresEvaluationSafetyError(
                f"Database {name!r} does not match the generated {expected_name_kind} pattern"
            )

        record = self._database_record(cursor, name)
        if record is None:
            raise PostgresEvaluationSafetyError(f"Managed database {name!r} does not exist")
        identity = self.preflight()
        if record.owner != identity.user:
            raise PostgresEvaluationSafetyError(
                f"Refusing database {name!r}: current admin role is not its owner"
            )

        marker = self._decode_marker(record.comment)
        if marker is None:
            raise PostgresEvaluationSafetyError(
                f"Refusing database {name!r}: Stropha evaluation marker is missing or invalid"
            )
        if marker.prefix != self._database_prefix or marker.kind not in allowed_marker_kinds:
            raise PostgresEvaluationSafetyError(
                f"Refusing database {name!r}: evaluation marker does not match the operation"
            )
        if expected_cache_key is not None and marker.cache_key != expected_cache_key:
            raise PostgresEvaluationSafetyError(
                f"Refusing database {name!r}: evaluation cache key does not match"
            )
        return marker

    @staticmethod
    def _comment_database(cursor: Any, name: str, comment: str) -> None:
        cursor.execute(
            sql.SQL("COMMENT ON DATABASE {} IS {}").format(
                sql.Identifier(name),
                sql.Literal(comment),
            )
        )

    @staticmethod
    def _active_connection_count(cursor: Any, name: str) -> int:
        cursor.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = %s",
            (name,),
        )
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def _drop_on_connection(
        self,
        cursor: Any,
        name: str,
        *,
        expected_name_kind: DatabaseKind,
        allowed_marker_kinds: set[DatabaseKind],
    ) -> None:
        self._validate_database(
            cursor,
            name,
            expected_name_kind=expected_name_kind,
            allowed_marker_kinds=allowed_marker_kinds,
        )
        cursor.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))

    def drop_database(self, name: str, *, expected_kind: DatabaseKind) -> None:
        """FORCE-drop one exactly named, marked, owned evaluation database."""
        self.preflight()
        with self._admin_connection("drop") as connection, connection.cursor() as cursor:
            self._drop_on_connection(
                cursor,
                name,
                expected_name_kind=expected_kind,
                allowed_marker_kinds={expected_kind},
            )

    def get_or_create_cache(
        self,
        cache_key: str,
        builder: DatabaseBuilder,
    ) -> DatabaseRef:
        """Return a marked canonical cache, building and publishing it if absent."""
        self._validate_cache_key(cache_key)
        self.preflight()
        canonical_name = self.cache_database_name(cache_key)
        staging_name = self._temporary_name("staging")
        self._ensure_not_protected(canonical_name)
        self._ensure_not_protected(staging_name)
        lock_key = int.from_bytes(
            hashlib.sha256(f"{self._database_prefix}\0{cache_key}".encode()).digest()[:8],
            signed=True,
        )

        with (
            self._admin_connection("cache creation") as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
            created = False
            try:
                existing = self._database_record(cursor, canonical_name)
                if existing is not None:
                    self._validate_database(
                        cursor,
                        canonical_name,
                        expected_name_kind="cache",
                        allowed_marker_kinds={"cache"},
                        expected_cache_key=cache_key,
                    )
                    return DatabaseRef(
                        name=canonical_name,
                        dsn=self._database_dsn(canonical_name),
                    )

                try:
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                            sql.Identifier(staging_name),
                            sql.Identifier(self._template_database),
                        )
                    )
                    created = True
                    self._comment_database(
                        cursor,
                        staging_name,
                        self._marker_comment("staging", cache_key),
                    )
                    staging_ref = DatabaseRef(
                        name=staging_name,
                        dsn=self._database_dsn(staging_name),
                    )
                    try:
                        builder(staging_ref)
                    except Exception as exc:
                        detail = self._safe_builder_error(exc)
                        raise _BuilderFailed(
                            f"Cache builder failed for database {staging_name!r}"
                            + (f": {detail}" if detail else "")
                        ) from None

                    cursor.execute(
                        sql.SQL("ALTER DATABASE {} WITH ALLOW_CONNECTIONS false").format(
                            sql.Identifier(staging_name)
                        )
                    )
                    if self._active_connection_count(cursor, staging_name):
                        raise PostgresEvaluationSafetyError(
                            "Cache builder returned with database connections still open"
                        )
                    self._comment_database(
                        cursor,
                        staging_name,
                        self._marker_comment("cache", cache_key),
                    )
                    cursor.execute(
                        sql.SQL("ALTER DATABASE {} RENAME TO {}").format(
                            sql.Identifier(staging_name),
                            sql.Identifier(canonical_name),
                        )
                    )
                    created = False
                except BaseException:
                    if created:
                        try:
                            self._drop_on_connection(
                                cursor,
                                staging_name,
                                expected_name_kind="staging",
                                allowed_marker_kinds={"staging", "cache"},
                            )
                        except Exception:
                            raise PostgresEvaluationError(
                                f"Cache build failed and staging cleanup was refused for "
                                f"{staging_name!r}"
                            ) from None
                    raise

                return DatabaseRef(
                    name=canonical_name,
                    dsn=self._database_dsn(canonical_name),
                )
            finally:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))

    @contextmanager
    def writable_clone(self, cache: DatabaseRef | str) -> Iterator[DatabaseRef]:
        """Yield a writable database cloned from ``cache`` and FORCE-drop it."""
        self.preflight()
        cache_name = cache.name if isinstance(cache, DatabaseRef) else cache
        work_name = self._temporary_name("work")
        self._ensure_not_protected(work_name)

        with (
            self._admin_connection("work clone creation") as connection,
            connection.cursor() as cursor,
        ):
            cache_marker = self._validate_database(
                cursor,
                cache_name,
                expected_name_kind="cache",
                allowed_marker_kinds={"cache"},
            )
            created = False
            try:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                        sql.Identifier(work_name),
                        sql.Identifier(cache_name),
                    )
                )
                created = True
                self._comment_database(
                    cursor,
                    work_name,
                    self._marker_comment("work", cache_marker.cache_key),
                )
            except BaseException:
                if created:
                    try:
                        self._drop_on_connection(
                            cursor,
                            work_name,
                            expected_name_kind="work",
                            allowed_marker_kinds={"work"},
                        )
                    except Exception:
                        raise PostgresEvaluationError(
                            f"Work clone creation failed and cleanup was refused for {work_name!r}"
                        ) from None
                raise

        work_ref = DatabaseRef(name=work_name, dsn=self._database_dsn(work_name))
        try:
            yield work_ref
        finally:
            self.drop_database(work_name, expected_kind="work")


__all__ = [
    "ClusterIdentity",
    "DatabaseBuilder",
    "DatabaseKind",
    "DatabaseRef",
    "PostgresEvaluationDatabaseManager",
    "PostgresEvaluationError",
    "PostgresEvaluationSafetyError",
    "redact_dsn",
]
