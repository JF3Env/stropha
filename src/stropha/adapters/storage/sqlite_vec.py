"""Storage adapter for sqlite-vec.

Subclasses the existing ``stropha.storage.sqlite.Storage`` so the entire
read/write API surface is inherited unchanged. Adds:

- ``Config`` (path) — user-facing knobs in YAML/env
- ``Stage`` introspection properties (``stage_name``, ``adapter_id``, …)
- ``health()`` probe — confirms the SQLite file is openable

Default path resolution mirrors :func:`stropha.config.models._default_index_path`
so that ``build_stages()`` and ``Config.resolve_index_path()`` ALWAYS agree
on which SQLite file to open. Without this, the CLI's storage adapter
would open ``~/.stropha/index.db`` while the MCP server's ``get_config``
tool would report ``./.stropha/index.db`` — silently splitting the index
into two files. See ``CLAUDE.md`` §3 for the invariant.

The path string is expanded (``~`` and ``$VARS``) at adapter
instantiation time.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from ...config.models import _default_index_path
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...storage.sqlite import Storage


class SqliteVecStorageConfig(BaseModel):
    path: str = Field(
        default_factory=lambda: str(_default_index_path()),
        description=(
            "Path to the SQLite DB. Defaults to ``./.stropha/index.db`` when "
            "the cwd has a ``.stropha/`` directory, else ``~/.stropha/index.db``. "
            "``~`` and ``$VARS`` are expanded at adapter instantiation."
        ),
    )


@register_adapter(stage="storage", name="sqlite-vec")
class SqliteVecStorage(Storage):
    """Storage stage backed by SQLite + sqlite-vec + FTS5."""

    Config = SqliteVecStorageConfig

    def __init__(
        self,
        config: SqliteVecStorageConfig | None = None,
        *,
        embedding_dim: int | None = None,
        force_rebuild: bool = False,
    ) -> None:
        if config is None:
            config = SqliteVecStorageConfig()
        if embedding_dim is None:
            raise ValueError(
                "SqliteVecStorage requires `embedding_dim=` (propagate from "
                "the active embedder)."
            )
        self._adapter_config = config
        resolved = Path(os.path.expandvars(config.path)).expanduser()
        # force_rebuild=True lets the storage drop dim-dependent tables
        # (vec_chunks) before the consistency check fires. Required when
        # switching embedder dims (e.g. bge-m3 1024 → voyage-code-3 512).
        super().__init__(
            resolved,
            embedding_dim=embedding_dim,
            force_rebuild=force_rebuild,
        )

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "storage"

    @property
    def adapter_name(self) -> str:
        return "sqlite-vec"

    @property
    def adapter_id(self) -> str:
        return f"sqlite-vec:dim={self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return SqliteVecStorageConfig

    def health(self) -> StageHealth:
        # The connection is open by the time we're here (super().__init__
        # raises on failure). Report DB size + chunk count as detail.
        try:
            stats = self.stats()
            return StageHealth(
                status="ready",
                message=f"sqlite-vec ready ({stats['chunks']} chunks)",
                detail={
                    "db_path": stats["db_path"],
                    "chunks": str(stats["chunks"]),
                    "dim": str(stats["index_dim"]),
                },
            )
        except Exception as exc:  # pragma: no cover  defensive
            return StageHealth(
                status="error",
                message=f"sqlite-vec probe failed: {exc}",
            )
