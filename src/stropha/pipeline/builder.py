"""Builder: resolved config dict → instantiated adapter objects.

Importing this module triggers ``stropha.adapters`` auto-registration so
``lookup_adapter`` resolves immediately.

Build order (Phase 2):
    walker → enricher → embedder → storage(embedder.dim) → retrieval(storage, embedder)

The chunker stage (Phase 3) is built lazily by the Pipeline using the
legacy ``stropha.ingest.chunker.Chunker`` until §10.3 ships its adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import adapters  # noqa: F401  — side-effect: populate registry
from ..errors import ConfigError
from ..logging import get_logger
from ..stages.chunker import ChunkerStage
from ..stages.embedder import EmbedderStage
from ..stages.enricher import EnricherStage
from ..stages.retrieval import RetrievalStage
from ..stages.storage import StorageStage
from ..stages.walker import WalkerStage
from .registry import lookup_adapter

log = get_logger(__name__)


@dataclass
class BuiltStages:
    """Set of constructed adapters returned by :func:`build_stages`.

    All 6 stages are adapter-aware after Phase 3. The Pipeline / CLI
    consumes this dataclass directly — never reaches into the registry.
    """

    walker: WalkerStage
    chunker: ChunkerStage
    enricher: EnricherStage
    embedder: EmbedderStage
    storage: StorageStage
    retrieval: RetrievalStage
    resolved: dict[str, Any]

    def indexing_hash(self) -> str:
        """Stable hash of the four adapters that determine index contents.

        WIN-176: the Meta-Harness index cache (WIN-177) keys on this
        hash. Two builds that produce byte-identical index databases
        share the same indexing_hash; flipping any walker, chunker,
        enricher, or embedder adapter_id forces a fresh hash, which
        invalidates the cache.

        Storage and retrieval are **deliberately excluded**: changing
        from sqlite-vec to lancedb (storage) or from hybrid-rrf to a
        different fusion (retrieval) does NOT change what chunks are
        in the index, only how they're queried back. The Meta-Harness
        wants those to share the cache.
        """
        import hashlib

        parts = [
            f"walker={getattr(self.walker, 'adapter_id', '?')}",
            f"chunker={getattr(self.chunker, 'adapter_id', '?')}",
            f"enricher={getattr(self.enricher, 'adapter_id', '?')}",
            f"embedder={getattr(self.embedder, 'adapter_id', '?')}",
        ]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        # Short prefix is plenty — 12 hex chars = 48 bits ≈ 280 trillion
        # buckets. Plenty of headroom for tens of thousands of candidates.
        return digest[:12]


def build_stages(
    resolved_config: dict[str, Any],
    *,
    open_storage: bool = True,
    force_rebuild: bool = False,
) -> BuiltStages:
    """Instantiate every adapter named in ``resolved_config``.

    ``open_storage=False`` skips storage + retrieval construction and
    returns ``None`` for those slots. Useful for ``stropha pipeline show``
    when the user has not yet populated an index file (avoids opening a
    SQLite connection just to print config).

    ``force_rebuild=True`` is propagated to the storage adapter so it can
    drop+recreate dim-dependent tables before the consistency check
    rejects a dim shift (e.g. ``bge-m3`` 1024 → ``voyage-code-3`` 512).
    Caller is responsible for actually triggering ``Pipeline.run(rebuild=True)``
    after — passing this flag without the rebuild leaves the index empty.
    """
    walker = _build_simple(resolved_config, "walker")
    chunker = _build_simple(resolved_config, "chunker")
    enricher = _build_simple(resolved_config, "enricher")
    embedder = _build_simple(resolved_config, "embedder")

    storage: StorageStage | None = None
    retrieval: RetrievalStage | None = None
    if open_storage:
        storage_section = resolved_config.get("storage") or {}
        storage_name = storage_section.get("adapter") or "sqlite-vec"
        storage_cls = lookup_adapter("storage", storage_name)
        try:
            storage_cfg = storage_cls.Config(**(storage_section.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for pipeline.storage adapter={storage_name!r}: {exc}"
            ) from exc
        # Forward force_rebuild — only the sqlite-vec adapter honours it
        # today, but the kwarg lets every future StorageStage opt in.
        try:
            storage = storage_cls(
                storage_cfg,
                embedding_dim=embedder.dim,
                force_rebuild=force_rebuild,
            )
        except TypeError:
            # Adapter not yet aware of force_rebuild — fall back to the
            # original signature. The user gets the dim-mismatch error if
            # any (i.e. behaviour matches pre-flag world).
            storage = storage_cls(storage_cfg, embedding_dim=embedder.dim)
        log.info(
            "pipeline.adapter.built",
            stage="storage",
            adapter=storage_name,
            adapter_id=storage.adapter_id,
        )

        # Late inject storage into enrichers that need it (e.g. graph-aware
        # enricher uses graph_nodes table). Adapters opt-in via setattr or
        # by accepting `storage` in __init__.
        if hasattr(enricher, "_storage") and getattr(enricher, "_storage", None) is None:
            try:
                enricher._storage = storage  # type: ignore[attr-defined]
                log.info(
                    "pipeline.adapter.injected",
                    stage="enricher",
                    adapter=enricher.adapter_name,
                    dependency="storage",
                )
            except (AttributeError, TypeError):
                pass

        retrieval_section = resolved_config.get("retrieval") or {}
        retrieval_name = retrieval_section.get("adapter") or "hybrid-rrf"
        retrieval_cls = lookup_adapter("retrieval", retrieval_name)
        try:
            retrieval_cfg = retrieval_cls.Config(**(retrieval_section.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for pipeline.retrieval adapter={retrieval_name!r}: {exc}"
            ) from exc
        retrieval = retrieval_cls(retrieval_cfg, storage=storage, embedder=embedder)
        log.info(
            "pipeline.adapter.built",
            stage="retrieval",
            adapter=retrieval_name,
            adapter_id=retrieval.adapter_id,
        )

    return BuiltStages(
        walker=walker,
        chunker=chunker,
        enricher=enricher,
        embedder=embedder,
        storage=storage,  # type: ignore[arg-type]
        retrieval=retrieval,  # type: ignore[arg-type]
        resolved=resolved_config,
    )


def _build_simple(resolved_config: dict[str, Any], stage: str) -> Any:
    """Build a stage that needs no cross-stage dependency injection."""
    section = resolved_config.get(stage) or {}
    adapter_name = section.get("adapter")
    if not adapter_name:
        raise ConfigError(f"pipeline.{stage}.adapter is required")
    cls = lookup_adapter(stage, adapter_name)
    config_payload = section.get("config") or {}
    try:
        cfg_obj = cls.Config(**config_payload)
    except Exception as exc:
        raise ConfigError(
            f"Invalid config for pipeline.{stage} adapter={adapter_name!r}: {exc}"
        ) from exc
    try:
        instance = cls(cfg_obj)
    except TypeError:
        instance = cls()
    log.info(
        "pipeline.adapter.built",
        stage=stage,
        adapter=adapter_name,
        adapter_id=getattr(instance, "adapter_id", "?"),
    )
    return instance
