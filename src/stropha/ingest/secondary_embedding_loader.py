"""Incremental backfill for the independent secondary vector space."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ..errors import ConfigError, EmbeddingError
from ..logging import get_logger
from ..secondary_embeddings import (
    SECONDARY_EMBEDDING_DIM,
    SecondaryEmbeddingVector,
)

if TYPE_CHECKING:
    from ..stages.embedder import EmbedderStage
    from ..stages.storage import StorageStage

log = get_logger(__name__)


@dataclass(frozen=True)
class SecondaryEmbeddingLoadResult:
    embedded: int
    skipped_already_current: int
    skipped_stale: int
    duration_ms: int
    embedder_id: str


class SecondaryEmbeddingLoader:
    """Backfill only chunks missing the active secondary identity/content hash."""

    def __init__(self, storage: StorageStage, embedder: EmbedderStage) -> None:
        if embedder.dim != SECONDARY_EMBEDDING_DIM:
            raise ConfigError(
                f"Secondary embedder requires dim={SECONDARY_EMBEDDING_DIM}, "
                f"got {embedder.dim}"
            )
        self._storage = storage
        self._embedder = embedder

    def load(self, *, batch_size: int | None = None) -> SecondaryEmbeddingLoadResult:
        start = datetime.now(UTC)
        size = batch_size or self._embedder.batch_size
        if size < 1:
            raise ValueError("batch_size must be >= 1")

        embedded = 0
        skipped_stale = 0
        while True:
            pending = self._storage.list_chunks_needing_secondary_embeddings(
                self._embedder.adapter_id,
                embedding_dim=SECONDARY_EMBEDDING_DIM,
                limit=size,
            )
            if not pending:
                break

            vectors = self._embedder.embed_documents([item["content"] for item in pending])
            if len(vectors) != len(pending):
                raise EmbeddingError(
                    "Secondary embedder returned "
                    f"{len(vectors)} vectors for {len(pending)} chunks"
                )
            bad_dim = next(
                (len(vector) for vector in vectors if len(vector) != SECONDARY_EMBEDDING_DIM),
                None,
            )
            if bad_dim is not None:
                raise EmbeddingError(
                    f"Secondary embedder returned dim={bad_dim}, "
                    f"expected {SECONDARY_EMBEDDING_DIM}"
                )

            payload: list[SecondaryEmbeddingVector] = [
                {
                    "chunk_id": item["chunk_id"],
                    "content_hash": item["content_hash"],
                    "embedding": vector,
                }
                for item, vector in zip(pending, vectors, strict=True)
            ]
            upserted = self._storage.upsert_secondary_embeddings(
                payload,
                self._embedder.adapter_id,
                embedding_dim=SECONDARY_EMBEDDING_DIM,
            )
            self._storage.commit()
            embedded += upserted
            skipped_stale += len(pending) - upserted
            if upserted == 0:
                break

        current = self._storage.count_secondary_embeddings(
            self._embedder.adapter_id,
            embedding_dim=SECONDARY_EMBEDDING_DIM,
        )
        result = SecondaryEmbeddingLoadResult(
            embedded=embedded,
            skipped_already_current=max(0, current - embedded),
            skipped_stale=skipped_stale,
            duration_ms=int((datetime.now(UTC) - start).total_seconds() * 1000),
            embedder_id=self._embedder.adapter_id,
        )
        log.info(
            "secondary_embedding_loader.done",
            embedded=result.embedded,
            skipped_already_current=result.skipped_already_current,
            skipped_stale=result.skipped_stale,
            duration_ms=result.duration_ms,
            embedder_id=result.embedder_id,
        )
        return result
