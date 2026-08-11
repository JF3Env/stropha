"""Independent Model2Vec cosine retrieval over secondary vectors."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter
from ....secondary_embeddings import SECONDARY_EMBEDDING_DIM
from ....stages.embedder import EmbedderStage
from ....stages.storage import StorageStage
from ...embedder.model2vec import Model2VecEmbedderConfig


class Model2VecCosineConfig(Model2VecEmbedderConfig):
    """Stream and secondary-embedder settings kept in one config block."""

    k: int = Field(default=50, ge=1, le=500)
    dim: Literal[256] = 256
    normalize: Literal[True] = True


@register_adapter(stage="retrieval-stream", name="model2vec-cosine")
class Model2VecCosineStream:
    Config = Model2VecCosineConfig

    def __init__(
        self,
        config: Model2VecCosineConfig | None = None,
        *,
        storage: StorageStage | None = None,
        secondary_embedder: EmbedderStage | None = None,
    ) -> None:
        if storage is None or secondary_embedder is None:
            raise ValueError(
                "Model2VecCosineStream requires storage= and secondary_embedder="
            )
        self._config = config or Model2VecCosineConfig()
        if secondary_embedder.dim != SECONDARY_EMBEDDING_DIM:
            raise ValueError(
                f"model2vec-cosine requires dim={SECONDARY_EMBEDDING_DIM}, "
                f"got {secondary_embedder.dim}"
            )
        self._storage = storage
        self._embedder = secondary_embedder

    @property
    def stage_name(self) -> str:
        return "retrieval-stream"

    @property
    def adapter_name(self) -> str:
        return "model2vec-cosine"

    @property
    def adapter_id(self) -> str:
        return f"model2vec-cosine:k={self._config.k}:embedder={self._embedder.adapter_id}"

    @property
    def config_schema(self) -> type[Model2VecCosineConfig]:
        return Model2VecCosineConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"model2vec-cosine (k={self._config.k})",
            detail={"embedder": self._embedder.adapter_id},
        )

    @property
    def k(self) -> int:
        return self._config.k

    def search(
        self,
        query: str,
        query_vec: list[float] | None,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        del query_vec  # The primary vector may be HyDE-rewritten and is a different space.
        query_vec = self._embedder.embed_query(query)
        if repo_id is None:
            return self._storage.search_secondary_embeddings(
                query_vec,
                self._embedder.adapter_id,
                embedding_dim=SECONDARY_EMBEDDING_DIM,
                k=self._config.k,
            )
        return self._storage.search_secondary_embeddings(
            query_vec,
            self._embedder.adapter_id,
            embedding_dim=SECONDARY_EMBEDDING_DIM,
            k=self._config.k,
            repo_id=repo_id,
        )
