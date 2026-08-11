"""Shared embedder adapter shell composed with a hardware-specific engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from ...pipeline.base import StageHealth
from ...stages.embedder import EmbedderStage


class EmbeddingEngine(Protocol):
    """Specialized execution component injected into a registered adapter."""

    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def health_detail(self) -> dict[str, str]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ComposedEmbedder(EmbedderStage):
    """Adapter protocol implementation shared by all local hardware engines."""

    Config: type[BaseModel]

    def __init__(
        self,
        *,
        adapter: str,
        config: BaseModel,
        batch_size: int,
        engine: EmbeddingEngine,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._batch_size = batch_size
        self._engine = engine

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return self._adapter

    @property
    def adapter_id(self) -> str:
        return f"{self.adapter_name}:{self.model_name}:{self.dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return self.Config

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=(
                f"{self.engine_name} embedder ready "
                f"({self.model_name}, dim={self.dim})"
            ),
            detail=self._engine.health_detail,
        )

    @property
    def engine_name(self) -> str:
        return self._engine.name

    @property
    def model_name(self) -> str:
        return self._engine.model_name

    @property
    def dim(self) -> int:
        return self._engine.dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._engine.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._engine.embed_query(text)
