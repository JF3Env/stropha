"""Model2Vec static embedder — ultra-fast, no forward pass.

Uses Model2Vec library with static token embeddings (lookup + pooling).
Default model: ``minishlab/potion-code-16M`` — 256-d, ~16M params, optimized for code.

Key benefits over transformer-based embedders:
- ~100x faster inference (table lookup vs. forward pass)
- ~4x smaller vectors (256-d vs. 1024-d)
- Lower memory footprint

Install: ``pip install model2vec`` or ``uv add model2vec``
Optional fast mode: ``pip install model2vec[fast]`` (uses optimized tokenizer)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.embedder import EmbedderStage

log = get_logger(__name__)


class Model2VecEmbedderConfig(BaseModel):
    """Config schema for the Model2Vec static embedder adapter."""

    model: str = Field(
        default="minishlab/potion-code-16M",
        description="HuggingFace model id for Model2Vec. Recommended: potion-code-16M for code.",
    )
    batch_size: int = Field(default=256, ge=1, le=4096)
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings to unit length (recommended for cosine similarity).",
    )


@register_adapter(stage="embedder", name="model2vec")
class Model2VecEmbedder(EmbedderStage):
    """Static embedding lookup — no transformer forward pass.
    
    Model2Vec uses pre-computed token embeddings and pools them for each text.
    This is ~100x faster than transformer inference but may have slightly
    lower quality for complex semantic queries. Excellent for code search
    where exact token matching matters.
    """

    Config = Model2VecEmbedderConfig

    def __init__(self, config: Model2VecEmbedderConfig | None = None) -> None:
        if config is None:
            config = Model2VecEmbedderConfig()
        self._config = config
        # ``StaticModel`` ships without type stubs, so ``_model_obj`` lives as
        # ``Any``. ``_load_model`` is invoked from ``__init__`` below and will
        # either populate this slot or raise — there is no observable ``None``
        # window after construction.
        self._model_obj: Any = None
        self._dim: int = 0
        
        # Lazy load to avoid import overhead when not used
        self._load_model()

    def _load_model(self) -> None:
        """Load the Model2Vec model."""
        try:
            from model2vec import StaticModel  # type: ignore[import-untyped,import-not-found]
        except ImportError as exc:
            raise EmbeddingError(
                "model2vec package is not installed. "
                "Run `pip install model2vec` or `uv add model2vec`. "
                "For faster tokenization: `pip install model2vec[fast]`"
            ) from exc
        
        try:
            self._model_obj = StaticModel.from_pretrained(self._config.model)
        except Exception as exc:
            raise EmbeddingError(
                f"Model2Vec failed to load {self._config.model!r}: {exc}"
            ) from exc
        
        # Probe to get dimensionality
        probe_embedding = self._model_obj.encode(["probe"])
        self._dim = probe_embedding.shape[1]
        
        log.info(
            "model2vec_embedder.ready",
            model=self._config.model,
            dim=self._dim,
            normalize=self._config.normalize,
        )

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "model2vec"

    @property
    def adapter_id(self) -> str:
        return f"model2vec:{self._config.model}:{self._dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return Model2VecEmbedderConfig

    def health(self) -> StageHealth:
        if self._model_obj is None:
            return StageHealth(
                status="error",
                message="Model not loaded",
                detail={"model": self._config.model},
            )
        return StageHealth(
            status="ready",
            message=f"Model2Vec static embedder loaded ({self._config.model}, dim={self._dim})",
            detail={
                "model": self._config.model,
                "dim": str(self._dim),
                "normalize": str(self._config.normalize),
            },
        )

    # ----- Embedder surface ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents using static token lookup + pooling."""
        if not texts:
            return []
        
        try:
            # Model2Vec encode returns numpy array
            embeddings = self._model_obj.encode(
                list(texts),
                normalize=self._config.normalize,
            )
            return [list(v) for v in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"Model2Vec embed failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        try:
            embedding = self._model_obj.encode(
                [text],
                normalize=self._config.normalize,
            )
            return list(embedding[0])
        except Exception as exc:
            raise EmbeddingError(f"Model2Vec query embed failed: {exc}") from exc
