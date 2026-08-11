"""bge-m3 embedder adapter — pre-configured native MLX backend.

``BAAI/bge-m3`` is a multilingual, multi-functional embedder (dense +
sparse + colbert) widely cited as the top open-source choice for code
retrieval beyond English-only models like ``mxbai-embed-large-v1``.

This adapter exists as a thin pre-configured wrapper around
:class:`stropha.adapters.embedder.mlx.MlxEmbedder` so Apple Silicon users can opt
in via ``--embedder bge-m3`` (or ``pipeline.embedder.adapter: bge-m3``)
without remembering the MLX registry alias. FastEmbed does not support
``BAAI/bge-m3``; routing through the existing MLX backend keeps the adapter
truthful instead of advertising an unusable ONNX model.

Per spec §15 / Phase 4: "Modelo de embedding self-hosted (bge-m3) como
fallback offline." This adapter delivers that surface.

Requires Apple Silicon and the ``mlx-embed`` optional dependency. The first
use downloads the model; subsequent calls run locally.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...pipeline.registry import register_adapter
from .mlx import MlxEmbedder, MlxEmbedderConfig, _is_mlx_available


class BgeM3Config(BaseModel):
    """bge-m3 is multilingual + multi-function; defaults tuned for code RAG."""

    model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "BGE-M3 model identifier. Both `BAAI/bge-m3` and the MLX alias "
            "`bge-m3` select the same model."
        ),
    )
    batch_size: int = Field(
        default=24, ge=1, le=256,
        description=(
            "bge-m3 is bigger than mxbai — smaller batches keep memory "
            "bounded on machines without a discrete GPU."
        ),
    )


@register_adapter(stage="embedder", name="bge-m3")
class BgeM3Embedder(MlxEmbedder):
    """MLX embedder pinned to the ``BAAI/bge-m3`` weights."""

    Config = BgeM3Config  # type: ignore[assignment]

    def __init__(self, config: BgeM3Config | None = None) -> None:
        cfg = config or BgeM3Config()
        if cfg.model not in {"BAAI/bge-m3", "bge-m3"}:
            raise ValueError("bge-m3 adapter model must be 'BAAI/bge-m3' or 'bge-m3'")
        if not _is_mlx_available():
            from ...errors import EmbeddingError

            raise EmbeddingError(
                "bge-m3 is not supported by FastEmbed. On Apple Silicon, install "
                "the native backend with `uv sync --extra mlx-embed`; on other "
                "platforms select a FastEmbed-supported `local` model."
            )
        self._bge_model = "BAAI/bge-m3"
        super().__init__(MlxEmbedderConfig(model="bge-m3", batch_size=cfg.batch_size))

    @property
    def adapter_name(self) -> str:
        return "bge-m3"

    @property
    def adapter_id(self) -> str:
        # Keep `bge-m3:` prefix so drift detection treats a switch from
        # `local:...` to `bge-m3:...` (even with the same underlying model
        # weights) as a real adapter change.
        return f"bge-m3:{self._bge_model}:{self.dim}"
