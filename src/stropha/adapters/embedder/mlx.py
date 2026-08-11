"""MLX-based embedder for Apple Silicon GPU acceleration.

Uses ``mlx-embedding-models`` for registry-backed BERT/RoBERTa models and
``mlx-lm`` for MLX-converted Qwen3 embedding checkpoints.

Supported models (from registry):
- mixedbread-large: mxbai-embed-large-v1, 1024-dim (same as local default)
- bge-m3: BAAI/bge-m3, 1024-dim, multilingual, 8K context
- bge-large: BAAI/bge-large-en-v1.5, 1024-dim
- nomic-text-v1.5: nomic-ai/nomic-embed-text-v1.5, 768-dim, 2K context
- bge-small: BAAI/bge-small-en-v1.5, 384-dim (fast)

Install dependency:
    uv sync --extra mlx-embed

Performance: ~3-5x faster than fastembed ONNX on M1/M2/M3 Macs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.registry import register_adapter
from .composed import ComposedEmbedder

log = get_logger(__name__)

# Models from mlx-embedding-models registry with their dimensions
_MLX_MODEL_REGISTRY = {
    "mixedbread-large": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
    "snowflake-lg": 1024,
    "nomic-text-v1.5": 768,
    "nomic-text-v1": 768,
    "bge-base": 768,
    "bge-small": 384,
    "minilm-l12": 384,
    "minilm-l6": 384,
    "gte-tiny": 384,
    "bge-micro": 384,
}

_MLX_MODEL_ALIASES = {
    "BAAI/bge-m3": "bge-m3",
    "BAAI/bge-large-en-v1.5": "bge-large",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "BAAI/bge-small-en-v1.5": "bge-small",
    "mixedbread-ai/mxbai-embed-large-v1": "mixedbread-large",
    "nomic-ai/nomic-embed-text-v1": "nomic-text-v1",
    "nomic-ai/nomic-embed-text-v1.5": "nomic-text-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
    "sentence-transformers/all-MiniLM-L12-v2": "minilm-l12",
}

_QWEN3_QUERY_INSTRUCTION = (
    "Given a code search query, retrieve relevant code passages that answer the query"
)
_QWEN3_MAX_LENGTH = 8192
_QWEN3_CACHE_LIMIT_BYTES = 512 * 1024 * 1024


def resolve_mlx_model(model: str) -> str:
    """Resolve a canonical Hugging Face ID to the MLX registry alias."""
    return _MLX_MODEL_ALIASES.get(model, model)


def _is_qwen3_embedding_model(model: str) -> bool:
    """Return whether ``model`` names an MLX-converted Qwen3 embedding checkpoint."""
    return model.rsplit("/", 1)[-1].lower().startswith("qwen3-embedding-")


class MlxEmbedderConfig(BaseModel):
    """Config schema for the MLX embedder adapter."""

    model: str = Field(
        default="mixedbread-large",
        description=(
            "Model name from mlx-embedding-models registry or an MLX-converted "
            "Qwen3-Embedding checkpoint. "
            "Canonical Hugging Face IDs such as `BAAI/bge-m3` are accepted and "
            "normalized to registry aliases. Qwen metadata is validated when saved; "
            "weights are loaded on the first embedding batch during indexing."
        ),
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Batch size for embedding. MLX can handle larger batches efficiently.",
    )


def _is_mlx_available() -> bool:
    """Check if MLX and at least one supported model runtime are available."""
    try:
        import mlx.core  # noqa: F401  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        return False
    try:
        import mlx_embedding_models  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        try:
            import mlx_lm  # noqa: F401  # type: ignore[import-untyped]
        except ImportError:
            return False
    return True


class MlxEngine:
    """Native Apple Silicon engine injected into ``ComposedEmbedder``."""

    def __init__(self, model: str) -> None:
        resolved_model = resolve_mlx_model(model)
        if resolved_model in _MLX_MODEL_REGISTRY:
            self._load_registry_model(model, resolved_model)
        elif _is_qwen3_embedding_model(resolved_model):
            self._prepare_qwen3_model(resolved_model)
        else:
            available = ", ".join(sorted(_MLX_MODEL_REGISTRY.keys()))
            raise EmbeddingError(
                f"Unknown MLX model {model!r}. Available registry models: {available}. "
                "MLX-converted Qwen3-Embedding checkpoints are also supported."
            )
        log.info(
            "mlx_embedder.ready",
            model=self._model_name,
            configured_model=model,
            dim=self._dim,
            runtime=self._runtime,
            loaded=self._model is not None,
        )

    def _load_registry_model(self, configured_model: str, resolved_model: str) -> None:
        try:
            from mlx_embedding_models.embedding import (  # type: ignore[import-untyped]
                EmbeddingModel,
            )
        except ImportError as exc:
            raise EmbeddingError(
                "mlx-embedding-models package is not installed. "
                "Install with: `uv sync --extra mlx-embed`"
            ) from exc
        try:
            self._model = EmbeddingModel.from_registry(resolved_model)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load MLX model {configured_model!r}: {exc}"
            ) from exc
        self._model_name = resolved_model
        self._dim = _MLX_MODEL_REGISTRY[resolved_model]
        self._runtime = "mlx-embedding-models"
        self._tokenizer: Any = None

    def _prepare_qwen3_model(self, model: str) -> None:
        try:
            from huggingface_hub import (  # type: ignore[import-untyped]
                hf_hub_download,
                try_to_load_from_cache,
            )
        except ImportError as exc:
            raise EmbeddingError(
                "huggingface-hub is required to validate Qwen3 checkpoints. "
                "Install with: `uv sync --extra mlx-embed`"
            ) from exc
        try:
            cached = try_to_load_from_cache(model, "config.json")
            config_path = cached if isinstance(cached, str) else hf_hub_download(
                model,
                filename="config.json",
            )
            model_config = json.loads(Path(config_path).read_text())
            if model_config.get("model_type") != "qwen3":
                raise ValueError("checkpoint config is not a Qwen3 model")
            hidden_size = model_config.get("hidden_size")
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise ValueError("model config does not define a positive hidden_size")
        except Exception as exc:
            raise EmbeddingError(f"Could not validate MLX Qwen3 model {model!r}: {exc}") from exc

        self._model = None
        self._tokenizer = None
        self._model_name = model
        self._dim = hidden_size
        self._runtime = "mlx-lm"
        self._load_lock = Lock()

    def _ensure_qwen3_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load_qwen3_weights()

    def _load_qwen3_weights(self) -> None:
        try:
            import mlx.core as mx  # type: ignore[import-untyped,import-not-found]
            from mlx_lm import load  # type: ignore[import-untyped]
        except ImportError as exc:
            raise EmbeddingError(
                "mlx-lm package is required for Qwen3 embedding checkpoints. "
                "Install with: `uv sync --extra mlx-embed`"
            ) from exc
        started = perf_counter()
        log.info(
            "mlx_qwen3.load_started",
            model=self._model_name,
            phase="download_or_materialize_weights",
        )
        try:
            mx.set_cache_limit(_QWEN3_CACHE_LIMIT_BYTES)
            loaded: Any = load(self._model_name)
            self._model, self._tokenizer = loaded[:2]
            hidden_size = getattr(getattr(self._model, "args", None), "hidden_size", None)
            if hidden_size != self._dim:
                raise ValueError(
                    f"loaded hidden_size {hidden_size!r} does not match validated dimension {self._dim}"
                )
            if getattr(self._model, "model", None) is None:
                raise ValueError("model does not expose its final hidden states")
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            mx.clear_cache()
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise EmbeddingError(
                f"Failed to load MLX Qwen3 model {self._model_name!r}: {exc}"
            ) from exc
        log.info(
            "mlx_qwen3.load_finished",
            model=self._model_name,
            dim=self._dim,
            elapsed_s=round(perf_counter() - started, 2),
        )

    @property
    def name(self) -> str:
        return "Apple MLX"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def health_detail(self) -> dict[str, str]:
        return {
            "model": self.model_name,
            "dim": str(self.dim),
            "execution": "MLX",
            "runtime": self._runtime,
            "weights_loaded": str(self._model is not None).lower(),
        }

    def _truncate_to_tokens(self, texts: list[str], max_tokens: int = 508) -> list[str]:
        """Truncate texts to fit within MLX embedding model's token limit.

        The mlx-embedding-models library has a max sequence length of 512 tokens.
        We use 508 to leave room for special tokens ([CLS], [SEP]) and any
        tokenizer quirks. The tokenizer's truncate=True with max_length should
        ensure the result fits.
        """
        tokenizer = self._model.tokenizer
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_tokens,
            padding=False,
            return_offsets_mapping=False,
        )

        # Decode back to text - this gives us properly truncated strings
        truncated = []
        for _i, input_ids in enumerate(encoded["input_ids"]):
            # Skip special tokens when decoding
            decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
            truncated.append(decoded)

        return truncated

    def _embed_qwen3(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_qwen3_loaded()
        try:
            import mlx.core as mx  # type: ignore[import-untyped,import-not-found]

            try:
                tokenizer = self._tokenizer
                if not callable(tokenizer):
                    tokenizer = getattr(tokenizer, "_tokenizer", None)
                if not callable(tokenizer):
                    raise TypeError("loaded tokenizer does not expose a callable HF tokenizer")
                encoded = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=_QWEN3_MAX_LENGTH,
                    return_tensors="np",
                )
                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])
                hidden_states = self._model.model(input_ids)

                positions = mx.arange(attention_mask.shape[1])
                last_indices = mx.max(mx.where(attention_mask != 0, positions, -1), axis=1)
                batch_indices = mx.arange(hidden_states.shape[0])
                pooled = hidden_states[batch_indices, last_indices]
                norms = mx.linalg.norm(pooled, axis=1, keepdims=True)
                normalized = pooled / mx.maximum(norms, 1e-12)
                mx.eval(normalized)
                return normalized.tolist()
            finally:
                mx.clear_cache()
        except Exception as exc:
            raise EmbeddingError(f"MLX Qwen3 embed failed: {exc}") from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple documents using MLX GPU acceleration."""
        if self._runtime == "mlx-lm":
            return self._embed_qwen3(texts)
        try:
            truncated = self._truncate_to_tokens(list(texts))
            embeddings = self._model.encode(truncated)
            return [list(emb) for emb in embeddings]
        except Exception as exc:
            log.error(
                "mlx_embed.failed",
                batch_size=len(texts),
                text_lengths=[len(t) for t in texts],
                error=str(exc),
            )
            raise EmbeddingError(f"MLX embed failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using MLX GPU acceleration."""
        if self._runtime == "mlx-lm":
            instructed = f"Instruct: {_QWEN3_QUERY_INSTRUCTION}\nQuery:{text}"
            return self._embed_qwen3([instructed])[0]
        try:
            truncated = self._truncate_to_tokens([text])[0]
            embeddings = self._model.encode([truncated])
            return list(embeddings[0])
        except Exception as exc:
            raise EmbeddingError(f"MLX query embed failed: {exc}") from exc


@register_adapter(stage="embedder", name="mlx")
class MlxEmbedder(ComposedEmbedder):
    """MLX adapter composed with the native Apple Silicon engine."""

    Config = MlxEmbedderConfig

    def __init__(self, config: MlxEmbedderConfig | None = None) -> None:
        cfg = config or MlxEmbedderConfig()
        super().__init__(
            adapter="mlx",
            config=cfg,
            batch_size=cfg.batch_size,
            engine=MlxEngine(cfg.model),
        )
