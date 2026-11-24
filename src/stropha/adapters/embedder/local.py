"""Local fastembed embedder (ONNX, no torch, no network).

Default model per ADR-008 of ``docs/architecture/stropha-graphify-integration.md``:
``mixedbread-ai/mxbai-embed-large-v1`` — top open-source English MTEB at
the ~1 GB scale, 1024-dim, stable on macOS aarch64.

Avoid ``jinaai/jina-embeddings-v2-base-code`` here — it has an ONNX-runtime
instability on aarch64 that hangs the process on the second consecutive
embed call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ...cpu import available_cpus
from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.registry import register_adapter
from .composed import ComposedEmbedder

log = get_logger(__name__)

OnnxProvider = str | tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class OnnxExecutionProvider:
    """Replaceable hardware component for the shared FastEmbed implementation."""

    name: str
    providers: tuple[OnnxProvider, ...]
    install_hint: str

    @property
    def required_provider(self) -> str:
        provider = self.providers[0]
        return provider[0] if isinstance(provider, tuple) else provider

    def ensure_available(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - FastEmbed normally provides it
            raise EmbeddingError("ONNX Runtime is not installed. Run `uv sync --extra local`.") from exc
        available = ort.get_available_providers()
        if self.required_provider not in available:
            raise EmbeddingError(
                f"{self.name} execution requires {self.required_provider}, but ONNX Runtime "
                f"only exposes {', '.join(available) or 'no providers'}. {self.install_hint}"
            )


CPU_EXECUTION = OnnxExecutionProvider(
    name="CPU",
    providers=("CPUExecutionProvider",),
    install_hint="Install the standard ONNX Runtime package.",
)


class FastEmbedEngine:
    """FastEmbed engine parameterized by one ONNX execution provider."""

    def __init__(
        self,
        model: str,
        execution: OnnxExecutionProvider,
        *,
        threads: int | None = None,
    ) -> None:
        self._model_name = model
        self._execution = execution
        # ONNX Runtime defaults to one thread per host core, which inside a
        # container oversubscribes the CPU quota and spends most time throttled.
        self._threads = threads if threads is not None else available_cpus()
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "fastembed package is not installed. Run `uv sync`."
            ) from exc
        execution.ensure_available()
        try:
            self._model = TextEmbedding(
                model_name=model,
                providers=list(execution.providers),
                threads=self._threads,
            )
        except Exception as exc:
            raise EmbeddingError(f"fastembed failed to load {model!r}: {exc}") from exc
        probe = next(iter(self._model.embed(["probe"])))
        self._dim = len(list(probe))
        log.info(
            "fastembed.ready",
            model=model,
            dim=self._dim,
            execution=execution.name,
            threads=self._threads,
        )

    @property
    def name(self) -> str:
        return f"FastEmbed / {self._execution.name}"

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
            "execution": self._execution.name,
            "provider": self._execution.required_provider,
            "threads": str(self._threads),
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return [list(v) for v in self._model.embed(list(texts))]
        except Exception as exc:
            raise EmbeddingError(f"FastEmbed document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return list(next(iter(self._model.query_embed([text]))))
        except Exception as exc:
            raise EmbeddingError(f"FastEmbed query embedding failed: {exc}") from exc


class LocalEmbedderConfig(BaseModel):
    """Config schema for the local fastembed adapter."""

    model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="HuggingFace model id supported by fastembed.",
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    threads: int | None = Field(
        default=None,
        ge=1,
        description=(
            "ONNX Runtime threads. Default: the CPUs this process may use "
            "(cgroup quota and affinity), not every core of the host. "
            "Does not change the embeddings, so it is not part of adapter_id."
        ),
    )


@register_adapter(stage="embedder", name="local")
class LocalEmbedder(ComposedEmbedder):
    """ONNX-based embedder, runs on CPU."""

    Config = LocalEmbedderConfig

    def __init__(
        self,
        config: LocalEmbedderConfig | None = None,
        *,
        model: str | None = None,
        execution: OnnxExecutionProvider = CPU_EXECUTION,
        adapter: str = "local",
    ) -> None:
        # Allow legacy keyword form ``LocalEmbedder(model="...")`` for back-compat
        # with code that hasn't migrated to passing a Config yet.
        if config is None:
            config = LocalEmbedderConfig(
                model=model or LocalEmbedderConfig.model_fields["model"].default
            )
        elif model is not None:
            raise TypeError("Pass either config= or model=, not both")
        super().__init__(
            adapter=adapter,
            config=config,
            batch_size=config.batch_size,
            engine=FastEmbedEngine(config.model, execution, threads=config.threads),
        )
