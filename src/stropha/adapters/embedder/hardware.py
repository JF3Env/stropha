"""Native hardware adapters composed with one shared embedder shell."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...pipeline.registry import register_adapter
from .composed import ComposedEmbedder


class HardwareEmbedderConfig(BaseModel):
    """Model and batching shared by native accelerator engines."""

    model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="SentenceTransformers-compatible Hugging Face model identifier.",
    )
    batch_size: int = Field(default=32, ge=1, le=512)
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings to unit length for cosine similarity.",
    )


@dataclass(frozen=True)
class NativeExecution:
    """Replaceable native runtime component selected by a thin adapter."""

    name: str
    backend: str
    device: str
    install_hint: str

    def ensure_available(self) -> None:
        if self.backend == "openvino":
            try:
                from openvino import Core  # type: ignore[import-untyped,import-not-found]
            except ImportError as exc:
                raise EmbeddingError(
                    f"{self.name} requires OpenVINO. {self.install_hint}"
                ) from exc
            devices = Core().available_devices
            if not any(device.upper().startswith("NPU") for device in devices):
                raise EmbeddingError(
                    f"{self.name} is unavailable; OpenVINO exposes "
                    f"{', '.join(devices) or 'no devices'}. {self.install_hint}"
                )
            return

        try:
            import torch  # type: ignore[import-untyped,import-not-found]
        except ImportError as exc:
            raise EmbeddingError(f"{self.name} requires PyTorch. {self.install_hint}") from exc
        if not torch.cuda.is_available():
            raise EmbeddingError(
                f"{self.name} is unavailable because PyTorch cannot access a GPU. "
                f"{self.install_hint}"
            )
        using_rocm = bool(getattr(torch.version, "hip", None))
        if self.backend == "rocm" and not using_rocm:
            raise EmbeddingError(
                f"AMD ROCm execution requires a ROCm PyTorch build. {self.install_hint}"
            )
        if self.backend == "cuda" and using_rocm:
            raise EmbeddingError(
                f"NVIDIA CUDA execution received a ROCm PyTorch build. {self.install_hint}"
            )


NVIDIA_EXECUTION = NativeExecution(
    name="NVIDIA CUDA / PyTorch",
    backend="cuda",
    device="cuda",
    install_hint="Run `uv sync --extra nvidia` with a CUDA-enabled PyTorch build.",
)

AMD_EXECUTION = NativeExecution(
    name="AMD ROCm / PyTorch",
    backend="rocm",
    device="cuda",  # PyTorch intentionally exposes ROCm through the cuda API.
    install_hint=(
        "Install the PyTorch build for your ROCm version, then run "
        "`uv sync --extra amd`."
    ),
)

INTEL_NPU_EXECUTION = NativeExecution(
    name="Intel NPU / OpenVINO",
    backend="openvino",
    device="NPU",
    install_hint="Run `uv sync --extra intel-npu` and install the Intel NPU driver.",
)


class SentenceTransformerEngine:
    """Native SentenceTransformers execution on CUDA, ROCm, or OpenVINO."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int,
        normalize: bool,
        execution: NativeExecution,
    ) -> None:
        execution.ensure_available()
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise EmbeddingError(
                f"{execution.name} requires SentenceTransformers. {execution.install_hint}"
            ) from exc
        try:
            self._model = SentenceTransformer(
                model,
                device=execution.device,
                backend="openvino" if execution.backend == "openvino" else "torch",
            )
        except Exception as exc:
            raise EmbeddingError(
                f"{execution.name} failed to load {model!r}: {exc}"
            ) from exc
        self._model_name = model
        self._batch_size = batch_size
        self._normalize = normalize
        self._execution = execution
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            dimension = len(self.embed_query("dimension probe"))
        self._dim = int(dimension)

    @property
    def name(self) -> str:
        return self._execution.name

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
            "backend": self._execution.backend,
            "device": self._execution.device,
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
            )
            return [list(vector) for vector in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"{self.name} document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            embeddings = self._model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
            )
            return list(embeddings[0])
        except Exception as exc:
            raise EmbeddingError(f"{self.name} query embedding failed: {exc}") from exc


class _HardwareEmbedder(ComposedEmbedder):
    """Thin adapter that injects one native engine into the shared shell."""

    Config = HardwareEmbedderConfig
    adapter: ClassVar[str]
    execution: ClassVar[NativeExecution]

    def __init__(self, config: HardwareEmbedderConfig | None = None) -> None:
        cfg = config or HardwareEmbedderConfig()
        super().__init__(
            adapter=self.adapter,
            config=cfg,
            batch_size=cfg.batch_size,
            engine=SentenceTransformerEngine(
                cfg.model,
                batch_size=cfg.batch_size,
                normalize=cfg.normalize,
                execution=self.execution,
            ),
        )

    @property
    def adapter_id(self) -> str:
        normalize = str(self._config.normalize).lower()  # type: ignore[attr-defined]
        return f"{super().adapter_id}:normalize={normalize}"


@register_adapter(stage="embedder", name="nvidia")
class NvidiaEmbedder(_HardwareEmbedder):
    """SentenceTransformers using native NVIDIA CUDA PyTorch execution."""

    adapter = "nvidia"
    execution = NVIDIA_EXECUTION


@register_adapter(stage="embedder", name="amd")
class AmdEmbedder(_HardwareEmbedder):
    """SentenceTransformers using native AMD ROCm PyTorch execution."""

    adapter = "amd"
    execution = AMD_EXECUTION


@register_adapter(stage="embedder", name="intel-npu")
class IntelNpuEmbedder(_HardwareEmbedder):
    """SentenceTransformers using native Intel NPU OpenVINO execution."""

    adapter = "intel-npu"
    execution = INTEL_NPU_EXECUTION
