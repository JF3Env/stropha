"""Tests for composed hardware-specific embedding adapters."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stropha.adapters.embedder.composed import ComposedEmbedder
from stropha.adapters.embedder.hardware import (
    AMD_EXECUTION,
    INTEL_NPU_EXECUTION,
    NVIDIA_EXECUTION,
    AmdEmbedder,
    HardwareEmbedderConfig,
    IntelNpuEmbedder,
    NvidiaEmbedder,
)
from stropha.adapters.embedder.mlx import (
    MlxEmbedder,
    MlxEmbedderConfig,
    MlxEngine,
    resolve_mlx_model,
)
from stropha.errors import EmbeddingError
from stropha.pipeline.registry import lookup_adapter


class _TokenizerWrapper:
    """Mirror mlx-lm's non-callable wrapper around a callable HF tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)

    def __getattr__(self, name: str) -> object:
        return getattr(self._tokenizer, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._tokenizer, name, value)


@pytest.mark.parametrize(
    ("name", "adapter"),
    [("nvidia", NvidiaEmbedder), ("amd", AmdEmbedder), ("intel-npu", IntelNpuEmbedder)],
)
def test_hardware_embedder_registered(name: str, adapter: type[ComposedEmbedder]) -> None:
    assert lookup_adapter("embedder", name) is adapter


@pytest.mark.parametrize(
    ("adapter", "execution"),
    [
        (NvidiaEmbedder, NVIDIA_EXECUTION),
        (AmdEmbedder, AMD_EXECUTION),
        (IntelNpuEmbedder, INTEL_NPU_EXECUTION),
    ],
)
def test_hardware_embedder_injects_only_execution_component(
    adapter: type[ComposedEmbedder],
    execution: object,
) -> None:
    engine = MagicMock()
    with patch(
        "stropha.adapters.embedder.hardware.SentenceTransformerEngine",
        return_value=engine,
    ) as engine_cls:
        instance = adapter(
            HardwareEmbedderConfig(model="BAAI/bge-small-en-v1.5", batch_size=8)
        )

    engine_cls.assert_called_once_with(
        "BAAI/bge-small-en-v1.5",
        batch_size=8,
        normalize=True,
        execution=execution,
    )
    assert instance.engine_name == engine.name


def test_execution_provider_reports_actionable_missing_runtime() -> None:
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(EmbeddingError, match="cannot access a GPU"),
    ):
        NVIDIA_EXECUTION.ensure_available()


def test_mlx_is_a_composed_peer_adapter() -> None:
    engine = MagicMock()
    engine.name = "Apple MLX"
    with patch("stropha.adapters.embedder.mlx.MlxEngine", return_value=engine) as engine_cls:
        instance = MlxEmbedder(MlxEmbedderConfig(model="bge-m3", batch_size=16))

    engine_cls.assert_called_once_with("bge-m3")
    assert isinstance(instance, ComposedEmbedder)
    assert instance.adapter_name == "mlx"
    assert instance.batch_size == 16


def test_mlx_resolves_canonical_bge_m3_model_id() -> None:
    assert resolve_mlx_model("BAAI/bge-m3") == "bge-m3"
    assert resolve_mlx_model("bge-m3") == "bge-m3"


def test_mlx_qwen3_uses_hidden_states_last_token_pooling_and_normalization(tmp_path) -> None:
    document_hidden_states = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [9.0, 9.0, 9.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )
    query_hidden_states = np.array(
        [[[1.0, 1.0, 0.0], [0.0, 3.0, 4.0], [8.0, 0.0, 0.0]]]
    )
    backbone = MagicMock(side_effect=[document_hidden_states, query_hidden_states])
    model = SimpleNamespace(args=SimpleNamespace(hidden_size=3), model=backbone)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "</s>"
    tokenizer.side_effect = [
        {
            "input_ids": np.array([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": np.array([[1, 1, 0], [1, 1, 1]]),
        },
        {
            "input_ids": np.array([[6, 7, 8]]),
            "attention_mask": np.array([[1, 1, 1]]),
        },
    ]
    tokenizer_wrapper = _TokenizerWrapper(tokenizer)

    mlx_core = ModuleType("mlx.core")
    mlx_core.array = np.asarray  # type: ignore[attr-defined]
    mlx_core.arange = np.arange  # type: ignore[attr-defined]
    mlx_core.max = np.max  # type: ignore[attr-defined]
    mlx_core.where = np.where  # type: ignore[attr-defined]
    mlx_core.maximum = np.maximum  # type: ignore[attr-defined]
    mlx_core.eval = lambda _value: None  # type: ignore[attr-defined]
    mlx_core.linalg = SimpleNamespace(norm=np.linalg.norm)  # type: ignore[attr-defined]
    mlx_core.set_cache_limit = MagicMock()  # type: ignore[attr-defined]
    mlx_core.clear_cache = MagicMock()  # type: ignore[attr-defined]
    mlx_package = ModuleType("mlx")
    mlx_package.core = mlx_core  # type: ignore[attr-defined]
    mlx_lm = ModuleType("mlx_lm")
    load = MagicMock(return_value=(model, tokenizer_wrapper))
    mlx_lm.load = load  # type: ignore[attr-defined]
    model_config = tmp_path / "config.json"
    model_config.write_text('{"model_type": "qwen3", "hidden_size": 3}')
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.try_to_load_from_cache = MagicMock(  # type: ignore[attr-defined]
        return_value=str(model_config)
    )
    huggingface_hub.hf_hub_download = MagicMock()  # type: ignore[attr-defined]

    modules = {
        "huggingface_hub": huggingface_hub,
        "mlx": mlx_package,
        "mlx.core": mlx_core,
        "mlx_lm": mlx_lm,
    }
    with patch.dict(sys.modules, modules):
        engine = MlxEngine("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
        load.assert_not_called()
        assert engine.health_detail["weights_loaded"] == "false"
        documents = engine.embed_documents(["short", "longer"])
        query = engine.embed_query("where is request tracing implemented?")

    load.assert_called_once_with("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
    assert engine.model_name == "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    assert engine.dim == 3
    assert engine.health_detail["runtime"] == "mlx-lm"
    assert engine.health_detail["weights_loaded"] == "true"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
    np.testing.assert_allclose(documents, [[0.0, 0.6, 0.8], [1.0, 0.0, 0.0]])
    assert query == pytest.approx([1.0, 0.0, 0.0])
    query_text = tokenizer.call_args_list[1].args[0][0]
    assert query_text.startswith("Instruct: Given a code search query")
    assert query_text.endswith("Query:where is request tracing implemented?")
    assert tokenizer.call_args_list[0].kwargs == {
        "padding": True,
        "truncation": True,
        "max_length": 8192,
        "return_tensors": "np",
    }
    mlx_core.set_cache_limit.assert_called_once_with(512 * 1024 * 1024)  # type: ignore[attr-defined]
    assert mlx_core.clear_cache.call_count == 3  # type: ignore[attr-defined]


def test_mlx_rejects_unknown_non_qwen_checkpoint() -> None:
    with pytest.raises(EmbeddingError, match="Qwen3-Embedding checkpoints"):
        MlxEngine("mlx-community/not-an-embedding-model")


def test_mlx_qwen3_rejects_non_qwen_metadata_without_loading_weights(tmp_path) -> None:
    model_config = tmp_path / "config.json"
    model_config.write_text('{"model_type": "llama", "hidden_size": 2560}')
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.try_to_load_from_cache = MagicMock(  # type: ignore[attr-defined]
        return_value=str(model_config)
    )
    huggingface_hub.hf_hub_download = MagicMock()  # type: ignore[attr-defined]
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = MagicMock()  # type: ignore[attr-defined]

    with (
        patch.dict(
            sys.modules,
            {"huggingface_hub": huggingface_hub, "mlx_lm": mlx_lm},
        ),
        pytest.raises(EmbeddingError, match="checkpoint config is not a Qwen3 model"),
    ):
        MlxEngine("mlx-community/Qwen3-Embedding-4B-invalid")

    mlx_lm.load.assert_not_called()  # type: ignore[attr-defined]
