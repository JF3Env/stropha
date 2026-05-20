"""Unit tests for the Model2Vec static embedder adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stropha.adapters.embedder.model2vec import (
    Model2VecEmbedder,
    Model2VecEmbedderConfig,
)
from stropha.errors import EmbeddingError
from stropha.pipeline.registry import lookup_adapter


class TestModel2VecEmbedderConfig:
    """Tests for Model2VecEmbedderConfig."""

    def test_default_config(self):
        config = Model2VecEmbedderConfig()
        assert config.model == "minishlab/potion-code-16M"
        assert config.batch_size == 256
        assert config.normalize is True

    def test_custom_config(self):
        config = Model2VecEmbedderConfig(
            model="custom/model",
            batch_size=128,
            normalize=False,
        )
        assert config.model == "custom/model"
        assert config.batch_size == 128
        assert config.normalize is False


def _create_mock_model2vec(dim: int = 256):
    """Helper to create mock model2vec module and model."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * dim])
    
    mock_static_model = MagicMock()
    mock_static_model.from_pretrained.return_value = mock_model
    
    mock_module = MagicMock()
    mock_module.StaticModel = mock_static_model
    
    return mock_module, mock_static_model, mock_model


class TestModel2VecEmbedderRegistration:
    """Tests for adapter registration."""

    def test_registered_in_adapter_registry(self):
        adapter_cls = lookup_adapter("embedder", "model2vec")
        assert adapter_cls is Model2VecEmbedder

    def test_stage_name(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            assert embedder.stage_name == "embedder"

    def test_adapter_name(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            assert embedder.adapter_name == "model2vec"


class TestModel2VecEmbedderInit:
    """Tests for Model2VecEmbedder initialization."""

    def test_init_with_default_config(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            
            mock_static_model.from_pretrained.assert_called_once_with("minishlab/potion-code-16M")
            assert embedder.model_name == "minishlab/potion-code-16M"
            assert embedder.dim == 256

    def test_init_with_custom_config(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec(dim=512)
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            config = Model2VecEmbedderConfig(model="custom/model-512d")
            embedder = Model2VecEmbedder(config=config)
            
            mock_static_model.from_pretrained.assert_called_once_with("custom/model-512d")
            assert embedder.model_name == "custom/model-512d"
            assert embedder.dim == 512

    def test_init_raises_on_missing_package(self):
        # Remove model2vec from sys.modules to simulate missing package
        with patch.dict("sys.modules", {"model2vec": None}):
            with pytest.raises(EmbeddingError, match="model2vec package is not installed"):
                Model2VecEmbedder()

    def test_init_raises_on_model_load_failure(self):
        mock_module = MagicMock()
        mock_static_model = MagicMock()
        mock_static_model.from_pretrained.side_effect = Exception("Model not found")
        mock_module.StaticModel = mock_static_model
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            with pytest.raises(EmbeddingError, match="Model2Vec failed to load"):
                Model2VecEmbedder()


class TestModel2VecEmbedderAdapterId:
    """Tests for adapter_id drift detection."""

    def test_adapter_id_format(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            assert embedder.adapter_id == "model2vec:minishlab/potion-code-16M:256"

    def test_adapter_id_changes_with_model(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder1 = Model2VecEmbedder()
            
            mock_model.encode.return_value = np.array([[0.1] * 512])
            config2 = Model2VecEmbedderConfig(model="other/model")
            embedder2 = Model2VecEmbedder(config=config2)
            
            assert embedder1.adapter_id != embedder2.adapter_id


class TestModel2VecEmbedderHealth:
    """Tests for health check."""

    def test_health_ready(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            health = embedder.health()
            
            assert health.status == "ready"
            assert "Model2Vec" in health.message
            assert health.detail["model"] == "minishlab/potion-code-16M"
            assert health.detail["dim"] == "256"


class TestModel2VecEmbedderEmbed:
    """Tests for embedding operations."""

    def test_embed_documents_single(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            result = embedder.embed_documents(["test text"])
            
            assert len(result) == 1
            assert result[0] == [0.1, 0.2, 0.3]

    def test_embed_documents_batch(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        # First call for probe, second for actual embed
        mock_model.encode.side_effect = [
            np.array([[0.1] * 256]),  # probe
            np.array([[0.1] * 256, [0.2] * 256, [0.3] * 256]),  # batch
        ]
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):

            embedder = Model2VecEmbedder()
            result = embedder.embed_documents(["text1", "text2", "text3"])
            
            assert len(result) == 3

    def test_embed_documents_empty(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            result = embedder.embed_documents([])
            
            assert result == []

    def test_embed_documents_passes_normalize(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            config = Model2VecEmbedderConfig(normalize=False)
            embedder = Model2VecEmbedder(config=config)
            embedder.embed_documents(["test"])
            
            # Check the embed call (not the probe call)
            calls = mock_model.encode.call_args_list
            assert len(calls) >= 2
            assert calls[-1][1]["normalize"] is False

    def test_embed_query(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            result = embedder.embed_query("search query")
            
            assert result == [0.5, 0.6, 0.7]

    def test_embed_documents_raises_on_error(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        mock_model.encode.side_effect = [
            np.array([[0.1] * 256]),  # probe succeeds
            Exception("Encoding failed"),  # batch fails
        ]
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            
            with pytest.raises(EmbeddingError, match="Model2Vec embed failed"):
                embedder.embed_documents(["test"])

    def test_embed_query_raises_on_error(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        mock_model.encode.side_effect = [
            np.array([[0.1] * 256]),  # probe succeeds
            Exception("Query failed"),  # query fails
        ]
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            
            with pytest.raises(EmbeddingError, match="Model2Vec query embed failed"):
                embedder.embed_query("test")


class TestModel2VecEmbedderBatchSize:
    """Tests for batch_size property."""

    def test_default_batch_size(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            embedder = Model2VecEmbedder()
            assert embedder.batch_size == 256

    def test_custom_batch_size(self):
        mock_module, mock_static_model, mock_model = _create_mock_model2vec()
        
        with patch.dict("sys.modules", {"model2vec": mock_module}):
            config = Model2VecEmbedderConfig(batch_size=512)
            embedder = Model2VecEmbedder(config=config)
            assert embedder.batch_size == 512
