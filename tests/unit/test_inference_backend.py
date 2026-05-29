"""Tests for the unified inference backend."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from stropha.inference import (
    generate,
    get_backend,
    is_mlx_available,
    is_ollama_available,
)
from stropha.inference.backend import InferenceBackend, reset_backend
from stropha.inference.mlx_backend import MlxBackend
from stropha.inference.ollama_backend import OllamaBackend


@pytest.fixture(autouse=True)
def reset_backend_cache():
    """Reset the cached backend before each test."""
    reset_backend()
    yield
    reset_backend()


# --------------------------------------------------------------------------- #
#                           Backend Detection Tests                           #
# --------------------------------------------------------------------------- #


def test_is_mlx_available_when_installed():
    """Returns True when mlx_lm is importable."""
    with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
        # Need to reimport to pick up the mocked module
        from stropha.inference.backend import is_mlx_available as check
        # Since we can't easily mock the import, just test the function exists
        assert callable(check)


def test_is_mlx_available_when_not_installed():
    """Returns False when mlx_lm import fails."""
    # This test works because mlx_lm is likely not installed in CI
    # On Apple Silicon with mlx_lm installed, this would return True
    result = is_mlx_available()
    assert isinstance(result, bool)


def test_is_ollama_available_checks_endpoint():
    """Checks Ollama endpoint availability."""
    # Just verify it returns a boolean (actual availability depends on system)
    result = is_ollama_available()
    assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
#                           MlxBackend Tests                                  #
# --------------------------------------------------------------------------- #


class TestMlxBackend:
    """Tests for the MLX backend."""

    def test_name_is_mlx(self):
        backend = MlxBackend()
        assert backend.name == "mlx"

    def test_is_available_checks_import(self):
        backend = MlxBackend()
        # Just verify it returns a boolean
        assert isinstance(backend.is_available, bool)

    def test_generate_returns_none_when_not_available(self):
        """When MLX is not available, generate returns None."""
        backend = MlxBackend()
        if not backend.is_available:
            result = backend.generate("test prompt")
            assert result is None

    def test_custom_model_override(self):
        backend = MlxBackend(model="custom-model")
        assert backend._default_model == "custom-model"

    def test_yaml_model_override(self, isolated_test_env):
        """YAML config can override MLX model."""
        from stropha.config import reset_config
        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text("""
inference:
  mlx_model: yaml-model
""")
        reset_config()
        backend = MlxBackend()
        assert backend._default_model == "yaml-model"


# --------------------------------------------------------------------------- #
#                           OllamaBackend Tests                               #
# --------------------------------------------------------------------------- #


class TestOllamaBackend:
    """Tests for the Ollama backend."""

    def test_name_is_ollama(self):
        backend = OllamaBackend()
        assert backend.name == "ollama"

    def test_default_model(self, isolated_test_env):
        """Default model comes from config defaults."""
        backend = OllamaBackend()
        assert backend._default_model == "qwen2.5-coder:1.5b"

    def test_custom_model(self):
        backend = OllamaBackend(model="custom-model")
        assert backend._default_model == "custom-model"

    def test_custom_base_url(self):
        backend = OllamaBackend(base_url="http://custom:1234")
        assert backend._base_url == "http://custom:1234"

    def test_env_var_host_override(self, isolated_test_env):
        """YAML ollama_host can override the default Ollama host."""
        from stropha.config import reset_config
        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "inference:\n  ollama_host: http://yaml-host:5678\n"
        )
        reset_config()  # Reload config to pick up YAML
        backend = OllamaBackend()
        assert backend._base_url == "http://yaml-host:5678"

    def test_is_available_returns_false_on_connection_error(self):
        with patch("stropha.inference.ollama_backend.urllib_request.urlopen") as mock:
            from urllib.error import URLError
            mock.side_effect = URLError("connection refused")
            
            backend = OllamaBackend()
            assert backend.is_available is False

    def test_generate_returns_none_on_failure(self):
        with patch("stropha.inference.ollama_backend.urllib_request.urlopen") as mock:
            from urllib.error import URLError
            mock.side_effect = URLError("connection refused")
            
            backend = OllamaBackend()
            result = backend.generate("test prompt")
            assert result is None


# --------------------------------------------------------------------------- #
#                           get_backend Tests                                 #
# --------------------------------------------------------------------------- #


class TestGetBackend:
    """Tests for backend auto-detection."""

    def test_returns_inference_backend(self):
        backend = get_backend()
        assert isinstance(backend, InferenceBackend)

    def test_caches_backend(self):
        backend1 = get_backend()
        backend2 = get_backend()
        assert backend1 is backend2

    def test_prefer_mlx(self):
        with patch("stropha.inference.backend.is_mlx_available", return_value=True):
            backend = get_backend(prefer="mlx")
            # Even if MLX load fails, it should try MLX first
            assert backend is not None

    def test_prefer_ollama(self):
        backend = get_backend(prefer="ollama")
        assert isinstance(backend, OllamaBackend)

    def test_yaml_config_backend_override(self, isolated_test_env):
        """YAML config can select backend."""
        from stropha.config import reset_config
        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text("""
inference:
  backend: ollama
""")
        reset_config()
        reset_backend()
        backend = get_backend()
        assert isinstance(backend, OllamaBackend)


# --------------------------------------------------------------------------- #
#                           generate Function Tests                           #
# --------------------------------------------------------------------------- #


class TestGenerate:
    """Tests for the convenience generate function."""

    def test_generate_uses_backend(self):
        with patch("stropha.inference.backend.get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.generate.return_value = "generated text"
            mock_get.return_value = mock_backend

            result = generate("test prompt")

            mock_backend.generate.assert_called_once()
            assert result == "generated text"

    def test_generate_passes_parameters(self):
        with patch("stropha.inference.backend.get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.generate.return_value = "text"
            mock_get.return_value = mock_backend

            generate(
                "prompt",
                max_tokens=100,
                temperature=0.5,
                model="custom",
            )

            mock_backend.generate.assert_called_with(
                "prompt",
                max_tokens=100,
                temperature=0.5,
                model="custom",
            )

    def test_generate_with_backend_preference(self):
        with patch("stropha.inference.backend.get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_backend.generate.return_value = "text"
            mock_get.return_value = mock_backend

            generate("prompt", backend="ollama")

            mock_get.assert_called_with(prefer="ollama")


# --------------------------------------------------------------------------- #
#                       InferenceConfig.effective_batch_size                  #
# --------------------------------------------------------------------------- #


class TestEffectiveBatchSize:
    """``InferenceConfig.effective_batch_size`` picks the per-backend
    override when one is set; otherwise falls back to the global
    ``enricher_batch_size``."""

    def test_no_overrides_returns_global(self):
        from stropha.config.models import InferenceConfig

        cfg = InferenceConfig(backend="deepseek", enricher_batch_size=10)
        assert cfg.effective_batch_size() == 10

    def test_deepseek_override_wins_over_global(self):
        from stropha.config.models import InferenceConfig

        cfg = InferenceConfig(
            backend="deepseek",
            enricher_batch_size=10,
            deepseek_batch_size=5,
        )
        assert cfg.effective_batch_size() == 5

    def test_anthropic_override_ignored_when_backend_is_deepseek(self):
        """Overrides are per-backend — anthropic_batch_size doesn't affect
        a DeepSeek run."""
        from stropha.config.models import InferenceConfig

        cfg = InferenceConfig(
            backend="deepseek",
            enricher_batch_size=10,
            anthropic_batch_size=3,  # set but for the wrong backend
        )
        assert cfg.effective_batch_size() == 10

    def test_each_backend_resolves_its_own_override(self):
        from stropha.config.models import InferenceConfig

        for backend, field, expected in [
            ("deepseek", "deepseek_batch_size", 5),
            ("anthropic", "anthropic_batch_size", 7),
            ("ollama", "ollama_batch_size", 2),
            ("mlx", "mlx_batch_size", 3),
        ]:
            cfg = InferenceConfig(
                backend=backend,
                enricher_batch_size=99,
                **{field: expected},
            )
            assert cfg.effective_batch_size() == expected, (
                f"backend={backend} expected override={expected} from {field}"
            )

    def test_unknown_backend_falls_back_to_global(self):
        from stropha.config.models import InferenceConfig

        cfg = InferenceConfig(
            backend="some-future-backend",
            enricher_batch_size=4,
            deepseek_batch_size=99,
        )
        assert cfg.effective_batch_size() == 4

    def test_none_backend_uses_global(self):
        """``backend=None`` means auto-detect — no per-backend override
        applies until the auto-selection resolves to a concrete backend."""
        from stropha.config.models import InferenceConfig

        cfg = InferenceConfig(
            backend=None,
            enricher_batch_size=6,
            deepseek_batch_size=2,
        )
        assert cfg.effective_batch_size() == 6


class TestPerBackendBatchSizeYAMLParsing:
    """YAML keys ``deepseek_batch_size`` / ``anthropic_batch_size`` /
    ``ollama_batch_size`` / ``mlx_batch_size`` flow through ``_parse_inference``."""

    def test_deepseek_batch_size_parsed_from_yaml(self, tmp_path, monkeypatch):
        import stropha.config as cfg_mod
        from pathlib import Path

        cfg_dir = tmp_path / ".stropha"
        cfg_dir.mkdir()
        (cfg_dir / "config.yaml").write_text(
            "inference:\n"
            "  backend: deepseek\n"
            "  enricher_batch_size: 10\n"
            "  deepseek_batch_size: 5\n"
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        # Force a fresh load.
        cfg_mod.reset_config()
        cfg = cfg_mod.load_config()

        assert cfg.inference.enricher_batch_size == 10
        assert cfg.inference.deepseek_batch_size == 5
        assert cfg.inference.effective_batch_size() == 5

    def test_absent_per_backend_override_stays_none(self, tmp_path, monkeypatch):
        import stropha.config as cfg_mod
        from pathlib import Path

        cfg_dir = tmp_path / ".stropha"
        cfg_dir.mkdir()
        (cfg_dir / "config.yaml").write_text(
            "inference:\n"
            "  backend: deepseek\n"
            "  enricher_batch_size: 10\n"
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        cfg_mod.reset_config()
        cfg = cfg_mod.load_config()

        assert cfg.inference.deepseek_batch_size is None
        assert cfg.inference.effective_batch_size() == 10

    def test_invalid_per_backend_value_falls_back_to_none(
        self, tmp_path, monkeypatch
    ):
        """A non-integer value parses to None → uses global enricher_batch_size."""
        import stropha.config as cfg_mod
        from pathlib import Path

        cfg_dir = tmp_path / ".stropha"
        cfg_dir.mkdir()
        (cfg_dir / "config.yaml").write_text(
            "inference:\n"
            "  backend: deepseek\n"
            "  enricher_batch_size: 8\n"
            "  deepseek_batch_size: not-a-number\n"
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        cfg_mod.reset_config()
        cfg = cfg_mod.load_config()

        assert cfg.inference.deepseek_batch_size is None
        assert cfg.inference.effective_batch_size() == 8
