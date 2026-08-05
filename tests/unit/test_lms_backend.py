"""Tests for the internal LMS OpenAI-compatible inference backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from stropha.config import get_config, reset_config
from stropha.inference import LmsBackend, get_backend
from stropha.inference.backend import reset_backend


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_lms_config_and_secret_parse(isolated_test_env) -> None:
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "inference:\n"
        "  backend: lms\n"
        "  lms_model: test-model\n"
        "  lms_base_url: https://lms.example/openai\n"
        "  lms_batch_size: 7\n"
        "secrets:\n"
        "  lms_api_key: secret\n"
    )
    reset_config()

    cfg = get_config()
    assert cfg.inference.lms_model == "test-model"
    assert cfg.inference.effective_batch_size() == 7
    assert cfg.secrets.has_lms is True


def test_non_string_lms_secret_is_treated_as_missing(isolated_test_env) -> None:
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text("secrets:\n  lms_api_key: true\n")
    reset_config()
    assert get_config().secrets.has_lms is False


def test_generate_uses_bearer_chat_completions() -> None:
    backend = LmsBackend(
        model="test-model",
        base_url="https://lms.example/openai/",
        api_key="secret",
        max_retries=0,
    )
    with patch(
        "stropha.inference.lms_backend.urllib_request.urlopen",
        return_value=_response({"choices": [{"message": {"content": "result"}}]}),
    ) as urlopen:
        assert backend.generate("hello", max_tokens=17, temperature=0.2) == "result"

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://lms.example/openai/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.data)
    assert body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 17,
        "temperature": 0.2,
        "stream": False,
        "think": False,
    }


def test_health_checks_exact_model() -> None:
    backend = LmsBackend(model="wanted", api_key="secret", max_retries=0)
    with patch(
        "stropha.inference.lms_backend.urllib_request.urlopen",
        return_value=_response({"data": [{"id": "wanted"}]}),
    ):
        assert backend.health_check() == (True, "LMS ready with wanted")


def test_explicit_lms_without_key_does_not_fallback(isolated_test_env) -> None:
    reset_config()
    reset_backend()
    backend = get_backend(prefer="lms")
    assert isinstance(backend, LmsBackend)
    assert backend.is_available is False


def test_identity_changes_with_model_or_endpoint() -> None:
    base = LmsBackend(model="one", base_url="https://a", api_key="secret")
    other_model = LmsBackend(model="two", base_url="https://a", api_key="secret")
    other_endpoint = LmsBackend(model="one", base_url="https://b", api_key="secret")
    assert base.identity != other_model.identity
    assert base.identity != other_endpoint.identity
