"""Tests for the local MoA Gateway inference backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock, patch
from urllib.error import URLError

from stropha.config import get_config, reset_config
from stropha.inference import MoaBackend, get_backend
from stropha.inference.backend import reset_backend


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_moa_config_and_secret_parse(isolated_test_env) -> None:
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "inference:\n"
        "  backend: moa\n"
        "  moa_model: test-model\n"
        "  moa_base_url: http://127.0.0.1:14598/v1\n"
        "  moa_batch_size: 7\n"
        "secrets:\n"
        "  moa_api_key: secret\n"
    )
    reset_config()

    cfg = get_config()
    assert cfg.inference.moa_model == "test-model"
    assert cfg.inference.effective_batch_size() == 7
    assert cfg.secrets.has_moa is True


def test_generate_uses_openai_chat_completions() -> None:
    backend = MoaBackend(
        model="moa-code",
        base_url="http://127.0.0.1:14598/v1/",
        api_key="secret",
        max_retries=0,
    )
    with patch(
        "stropha.inference.moa_backend.urllib_request.urlopen",
        return_value=_response({"choices": [{"message": {"content": "result"}}]}),
    ) as urlopen:
        assert backend.generate("hello", max_tokens=17, temperature=0.2) == "result"

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:14598/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.data) == {
        "model": "moa-code",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 17,
        "temperature": 0.2,
        "stream": False,
    }


def test_health_and_model_discovery() -> None:
    backend = MoaBackend(model="moa-code", max_retries=0)
    payload = {"data": [{"id": "moa-code"}, {"id": "direct-code"}]}
    with patch(
        "stropha.inference.moa_backend.urllib_request.urlopen",
        return_value=_response(payload),
    ):
        assert backend.health_check() == (True, "MoA ready with moa-code")
        assert backend.list_models() == ["moa-code", "direct-code"]


def test_generate_returns_none_when_gateway_is_unreachable() -> None:
    backend = MoaBackend(max_retries=0)
    with patch(
        "stropha.inference.moa_backend.urllib_request.urlopen",
        side_effect=URLError("connection refused"),
    ):
        assert backend.generate("hello") is None
        assert backend.health_check()[0] is False
        assert backend.list_models() == []


def test_explicit_moa_does_not_fallback_when_unavailable() -> None:
    reset_backend()
    with patch.object(
        MoaBackend,
        "is_available",
        new_callable=PropertyMock,
        return_value=False,
    ):
        backend = get_backend(prefer="moa")
    assert isinstance(backend, MoaBackend)


def test_identity_changes_with_model_or_endpoint() -> None:
    base = MoaBackend(model="one", base_url="http://127.0.0.1:14598/v1")
    other_model = MoaBackend(model="two", base_url="http://127.0.0.1:14598/v1")
    other_endpoint = MoaBackend(model="one", base_url="http://127.0.0.1:9999/v1")
    assert base.identity != other_model.identity
    assert base.identity != other_endpoint.identity
