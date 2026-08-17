"""Tests for the internal LMS OpenAI-compatible inference backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from stropha.config import get_config, reset_config
from stropha.errors import InferenceError, InferenceTimeoutError
from stropha.inference import LmsBackend, generate, get_backend
from stropha.inference.backend import reset_backend


def _response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://lms.example/openai/chat/completions"),
    )


def test_lms_config_and_secret_parse(isolated_test_env) -> None:
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "inference:\n"
        "  backend: lms\n"
        "  lms_model: test-model\n"
        "  lms_base_url: https://lms.example/openai\n"
        "  lms_batch_size: 7\n"
        "  lms_max_connections: 6\n"
        "  lms_max_keepalive_connections: 3\n"
        "secrets:\n"
        "  lms_api_key: secret\n"
    )
    reset_config()

    cfg = get_config()
    assert cfg.inference.lms_model == "test-model"
    assert cfg.inference.effective_batch_size() == 7
    assert cfg.inference.lms_max_connections == 6
    assert cfg.inference.lms_max_keepalive_connections == 3
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
        "httpx.Client.post",
        return_value=_response({"choices": [{"message": {"content": "result"}}]}),
    ) as post:
        assert backend.generate("hello", max_tokens=17, temperature=0.2) == "result"

    assert post.call_args.args[0] == "https://lms.example/openai/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
    body = post.call_args.kwargs["json"]
    assert body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 17,
        "temperature": 0.2,
        "stream": False,
        "think": False,
    }


def test_client_pool_has_bounded_configurable_limits() -> None:
    with patch("stropha.inference.lms_backend.httpx.Client") as client_type:
        backend = LmsBackend(
            api_key="secret",
            max_connections=6,
            max_keepalive_connections=3,
        )
        backend.close()

    limits = client_type.call_args.kwargs["limits"]
    assert limits.max_connections == 6
    assert limits.max_keepalive_connections == 3
    client_type.return_value.close.assert_called_once_with()


def test_generate_reuses_injected_client() -> None:
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _response(
        {"choices": [{"message": {"content": "result"}}]}
    )
    backend = LmsBackend(api_key="secret", client=client)

    assert backend.generate("one") == "result"
    assert backend.generate("two") == "result"
    backend.close()

    assert client.post.call_count == 2
    client.close.assert_not_called()


def test_generate_retries_transient_http_status() -> None:
    backend = LmsBackend(api_key="secret", max_retries=1)
    with (
        patch(
            "httpx.Client.post",
            side_effect=[
                _response({}, status_code=503),
                _response({"choices": [{"message": {"content": "result"}}]}),
            ],
        ) as post,
        patch("stropha.inference.lms_backend.time.sleep") as sleep,
    ):
        assert backend.generate("hello") == "result"

    assert post.call_count == 2
    sleep.assert_called_once_with(1)


def test_identity_changes_with_reliability_controls() -> None:
    base = LmsBackend(api_key="secret", timeout=30.0, max_retries=1)
    assert base.identity != LmsBackend(
        api_key="secret", timeout=31.0, max_retries=1
    ).identity
    assert base.identity != LmsBackend(
        api_key="secret", timeout=30.0, max_retries=2
    ).identity


def test_generate_supports_structured_ranking_controls() -> None:
    backend = LmsBackend(
        model="test-model",
        base_url="https://lms.example/openai/",
        api_key="secret",
        timeout=30.0,
        max_retries=0,
        think=True,
    )
    schema = {
        "type": "object",
        "properties": {"order": {"type": "array"}},
        "required": ["order"],
    }
    with patch(
        "httpx.Client.post",
        return_value=_response({"choices": [{"message": {"content": "{}"}}]}),
    ) as post:
        assert (
            backend.generate(
                "rank",
                json_schema=schema,
                think=False,
                top_p=0.95,
                top_k=40,
                presence_penalty=1.0,
                timeout=4.5,
            )
            == "{}"
        )

    body = post.call_args.kwargs["json"]
    assert body["think"] is False
    assert body["top_p"] == 0.95
    assert body["top_k"] == 40
    assert body["presence_penalty"] == 1.0
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "stropha_response",
            "strict": True,
            "schema": schema,
        },
    }
    assert post.call_args.kwargs["timeout"] == 4.5


def test_dispatcher_forwards_lms_sampling_controls() -> None:
    client = MagicMock(spec=httpx.Client)
    client.post.return_value = _response(
        {"choices": [{"message": {"content": "result"}}]}
    )
    backend = LmsBackend(api_key="secret", client=client)
    with patch("stropha.inference.backend.get_backend", return_value=backend):
        assert (
            generate(
                "hello",
                top_p=0.95,
                top_k=40,
                presence_penalty=1.0,
            )
            == "result"
        )

    body = client.post.call_args.kwargs["json"]
    assert body["top_p"] == 0.95
    assert body["top_k"] == 40
    assert body["presence_penalty"] == 1.0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("timeout"), InferenceTimeoutError),
        (httpx.ConnectError("offline"), InferenceError),
    ],
)
def test_generate_can_raise_typed_errors_for_diagnostics(
    error: Exception,
    expected: type[Exception],
) -> None:
    backend = LmsBackend(api_key="secret", max_retries=0)
    with (
        patch(
            "httpx.Client.post",
            side_effect=error,
        ),
        pytest.raises(expected),
    ):
        backend.generate("rank", raise_errors=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_generate_raises_for_empty_success_response(payload: dict) -> None:
    backend = LmsBackend(api_key="secret", max_retries=0)
    with (
        patch(
            "httpx.Client.post",
            return_value=_response(payload),
        ),
        pytest.raises(InferenceError),
    ):
        backend.generate("rank", raise_errors=True)


def test_generate_raises_without_api_key_when_requested(isolated_test_env) -> None:
    reset_config()
    with pytest.raises(InferenceError, match="API key"):
        LmsBackend(api_key="").generate("rank", raise_errors=True)


def test_health_checks_exact_model() -> None:
    backend = LmsBackend(model="wanted", api_key="secret", max_retries=0)
    with patch(
        "httpx.Client.get",
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
