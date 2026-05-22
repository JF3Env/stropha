"""Tests for the DeepSeek inference backend."""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from stropha.inference import (
    DeepSeekBackend,
    generate,
    get_backend,
    is_deepseek_available,
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


def _http_response(payload: dict) -> MagicMock:
    """Build an urlopen-compatible context-manager mock with the given JSON body."""
    body = json.dumps(payload).encode("utf-8")
    response = MagicMock()
    response.read.return_value = body
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


# --------------------------------------------------------------------------- #
#                         DeepSeekBackend Unit Tests                          #
# --------------------------------------------------------------------------- #


class TestDeepSeekBackend:
    """Tests for the DeepSeek backend."""

    def test_name_is_deepseek(self):
        backend = DeepSeekBackend(api_key="sk-test")
        assert backend.name == "deepseek"

    def test_default_model_from_config(self, isolated_test_env):
        """Default model comes from InferenceConfig defaults."""
        backend = DeepSeekBackend(api_key="sk-test")
        assert backend._default_model == "deepseek-chat"

    def test_custom_model_overrides_config(self):
        backend = DeepSeekBackend(model="deepseek-reasoner", api_key="sk-test")
        assert backend._default_model == "deepseek-reasoner"

    def test_custom_base_url(self):
        backend = DeepSeekBackend(
            base_url="https://proxy.example.com/v1/", api_key="sk-test"
        )
        # Trailing slash is stripped.
        assert backend._base_url == "https://proxy.example.com/v1"

    def test_env_var_api_key_pickup(self, isolated_test_env, monkeypatch):
        """DEEPSEEK_API_KEY env var feeds SecretsConfig.deepseek_api_key."""
        from stropha.config import reset_config

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        reset_config()
        backend = DeepSeekBackend()
        assert backend._api_key == "sk-from-env"
        assert backend.is_available is True

    def test_is_available_false_without_api_key(self, isolated_test_env, monkeypatch):
        """No DEEPSEEK_API_KEY → is_available returns False.

        Sets the env var to empty string so ``load_dotenv(override=False)``
        does NOT re-introduce the project's real key via dotenv's
        walk-up-from-__file__ search. This is more robust than ``delenv``.
        """
        from stropha.config import reset_config

        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        reset_config()
        backend = DeepSeekBackend()
        assert backend._api_key in (None, "")
        assert backend.is_available is False

    def test_is_available_true_with_api_key(self):
        backend = DeepSeekBackend(api_key="sk-test")
        assert backend.is_available is True

    def test_is_available_false_with_blank_key(self):
        backend = DeepSeekBackend(api_key="   ")
        assert backend.is_available is False

    # ---- generate ---------------------------------------------------------

    def test_generate_returns_text_on_success(self):
        payload = {
            "choices": [
                {"message": {"role": "assistant", "content": "hello world"}}
            ]
        }
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = DeepSeekBackend(api_key="sk-test")
            result = backend.generate("ping", max_tokens=8, temperature=0.0)
            assert result == "hello world"

    def test_generate_sends_bearer_token_and_chat_messages(self):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = DeepSeekBackend(api_key="sk-secret")
            backend.generate("hi", max_tokens=10, temperature=0.3)
            assert mock.call_count == 1
            req = mock.call_args[0][0]
            assert req.full_url.endswith("/chat/completions")
            assert req.headers["Authorization"] == "Bearer sk-secret"
            assert req.headers["Content-type"] == "application/json"
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == "deepseek-chat"
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            assert body["max_tokens"] == 10
            assert body["temperature"] == 0.3
            assert body["stream"] is False

    def test_generate_respects_per_call_model_override(self):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = DeepSeekBackend(api_key="sk-test")
            backend.generate("x", model="deepseek-reasoner")
            body = json.loads(mock.call_args[0][0].data.decode("utf-8"))
            assert body["model"] == "deepseek-reasoner"

    def test_generate_returns_none_when_no_api_key(self, isolated_test_env, monkeypatch):
        from stropha.config import reset_config

        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        reset_config()
        backend = DeepSeekBackend()
        assert backend.generate("anything") is None

    def test_generate_returns_none_on_connection_error(self):
        from urllib.error import URLError

        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = URLError("dns failure")
            # max_retries=0 so the test asserts the terminal outcome without
            # sleeping through the backoff schedule.
            backend = DeepSeekBackend(api_key="sk-test", max_retries=0)
            assert backend.generate("x") is None
            assert mock.call_count == 1

    def test_generate_returns_none_on_http_error(self):
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = HTTPError(
                url="https://api.deepseek.com/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad key"}'),
            )
            backend = DeepSeekBackend(api_key="sk-bad", max_retries=0)
            assert backend.generate("x") is None
            # 401 is permanent — must NOT retry even with retries > 0.
            assert mock.call_count == 1

    def test_generate_does_not_retry_on_4xx(self):
        """HTTP 4xx is permanent (bad key, bad request, rate limit) — no retry."""
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = HTTPError(
                url="https://api.deepseek.com/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"rate"}'),
            )
            backend = DeepSeekBackend(api_key="sk-test", max_retries=5)
            assert backend.generate("x") is None
            assert mock.call_count == 1  # No retry on 4xx

    def test_generate_retries_on_timeout(self):
        """Network timeouts are transient — retry with backoff."""
        from urllib.error import URLError

        good_payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.deepseek_backend.time.sleep"
        ) as sleep_mock:
            # First call times out, second succeeds.
            mock.side_effect = [URLError("timed out"), _http_response(good_payload)]
            backend = DeepSeekBackend(api_key="sk-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            assert mock.call_count == 2  # 1 fail + 1 success
            sleep_mock.assert_called_once_with(1)  # backoff: 2^0 = 1s

    def test_generate_retries_then_gives_up(self):
        """After max_retries exhausted, returns None."""
        from urllib.error import URLError

        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.deepseek_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = URLError("timed out")
            backend = DeepSeekBackend(api_key="sk-test", max_retries=2)
            result = backend.generate("x")
            assert result is None
            assert mock.call_count == 3  # initial + 2 retries
            # Backoff schedule: 1s, 2s (2^0, 2^1).
            assert sleep_mock.call_args_list == [((1,),), ((2,),)]

    def test_generate_retries_on_5xx(self):
        """HTTP 5xx is transient — retry."""
        from urllib.error import HTTPError

        good_payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.deepseek_backend.time.sleep"
        ):
            mock.side_effect = [
                HTTPError(
                    url="https://api.deepseek.com/v1/chat/completions",
                    code=503,
                    msg="Service Unavailable",
                    hdrs=None,
                    fp=io.BytesIO(b""),
                ),
                _http_response(good_payload),
            ]
            backend = DeepSeekBackend(api_key="sk-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            assert mock.call_count == 2

    def test_default_timeout_and_max_retries_from_config(self, isolated_test_env):
        """When not overridden, the backend pulls timeout / max_retries from YAML."""
        from stropha.config import reset_config

        (isolated_test_env / "stropha.yaml").write_text(
            "inference:\n"
            "  deepseek_timeout: 30\n"
            "  deepseek_max_retries: 5\n"
        )
        reset_config()
        backend = DeepSeekBackend(api_key="sk-test")
        assert backend._timeout == 30.0
        assert backend._max_retries == 5

    def test_generate_returns_none_on_empty_choices(self):
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response({"choices": []})
            backend = DeepSeekBackend(api_key="sk-test")
            assert backend.generate("x") is None

    def test_generate_returns_none_on_empty_content(self):
        payload = {"choices": [{"message": {"content": "   "}}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = DeepSeekBackend(api_key="sk-test")
            assert backend.generate("x") is None

    def test_generate_returns_none_on_malformed_json(self):
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            response = MagicMock()
            response.read.return_value = b"not json"
            cm = MagicMock()
            cm.__enter__.return_value = response
            cm.__exit__.return_value = False
            mock.return_value = cm
            backend = DeepSeekBackend(api_key="sk-test")
            assert backend.generate("x") is None

    # ---- health_check -----------------------------------------------------

    def test_health_check_missing_key(self, isolated_test_env, monkeypatch):
        from stropha.config import reset_config

        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        reset_config()
        backend = DeepSeekBackend()
        ok, message = backend.health_check()
        assert ok is False
        assert "DEEPSEEK_API_KEY" in message

    def test_health_check_ok_when_model_listed(self):
        listing = {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}
        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(listing)
            backend = DeepSeekBackend(api_key="sk-test")
            ok, message = backend.health_check()
            assert ok is True
            assert "deepseek-chat" in message

    def test_health_check_auth_failure(self):
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.deepseek_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = HTTPError(
                url="https://api.deepseek.com/v1/models",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b""),
            )
            backend = DeepSeekBackend(api_key="sk-bad")
            ok, message = backend.health_check()
            assert ok is False
            assert "auth" in message.lower()


# --------------------------------------------------------------------------- #
#                        Module-level helpers                                 #
# --------------------------------------------------------------------------- #


def test_is_deepseek_available_reflects_secret(isolated_test_env, monkeypatch):
    from stropha.config import reset_config

    # Missing key → False.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    reset_config()
    assert is_deepseek_available() is False

    # Present key → True.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-y")
    reset_config()
    assert is_deepseek_available() is True


# --------------------------------------------------------------------------- #
#                  get_backend(prefer="deepseek") routing                     #
# --------------------------------------------------------------------------- #


class TestGetBackendDeepSeek:
    def test_prefer_deepseek_with_key(self, isolated_test_env):
        from stropha.config import reset_config

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-y"}):
            reset_config()
            backend = get_backend(prefer="deepseek")
            assert isinstance(backend, DeepSeekBackend)

    def test_prefer_deepseek_without_key_does_not_silent_fallback(
        self, isolated_test_env, monkeypatch
    ):
        """When the user explicitly asks for DeepSeek but no key is set, we
        still return a DeepSeekBackend (so health_check surfaces the issue)
        instead of silently dropping to a local model. ``generate`` will
        return ``None`` per the contract."""
        from stropha.config import reset_config

        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        reset_config()
        backend = get_backend(prefer="deepseek")
        assert isinstance(backend, DeepSeekBackend)
        assert backend.is_available is False

    def test_yaml_backend_deepseek(self, isolated_test_env):
        """YAML inference.backend=deepseek selects DeepSeek when key present."""
        from stropha.config import reset_config

        (isolated_test_env / "stropha.yaml").write_text(
            "inference:\n  backend: deepseek\n"
        )
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-y"}):
            reset_config()
            backend = get_backend()
            assert isinstance(backend, DeepSeekBackend)

    def test_auto_detect_never_picks_deepseek(self, isolated_test_env):
        """Auto-detection MUST NOT route to DeepSeek even when API key is
        set — local-only-by-default invariant (CLAUDE.md §3.6)."""
        from stropha.config import reset_config

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-y"}):
            reset_config()
            # No backend preference → auto-detect → must pick MLX or Ollama.
            backend = get_backend()
            assert not isinstance(backend, DeepSeekBackend)
            assert isinstance(backend, (MlxBackend, OllamaBackend))


# --------------------------------------------------------------------------- #
#                   generate() dispatches when backend=deepseek               #
# --------------------------------------------------------------------------- #


def test_generate_with_deepseek_preference_routes_to_deepseek(isolated_test_env):
    """The convenience helper passes prefer='deepseek' through."""
    with patch("stropha.inference.backend.get_backend") as mock_get:
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.generate.return_value = "from-deepseek"
        mock_get.return_value = mock_backend

        out = generate("hi", backend="deepseek")

        mock_get.assert_called_with(prefer="deepseek")
        assert out == "from-deepseek"
