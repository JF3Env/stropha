"""Tests for the Anthropic inference backend."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from stropha.inference import (
    AnthropicBackend,
    DeepSeekBackend,
    generate,
    get_backend,
    is_anthropic_available,
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


def _ok_payload(text: str = "hello world") -> dict:
    """Build a minimal valid Anthropic Messages API response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
    }


# --------------------------------------------------------------------------- #
#                       AnthropicBackend Unit Tests                           #
# --------------------------------------------------------------------------- #


class TestAnthropicBackend:
    """Tests for the Anthropic backend."""

    def test_name_is_anthropic(self):
        backend = AnthropicBackend(api_key="sk-ant-test")
        assert backend.name == "anthropic"

    def test_default_model_from_config(self, isolated_test_env):
        """Default model comes from InferenceConfig defaults."""
        backend = AnthropicBackend(api_key="sk-ant-test")
        assert backend._default_model == "claude-haiku-4-5"

    def test_custom_model_overrides_config(self):
        backend = AnthropicBackend(model="claude-opus-4-7", api_key="sk-ant-test")
        assert backend._default_model == "claude-opus-4-7"

    def test_custom_base_url(self):
        backend = AnthropicBackend(
            base_url="https://proxy.example.com/v1/", api_key="sk-ant-test"
        )
        # Trailing slash is stripped.
        assert backend._base_url == "https://proxy.example.com/v1"

    def test_yaml_api_key_pickup(self, isolated_test_env):
        """anthropic_api_key in YAML secrets feeds SecretsConfig.anthropic_api_key."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        (isolated_test_env / ".stropha" / "config.yaml").write_text(
            "secrets:\n  anthropic_api_key: sk-ant-from-yaml\n"
        )
        reset_config()
        backend = AnthropicBackend()
        assert backend._api_key == "sk-ant-from-yaml"
        assert backend.is_available is True

    def test_is_available_false_without_api_key(self, isolated_test_env):
        """No anthropic_api_key in YAML → is_available returns False."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        (isolated_test_env / ".stropha" / "config.yaml").write_text("{}\n")
        reset_config()
        backend = AnthropicBackend()
        assert backend._api_key in (None, "")
        assert backend.is_available is False

    def test_is_available_true_with_api_key(self):
        backend = AnthropicBackend(api_key="sk-ant-test")
        assert backend.is_available is True

    def test_is_available_false_with_blank_key(self):
        backend = AnthropicBackend(api_key="   ")
        assert backend.is_available is False

    # ---- generate ---------------------------------------------------------

    def test_generate_returns_text_on_success(self):
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(_ok_payload("hello world"))
            backend = AnthropicBackend(api_key="sk-ant-test")
            result = backend.generate("ping", max_tokens=8, temperature=0.0)
            assert result == "hello world"

    def test_generate_sends_x_api_key_and_anthropic_version(self):
        """Anthropic uses x-api-key (NOT Bearer) and requires anthropic-version."""
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(_ok_payload("ok"))
            backend = AnthropicBackend(api_key="sk-ant-secret")
            backend.generate("hi", max_tokens=10, temperature=0.3)
            assert mock.call_count == 1
            req = mock.call_args[0][0]
            assert req.full_url.endswith("/messages")
            # NB: urllib normalises header keys to title-case via .header_items()
            # but `req.headers` keeps the original casing as the dict key.
            assert req.headers["X-api-key"] == "sk-ant-secret"
            assert req.headers["Anthropic-version"] == "2023-06-01"
            assert req.headers["Content-type"] == "application/json"
            # Must NOT send Authorization: Bearer
            assert "Authorization" not in req.headers
            body = json.loads(req.data.decode("utf-8"))
            assert body["model"] == "claude-haiku-4-5"
            assert body["messages"] == [{"role": "user", "content": "hi"}]
            assert body["max_tokens"] == 10
            assert body["temperature"] == 0.3

    def test_generate_respects_per_call_model_override(self):
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(_ok_payload("ok"))
            backend = AnthropicBackend(api_key="sk-ant-test")
            backend.generate("x", model="claude-opus-4-7")
            body = json.loads(mock.call_args[0][0].data.decode("utf-8"))
            assert body["model"] == "claude-opus-4-7"

    def test_generate_concatenates_multiple_text_blocks(self):
        """Anthropic can split a response into multiple content blocks."""
        payload = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        }
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = AnthropicBackend(api_key="sk-ant-test")
            assert backend.generate("x") == "Hello world"

    def test_generate_skips_non_text_blocks(self):
        """tool_use / image blocks must be ignored, only text concatenated."""
        payload = {
            "content": [
                {"type": "tool_use", "name": "calc", "input": {}},
                {"type": "text", "text": "answer"},
            ]
        }
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = AnthropicBackend(api_key="sk-ant-test")
            assert backend.generate("x") == "answer"

    def test_generate_returns_none_when_no_api_key(self, isolated_test_env, monkeypatch):
        from stropha.config import reset_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        reset_config()
        backend = AnthropicBackend()
        assert backend.generate("anything") is None

    def test_generate_returns_none_on_connection_error(self):
        from urllib.error import URLError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = URLError("dns failure")
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=0)
            assert backend.generate("x") is None
            assert mock.call_count == 1

    def test_generate_returns_none_on_http_error(self):
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad key"}'),
            )
            backend = AnthropicBackend(api_key="sk-ant-bad", max_retries=0)
            assert backend.generate("x") is None
            assert mock.call_count == 1

    def test_generate_does_not_retry_on_permanent_4xx(self):
        """HTTP 4xx other than 429 (bad key, bad request) is permanent — no retry."""
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.side_effect = HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"bad key"}'),
            )
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=5)
            assert backend.generate("x") is None
            assert mock.call_count == 1  # No retry on permanent 4xx

    def test_generate_retries_on_429_with_exponential_backoff_when_no_header(self):
        """429 without retry-after header → retry with exponential backoff."""
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = [
                HTTPError(
                    url="https://api.anthropic.com/v1/messages",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=None,  # No retry-after
                    fp=io.BytesIO(b'{"error":"rate"}'),
                ),
                _http_response(_ok_payload("ok")),
            ]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            assert mock.call_count == 2  # 1 throttle + 1 success
            sleep_mock.assert_called_once_with(1)  # backoff: 2^0 = 1s

    def test_generate_honours_retry_after_header_on_429(self):
        """429 with retry-after header → sleep exactly that many seconds."""
        from email.message import Message
        from urllib.error import HTTPError

        hdrs = Message()
        hdrs["retry-after"] = "7"

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = [
                HTTPError(
                    url="https://api.anthropic.com/v1/messages",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=hdrs,
                    fp=io.BytesIO(b'{"error":"rate"}'),
                ),
                _http_response(_ok_payload("ok")),
            ]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            sleep_mock.assert_called_once_with(7.0)

    def test_generate_caps_retry_after_at_max(self):
        """A misconfigured server sending retry-after=99999 must not stall us."""
        from email.message import Message
        from urllib.error import HTTPError

        hdrs = Message()
        hdrs["retry-after"] = "99999"

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = [
                HTTPError(
                    url="https://api.anthropic.com/v1/messages",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=hdrs,
                    fp=io.BytesIO(b""),
                ),
                _http_response(_ok_payload("ok")),
            ]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            backend.generate("x")
            # Capped at _MAX_RETRY_AFTER_SECONDS = 60.0
            sleep_mock.assert_called_once_with(60.0)

    def test_generate_gives_up_on_429_after_max_retries(self):
        """After exhausting retries on 429, returns None — caller falls back."""
        from urllib.error import HTTPError

        def make_429() -> HTTPError:
            return HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"rate"}'),
            )

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = [make_429(), make_429(), make_429()]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            assert backend.generate("x") is None
            assert mock.call_count == 3  # initial + 2 retries
            assert sleep_mock.call_count == 2

    def test_generate_retries_on_timeout(self):
        """Network timeouts are transient — retry with backoff."""
        from urllib.error import URLError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = [URLError("timed out"), _http_response(_ok_payload("ok"))]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            assert mock.call_count == 2  # 1 fail + 1 success
            sleep_mock.assert_called_once_with(1)  # backoff: 2^0 = 1s

    def test_generate_retries_then_gives_up(self):
        """After max_retries exhausted, returns None."""
        from urllib.error import URLError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ) as sleep_mock:
            mock.side_effect = URLError("timed out")
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            result = backend.generate("x")
            assert result is None
            assert mock.call_count == 3  # initial + 2 retries
            assert sleep_mock.call_args_list == [((1,),), ((2,),)]

    def test_generate_retries_on_5xx(self):
        """HTTP 5xx is transient — retry."""
        from urllib.error import HTTPError

        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock, patch(
            "stropha.inference.anthropic_backend.time.sleep"
        ):
            mock.side_effect = [
                HTTPError(
                    url="https://api.anthropic.com/v1/messages",
                    code=503,
                    msg="Service Unavailable",
                    hdrs=None,
                    fp=io.BytesIO(b""),
                ),
                _http_response(_ok_payload("ok")),
            ]
            backend = AnthropicBackend(api_key="sk-ant-test", max_retries=2)
            result = backend.generate("x")
            assert result == "ok"
            assert mock.call_count == 2

    def test_default_timeout_and_max_retries_from_config(self, isolated_test_env):
        """When not overridden, the backend pulls timeout / max_retries from YAML."""
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "inference:\n"
            "  anthropic_timeout: 30\n"
            "  anthropic_max_retries: 5\n"
        )
        reset_config()
        backend = AnthropicBackend(api_key="sk-ant-test")
        assert backend._timeout == 30.0
        assert backend._max_retries == 5

    def test_anthropic_version_override(self, isolated_test_env):
        """Custom anthropic_version is forwarded to the request header."""
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "inference:\n  anthropic_version: '2024-01-01'\n"
        )
        reset_config()
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(_ok_payload("ok"))
            backend = AnthropicBackend(api_key="sk-ant-test")
            backend.generate("x")
            req = mock.call_args[0][0]
            assert req.headers["Anthropic-version"] == "2024-01-01"

    def test_generate_returns_none_on_empty_content(self):
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response({"content": []})
            backend = AnthropicBackend(api_key="sk-ant-test")
            assert backend.generate("x") is None

    def test_generate_returns_none_on_whitespace_only_content(self):
        payload = {"content": [{"type": "text", "text": "   "}]}
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            mock.return_value = _http_response(payload)
            backend = AnthropicBackend(api_key="sk-ant-test")
            assert backend.generate("x") is None

    def test_generate_returns_none_on_malformed_json(self):
        with patch(
            "stropha.inference.anthropic_backend.urllib_request.urlopen"
        ) as mock:
            response = MagicMock()
            response.read.return_value = b"not json"
            cm = MagicMock()
            cm.__enter__.return_value = response
            cm.__exit__.return_value = False
            mock.return_value = cm
            backend = AnthropicBackend(api_key="sk-ant-test")
            assert backend.generate("x") is None

    # ---- health_check -----------------------------------------------------

    def test_health_check_missing_key(self, isolated_test_env):
        from stropha.config import reset_config

        # No key in YAML → missing
        reset_config()
        backend = AnthropicBackend()
        ok, message = backend.health_check()
        assert ok is False
        assert "secrets.anthropic_api_key" in message
        assert "~/.stropha/config.yaml" in message

    def test_health_check_ok_with_key(self):
        """Anthropic health_check is offline — just verifies the key exists."""
        backend = AnthropicBackend(api_key="sk-ant-test")
        ok, message = backend.health_check()
        assert ok is True
        assert "claude-haiku-4-5" in message


# --------------------------------------------------------------------------- #
#                        Module-level helpers                                 #
# --------------------------------------------------------------------------- #


def test_is_anthropic_available_reflects_secret(isolated_test_env):
    from stropha.config import reset_config

    # Missing key → False (no .stropha/config.yaml written).
    reset_config()
    assert is_anthropic_available() is False

    # Present key → True.
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text("secrets:\n  anthropic_api_key: sk-ant-y\n")
    reset_config()
    assert is_anthropic_available() is True


# --------------------------------------------------------------------------- #
#                  get_backend(prefer="anthropic") routing                    #
# --------------------------------------------------------------------------- #


class TestGetBackendAnthropic:
    def test_prefer_anthropic_with_key(self, isolated_test_env):
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text("secrets:\n  anthropic_api_key: sk-ant-y\n")
        reset_config()
        backend = get_backend(prefer="anthropic")
        assert isinstance(backend, AnthropicBackend)

    def test_prefer_anthropic_without_key_does_not_silent_fallback(
        self, isolated_test_env
    ):
        """When the user explicitly asks for Anthropic but no key is set, we
        still return an AnthropicBackend (so health_check surfaces the issue)
        instead of silently dropping to a local model. ``generate`` will
        return ``None`` per the contract."""
        from stropha.config import reset_config

        # No key in YAML → is_available=False
        reset_config()
        backend = get_backend(prefer="anthropic")
        assert isinstance(backend, AnthropicBackend)
        assert backend.is_available is False

    def test_yaml_backend_anthropic(self, isolated_test_env):
        """YAML inference.backend=anthropic selects Anthropic when key present."""
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "inference:\n  backend: anthropic\n"
            "secrets:\n  anthropic_api_key: sk-ant-y\n"
        )
        reset_config()
        backend = get_backend()
        assert isinstance(backend, AnthropicBackend)

    def test_auto_detect_never_picks_anthropic(self, isolated_test_env):
        """Auto-detection MUST NOT route to Anthropic — local-only-by-default
        invariant (CLAUDE.md §3.6). Even with key in YAML, no backend preference
        means auto-detect picks local."""
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text("secrets:\n  anthropic_api_key: sk-ant-y\n")
        reset_config()
        # No backend preference → auto-detect → must pick MLX or Ollama.
        backend = get_backend()
        assert not isinstance(backend, AnthropicBackend)
        assert isinstance(backend, (MlxBackend, OllamaBackend))

    def test_anthropic_and_deepseek_keys_both_set_auto_picks_local(
        self, isolated_test_env
    ):
        """Both cloud keys present in YAML, no preference → still picks local."""
        from stropha.config import reset_config

        cfg_dir = isolated_test_env / ".stropha"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "secrets:\n  anthropic_api_key: sk-ant-y\n  deepseek_api_key: sk-y\n"
        )
        reset_config()
        backend = get_backend()
        assert not isinstance(backend, AnthropicBackend)
        assert not isinstance(backend, DeepSeekBackend)


# --------------------------------------------------------------------------- #
#                  generate() dispatches when backend=anthropic               #
# --------------------------------------------------------------------------- #


def test_generate_with_anthropic_preference_routes_to_anthropic(isolated_test_env):
    """The convenience helper passes prefer='anthropic' through."""
    with patch("stropha.inference.backend.get_backend") as mock_get:
        mock_backend = MagicMock(spec=InferenceBackend)
        mock_backend.generate.return_value = "from-anthropic"
        mock_get.return_value = mock_backend

        out = generate("hi", backend="anthropic")

        mock_get.assert_called_with(prefer="anthropic")
        assert out == "from-anthropic"
