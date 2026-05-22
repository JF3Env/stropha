"""DeepSeek inference backend — cloud LLM via OpenAI-compatible API.

Calls DeepSeek's hosted chat-completions endpoint (``/v1/chat/completions``).
Requires the ``DEEPSEEK_API_KEY`` environment variable.

This backend is **opt-in** because it makes outbound network calls. The
``CLAUDE.md`` §3.6 "local-only by default" invariant is preserved by
:func:`stropha.inference.backend.get_backend` — DeepSeek is never
auto-selected, only when ``inference.backend: deepseek`` is set in
``stropha.yaml`` (or ``prefer='deepseek'`` is passed explicitly).

Configuration (stropha.yaml):
    inference:
      backend: deepseek
      deepseek_model: deepseek-chat
      deepseek_base_url: https://api.deepseek.com/v1

Authentication (env only — never YAML):
    DEEPSEEK_API_KEY=sk-...

API contract (mirrors :class:`OllamaBackend`):
    - ``generate`` MUST swallow every exception and return ``None`` on
      failure. Consumers (HyDE, contextual enricher, query rewrite, …)
      rely on the graceful-degradation contract.
    - Uses stdlib ``urllib.request`` — no new dependencies.
"""

from __future__ import annotations

import json
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..config import get_config
from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)


class DeepSeekBackend(InferenceBackend):
    """DeepSeek HTTPS-based inference backend (OpenAI-compatible)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the DeepSeek backend.

        Args:
            model: Default model id (e.g. ``deepseek-chat``,
                ``deepseek-reasoner``). Falls back to
                ``inference.deepseek_model`` in config.
            base_url: API base URL. Falls back to
                ``inference.deepseek_base_url`` in config.
            api_key: Bearer token. Falls back to ``DEEPSEEK_API_KEY``
                env var (via ``secrets.deepseek_api_key``).
            timeout: Per-request timeout in seconds. Cloud round-trips
                are slower than localhost so default is 60s vs 30s for
                Ollama.
        """
        cfg = get_config()
        self._default_model = model or cfg.inference.deepseek_model
        self._base_url = (base_url or cfg.inference.deepseek_base_url).rstrip("/")
        self._api_key = api_key or cfg.secrets.deepseek_api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def is_available(self) -> bool:
        """Cheap availability probe — just checks the API key is set.

        We deliberately do NOT make a network round-trip here. Probing a
        cloud endpoint on every ``is_available`` call would slow down
        startup and burn quota. Real failures surface as ``generate`` →
        ``None`` with a structured log entry.
        """
        return bool(self._api_key and self._api_key.strip())

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text via DeepSeek chat completions.

        Args:
            prompt: User prompt (sent as a single ``user`` message).
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature (0.0 = deterministic).
            model: Optional per-call model override.

        Returns:
            The generated text (stripped), or ``None`` on any failure
            (missing key, network error, bad response, empty content).
        """
        if not self._api_key:
            log.warning("deepseek.generate_skipped", reason="DEEPSEEK_API_KEY not set")
            return None

        target_model = model or self._default_model

        body = json.dumps(
            {
                "model": target_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")

        try:
            req = urllib_request.Request(
                f"{self._base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            choices = payload.get("choices") or []
            if not choices:
                log.warning("deepseek.generate_failed", reason="no choices in response")
                return None
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            return text if text else None

        except urllib_error.HTTPError as exc:
            # Common: 401 (bad key), 402 (out of credits), 429 (rate limit).
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            log.warning(
                "deepseek.generate_failed",
                status=exc.code,
                reason=exc.reason,
                body=err_body[:500],
            )
            return None
        except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            log.warning("deepseek.generate_failed", error=str(exc))
            return None

    def health_check(self) -> tuple[bool, str]:
        """Check DeepSeek backend health.

        Verifies the API key is present and the endpoint responds to a
        minimal ``models`` listing call. Does NOT issue a generation
        request (to avoid burning quota on every health probe).
        """
        if not self.is_available:
            return False, "DeepSeek API key not set (DEEPSEEK_API_KEY)"

        try:
            req = urllib_request.Request(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # OpenAI-compat shape: {"data": [{"id": "deepseek-chat", ...}, ...]}
                models = [m.get("id") for m in (data.get("data") or [])]
                if self._default_model in models or not models:
                    return True, f"DeepSeek ready with {self._default_model}"
                return (
                    True,
                    f"DeepSeek reachable (model {self._default_model} not "
                    f"in /models list; will attempt at generate-time)",
                )
        except urllib_error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, f"DeepSeek auth failed ({exc.code}); check DEEPSEEK_API_KEY"
            return False, f"DeepSeek error {exc.code}: {exc.reason}"
        except Exception as exc:
            return False, f"DeepSeek unreachable: {exc}"
