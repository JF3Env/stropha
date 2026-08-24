"""DeepSeek inference backend — cloud LLM via OpenAI-compatible API.

Calls DeepSeek's hosted chat-completions endpoint (``/v1/chat/completions``).
Requires ``secrets.deepseek_api_key`` in ``~/.stropha/config.yaml``.

This backend is **opt-in** because it makes outbound network calls. The
The ``CLAUDE.md`` **Local-only by default** invariant is preserved by
:func:`stropha.inference.backend.get_backend` — DeepSeek is never
auto-selected, only when ``inference.backend: deepseek`` is set in
``~/.stropha/config.yaml`` (or ``prefer='deepseek'`` is passed explicitly).

Configuration (``~/.stropha/config.yaml``):
    inference:
      backend: deepseek
      deepseek_model: deepseek-chat
      deepseek_base_url: https://api.deepseek.com/v1

    secrets:
      deepseek_api_key: sk-...

API contract (mirrors :class:`OllamaBackend`):
    - ``generate`` MUST swallow every exception and return ``None`` on
      failure. Consumers (HyDE, contextual enricher, query rewrite, …)
      rely on the graceful-degradation contract.
    - Uses stdlib ``urllib.request`` — no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..config import get_config
from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)

# Errors that are worth retrying — network-level transient failures.
# HTTPError (4xx/5xx) is handled separately because 4xx semantics depend
# on the status code (429 is retryable, 401/400 are not).
_RETRYABLE_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    urllib_error.URLError,
    OSError,
    TimeoutError,
)

# Upper bound for honouring server-sent ``retry-after`` headers. DeepSeek
# typically returns small values (seconds), but a misconfigured proxy could
# inject a very large value that would stall the indexer indefinitely.
_MAX_RETRY_AFTER_SECONDS: float = 60.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``retry-after`` header value.

    Per RFC 7231 §7.1.3 the value is either a non-negative decimal integer
    (seconds) or an HTTP-date. DeepSeek uses the seconds form. Returns
    ``None`` for missing / malformed input so the caller can fall back to
    exponential backoff.
    """
    if not value:
        return None
    try:
        secs = float(value.strip())
    except (TypeError, ValueError):
        return None
    if secs < 0:
        return None
    return min(secs, _MAX_RETRY_AFTER_SECONDS)


class DeepSeekBackend(InferenceBackend):
    """DeepSeek HTTPS-based inference backend (OpenAI-compatible)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the DeepSeek backend.

        Args:
            model: Default model id (e.g. ``deepseek-chat``,
                ``deepseek-reasoner``). Falls back to
                ``inference.deepseek_model`` in config.
            base_url: API base URL. Falls back to
                ``inference.deepseek_base_url`` in config.
            api_key: Bearer token. Falls back to ``secrets.deepseek_api_key``
                in ``~/.stropha/config.yaml``.
            timeout: Per-request timeout in seconds. Falls back to
                ``inference.deepseek_timeout`` (default 120s — the
                China-routed DeepSeek endpoint can be slow under load).
            max_retries: Extra attempts on transient network errors.
                Falls back to ``inference.deepseek_max_retries``
                (default 2 → up to 3 total attempts with 1s/2s
                exponential backoff).
        """
        cfg = get_config()
        self._default_model = model or cfg.inference.deepseek_model
        self._base_url = (base_url or cfg.inference.deepseek_base_url).rstrip("/")
        self._api_key = api_key or cfg.secrets.deepseek_api_key
        self._timeout = timeout if timeout is not None else cfg.inference.deepseek_timeout
        self._max_retries = (
            max_retries if max_retries is not None else cfg.inference.deepseek_max_retries
        )

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def identity(self) -> str:
        endpoint_hash = hashlib.sha256(self._base_url.encode()).hexdigest()[:8]
        # Drift identity covers only inputs that change generated text.
        # Transport-reliability controls (timeout, retries) are deliberately
        # excluded: they alter how hard we try to obtain a completion, never
        # the completion itself, and folding them in invalidated every
        # enricher_id — and therefore every chunk in every repo — whenever an
        # operational knob was tuned (WIN-361 follow-up).
        return f"deepseek:model={self._default_model}:endpoint={endpoint_hash}"

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
            log.warning(
                "deepseek.generate_skipped",
                reason="secrets.deepseek_api_key not set in ~/.stropha/config.yaml",
            )
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

        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
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
                    log.warning(
                        "deepseek.generate_failed",
                        reason="no choices in response",
                        attempt=attempt,
                    )
                    return None
                message = choices[0].get("message") or {}
                text = (message.get("content") or "").strip()
                return text if text else None

            except urllib_error.HTTPError as exc:
                # Status code policy:
                #   - 429 (rate limit): transient. Honour ``retry-after``
                #     header when present; otherwise exponential backoff.
                #     DeepSeek returns the header in seconds.
                #   - 5xx (server error): transient. Exponential backoff.
                #   - Other 4xx (401, 400, 404): permanent. Bad key / bad
                #     request / wrong model — no amount of retrying fixes it.
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""

                is_rate_limit = exc.code == 429
                is_server_error = exc.code >= 500
                is_retryable = is_rate_limit or is_server_error

                if is_retryable and attempt < max_attempts:
                    if is_rate_limit:
                        # DeepSeek sets ``retry-after`` in seconds. exc.headers
                        # is an http.client.HTTPMessage; ``.get`` returns None
                        # when absent.
                        header_value = (
                            exc.headers.get("retry-after") if exc.headers else None
                        )
                        backoff = _parse_retry_after(header_value)
                        if backoff is None:
                            # Server didn't tell us how long; fall back to
                            # exponential backoff (1s, 2s, 4s, …).
                            backoff = 2 ** (attempt - 1)
                    else:
                        backoff = 2 ** (attempt - 1)
                    log.warning(
                        "deepseek.generate_retry",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        backoff_s=backoff,
                        status=exc.code,
                        reason=exc.reason,
                        retry_after_header=(
                            exc.headers.get("retry-after")
                            if (is_rate_limit and exc.headers)
                            else None
                        ),
                    )
                    time.sleep(backoff)
                    continue
                log.warning(
                    "deepseek.generate_failed",
                    status=exc.code,
                    reason=exc.reason,
                    body=err_body[:500],
                    attempt=attempt,
                )
                return None

            except _RETRYABLE_NETWORK_ERRORS as exc:
                # Connection error / timeout / DNS — retry with backoff.
                if attempt < max_attempts:
                    backoff = 2 ** (attempt - 1)
                    log.warning(
                        "deepseek.generate_retry",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        backoff_s=backoff,
                        error=str(exc),
                    )
                    time.sleep(backoff)
                    continue
                log.warning(
                    "deepseek.generate_failed",
                    error=str(exc),
                    attempt=attempt,
                )
                return None

            except json.JSONDecodeError as exc:
                # Malformed body — not retryable, the server returned garbage.
                log.warning("deepseek.generate_failed", error=str(exc), attempt=attempt)
                return None

        # Loop exited without returning — should not happen but guard anyway.
        return None

    def health_check(self) -> tuple[bool, str]:
        """Check DeepSeek backend health.

        Verifies the API key is present and the endpoint responds to a
        minimal ``models`` listing call. Does NOT issue a generation
        request (to avoid burning quota on every health probe).
        """
        if not self.is_available:
            return False, (
                "DeepSeek API key not set "
                "(secrets.deepseek_api_key in ~/.stropha/config.yaml)"
            )

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
                return False, (
                    f"DeepSeek auth failed ({exc.code}); check "
                    "secrets.deepseek_api_key in ~/.stropha/config.yaml"
                )
            return False, f"DeepSeek error {exc.code}: {exc.reason}"
        except Exception as exc:
            return False, f"DeepSeek unreachable: {exc}"
