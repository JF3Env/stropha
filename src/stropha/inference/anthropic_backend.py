"""Anthropic inference backend — cloud LLM via the Messages API.

Calls Anthropic's hosted Messages endpoint (``/v1/messages``). Requires
``secrets.anthropic_api_key`` in ``~/.stropha/config.yaml``.

This backend is **opt-in** because it makes outbound network calls. The
The ``CLAUDE.md`` **Local-only by default** invariant is preserved by
:func:`stropha.inference.backend.get_backend` — Anthropic is never
auto-selected, only when ``inference.backend: anthropic`` is set in
``~/.stropha/config.yaml`` (or ``prefer='anthropic'`` is passed explicitly).

Configuration (``~/.stropha/config.yaml``):
    inference:
      backend: anthropic
      anthropic_model: claude-haiku-4-5
      anthropic_base_url: https://api.anthropic.com/v1
      anthropic_version: 2023-06-01

    secrets:
      anthropic_api_key: sk-ant-...

API contract (mirrors :class:`DeepSeekBackend`):
    - ``generate`` MUST swallow every exception and return ``None`` on
      failure. Consumers (HyDE, contextual enricher, query rewrite, …)
      rely on the graceful-degradation contract.
    - Uses stdlib ``urllib.request`` — no new dependencies.
    - Messages API differs from OpenAI-compatible APIs:
        * Header: ``x-api-key`` (not ``Authorization: Bearer``)
        * Header: ``anthropic-version`` (required)
        * Response: ``{"content": [{"type": "text", "text": "..."}]}``
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

# Upper bound for honouring server-sent ``retry-after`` headers. Anthropic
# typically returns small values (seconds), but a misconfigured proxy could
# inject a very large value that would stall the indexer indefinitely.
_MAX_RETRY_AFTER_SECONDS: float = 60.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``retry-after`` header value.

    Per RFC 7231 §7.1.3 the value is either a non-negative decimal integer
    (seconds) or an HTTP-date. Anthropic uses the seconds form. Returns
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


class AnthropicBackend(InferenceBackend):
    """Anthropic Messages API inference backend."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        anthropic_version: str | None = None,
    ) -> None:
        """Initialize the Anthropic backend.

        Args:
            model: Default model id (e.g. ``claude-haiku-4-5``,
                ``claude-sonnet-4-5``, ``claude-opus-4-7``). Falls back
                to ``inference.anthropic_model`` in config.
            base_url: API base URL. Falls back to
                ``inference.anthropic_base_url`` in config.
            api_key: API key. Falls back to ``secrets.anthropic_api_key``
                in ``~/.stropha/config.yaml``.
            timeout: Per-request timeout in seconds. Falls back to
                ``inference.anthropic_timeout`` (default 60s).
            max_retries: Extra attempts on transient network errors.
                Falls back to ``inference.anthropic_max_retries``
                (default 2 → up to 3 total attempts with 1s/2s
                exponential backoff).
            anthropic_version: API version header. Falls back to
                ``inference.anthropic_version``.
        """
        cfg = get_config()
        self._default_model = model or cfg.inference.anthropic_model
        self._base_url = (base_url or cfg.inference.anthropic_base_url).rstrip("/")
        self._api_key = api_key or cfg.secrets.anthropic_api_key
        self._timeout = (
            timeout if timeout is not None else cfg.inference.anthropic_timeout
        )
        self._max_retries = (
            max_retries
            if max_retries is not None
            else cfg.inference.anthropic_max_retries
        )
        self._version = anthropic_version or cfg.inference.anthropic_version

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def identity(self) -> str:
        endpoint_hash = hashlib.sha256(self._base_url.encode()).hexdigest()[:8]
        # Drift identity covers only inputs that change generated text.
        # Transport-reliability controls (timeout, retries) are deliberately
        # excluded: they alter how hard we try to obtain a completion, never
        # the completion itself, and folding them in invalidated every
        # enricher_id — and therefore every chunk in every repo — whenever an
        # operational knob was tuned (WIN-361 follow-up).
        return (
            f"anthropic:model={self._default_model}:endpoint={endpoint_hash}"
            f":version={self._version}"
        )

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
        """Generate text via Anthropic Messages API.

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
                "anthropic.generate_skipped",
                reason="secrets.anthropic_api_key not set in ~/.stropha/config.yaml",
            )
            return None

        target_model = model or self._default_model

        body = json.dumps(
            {
                "model": target_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")

        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                req = urllib_request.Request(
                    f"{self._base_url}/messages",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self._api_key,
                        "anthropic-version": self._version,
                    },
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))

                # Messages API response shape:
                # {"content": [{"type": "text", "text": "..."}], ...}
                content_blocks = payload.get("content") or []
                if not content_blocks:
                    log.warning(
                        "anthropic.generate_failed",
                        reason="no content in response",
                        attempt=attempt,
                    )
                    return None

                # Concatenate all text blocks (Anthropic can return
                # multiple blocks; tool_use blocks are skipped).
                parts: list[str] = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        parts.append(block.get("text") or "")
                text = "".join(parts).strip()
                return text if text else None

            except urllib_error.HTTPError as exc:
                # Status code policy:
                #   - 429 (rate limit): transient. Honour ``retry-after``
                #     header when present; otherwise exponential backoff.
                #     Anthropic returns the header in seconds.
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
                        # Anthropic sets ``retry-after`` in seconds. exc.headers
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
                        "anthropic.generate_retry",
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
                    "anthropic.generate_failed",
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
                        "anthropic.generate_retry",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        backoff_s=backoff,
                        error=str(exc),
                    )
                    time.sleep(backoff)
                    continue
                log.warning(
                    "anthropic.generate_failed",
                    error=str(exc),
                    attempt=attempt,
                )
                return None

            except json.JSONDecodeError as exc:
                # Malformed body — not retryable, the server returned garbage.
                log.warning("anthropic.generate_failed", error=str(exc), attempt=attempt)
                return None

        # Loop exited without returning — should not happen but guard anyway.
        return None

    def health_check(self) -> tuple[bool, str]:
        """Check Anthropic backend health.

        Verifies the API key is present. Does NOT issue any network
        request — Anthropic does not expose a free ``/models`` endpoint
        that we can probe without burning quota, and the key validity
        will surface naturally on the first ``generate`` call.
        """
        if not self.is_available:
            return False, (
                "Anthropic API key not set "
                "(secrets.anthropic_api_key in ~/.stropha/config.yaml)"
            )
        return True, f"Anthropic ready with {self._default_model}"
