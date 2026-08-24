"""Internal LMS inference backend using the OpenAI-compatible chat API."""

from __future__ import annotations

import hashlib
import time
from contextlib import suppress

import httpx

from ..config import get_config
from ..errors import InferenceError, InferenceTimeoutError
from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)

_MISSING_BASE_URL = "LMS endpoint not set (inference.lms_base_url)"

_RETRYABLE_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.RequestError,
    OSError,
    TimeoutError,
)


class LmsBackend(InferenceBackend):
    """LMS chat-completions backend with graceful failure semantics."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        think: bool | None = None,
        max_connections: int | None = None,
        max_keepalive_connections: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        cfg = get_config()
        self._default_model = model or cfg.inference.lms_model
        self._base_url = (base_url or cfg.inference.lms_base_url or "").rstrip("/")
        self._api_key = api_key or cfg.secrets.lms_api_key
        self._timeout = timeout if timeout is not None else cfg.inference.lms_timeout
        self._max_retries = (
            max_retries if max_retries is not None else cfg.inference.lms_max_retries
        )
        self._think = think if think is not None else cfg.inference.lms_think
        connection_limit = max_connections or cfg.inference.lms_max_connections
        keepalive_limit = (
            max_keepalive_connections
            if max_keepalive_connections is not None
            else cfg.inference.lms_max_keepalive_connections
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(
            limits=httpx.Limits(
                max_connections=max(connection_limit, 1),
                max_keepalive_connections=max(min(keepalive_limit, connection_limit), 0),
            )
        )

    @property
    def name(self) -> str:
        return "lms"

    @property
    def identity(self) -> str:
        endpoint_hash = hashlib.sha256(self._base_url.encode()).hexdigest()[:8]
        # Drift identity covers only inputs that change generated text.
        # Transport-reliability controls (timeout, retries) are deliberately
        # excluded: they alter how hard we try to obtain a completion, never
        # the completion itself, and folding them in invalidated every
        # enricher_id — and therefore every chunk in every repo — whenever an
        # operational knob was tuned (WIN-361 follow-up).
        return f"lms:model={self._default_model}:endpoint={endpoint_hash}:think={int(self._think)}"

    @property
    def is_available(self) -> bool:
        return bool(self._base_url and self._api_key and self._api_key.strip())

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
        json_schema: dict[str, object] | None = None,
        think: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        timeout: float | None = None,
        raise_errors: bool = False,
    ) -> str | None:
        if not self._base_url:
            log.warning("lms.generate_skipped", reason=_MISSING_BASE_URL)
            if raise_errors:
                raise InferenceError(_MISSING_BASE_URL)
            return None
        if not self._api_key:
            log.warning("lms.generate_skipped", reason="LMS API key not set")
            if raise_errors:
                raise InferenceError("LMS API key not set")
            return None

        request_body: dict[str, object] = {
            "model": model or self._default_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "think": self._think if think is None else think,
        }
        if json_schema is not None:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "stropha_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        if top_p is not None:
            request_body["top_p"] = top_p
        if top_k is not None:
            request_body["top_k"] = top_k
        if presence_penalty is not None:
            request_body["presence_penalty"] = presence_penalty
        request_timeout = self._timeout if timeout is None else timeout

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    json=request_body,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    log.warning("lms.generate_failed", reason="no choices", attempt=attempt)
                    if raise_errors:
                        raise InferenceError("LMS returned no choices")
                    return None
                text = ((choices[0].get("message") or {}).get("content") or "").strip()
                if not text and raise_errors:
                    raise InferenceError("LMS returned empty content")
                return text or None
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status == 429 or 500 <= status < 600
                if not retryable or attempt >= attempts:
                    log.warning("lms.generate_failed", status=status, attempt=attempt)
                    if raise_errors:
                        raise InferenceError(f"LMS request failed with HTTP {status}") from exc
                    return None
                delay = 2 ** (attempt - 1)
                retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    with suppress(ValueError):
                        delay = min(max(float(retry_after), 0.0), 60.0)
                time.sleep(delay)
            except _RETRYABLE_NETWORK_ERRORS as exc:
                if attempt >= attempts:
                    log.warning(
                        "lms.generate_failed",
                        error=type(exc).__name__,
                        attempt=attempt,
                    )
                    if raise_errors:
                        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                            raise InferenceTimeoutError("LMS request timed out") from exc
                        raise InferenceError(
                            f"LMS request failed with {type(exc).__name__}"
                        ) from exc
                    return None
                time.sleep(2 ** (attempt - 1))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("lms.generate_failed", error=type(exc).__name__)
                if raise_errors:
                    raise InferenceError(
                        f"LMS returned an invalid {type(exc).__name__} response"
                    ) from exc
                return None
            except InferenceError:
                raise
            except Exception as exc:
                log.warning("lms.generate_failed", error=type(exc).__name__)
                if raise_errors:
                    raise InferenceError(
                        f"LMS request failed with {type(exc).__name__}"
                    ) from exc
                return None
        return None

    def health_check(self) -> tuple[bool, str]:
        if not self._base_url:
            return False, _MISSING_BASE_URL
        if not self.is_available:
            return False, "LMS API key not set (secrets.lms_api_key)"
        try:
            response = self._client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            models = [m.get("id") for m in (data.get("data") or [])]
            if self._default_model in models or not models:
                return True, f"LMS ready with {self._default_model}"
            return False, f"LMS model {self._default_model} not found"
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                return False, f"LMS auth failed ({status}); check secrets.lms_api_key"
            return False, f"LMS error {status}: {exc.response.reason_phrase}"
        except Exception as exc:
            return False, f"LMS unreachable: {exc}"
