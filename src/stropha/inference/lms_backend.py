"""Internal LMS inference backend using the OpenAI-compatible chat API."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..config import get_config
from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)

_RETRYABLE_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    urllib_error.URLError,
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
    ) -> None:
        cfg = get_config()
        self._default_model = model or cfg.inference.lms_model
        self._base_url = (base_url or cfg.inference.lms_base_url).rstrip("/")
        self._api_key = api_key or cfg.secrets.lms_api_key
        self._timeout = timeout if timeout is not None else cfg.inference.lms_timeout
        self._max_retries = (
            max_retries if max_retries is not None else cfg.inference.lms_max_retries
        )
        self._think = think if think is not None else cfg.inference.lms_think

    @property
    def name(self) -> str:
        return "lms"

    @property
    def identity(self) -> str:
        endpoint_hash = hashlib.sha256(self._base_url.encode()).hexdigest()[:8]
        return (
            f"lms:model={self._default_model}:endpoint={endpoint_hash}"
            f":think={int(self._think)}"
        )

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        if not self._api_key:
            log.warning("lms.generate_skipped", reason="LMS API key not set")
            return None

        body = json.dumps(
            {
                "model": model or self._default_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "think": self._think,
            }
        ).encode("utf-8")

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
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
                    log.warning("lms.generate_failed", reason="no choices", attempt=attempt)
                    return None
                text = ((choices[0].get("message") or {}).get("content") or "").strip()
                return text or None
            except urllib_error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= attempts:
                    log.warning("lms.generate_failed", status=exc.code, attempt=attempt)
                    return None
                delay = 2 ** (attempt - 1)
                retry_after = exc.headers.get("retry-after") if exc.headers else None
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
                    return None
                time.sleep(2 ** (attempt - 1))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("lms.generate_failed", error=type(exc).__name__)
                return None
            except Exception as exc:
                log.warning("lms.generate_failed", error=type(exc).__name__)
                return None
        return None

    def health_check(self) -> tuple[bool, str]:
        if not self.is_available:
            return False, "LMS API key not set (secrets.lms_api_key)"
        try:
            req = urllib_request.Request(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in (data.get("data") or [])]
            if self._default_model in models or not models:
                return True, f"LMS ready with {self._default_model}"
            return False, f"LMS model {self._default_model} not found"
        except urllib_error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, f"LMS auth failed ({exc.code}); check secrets.lms_api_key"
            return False, f"LMS error {exc.code}: {exc.reason}"
        except Exception as exc:
            return False, f"LMS unreachable: {exc}"
