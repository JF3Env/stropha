"""Local MoA Gateway inference backend using its OpenAI-compatible API."""

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

_RETRYABLE_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    urllib_error.URLError,
    OSError,
    TimeoutError,
)


class MoaBackend(InferenceBackend):
    """Generate through a locally running MoA Gateway profile."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        cfg = get_config()
        self._default_model = model or cfg.inference.moa_model
        self._base_url = (base_url or cfg.inference.moa_base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else cfg.secrets.moa_api_key
        self._timeout = timeout if timeout is not None else cfg.inference.moa_timeout
        self._max_retries = (
            max_retries if max_retries is not None else cfg.inference.moa_max_retries
        )

    @property
    def name(self) -> str:
        return "moa"

    @property
    def identity(self) -> str:
        endpoint_hash = hashlib.sha256(self._base_url.encode()).hexdigest()[:8]
        return (
            f"moa:model={self._default_model}:endpoint={endpoint_hash}"
            f":timeout={self._timeout}:retries={self._max_retries}"
        )

    @property
    def is_available(self) -> bool:
        healthy, _ = self.health_check()
        return healthy

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        body = json.dumps(
            {
                "model": model or self._default_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                req = urllib_request.Request(
                    f"{self._base_url}/chat/completions",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                choices = payload.get("choices") or []
                if not choices:
                    log.warning("moa.generate_failed", reason="no choices", attempt=attempt)
                    return None
                text = ((choices[0].get("message") or {}).get("content") or "").strip()
                return text or None
            except urllib_error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= attempts:
                    log.warning("moa.generate_failed", status=exc.code, attempt=attempt)
                    return None
            except _RETRYABLE_NETWORK_ERRORS as exc:
                if attempt >= attempts:
                    log.warning(
                        "moa.generate_failed",
                        error=type(exc).__name__,
                        attempt=attempt,
                    )
                    return None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("moa.generate_failed", error=type(exc).__name__)
                return None
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
        return None

    def health_check(self) -> tuple[bool, str]:
        try:
            req = urllib_request.Request(
                f"{self._base_url}/models",
                headers=self._auth_headers(),
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=min(self._timeout, 5.0)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = [item.get("id") for item in payload.get("data") or []]
            if self._default_model in models:
                return True, f"MoA ready with {self._default_model}"
            return False, f"MoA model {self._default_model} not found"
        except urllib_error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, "MoA authentication failed; check secrets.moa_api_key"
            return False, f"MoA error {exc.code}: {exc.reason}"
        except Exception as exc:
            return False, f"MoA unreachable: {exc}"

    def list_models(self) -> list[str]:
        """Return model aliases advertised by the local gateway."""
        try:
            req = urllib_request.Request(
                f"{self._base_url}/models?limit=1000",
                headers=self._auth_headers(),
                method="GET",
            )
            with urllib_request.urlopen(req, timeout=min(self._timeout, 5.0)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return [str(item["id"]) for item in payload.get("data") or [] if item.get("id")]
        except Exception:
            return []

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
