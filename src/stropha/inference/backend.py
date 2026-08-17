"""Base inference backend and auto-detection logic.

Auto-detection prefers local MLX when available and falls back to Ollama. Other
local or remote OpenAI-compatible backends require explicit selection.

Configuration (``~/.stropha/config.yaml``):
    inference:
      backend: null  # "mlx", "ollama", "moa", cloud backend, or null
      mlx_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
      ollama_model: qwen2.5-coder:1.5b
      ollama_host: http://localhost:11434  # also reads OLLAMA_HOST env
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from ..config import get_config
from ..logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Cached backend instance
_BACKEND: InferenceBackend | None = None
_INFERENCE_CALLS: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "stropha_inference_calls",
    default=None,
)


@contextmanager
def capture_inference_calls() -> Iterator[list[dict[str, object]]]:
    """Capture centrally dispatched inference calls in the current context."""
    calls: list[dict[str, object]] = []
    token = _INFERENCE_CALLS.set(calls)
    try:
        yield calls
    finally:
        _INFERENCE_CALLS.reset(token)


class InferenceBackend(ABC):
    """Abstract base class for local LLM inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'mlx', 'ollama')."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and ready."""
        ...

    @property
    def identity(self) -> str:
        """Stable non-secret identity for output drift detection."""
        return self.name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            model: Optional model override.

        Returns:
            Generated text, or None on failure.
        """
        ...

    def health_check(self) -> tuple[bool, str]:
        """Check backend health.

        Returns:
            (is_healthy, message) tuple.
        """
        if self.is_available:
            return True, f"{self.name} backend ready"
        return False, f"{self.name} backend not available"

    def close(self) -> None:
        """Release backend resources, if any."""
        return None


def is_mlx_available() -> bool:
    """Check if MLX is available (Apple Silicon only)."""
    try:
        import mlx_lm  # noqa: F401

        return True
    except ImportError:
        return False


def is_ollama_available() -> bool:
    """Check if Ollama daemon is reachable."""
    import json
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    ollama_url = get_config().inference.ollama_host

    try:
        req = urllib_request.Request(
            f"{ollama_url.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib_request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "models" in data
    except (urllib_error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False


def is_deepseek_available() -> bool:
    """Cheap availability probe for DeepSeek.

    Only checks that ``secrets.deepseek_api_key`` is set — we deliberately do
    NOT make a network round-trip on every probe. Real failures surface
    inside :meth:`DeepSeekBackend.generate` as ``None`` with a logged
    error. This matches the local-first invariant (no surprise network
    calls during startup).
    """
    return get_config().secrets.has_deepseek


def is_anthropic_available() -> bool:
    """Cheap availability probe for Anthropic.

    Only checks that ``secrets.anthropic_api_key`` is set — mirrors the
    DeepSeek policy: never auto-probe a cloud endpoint at startup.
    """
    return get_config().secrets.has_anthropic


def is_lms_available() -> bool:
    """Cheap LMS availability probe without making a network request."""
    return get_config().secrets.has_lms


def is_moa_available() -> bool:
    """Check whether the configured local MoA Gateway is reachable."""
    from .moa_backend import MoaBackend

    return MoaBackend().is_available


def get_backend(prefer: str | None = None) -> InferenceBackend:
    """Get the best available inference backend.

    Args:
        prefer: Preferred backend (``'mlx'``, ``'ollama'``, or
            ``'deepseek'``). If not available, falls back to the next
            local option. If ``None``, auto-detects from config.

    Returns:
        An InferenceBackend instance.

    The backend is cached globally for reuse.

    Auto-detection NEVER selects ``deepseek``, ``anthropic``, or ``lms`` — these
    are cloud backends and must be opted into explicitly via
    ``inference.backend: deepseek|anthropic`` in ``~/.stropha/config.yaml`` (or
    ``prefer='deepseek'|'anthropic'``). This preserves the
    ``CLAUDE.md`` **Local-only by default** invariant.
    """
    global _BACKEND

    # Check for preference from config
    if prefer is None:
        config_prefer = get_config().inference.backend
        if config_prefer:
            prefer = config_prefer.lower()

    # Return cached backend if it exists and matches preference (or none asked).
    if _BACKEND is not None and (prefer is None or _BACKEND.name == prefer):
        return _BACKEND

    from .anthropic_backend import AnthropicBackend
    from .deepseek_backend import DeepSeekBackend
    from .lms_backend import LmsBackend
    from .mlx_backend import MlxBackend
    from .moa_backend import MoaBackend
    from .ollama_backend import OllamaBackend

    # Try preferred backend first. The local ``backend`` slot is the
    # abstract base so mypy doesn't pin it to the first concrete class
    # we touch on a given branch.
    backend: InferenceBackend
    if prefer == "mlx":
        backend = MlxBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="mlx")
            return _BACKEND
        log.info("inference.mlx_unavailable_fallback_ollama")
        _BACKEND = OllamaBackend()
        return _BACKEND

    if prefer == "ollama":
        backend = OllamaBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="ollama")
            return _BACKEND
        log.info("inference.ollama_unavailable_fallback_mlx")
        _BACKEND = MlxBackend()
        return _BACKEND

    if prefer == "deepseek":
        backend = DeepSeekBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="deepseek")
            return _BACKEND
        # Configured API key missing → don't silently fall back to a local
        # model (the user explicitly asked for DeepSeek). Cache the
        # instance anyway so `health_check()` reports the actionable
        # YAML-secret guidance. Consumers see ``None`` from
        # ``generate`` per the contract.
        log.warning("inference.deepseek_no_api_key")
        _BACKEND = backend
        return _BACKEND

    if prefer == "anthropic":
        backend = AnthropicBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="anthropic")
            return _BACKEND
        # Same policy as DeepSeek: never silently fall back when the
        # user explicitly asked for Anthropic. Cache so health_check
        # reports actionable guidance.
        log.warning("inference.anthropic_no_api_key")
        _BACKEND = backend
        return _BACKEND

    if prefer == "lms":
        backend = LmsBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="lms")
            return _BACKEND
        log.warning("inference.lms_no_api_key")
        _BACKEND = backend
        return _BACKEND

    if prefer == "moa":
        backend = MoaBackend()
        if backend.is_available:
            _BACKEND = backend
            log.info("inference.backend_selected", backend="moa")
            return _BACKEND
        log.warning("inference.moa_unavailable")
        _BACKEND = backend
        return _BACKEND

    # Auto-detect: prefer MLX (faster, more efficient). DeepSeek and
    # Anthropic are NEVER auto-selected — see docstring.
    if is_mlx_available():
        _BACKEND = MlxBackend()
        log.info("inference.backend_auto_selected", backend="mlx")
        return _BACKEND

    # Fall back to Ollama
    _BACKEND = OllamaBackend()
    log.info("inference.backend_auto_selected", backend="ollama")
    return _BACKEND


def generate(
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: str | None = None,
    backend: str | None = None,
    json_schema: dict[str, object] | None = None,
    think: bool | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    timeout: float | None = None,
    raise_errors: bool = False,
) -> str | None:
    """Generate text using the best available backend.

    Convenience function that auto-selects and caches the backend.

    Args:
        prompt: The input prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        model: Optional model override.
        backend: Optional backend preference ('mlx' or 'ollama').
        json_schema: Optional JSON schema for backends that support structured output.
        think: Optional per-call reasoning override for supporting backends.
        top_p: Optional nucleus-sampling threshold for supporting backends.
        top_k: Optional sampling candidate limit for supporting backends.
        presence_penalty: Optional repetition penalty for supporting backends.
        timeout: Optional per-call timeout in seconds for supporting backends.
        raise_errors: Raise typed LMS errors instead of returning None.

    Returns:
        Generated text, or None on failure.
    """
    started = time.perf_counter()
    be = get_backend(prefer=backend)
    result: str | None = None
    status = "failed"
    try:
        from ..errors import InferenceError
        from .lms_backend import LmsBackend

        if isinstance(be, LmsBackend):
            result = be.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                json_schema=json_schema,
                think=think,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                timeout=timeout,
                raise_errors=raise_errors,
            )
        else:
            if timeout is not None:
                raise InferenceError(
                    f"{be.name} inference does not support a per-call timeout"
                )
            result = be.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
            )
        if raise_errors and result is None:
            raise InferenceError(f"{be.name} inference returned no output")
        status = "completed" if result else "failed"
        return result
    except Exception:
        status = "error"
        raise
    finally:
        calls = _INFERENCE_CALLS.get()
        if calls is not None:
            calls.append(
                {
                    "started_perf": started,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "backend": be.name,
                    "model": model,
                    "max_tokens": max_tokens,
                    "status": status,
                    "output_chars": len(result) if result else 0,
                }
            )


def reset_backend() -> None:
    """Reset the cached backend. Useful for testing."""
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.close()
    _BACKEND = None
