"""LLM inference backends.

This module provides a unified interface for LLM inference,
supporting five backends:

- ``mlx``       — local, Apple Silicon native (default when available)
- ``ollama``    — local, HTTP daemon (fallback)
- ``deepseek``  — cloud, OpenAI-compatible API (opt-in, requires
  ``DEEPSEEK_API_KEY``)
- ``anthropic`` — cloud, Messages API (opt-in, requires
   ``ANTHROPIC_API_KEY``)
- ``lms``       — internal OpenAI-compatible API (opt-in)

Per ``CLAUDE.md`` §3.6, cloud backends are never auto-selected — they
must be requested explicitly via ``inference.backend`` in
``stropha.yaml`` or ``prefer='deepseek'|'anthropic'`` in code.

Usage:
    from stropha.inference import get_backend, generate

    # Auto-detect best LOCAL backend
    backend = get_backend()
    response = backend.generate("Your prompt here")

    # Explicit cloud opt-in
    backend = get_backend(prefer="anthropic")

    # Or use the convenience helper
    response = generate("Your prompt here")
"""

from .anthropic_backend import AnthropicBackend
from .backend import (
    InferenceBackend,
    capture_inference_calls,
    generate,
    get_backend,
    is_anthropic_available,
    is_deepseek_available,
    is_lms_available,
    is_mlx_available,
    is_moa_available,
    is_ollama_available,
)
from .deepseek_backend import DeepSeekBackend
from .lms_backend import LmsBackend
from .mlx_backend import MlxBackend
from .moa_backend import MoaBackend
from .ollama_backend import OllamaBackend

__all__ = [
    "AnthropicBackend",
    "DeepSeekBackend",
    "InferenceBackend",
    "LmsBackend",
    "MlxBackend",
    "MoaBackend",
    "OllamaBackend",
    "capture_inference_calls",
    "generate",
    "get_backend",
    "is_anthropic_available",
    "is_deepseek_available",
    "is_lms_available",
    "is_mlx_available",
    "is_moa_available",
    "is_ollama_available",
]
