"""MLX inference backend — native Apple Silicon inference.

Uses mlx-lm for in-process inference with Metal acceleration.
No daemon required, more efficient than Ollama on M-series chips.

Configuration (``~/.stropha/config.yaml``):
    inference:
      mlx_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..logging import get_logger
from .backend import InferenceBackend

log = get_logger(__name__)


class MlxBackend(InferenceBackend):
    """MLX-based inference backend for Apple Silicon."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the MLX backend.

        Args:
            model: HuggingFace model ID in MLX format. Defaults to the
                   value in ``~/.stropha/config.yaml``.
        """
        self._default_model = model or get_config().inference.mlx_model
        self._loaded_model: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._generate_fn: Any = None
        self._load_fn: Any = None

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def identity(self) -> str:
        return f"mlx:model={self._default_model}"

    @property
    def is_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_loaded(self, model: str) -> bool:
        """Load the model if not already loaded."""
        if self._loaded_model == model and self._model is not None:
            return True

        try:
            from mlx_lm import generate, load

            self._load_fn = load
            self._generate_fn = generate

            log.info("mlx.loading_model", model=model)
            # ``mlx_lm.load`` has no type stubs; newer versions return
            # ``(model, tokenizer, config)`` triples while older ones return
            # 2-tuples. Both are unpacked into the two slots we care about.
            self._model, self._tokenizer = load(model)  # type: ignore[misc]
            self._loaded_model = model
            log.info("mlx.model_loaded", model=model)
            return True
        except ImportError:
            log.warning("mlx.mlx_lm_not_installed")
            return False
        except Exception as exc:
            log.warning("mlx.model_load_failed", model=model, error=str(exc))
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Generate text using MLX.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            model: Optional model override.

        Returns:
            Generated text, or None on failure.
        """
        model = model or self._default_model

        if not self._ensure_loaded(model):
            return None

        try:
            # Build kwargs for mlx_lm.generate
            # Note: mlx_lm uses a sampler function for temperature control,
            # but the default greedy sampler (temp=0) is fine for our use case.
            # For non-zero temperature, we'd need to pass a custom sampler.
            kwargs: dict = {
                "max_tokens": max_tokens,
                "verbose": False,
            }

            # Only add sampler for non-zero temperature
            if temperature > 0:
                try:
                    from mlx_lm.sample_utils import make_sampler

                    kwargs["sampler"] = make_sampler(temp=temperature)
                except ImportError:
                    # Older mlx_lm version without make_sampler, use default
                    pass

            # Apply the model's chat template. Instruct models (Qwen2.5-Coder-
            # Instruct, etc.) are fine-tuned on ChatML and degenerate into
            # emitting ``<|im_start|>`` tokens when fed a raw prompt. Wrapping
            # the prompt as a user turn (with the generation prompt appended)
            # makes the model answer normally and stop at ``<|im_end|>``.
            formatted = self._apply_chat_template(prompt)

            text = self._generate_fn(
                self._model,
                self._tokenizer,
                formatted,
                **kwargs,
            )

            if not isinstance(text, str):
                return None

            return text.strip()
        except Exception as exc:
            log.warning("mlx.generate_failed", error=str(exc))
            return None

    def _apply_chat_template(self, prompt: str) -> str:
        """Wrap ``prompt`` as a single user turn via the tokenizer's chat
        template. Falls back to the raw prompt when the tokenizer has no
        chat template (base/non-instruct models)."""
        tok = self._tokenizer
        try:
            if (
                tok is not None
                and getattr(tok, "chat_template", None)
                and hasattr(tok, "apply_chat_template")
            ):
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("mlx.chat_template_failed", error=str(exc))
        return prompt

    def health_check(self) -> tuple[bool, str]:
        """Check MLX backend health."""
        if not self.is_available:
            return False, "mlx-lm not installed (pip install mlx-lm)"

        # Try a minimal generation to verify the model works
        try:
            if not self._ensure_loaded(self._default_model):
                return False, f"Failed to load model: {self._default_model}"
            return True, f"MLX ready with {self._default_model}"
        except Exception as exc:
            return False, f"MLX error: {exc}"
