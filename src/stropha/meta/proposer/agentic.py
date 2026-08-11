"""Cloud agentic proposer (WIN-191) — Anthropic / DeepSeek opt-in.

The cloud path. Pays per-token but routinely beats the local 1.5B
LLM at structured-output tasks like 'emit a single config JSON'.
Opt-in only — never auto-selected by the loop. Per the CLAUDE.md
**Local-only by default** invariant, the
local-only default invariant stays intact; this proposer is loaded
explicitly via ``inference.backend: anthropic`` (or deepseek) in
``~/.stropha/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...inference import InferenceBackend, get_backend
from ...logging import get_logger
from ..search_space import SearchSpace
from . import load_skill
from ._agentic_base import AgenticProposerBase

log = get_logger(__name__)


@dataclass
class CloudAgenticProposer(AgenticProposerBase):
    """Agentic proposer routed through a cloud backend.

    Args (in addition to those on :class:`AgenticProposerBase`):
        prefer_backend: ``"anthropic"`` or ``"deepseek"``. Required —
            the cloud proposer never auto-selects (preserves the
            local-only invariant).
        max_tokens: response budget. Default 1500 (cloud is cheaper
            per token than the local model is, so we can ask for more).
        temperature: sampling temperature.
    """

    prefer_backend: str = "anthropic"
    max_tokens: int = 1500
    temperature: float = 0.3
    _backend: InferenceBackend | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prefer_backend not in ("anthropic", "deepseek"):
            raise ValueError(
                f"cloud agentic proposer needs prefer_backend in "
                f"('anthropic', 'deepseek'); got {self.prefer_backend!r}"
            )
        try:
            self._backend = get_backend(prefer=self.prefer_backend)
        except Exception as exc:
            log.warning(
                "cloud_agentic.backend_unavailable",
                prefer=self.prefer_backend, error=str(exc),
            )
            self._backend = None

    def _call_llm(self, prompt: str) -> str | None:
        if self._backend is None:
            return None
        return self._backend.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


def make_cloud_agentic_proposer(
    space: SearchSpace,
    *,
    prefer_backend: str = "anthropic",
    env: dict[str, str] | None = None,
    fallback_seed: int | None = None,
) -> CloudAgenticProposer:
    """Factory — loads the skill prompt + wires the chosen cloud backend."""
    return CloudAgenticProposer(
        space=space,
        skill=load_skill(),
        backend_name=prefer_backend,
        env=env,
        fallback_seed=fallback_seed,
        prefer_backend=prefer_backend,
    )


__all__ = ["CloudAgenticProposer", "make_cloud_agentic_proposer"]
