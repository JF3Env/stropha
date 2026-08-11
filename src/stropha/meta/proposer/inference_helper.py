"""Local agentic proposer (WIN-190) — uses stropha.inference backends.

The RECOMMENDED default for the Meta-Harness loop on a developer
laptop. Uses whatever local LLM ``stropha.inference.get_backend``
resolves (mlx on Apple Silicon, ollama elsewhere). No outbound
network call — preserves the CLAUDE.md **Local-only by default** invariant.

If the local backend is unavailable (no mlx, no ollama daemon), the
proposer transparently falls back to the bandit. The loop never sees
a None.
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
class LocalAgenticProposer(AgenticProposerBase):
    """Agentic proposer powered by a local stropha.inference backend.

    Args (in addition to those on :class:`AgenticProposerBase`):
        prefer_backend: passed to ``stropha.inference.get_backend``
            (``"mlx"`` / ``"ollama"`` / None for auto).
        max_tokens: generation budget for the LLM response.
        temperature: sampling temperature; default 0.2 for slight
            diversity without losing coherence.
    """

    prefer_backend: str | None = None
    max_tokens: int = 800
    temperature: float = 0.2
    _backend: InferenceBackend | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            self._backend = get_backend(prefer=self.prefer_backend)
        except Exception as exc:
            log.warning(
                "local_agentic.backend_unavailable",
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


def make_local_agentic_proposer(
    space: SearchSpace,
    *,
    prefer_backend: str | None = None,
    env: dict[str, str] | None = None,
    fallback_seed: int | None = None,
) -> LocalAgenticProposer:
    """Factory — loads the skill prompt + wires the backend."""
    return LocalAgenticProposer(
        space=space,
        skill=load_skill(),
        backend_name=prefer_backend or "auto",
        env=env,
        fallback_seed=fallback_seed,
        prefer_backend=prefer_backend,
    )


__all__ = ["LocalAgenticProposer", "make_local_agentic_proposer"]
