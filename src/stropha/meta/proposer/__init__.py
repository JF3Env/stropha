"""Proposers for the Meta-Harness outer loop (WIN-186 epic).

Three proposers ship under this package:

- :mod:`bandit` (WIN-189) — zero-LLM Thompson sampling over the manifest.
- :mod:`inference_helper` (WIN-190) — local agentic, recommended default.
- :mod:`agentic` (WIN-191) — cloud agentic, opt-in.

Each module exports a ``propose(archive)`` callable matching the
:data:`stropha.meta.loop.ProposerFn` protocol.

``skill.md`` (WIN-188) is the operational guide every proposer's prompt
references. Read it before tuning prompts.
"""

from __future__ import annotations

from pathlib import Path

SKILL_PATH: Path = Path(__file__).resolve().parent / "skill.md"
"""Path to the operational skill prompt (WIN-188)."""


def load_skill() -> str:
    """Read the skill prompt text from disk.

    The agentic proposers (WIN-190, WIN-191) call this once at
    construction time to embed the skill into their system prompt.
    """
    return SKILL_PATH.read_text(encoding="utf-8")


# Imported last to avoid circular import (these modules import load_skill).
from .agentic import CloudAgenticProposer, make_cloud_agentic_proposer  # noqa: E402
from .bandit import BanditProposer, make_bandit_proposer  # noqa: E402
from .inference_helper import LocalAgenticProposer, make_local_agentic_proposer  # noqa: E402

__all__ = [
    "SKILL_PATH",
    "BanditProposer",
    "CloudAgenticProposer",
    "LocalAgenticProposer",
    "load_skill",
    "make_bandit_proposer",
    "make_cloud_agentic_proposer",
    "make_local_agentic_proposer",
]
