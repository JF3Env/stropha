"""Code-search proposer (WIN-196 — EPIC 1.6 Phase 5).

The plain agentic proposers (WIN-190 local, WIN-191 cloud) only choose
values along the **config** axis: which adapter to use at each stage,
which retrieval feature to enable, what cache size to try next. The
**code** proposer can also propose *new adapter source files*, which
the ``ArtifactStore`` materialises under
``candidates/<id>/adapters/<rel_path>``.

Why this matters
----------------

Per ``docs/architecture/stropha-meta-harness.md`` §Phase 5: the paper's
"search over harness code" is the full Meta-Harness regime. Letting the
proposer write code (not just pick from a manifest) opens up:

- a new adapter that does X — e.g. a tokenizer-aware chunker the
  current registry doesn't ship;
- a fix to a retrieval-stream edge case the proposer noticed in traces;
- an experimental enricher that fuses two existing strategies.

The safety arguments (paper §5, RFC §7 risk table):

- **Inspectable.** The candidate's source lands on disk in plain text;
  ``git diff`` shows everything the proposer emitted.
- **Sandboxed.** WIN-198 will load each candidate's adapters in a
  subprocess so a runaway proposed adapter cannot kill the loop.
- **Test-gated.** WIN-197 will run ``uv run pytest -q`` *before*
  scoring; any candidate that breaks the suite is rejected.
- **Production-isolated.** ``src/stropha/adapters/`` is never written
  to. The proposer emits relative paths; the store rejects anything
  that resolves outside ``candidates/<id>/adapters/``. WIN-200 will add
  end-to-end safety tests that confirm this invariant.

Output contract
---------------

The LLM is asked to return ONE JSON object::

    {
      "config": { "<dim>": <value>, ... },
      "code_artifacts": { "<rel_path>": "<python source>", ... }
    }

Both keys are optional. When ``config`` is omitted the whole object is
treated as a config dict (back-compat with the plain agentic proposer
prompt). When ``code_artifacts`` is omitted, the proposer behaves
exactly like its plain agentic parent.

Surviving artifacts are stuffed back into the returned config dict
under :data:`stropha.meta.store.CODE_ARTIFACTS_KEY`, where the store's
``register_candidate`` pops them. The loop never sees the key — the
``ProposerFn -> dict`` protocol is preserved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ...logging import get_logger
from ..code_candidate import (
    MAX_CODE_ARTIFACT_BYTES,
    MAX_CODE_ARTIFACTS,
    CodeCandidateError,
    validate_code_candidate,
    validate_code_candidate_source,
)
from ..fitness import ScoreCard
from ..search_space import SearchSpace
from ..store import CODE_ARTIFACTS_KEY, CandidateRecord
from . import load_skill
from ._agentic_base import (
    AgenticProposerBase,
    _clamp_to_space,
    _extract_json,
    _parse_proposal,
)
from .agentic import CloudAgenticProposer
from .history import estimate_tokens
from .inference_helper import LocalAgenticProposer

log = get_logger(__name__)
_RERANKER_PATH_RE = re.compile(
    r"adapters/reranker/[A-Za-z0-9][A-Za-z0-9_-]*\.py"
)


# A proposed adapter source MUST contain this token. The registry decorator
# is what wires the adapter into the stage; a file that doesn't decorate
# anything is dead code at best, and at worst a sign the LLM forgot the
# contract. Cheap sanity gate; the subprocess sandbox in WIN-198 is the
# real safety net.
REGISTER_ADAPTER_TOKEN = "@register_adapter"


# Path constraints — kept here so the proposer rejects obvious abuse
# before any byte hits the store. The store re-validates everything
# defensively (defence-in-depth, see _validate_artifact_rel_path in
# meta/store.py), but the proposer-side validator avoids the round-trip
# entirely for the common case of a malformed LLM response.
MAX_ARTIFACT_SOURCE_BYTES = MAX_CODE_ARTIFACT_BYTES
MAX_ARTIFACTS_PER_CANDIDATE = MAX_CODE_ARTIFACTS


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_valid_artifact_path(rel_path: str) -> bool:
    """Return True when ``rel_path`` is a safe ``adapters/...<.py>`` path.

    Rejects: absolute paths, backslashes, drive prefixes, empty / dot /
    parent segments, any non-whitelisted character, any path not ending
    in ``.py``, and any path not rooted at ``adapters/``.

    The ``adapters/`` prefix is part of the contract surfaced to the
    LLM (see :func:`build_code_prompt_tail`). The store re-checks the
    prefix in :meth:`ArtifactStore._write_code_artifacts` and rejects
    anything that resolves outside ``<candidate>/adapters/``.

    We duplicate the proposer-side checks so the proposer can drop bad
    paths quietly rather than failing the whole proposal on one bad
    entry — the LLM still gets to evaluate its config dict; only the
    malformed artifact is dropped.
    """
    if not isinstance(rel_path, str):
        return False
    return _RERANKER_PATH_RE.fullmatch(rel_path) is not None


def _is_valid_artifact_source(source: str) -> bool:
    """Return True when ``source`` looks like a real adapter module.

    Full AST and identity validation needs the artifact path and candidate
    config, so this compatibility helper checks only type, size, and syntax.
    """
    if not isinstance(source, str) or not source.strip():
        return False
    if len(source.encode("utf-8")) > MAX_ARTIFACT_SOURCE_BYTES:
        return False
    try:
        validate_code_candidate_source(source)
    except CodeCandidateError:
        return False
    return True


def _validate_artifacts(
    raw: Any, config: dict[str, Any] | None = None
) -> dict[str, str]:
    """Return the subset of ``raw`` that survives every validation rule.

    Bad entries are logged at warning level and dropped — a single
    malformed file MUST NOT abort the whole proposal. The proposer
    should still get to evaluate its config; the LLM can do better next
    iteration.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        log.warning("code_proposer.bad_artifacts_type", got=type(raw).__name__)
        return {}
    if config is None:
        if len(raw) != 1:
            return {}
        rel_path, source = next(iter(raw.items()))
        return (
            {rel_path: source}
            if _is_valid_artifact_path(rel_path) and _is_valid_artifact_source(source)
            else {}
        )
    try:
        spec = validate_code_candidate(config, raw)
    except CodeCandidateError as exc:
        log.warning("code_proposer.bad_code_candidate", error=str(exc))
        return {}
    return {spec.rel_path: spec.source}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


# Appended to the standard agentic prompt. Keeps the original instructions
# (single JSON object, stay within manifest, no prose) and just adds the
# code_artifacts option.
_CODE_PROMPT_TAIL = """\

## Code-search option (Phase 5)

You MAY also propose exactly ONE brand-new reranker adapter Python file in
addition to your config dict. Wrap them inside a top-level
``code_artifacts`` object mapping a relative path (under
``adapters/reranker/<name>.py``) to the file's full source. The source
MUST decorate exactly one top-level class with
``@register_adapter(stage="reranker", name="<name>")`` and the config's
``reranker`` value MUST select that same name.

When you propose code:

- Use forward-slash relative paths only. No leading ``/``, no ``..``,
  no backslashes.
- Stick to ``adapters/reranker/<name>.py`` shape.
- Keep each file under {max_bytes} bytes.
- A maximum of {max_files} files per proposal.

If you DO NOT have a clear code change to make, omit
``code_artifacts`` entirely. The loop will still evaluate your config.

Schema:

    {{
      "config": {{ "<dim>": <value>, ... }},
      "hypothesis": {{ "statement": "...", "evidence_candidate_ids": [...] }},
      "code_artifacts": {{ "<rel_path>": "<full python source>", ... }}
    }}

Return ONE JSON object — and ONLY that object — and nothing outside it.
"""


def build_code_prompt_tail() -> str:
    """Return the formatted Phase-5 tail appended to the base prompt."""
    return _CODE_PROMPT_TAIL.format(
        max_bytes=MAX_ARTIFACT_SOURCE_BYTES,
        max_files=MAX_ARTIFACTS_PER_CANDIDATE,
    )


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class CodeProposerMixin:
    """Shared override of ``__call__`` for any code-search proposer.

    The mixin is applied to :class:`LocalAgenticProposer` and
    :class:`CloudAgenticProposer` (the two concrete agentic proposers).
    It depends on attributes provided by :class:`AgenticProposerBase`
    (``space``, ``backend_name``, ``env``, ``_fallback_propose``, …)
    and overrides:

    - :meth:`_build_prompt` to append the Phase-5 schema.
    - :meth:`__call__` to parse both ``config`` and ``code_artifacts``.

    Concrete classes (:class:`LocalCodeProposer`,
    :class:`CloudCodeProposer`) inherit from the mixin AND the
    underlying agentic proposer to compose the behaviour.
    """

    def _build_prompt(  # type: ignore[override]
        self,
        archive: list[tuple[CandidateRecord, ScoreCard]],
    ) -> str:
        # Defer to the agentic base for the standard prompt body, then
        # append the code-artifacts schema. The base method exists on
        # AgenticProposerBase; the mixin assumes it via MRO.
        base = AgenticProposerBase._build_prompt(self, archive)  # type: ignore[arg-type]
        return base + build_code_prompt_tail()

    def _reserved_prompt_tokens(self) -> int:
        return estimate_tokens(build_code_prompt_tail())

    def __call__(  # type: ignore[override]
        self,
        archive: Iterable[tuple[CandidateRecord, ScoreCard]],
    ) -> dict[str, Any]:
        archive_list = list(archive)
        prompt = self._build_prompt(archive_list)
        try:
            raw = self._call_llm(prompt)  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning(
                "code_proposer.llm_failed",
                backend=getattr(self, "backend_name", "?"),
                error=str(exc),
            )
            return self._fallback_propose(archive_list)  # type: ignore[attr-defined]
        if raw is None or not raw.strip():
            log.warning(
                "code_proposer.empty_response",
                backend=getattr(self, "backend_name", "?"),
            )
            return self._fallback_propose(archive_list)  # type: ignore[attr-defined]

        try:
            obj = _extract_json(raw)
        except ValueError as exc:
            log.warning(
                "code_proposer.bad_json",
                backend=getattr(self, "backend_name", "?"),
                error=str(exc),
                preview=raw[:120],
            )
            return self._fallback_propose(archive_list)  # type: ignore[attr-defined]

        # Two shapes accepted:
        # 1. Phase-5: ``{"config": {...}, "code_artifacts": {...}}``
        # 2. Back-compat: ``{...}`` interpreted as the plain config dict.
        # Heuristic: if "config" key is itself a dict, treat object as a
        # wrapper. Otherwise treat the whole thing as the config.
        config_raw: Any
        artifacts_raw: Any
        try:
            proposal = _parse_proposal(obj)
        except ValueError as exc:
            log.warning("code_proposer.bad_proposal", error=str(exc))
            return self._fallback_propose(archive_list)  # type: ignore[attr-defined]
        self.__dict__["last_proposal"] = proposal
        config_raw = proposal.config
        artifacts_raw = proposal.code_artifacts

        space: SearchSpace = self.space  # type: ignore[attr-defined]
        cleaned = _clamp_to_space(config_raw, space)
        env = getattr(self, "env", None) or {}
        constrained = space.apply_constraints(cleaned, env=env)

        validation_config = dict(config_raw) if isinstance(config_raw, dict) else {}
        artifacts = _validate_artifacts(artifacts_raw, validation_config)
        if artifacts:
            constrained["reranker"] = validation_config["reranker"]
            constrained[CODE_ARTIFACTS_KEY] = artifacts
            log.info(
                "code_proposer.artifacts_attached",
                backend=getattr(self, "backend_name", "?"),
                count=len(artifacts),
            )
        return constrained


# ---------------------------------------------------------------------------
# Concrete proposers
# ---------------------------------------------------------------------------


@dataclass
class LocalCodeProposer(CodeProposerMixin, LocalAgenticProposer):
    """Code-search proposer routed through the local inference backend.

    Inherits :class:`LocalAgenticProposer`'s backend selection
    (``prefer_backend`` → mlx / ollama / None=auto) and overlays
    :class:`CodeProposerMixin`'s code-aware ``__call__``.
    """

    # ``LocalAgenticProposer`` already declares everything we need;
    # the mixin overrides ``_build_prompt`` + ``__call__``.
    enable_code: bool = field(default=True)


@dataclass
class CloudCodeProposer(CodeProposerMixin, CloudAgenticProposer):
    """Code-search proposer routed through a cloud backend.

    Same shape as :class:`LocalCodeProposer` but uses
    :class:`CloudAgenticProposer` (Anthropic or DeepSeek). Opt-in only —
    requires explicit ``inference.backend`` configuration; never auto-
    selected per the CLAUDE.md **Local-only by default** invariant.
    """

    enable_code: bool = field(default=True)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_local_code_proposer(
    space: SearchSpace,
    *,
    prefer_backend: str | None = None,
    env: dict[str, str] | None = None,
    fallback_seed: int | None = None,
) -> LocalCodeProposer:
    """Factory — loads the skill prompt + wires a local backend.

    Mirrors :func:`make_local_agentic_proposer`'s signature so callers
    can swap the two with a single import change.
    """
    return LocalCodeProposer(
        space=space,
        skill=load_skill(),
        backend_name=prefer_backend or "auto",
        env=env,
        fallback_seed=fallback_seed,
        prefer_backend=prefer_backend,
    )


def make_cloud_code_proposer(
    space: SearchSpace,
    *,
    prefer_backend: str = "anthropic",
    env: dict[str, str] | None = None,
    fallback_seed: int | None = None,
) -> CloudCodeProposer:
    """Factory — loads the skill prompt + wires a cloud backend.

    Mirrors :func:`make_cloud_agentic_proposer` so the call sites stay
    symmetric.
    """
    return CloudCodeProposer(
        space=space,
        skill=load_skill(),
        backend_name=prefer_backend,
        env=env,
        fallback_seed=fallback_seed,
        prefer_backend=prefer_backend,
    )


__all__ = [
    "CODE_ARTIFACTS_KEY",
    "MAX_ARTIFACTS_PER_CANDIDATE",
    "MAX_ARTIFACT_SOURCE_BYTES",
    "REGISTER_ADAPTER_TOKEN",
    "CloudCodeProposer",
    "CodeProposerMixin",
    "LocalCodeProposer",
    "build_code_prompt_tail",
    "make_cloud_code_proposer",
    "make_local_code_proposer",
]
