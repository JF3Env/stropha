"""Shared base for the agentic proposers (WIN-190 local, WIN-191 cloud).

Both proposers follow the same shape:

1. Serialise the recent archive into a compact prompt slice.
2. Ask the LLM, with the skill prompt as system context, to emit ONE
   JSON object representing the next candidate config.
3. Parse the JSON; clamp every dimension against the search space;
   apply the constraint engine; return the config.

The only difference between local and cloud is the
:class:`stropha.inference.InferenceBackend` instance the proposer
holds. The base class owns prompt assembly, JSON parsing, and
fallback-to-bandit when the LLM call fails — local and cloud just
inherit and pick a backend.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ...logging import get_logger
from ..fitness import ScoreCard
from ..search_space import SearchSpace
from ..store import CandidateRecord
from .bandit import BanditProposer
from .history import estimate_tokens, pack_history

log = get_logger(__name__)


MAX_PROMPT_BYTES = 50_000
MAX_PROMPT_TOKENS = MAX_PROMPT_BYTES // 4
MAX_ARCHIVE_ENTRIES = 10


class Hypothesis(BaseModel):
    """Testable rationale carried with an agentic proposal."""

    statement: str
    evidence_candidate_ids: list[str] = Field(default_factory=list)
    evidence_trace_ids: list[str] = Field(default_factory=list)
    expected_objective_deltas: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None

    @field_validator("statement")
    @classmethod
    def _statement_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("hypothesis statement must not be blank")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_is_probability(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class Proposal(BaseModel):
    """Typed LLM response; callers still return its config to the loop."""

    config: dict[str, Any] = Field(default_factory=dict)
    hypothesis: Hypothesis | None = None
    code_artifacts: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


def _parse_proposal(obj: dict[str, Any]) -> Proposal:
    """Parse the typed envelope, retaining the legacy flat-config shape."""
    envelope_keys = {"config", "hypothesis", "code_artifacts", "usage"}
    if envelope_keys.intersection(obj):
        try:
            return Proposal.model_validate(obj)
        except ValidationError as exc:
            raise ValueError(f"invalid typed proposal: {exc}") from exc
    return Proposal(config=obj)


def _summarise_card(card: ScoreCard) -> dict[str, Any]:
    """Compact ScoreCard summary for the prompt."""
    return {
        "quality": round(card.quality, 4),
        "context_efficiency": round(card.context_efficiency, 4),
        "outcome_score": round(card.outcome_score, 4),
        "latency_ms": round(card.latency_ms, 1),
        "tier_counts": dict(card.tier_counts),
    }


def _summarise_archive(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    *,
    limit: int = MAX_ARCHIVE_ENTRIES,
) -> list[dict[str, Any]]:
    """Most-recent-N candidates as JSON-safe dicts."""
    out: list[dict[str, Any]] = []
    for record, card in archive[-limit:]:
        out.append({
            "id": record.id,
            "config": record.config,
            "scores": _summarise_card(card),
            "label": record.label,
        })
    return out


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM response.

    LLMs frequently wrap their output in fences, prefix with prose, or
    emit trailing commentary. This helper:

    1. Strips ```json``` fences.
    2. Picks the first complete ``{...}`` span.
    3. Tries ``json.loads`` directly first, falls back to trimming
       trailing commas.

    Returns the parsed dict, or raises ``ValueError`` if nothing in
    ``text`` looks like valid JSON.
    """
    cleaned = text.strip()
    # Drop common fence patterns.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # Find the first balanced { ... } span.
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("response contains no JSON object")
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("response has unbalanced braces")
    candidate = cleaned[start:end]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Trailing comma forgiveness — drop ", }" and ", ]" sequences.
        trimmed = re.sub(r",(\s*[}\]])", r"\1", candidate)
        obj = json.loads(trimmed)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root is not a dict (got {type(obj).__name__})")
    return obj


def _clamp_to_space(config: dict[str, Any], space: SearchSpace) -> dict[str, Any]:
    """Force ``config`` keys to land within the search-space bounds.

    A free-form LLM might propose ``walker: my-cool-walker`` that isn't
    in the registry. We drop unknown choices back to the dimension's
    default rather than letting the evaluator choke later.
    """
    cleaned: dict[str, Any] = {}
    for dim in space.dimensions:
        value = config.get(dim.name, dim.default)
        if dim.type == "choice" and dim.choices and value not in dim.choices:
            value = dim.default if dim.default in dim.choices else dim.choices[0]
        elif dim.type == "int" and dim.range is not None:
            try:
                v = int(value)
            except (TypeError, ValueError):
                v = int(dim.default) if dim.default is not None else dim.range[0]
            lo, hi = dim.range
            v = max(lo, min(hi, v))
            value = v
        elif dim.type == "bool":
            value = bool(value)
        cleaned[dim.name] = value
    return cleaned


@dataclass
class AgenticProposerBase:
    """Common surface for local + cloud agentic proposers.

    Concrete subclasses provide ``_call_llm(prompt) -> str`` which
    talks to a specific backend; the base owns prompt assembly + JSON
    parsing + fallback-to-bandit when the LLM call fails.
    """

    space: SearchSpace
    skill: str  # operational skill prompt (WIN-188)
    backend_name: str
    env: dict[str, str] | None = None
    fallback_seed: int | None = None
    _fallback: BanditProposer | None = field(default=None, init=False, repr=False)
    last_proposal: Proposal | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._fallback = BanditProposer(
            space=self.space,
            seed=self.fallback_seed,
            env=self.env,
        )

    # ------------------------------------------------------------------ #
    # ProposerFn protocol                                                #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        archive: Iterable[tuple[CandidateRecord, ScoreCard]],
    ) -> dict[str, Any]:
        archive_list = list(archive)
        self.last_proposal = None
        prompt = self._build_prompt(archive_list)
        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            log.warning(
                "agentic_proposer.llm_failed",
                backend=self.backend_name, error=str(exc),
            )
            return self._fallback_propose(archive_list)
        if raw is None or not raw.strip():
            log.warning(
                "agentic_proposer.empty_response",
                backend=self.backend_name,
            )
            return self._fallback_propose(archive_list)
        try:
            proposal = _parse_proposal(_extract_json(raw))
        except ValueError as exc:
            log.warning(
                "agentic_proposer.bad_json",
                backend=self.backend_name, error=str(exc),
                preview=raw[:120],
            )
            return self._fallback_propose(archive_list)
        self.last_proposal = proposal
        clamped = _clamp_to_space(proposal.config, self.space)
        return self.space.apply_constraints(clamped, env=self.env or {})

    # ------------------------------------------------------------------ #
    # Internal hooks                                                     #
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str) -> str | None:
        """Subclass hook — return the LLM's raw response text."""
        raise NotImplementedError

    def _fallback_propose(
        self,
        archive: list[tuple[CandidateRecord, ScoreCard]],
    ) -> dict[str, Any]:
        """Bandit fallback when the LLM is unhappy."""
        assert self._fallback is not None
        return self._fallback(archive)

    def _build_prompt(
        self,
        archive: list[tuple[CandidateRecord, ScoreCard]],
    ) -> str:
        manifest = _summarise_manifest(self.space)
        prefix = (
            f"{self.skill}\n\n"
            "## Manifest (valid choices + ranges)\n\n"
            f"{json.dumps(manifest, indent=2)}\n\n"
        )
        suffix = (
            "## Your task\n\n"
            "Return one typed proposal JSON object — and ONLY that object — "
            "with `config` and a testable `hypothesis`. The hypothesis may "
            "include `evidence_candidate_ids`, `evidence_trace_ids`, "
            "`expected_objective_deltas`, and `confidence` (0..1). "
            "Stay within the manifest. Legacy flat configuration JSON remains "
            "accepted, but the typed shape is preferred.\n\n"
            "Schema: {\"config\": {...}, \"hypothesis\": {\"statement\": "
            "\"...\", \"evidence_candidate_ids\": [...], "
            "\"evidence_trace_ids\": [...], \"expected_objective_deltas\": "
            "{...}, \"confidence\": 0.0}}\n"
        )
        fixed_tokens = (
            estimate_tokens(prefix)
            + estimate_tokens(suffix)
            + self._reserved_prompt_tokens()
        )
        history = pack_history(
            archive,
            budget_tokens=MAX_PROMPT_TOKENS,
            fixed_tokens=fixed_tokens,
        )
        payload = (
            prefix
            + "## Full history bundle\n\n"
            + json.dumps(history.to_dict(), separators=(",", ":"), default=str)
            + "\n\n"
            + suffix
        )
        if len(payload.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("token estimate exceeded the hard prompt byte limit")
        return payload

    def _reserved_prompt_tokens(self) -> int:
        """Tokens a subclass will append after the standard prompt."""
        return 0


def _summarise_manifest(space: SearchSpace) -> dict[str, Any]:
    """Compact, LLM-friendly representation of the search space."""
    stages = {}
    features = {}
    for dim in space.dimensions:
        view = {"indexing": dim.indexing, "default": dim.default}
        if dim.type == "choice":
            view["choices"] = list(dim.choices)
        if dim.type == "int":
            view["type"] = "int"
            if dim.range is not None:
                view["range"] = list(dim.range)
            if dim.step is not None:
                view["step"] = dim.step
        if dim.type == "bool":
            view["type"] = "bool"
        if dim.kind == "stage":
            stages[dim.name] = view
        else:
            features[dim.name] = view
    return {"stages": stages, "features": features}


__all__ = [
    "MAX_ARCHIVE_ENTRIES",
    "MAX_PROMPT_BYTES",
    "MAX_PROMPT_TOKENS",
    "AgenticProposerBase",
    "Hypothesis",
    "Proposal",
]
