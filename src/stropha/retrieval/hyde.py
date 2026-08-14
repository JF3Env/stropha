"""HyDE — Hypothetical Document Embeddings (Phase 3 §6.3).

Idea (Gao et al., 2022): generate a hypothetical answer to the query
using an LLM, then embed *that answer* instead of the query itself. The
hypothetical document lives in the same space as your indexed chunks,
so the dense stream sees a much better neighbour match than it does
from the raw question.

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to the raw query when:
- No inference backend available
- The generation times out
- The generated text is empty
- ``retrieval_features.hyde_enabled`` is false in the unified config

Configuration (``~/.stropha/config.yaml``):
    retrieval_features:
      hyde_enabled: true
      hyde_code_tuned: true

Drift-safety note: HyDE changes only the *query path*. It does not
mutate stored chunks or their embeddings, so toggling it on and off is
zero-cost — no re-index needed.

Usage:

    from stropha.retrieval.hyde import maybe_hyde_rewrite

    text_to_embed = maybe_hyde_rewrite(user_query) or user_query
    qvec = embedder.embed_query(text_to_embed)
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Literal

from ..config import get_config
from ..logging import get_logger
from .query_classify import QueryClassification, classify_query

log = get_logger(__name__)

# Code-tuned prompts for different query types
# The hypothetical document should look like actual code/docs from the codebase

_PROMPT_CODE_FUNCTION = """\
Write a hypothetical code snippet that would answer this question about a function or method.
Include: function signature, brief docstring, and key implementation lines.

Question: {query}

```
"""

_PROMPT_CODE_CLASS = """\
Write a hypothetical code snippet that would answer this question about a class or type.
Include: class definition, key attributes, and important method signatures.

Question: {query}

```
"""

_PROMPT_CODE_HOW = """\
Write a hypothetical code comment or docstring that would explain how this works.
Be specific about the implementation approach, algorithms used, or data flow.

Question: {query}

/**
"""

_PROMPT_CODE_WHERE = """\
Write a hypothetical code snippet showing where this functionality is implemented.
Include: the file path as a comment, class/function name, and key lines.

Question: {query}

// File: 
"""

_PROMPT_CODE_TEST = """\
Write a hypothetical test case that would test this functionality.
Include: test method name, setup, assertion, and what it verifies.

Question: {query}

@Test
"""

_PROMPT_CODE_ERROR = """\
Write a hypothetical code snippet showing how this error or bug would be fixed.
Include: the problematic code pattern and the corrected version.

Question: {query}

// Before (bug):
"""

_PROMPT_CODE_DEFAULT = """\
Write a 2-4 line hypothetical code snippet or documentation that answers this developer question.
Match the style of real production code with proper naming and structure.

Question: {query}

"""

# Query pattern detection for selecting the best prompt
_PATTERNS = [
    (re.compile(r'\b(function|method|def|fn)\b', re.I), _PROMPT_CODE_FUNCTION),
    (re.compile(r'\b(class|type|interface|struct|enum)\b', re.I), _PROMPT_CODE_CLASS),
    (re.compile(r'^how\b', re.I), _PROMPT_CODE_HOW),
    (re.compile(r'\b(where|find|locate)\b', re.I), _PROMPT_CODE_WHERE),
    (re.compile(r'\b(tests?|specs?|asserts?|assertions?)\b', re.I), _PROMPT_CODE_TEST),
    (re.compile(r'\b(bug|fix|error|issue|debug)\b', re.I), _PROMPT_CODE_ERROR),
]

# Default prompt when no code-specific query pattern matches.
_DEFAULT_PROMPT = (
    "Write a 2-3 sentence imaginary code snippet or doc paragraph that would "
    "directly answer this developer question:\n\n"
    "QUESTION: {query}\n\n"
    "Reply with just the snippet/paragraph — no preamble, no \"sure here is\"."
)
_POLICY_VERSION = "behavior-description-v2"
_BEHAVIOR_PREFIXES = ("when ", "after ", "before ", "given ", "once ")
_STRUCTURAL_PREFIXES = (
    "find callers",
    "find tests",
    "tests for",
    "trace ",
    "what calls",
    "what relates",
    "what uses",
    "who calls",
)

HydeReason = Literal[
    "success",
    "disabled",
    "empty_query",
    "symbol_query",
    "structural_query",
    "not_natural",
    "too_short",
    "not_behavior",
    "empty_response",
    "backend_error",
]


@dataclass(frozen=True)
class HydeResult:
    """Hypothetical document plus selection and generation measurements."""

    text: str | None
    reason: HydeReason
    generation_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    latency_ms: float = 0.0


def policy_id(*, code_tuned: bool = True) -> str:
    """Stable identity for selection and every possible prompt template."""
    prompts = (
        _PROMPT_CODE_FUNCTION,
        _PROMPT_CODE_CLASS,
        _PROMPT_CODE_HOW,
        _PROMPT_CODE_WHERE,
        _PROMPT_CODE_TEST,
        _PROMPT_CODE_ERROR,
        _PROMPT_CODE_DEFAULT,
    ) if code_tuned else (_DEFAULT_PROMPT,)
    prompt_hash = hashlib.sha256("\0".join(prompts).encode("utf-8")).hexdigest()[:8]
    return f"{_POLICY_VERSION}:code={int(code_tuned)}:p={prompt_hash}"


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _has_obvious_identifier(classification: QueryClassification) -> bool:
    first = classification.tokens[0] if classification.tokens else ""
    for token in classification.symbol_tokens:
        if "-" in token:
            continue
        if token.isupper() or (token.endswith("s") and token[:-1].isupper()):
            continue
        if token == first and token.istitle():
            continue
        return True
    return False


def _policy_reason(query: str) -> HydeReason | None:
    stripped = query.strip()
    if not stripped:
        return "empty_query"

    classification = classify_query(stripped)
    if _has_obvious_identifier(classification):
        return "symbol_query"
    if not classification.is_natural:
        return "not_natural"
    if len(classification.tokens) < 5:
        return "too_short"

    lowered = stripped.lower()
    if lowered.startswith(_STRUCTURAL_PREFIXES):
        return "structural_query"
    is_behavior = lowered.startswith(_BEHAVIOR_PREFIXES) or "what happens" in lowered
    if not is_behavior:
        return "not_behavior"
    return None


def _select_prompt(query: str) -> str:
    """Select the best code-tuned prompt based on query patterns."""
    for pattern, prompt in _PATTERNS:
        if pattern.search(query):
            return prompt
    return _PROMPT_CODE_DEFAULT


def _enabled() -> bool:
    """Check whether unified configuration enables HyDE."""
    return get_config().retrieval_features.hyde_enabled


def _use_code_tuned() -> bool:
    """Check if code-tuned prompts should be used (default: yes)."""
    return get_config().retrieval_features.hyde_code_tuned


def _prompt_template(query: str, *, code_tuned: bool | None = None) -> str:
    """Get the appropriate prompt template for the query."""
    # Use code-tuned prompts by default
    if _use_code_tuned() if code_tuned is None else code_tuned:
        return _select_prompt(query)
    
    return _DEFAULT_PROMPT


def generate_hypothetical(
    query: str,
    *,
    force_enabled: bool = False,
    code_tuned: bool | None = None,
    apply_policy: bool = True,
) -> HydeResult:
    """Selectively generate a hypothetical document and never raise.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Caller is expected to use the returned text *instead of* the raw
    query for the dense embedding step; the BM25 / symbol streams keep
    using the raw query (HyDE doesn't help literal-match lanes).

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check. Used when caller
            already determined HyDE should run (e.g., config flag).
        code_tuned: Explicit prompt selection. ``None`` inherits the unified
            runtime setting; candidate evaluation passes a bool to stay isolated.

    The returned result always explains whether generation ran or was skipped.
    """
    if not force_enabled and not _enabled():
        return HydeResult(None, "disabled")
    reason = _policy_reason(query) if apply_policy else None
    if not query.strip():
        reason = "empty_query"
    if reason is not None:
        return HydeResult(None, reason)

    effective_code_tuned = _use_code_tuned() if code_tuned is None else code_tuned
    prompt = _prompt_template(query, code_tuned=effective_code_tuned).format(query=query)
    
    log.debug("hyde.prompt_selected", query=query[:50], prompt_start=prompt[:30])

    started = time.perf_counter()
    try:
        from ..inference import generate

        text = generate(
            prompt,
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.warning("hyde.generation_failed", error=str(exc))
        return HydeResult(
            None,
            "backend_error",
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not text or not text.strip():
        log.info("hyde.skip_generation_failed")
        return HydeResult(
            None,
            "empty_response",
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    # Clean up the response - remove markdown code fences if present
    raw_text = text
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    # Cap to keep dense embedding time bounded.
    text = text[:2000]
    if not text.strip():
        return HydeResult(
            None,
            "empty_response",
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            estimated_output_tokens=_estimate_tokens(raw_text),
            latency_ms=latency_ms,
        )
    return HydeResult(
        text,
        "success",
        generation_calls=1,
        estimated_input_tokens=_estimate_tokens(prompt),
        estimated_output_tokens=_estimate_tokens(raw_text),
        latency_ms=latency_ms,
    )


def maybe_hyde_rewrite(
    query: str,
    *,
    force_enabled: bool = False,
    code_tuned: bool | None = None,
) -> str | None:
    """Return hypothetical text while preserving the legacy non-selective API."""
    return generate_hypothetical(
        query,
        force_enabled=force_enabled,
        code_tuned=code_tuned,
        apply_policy=False,
    ).text


def get_prompt_type(query: str) -> str:
    """Return the prompt type that would be used for this query (for debugging)."""
    for pattern, prompt in _PATTERNS:
        if pattern.search(query):
            # Extract the prompt name from the variable name
            for name, val in globals().items():
                if val is prompt and name.startswith("_PROMPT_CODE_"):
                    return name.replace("_PROMPT_CODE_", "").lower()
    return "default"
