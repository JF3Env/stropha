"""Query Rewriting — transform user queries into more searchable form.

Idea: LLM rewrites natural-language queries into forms that are more likely
to match code identifiers and technical terms. Works for all streams (dense,
sparse, symbol), unlike HyDE which only helps dense.

Examples:
    "onde tem teste pra fsrs" → "FSRS calculator test unit test FsrsCalculatorTest"
    "como funciona mastery" → "mastery streak transition REVIEW phase acquisition"

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to original query when:
- No inference backend available
- The generation times out
- The generated text is empty
- ``retrieval_features.query_rewrite_enabled`` is false in the unified config

Configuration (``~/.stropha/config.yaml``):
    retrieval_features:
      query_rewrite_enabled: true

Usage:
    from stropha.retrieval.query_rewrite import maybe_rewrite_query

    rewritten = maybe_rewrite_query(user_query) or user_query
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Literal

from ..config import get_config
from ..logging import get_logger
from .query_classify import classify_query
from .query_router import QueryIntent, route_query

log = get_logger(__name__)

_DEFAULT_PROMPT = (
    "Rewrite this developer question into search terms that would match code.\n"
    "Include: class names, method names, technical terms, file patterns.\n"
    "Keep the original intent but expand with likely code identifiers.\n\n"
    "QUESTION: {query}\n\n"
    "SEARCH TERMS (just the terms, no explanation):"
)

DEFAULT_MAX_TERMS = 6
_POLICY_VERSION = "short-natural-v1"

RewriteReason = Literal[
    "success",
    "disabled",
    "empty_query",
    "symbol_query",
    "structural_query",
    "not_natural",
    "too_long",
    "empty_response",
    "backend_error",
]


@dataclass(frozen=True)
class QueryRewriteResult:
    """Rewrite output plus the measurements needed for promotion evidence."""

    query: str | None
    reason: RewriteReason
    generation_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    latency_ms: float = 0.0


def policy_id(max_terms: int = DEFAULT_MAX_TERMS) -> str:
    """Stable identity for the deterministic policy and generation prompt."""
    prompt_hash = hashlib.sha256(_DEFAULT_PROMPT.encode("utf-8")).hexdigest()[:8]
    return f"{_POLICY_VERSION}:max={max_terms}:p={prompt_hash}"


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _policy_reason(query: str, max_terms: int) -> RewriteReason | None:
    stripped = query.strip()
    if not stripped:
        return "empty_query"

    classification = classify_query(stripped)
    if classification.is_symbol_like:
        return "symbol_query"

    routed = route_query(stripped, use_llm=False)
    if routed.intent in {
        QueryIntent.FIND_CALLERS,
        QueryIntent.FIND_TESTS,
        QueryIntent.FIND_RELATED,
        QueryIntent.TRACE_FEATURE,
    }:
        return "structural_query"
    if routed.intent is QueryIntent.GET_SYMBOL:
        return "symbol_query"
    if not classification.is_natural:
        return "not_natural"
    if len(classification.tokens) > max_terms:
        return "too_long"
    return None


def _enabled() -> bool:
    """Check whether unified configuration enables query rewriting."""
    return get_config().retrieval_features.query_rewrite_enabled


def _prompt_template() -> str:
    """Get prompt template (currently hardcoded)."""
    return _DEFAULT_PROMPT


def rewrite_query(
    query: str,
    *,
    force_enabled: bool = False,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> QueryRewriteResult:
    """Selectively rewrite a short natural-language query and never raise."""
    if not force_enabled and not _enabled():
        return QueryRewriteResult(query=None, reason="disabled")

    max_terms = max(1, min(20, max_terms))
    reason = _policy_reason(query, max_terms)
    if reason is not None:
        return QueryRewriteResult(query=None, reason=reason)

    prompt = _prompt_template().format(query=query)
    started = time.perf_counter()
    try:
        from ..inference import generate

        text = generate(
            prompt,
            max_tokens=128,
            temperature=0.0,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.warning("query_rewrite.generation_failed", error=str(exc))
        return QueryRewriteResult(
            query=None,
            reason="backend_error",
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not text or not text.strip():
        log.info("query_rewrite.skip_generation_failed")
        return QueryRewriteResult(
            query=None,
            reason="empty_response",
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    combined = f"{query} {text}"[:500]
    return QueryRewriteResult(
        query=combined,
        reason="success",
        generation_calls=1,
        estimated_input_tokens=_estimate_tokens(prompt),
        estimated_output_tokens=_estimate_tokens(text),
        latency_ms=latency_ms,
    )


def maybe_rewrite_query(
    query: str,
    *,
    force_enabled: bool = False,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> str | None:
    """Return a rewritten query optimized for code search, or None on fail/skip.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Unlike HyDE, the rewritten query is used for ALL streams (dense, sparse,
    symbol), not just dense. The goal is to expand natural language into
    code-like terms.

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check. Used when caller
            already determined query rewriting should run (e.g., config flag).

    Never raises. Returns None on failure, letting caller use original query.
    """
    return rewrite_query(
        query,
        force_enabled=force_enabled,
        max_terms=max_terms,
    ).query
