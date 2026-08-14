"""Multi-Query Expansion — generate paraphrases and fuse results via RRF.

Idea: Instead of searching with a single query, generate 3-5 paraphrases
of the original query using an LLM, search with each paraphrase, and
fuse the results via Reciprocal Rank Fusion (RRF).

This increases recall by capturing different ways users might ask
about the same concept:

    "how to authenticate users"
    → Paraphrases:
       1. "authentication login flow user session"
       2. "security middleware auth check validation"
       3. "user credentials token JWT verification"
    → Search each, RRF fuse

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Falls back to single-query search when:
- No inference backend available
- Generation times out
- Generated paraphrases are empty
- ``retrieval_features.multi_query_enabled`` is false in the unified config

Caching: Paraphrases are cached in-memory by query to avoid repeated
LLM calls for the same query.

Configuration (``~/.stropha/config.yaml``):
    retrieval_features:
      multi_query_enabled: true
      multi_query_count: 3  # number of paraphrases

Usage:
    from stropha.retrieval.multi_query import MultiQueryExpander

    expander = MultiQueryExpander()
    paraphrases = expander.expand(query)
    # Returns list of 3-5 paraphrases including the original query
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
    "Generate {num_paraphrases} different search queries that a developer might use "
    "to find the same code as this query. Each paraphrase should focus on different "
    "aspects: synonyms, related concepts, or technical terms.\n\n"
    "Original query: {query}\n\n"
    "Return ONLY the paraphrases, one per line, no numbers or bullets:"
)
DEFAULT_MIN_TERMS = 7
_POLICY_VERSION = "broad-conceptual-v1"
_CONCEPTUAL_TERMS = frozenset(
    {
        "architecture",
        "architectural",
        "design",
        "interact",
        "interaction",
        "interactions",
        "lifecycle",
        "system",
        "workflow",
    }
)

MultiQueryReason = Literal[
    "success",
    "cache_hit",
    "disabled",
    "empty_query",
    "symbol_query",
    "structural_query",
    "not_natural",
    "too_short",
    "not_conceptual",
    "empty_response",
    "no_valid_paraphrases",
    "backend_error",
]


@dataclass(frozen=True)
class MultiQueryResult:
    """Expanded queries plus generation and deduplication measurements."""

    queries: tuple[str, ...]
    reason: MultiQueryReason
    requested_paraphrases: int
    raw_paraphrases: int = 0
    duplicate_paraphrases: int = 0
    generation_calls: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    latency_ms: float = 0.0


def policy_id(min_terms: int = DEFAULT_MIN_TERMS) -> str:
    """Stable identity for deterministic selection and the generation prompt."""
    prompt_hash = hashlib.sha256(_DEFAULT_PROMPT.encode("utf-8")).hexdigest()[:8]
    return f"{_POLICY_VERSION}:min={min_terms}:p={prompt_hash}"


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _policy_reason(query: str, min_terms: int) -> MultiQueryReason | None:
    stripped = query.strip()
    if not stripped:
        return "empty_query"

    classification = classify_query(stripped)
    if classification.symbol_tokens:
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
    if len(classification.tokens) < min_terms:
        return "too_short"

    lowered = [token.lower() for token in classification.tokens]
    conceptual = lowered[0] in {"how", "why"} or bool(_CONCEPTUAL_TERMS.intersection(lowered))
    if not conceptual:
        return "not_conceptual"
    return None

# In-memory cache for paraphrases (keyed by query hash)
_PARAPHRASE_CACHE: dict[str, tuple[str, ...]] = {}
_CACHE_MAX_SIZE = 1000


def _enabled() -> bool:
    """Check whether unified configuration enables multi-query expansion."""
    return get_config().retrieval_features.multi_query_enabled


def _prompt_template() -> str:
    """Get prompt template (currently hardcoded)."""
    return _DEFAULT_PROMPT


def _num_paraphrases() -> int:
    """Get the configured number of paraphrases."""
    return get_config().retrieval_features.multi_query_count


def _cache_key(query: str, num_paraphrases: int | None = None) -> str:
    """Generate a cache key for the query."""
    count = num_paraphrases or _num_paraphrases()
    payload = f"{query.strip().lower()}\0{count}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_cached(query: str, num_paraphrases: int) -> tuple[str, ...] | None:
    """Get cached paraphrases for a query."""
    key = _cache_key(query, num_paraphrases)
    return _PARAPHRASE_CACHE.get(key)


def _set_cached(query: str, num_paraphrases: int, paraphrases: tuple[str, ...]) -> None:
    """Cache paraphrases for a query, with LRU eviction."""
    global _PARAPHRASE_CACHE
    if len(_PARAPHRASE_CACHE) >= _CACHE_MAX_SIZE:
        # Simple eviction: remove oldest half
        keys = list(_PARAPHRASE_CACHE.keys())
        for k in keys[: len(keys) // 2]:
            del _PARAPHRASE_CACHE[k]
    key = _cache_key(query, num_paraphrases)
    _PARAPHRASE_CACHE[key] = paraphrases


def expand_query(
    query: str,
    *,
    force_enabled: bool = False,
    num_paraphrases: int | None = None,
    min_terms: int | None = None,
    apply_policy: bool = True,
) -> MultiQueryResult:
    """Selectively expand a broad conceptual query and never raise.

    Uses MLX on Apple Silicon, falls back to Ollama on other platforms.

    Args:
        query: The user's search query.
        force_enabled: If True, skip the _enabled() check.
        num_paraphrases: Number of paraphrases to generate (default: 3).
        apply_policy: Select only broad conceptual queries when true.

    Returns a structured result whose queries always include the original first.
    """
    n = max(1, min(5, num_paraphrases or _num_paraphrases()))
    if not force_enabled and not _enabled():
        return MultiQueryResult((query,), "disabled", n)

    effective_min_terms = min_terms or get_config().retrieval_features.multi_query_min_terms
    effective_min_terms = max(1, min(30, effective_min_terms))
    reason = _policy_reason(query, effective_min_terms) if apply_policy else None
    if not query.strip():
        reason = "empty_query"
    if reason is not None:
        return MultiQueryResult((query,), reason, n)

    # Check cache first
    cached = _get_cached(query, n)
    if cached is not None:
        log.debug("multi_query.cache_hit", query=query[:50])
        return MultiQueryResult(cached, "cache_hit", n)

    prompt = _prompt_template().format(query=query, num_paraphrases=n)

    started = time.perf_counter()
    try:
        from ..inference import generate

        text = generate(
            prompt,
            max_tokens=256,
            temperature=0.7,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.warning("multi_query.generation_failed", error=str(exc))
        return MultiQueryResult(
            (query,),
            "backend_error",
            n,
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not text or not text.strip():
        log.info("multi_query.skip_generation_failed")
        return MultiQueryResult(
            (query,),
            "empty_response",
            n,
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            latency_ms=latency_ms,
        )

    # Parse paraphrases (one per line)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Filter out lines that are too short or look like metadata
    paraphrases = [
        line
        for line in lines
        if len(line) > 5 and not line.startswith(("1.", "2.", "3.", "-", "*", "#"))
    ]

    if not paraphrases:
        log.info("multi_query.no_valid_paraphrases")
        return MultiQueryResult(
            (query,),
            "no_valid_paraphrases",
            n,
            raw_paraphrases=len(lines),
            generation_calls=1,
            estimated_input_tokens=_estimate_tokens(prompt),
            estimated_output_tokens=_estimate_tokens(text),
            latency_ms=latency_ms,
        )

    # Always include original query first, then paraphrases
    # Dedupe and limit
    seen = {query.lower()}
    result = [query]
    duplicates = 0
    for p in paraphrases:
        normalized = p.lower()
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        if len(result) < n + 1:
            result.append(p[:200])  # Cap individual paraphrase length

    log.debug(
        "multi_query.generated",
        original=query[:50],
        count=len(result) - 1,
    )

    # Cache the result
    queries = tuple(result)
    _set_cached(query, n, queries)

    return MultiQueryResult(
        queries,
        "success",
        n,
        raw_paraphrases=len(lines),
        duplicate_paraphrases=duplicates,
        generation_calls=1,
        estimated_input_tokens=_estimate_tokens(prompt),
        estimated_output_tokens=_estimate_tokens(text),
        latency_ms=latency_ms,
    )


def generate_paraphrases(
    query: str,
    *,
    force_enabled: bool = False,
    num_paraphrases: int | None = None,
    min_terms: int | None = None,
) -> list[str]:
    """Return expanded queries, preserving the legacy list-returning API."""
    return list(
        expand_query(
            query,
            force_enabled=force_enabled,
            num_paraphrases=num_paraphrases,
            min_terms=min_terms,
            apply_policy=False,
        ).queries
    )


class MultiQueryExpander:
    """Stateful wrapper for multi-query expansion with configuration."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        num_paraphrases: int = 3,
    ) -> None:
        """Initialize the expander.

        Args:
            enabled: If True, multi-query expansion is active.
            num_paraphrases: Number of paraphrases to generate.
        """
        self._enabled = enabled
        self._num_paraphrases = num_paraphrases

    def expand(self, query: str) -> list[str]:
        """Expand query into multiple paraphrases.

        Returns:
            List of queries including the original. Length 1 if disabled
            or on failure.
        """
        return generate_paraphrases(
            query,
            force_enabled=self._enabled,
            num_paraphrases=self._num_paraphrases,
        )

    @property
    def is_enabled(self) -> bool:
        """Check explicit or unified-config multi-query enablement."""
        return self._enabled or _enabled()


def clear_cache() -> None:
    """Clear the paraphrase cache. Useful for testing."""
    global _PARAPHRASE_CACHE
    _PARAPHRASE_CACHE = {}
