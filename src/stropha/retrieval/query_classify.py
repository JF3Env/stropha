"""Query classification for adaptive retrieval.

Classifies queries as symbol-like (code identifiers) or natural-language
to enable adaptive weighting of retrieval streams.

Symbol-like queries benefit from higher lexical/symbol stream weights,
while natural-language queries benefit from dense semantic matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Query classification types."""
    
    SYMBOL = "symbol"        # Code identifier (CamelCase, snake_case, qualified)
    NATURAL = "natural"      # Natural language description
    MIXED = "mixed"          # Contains both symbol and natural language tokens


@dataclass
class QueryClassification:
    """Result of query classification."""
    
    query_type: QueryType
    symbol_ratio: float  # 0.0 to 1.0 — ratio of symbol-like tokens
    tokens: list[str]    # Tokenized query
    symbol_tokens: list[str]  # Tokens identified as symbols
    
    @property
    def is_symbol_like(self) -> bool:
        """True if query is predominantly symbol-like (>= 80% symbol tokens)."""
        return self.symbol_ratio >= 0.8
    
    @property
    def is_natural(self) -> bool:
        """True if query is predominantly natural language (<= 20% symbol tokens)."""
        return self.symbol_ratio <= 0.2


# Patterns for identifying symbol-like tokens
_CAMEL_CASE = re.compile(r"^[A-Z][a-z]+(?:[A-Z][a-z]*)+$")  # CamelCase
_PASCAL_CASE = re.compile(r"^[A-Z][a-z]+(?:[A-Z][a-z]*)*$")  # PascalCase
_SNAKE_CASE = re.compile(r"^[a-z]+(?:_[a-z]+)+$")           # snake_case
_SCREAMING_SNAKE = re.compile(r"^[A-Z]+(?:_[A-Z]+)+$")      # SCREAMING_SNAKE
_QUALIFIED = re.compile(r"^[\w]+(?:\.[\w]+)+$")              # module.Class.method
_KEBAB_CASE = re.compile(r"^[a-z]+(?:-[a-z]+)+$")           # kebab-case (file names)

# Common natural language words that are NOT symbols even if they match patterns
_NATURAL_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "but", "and", "or", "if", "because",
    "what", "which", "who", "this", "that", "these", "those", "it",
    "find", "get", "set", "add", "remove", "delete", "create", "update",
    "where", "how", "does", "work", "function", "method", "class", "file",
    "code", "implement", "call", "use", "return", "handle", "process",
})


def _is_symbol_token(token: str) -> bool:
    """Check if a token looks like a code symbol."""
    # Skip common natural language words
    if token.lower() in _NATURAL_WORDS:
        return False
    
    # Check against symbol patterns
    if _CAMEL_CASE.match(token):
        return True
    if _PASCAL_CASE.match(token) and len(token) > 3:  # Avoid matching "The", "And"
        return True
    if _SNAKE_CASE.match(token):
        return True
    if _SCREAMING_SNAKE.match(token):
        return True
    if _QUALIFIED.match(token):
        return True
    if _KEBAB_CASE.match(token):
        return True
    
    # Check for mixed case within word (e.g., "getUserById")
    if re.search(r"[a-z][A-Z]", token):
        return True
    
    # Check for underscores or dots in middle of word
    if "_" in token[1:-1] or "." in token:
        return True
    
    return False


def _tokenize_query(query: str) -> list[str]:
    """Tokenize query into meaningful tokens."""
    # Split on whitespace and common delimiters, keeping identifiers intact
    tokens = re.split(r"[\s,;:!?\"\']", query)
    # Filter empty tokens and strip punctuation from edges
    return [t.strip("()[]{}") for t in tokens if t.strip("()[]{}")]


def classify_query(query: str) -> QueryClassification:
    """Classify a query as symbol-like, natural-language, or mixed.
    
    Args:
        query: The search query string.
        
    Returns:
        QueryClassification with type, ratio, and token analysis.
    """
    tokens = _tokenize_query(query)
    
    if not tokens:
        return QueryClassification(
            query_type=QueryType.NATURAL,
            symbol_ratio=0.0,
            tokens=[],
            symbol_tokens=[],
        )
    
    symbol_tokens = [t for t in tokens if _is_symbol_token(t)]
    symbol_ratio = len(symbol_tokens) / len(tokens)
    
    if symbol_ratio >= 0.8:
        query_type = QueryType.SYMBOL
    elif symbol_ratio <= 0.2:
        query_type = QueryType.NATURAL
    else:
        query_type = QueryType.MIXED
    
    return QueryClassification(
        query_type=query_type,
        symbol_ratio=symbol_ratio,
        tokens=tokens,
        symbol_tokens=symbol_tokens,
    )


@dataclass
class StreamWeights:
    """Weights for each retrieval stream in RRF fusion."""
    
    dense: float = 1.0    # vec-cosine
    sparse: float = 1.0   # fts5-bm25
    symbol: float = 1.0   # like-tokens
    graph_vec: float = 1.0  # graph-vec (if enabled)
    
    def to_dict(self) -> dict[str, float]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "symbol": self.symbol,
            "graph_vec": self.graph_vec,
        }


# Default weights for each query type
SYMBOL_WEIGHTS = StreamWeights(
    dense=0.5,
    sparse=1.0,
    symbol=1.5,
    graph_vec=0.5,
)

NATURAL_WEIGHTS = StreamWeights(
    dense=1.2,
    sparse=0.8,
    symbol=0.5,
    graph_vec=1.0,
)

MIXED_WEIGHTS = StreamWeights(
    dense=1.0,
    sparse=1.0,
    symbol=1.0,
    graph_vec=1.0,
)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a when t=0, b when t=1."""
    return a + (b - a) * t


def _interpolate_weights(ratio: float) -> StreamWeights:
    """Continuous interpolation between NATURAL and SYMBOL weights.

    ``ratio`` is the symbol_ratio from classify_query (0.0 = pure natural,
    1.0 = pure symbol). Instead of snapping to one of three fixed buckets,
    each stream weight is smoothly interpolated so a query with 40% symbol
    tokens gets 40% of the symbol weight shift.

    WIN-298: replaces the binary SYMBOL/MIXED/NATURAL lookup.
    """
    return StreamWeights(
        dense=round(_lerp(NATURAL_WEIGHTS.dense, SYMBOL_WEIGHTS.dense, ratio), 4),
        sparse=round(_lerp(NATURAL_WEIGHTS.sparse, SYMBOL_WEIGHTS.sparse, ratio), 4),
        symbol=round(_lerp(NATURAL_WEIGHTS.symbol, SYMBOL_WEIGHTS.symbol, ratio), 4),
        graph_vec=round(_lerp(NATURAL_WEIGHTS.graph_vec, SYMBOL_WEIGHTS.graph_vec, ratio), 4),
    )


def get_adaptive_weights(query: str) -> StreamWeights:
    """Get adaptive stream weights based on query classification.

    Uses continuous interpolation (WIN-298): the symbol_ratio from query
    classification drives a smooth blend between NATURAL_WEIGHTS (ratio=0)
    and SYMBOL_WEIGHTS (ratio=1). A query with 40% symbol tokens gets 40%
    of the weight shift — no more hard buckets.

    Args:
        query: The search query string.

    Returns:
        StreamWeights smoothly interpolated for the query's symbol ratio.
    """
    classification = classify_query(query)
    return _interpolate_weights(classification.symbol_ratio)
