"""Code-aware reranker — applies code-specific heuristics for better ranking.

This reranker applies several code-aware signals to improve ranking quality
for code search, inspired by semble's approach:

1. **Definition boost**: Chunks that DEFINE a symbol matching the query
   get a score boost (definitions are usually more valuable than usages).

2. **Identifier stem match**: Stemmed identifier matching (getUserById ↔
   get_user_by_id ↔ GetUserByID) boosts chunks with similar identifiers.

3. **Noise penalty**: Test files, legacy code, compatibility layers, and
   .d.ts files get penalized (unless the query explicitly asks for tests).

4. **File coherence**: Groups results by file and reorders within groups
   to preserve adjacency, improving context when multiple chunks from the
   same file match.

YAML configuration example:
    retrieval:
      adapter: hybrid-rrf
      config:
        reranker:
          adapter: code-aware
          config:
            definition_boost: 1.5
            stem_match_boost: 1.2
            noise_penalty: 0.7
            file_coherence: true
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

from ....models import SearchHit
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class CodeAwareRerankerConfig(BaseModel):
    """Configuration for code-aware reranking heuristics."""

    definition_boost: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Score multiplier for chunks that define symbols matching the query.",
    )
    stem_match_boost: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Score multiplier for identifier stem matches.",
    )
    noise_penalty: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Score multiplier for test/legacy/compat files.",
    )
    file_coherence: bool = Field(
        default=True,
        description="Reorder hits to group same-file results together.",
    )
    file_coherence_boost: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Score boost for files with multiple high-scoring chunks (indicates strong relevance).",
    )
    noise_patterns: list[str] = Field(
        default_factory=lambda: [
            # Python
            r"test_.*\.py$",
            r".*_test\.py$",
            # Go
            r".*_test\.go$",
            # Java/Kotlin
            r".*Tests?\.java$",
            r".*Tests?\.kt$",
            r".*IT\.java$",  # Integration tests
            # JavaScript/TypeScript
            r".*\.test\.[jt]sx?$",
            r".*\.spec\.[jt]sx?$",
            r".*/__tests__/.*",
            # Ruby
            r".*_spec\.rb$",
            r".*/spec/.*",
            # C#
            r".*Tests?\.cs$",
            # PHP
            r".*Test\.php$",
            # Rust
            r".*/tests/.*\.rs$",
            # Generic patterns
            r".*/tests?/.*",
            r".*/legacy/.*",
            r".*/deprecated/.*",
            r".*/compat/.*",
            r".*\.d\.ts$",
            r".*/examples?/.*",
            r".*/fixtures?/.*",
            r".*/mocks?/.*",
            r".*/stubs?/.*",
            r".*/fakes?/.*",
            # Re-export barrels (lower penalty tier)
            r".*/__init__\.py$",
            r".*/index\.[jt]s$",
            r".*/package-info\.java$",
        ],
        description="Regex patterns for noisy file paths to penalize.",
    )
    definition_kinds: list[str] = Field(
        default_factory=lambda: [
            # Object-oriented
            "class", "interface", "type", "enum", "struct", "trait", "protocol",
            # Functions
            "function", "method", "func", "def", "impl",
            # Variables/constants (definitions)
            "const", "constant", "let", "var",
            # Modules
            "module", "namespace", "package",
            # SQL DDL
            "table", "view", "index", "procedure", "trigger",
            # Other
            "macro", "typedef", "extension",
        ],
        description="Chunk kinds that count as definitions.",
    )


# Pre-compiled regex patterns for identifier parsing
# Matches: initial lowercase word, CamelCase words, all-caps sequences, digits
_CAMEL_SPLIT = re.compile(r"([a-z]+(?=[A-Z])|[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[a-z]+$|\d+)")
_SNAKE_SPLIT = re.compile(r"[_\-.]")
_QUALIFIED_SPLIT = re.compile(r"[./]")


def _tokenize_identifier(identifier: str) -> set[str]:
    """Split an identifier into stemmed tokens.
    
    Examples:
        getUserById -> {get, user, by, id}
        get_user_by_id -> {get, user, by, id}
        GetUserByID -> {get, user, by, id}
    """
    # First split on qualified name separators
    parts = _QUALIFIED_SPLIT.split(identifier)
    tokens: set[str] = set()
    
    for part in parts:
        # Split on underscores/dashes
        snake_parts = _SNAKE_SPLIT.split(part)
        for sp in snake_parts:
            if not sp:
                continue
            # Split CamelCase
            camel_parts = _CAMEL_SPLIT.findall(sp)
            if camel_parts:
                for cp in camel_parts:
                    if cp:
                        tokens.add(cp.lower())
            else:
                tokens.add(sp.lower())
    
    return tokens


def _query_contains_test_keywords(query: str) -> bool:
    """Check if query explicitly asks for tests."""
    test_keywords = {"test", "tests", "testing", "spec", "specs", "mock", "fixture"}
    query_words = set(query.lower().split())
    return bool(query_words & test_keywords)


@register_adapter(stage="reranker", name="code-aware")
class CodeAwareReranker:
    """Applies code-aware heuristics to improve search ranking.
    
    This reranker adjusts scores based on:
    - Whether a chunk defines a symbol (vs. just using it)
    - How well identifiers match the query (accounting for naming conventions)
    - Whether the file is a test/legacy/compat file
    - Grouping results by file for better context
    """

    Config = CodeAwareRerankerConfig

    def __init__(self, config: CodeAwareRerankerConfig | None = None) -> None:
        self._config = config or CodeAwareRerankerConfig()
        # Pre-compile noise patterns
        self._noise_patterns = [
            re.compile(p, re.IGNORECASE) for p in self._config.noise_patterns
        ]
        self._definition_kinds = set(k.lower() for k in self._config.definition_kinds)

    @property
    def stage_name(self) -> str:
        return "reranker"

    @property
    def adapter_name(self) -> str:
        return "code-aware"

    @property
    def adapter_id(self) -> str:
        # Include config in adapter_id for drift detection
        return (
            f"reranker:code-aware:v1"
            f":def={self._config.definition_boost}"
            f":stem={self._config.stem_match_boost}"
            f":noise={self._config.noise_penalty}"
            f":coherence={self._config.file_coherence}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return CodeAwareRerankerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"code-aware reranker ({len(self._noise_patterns)} noise patterns)",
            detail={
                "definition_boost": str(self._config.definition_boost),
                "stem_match_boost": str(self._config.stem_match_boost),
                "noise_penalty": str(self._config.noise_penalty),
                "file_coherence": str(self._config.file_coherence),
            },
        )

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Re-score hits using code-aware heuristics.
        
        Args:
            query: The user query string.
            hits: Candidate hits to rerank.
            top_k: Number of results to return.
            
        Returns:
            Reranked hits with adjusted scores.
        """
        if not hits:
            return []
        
        # Tokenize query for matching
        query_tokens = _tokenize_identifier(query)
        skip_noise_penalty = _query_contains_test_keywords(query)
        
        # Score each hit
        scored_hits: list[tuple[float, SearchHit]] = []
        for hit in hits:
            score = hit.score or 0.0
            
            # 1. Definition boost
            score = self._apply_definition_boost(hit, query_tokens, score)
            
            # 2. Identifier stem match boost
            score = self._apply_stem_match_boost(hit, query_tokens, score)
            
            # 3. Noise penalty
            if not skip_noise_penalty:
                score = self._apply_noise_penalty(hit, score)
            
            scored_hits.append((score, hit))
        
        # Sort by adjusted score
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        
        # 4. File coherence grouping (optional)
        if self._config.file_coherence:
            scored_hits = self._apply_file_coherence(scored_hits)
        
        # Create result with updated scores and ranks
        result: list[SearchHit] = []
        for new_rank, (score, hit) in enumerate(scored_hits[:top_k], start=1):
            result.append(hit.model_copy(update={"score": score, "rank": new_rank}))
        
        return result

    def _apply_definition_boost(
        self,
        hit: SearchHit,
        query_tokens: set[str],
        score: float,
    ) -> float:
        """Boost chunks that define symbols matching the query."""
        # Check if this chunk is a definition type
        kind = (hit.kind or "").lower()
        if kind not in self._definition_kinds:
            return score
        
        # Check if the symbol matches query tokens
        symbol = hit.symbol or ""
        symbol_tokens = _tokenize_identifier(symbol)
        
        # Boost if there's significant overlap
        if query_tokens and symbol_tokens:
            overlap = len(query_tokens & symbol_tokens)
            coverage = overlap / len(query_tokens)
            if coverage >= 0.5:  # At least 50% of query tokens match
                score *= self._config.definition_boost
        
        return score

    def _apply_stem_match_boost(
        self,
        hit: SearchHit,
        query_tokens: set[str],
        score: float,
    ) -> float:
        """Boost chunks with good identifier stem matches."""
        if not query_tokens:
            return score
        
        # Tokenize all identifiers in the chunk
        symbol = hit.symbol or ""
        snippet = hit.snippet or ""
        
        # Extract potential identifiers from snippet content
        # This is a simplified heuristic — just look for CamelCase/snake_case words
        potential_ids = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", snippet)
        
        all_tokens: set[str] = _tokenize_identifier(symbol)
        for pid in potential_ids[:50]:  # Limit to avoid performance issues
            all_tokens |= _tokenize_identifier(pid)
        
        # Calculate overlap score
        if all_tokens:
            overlap = len(query_tokens & all_tokens)
            if overlap >= 2:  # At least 2 token matches
                coverage = overlap / len(query_tokens)
                boost = 1.0 + (self._config.stem_match_boost - 1.0) * coverage
                score *= boost
        
        return score

    def _apply_noise_penalty(self, hit: SearchHit, score: float) -> float:
        """Penalize test/legacy/compat files."""
        path = hit.rel_path or ""
        
        for pattern in self._noise_patterns:
            if pattern.search(path):
                score *= self._config.noise_penalty
                break  # Only apply penalty once
        
        return score

    def _apply_file_coherence(
        self,
        scored_hits: list[tuple[float, SearchHit]],
    ) -> list[tuple[float, SearchHit]]:
        """Reorder to group same-file results together and boost multi-hit files.
        
        This preserves the overall ranking but:
        1. Moves hits from the same file adjacent to each other
        2. Sorts by line number within the file
        3. Boosts scores for files with multiple high-scoring chunks
        """
        if len(scored_hits) <= 2:
            return scored_hits
        
        # Group by file
        by_file: dict[str, list[tuple[float, SearchHit, int]]] = defaultdict(list)
        for idx, (score, hit) in enumerate(scored_hits):
            path = hit.rel_path or ""
            by_file[path].append((score, hit, idx))
        
        # Apply file coherence boost for files with multiple hits
        # Files with 2+ high-scoring chunks are likely very relevant
        max_score = max(s for s, _ in scored_hits) if scored_hits else 1.0
        score_threshold = max_score * 0.5  # Consider "high-scoring" if > 50% of max
        
        boosted_by_file: dict[str, list[tuple[float, SearchHit, int]]] = {}
        for path, file_hits in by_file.items():
            # Count high-scoring hits in this file
            high_score_count = sum(1 for s, _, _ in file_hits if s >= score_threshold)
            
            # Apply coherence boost if file has 2+ high-scoring chunks
            if high_score_count >= 2:
                boost = self._config.file_coherence_boost
                boosted_hits = [(s * boost, h, i) for s, h, i in file_hits]
                boosted_by_file[path] = boosted_hits
            else:
                boosted_by_file[path] = file_hits
        
        # Sort each file's hits by line number
        for path in boosted_by_file:
            boosted_by_file[path].sort(key=lambda x: x[1].start_line or 0)
        
        # Rebuild the list, preserving first occurrence order
        seen_files: set[str] = set()
        result: list[tuple[float, SearchHit]] = []
        
        for score, hit in scored_hits:
            path = hit.rel_path or ""
            if path not in seen_files:
                seen_files.add(path)
                # Add all hits from this file, in line order
                for file_score, file_hit, _ in boosted_by_file[path]:
                    result.append((file_score, file_hit))
        
        return result
