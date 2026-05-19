"""Tokens-saved tracking for RAG value measurement.

Tracks how many tokens were saved by using stropha's RAG search instead
of naive grep + read. This helps users understand the value delivered
by semantic search.

The basic formula:
- baseline_tokens = estimated total tokens to read all matching files
- returned_tokens = sum of content lengths in search results
- tokens_saved = baseline_tokens - returned_tokens
- savings_pct = tokens_saved / baseline_tokens * 100

Savings are logged via structlog and can be aggregated by `stropha cost --savings`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging import get_logger
from ..models import SearchHit

log = get_logger(__name__)


@dataclass
class SearchSavings:
    """Token savings from a single search query."""
    
    query: str
    returned_tokens: int
    baseline_tokens: int
    results_count: int
    
    @property
    def tokens_saved(self) -> int:
        """Tokens saved compared to naive read."""
        return max(0, self.baseline_tokens - self.returned_tokens)
    
    @property
    def savings_pct(self) -> float:
        """Percentage of tokens saved (0-100)."""
        if self.baseline_tokens <= 0:
            return 0.0
        return (self.tokens_saved / self.baseline_tokens) * 100


def estimate_file_tokens(file_size_bytes: int) -> int:
    """Estimate token count from file size.
    
    Rough estimate: ~4 bytes per token for code (mix of keywords,
    identifiers, whitespace). This is a conservative estimate;
    actual token counts depend on the tokenizer.
    """
    return file_size_bytes // 4


def estimate_content_tokens(content: str) -> int:
    """Estimate token count from content string.
    
    Simple approximation: split on whitespace and punctuation,
    counting each as roughly one token.
    """
    # Count words (split on whitespace)
    words = len(content.split())
    # Add estimate for punctuation/operators (roughly 20% extra)
    return int(words * 1.2)


def calculate_savings(
    query: str,
    hits: list[SearchHit],
    repo_total_size_bytes: int | None = None,
    average_file_size_bytes: int = 5000,
    files_in_repo: int = 100,
) -> SearchSavings:
    """Calculate token savings from search results.
    
    Args:
        query: The search query string.
        hits: Search results returned.
        repo_total_size_bytes: Total size of repo in bytes (if known).
        average_file_size_bytes: Average file size estimate (default 5KB).
        files_in_repo: Estimated number of files in repo.
        
    Returns:
        SearchSavings with token counts and savings percentage.
    """
    # Calculate returned tokens (what we actually return)
    returned_tokens = sum(
        estimate_content_tokens(hit.snippet or "")
        for hit in hits
    )
    
    # Estimate baseline tokens (naive approach: grep + read matching files)
    # We assume a naive approach would need to read ~10% of the repo
    # to find similar results (conservative estimate)
    if repo_total_size_bytes:
        # Estimate reading 10% of repo for similar coverage
        baseline_bytes = repo_total_size_bytes // 10
    else:
        # Fallback: estimate based on average file size × files to read
        # Assume reading ~10 files to get similar coverage
        baseline_bytes = average_file_size_bytes * min(10, files_in_repo)
    
    baseline_tokens = estimate_file_tokens(baseline_bytes)
    
    return SearchSavings(
        query=query,
        returned_tokens=returned_tokens,
        baseline_tokens=baseline_tokens,
        results_count=len(hits),
    )


def log_savings(savings: SearchSavings) -> None:
    """Log savings to structlog for aggregation by `stropha cost`.
    
    The log entry uses event="search.served" which can be parsed
    by the cost aggregator.
    """
    log.info(
        "search.served",
        query=savings.query[:100],  # Truncate long queries
        results_count=savings.results_count,
        returned_tokens=savings.returned_tokens,
        baseline_tokens=savings.baseline_tokens,
        tokens_saved=savings.tokens_saved,
        savings_pct=round(savings.savings_pct, 1),
    )


def track_search_savings(
    query: str,
    hits: list[SearchHit],
    repo_size_bytes: int | None = None,
) -> SearchSavings:
    """Calculate and log token savings for a search.
    
    Convenience function that calculates savings and logs them.
    
    Args:
        query: The search query.
        hits: Search results.
        repo_size_bytes: Total repo size in bytes (optional).
        
    Returns:
        SearchSavings with the calculated metrics.
    """
    savings = calculate_savings(
        query=query,
        hits=hits,
        repo_total_size_bytes=repo_size_bytes,
    )
    log_savings(savings)
    return savings
