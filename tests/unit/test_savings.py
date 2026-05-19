"""Unit tests for token savings tracking."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stropha.models import SearchHit
from stropha.retrieval.savings import (
    SearchSavings,
    calculate_savings,
    estimate_content_tokens,
    estimate_file_tokens,
    log_savings,
    track_search_savings,
)


def _make_hit(
    chunk_id: str = "1",
    rel_path: str = "test.py",
    snippet: str = "def foo(): pass",
    symbol: str | None = "foo",
    kind: str = "function",
    rank: int = 1,
    score: float = 1.0,
    language: str = "python",
    start_line: int = 1,
    end_line: int = 5,
) -> SearchHit:
    """Helper to create SearchHit objects for tests."""
    return SearchHit(
        chunk_id=chunk_id,
        rel_path=rel_path,
        snippet=snippet,
        symbol=symbol,
        kind=kind,
        rank=rank,
        score=score,
        language=language,
        start_line=start_line,
        end_line=end_line,
    )


class TestSearchSavings:
    """Tests for SearchSavings dataclass."""

    def test_tokens_saved_positive(self):
        savings = SearchSavings(
            query="test query",
            returned_tokens=100,
            baseline_tokens=1000,
            results_count=5,
        )
        assert savings.tokens_saved == 900
        assert savings.savings_pct == 90.0

    def test_tokens_saved_zero(self):
        savings = SearchSavings(
            query="test",
            returned_tokens=1000,
            baseline_tokens=1000,
            results_count=10,
        )
        assert savings.tokens_saved == 0
        assert savings.savings_pct == 0.0

    def test_tokens_saved_negative_clamped(self):
        # If returned > baseline, savings should be 0 (not negative)
        savings = SearchSavings(
            query="test",
            returned_tokens=2000,
            baseline_tokens=1000,
            results_count=20,
        )
        assert savings.tokens_saved == 0

    def test_savings_pct_zero_baseline(self):
        savings = SearchSavings(
            query="test",
            returned_tokens=100,
            baseline_tokens=0,
            results_count=1,
        )
        assert savings.savings_pct == 0.0


class TestEstimateFunctions:
    """Tests for token estimation functions."""

    def test_estimate_file_tokens(self):
        # 1000 bytes = ~250 tokens (4 bytes/token)
        assert estimate_file_tokens(1000) == 250
        assert estimate_file_tokens(0) == 0
        assert estimate_file_tokens(4) == 1

    def test_estimate_content_tokens(self):
        # Simple content with words
        content = "def hello_world(): return 42"
        tokens = estimate_content_tokens(content)
        # 4 words × 1.2 = ~5 tokens (rounded)
        assert tokens >= 4
        assert tokens <= 10

    def test_estimate_content_tokens_empty(self):
        assert estimate_content_tokens("") == 0


class TestCalculateSavings:
    """Tests for calculate_savings function."""

    def test_calculate_with_hits(self):
        hits = [
            _make_hit(
                chunk_id="1",
                rel_path="a.py",
                snippet="def foo(): pass",
                rank=1,
                score=1.0,
            ),
            _make_hit(
                chunk_id="2",
                rel_path="b.py",
                snippet="class Bar: pass",
                rank=2,
                score=0.9,
            ),
        ]
        
        savings = calculate_savings(
            query="foo bar",
            hits=hits,
            repo_total_size_bytes=100000,
        )
        
        assert savings.query == "foo bar"
        assert savings.results_count == 2
        assert savings.returned_tokens > 0
        assert savings.baseline_tokens > 0

    def test_calculate_with_repo_size(self):
        hits = [
            _make_hit(
                chunk_id="1",
                rel_path="a.py",
                snippet="small",
                rank=1,
                score=1.0,
            ),
        ]
        
        # Large repo = more savings potential
        savings_large = calculate_savings(
            query="test",
            hits=hits,
            repo_total_size_bytes=1000000,
        )
        
        savings_small = calculate_savings(
            query="test",
            hits=hits,
            repo_total_size_bytes=10000,
        )
        
        # Same returned tokens, but baseline is proportional to repo size
        assert savings_large.baseline_tokens > savings_small.baseline_tokens

    def test_calculate_without_repo_size(self):
        hits = [
            _make_hit(
                chunk_id="1",
                rel_path="a.py",
                snippet="test",
                rank=1,
                score=1.0,
            ),
        ]
        
        # Should use fallback estimation
        savings = calculate_savings(
            query="test",
            hits=hits,
            repo_total_size_bytes=None,
        )
        
        assert savings.baseline_tokens > 0

    def test_calculate_empty_hits(self):
        savings = calculate_savings(
            query="test",
            hits=[],
            repo_total_size_bytes=100000,
        )
        
        assert savings.returned_tokens == 0
        assert savings.results_count == 0
        assert savings.tokens_saved > 0  # All baseline tokens saved

    def test_calculate_with_none_content(self):
        hits = [
            _make_hit(
                chunk_id="1",
                rel_path="a.py",
                snippet="",  # Empty snippet instead of None
                rank=1,
                score=1.0,
            ),
        ]
        
        # Should handle empty snippet gracefully
        savings = calculate_savings(
            query="test",
            hits=hits,
            repo_total_size_bytes=100000,
        )
        
        assert savings.returned_tokens == 0


class TestLogSavings:
    """Tests for log_savings function."""

    def test_log_savings_called(self):
        savings = SearchSavings(
            query="test query",
            returned_tokens=100,
            baseline_tokens=1000,
            results_count=5,
        )
        
        with patch("stropha.retrieval.savings.log") as mock_log:
            log_savings(savings)
            
            mock_log.info.assert_called_once()
            call_args = mock_log.info.call_args
            assert call_args[0][0] == "search.served"
            assert call_args[1]["returned_tokens"] == 100
            assert call_args[1]["baseline_tokens"] == 1000
            assert call_args[1]["tokens_saved"] == 900
            assert call_args[1]["savings_pct"] == 90.0

    def test_log_savings_truncates_long_query(self):
        long_query = "x" * 200
        savings = SearchSavings(
            query=long_query,
            returned_tokens=100,
            baseline_tokens=1000,
            results_count=5,
        )
        
        with patch("stropha.retrieval.savings.log") as mock_log:
            log_savings(savings)
            
            call_args = mock_log.info.call_args
            assert len(call_args[1]["query"]) == 100


class TestTrackSearchSavings:
    """Tests for track_search_savings convenience function."""

    def test_track_calculates_and_logs(self):
        hits = [
            _make_hit(
                chunk_id="1",
                rel_path="a.py",
                snippet="def foo(): pass",
                rank=1,
                score=1.0,
            ),
        ]
        
        with patch("stropha.retrieval.savings.log"):
            savings = track_search_savings(
                query="foo",
                hits=hits,
                repo_size_bytes=100000,
            )
            
            assert isinstance(savings, SearchSavings)
            assert savings.query == "foo"
            assert savings.results_count == 1


class TestCostReportSavings:
    """Tests for savings in CostReport."""

    def test_aggregate_search_served_events(self):
        import json
        import tempfile
        from pathlib import Path
        
        from stropha.cost import aggregate
        
        # Create a temp log file with search.served events
        events = [
            {
                "event": "search.served",
                "query": "test1",
                "results_count": 5,
                "returned_tokens": 100,
                "baseline_tokens": 1000,
                "tokens_saved": 900,
            },
            {
                "event": "search.served",
                "query": "test2",
                "results_count": 3,
                "returned_tokens": 200,
                "baseline_tokens": 800,
                "tokens_saved": 600,
            },
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
            temp_path = Path(f.name)
        
        try:
            report = aggregate([temp_path])
            
            assert report.savings.total_searches == 2
            assert report.savings.total_results_returned == 8
            assert report.savings.total_returned_tokens == 300
            assert report.savings.total_baseline_tokens == 1800
            assert report.savings.total_tokens_saved == 1500
            assert report.savings.avg_savings_pct > 80
        finally:
            temp_path.unlink()

    def test_savings_in_as_dict(self):
        from stropha.cost import CostReport, SavingsCost
        
        report = CostReport()
        report.savings = SavingsCost(
            total_searches=10,
            total_returned_tokens=1000,
            total_baseline_tokens=10000,
            total_tokens_saved=9000,
            total_results_returned=50,
        )
        
        d = report.as_dict()
        
        assert "savings" in d
        assert d["savings"]["total_searches"] == 10
        assert d["savings"]["avg_savings_pct"] == 90.0
        assert d["savings"]["avg_results_per_search"] == 5.0
