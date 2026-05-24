"""Unit tests for query classification and adaptive weighting."""

from __future__ import annotations

import pytest

from stropha.retrieval.query_classify import (
    QueryClassification,
    QueryType,
    StreamWeights,
    classify_query,
    get_adaptive_weights,
    _interpolate_weights,
    SYMBOL_WEIGHTS,
    NATURAL_WEIGHTS,
    MIXED_WEIGHTS,
)


class TestQueryClassification:
    """Tests for QueryClassification dataclass."""

    def test_is_symbol_like_above_threshold(self):
        classification = QueryClassification(
            query_type=QueryType.SYMBOL,
            symbol_ratio=0.85,
            tokens=["getUserById", "UserService"],
            symbol_tokens=["getUserById", "UserService"],
        )
        assert classification.is_symbol_like is True
        assert classification.is_natural is False

    def test_is_natural_below_threshold(self):
        classification = QueryClassification(
            query_type=QueryType.NATURAL,
            symbol_ratio=0.1,
            tokens=["how", "does", "authentication", "work"],
            symbol_tokens=[],
        )
        assert classification.is_symbol_like is False
        assert classification.is_natural is True

    def test_mixed_neither_extreme(self):
        classification = QueryClassification(
            query_type=QueryType.MIXED,
            symbol_ratio=0.5,
            tokens=["find", "UserService", "method"],
            symbol_tokens=["UserService"],
        )
        assert classification.is_symbol_like is False
        assert classification.is_natural is False


class TestClassifyQuery:
    """Tests for classify_query function."""

    # Symbol-like queries
    def test_camel_case_symbol(self):
        result = classify_query("getUserById")
        assert result.query_type == QueryType.SYMBOL
        assert result.symbol_ratio >= 0.8
        assert "getUserById" in result.symbol_tokens

    def test_pascal_case_symbol(self):
        result = classify_query("UserService")
        assert result.query_type == QueryType.SYMBOL
        assert "UserService" in result.symbol_tokens

    def test_snake_case_symbol(self):
        result = classify_query("get_user_by_id")
        assert result.query_type == QueryType.SYMBOL
        assert "get_user_by_id" in result.symbol_tokens

    def test_qualified_name_symbol(self):
        result = classify_query("com.example.UserService.findById")
        assert result.query_type == QueryType.SYMBOL
        assert result.symbol_ratio >= 0.8

    def test_multiple_symbols(self):
        result = classify_query("UserService getUserById")
        assert result.query_type == QueryType.SYMBOL
        assert len(result.symbol_tokens) == 2

    def test_screaming_snake_case(self):
        result = classify_query("MAX_RETRY_COUNT")
        assert "MAX_RETRY_COUNT" in result.symbol_tokens

    # Natural language queries
    def test_simple_question(self):
        result = classify_query("how does authentication work")
        assert result.query_type == QueryType.NATURAL
        assert result.symbol_ratio <= 0.2

    def test_description_query(self):
        result = classify_query("where is the login flow implemented")
        assert result.query_type == QueryType.NATURAL

    def test_natural_with_common_words(self):
        result = classify_query("find the function that handles user registration")
        assert result.query_type == QueryType.NATURAL

    # Mixed queries
    def test_mixed_symbol_and_natural(self):
        result = classify_query("where is UserService implemented")
        assert result.query_type == QueryType.MIXED
        assert 0.2 < result.symbol_ratio < 0.8
        assert "UserService" in result.symbol_tokens

    def test_mixed_with_qualified_name(self):
        result = classify_query("what does config.database.host do")
        # "config.database.host" is a qualified name (symbol)
        assert "config.database.host" in result.symbol_tokens

    # Edge cases
    def test_empty_query(self):
        result = classify_query("")
        assert result.query_type == QueryType.NATURAL
        assert result.symbol_ratio == 0.0
        assert result.tokens == []

    def test_whitespace_only_query(self):
        result = classify_query("   ")
        assert result.query_type == QueryType.NATURAL
        assert result.tokens == []

    def test_query_with_punctuation(self):
        result = classify_query("what is UserService?")
        assert "UserService" in result.symbol_tokens

    def test_query_with_delimiters(self):
        result = classify_query("UserService, UserRepository")
        assert len(result.symbol_tokens) == 2

    def test_short_pascal_case_not_symbol(self):
        # Short words like "The", "And" shouldn't be marked as symbols
        result = classify_query("The And Or")
        assert len(result.symbol_tokens) == 0

    def test_common_words_not_symbols(self):
        # Words like "find", "get", "create" are natural language
        result = classify_query("find get create")
        assert result.query_type == QueryType.NATURAL


class TestStreamWeights:
    """Tests for StreamWeights dataclass."""

    def test_default_weights(self):
        weights = StreamWeights()
        assert weights.dense == 1.0
        assert weights.sparse == 1.0
        assert weights.symbol == 1.0
        assert weights.graph_vec == 1.0

    def test_custom_weights(self):
        weights = StreamWeights(dense=0.5, sparse=1.5)
        assert weights.dense == 0.5
        assert weights.sparse == 1.5

    def test_to_dict(self):
        weights = StreamWeights(dense=0.5, sparse=1.5, symbol=2.0, graph_vec=0.8)
        d = weights.to_dict()
        assert d == {"dense": 0.5, "sparse": 1.5, "symbol": 2.0, "graph_vec": 0.8}


class TestGetAdaptiveWeights:
    """Tests for get_adaptive_weights function."""

    def test_symbol_query_weights(self):
        weights = get_adaptive_weights("getUserById")
        assert weights.symbol > weights.dense  # Symbol stream boosted

    def test_natural_query_weights(self):
        weights = get_adaptive_weights("how does authentication work")
        assert weights.dense > weights.symbol  # Dense stream boosted

    def test_mixed_query_weights_interpolated(self):
        """Mixed queries should produce weights between the two extremes."""
        weights = get_adaptive_weights("find UserService method")
        # Should be between natural and symbol weights for each stream
        assert NATURAL_WEIGHTS.dense >= weights.dense >= SYMBOL_WEIGHTS.dense
        assert SYMBOL_WEIGHTS.symbol >= weights.symbol >= NATURAL_WEIGHTS.symbol

    def test_symbol_weights_values(self):
        """Verify symbol weights favor lexical/symbol streams."""
        assert SYMBOL_WEIGHTS.symbol > SYMBOL_WEIGHTS.dense
        assert SYMBOL_WEIGHTS.sparse >= SYMBOL_WEIGHTS.dense

    def test_natural_weights_values(self):
        """Verify natural weights favor dense semantic matching."""
        assert NATURAL_WEIGHTS.dense > NATURAL_WEIGHTS.symbol
        assert NATURAL_WEIGHTS.graph_vec >= NATURAL_WEIGHTS.symbol


class TestContinuousInterpolation:
    """Tests for WIN-298 continuous adaptive weighting."""

    def test_ratio_zero_gives_natural_weights(self):
        """symbol_ratio=0 should produce exactly NATURAL_WEIGHTS."""
        weights = _interpolate_weights(0.0)
        assert weights.dense == NATURAL_WEIGHTS.dense
        assert weights.sparse == NATURAL_WEIGHTS.sparse
        assert weights.symbol == NATURAL_WEIGHTS.symbol
        assert weights.graph_vec == NATURAL_WEIGHTS.graph_vec

    def test_ratio_one_gives_symbol_weights(self):
        """symbol_ratio=1 should produce exactly SYMBOL_WEIGHTS."""
        weights = _interpolate_weights(1.0)
        assert weights.dense == SYMBOL_WEIGHTS.dense
        assert weights.sparse == SYMBOL_WEIGHTS.sparse
        assert weights.symbol == SYMBOL_WEIGHTS.symbol
        assert weights.graph_vec == SYMBOL_WEIGHTS.graph_vec

    def test_ratio_half_is_midpoint(self):
        """symbol_ratio=0.5 should be the exact midpoint."""
        weights = _interpolate_weights(0.5)
        expected_dense = (NATURAL_WEIGHTS.dense + SYMBOL_WEIGHTS.dense) / 2
        expected_symbol = (NATURAL_WEIGHTS.symbol + SYMBOL_WEIGHTS.symbol) / 2
        assert abs(weights.dense - expected_dense) < 0.001
        assert abs(weights.symbol - expected_symbol) < 0.001

    def test_monotonic_dense_decreases_with_ratio(self):
        """Dense weight should decrease as ratio increases (natural → symbol)."""
        w0 = _interpolate_weights(0.0)
        w25 = _interpolate_weights(0.25)
        w50 = _interpolate_weights(0.5)
        w75 = _interpolate_weights(0.75)
        w1 = _interpolate_weights(1.0)
        assert w0.dense >= w25.dense >= w50.dense >= w75.dense >= w1.dense

    def test_monotonic_symbol_increases_with_ratio(self):
        """Symbol weight should increase as ratio increases."""
        w0 = _interpolate_weights(0.0)
        w25 = _interpolate_weights(0.25)
        w50 = _interpolate_weights(0.5)
        w75 = _interpolate_weights(0.75)
        w1 = _interpolate_weights(1.0)
        assert w0.symbol <= w25.symbol <= w50.symbol <= w75.symbol <= w1.symbol

    def test_gradual_shift_not_sudden_jump(self):
        """Adjacent ratios should produce similar weights (no sudden jumps)."""
        w40 = _interpolate_weights(0.4)
        w50 = _interpolate_weights(0.5)
        # The step should be small — max delta per 0.1 step in any stream
        delta_dense = abs(w50.dense - w40.dense)
        delta_symbol = abs(w50.symbol - w40.symbol)
        assert delta_dense < 0.2  # At most 0.2 change per 0.1 ratio step
        assert delta_symbol < 0.2

    def test_continuous_weights_respects_stream_ordering(self):
        """At any ratio, the relative ordering should make sense."""
        # At ratio=0.3 (mostly natural), dense should still be >= symbol
        w = _interpolate_weights(0.3)
        assert w.dense >= w.symbol
        
        # At ratio=0.7 (mostly symbol), symbol should be > dense
        w = _interpolate_weights(0.7)
        assert w.symbol > w.dense


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
):
    """Helper to create SearchHit objects for tests."""
    from stropha.models import SearchHit
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


class TestWeightedRRF:
    """Tests for weighted RRF fusion."""

    def test_rrf_fuse_with_weights(self):
        from stropha.retrieval.rrf import rrf_fuse

        # Create mock hits
        hits1 = [
            _make_hit(chunk_id="a", rel_path="a.py", snippet="a", rank=1, score=1.0),
            _make_hit(chunk_id="b", rel_path="b.py", snippet="b", rank=2, score=0.9),
        ]
        hits2 = [
            _make_hit(chunk_id="b", rel_path="b.py", snippet="b", rank=1, score=1.0),
            _make_hit(chunk_id="c", rel_path="c.py", snippet="c", rank=2, score=0.9),
        ]

        # Without weights (equal)
        result_equal = rrf_fuse(hits1, hits2, k=60, top_k=3)
        
        # With weights (second list boosted)
        result_weighted = rrf_fuse(hits1, hits2, k=60, top_k=3, weights=[0.5, 1.5])
        
        # "b" should rank higher with weighted fusion since it's rank 1 in the boosted list
        assert result_weighted[0].chunk_id == "b"

    def test_rrf_fuse_named(self):
        from stropha.retrieval.rrf import rrf_fuse_named

        hits_dense = [
            _make_hit(chunk_id="a", rel_path="a.py", snippet="a", rank=1, score=1.0),
        ]
        hits_sparse = [
            _make_hit(chunk_id="b", rel_path="b.py", snippet="b", rank=1, score=1.0),
        ]

        result = rrf_fuse_named(
            {"dense": hits_dense, "sparse": hits_sparse},
            k=60,
            top_k=2,
            weights={"dense": 1.0, "sparse": 2.0},
        )
        
        # "b" should rank higher due to higher weight
        assert result[0].chunk_id == "b"

    def test_rrf_fuse_weight_length_mismatch(self):
        from stropha.retrieval.rrf import rrf_fuse

        hits1 = [_make_hit(chunk_id="a", rel_path="a.py", snippet="a", rank=1, score=1.0)]
        hits2 = [_make_hit(chunk_id="b", rel_path="b.py", snippet="b", rank=1, score=1.0)]

        with pytest.raises(ValueError, match="weights length"):
            rrf_fuse(hits1, hits2, weights=[1.0])  # Only 1 weight for 2 lists
