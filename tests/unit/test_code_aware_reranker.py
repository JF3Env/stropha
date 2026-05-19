"""Unit tests for the code-aware reranker."""

from __future__ import annotations

import pytest

from stropha.adapters.retrieval.reranker.code_aware import (
    CodeAwareReranker,
    CodeAwareRerankerConfig,
    _tokenize_identifier,
    _query_contains_test_keywords,
)
from stropha.models import SearchHit
from stropha.pipeline.registry import lookup_adapter


def _make_hit(
    chunk_id: str = "1",
    rel_path: str = "test.py",
    content: str = "def foo(): pass",
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
        snippet=content[:100],
        symbol=symbol,
        kind=kind,
        rank=rank,
        score=score,
        language=language,
        start_line=start_line,
        end_line=end_line,
    )


class TestTokenizeIdentifier:
    """Tests for identifier tokenization."""

    def test_camel_case(self):
        tokens = _tokenize_identifier("getUserById")
        assert tokens == {"get", "user", "by", "id"}

    def test_pascal_case(self):
        tokens = _tokenize_identifier("UserService")
        assert tokens == {"user", "service"}

    def test_snake_case(self):
        tokens = _tokenize_identifier("get_user_by_id")
        assert tokens == {"get", "user", "by", "id"}

    def test_screaming_snake(self):
        tokens = _tokenize_identifier("MAX_RETRY_COUNT")
        assert tokens == {"max", "retry", "count"}

    def test_qualified_name(self):
        tokens = _tokenize_identifier("com.example.UserService")
        assert "com" in tokens
        assert "example" in tokens
        assert "user" in tokens
        assert "service" in tokens

    def test_mixed_styles(self):
        tokens = _tokenize_identifier("UserService.get_user_by_id")
        assert "user" in tokens
        assert "service" in tokens
        assert "get" in tokens
        assert "id" in tokens

    def test_single_word(self):
        tokens = _tokenize_identifier("user")
        assert tokens == {"user"}

    def test_empty_string(self):
        tokens = _tokenize_identifier("")
        assert tokens == set()


class TestQueryContainsTestKeywords:
    """Tests for test keyword detection."""

    def test_contains_test(self):
        assert _query_contains_test_keywords("find the test for UserService")
        assert _query_contains_test_keywords("tests for login")
        assert _query_contains_test_keywords("testing authentication")

    def test_contains_spec(self):
        assert _query_contains_test_keywords("spec for parser")
        assert _query_contains_test_keywords("specs")

    def test_no_test_keywords(self):
        assert not _query_contains_test_keywords("how does authentication work")
        assert not _query_contains_test_keywords("UserService.login")


class TestCodeAwareRerankerConfig:
    """Tests for CodeAwareRerankerConfig."""

    def test_default_config(self):
        config = CodeAwareRerankerConfig()
        assert config.definition_boost == 1.5
        assert config.stem_match_boost == 1.2
        assert config.noise_penalty == 0.7
        assert config.file_coherence is True
        assert len(config.noise_patterns) > 0
        assert len(config.definition_kinds) > 0

    def test_custom_config(self):
        config = CodeAwareRerankerConfig(
            definition_boost=2.0,
            noise_penalty=0.5,
        )
        assert config.definition_boost == 2.0
        assert config.noise_penalty == 0.5


class TestCodeAwareRerankerRegistration:
    """Tests for adapter registration."""

    def test_registered_in_registry(self):
        adapter_cls = lookup_adapter("reranker", "code-aware")
        assert adapter_cls is CodeAwareReranker

    def test_stage_name(self):
        reranker = CodeAwareReranker()
        assert reranker.stage_name == "reranker"

    def test_adapter_name(self):
        reranker = CodeAwareReranker()
        assert reranker.adapter_name == "code-aware"

    def test_adapter_id_includes_config(self):
        config = CodeAwareRerankerConfig(definition_boost=2.0)
        reranker = CodeAwareReranker(config)
        assert "def=2.0" in reranker.adapter_id

    def test_health(self):
        reranker = CodeAwareReranker()
        health = reranker.health()
        assert health.status == "ready"
        assert "code-aware" in health.message


class TestCodeAwareRerankerDefinitionBoost:
    """Tests for definition boost heuristic."""

    def test_boosts_definition_matching_query(self):
        reranker = CodeAwareReranker()
        
        # Definition chunk
        definition_hit = _make_hit(
            chunk_id="1",
            rel_path="user_service.py",
            content="class UserService:",
            symbol="UserService",
            kind="class",
            rank=2,
            score=0.8,
        )
        
        # Usage chunk
        usage_hit = _make_hit(
            chunk_id="2",
            rel_path="main.py",
            content="service = UserService()",
            symbol="main",
            kind="function",
            rank=1,
            score=0.9,
        )
        
        result = reranker.rerank("UserService", [usage_hit, definition_hit])
        
        # Definition should be boosted and potentially move up
        definition_result = next(h for h in result if h.chunk_id == "1")
        assert definition_result.score > 0.8  # Boosted

    def test_no_boost_for_non_definition_kinds(self):
        reranker = CodeAwareReranker()
        
        hit = _make_hit(
            chunk_id="1",
            rel_path="main.py",
            content="x = UserService()",
            symbol="x",
            kind="variable",  # Not a definition kind
            rank=1,
            score=1.0,
        )
        
        result = reranker.rerank("UserService", [hit])
        
        # Score should not be boosted (may have slight stem match boost)
        assert result[0].score <= 1.0 * 1.2  # At most stem match boost


class TestCodeAwareRerankerStemMatch:
    """Tests for identifier stem match boost."""

    def test_boosts_stem_matches(self):
        reranker = CodeAwareReranker()
        
        hit = _make_hit(
            chunk_id="1",
            rel_path="service.py",
            content="def get_user_by_id(user_id): pass",
            symbol="get_user_by_id",
            rank=1,
            score=1.0,
        )
        
        # Query uses different naming convention
        result = reranker.rerank("getUserById", [hit])
        
        # Should be boosted due to stem match
        assert result[0].score > 1.0


class TestCodeAwareRerankerNoisePenalty:
    """Tests for noise penalty heuristic."""

    def test_penalizes_test_files(self):
        reranker = CodeAwareReranker()
        
        test_hit = _make_hit(
            chunk_id="1",
            rel_path="tests/test_user_service.py",
            content="def test_login():",
            rank=1,
            score=1.0,
        )
        
        result = reranker.rerank("login", [test_hit])
        
        # Test file should be penalized
        assert result[0].score < 1.0

    def test_penalizes_spec_files(self):
        reranker = CodeAwareReranker()
        
        hit = _make_hit(
            chunk_id="1",
            rel_path="auth.spec.ts",
            content="it('should login')",
            language="typescript",
            rank=1,
            score=1.0,
        )
        
        result = reranker.rerank("login", [hit])
        assert result[0].score < 1.0

    def test_penalizes_legacy_files(self):
        reranker = CodeAwareReranker()
        
        hit = _make_hit(
            chunk_id="1",
            rel_path="src/legacy/old_auth.py",
            content="def login():",
            rank=1,
            score=1.0,
        )
        
        result = reranker.rerank("login", [hit])
        assert result[0].score < 1.0

    def test_no_penalty_when_query_asks_for_tests(self):
        reranker = CodeAwareReranker()
        
        test_hit = _make_hit(
            chunk_id="1",
            rel_path="tests/test_login.py",
            content="def test_login():",
            rank=1,
            score=1.0,
        )
        
        # Query explicitly asks for tests
        result = reranker.rerank("tests for login", [test_hit])
        
        # Should NOT be penalized
        assert result[0].score >= 1.0


class TestCodeAwareRerankerFileCoherence:
    """Tests for file coherence grouping."""

    def test_groups_same_file_hits(self):
        config = CodeAwareRerankerConfig(
            file_coherence=True,
            definition_boost=1.0,  # Disable other boosts for clarity
            stem_match_boost=1.0,
            noise_penalty=1.0,
        )
        reranker = CodeAwareReranker(config)
        
        hits = [
            _make_hit(chunk_id="1", rel_path="a.py", content="a", start_line=10, rank=1, score=1.0),
            _make_hit(chunk_id="2", rel_path="b.py", content="b", start_line=5, rank=2, score=0.9),
            _make_hit(chunk_id="3", rel_path="a.py", content="c", start_line=5, rank=3, score=0.8),
        ]
        
        result = reranker.rerank("query", hits)
        
        # Hits from a.py should be grouped together
        a_indices = [i for i, h in enumerate(result) if h.rel_path == "a.py"]
        assert abs(a_indices[0] - a_indices[1]) == 1  # Adjacent

    def test_preserves_order_within_file(self):
        config = CodeAwareRerankerConfig(
            file_coherence=True,
            definition_boost=1.0,
            stem_match_boost=1.0,
            noise_penalty=1.0,
        )
        reranker = CodeAwareReranker(config)
        
        hits = [
            _make_hit(chunk_id="1", rel_path="a.py", content="a", start_line=20, rank=1, score=1.0),
            _make_hit(chunk_id="2", rel_path="a.py", content="b", start_line=10, rank=2, score=0.9),
            _make_hit(chunk_id="3", rel_path="a.py", content="c", start_line=15, rank=3, score=0.8),
        ]
        
        result = reranker.rerank("query", hits)
        
        # Within a.py, should be ordered by line number
        lines = [h.start_line for h in result]
        assert lines == sorted(lines)

    def test_disabled_file_coherence(self):
        config = CodeAwareRerankerConfig(file_coherence=False)
        reranker = CodeAwareReranker(config)
        
        hits = [
            _make_hit(chunk_id="1", rel_path="a.py", content="a", rank=1, score=1.0),
            _make_hit(chunk_id="2", rel_path="b.py", content="b", rank=2, score=0.9),
            _make_hit(chunk_id="3", rel_path="a.py", content="c", rank=3, score=0.8),
        ]
        
        result = reranker.rerank("query", hits)
        
        # Without coherence, order is by score only (with boosts applied)
        # Just verify it doesn't crash
        assert len(result) == 3


class TestCodeAwareRerankerEdgeCases:
    """Tests for edge cases."""

    def test_empty_hits(self):
        reranker = CodeAwareReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_single_hit(self):
        reranker = CodeAwareReranker()
        hit = _make_hit(chunk_id="1", rel_path="a.py", content="a", rank=1, score=1.0)
        result = reranker.rerank("query", [hit])
        assert len(result) == 1

    def test_top_k_truncation(self):
        reranker = CodeAwareReranker()
        hits = [
            _make_hit(chunk_id=str(i), rel_path="a.py", content="a", rank=i, score=1.0 - i * 0.1)
            for i in range(10)
        ]
        
        result = reranker.rerank("query", hits, top_k=5)
        assert len(result) == 5

    def test_ranks_are_updated(self):
        reranker = CodeAwareReranker()
        hits = [
            _make_hit(chunk_id="1", rel_path="a.py", content="a", rank=5, score=0.8),
            _make_hit(chunk_id="2", rel_path="b.py", content="b", rank=10, score=0.9),
        ]
        
        result = reranker.rerank("query", hits)
        
        # Ranks should be 1, 2 after reranking
        ranks = [h.rank for h in result]
        assert ranks == [1, 2]

    def test_none_values_handled(self):
        reranker = CodeAwareReranker()
        hit = _make_hit(
            chunk_id="1",
            rel_path="a.py",
            content="a",
            symbol=None,
            kind="file",  # kind is required, can't be None
            rank=1,
            score=1.0,
        )
        
        # Should not crash
        result = reranker.rerank("query", [hit])
        assert len(result) == 1
