"""Unit tests for multi-query expansion.

Tests cover:
- Enable/disable toggles (env var + config)
- Paraphrase generation via inference backend mock
- Cache behavior
- Graceful fallback on failures
- Integration with HybridRrfRetrieval
"""

from __future__ import annotations

from unittest.mock import patch

from stropha.retrieval.multi_query import (
    MultiQueryExpander,
    _cache_key,
    clear_cache,
    expand_query,
    generate_paraphrases,
)

# --------------------------------------------------------------------------- Basic


def test_multi_query_disabled_by_default(isolated_test_env) -> None:
    """Multi-query is disabled by default."""
    clear_cache()
    
    result = generate_paraphrases("how does authentication work")
    
    assert result == ["how does authentication work"]


def test_multi_query_returns_paraphrases_when_enabled(isolated_test_env) -> None:
    """Multi-query generates paraphrases when enabled via YAML."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
""")
    reset_config()
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "authentication login flow user session\n"
            "security middleware auth check validation\n"
            "user credentials token JWT verification"
        )
        result = generate_paraphrases("how does authentication work")
    
    assert len(result) >= 2  # Original + at least 1 paraphrase
    assert result[0] == "how does authentication work"  # Original first
    assert "authentication login flow user session" in result


def test_multi_query_force_enabled(monkeypatch) -> None:
    """force_enabled bypasses env var check."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "search terms expanded query"
        result = generate_paraphrases(
            "how does auth work",
            force_enabled=True,
        )
    
    assert len(result) >= 2


def test_multi_query_empty_query() -> None:
    """Empty query returns itself."""
    clear_cache()
    
    assert generate_paraphrases("", force_enabled=True) == [""]
    assert generate_paraphrases("  ", force_enabled=True) == ["  "]


def test_multi_query_generation_failure_fallback(monkeypatch) -> None:
    """Fallback to original query when inference fails."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None  # Simulate failure
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


def test_multi_query_empty_response_fallback(monkeypatch) -> None:
    """Fallback to original query when response is empty."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""
        result = generate_paraphrases("test query")
    
    assert result == ["test query"]


def test_selective_policy_expands_only_broad_conceptual_queries() -> None:
    clear_cache()
    query = "how do retrieval streams interact across the search architecture lifecycle"

    with patch("stropha.inference.generate", return_value="fusion ranking flow\nsearch pipeline design"):
        expanded = expand_query(
            query,
            force_enabled=True,
            num_paraphrases=2,
            min_terms=7,
        )

    assert expanded.reason == "success"
    assert expanded.queries == (query, "fusion ranking flow", "search pipeline design")
    assert expanded.generation_calls == 1
    assert expanded.estimated_input_tokens > 0
    assert expanded.estimated_output_tokens > 0

    with patch("stropha.inference.generate") as generate:
        symbol = expand_query("UserService", force_enabled=True, min_terms=1)
        structural = expand_query(
            "who calls the authentication request handler middleware",
            force_enabled=True,
            min_terms=1,
        )
        short = expand_query("how does authentication work", force_enabled=True)

    assert symbol.reason == "symbol_query"
    assert structural.reason == "structural_query"
    assert short.reason == "too_short"
    generate.assert_not_called()


def test_structured_expansion_handles_backend_exception() -> None:
    clear_cache()
    query = "how do retrieval streams interact across the search architecture lifecycle"

    with patch("stropha.inference.generate", side_effect=RuntimeError("offline")):
        result = expand_query(query, force_enabled=True, min_terms=7)

    assert result.queries == (query,)
    assert result.reason == "backend_error"
    assert result.generation_calls == 1
    assert result.estimated_input_tokens > 0


# --------------------------------------------------------------------------- Cache


def test_multi_query_cache_hit(monkeypatch) -> None:
    """Cached paraphrases are returned without LLM call."""
    monkeypatch.setenv("STROPHA_MULTI_QUERY_ENABLED", "1")
    clear_cache()
    
    query = "how does caching work"
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "caching layer memory storage"
        # First call - hits LLM
        result1 = generate_paraphrases(query)
        call_count_1 = mock_gen.call_count
        
        # Second call - should use cache
        result2 = generate_paraphrases(query)
        call_count_2 = mock_gen.call_count
    
    assert result1 == result2
    assert call_count_1 == call_count_2  # No new LLM call


def test_multi_query_cache_key_normalization() -> None:
    """Cache keys normalize case and whitespace."""
    key1 = _cache_key("How Does Auth Work")
    key2 = _cache_key("how does auth work")
    key3 = _cache_key("  how does auth work  ")
    
    # All should produce same key (lowercase + strip)
    assert key1 == key2
    assert key1 == key3


def test_multi_query_cache_separates_paraphrase_counts() -> None:
    clear_cache()

    with patch("stropha.inference.generate", return_value="one query\ntwo query\nthree query") as generate:
        generate_paraphrases("test query", force_enabled=True, num_paraphrases=2)
        generate_paraphrases("test query", force_enabled=True, num_paraphrases=3)

    assert generate.call_count == 2


def test_cache_clear() -> None:
    """clear_cache empties the cache."""
    clear_cache()
    
    # Simulate a cached entry via the public API
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "cached result"
        generate_paraphrases("cached query", force_enabled=True)
    
    clear_cache()
    
    # Now should call LLM again
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "new result"
        generate_paraphrases("cached query", force_enabled=True)
        assert mock_gen.called  # Had to call LLM


# --------------------------------------------------------------------------- Expander Class


def test_expander_class_disabled() -> None:
    """MultiQueryExpander respects enabled flag."""
    clear_cache()
    
    expander = MultiQueryExpander(enabled=False)
    result = expander.expand("test query")
    
    assert result == ["test query"]
    assert not expander.is_enabled


def test_expander_class_enabled(monkeypatch) -> None:
    """MultiQueryExpander generates paraphrases when enabled."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "expanded search terms"
        expander = MultiQueryExpander(enabled=True, num_paraphrases=3)
        result = expander.expand("test query")
    
    assert len(result) >= 2
    assert expander.is_enabled


def test_expander_respects_num_paraphrases(monkeypatch) -> None:
    """MultiQueryExpander limits number of paraphrases."""
    monkeypatch.delenv("STROPHA_MULTI_QUERY_ENABLED", raising=False)
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "paraphrase one\n"
            "paraphrase two\n"
            "paraphrase three\n"
            "paraphrase four\n"
            "paraphrase five"
        )
        expander = MultiQueryExpander(enabled=True, num_paraphrases=2)
        result = expander.expand("test query")
    
    # Original + 2 paraphrases max
    assert len(result) <= 3


# --------------------------------------------------------------------------- YAML Config


def test_yaml_config_count(isolated_test_env) -> None:
    """multi_query_count in YAML is respected."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
  multi_query_count: 5
  multi_query_min_terms: 9
""")
    reset_config()
    clear_cache()

    from stropha.config import get_config

    assert get_config().retrieval_features.multi_query_min_terms == 9
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "p1\np2\np3\np4\np5"
        generate_paraphrases("test query")
        
        # Should use 5 paraphrases
        mock_gen.assert_called_once()
        # The call should have a prompt with {num_paraphrases}=5
        call_args = mock_gen.call_args
        assert "5" in call_args[0][0]  # Prompt contains "5"


# --------------------------------------------------------------------------- Filtering


def test_paraphrases_deduplicated(isolated_test_env) -> None:
    """Duplicate paraphrases are removed."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
""")
    reset_config()
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "same query text\n"
            "same query text\n"
            "different query text"
        )
        result = generate_paraphrases("test query")
    
    # Should deduplicate
    assert len(result) == 3  # original + 2 unique
    assert result.count("same query text") == 1


def test_original_query_always_first(isolated_test_env) -> None:
    """Original query is always first in results."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
""")
    reset_config()
    clear_cache()
    
    original = "my original query"
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "paraphrase one\nparaphrase two"
        result = generate_paraphrases(original)
    
    assert result[0] == original


def test_numbered_lists_filtered(isolated_test_env) -> None:
    """Numbered list markers are filtered out."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
""")
    reset_config()
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "1. should be filtered\n"
            "valid paraphrase\n"
            "- also filtered\n"
            "* and this"
        )
        result = generate_paraphrases("test query")
    
    # Only "valid paraphrase" should remain (plus original)
    assert "valid paraphrase" in result
    assert not any(p.startswith("1.") for p in result)
    assert not any(p.startswith("-") for p in result)
    assert not any(p.startswith("*") for p in result)


def test_short_paraphrases_filtered(isolated_test_env) -> None:
    """Very short paraphrases are filtered."""
    from stropha.config import reset_config
    (isolated_test_env / ".stropha").mkdir(exist_ok=True)
    yaml_path = isolated_test_env / ".stropha" / "config.yaml"
    yaml_path.write_text("""
retrieval_features:
  multi_query_enabled: true
""")
    reset_config()
    clear_cache()
    
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "ok\nab\nvalid paraphrase here"
        result = generate_paraphrases("test query")
    
    # "ok" and "ab" should be filtered (< 6 chars)
    assert "ok" not in result
    assert "ab" not in result
    assert "valid paraphrase here" in result
