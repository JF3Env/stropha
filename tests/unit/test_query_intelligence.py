"""Tests for query intelligence features (HyDE, Query Rewriting)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stropha.retrieval.hyde import HydeResult, maybe_hyde_rewrite
from stropha.retrieval.hyde import _enabled as hyde_enabled
from stropha.retrieval.multi_query import MultiQueryResult
from stropha.retrieval.query_rewrite import (
    QueryRewriteResult,
    _enabled,
    maybe_rewrite_query,
    rewrite_query,
)

rewrite_enabled = _enabled


class TestHyDE:
    """Tests for HyDE (Hypothetical Document Embeddings)."""

    def test_hyde_disabled_by_default(self, isolated_test_env) -> None:
        """HyDE should be disabled by default."""
        assert not hyde_enabled()
        result = maybe_hyde_rewrite("test query")
        assert result is None

    def test_hyde_returns_none_for_empty_query(self, isolated_test_env) -> None:
        """HyDE should return None for empty queries even when enabled."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  hyde_enabled: true
""")
        reset_config()
        assert maybe_hyde_rewrite("") is None
        assert maybe_hyde_rewrite("   ") is None

    @patch("stropha.inference.generate")
    def test_hyde_calls_inference_backend(self, mock_gen: MagicMock, isolated_test_env) -> None:
        """HyDE should call inference backend when enabled."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  hyde_enabled: true
""")
        reset_config()
        mock_gen.return_value = "def calculate_sum(a, b): return a + b"

        result = maybe_hyde_rewrite("how to add two numbers")
        assert result is not None
        assert "calculate_sum" in result or "return" in result

    @patch("stropha.inference.generate")
    def test_hyde_handles_inference_failure(self, mock_gen: MagicMock, isolated_test_env) -> None:
        """HyDE should return None when inference fails."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  hyde_enabled: true
""")
        reset_config()
        mock_gen.return_value = None  # Simulate failure

        result = maybe_hyde_rewrite("test query")
        assert result is None

    @patch("stropha.inference.generate")
    def test_hyde_handles_empty_response(self, mock_gen: MagicMock, isolated_test_env) -> None:
        """HyDE should return None when inference returns empty."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  hyde_enabled: true
""")
        reset_config()
        mock_gen.return_value = ""

        result = maybe_hyde_rewrite("test query")
        assert result is None


class TestQueryRewrite:
    """Tests for Query Rewriting."""

    def test_rewrite_disabled_by_default(self, isolated_test_env) -> None:
        """Query rewriting should be disabled by default."""
        assert not rewrite_enabled()
        result = maybe_rewrite_query("test query")
        assert result is None

    def test_rewrite_returns_none_for_empty_query(self, isolated_test_env) -> None:
        """Query rewriting should return None for empty queries even when enabled."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()
        assert maybe_rewrite_query("") is None
        assert maybe_rewrite_query("   ") is None

    @patch("stropha.inference.generate")
    def test_rewrite_calls_inference_backend(self, mock_gen: MagicMock, isolated_test_env) -> None:
        """Query rewriting should call inference backend when enabled."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()
        mock_gen.return_value = "FSRS calculator test FsrsCalculatorTest"

        result = maybe_rewrite_query("onde tem teste pra fsrs")
        assert result is not None
        # Should contain original query + rewritten terms
        assert "fsrs" in result.lower()

    @patch("stropha.inference.generate")
    def test_rewrite_combines_original_and_expanded(
        self, mock_gen: MagicMock, isolated_test_env
    ) -> None:
        """Query rewriting should combine original query with expanded terms."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()
        mock_gen.return_value = "authentication login security"

        result = maybe_rewrite_query("how does auth work")
        assert result is not None
        # Should contain both original and expanded
        assert "how does auth work" in result
        assert "authentication" in result

    @patch("stropha.inference.generate")
    def test_rewrite_handles_inference_failure(
        self, mock_gen: MagicMock, isolated_test_env
    ) -> None:
        """Query rewriting should return None when inference fails."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()
        mock_gen.return_value = None  # Simulate failure

        result = maybe_rewrite_query("test query")
        assert result is None

    @patch("stropha.inference.generate")
    def test_rewrite_truncates_long_output(self, mock_gen: MagicMock, isolated_test_env) -> None:
        """Query rewriting should truncate very long outputs."""
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()
        long_response = "term " * 200  # 1000 chars
        mock_gen.return_value = long_response

        result = maybe_rewrite_query("short query")
        assert result is not None
        assert len(result) <= 500

    @pytest.mark.parametrize(
        ("query", "reason"),
        [
            ("UserService.authenticate", "symbol_query"),
            ("what calls UserService", "structural_query"),
            ("tests for UserService", "structural_query"),
            ("what relates to UserService", "structural_query"),
            ("trace feature user login", "structural_query"),
            ("how does authentication flow through every service layer", "too_long"),
        ],
    )
    @patch("stropha.inference.generate")
    def test_rewrite_policy_skips_non_candidates(
        self,
        mock_gen: MagicMock,
        query: str,
        reason: str,
    ) -> None:
        result = rewrite_query(query, force_enabled=True)

        assert result.query is None
        assert result.reason == reason
        assert result.generation_calls == 0
        mock_gen.assert_not_called()

    @patch("stropha.inference.generate", side_effect=RuntimeError("backend down"))
    def test_rewrite_handles_inference_exception(self, mock_gen: MagicMock) -> None:
        result = rewrite_query("how does auth work", force_enabled=True)

        assert result.query is None
        assert result.reason == "backend_error"
        assert result.generation_calls == 1
        assert result.estimated_input_tokens > 0
        assert result.latency_ms >= 0


class TestHybridRrfWithQueryIntelligence:
    """Tests for hybrid-rrf integration with HyDE and Query Rewriting."""

    def test_config_accepts_hyde_flag(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig

        config = HybridRrfConfig(hyde_enabled=True)
        assert config.hyde_enabled is True

        config = HybridRrfConfig(hyde_enabled=False)
        assert config.hyde_enabled is False

    def test_config_accepts_query_rewrite_flag(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig

        config = HybridRrfConfig(query_rewrite_enabled=True)
        assert config.query_rewrite_enabled is True

        config = HybridRrfConfig(query_rewrite_enabled=False)
        assert config.query_rewrite_enabled is False

    def test_config_defaults_disabled(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig

        config = HybridRrfConfig()
        assert config.hyde_enabled is False
        assert config.query_rewrite_enabled is False

    def test_config_accepts_selective_multi_query_policy(self) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig

        config = HybridRrfConfig(
            multi_query_enabled=True,
            multi_query_count=2,
            multi_query_min_terms=8,
        )
        assert config.multi_query_enabled
        assert config.multi_query_count == 2
        assert config.multi_query_min_terms == 8


class TestHybridRrfSearchIntegration:
    """Integration tests for HyDE and query rewrite in hybrid-rrf.search()."""

    @pytest.fixture
    def recording_storage(self):
        """Storage stub that records search calls and queries."""
        from stropha.models import SearchHit

        class _RecordingStorage:
            def __init__(self) -> None:
                self.dense_calls: list[tuple[list[float], int]] = []
                self.bm25_calls: list[tuple[str, int]] = []
                self.symbol_calls: list[tuple[str, int]] = []

            @staticmethod
            def _hit(rank: int, chunk_id: str, score: float = 1.0) -> SearchHit:
                return SearchHit(
                    rank=rank,
                    score=score,
                    rel_path=f"file_{chunk_id}.py",
                    language="python",
                    kind="file",
                    symbol=None,
                    start_line=1,
                    end_line=10,
                    snippet=f"snippet {chunk_id}",
                    chunk_id=chunk_id,
                )

            def search_dense(
                self, vec: list[float], k: int, *, repo_id: int | None = None
            ) -> list[SearchHit]:
                self.dense_calls.append((vec, k))
                return [self._hit(i + 1, f"d{i}") for i in range(min(k, 3))]

            def search_bm25(
                self, query: str, k: int, *, repo_id: int | None = None
            ) -> list[SearchHit]:
                self.bm25_calls.append((query, k))
                return [self._hit(i + 1, f"b{i}") for i in range(min(k, 3))]

            def search_symbol_tokens(
                self, query: str, k: int, *, repo_id: int | None = None
            ) -> list[SearchHit]:
                self.symbol_calls.append((query, k))
                return [self._hit(i + 1, f"s{i}") for i in range(min(k, 3))]

        return _RecordingStorage()

    @pytest.fixture
    def recording_embedder(self):
        """Embedder stub that records which text was embedded."""
        from pydantic import BaseModel

        from stropha.pipeline.base import StageHealth

        class _Config(BaseModel):
            pass

        class _RecordingEmbedder:
            def __init__(self) -> None:
                self.query_texts: list[str] = []

            @property
            def stage_name(self) -> str:
                return "embedder"

            @property
            def adapter_name(self) -> str:
                return "stub"

            @property
            def adapter_id(self) -> str:
                return "stub"

            @property
            def model_name(self) -> str:
                return "stub"

            @property
            def dim(self) -> int:
                return 4

            @property
            def batch_size(self) -> int:
                return 16

            @property
            def config_schema(self):
                return _Config

            def health(self):
                return StageHealth(status="ready", message="stub")

            def embed_documents(self, texts):
                return [[0.0] * 4 for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                self.query_texts.append(text)
                return [1.0, 0.0, 0.0, 0.0]

        return _RecordingEmbedder()

    @patch("stropha.inference.generate")
    def test_search_uses_hyde_when_config_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When hyde_enabled=True, search should use HyDE-rewritten query for embedding."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        mock_gen.return_value = "def hypothetical_code(): pass"

        config = HybridRrfConfig(hyde_enabled=True)
        retrieval = HybridRrfRetrieval(
            config, storage=recording_storage, embedder=recording_embedder
        )

        retrieval.search("When a user submits login credentials to the server")

        # Embedder should receive the HyDE-rewritten query (hypothetical code)
        assert len(recording_embedder.query_texts) == 1
        assert "hypothetical_code" in recording_embedder.query_texts[0]

    @patch("stropha.inference.generate")
    def test_search_uses_hyde_when_yaml_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder, isolated_test_env
    ) -> None:
        """When hyde_enabled in YAML, search should use HyDE."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  hyde_enabled: true
""")
        reset_config()

        mock_gen.return_value = "class AuthService: pass"

        config = HybridRrfConfig()  # Will pick up from YAML
        retrieval = HybridRrfRetrieval(
            config, storage=recording_storage, embedder=recording_embedder
        )

        retrieval.search("When a user starts the authentication flow")

        assert len(recording_embedder.query_texts) == 1
        assert "AuthService" in recording_embedder.query_texts[0]

    @patch("stropha.inference.generate")
    def test_search_uses_query_rewrite_when_config_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder
    ) -> None:
        """When query_rewrite_enabled=True, search should use rewritten query for all streams."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        mock_gen.return_value = "FSRS calculator FsrsCalculator test"

        config = HybridRrfConfig(query_rewrite_enabled=True)
        retrieval = HybridRrfRetrieval(
            config, storage=recording_storage, embedder=recording_embedder
        )

        retrieval.search("onde tem teste pra fsrs")

        # BM25 and symbol streams should receive the rewritten query
        assert len(recording_storage.bm25_calls) == 1
        bm25_query = recording_storage.bm25_calls[0][0]
        # Rewritten query combines original + expanded terms
        assert "fsrs" in bm25_query.lower()
        assert "FSRS" in bm25_query or "FsrsCalculator" in bm25_query

    @patch("stropha.inference.generate")
    def test_search_uses_query_rewrite_when_yaml_enabled(
        self, mock_gen: MagicMock, recording_storage, recording_embedder, isolated_test_env
    ) -> None:
        """When query_rewrite_enabled in YAML, search should use rewriting."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        yaml_path = isolated_test_env / ".stropha" / "config.yaml"
        yaml_path.write_text("""
retrieval_features:
  query_rewrite_enabled: true
""")
        reset_config()

        mock_gen.return_value = "mastery streak transition"

        config = HybridRrfConfig()  # Will pick up from YAML
        retrieval = HybridRrfRetrieval(
            config, storage=recording_storage, embedder=recording_embedder
        )

        retrieval.search("como funciona mastery")

        assert len(recording_storage.bm25_calls) == 1
        bm25_query = recording_storage.bm25_calls[0][0]
        assert "mastery" in bm25_query.lower()
        assert "streak" in bm25_query or "transition" in bm25_query

    def test_search_uses_both_hyde_and_query_rewrite(
        self, recording_storage, recording_embedder
    ) -> None:
        """Explicitly enabled expansions retain their independent stream effects."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        # Mock both functions directly since they both use the inference backend
        with (
            patch("stropha.retrieval.query_rewrite.rewrite_query") as mock_rewrite,
            patch("stropha.retrieval.hyde.generate_hypothetical") as mock_hyde,
        ):
            # Query rewrite returns expanded terms
            mock_rewrite.return_value = QueryRewriteResult(
                query="how does auth work authentication login AuthService",
                reason="success",
                generation_calls=1,
            )
            mock_hyde.return_value = HydeResult(
                "class LoginHandler { authenticate() {} }",
                "success",
                generation_calls=1,
            )

            config = HybridRrfConfig(hyde_enabled=True, query_rewrite_enabled=True)
            retrieval = HybridRrfRetrieval(
                config, storage=recording_storage, embedder=recording_embedder
            )

            retrieval.search("how does auth work")

            mock_rewrite.assert_called_once()
            mock_hyde.assert_called_once()

            # Verify force_enabled was passed
            _, kwargs = mock_rewrite.call_args
            assert kwargs.get("force_enabled") is True

            _, kwargs = mock_hyde.call_args
            assert kwargs.get("force_enabled") is True

            # Dense embedding should use HyDE output.
            assert len(recording_embedder.query_texts) == 1
            assert "LoginHandler" in recording_embedder.query_texts[0]

            # BM25/symbol should use query rewrite output
            assert len(recording_storage.bm25_calls) == 1
            bm25_query = recording_storage.bm25_calls[0][0]
            assert "authentication" in bm25_query or "AuthService" in bm25_query

    def test_search_fallback_when_hyde_fails(self, recording_storage, recording_embedder) -> None:
        """When HyDE fails, search should fall back to original query."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        with patch("stropha.inference.generate") as mock_gen:
            mock_gen.return_value = None  # Simulate failure

            config = HybridRrfConfig(hyde_enabled=True)
            retrieval = HybridRrfRetrieval(
                config, storage=recording_storage, embedder=recording_embedder
            )

            query = "When a user submits credentials the server validates them"
            retrieval.search(query)

            # Should still embed something (the original query)
            assert len(recording_embedder.query_texts) == 1
            assert recording_embedder.query_texts[0] == query

    def test_search_fallback_when_query_rewrite_fails(
        self, recording_storage, recording_embedder
    ) -> None:
        """When query rewrite fails, search should fall back to original query."""
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        with patch("stropha.inference.generate") as mock_gen:
            mock_gen.return_value = None  # Simulate failure

            config = HybridRrfConfig(query_rewrite_enabled=True)
            retrieval = HybridRrfRetrieval(
                config, storage=recording_storage, embedder=recording_embedder
            )

            retrieval.search("original query")

            # BM25 should receive the original query
            assert len(recording_storage.bm25_calls) == 1
            assert recording_storage.bm25_calls[0][0] == "original query"

    def test_candidate_config_does_not_inherit_global_feature_toggles(
        self, recording_storage, recording_embedder, isolated_test_env
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
        from stropha.config import reset_config

        (isolated_test_env / ".stropha").mkdir(exist_ok=True)
        (isolated_test_env / ".stropha" / "config.yaml").write_text(
            "retrieval_features:\n"
            "  hyde_enabled: true\n"
            "  query_rewrite_enabled: true\n"
            "  multi_query_enabled: true\n"
        )
        reset_config()
        retrieval = HybridRrfRetrieval(
            HybridRrfConfig(inherit_global_features=False),
            storage=recording_storage,
            embedder=recording_embedder,
        )

        with patch("stropha.inference.generate") as generate:
            retrieval.search("original query")

        generate.assert_not_called()
        assert recording_embedder.query_texts == ["original query"]
        assert recording_storage.bm25_calls[0][0] == "original query"

    def test_explicit_hyde_prompt_style_reaches_rewriter(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        retrieval = HybridRrfRetrieval(
            HybridRrfConfig(
                hyde_enabled=True,
                hyde_code_tuned=False,
                inherit_global_features=False,
            ),
            storage=recording_storage,
            embedder=recording_embedder,
        )
        with patch(
            "stropha.retrieval.hyde.generate_hypothetical",
            return_value=HydeResult("hypothetical", "success", generation_calls=1),
        ) as rewrite:
            retrieval.search("When a request arrives the service validates its payload")

        rewrite.assert_called_once_with(
            "When a request arrives the service validates its payload",
            force_enabled=True,
            code_tuned=False,
        )

    def test_query_rewrite_diagnostics_record_success_skip_and_fallback(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        retrieval = HybridRrfRetrieval(
            HybridRrfConfig(query_rewrite_enabled=True, inherit_global_features=False),
            storage=recording_storage,
            embedder=recording_embedder,
        )
        outcomes = [
            QueryRewriteResult(
                query="how does auth work AuthService",
                reason="success",
                generation_calls=1,
                estimated_input_tokens=10,
                estimated_output_tokens=4,
                latency_ms=5.0,
            ),
            QueryRewriteResult(query=None, reason="symbol_query"),
            QueryRewriteResult(
                query=None,
                reason="backend_error",
                generation_calls=1,
                estimated_input_tokens=8,
                latency_ms=7.0,
            ),
        ]
        with patch("stropha.retrieval.query_rewrite.rewrite_query", side_effect=outcomes):
            retrieval.search("how does auth work")
            retrieval.search("UserService")
            retrieval.search("how does login work")

        diagnostics = retrieval.diagnostics()["query_rewrite"]
        assert diagnostics == {
            "enabled": True,
            "policy_id": diagnostics["policy_id"],
            "max_terms": 6,
            "generation_calls": 2,
            "successes": 1,
            "fallbacks": 1,
            "fallback_rate": 0.5,
            "estimated_input_tokens": 18,
            "estimated_output_tokens": 4,
            "estimated_total_tokens": 22,
            "latency_ms": 12.0,
            "latency_p95_ms": 7.0,
            "skips": {
                "empty_query": 0,
                "symbol_query": 1,
                "structural_query": 0,
                "not_natural": 0,
                "too_long": 0,
            },
        }

        retrieval.reset_diagnostics()
        assert retrieval.diagnostics()["query_rewrite"]["generation_calls"] == 0

    def test_adapter_id_tracks_effective_rewrite_policy_and_backend(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        with patch("stropha.inference.get_backend") as get_backend:
            get_backend.return_value.identity = "local:test-model"
            first = HybridRrfRetrieval(
                HybridRrfConfig(
                    query_rewrite_enabled=True,
                    query_rewrite_max_terms=4,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id
            second = HybridRrfRetrieval(
                HybridRrfConfig(
                    query_rewrite_enabled=True,
                    query_rewrite_max_terms=6,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id

        assert "rewrite=short-natural-v1:max=4" in first
        assert first != second

    def test_multi_query_diagnostics_record_policy_cost_and_duplicates(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        retrieval = HybridRrfRetrieval(
            HybridRrfConfig(
                multi_query_enabled=True,
                multi_query_count=2,
                multi_query_min_terms=7,
                inherit_global_features=False,
            ),
            storage=recording_storage,
            embedder=recording_embedder,
        )
        outcomes = [
            MultiQueryResult(
                ("broad query", "alternate query"),
                "success",
                requested_paraphrases=2,
                raw_paraphrases=3,
                duplicate_paraphrases=1,
                generation_calls=1,
                estimated_input_tokens=10,
                estimated_output_tokens=4,
                latency_ms=5.0,
            ),
            MultiQueryResult(("short query",), "too_short", requested_paraphrases=2),
            MultiQueryResult(
                ("failed query",),
                "backend_error",
                requested_paraphrases=2,
                generation_calls=1,
                estimated_input_tokens=8,
                latency_ms=7.0,
            ),
        ]
        with patch("stropha.retrieval.multi_query.expand_query", side_effect=outcomes):
            retrieval.search("broad query")
            retrieval.search("short query")
            retrieval.search("failed query")

        diagnostics = retrieval.diagnostics()["multi_query"]
        assert diagnostics["enabled"] is True
        assert diagnostics["count"] == 2
        assert diagnostics["min_terms"] == 7
        assert diagnostics["generation_calls"] == 2
        assert diagnostics["successes"] == 1
        assert diagnostics["fallbacks"] == 1
        assert diagnostics["fallback_rate"] == 0.5
        assert diagnostics["requested_paraphrases"] == 6
        assert diagnostics["generated_paraphrases"] == 1
        assert diagnostics["raw_paraphrases"] == 3
        assert diagnostics["duplicate_paraphrases"] == 1
        assert diagnostics["searches"] == 2
        assert diagnostics["estimated_total_tokens"] == 22
        assert diagnostics["latency_ms"] == 12.0
        assert diagnostics["latency_p95_ms"] == 7.0
        assert diagnostics["candidate_duplicates_before_fusion"] > 0
        assert diagnostics["candidate_duplicate_rate_before_fusion"] == 0.5
        assert diagnostics["candidate_duplicates_after_fusion"] == 0
        assert diagnostics["skips"]["too_short"] == 1

        retrieval.reset_diagnostics()
        assert retrieval.diagnostics()["multi_query"]["generation_calls"] == 0

    def test_adapter_id_tracks_effective_multi_query_policy_and_backend(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        with patch("stropha.inference.get_backend") as get_backend:
            get_backend.return_value.identity = "local:test-model"
            count_two = HybridRrfRetrieval(
                HybridRrfConfig(
                    multi_query_enabled=True,
                    multi_query_count=2,
                    multi_query_min_terms=7,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id
            count_three = HybridRrfRetrieval(
                HybridRrfConfig(
                    multi_query_enabled=True,
                    multi_query_count=3,
                    multi_query_min_terms=7,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id

        assert "mq=broad-conceptual-v1:min=7" in count_two
        assert "count=2" in count_two
        assert count_two != count_three

    def test_hyde_diagnostics_record_policy_cost_and_fallback(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        retrieval = HybridRrfRetrieval(
            HybridRrfConfig(hyde_enabled=True, inherit_global_features=False),
            storage=recording_storage,
            embedder=recording_embedder,
        )
        outcomes = [
            HydeResult(
                "hypothetical code",
                "success",
                generation_calls=1,
                estimated_input_tokens=10,
                estimated_output_tokens=4,
                latency_ms=5.0,
            ),
            HydeResult(None, "not_behavior"),
            HydeResult(
                None,
                "backend_error",
                generation_calls=1,
                estimated_input_tokens=8,
                latency_ms=7.0,
            ),
        ]
        with patch("stropha.retrieval.hyde.generate_hypothetical", side_effect=outcomes):
            retrieval.search("When an agent submits prose the auditor checks citations")
            retrieval.search("How does citation auditing work across repositories")
            retrieval.search("After citations are parsed the auditor resolves references")

        diagnostics = retrieval.diagnostics()["hyde"]
        assert diagnostics["enabled"] is True
        assert diagnostics["code_tuned"] is True
        assert diagnostics["generation_calls"] == 2
        assert diagnostics["successes"] == 1
        assert diagnostics["fallbacks"] == 1
        assert diagnostics["fallback_rate"] == 0.5
        assert diagnostics["estimated_total_tokens"] == 22
        assert diagnostics["latency_ms"] == 12.0
        assert diagnostics["latency_p95_ms"] == 7.0
        assert diagnostics["skips"]["not_behavior"] == 1

        retrieval.reset_diagnostics()
        assert retrieval.diagnostics()["hyde"]["generation_calls"] == 0

    def test_adapter_id_tracks_effective_hyde_policy_and_backend(
        self, recording_storage, recording_embedder
    ) -> None:
        from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval

        with patch("stropha.inference.get_backend") as get_backend:
            get_backend.return_value.identity = "local:test-model"
            code_tuned = HybridRrfRetrieval(
                HybridRrfConfig(
                    hyde_enabled=True,
                    hyde_code_tuned=True,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id
            generic = HybridRrfRetrieval(
                HybridRrfConfig(
                    hyde_enabled=True,
                    hyde_code_tuned=False,
                    inherit_global_features=False,
                ),
                storage=recording_storage,
                embedder=recording_embedder,
            ).adapter_id

        assert "hyde=behavior-description-v1:code=1" in code_tuned
        assert "llm=" in code_tuned
        assert code_tuned != generic
