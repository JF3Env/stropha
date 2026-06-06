"""Unit tests for the pluggable router stage (WIN-234, EPIC 1.10 #25).

Covers:
- ``RouterStage`` Protocol surface.
- ``PatternRouter`` adapter wrapping the legacy ``route_query``.
- ``NoopRouter`` adapter for the A/B baseline.
- Pipeline builder constructs a router with a sensible default.
- ``BuiltStages.router`` field.
- ``smart_search`` uses the injected router.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stropha.adapters.router.noop import NoopRouter, NoopRouterConfig
from stropha.adapters.router.pattern import PatternRouter, PatternRouterConfig
from stropha.pipeline.registry import lookup_adapter
from stropha.retrieval.query_router import QueryIntent, RoutedQuery
from stropha.stages.router import RouterStage

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_pattern_router_registered(self) -> None:
        cls = lookup_adapter("router", "pattern")
        assert cls is PatternRouter

    def test_noop_router_registered(self) -> None:
        cls = lookup_adapter("router", "noop")
        assert cls is NoopRouter


# ---------------------------------------------------------------------------
# Pattern router
# ---------------------------------------------------------------------------


class TestPatternRouter:
    def test_implements_router_stage(self) -> None:
        router = PatternRouter()
        assert isinstance(router, RouterStage)

    def test_default_adapter_id(self) -> None:
        router = PatternRouter()
        assert router.adapter_id == "pattern:llm=0"

    def test_llm_fallback_in_adapter_id(self) -> None:
        router = PatternRouter(PatternRouterConfig(use_llm_fallback=True))
        assert router.adapter_id == "pattern:llm=1"

    def test_classifies_callers_query(self) -> None:
        router = PatternRouter()
        result = router.route("what calls FsrsCalculator")
        assert result.intent == QueryIntent.FIND_CALLERS
        assert result.symbol == "FsrsCalculator"

    def test_classifies_tests_query(self) -> None:
        router = PatternRouter()
        result = router.route("tests for UserService")
        assert result.intent == QueryIntent.FIND_TESTS
        assert result.symbol == "UserService"

    def test_falls_back_to_search_code_on_unrelated_query(
        self,
    ) -> None:
        router = PatternRouter()
        result = router.route("how does the auth flow work")
        assert result.intent in (
            QueryIntent.SEARCH_CODE,
            QueryIntent.TRACE_FEATURE,  # "how does ... work" pattern
        )

    def test_route_query_exception_returns_searchcode_fallback(
        self,
    ) -> None:
        router = PatternRouter()
        with patch(
            "stropha.adapters.router.pattern.route_query",
            side_effect=RuntimeError("boom"),
        ):
            result = router.route("anything")
        assert result.intent == QueryIntent.SEARCH_CODE
        assert "pattern router exception" in result.reason

    def test_health_ready(self) -> None:
        h = PatternRouter().health()
        assert h.status == "ready"

    def test_stage_metadata(self) -> None:
        router = PatternRouter()
        assert router.stage_name == "router"
        assert router.adapter_name == "pattern"
        assert router.config_schema is PatternRouterConfig


# ---------------------------------------------------------------------------
# Noop router
# ---------------------------------------------------------------------------


class TestNoopRouter:
    def test_implements_router_stage(self) -> None:
        router = NoopRouter()
        assert isinstance(router, RouterStage)

    def test_always_returns_search_code(self) -> None:
        router = NoopRouter()
        for q in ("what calls Foo", "tests for Bar", "how does X work"):
            result = router.route(q)
            assert result.intent == QueryIntent.SEARCH_CODE
            assert result.symbol is None
            assert result.confidence == 1.0
            assert "noop" in result.reason

    def test_adapter_id_stable(self) -> None:
        assert NoopRouter().adapter_id == "noop"

    def test_health_ready(self) -> None:
        h = NoopRouter().health()
        assert h.status == "ready"

    def test_stage_metadata(self) -> None:
        router = NoopRouter()
        assert router.stage_name == "router"
        assert router.adapter_name == "noop"
        assert router.config_schema is NoopRouterConfig


# ---------------------------------------------------------------------------
# Pipeline builder integration
# ---------------------------------------------------------------------------


class TestBuilderRouter:
    def _minimal_resolved(self, router_section: dict | None = None) -> dict:
        base = {
            "walker": {"adapter": "git-ls-files", "config": {}},
            "chunker": {
                "adapter": "tree-sitter-dispatch",
                "config": {},
            },
            "enricher": {"adapter": "noop", "config": {}},
            "embedder": {"adapter": "local", "config": {}},
            "storage": {"adapter": "sqlite-vec", "config": {}},
            "retrieval": {"adapter": "hybrid-rrf", "config": {}},
        }
        if router_section is not None:
            base["router"] = router_section
        return base

    def test_default_router_is_pattern(self, tmp_path) -> None:
        from stropha.pipeline.builder import build_stages

        cfg = self._minimal_resolved()
        cfg["storage"]["config"] = {"path": str(tmp_path / "i.db")}
        built = build_stages(cfg)
        try:
            assert built.router is not None
            assert built.router.adapter_name == "pattern"
        finally:
            if built.storage is not None:
                built.storage.close()

    def test_explicit_noop_router(self, tmp_path) -> None:
        from stropha.pipeline.builder import build_stages

        cfg = self._minimal_resolved({"adapter": "noop", "config": {}})
        cfg["storage"]["config"] = {"path": str(tmp_path / "i.db")}
        built = build_stages(cfg)
        try:
            assert built.router is not None
            assert built.router.adapter_name == "noop"
        finally:
            if built.storage is not None:
                built.storage.close()

    def test_unknown_router_logged_no_crash(self, tmp_path) -> None:
        """An unknown router adapter doesn't crash the build — it
        logs + falls back to ``router=None`` so smart_search drops
        to the legacy route_query path."""
        from stropha.pipeline.builder import build_stages

        cfg = self._minimal_resolved({
            "adapter": "ghost-router-99", "config": {},
        })
        cfg["storage"]["config"] = {"path": str(tmp_path / "i.db")}
        built = build_stages(cfg)
        try:
            assert built.router is None
        finally:
            if built.storage is not None:
                built.storage.close()

    def test_router_invalid_config_raises(self, tmp_path) -> None:
        """A config schema violation surfaces as ConfigError, not a
        silent swallow — matches every other stage's behaviour."""
        from stropha.errors import ConfigError
        from stropha.pipeline.builder import build_stages

        cfg = self._minimal_resolved({
            "adapter": "pattern",
            "config": {"use_llm_fallback": "not-a-bool"},
        })
        cfg["storage"]["config"] = {"path": str(tmp_path / "i.db")}
        with pytest.raises(ConfigError):
            build_stages(cfg)


# ---------------------------------------------------------------------------
# smart_search uses the injected router
# ---------------------------------------------------------------------------


class TestSmartSearchUsesRouter:
    def test_router_route_called(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        # Mock router returns a fixed RoutedQuery.
        router = MagicMock()
        router.adapter_name = "mock"
        router.route.return_value = RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol=None,
            original_query="q",
            confidence=1.0,
            reason="mock",
        )
        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage.search_dense.return_value = []
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        smart_search(
            "q", storage, embedder, top_k=5, router=router,
        )
        router.route.assert_called_once_with("q")

    def test_router_exception_falls_back_to_searchcode(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.adapter_name = "broken"
        router.route.side_effect = RuntimeError("boom")
        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage.search_dense.return_value = []
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        result = smart_search(
            "q", storage, embedder, top_k=5, router=router,
        )
        # Fallback should be SEARCH_CODE with the failure recorded.
        assert result.routed.intent == QueryIntent.SEARCH_CODE
        assert "raised" in result.routed.reason

    def test_use_router_false_skips_router(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage.search_dense.return_value = []
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        result = smart_search(
            "q", storage, embedder, top_k=5,
            router=router, use_router=False,
        )
        router.route.assert_not_called()
        assert result.routed.intent == QueryIntent.SEARCH_CODE
        assert "disabled" in result.routed.reason

    def test_no_router_falls_back_to_legacy_route_query(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        storage = MagicMock()
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage.search_dense.return_value = []
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        with patch(
            "stropha.retrieval.smart_search.route_query",
            return_value=RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=None, original_query="q",
                confidence=1.0, reason="legacy",
            ),
        ) as mock_legacy:
            smart_search("q", storage, embedder, top_k=5)
        mock_legacy.assert_called_once()
