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

    def test_default_explicitly_disables_llm_classification(self) -> None:
        router = PatternRouter()
        with patch("stropha.adapters.router.pattern.route_query") as route:
            route.return_value = RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=None,
                original_query="q",
                confidence=0.5,
                reason="fallback",
            )
            router.route("q")
        route.assert_called_once_with("q", use_llm=False)

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
        assert result.intent == QueryIntent.SEARCH_CODE

    @pytest.mark.parametrize(
        ("query", "intent", "symbol"),
        [
            ("what calls FsrsCalculator", QueryIntent.FIND_CALLERS, "FsrsCalculator"),
            ("who uses UserService.authenticate", QueryIntent.FIND_CALLERS, "UserService.authenticate"),
            ("callers of build_stages", QueryIntent.FIND_CALLERS, "build_stages"),
            ("where is Pipeline.run called", QueryIntent.FIND_CALLERS, "Pipeline.run"),
            ("tests for StreamWeightProfiles", QueryIntent.FIND_TESTS, "StreamWeightProfiles"),
            ("how is smart_search tested", QueryIntent.FIND_TESTS, "smart_search"),
            ("test coverage for PatternRouter", QueryIntent.FIND_TESTS, "PatternRouter"),
            ("what relates to SearchEngine", QueryIntent.FIND_RELATED, "SearchEngine"),
            ("related symbols for QueryIntent", QueryIntent.FIND_RELATED, "QueryIntent"),
            ("Pipeline dependencies", QueryIntent.FIND_RELATED, "Pipeline"),
            ("trace feature user submits answer", QueryIntent.TRACE_FEATURE, "user submits answer"),
            ("follow the checkout flow", QueryIntent.TRACE_FEATURE, "checkout"),
            ("how does request handling flow", QueryIntent.TRACE_FEATURE, "request handling"),
            ("execution path for indexing", QueryIntent.TRACE_FEATURE, "indexing"),
            ("where is HybridRrfRetrieval defined", QueryIntent.GET_SYMBOL, "HybridRrfRetrieval"),
            ("definition of StreamWeightProfile", QueryIntent.GET_SYMBOL, "StreamWeightProfile"),
            ("show me PatternRouter", QueryIntent.GET_SYMBOL, "PatternRouter"),
            ("FsrsCalculator", QueryIntent.GET_SYMBOL, "FsrsCalculator"),
            ("how does authentication work", QueryIntent.SEARCH_CODE, None),
            ("explain retrieval quality", QueryIntent.SEARCH_CODE, None),
        ],
    )
    def test_deterministic_routing_benchmark(
        self,
        query: str,
        intent: QueryIntent,
        symbol: str | None,
    ) -> None:
        result = PatternRouter().route(query)
        assert result.intent == intent
        assert result.symbol == symbol

    def test_deterministic_routing_accuracy_gate(self) -> None:
        cases = [
            ("what calls FsrsCalculator", QueryIntent.FIND_CALLERS, "FsrsCalculator"),
            ("who uses UserService", QueryIntent.FIND_CALLERS, "UserService"),
            ("tests for SearchEngine", QueryIntent.FIND_TESTS, "SearchEngine"),
            ("how is smart_search tested", QueryIntent.FIND_TESTS, "smart_search"),
            ("what relates to Pipeline", QueryIntent.FIND_RELATED, "Pipeline"),
            ("Pipeline dependencies", QueryIntent.FIND_RELATED, "Pipeline"),
            ("trace feature user login", QueryIntent.TRACE_FEATURE, "user login"),
            ("how does request handling flow", QueryIntent.TRACE_FEATURE, "request handling"),
            ("definition of PatternRouter", QueryIntent.GET_SYMBOL, "PatternRouter"),
            ("where is SearchEngine defined", QueryIntent.GET_SYMBOL, "SearchEngine"),
            ("how does authentication work", QueryIntent.SEARCH_CODE, None),
            ("explain retrieval quality", QueryIntent.SEARCH_CODE, None),
            ("find code handling malformed profiles", QueryIntent.SEARCH_CODE, None),
            ("why is RRF used here", QueryIntent.SEARCH_CODE, None),
            ("show examples of storage migration", QueryIntent.SEARCH_CODE, None),
            ("what validates user input", QueryIntent.SEARCH_CODE, None),
            ("how are adapters registered", QueryIntent.SEARCH_CODE, None),
            ("describe the indexing pipeline", QueryIntent.SEARCH_CODE, None),
            ("where is query routing configured", QueryIntent.SEARCH_CODE, None),
            ("what happens after a commit", QueryIntent.SEARCH_CODE, None),
        ]
        correct = sum(
            (result := PatternRouter().route(query)).intent == intent
            and result.symbol == symbol
            for query, intent, symbol in cases
        )

        assert correct / len(cases) >= 0.95

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

    @pytest.mark.parametrize(
        ("intent", "symbol", "tool", "patched"),
        [
            (QueryIntent.FIND_CALLERS, "Foo", "find_callers", "find_callers"),
            (QueryIntent.FIND_TESTS, "Foo", "find_tests_for", "find_tests_for"),
            (QueryIntent.FIND_RELATED, "Foo", "find_related", "find_related"),
            (QueryIntent.TRACE_FEATURE, "feature flow", "trace_feature", "trace_feature"),
        ],
    )
    def test_graph_routes_preserve_repository_scope(
        self,
        intent: QueryIntent,
        symbol: str,
        tool: str,
        patched: str,
    ) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.route.return_value = RoutedQuery(
            intent=intent,
            symbol=symbol,
            original_query="q",
            confidence=0.9,
            reason="benchmark route",
        )
        graph_result = {"paths": []} if intent == QueryIntent.TRACE_FEATURE else {}
        with (
            patch("stropha.retrieval.graph.graph_loaded", return_value=True),
            patch(f"stropha.retrieval.graph.{patched}", return_value=graph_result) as dispatch,
        ):
            result = smart_search(
                "q",
                MagicMock(),
                MagicMock(),
                router=router,
                repo_id=42,
            )

        assert result.tool_used == tool
        assert dispatch.call_args.kwargs["repo_id"] == 42

    def test_symbol_route_preserves_repository_scope(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.route.return_value = RoutedQuery(
            intent=QueryIntent.GET_SYMBOL,
            symbol="Foo",
            original_query="q",
            confidence=0.9,
            reason="benchmark route",
        )
        storage = MagicMock()
        storage.lookup_by_symbol.return_value = []

        result = smart_search("q", storage, MagicMock(), router=router, repo_id=42)

        assert result.tool_used == "get_symbol"
        storage.lookup_by_symbol.assert_called_once_with("Foo", limit=10, repo_id=42)

    def test_trace_route_converts_flat_graph_nodes_to_hits(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.route.return_value = RoutedQuery(
            intent=QueryIntent.TRACE_FEATURE,
            symbol="feature flow",
            original_query="trace feature flow",
            confidence=0.9,
            reason="trace pattern",
        )
        node = {
            "node_id": "repo:src_feature_run",
            "label": "run()",
            "source_file": "src/feature.py",
            "source_location": "L12",
        }
        with (
            patch("stropha.retrieval.graph.graph_loaded", return_value=True),
            patch(
                "stropha.retrieval.graph.trace_feature",
                return_value={"paths": [{"chain": [node]}], "nodes": [node]},
            ),
        ):
            result = smart_search(
                "trace feature flow",
                MagicMock(),
                MagicMock(),
                router=router,
                repo_id=42,
            )

        assert result.tool_used == "trace_feature"
        assert [hit.rel_path for hit in result.hits] == ["src/feature.py"]

    def test_trace_route_preserves_path_chain_order(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.route.return_value = RoutedQuery(
            intent=QueryIntent.TRACE_FEATURE,
            symbol="feature flow",
            original_query="trace feature flow",
            confidence=0.9,
            reason="trace pattern",
        )
        entry = {
            "node_id": "repo:entry",
            "label": "entry()",
            "source_file": "src/entry.py",
        }
        called = {
            "node_id": "repo:called",
            "label": "called()",
            "source_file": "src/called.py",
        }
        unrelated = {
            "node_id": "repo:unrelated",
            "label": "unrelated()",
            "source_file": "src/unrelated.py",
        }
        with (
            patch("stropha.retrieval.graph.graph_loaded", return_value=True),
            patch(
                "stropha.retrieval.graph.trace_feature",
                return_value={
                    "paths": [{"chain": [entry, called]}],
                    "nodes": [unrelated, called, entry],
                },
            ),
        ):
            result = smart_search(
                "trace feature flow",
                MagicMock(),
                MagicMock(),
                top_k=2,
                router=router,
                repo_id=42,
            )

        assert [hit.rel_path for hit in result.hits] == ["src/entry.py", "src/called.py"]

    def test_graph_unavailable_fallback_is_explicit_and_scoped(self) -> None:
        from stropha.retrieval.smart_search import smart_search

        router = MagicMock()
        router.route.return_value = RoutedQuery(
            intent=QueryIntent.FIND_CALLERS,
            symbol="Foo",
            original_query="what calls Foo",
            confidence=0.9,
            reason="callers pattern",
        )
        engine = MagicMock()
        engine.search.return_value = []
        storage = MagicMock()
        with patch("stropha.retrieval.graph.graph_loaded", return_value=False) as loaded:
            result = smart_search(
                "what calls Foo",
                storage,
                MagicMock(),
                router=router,
                search_engine=engine,
                repo_id=42,
            )

        assert result.tool_used == "search_code"
        assert result.routed.intent == QueryIntent.SEARCH_CODE
        assert "graph not loaded" in result.routed.reason
        loaded.assert_called_once_with(storage, repo_id=42)
        engine.search.assert_called_once_with("what calls Foo", top_k=10, repo_id=42)


class TestSmartSearchMcpMetadata:
    def test_response_surfaces_route_reason_and_repository_scope(self) -> None:
        from stropha.retrieval.smart_search import SmartSearchResult
        from stropha.server_discovery_tools import smart_search_tool_impl

        routed = RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol="Foo",
            original_query="what calls Foo",
            confidence=0.45,
            reason="graph not loaded, fallback from find_callers",
        )
        result = SmartSearchResult(hits=[], routed=routed, tool_used="search_code")
        app = MagicMock()
        search = MagicMock(return_value=result)

        response = smart_search_tool_impl(
            "what calls Foo",
            10,
            False,
            app,
            to_result=MagicMock(),
            smart_search_fn=search,
            graph_loaded_fn=MagicMock(return_value=False),
            repo_id=42,
            log=MagicMock(),
        )

        assert response.route_reason == routed.reason
        assert response.confidence == 0.45
        assert response.graph_available is False
        assert search.call_args.kwargs["repo_id"] == 42
