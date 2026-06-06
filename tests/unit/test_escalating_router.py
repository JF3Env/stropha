"""Unit tests for confidence-gated escalating router + LLM router
(WIN-235, EPIC 1.10 #26)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stropha.adapters.router.escalating import (
    EscalatingRouter,
    EscalatingRouterConfig,
    _build_subrouter,
)
from stropha.adapters.router.llm import LLMRouter, LLMRouterConfig
from stropha.adapters.router.noop import NoopRouter
from stropha.adapters.router.pattern import PatternRouter
from stropha.pipeline.registry import lookup_adapter
from stropha.retrieval.query_router import QueryIntent, RoutedQuery
from stropha.stages.router import RouterStage

# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class TestLLMRouter:
    def test_registered(self) -> None:
        assert lookup_adapter("router", "llm") is LLMRouter

    def test_implements_router_stage(self) -> None:
        assert isinstance(LLMRouter(), RouterStage)

    def test_routes_via_llm_classify(self) -> None:
        with patch(
            "stropha.adapters.router.llm._llm_classify",
            return_value=RoutedQuery(
                intent=QueryIntent.FIND_CALLERS,
                symbol="Foo",
                original_query="who calls Foo",
                confidence=0.85,
                reason="LLM classification",
            ),
        ):
            result = LLMRouter().route("who calls Foo")
        assert result.intent == QueryIntent.FIND_CALLERS
        assert result.symbol == "Foo"

    def test_falls_back_to_searchcode_on_none(self) -> None:
        with patch(
            "stropha.adapters.router.llm._llm_classify",
            return_value=None,
        ):
            result = LLMRouter().route("anything")
        assert result.intent == QueryIntent.SEARCH_CODE
        assert "fallback" in result.reason

    def test_health_ready(self) -> None:
        assert LLMRouter().health().status == "ready"

    def test_stage_metadata(self) -> None:
        r = LLMRouter()
        assert r.stage_name == "router"
        assert r.adapter_name == "llm"
        assert r.adapter_id == "llm"
        assert r.config_schema is LLMRouterConfig


# ---------------------------------------------------------------------------
# EscalatingRouter
# ---------------------------------------------------------------------------


def _stub_router(name: str, *, intent=QueryIntent.SEARCH_CODE,
                 confidence=0.5, symbol=None, reason="stub") -> MagicMock:
    """Build a MagicMock that quacks like a RouterStage."""
    r = MagicMock()
    r.adapter_name = name
    r.adapter_id = name
    r.route.return_value = RoutedQuery(
        intent=intent,
        symbol=symbol,
        original_query="",
        confidence=confidence,
        reason=reason,
    )
    return r


class TestEscalatingRouter:
    def test_registered(self) -> None:
        assert lookup_adapter("router", "escalating") is EscalatingRouter

    def test_primary_kept_when_confident(self) -> None:
        primary = _stub_router("p", confidence=0.95)
        escalation = _stub_router("e", confidence=0.99)
        router = EscalatingRouter(
            primary=primary, escalation=escalation,
        )
        result = router.route("q")
        # Primary above threshold (0.7) → escalation not called.
        escalation.route.assert_not_called()
        assert result.confidence == 0.95

    def test_primary_below_threshold_triggers_escalation(self) -> None:
        primary = _stub_router("p", confidence=0.3)
        escalation = _stub_router("e", confidence=0.9,
                                  intent=QueryIntent.FIND_CALLERS)
        router = EscalatingRouter(
            primary=primary, escalation=escalation,
        )
        result = router.route("q")
        escalation.route.assert_called_once()
        # Escalation wins (its confidence > primary's).
        assert result.intent == QueryIntent.FIND_CALLERS
        assert "escalated from" in result.reason

    def test_escalation_without_improvement_keeps_primary(self) -> None:
        primary = _stub_router("p", confidence=0.4,
                               intent=QueryIntent.SEARCH_CODE)
        escalation = _stub_router("e", confidence=0.3,
                                  intent=QueryIntent.FIND_RELATED)
        router = EscalatingRouter(
            primary=primary, escalation=escalation,
        )
        result = router.route("q")
        # Escalation called but did NOT improve; primary's verdict
        # survives.
        escalation.route.assert_called_once()
        assert result.intent == QueryIntent.SEARCH_CODE
        assert "did not improve" in result.reason

    def test_threshold_boundary_inclusive(self) -> None:
        """A primary at exactly the threshold counts as confident."""
        primary = _stub_router("p", confidence=0.7)
        escalation = _stub_router("e", confidence=0.9)
        router = EscalatingRouter(
            EscalatingRouterConfig(threshold=0.7),
            primary=primary, escalation=escalation,
        )
        router.route("q")
        escalation.route.assert_not_called()

    def test_primary_exception_degrades_to_zero(self) -> None:
        primary = MagicMock()
        primary.adapter_name = "p"
        primary.route.side_effect = RuntimeError("boom")
        escalation = _stub_router("e", confidence=0.9)
        router = EscalatingRouter(
            primary=primary, escalation=escalation,
        )
        result = router.route("q")
        # Primary degraded to 0.0; escalation runs + wins.
        escalation.route.assert_called_once()
        assert result.confidence == 0.9

    def test_escalation_exception_keeps_primary(self) -> None:
        primary = _stub_router("p", confidence=0.3)
        escalation = MagicMock()
        escalation.adapter_name = "e"
        escalation.route.side_effect = RuntimeError("boom")
        router = EscalatingRouter(
            primary=primary, escalation=escalation,
        )
        result = router.route("q")
        # Escalation raised → degraded to 0.0, which doesn't beat
        # primary's 0.3.
        assert result.confidence == 0.3

    def test_adapter_id_includes_subrouter_ids_and_threshold(
        self,
    ) -> None:
        primary = _stub_router("pattern")
        primary.adapter_id = "pattern:llm=0"
        escalation = _stub_router("llm")
        escalation.adapter_id = "llm"
        router = EscalatingRouter(
            EscalatingRouterConfig(threshold=0.65),
            primary=primary, escalation=escalation,
        )
        assert (
            router.adapter_id
            == "escalating:primary=pattern:llm=0:escalation=llm:t=0.65"
        )

    def test_health_ready_mentions_subrouters_and_threshold(
        self,
    ) -> None:
        primary = _stub_router("p")
        escalation = _stub_router("e")
        router = EscalatingRouter(
            EscalatingRouterConfig(threshold=0.55),
            primary=primary, escalation=escalation,
        )
        h = router.health()
        assert h.status == "ready"
        assert "p" in h.message
        assert "e" in h.message
        assert "0.55" in h.message

    def test_stage_metadata(self) -> None:
        router = EscalatingRouter(
            primary=_stub_router("p"), escalation=_stub_router("e"),
        )
        assert router.stage_name == "router"
        assert router.adapter_name == "escalating"
        assert router.config_schema is EscalatingRouterConfig


# ---------------------------------------------------------------------------
# _build_subrouter helper
# ---------------------------------------------------------------------------


class TestBuildSubrouter:
    def test_builds_pattern(self) -> None:
        r = _build_subrouter("pattern")
        assert isinstance(r, PatternRouter)

    def test_builds_noop(self) -> None:
        r = _build_subrouter("noop")
        assert isinstance(r, NoopRouter)

    def test_builds_llm(self) -> None:
        r = _build_subrouter("llm")
        assert isinstance(r, LLMRouter)

    def test_fallback_when_no_arg_init_fails(self) -> None:
        """Adapter that doesn't accept no-arg should still build via
        the cls.Config() fallback path."""

        class _FakeRouter:
            Config = type("C", (), {"__init__": lambda self: None})

            def __init__(self, config: object) -> None:
                self._config = config

        with patch(
            "stropha.adapters.router.escalating.lookup_adapter",
            return_value=_FakeRouter,
        ):
            result = _build_subrouter("fake")
        assert isinstance(result, _FakeRouter)


# ---------------------------------------------------------------------------
# End-to-end via the pattern + LLM stack
# ---------------------------------------------------------------------------


class TestEscalatingEndToEnd:
    def test_obvious_query_takes_fast_path(self) -> None:
        """A clear 'tests for X' query matches the pattern at 0.9
        confidence; the LLM should not be called."""
        with patch(
            "stropha.adapters.router.llm._llm_classify",
        ) as mock_llm:
            router = EscalatingRouter()  # default: pattern + llm
            result = router.route("tests for UserService")
        # LLM not called — pattern at 0.9 is above the 0.7 threshold.
        mock_llm.assert_not_called()
        assert result.intent == QueryIntent.FIND_TESTS

    def test_ambiguous_query_escalates_to_llm(self) -> None:
        """A query that doesn't match any pattern (low confidence)
        triggers the LLM call."""
        with patch(
            "stropha.adapters.router.llm._llm_classify",
            return_value=RoutedQuery(
                intent=QueryIntent.FIND_CALLERS,
                symbol="Foo",
                original_query="ambiguous",
                confidence=0.8,
                reason="LLM classification",
            ),
        ) as mock_llm:
            router = EscalatingRouter()
            # An ambiguous query — won't match the regex set.
            result = router.route(
                "tell me everything about your scheduler"
            )
        mock_llm.assert_called_once()
        # The LLM's verdict won.
        assert result.intent == QueryIntent.FIND_CALLERS
        assert result.confidence == 0.8
