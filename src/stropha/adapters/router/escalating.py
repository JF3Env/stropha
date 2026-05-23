"""Confidence-gated escalating router (WIN-235, EPIC 1.10 #26).

The agent makes hundreds of search calls per session; the typical
distribution is ~95% obvious queries (clear "what calls X", "tests
for Y") + ~5% ambiguous queries that benefit from an LLM
classifier. Calling the LLM on every query inverts the cost shape:
cheap-when-cheap, expensive-when-needed.

``EscalatingRouter`` composes two routers (primary + escalation)
into one Stage:

1. Run the primary (cheap, e.g. pattern-based).
2. If primary confidence >= threshold, return that.
3. Otherwise, run the escalation (expensive, e.g. LLM-based) and
   return its result — but ONLY if the escalation's confidence
   actually beats the primary's. The escalation never overrides
   a confident pattern match it didn't itself improve on.

This is the production-shape router: cheap path 95% of the time,
LLM path only for genuine ambiguity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...logging import get_logger
from ...pipeline.base import StageHealth
from ...pipeline.registry import lookup_adapter, register_adapter
from ...retrieval.query_router import RoutedQuery
from ...stages.router import RouterStage

log = get_logger(__name__)


class EscalatingRouterConfig(BaseModel):
    """Two-router composition + the confidence threshold."""

    primary: str = Field(
        default="pattern",
        description=(
            "Adapter name for the cheap first-pass router. "
            "Defaults to 'pattern'."
        ),
    )
    escalation: str = Field(
        default="llm",
        description=(
            "Adapter name for the fallback router. Defaults to "
            "'llm'. Set to 'noop' to disable escalation entirely."
        ),
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence below which the primary's verdict is "
            "considered uncertain and escalation runs. 0.7 catches "
            "the WIN-235 'cheap path 95% of the time' target."
        ),
    )


@register_adapter(stage="router", name="escalating")
class EscalatingRouter(RouterStage):
    """Composite router: primary first, escalation when uncertain."""

    Config = EscalatingRouterConfig

    def __init__(
        self,
        config: EscalatingRouterConfig | None = None,
        *,
        primary: RouterStage | None = None,
        escalation: RouterStage | None = None,
    ) -> None:
        self._config = config or EscalatingRouterConfig()
        # Build sub-routers from the registry unless the caller
        # injected them directly (used in tests).
        self._primary = primary or _build_subrouter(self._config.primary)
        self._escalation = escalation or _build_subrouter(
            self._config.escalation
        )

    @property
    def stage_name(self) -> str:
        return "router"

    @property
    def adapter_name(self) -> str:
        return "escalating"

    @property
    def adapter_id(self) -> str:
        # Both sub-router ids + threshold are baked in — drift fires
        # on any composition change.
        p = getattr(self._primary, "adapter_id", "?")
        e = getattr(self._escalation, "adapter_id", "?")
        return (
            f"escalating:primary={p}:escalation={e}"
            f":t={self._config.threshold:.2f}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return EscalatingRouterConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=(
                "escalating router (primary="
                f"{getattr(self._primary, 'adapter_name', '?')}, "
                "escalation="
                f"{getattr(self._escalation, 'adapter_name', '?')}, "
                f"threshold={self._config.threshold:.2f})"
            ),
        )

    def route(self, query: str) -> RoutedQuery:
        """Two-pass classification with confidence gating.

        Step 1: ``primary.route(query)``. If confidence ≥ threshold,
        done — fast path.

        Step 2: ``escalation.route(query)``. Take the escalation
        verdict ONLY when its confidence beats the primary's. Tag
        the reason so the operator can tell which path was taken.

        Defensive: if either sub-router raises, treat its verdict
        as a SEARCH_CODE fallback with confidence 0; the other
        sub-router still gets a chance.
        """
        primary_result = self._safe_route(self._primary, query)
        if primary_result.confidence >= self._config.threshold:
            log.info(
                "escalating_router.primary_kept",
                confidence=primary_result.confidence,
                threshold=self._config.threshold,
                intent=primary_result.intent.value,
            )
            return primary_result

        # Primary uncertain — escalate.
        escalation_result = self._safe_route(self._escalation, query)
        # Only override when escalation actually improves confidence.
        if escalation_result.confidence > primary_result.confidence:
            log.info(
                "escalating_router.escalated",
                primary_confidence=primary_result.confidence,
                escalation_confidence=escalation_result.confidence,
                escalation_intent=escalation_result.intent.value,
            )
            # Tag the reason so the agent + operator know what
            # happened.
            return RoutedQuery(
                intent=escalation_result.intent,
                symbol=escalation_result.symbol,
                original_query=escalation_result.original_query,
                confidence=escalation_result.confidence,
                reason=(
                    f"escalated from "
                    f"{getattr(self._primary, 'adapter_name', '?')} "
                    f"(conf {primary_result.confidence:.2f}) → "
                    f"{getattr(self._escalation, 'adapter_name', '?')} "
                    f"(conf {escalation_result.confidence:.2f}); "
                    f"original: {escalation_result.reason}"
                ),
            )
        # Escalation didn't improve — keep the primary's verdict
        # but note the attempt.
        log.info(
            "escalating_router.escalation_no_improvement",
            primary_confidence=primary_result.confidence,
            escalation_confidence=escalation_result.confidence,
        )
        return RoutedQuery(
            intent=primary_result.intent,
            symbol=primary_result.symbol,
            original_query=primary_result.original_query,
            confidence=primary_result.confidence,
            reason=(
                f"{primary_result.reason}; "
                f"escalation '{getattr(self._escalation, 'adapter_name', '?')}' "
                f"did not improve (conf "
                f"{escalation_result.confidence:.2f} ≤ "
                f"{primary_result.confidence:.2f})"
            ),
        )

    def _safe_route(self, router: RouterStage, query: str) -> RoutedQuery:
        """Wrap any router with the defensive contract."""
        from ...retrieval.query_router import QueryIntent

        try:
            return router.route(query)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "escalating_router.subrouter_failed",
                adapter=getattr(router, "adapter_name", "?"),
                error=str(exc),
            )
            return RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=None,
                original_query=query,
                confidence=0.0,
                reason=(
                    f"sub-router '{getattr(router, 'adapter_name', '?')}' "
                    f"raised: {exc!r}"
                ),
            )


def _build_subrouter(name: str) -> RouterStage:
    """Build a sub-router by adapter name with default config."""
    cls = lookup_adapter("router", name)
    # Sub-routers must be no-arg constructible (they take their
    # config object via __init__(config=...)).
    try:
        return cls()
    except TypeError:
        # Fallback: pass an empty Config when no-arg fails.
        return cls(cls.Config())  # type: ignore[call-arg]
