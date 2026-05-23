"""LLM-only router adapter (WIN-235, EPIC 1.10 #26).

Delegates EVERY query to the LLM classifier. Used as the
"escalation" router that the :class:`EscalatingRouter` falls
through to when the cheap pattern-based first pass is uncertain.

Standalone use is also fine — operators can pick this directly
when they want max routing accuracy and don't mind the per-query
LLM cost.

Latency / cost trade-off: each call is ~1-3s + a few hundred
tokens; the pattern router handles 95% of queries in microseconds.
The escalating composite (WIN-235) is the right shape for
production.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...retrieval.query_router import (
    QueryIntent,
    RoutedQuery,
    _llm_classify,
)
from ...stages.router import RouterStage


class LLMRouterConfig(BaseModel):
    """Config schema — no per-adapter knobs yet.

    Future stories may surface the prompt template or temperature;
    for the MVP we reuse the existing :func:`_llm_classify` helper.
    """

    description: str = Field(
        default="LLM-only router",
        description="Human-readable label (does not affect behaviour).",
    )


@register_adapter(stage="router", name="llm")
class LLMRouter(RouterStage):
    """Always-LLM router. Escalation target for WIN-235."""

    Config = LLMRouterConfig

    def __init__(self, config: LLMRouterConfig | None = None) -> None:
        self._config = config or LLMRouterConfig()

    @property
    def stage_name(self) -> str:
        return "router"

    @property
    def adapter_name(self) -> str:
        return "llm"

    @property
    def adapter_id(self) -> str:
        return "llm"

    @property
    def config_schema(self) -> type[BaseModel]:
        return LLMRouterConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="llm router (uses _llm_classify from query_router)",
        )

    def route(self, query: str) -> RoutedQuery:
        """Call the LLM classifier; fall back to SEARCH_CODE on any
        failure so the agent's call never aborts."""
        result = _llm_classify(query)
        if result is None:
            return RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=None,
                original_query=query,
                confidence=0.2,
                reason="LLM classification returned None — fallback",
            )
        return result
