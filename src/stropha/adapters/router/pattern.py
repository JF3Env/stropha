"""Pattern-based router adapter (WIN-234, EPIC 1.10 #25).

Wraps the long-standing :func:`stropha.retrieval.query_router.route_query`
regex-based classifier as a first-class Stage so the pipeline
adapter framework can swap it for an LLM-backed or learned router
without touching the dispatch layer.

Zero LLM cost. Microseconds per call. The default router unless
the operator explicitly selects another.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...retrieval.query_router import RoutedQuery, route_query
from ...stages.router import RouterStage


class PatternRouterConfig(BaseModel):
    """Config for the pattern-based router.

    ``use_llm_fallback=True`` allows the underlying ``route_query`` to
    consult an LLM for ambiguous cases. ``False`` keeps this adapter fully
    deterministic regardless of the global router setting.
    """

    use_llm_fallback: bool = Field(
        default=False,
        description=(
            "Force route_query to escalate ambiguous queries to LLM "
            "classification. False delegates to the global "
            "retrieval_features.query_router_llm setting."
        ),
    )


@register_adapter(stage="router", name="pattern")
class PatternRouter(RouterStage):
    """First-class adapter wrapping the legacy ``route_query``."""

    Config = PatternRouterConfig

    def __init__(self, config: PatternRouterConfig | None = None) -> None:
        self._config = config or PatternRouterConfig()

    @property
    def stage_name(self) -> str:
        return "router"

    @property
    def adapter_name(self) -> str:
        return "pattern"

    @property
    def adapter_id(self) -> str:
        # llm_fallback IS in the id — toggling it changes
        # classification behaviour, so drift-detect should fire on
        # change. (Even though it's a query-time switch and doesn't
        # affect indexed data, downstream golden-eval consumers
        # benefit from a distinct id.)
        return f"pattern:llm={'1' if self._config.use_llm_fallback else '0'}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return PatternRouterConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="pattern router (regex; LLM fallback "
                    f"{'enabled' if self._config.use_llm_fallback else 'disabled'})",
        )

    def route(self, query: str) -> RoutedQuery:
        """Delegate to the legacy classifier.

        Caught + degraded contract: defensive ``except Exception``
        ensures the router never aborts the caller.
        """
        try:
            return route_query(
                query,
                use_llm=self._config.use_llm_fallback,
            )
        except Exception as exc:
            # Emergency fallback to SEARCH_CODE with low confidence.
            from ...retrieval.query_router import QueryIntent

            return RoutedQuery(
                intent=QueryIntent.SEARCH_CODE,
                symbol=None,
                original_query=query,
                confidence=0.1,
                reason=f"pattern router exception: {exc!r}",
            )
