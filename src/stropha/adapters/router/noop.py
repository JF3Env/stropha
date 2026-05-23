"""No-op router adapter (WIN-234).

Every query routes to ``SEARCH_CODE`` with confidence 1.0. Useful
as an A/B baseline when measuring whether the smart router earns
its keep, and as the fallback when the operator wants to disable
routing entirely without removing ``smart_search`` from the
toolchain.
"""

from __future__ import annotations

from pydantic import BaseModel

from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...retrieval.query_router import QueryIntent, RoutedQuery
from ...stages.router import RouterStage


class NoopRouterConfig(BaseModel):
    """Empty config — no knobs."""


@register_adapter(stage="router", name="noop")
class NoopRouter(RouterStage):
    """Always returns ``SEARCH_CODE`` — disables routing."""

    Config = NoopRouterConfig

    def __init__(self, config: NoopRouterConfig | None = None) -> None:
        self._config = config or NoopRouterConfig()

    @property
    def stage_name(self) -> str:
        return "router"

    @property
    def adapter_name(self) -> str:
        return "noop"

    @property
    def adapter_id(self) -> str:
        return "noop"

    @property
    def config_schema(self) -> type[BaseModel]:
        return NoopRouterConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="noop router (always SEARCH_CODE)",
        )

    def route(self, query: str) -> RoutedQuery:
        return RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol=None,
            original_query=query,
            confidence=1.0,
            reason="routing disabled (noop router)",
        )
