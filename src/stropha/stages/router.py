"""RouterStage protocol — pluggable query routing (WIN-234).

Query routing classifies a natural-language query into one of the
``QueryIntent`` values (FIND_CALLERS / FIND_TESTS / SEARCH_CODE /
…) so the dispatch layer can pick the right tool. Today the
routing logic is hard-wired in :mod:`stropha.retrieval.query_router`;
WIN-234 turns it into a Stage so:

1. The Meta-Harness can treat router choice as a search-space
   dimension (rule-based vs. LLM-augmented vs. learned).
2. Operators can A/B different routers without touching the
   ``smart_search`` call path.
3. Future routers (LLM ranker, ML classifier, hand-tuned
   per-repo rules) plug in via ``@register_adapter`` without
   modifying the dispatch layer.

Routers do NOT execute the routed query — they ONLY classify it.
Dispatch stays in the caller (e.g. ``smart_search`` / the MCP
tool layer).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..pipeline.base import StageHealth
from ..retrieval.query_router import RoutedQuery


@runtime_checkable
class RouterStage(Protocol):
    """Classify a query into a ``RoutedQuery`` (intent + symbol)."""

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    def route(self, query: str) -> RoutedQuery:
        """Return a :class:`RoutedQuery` describing the agent's intent.

        Routers MUST NOT raise. On any internal failure, fall back to
        ``QueryIntent.SEARCH_CODE`` with a low confidence + a textual
        reason — the dispatch layer's job is to surface the fallback
        explicitly, not the router's job to abort the agent's call.
        """
        ...
