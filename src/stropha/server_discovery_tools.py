"""Implementation helpers for discovery-oriented MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .errors import StrophaError
from .server_models import OutlineEntry, RepoSummary, SearchResult, SmartSearchResponse
from .server_runtime import AppContext

ToResult = Callable[[Any], SearchResult]


def smart_search_tool_impl(
    query: str,
    top_k: int,
    use_llm_router: bool,
    app: AppContext,
    *,
    to_result: ToResult,
    smart_search_fn: Callable[..., Any],
    graph_loaded_fn: Callable[..., bool],
    repo_id: int | None,
    log: Any,
) -> SmartSearchResponse:
    top_k = max(1, min(top_k, 30))
    try:
        configured_router = getattr(app.built, "router", None)
        result = smart_search_fn(
            query,
            app.storage,
            app.embedder,
            top_k=top_k,
            use_router=True,
            use_llm_router=use_llm_router,
            router=configured_router,
            search_engine=app.search_engine,
            repo_id=repo_id,
        )
        return SmartSearchResponse(
            results=[to_result(hit) for hit in result.hits],
            total_candidates=len(result.hits),
            query=query,
            intent=result.routed.intent.value,
            symbol=result.routed.symbol,
            tool_used=result.tool_used,
            confidence=result.routed.confidence,
            graph_available=graph_loaded_fn(app.storage, repo_id=repo_id),
        )
    except StrophaError as exc:
        log.warning("mcp.smart_search.error", error=str(exc))
        return SmartSearchResponse(
            results=[],
            total_candidates=0,
            query=query,
            intent="search_code",
            symbol=None,
            tool_used="search_code",
            confidence=0.0,
            graph_available=graph_loaded_fn(app.storage, repo_id=repo_id),
        )


def get_symbol_impl(
    symbol: str,
    limit: int,
    app: AppContext,
    *,
    to_result: ToResult,
    repo_id: int | None,
) -> list[SearchResult]:
    bounded_limit = max(1, min(limit, 20))
    if repo_id is None:
        hits = app.storage.lookup_by_symbol(symbol, limit=bounded_limit)
    else:
        hits = app.storage.lookup_by_symbol(symbol, limit=bounded_limit, repo_id=repo_id)
    return [to_result(hit) for hit in hits]


def list_repos_impl(app: AppContext) -> list[RepoSummary]:
    return [
        RepoSummary(
            normalized_key=repo.normalized_key,
            url=repo.url,
            local_path=repo.local_path,
            default_branch=repo.default_branch,
            head_commit=repo.head_commit,
            files=repo.files,
            chunks=repo.chunks,
            last_indexed_at=repo.last_indexed_at,
        )
        for repo in app.storage.list_repos()
    ]


def get_file_outline_impl(
    path: str,
    app: AppContext,
    *,
    repo_id: int | None,
) -> list[OutlineEntry]:
    if repo_id is None:
        rows = app.storage.file_outline(path)
    else:
        rows = app.storage.file_outline(path, repo_id=repo_id)
    return [
        OutlineEntry(
            chunk_id=row["chunk_id"],
            kind=row["kind"],
            symbol=row["symbol"],
            parent_chunk_id=row["parent_chunk_id"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
        )
        for row in rows
    ]
