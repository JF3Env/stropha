"""MCP server entry point.

Wires Config → configured pipeline stages inside a FastMCP lifespan and
registers the MCP tool surface. Core discovery examples include:

- `search_code`         — hybrid retrieval (dense + BM25 + RRF).
- `get_symbol`          — direct symbol lookup.
- `get_file_outline`    — symbolic outline of a single file.
- `list_repos`          — enumerate repositories present in the index.
- `index_stats` resource (stropha://stats).

Search results include a ``repo`` field (URL, default branch, HEAD) when
repository metadata is available, so a remote MCP client can ``git clone``
the source. The server speaks
the MCP ``streamable-http`` transport at ``http://<host>:<port>/mcp``;
``bin/stropha-mcp-ensure`` is the canonical daemon launcher and MCP
clients connect to it as a ``remote`` server.

Background file watcher:
When ``mcp.watch`` is enabled in ``~/.stropha/config.yaml`` (the default), the
shared runtime polls the target repository and triggers a debounced light
re-index so MCP queries see saved edits. Disable that YAML setting when manual
indexing or the post-commit hook is sufficient.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, get_config, get_config_info, reset_config
from .errors import ConfigError, EmbeddingDimensionMismatchError, StrophaError
from .logging import configure_logging, get_logger
from .meta.decorator import traced
from .pipeline import build_stages, load_pipeline_config
from .retrieval.graph import (
    find_callers as graph_find_callers,
)
from .retrieval.graph import (
    find_rationale as graph_find_rationale,
)
from .retrieval.graph import (
    find_related as graph_find_related,
)
from .retrieval.graph import (
    find_tests_for as graph_find_tests_for,
)
from .retrieval.graph import (
    get_community as graph_get_community,
)
from .retrieval.graph import (
    graph_loaded,
    has_rationale_edges,
)
from .retrieval.graph import (
    trace_feature as graph_trace_feature,
)
from .retrieval.savings import track_search_savings
from .retrieval.smart_search import smart_search
from .server_discovery_tools import (
    get_file_outline_impl,
    get_symbol_impl,
    list_repos_impl,
    smart_search_tool_impl,
)
from .server_frontend import (
    answer_index_question,
    config_path,
    configuration_catalog,
    merge_config_document,
    read_config_document,
    redact_config_document,
    write_config_document,
)
from .server_graph_tools import (
    assert_snippet_freshness_impl,
    audit_citations_impl,
    blast_radius_impl,
    check_symbols_impl,
    contract_tests_impl,
    find_callers_impl,
    find_rationale_impl,
    find_related_impl,
    find_tests_for_impl,
    get_community_impl,
    spec_consistency_impl,
    test_impact_impl,
    trace_feature_impl,
    verify_claim_impl,
)
from .server_graph_tools import (
    default_spec_loader as server_default_spec_loader,
)
from .server_http import RestartTolerantFastMCP
from .server_memory_tools import (
    bootstrap_environment_impl,
    cache_prune_impl,
    cache_stats_impl,
    check_staleness_impl,
    co_occurrence_impl,
    entry_from_record,
    episode_to_entry,
    forget_impl,
    recall_episodes_impl,
    recall_impl,
    record_episode_impl,
    remember_impl,
    scratchpad_append_impl,
    scratchpad_clear_impl,
    scratchpad_read_impl,
    scratchpad_write_impl,
)
from .server_memory_tools import (
    current_head_commit as server_current_head_commit,
)
from .server_memory_tools import (
    resolve_target_repo_id as server_resolve_target_repo_id,
)
from .server_models import (
    BlastRadiusResponse,
    BootstrapResponse,
    CitationAuditResponse,
    ContractTestsResponse,
    CoOccurrenceResponse,
    CrossSessionCachePruneResponse,
    CrossSessionCacheStats,
    EpisodeEntry,
    ExpandChunkResponse,
    ExpandChunksResponse,
    FacetCounts,  # noqa: F401 — re-exported for test_search_filters.py
    ForgetResponse,
    FreshnessResponse,
    HallucinationGuardResponse,
    ListPinsResponse,
    MemoryEntry,
    OutlineEntry,
    PinResponse,
    RecallEpisodesResponse,
    RecallResponse,
    RecordEpisodeResponse,
    RememberResponse,
    RepoInfo,
    RepoSummary,
    ResetDedupResponse,
    ScratchpadResponse,
    SearchResponse,
    SearchResult,
    SmartSearchResponse,
    SpecConsistencyResponse,
    StalenessResponse,
    StatsPayload,
    TestImpactResponse,
    VerdictResponse,
)
from .server_runtime import (
    AppContext,
    chunk_belongs_to_repo,
    ctx_from_mcp,
    get_shared_runtime,
    persist_session_state,
    reload_shared_runtime,
)
from .server_runtime import (
    lifespan as server_lifespan,
)
from .server_search import apply_filters as server_apply_filters
from .server_search import search_code_impl
from .server_session_tools import (
    expand_chunk_impl,
    expand_chunks_impl,
    list_pins_impl,
    pin_chunks_impl,
    reset_dedup_impl,
    unpin_chunks_impl,
)
from .watch import WatchController

log = get_logger(__name__)


# ---- composition root ------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: RestartTolerantFastMCP) -> AsyncIterator[AppContext]:
    async with server_lifespan(
        app,
        version=__version__,
        get_config_fn=get_config,
        configure_logging_fn=configure_logging,
        load_pipeline_config_fn=load_pipeline_config,
        build_stages_fn=build_stages,
        watch_controller_cls=WatchController,
    ) as app_ctx:
        yield app_ctx


_INSTRUCTIONS = """\
stropha_rag — Semantic code search and structural analysis over the indexed codebase.

IMPORTANT: When this MCP server is available, ALWAYS prefer stropha tools over:
- Grep/ripgrep for searching code content
- Read/cat for exploring unfamiliar files
- Glob/find for discovering relevant files
- Manual directory traversal

stropha provides semantic understanding, not just text matching. It finds code by
INTENT ("how does authentication work?") and by STRUCTURE ("who calls this method?").

TOOL SELECTION GUIDE (use the most specific tool):

| Query type                              | Tool              | Why                          |
|-----------------------------------------|-------------------|------------------------------|
| Conceptual / free-text                  | search_code       | Semantic + lexical hybrid    |
| Known class/method/function name        | get_symbol        | Exact match, faster          |
| "Where is X defined?"                   | get_symbol        | Direct lookup                |
| "How does X work?"                      | search_code       | Semantic understanding       |
| "Show me examples of X"                 | search_code       | Finds related patterns       |
| "What's in this file?"                  | get_file_outline  | Structure before full read   |
| "Who calls X?" / "What uses X?"         | find_callers      | Graph traversal (structural) |
| "What tests cover X?"                   | find_tests_for    | Graph + path heuristics      |
| "What relates to X?"                    | find_related      | Bidirectional graph walk     |
| "What's in the same module as X?"       | get_community     | Cluster grouping             |
| "Why was X built this way?"             | find_rationale    | Links to docs/ADRs           |
| "Trace feature X through the code"      | trace_feature     | DFS from entry points        |

WORKFLOW TIPS:
1. Start with search_code or get_symbol to find entry points
2. Use get_file_outline before reading a full file — saves context tokens
3. Use find_callers/find_related to understand dependencies
4. Results include repo.local_path + rel_path for direct file access

The find_* and trace_* tools require a graphify graph in `graphify-out/graph.json`.
When the graph is missing they return {"graph_loaded": false, ...} —
run `graphify .` (or the post-commit hook installed by `stropha hook install`)
to bootstrap.
"""


mcp = RestartTolerantFastMCP(
    name="stropha_rag",
    instructions=_INSTRUCTIONS,
    # FastMCP types ``lifespan`` against ``Callable[[FastMCP[Any]], ...]`` —
    # our ``_lifespan`` is narrower (``RestartTolerantFastMCP``, our
    # subclass). The runtime contract is identical; the cast is a no-op.
    lifespan=_lifespan,  # type: ignore[arg-type]
)


# ---- health endpoint (SSE/HTTP transports only) ----------------------------


@mcp.custom_route("/health", methods=["GET"])
async def _health_check(request):
    """Liveness probe. Returns 200 when the server is accepting connections."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "version": __version__})


# ---- frontend JSON API -----------------------------------------------------

_FRONTEND_UPDATE_LOCK_ATTR = "_stropha_frontend_update_lock"


async def _frontend_runtime():
    return await get_shared_runtime(
        mcp,
        version=__version__,
        get_config_fn=get_config,
        configure_logging_fn=configure_logging,
        load_pipeline_config_fn=load_pipeline_config,
        build_stages_fn=build_stages,
        watch_controller_cls=WatchController,
    )


async def _frontend_reload_runtime():
    return await reload_shared_runtime(
        mcp,
        get_config_fn=get_config,
        configure_logging_fn=configure_logging,
        load_pipeline_config_fn=load_pipeline_config,
        build_stages_fn=build_stages,
        watch_controller_cls=WatchController,
    )


def _frontend_cors_headers(request) -> dict[str, str]:
    from urllib.parse import urlsplit

    origin = request.headers.get("origin")
    if not origin:
        return {}
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, PATCH, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Vary": "Origin",
    }


def _frontend_is_local(request) -> bool:
    from ipaddress import ip_address

    if request.client is None:
        return False
    try:
        return ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def _frontend_json(request, payload: dict, *, status_code: int = 200):
    from starlette.responses import JSONResponse

    return JSONResponse(
        payload,
        status_code=status_code,
        headers=_frontend_cors_headers(request),
    )


@mcp.custom_route("/", methods=["GET"])
async def _frontend_root(request):
    from starlette.responses import RedirectResponse

    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    return RedirectResponse("/dashboard", status_code=307)


@mcp.custom_route("/dashboard", methods=["GET"])
async def _frontend_dashboard(request):
    from pathlib import Path

    from starlette.responses import FileResponse

    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    page = Path(__file__).with_name("dashboard") / "index.html"
    return FileResponse(page, media_type="text/html")


@mcp.custom_route("/api/status", methods=["GET", "OPTIONS"])
async def _frontend_status(request):
    """Report the active pipeline and local MoA connectivity."""
    import asyncio

    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    if request.method == "OPTIONS":
        return _frontend_json(request, {})
    try:
        runtime = await _frontend_runtime()
        from .inference import get_backend
        from .inference.moa_backend import MoaBackend

        active = await asyncio.to_thread(
            get_backend,
            prefer=runtime.config.inference.backend,
        )
        moa = active if isinstance(active, MoaBackend) else MoaBackend()
        if moa is active:
            active_health = moa_health = await asyncio.to_thread(moa.health_check)
        else:
            active_health, moa_health = await asyncio.gather(
                asyncio.to_thread(active.health_check),
                asyncio.to_thread(moa.health_check),
            )
        models = await asyncio.to_thread(moa.list_models)
        stats = await asyncio.to_thread(runtime.built.storage.stats)
        repos = stats.get("repos") or []
    except Exception as exc:
        log.warning("frontend.status_failed", error=str(exc))
        return _frontend_json(request, {"error": "Could not inspect runtime"}, status_code=503)
    return _frontend_json(
        request,
        {
            "version": __version__,
            "pipeline": {
                "embedder": runtime.built.embedder.adapter_id,
                "storage": runtime.built.storage.adapter_id,
                "retrieval": runtime.built.retrieval.adapter_id,
            },
            "index": {
                "chunks": stats.get("chunks", 0),
                "files": stats.get("files", 0),
                "repos": len(repos) if isinstance(repos, list) else repos,
            },
            "inference": {
                "backend": active.name,
                "identity": active.identity,
                "healthy": active_health[0],
                "message": active_health[1],
            },
            "moa": {
                "endpoint": runtime.config.inference.moa_base_url,
                "model": runtime.config.inference.moa_model,
                "healthy": moa_health[0],
                "message": moa_health[1],
                "models": models,
            },
        },
    )


@mcp.custom_route("/api/config", methods=["GET", "PATCH", "OPTIONS"])
async def _frontend_config(request):
    """Read or atomically update the daemon's live configuration."""
    import asyncio

    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    if request.method == "OPTIONS":
        return _frontend_json(request, {})
    if request.method == "GET":
        try:
            document = read_config_document()
        except ConfigError as exc:
            return _frontend_json(request, {"error": str(exc)}, status_code=500)
        return _frontend_json(
            request,
            {"config": redact_config_document(document), "version": __version__},
        )

    try:
        patch = await request.json()
    except Exception:
        return _frontend_json(request, {"error": "Request body must be JSON"}, status_code=400)
    if not isinstance(patch, dict):
        return _frontend_json(
            request, {"error": "Configuration patch must be an object"}, status_code=422
        )

    lock = getattr(mcp, _FRONTEND_UPDATE_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(mcp, _FRONTEND_UPDATE_LOCK_ATTR, lock)

    async with lock:
        path = config_path()
        existed = path.is_file()
        previous: dict = {}
        try:
            previous = read_config_document(path)
            candidate = merge_config_document(previous, patch)
            write_config_document(candidate, path)
            reset_config()
            from .inference.backend import reset_backend

            reset_backend()
            runtime = await _frontend_reload_runtime()
        except EmbeddingDimensionMismatchError as exc:
            # The candidate is valid but fixed-width vector tables require a
            # rebuild before it can replace the current runtime.
            runtime = await _frontend_runtime()
            return _frontend_json(
                request,
                {
                    "config": redact_config_document(candidate),
                    "active": {
                        "embedder": runtime.built.embedder.adapter_id,
                        "storage": runtime.built.storage.adapter_id,
                        "retrieval": runtime.built.retrieval.adapter_id,
                        "inference": runtime.config.inference.backend,
                    },
                    "pending_rebuild": True,
                    "message": str(exc),
                },
                status_code=202,
            )
        except Exception as exc:
            if existed:
                write_config_document(previous, path)
            else:
                path.unlink(missing_ok=True)
            reset_config()
            from .inference.backend import reset_backend

            reset_backend()
            status = 422 if isinstance(exc, (StrophaError, ValueError)) else 500
            message = str(exc) if status == 422 else "Could not apply configuration"
            return _frontend_json(request, {"error": message}, status_code=status)

    return _frontend_json(
        request,
        {
            "config": redact_config_document(candidate),
            "active": {
                "embedder": runtime.built.embedder.adapter_id,
                "storage": runtime.built.storage.adapter_id,
                "retrieval": runtime.built.retrieval.adapter_id,
                "inference": runtime.config.inference.backend,
            },
        },
    )


@mcp.custom_route("/api/config/schema", methods=["GET", "OPTIONS"])
async def _frontend_config_schema(request):
    """Return safe field metadata and registered adapter schemas for the dashboard."""
    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    if request.method == "OPTIONS":
        return _frontend_json(request, {})
    return _frontend_json(request, configuration_catalog())


@mcp.custom_route("/api/query", methods=["POST", "OPTIONS"])
async def _frontend_query(request):
    """Answer a frontend question from retrieved index evidence."""
    import asyncio

    if not _frontend_is_local(request):
        return _frontend_json(request, {"error": "Frontend API is local-only"}, status_code=403)
    if request.method == "OPTIONS":
        return _frontend_json(request, {})
    try:
        payload = await request.json()
    except Exception:
        return _frontend_json(request, {"error": "Request body must be JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return _frontend_json(request, {"error": "Request body must be an object"}, status_code=422)

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return _frontend_json(request, {"error": "question is required"}, status_code=422)
    question = question.strip()
    if len(question) > 8000:
        return _frontend_json(
            request, {"error": "question exceeds 8000 characters"}, status_code=422
        )
    try:
        top_k = int(payload.get("top_k", 8))
        max_tokens = int(payload.get("max_tokens", 512))
    except (TypeError, ValueError):
        return _frontend_json(
            request, {"error": "top_k and max_tokens must be integers"}, status_code=422
        )
    if not 1 <= top_k <= 20 or not 64 <= max_tokens <= 2048:
        return _frontend_json(
            request,
            {"error": "top_k must be 1..20 and max_tokens must be 64..2048"},
            status_code=422,
        )
    include_other_repos = payload.get("include_other_repos", False)
    if not isinstance(include_other_repos, bool):
        return _frontend_json(
            request, {"error": "include_other_repos must be a boolean"}, status_code=422
        )
    raw_history = payload.get("history", [])
    if not isinstance(raw_history, list):
        return _frontend_json(request, {"error": "history must be an array"}, status_code=422)
    history: list[dict[str, str]] = []
    history_chars = 0
    for item in raw_history[-12:]:
        if not isinstance(item, dict):
            return _frontend_json(
                request, {"error": "history entries must be objects"}, status_code=422
            )
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return _frontend_json(
                request,
                {"error": "history entries require user/assistant role and text content"},
                status_code=422,
            )
        history_chars += len(content)
        if history_chars > 24000:
            return _frontend_json(
                request, {"error": "history exceeds 24000 characters"}, status_code=422
            )
        history.append({"role": role, "content": content})

    try:
        runtime = await _frontend_runtime()
        from .inference import generate

        result = await asyncio.to_thread(
            answer_index_question,
            runtime,
            question,
            top_k=top_k,
            max_tokens=max_tokens,
            generate_fn=generate,
            history=history,
            include_other_repos=include_other_repos,
        )
    except Exception as exc:
        log.warning("frontend.query_failed", error=str(exc))
        return _frontend_json(
            request,
            {"error": "Could not answer from the configured index and inference backend"},
            status_code=503,
        )
    return _frontend_json(request, result)


# ---- response models --------------------------------------------------------

# ---- helpers ---------------------------------------------------------------


def _ctx(mcp_ctx: Context) -> AppContext:
    return ctx_from_mcp(mcp_ctx)


def _persist_session_state(app: AppContext) -> None:
    persist_session_state(app)


def _to_result(hit) -> SearchResult:  # type: ignore[no-untyped-def]
    repo: RepoInfo | None = None
    if hit.repo is not None:
        repo = RepoInfo(
            normalized_key=hit.repo.normalized_key,
            url=hit.repo.url,
            local_path=hit.repo.local_path,
            default_branch=hit.repo.default_branch,
            head_commit=hit.repo.head_commit,
        )
    return SearchResult(
        rank=hit.rank,
        score=hit.score,
        path=hit.rel_path,
        start_line=hit.start_line,
        end_line=hit.end_line,
        language=hit.language,
        kind=hit.kind,
        symbol=hit.symbol,
        snippet=hit.snippet,
        chunk_id=hit.chunk_id,
        repo=repo,
        # WIN-214: pass through per-stream RRF contributions so the
        # agent can self-audit which retrieval lanes contributed.
        streams=hit.streams,
    )


# ---- tools ------------------------------------------------------------------


def _apply_filters(
    hits: list,
    *,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    kind: list[str] | None = None,
    exclude_tests: bool = False,
) -> list:
    """Apply post-retrieval filters to search hits.

    Note: This is post-filtering on already-retrieved candidates. For large
    result sets, pre-filtering in SQL would be more efficient (future work).
    """
    return server_apply_filters(
        hits,
        language=language,
        path_prefix=path_prefix,
        kind=kind,
        exclude_tests=exclude_tests,
    )


@mcp.tool(
    title="Search code",
    description=(
        "PREFER THIS over Grep/ripgrep for code search. "
        "Hybrid semantic + lexical search understands intent, not just text. "
        "Use for: 'where is X', 'how does Y work', 'show examples of Z', "
        "'find code that handles authentication', 'what validates user input'. "
        "For a known exact symbol name (e.g., 'UserService.createUser'), use get_symbol instead. "
        "Filters: language=['java','python'], path_prefix='src/', kind=['method','class'], exclude_tests=True. "
        "Set recursive=True to auto-merge related chunks for more context. "
        "Results default to the current repository; repo_url also accepts an exact indexed "
        "absolute local path or normalized key, or set include_other_repos=True explicitly "
        "to search every indexed repository. "
        "Results include file path (relative to repo root), line numbers, and snippet."
    ),
)
@traced("search_code")
def search_code(
    query: str,
    top_k: int = 10,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    kind: list[str] | None = None,
    exclude_tests: bool = False,
    include_facets: bool = False,
    recursive: bool = False,
    repo_url: str | None = None,
    refresh: bool = False,
    context_budget: int | None = None,
    compress: bool = False,
    card_level: str = "full",
    relevance_decay: bool = False,
    dedup: bool = False,
    conversation_aware: bool = False,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> SearchResponse:
    """Hybrid search (dense + BM25, fused via RRF). Returns top_k results
    with path, line range, language, kind, symbol, and a snippet.

    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
        language: Filter by language(s), e.g. ["java", "python"].
        path_prefix: Filter by path prefix, e.g. "backend/src/".
        kind: Filter by chunk kind(s), e.g. ["method", "class", "function"].
        exclude_tests: If True, exclude test files from results.
        include_facets: If True, include facet counts (language, kind, repo distribution).
        recursive: If True, auto-merge sibling chunks (same parent) and adjacent chunks.
        repo_url: Optional git URL, indexed absolute local path, or normalized repository key.
            Git URLs are cloned/indexed on first access. Local/key selectors avoid
            the shared daemon's process-level default when it differs from the caller.
        include_other_repos: Search all indexed repositories. False scopes results
            to the current target repository. Ignored when repo_url is provided.
        refresh: If True with repo_url, re-clone and re-index the remote repository.
        context_budget: Optional token budget for result packing (WIN-208). When
            set, results are greedily packed in policy order up to this many
            estimated tokens. Score order is the default; opt-in diversity uses
            a stable second pass. Remaining hits are reported via
            ``spilled_count``. Pass when your
            context window is tight; omit for the classic top_k behaviour.
        compress: Extractive compression (WIN-210). When True, each hit's
            snippet is rewritten to keep only the signature, lines matching
            the query, a small surrounding context window, and an
            ``# ... (N lines elided)`` marker for skipped blocks. Doubles
            useful-signal-per-token without losing recall.
        card_level: Disclosure level for each hit's snippet (WIN-212).
            One of "line" (path:lines symbol [kind]), "paragraph"
            (signature + docstring), or "full" (default, classic).
            Lower levels make scanning a result set cheaper; the agent
            can expand specific hits via get_symbol once it has triaged.
        relevance_decay: WIN-213 — when True, apply per-rank card-level
            decay: top hit at "full", ranks 1-2 at "paragraph", ranks 3+
            at "line". Allocates tokens to the most-relevant spans
            instead of spreading them evenly. Overrides card_level when
            both are set.
        dedup: WIN-217 — when True, filter out hits already returned in
            this MCP session. Saves tokens on iterative refinement
            queries that would otherwise re-deliver the same spans. Use
            ``reset_dedup`` to clear the session's seen-set when you
            want a fresh slate.
        conversation_aware: WIN-224 — when True, bias the dense
            semantic retrieval toward the recent conversation thread
            by appending the last few queries as additional context
            to the embedder. BM25 and symbol lanes keep the literal
            query so identifier matches stay clean. Lets follow-up
            queries like "tests for it" carry implicit references
            from earlier turns into the dense match.
    """
    from .tools.git_url import is_git_url, search_git_url_scoped

    remote_repo_selector = bool(repo_url and is_git_url(repo_url))
    if repo_url and not remote_repo_selector and include_other_repos:
        return SearchResponse(results=[], total_candidates=0, query=query, facets=None)

    return search_code_impl(
        query,
        top_k,
        language,
        path_prefix,
        kind,
        exclude_tests,
        include_facets,
        recursive,
        repo_url,
        include_other_repos,
        refresh,
        context_budget,
        compress,
        card_level,
        relevance_decay,
        dedup,
        conversation_aware,
        _ctx(ctx),
        to_result=_to_result,
        persist_session_state=_persist_session_state,
        track_search_savings_fn=track_search_savings,
        is_git_url_fn=is_git_url,
        search_git_url_fn=search_git_url_scoped,
        apply_filters_fn=_apply_filters,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        stropha_error_cls=StrophaError,
        log=log,
    )


@mcp.tool(
    title="Reset session dedup",
    description=(
        "WIN-217 — clear the session's seen-chunk-ids set. Use this when "
        "the agent has finished one logical task and wants a fresh slate "
        "for the next retrieval round, OR when an iterative search keeps "
        "returning empty because everything got dedup'd away. The cleared "
        "state is persisted for durable sessions. Returns the count of ids "
        "dropped."
    ),
)
@traced("reset_dedup")
def reset_dedup(*, ctx: Context) -> ResetDedupResponse:
    """Drop every chunk_id from the session's seen-set."""
    return reset_dedup_impl(
        _ctx(ctx),
        persist_session_state=_persist_session_state,
        log=log,
    )


@mcp.tool(
    title="Pin chunks to session working set",
    description=(
        "WIN-218 — pin one or more chunk_ids so subsequent search_code "
        "calls always include them at the top of the result list. Use "
        "for the inverse of reset_dedup: when the agent has determined "
        "a working set and wants those chunks to stay visible across "
        "queries without re-issuing the search. Pinning wins over "
        "dedup (WIN-217): pinned chunks are NEVER filtered out, even "
        "after they've been served. Idempotent — pinning the same id "
        "twice is a no-op. Defaults to the current repository; "
        "include_other_repos=True permits pins from any indexed repository."
    ),
)
@traced("pin_chunks")
def pin_chunks(
    chunk_ids: list[str],
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> PinResponse:
    """Add ``chunk_ids`` to the session's pin set."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return PinResponse(pinned=[], added=0)
    return pin_chunks_impl(
        chunk_ids,
        app,
        repo_id=repo_id,
        persist_session_state=_persist_session_state,
        log=log,
    )


@mcp.tool(
    title="Unpin chunks",
    description=(
        "WIN-218 — remove one or more chunk_ids from the session's pin "
        "set. Use when the working set has shifted: the chunks that "
        "mattered last turn are no longer relevant. Ids that weren't "
        "in the set are silently ignored — the response surfaces the "
        "actual removal count."
    ),
)
@traced("unpin_chunks")
def unpin_chunks(chunk_ids: list[str], *, ctx: Context) -> PinResponse:
    """Drop ``chunk_ids`` from the session's pin set."""
    return unpin_chunks_impl(
        chunk_ids,
        _ctx(ctx),
        persist_session_state=_persist_session_state,
        log=log,
    )


@mcp.tool(
    title="List pinned chunks",
    description=(
        "WIN-218 — return the current pin set with each chunk's metadata "
        "(path, line range, symbol). Missing ids (chunks deleted after "
        "they were pinned) appear with found=False so the agent can "
        "decide whether to unpin them. Defaults to the current repository; "
        "include_other_repos=True resolves pins from every indexed repository."
    ),
)
@traced("list_pins")
def list_pins(
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> ListPinsResponse:
    """Resolve every pinned chunk_id into a structured response."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return ListPinsResponse(entries=[], found_count=0, missing_count=0)
    return list_pins_impl(app, to_result=_to_result, repo_id=repo_id)


@mcp.tool(
    title="Smart search",
    description=(
        "BEST DEFAULT CHOICE when unsure which stropha tool to use. "
        "Automatically analyzes query intent and routes to the optimal tool:\n"
        "- 'what calls X' → find_callers (graph)\n"
        "- 'tests for X' → find_tests_for (graph)\n"
        "- 'what relates to X' → find_related (graph)\n"
        "- 'trace feature X' → trace_feature (graph)\n"
        "- Conceptual questions → search_code (semantic)\n\n"
        "Returns results plus metadata explaining which tool was used and why. "
        "Prefer this over manually choosing tools when the query type is ambiguous. "
        "Defaults to the current repository; set include_other_repos=True only "
        "for an explicit cross-repository search."
    ),
)
@traced("smart_search")
def smart_search_tool(
    query: str,
    top_k: int = 10,
    use_llm_router: bool = False,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> SmartSearchResponse:
    """Smart search with automatic query routing.

    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
        use_llm_router: If True, use LLM for ambiguous queries (slower but more accurate).
        include_other_repos: Search every indexed repository. False scopes
            all semantic, symbol, and graph routes to the current repository.
    """
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return SmartSearchResponse(
            results=[],
            total_candidates=0,
            query=query,
            intent="search_code",
            symbol=None,
            tool_used="search_code",
            confidence=0.0,
            route_reason="current repository is not indexed",
            graph_available=False,
        )
    return smart_search_tool_impl(
        query,
        top_k,
        use_llm_router,
        app,
        to_result=_to_result,
        smart_search_fn=smart_search,
        graph_loaded_fn=graph_loaded,
        repo_id=repo_id,
        log=log,
    )


@mcp.tool(
    title="Get symbol",
    description=(
        "PREFER THIS when you know the exact class/method/function name. "
        "Faster and more precise than search_code — direct symbol lookup. "
        "Use for: 'where is UserService defined', 'find the createUser method'. "
        "Accepts: simple names ('FsrsCalculator'), qualified names ('StudyService.submitAnswer'), "
        "or partial matches ('submitAnswer'). "
        "Defaults to the current repository; include_other_repos=True broadens the lookup. "
        "Returns file path, line numbers, and code snippet for each match."
    ),
)
@traced("get_symbol")
def get_symbol(
    symbol: str,
    limit: int = 5,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> list[SearchResult]:
    """Find by symbol name. Returns up to `limit` matches, exact-match first."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return []
    return get_symbol_impl(symbol, limit, app, to_result=_to_result, repo_id=repo_id)


@mcp.tool(
    title="List repositories",
    description=(
        "List all repositories in the index with file/chunk counts and paths. "
        "Each entry includes: normalized_key, url (for git clone), local_path (filesystem), "
        "and indexing stats. Use this to understand what code is searchable."
    ),
)
@traced("list_repos")
def list_repos(*, ctx: Context) -> list[RepoSummary]:
    """Aggregate per-repo stats. Cheap query — single GROUP BY.

    Returns both the git origin URL and local filesystem path for each
    repo, so callers can access files via either method.
    """
    return list_repos_impl(_ctx(ctx))


@mcp.tool(
    title="Get file outline",
    description=(
        "ALWAYS USE BEFORE reading a full file — saves context tokens. "
        "Returns the symbolic structure: classes, methods, functions with their line ranges. "
        "Use this to decide WHICH parts of a file to read, instead of reading the whole file. "
        "Example: get_file_outline('src/services/user.py') → shows all classes/methods, "
        "then you can Read just lines 45-67 for the specific method you need. Defaults to "
        "the current repository; include_other_repos=True broadens path matching."
    ),
)
@traced("get_file_outline")
def get_file_outline(
    path: str,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> list[OutlineEntry]:
    """Return the chunk-level outline of a file path (repo-relative)."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return []
    return get_file_outline_impl(path, app, repo_id=repo_id)


@mcp.tool(
    title="Expand chunk by id (just-in-time dereference)",
    description=(
        "WIN-216 — dereference a lightweight chunk handle into the full "
        "SearchResult. Use this after a search_code(card_level='line') or "
        "an assemble_context bundle: the agent gets cheap handles first, "
        "then expands only the chunks it actually needs. Mirrors Claude "
        "Code's file-on-demand pattern. Returns ``found=False`` when the "
        "chunk_id is unknown (e.g. the index was rebuilt and the id is "
        "stale) — never raises. Defaults to the current repository; "
        "include_other_repos=True permits dereferencing any indexed repository."
    ),
)
@traced("expand_chunk")
def expand_chunk(
    chunk_id: str,
    card_level: str = "full",
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> ExpandChunkResponse:
    """Dereference a chunk_id into a SearchResult.

    Args:
        chunk_id: the opaque id returned by ``search_code``,
            ``get_symbol``, ``get_file_outline``, or the
            ``assemble_context`` bundle.
        card_level: optional level to render at — "line",
            "paragraph", or "full" (default). Lets the agent ask
            for just a paragraph when it only needs the signature.

    Returns:
        :class:`ExpandChunkResponse` with ``found`` and (when found)
        the rendered :class:`SearchResult`.
    """
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return ExpandChunkResponse(chunk_id=chunk_id, found=False, result=None)
    return expand_chunk_impl(chunk_id, card_level, app, to_result=_to_result, repo_id=repo_id)


@mcp.tool(
    title="Expand chunks (batched)",
    description=(
        "WIN-216 batch variant of expand_chunk. Pass a list of chunk_ids "
        "to dereference them in one MCP call. Useful when assemble_context "
        "returned a bundle of line-card handles and the agent decided to "
        "expand several at once. Missing chunk_ids appear in the response "
        "with found=False; the call never partially fails. Defaults to the "
        "current repository; include_other_repos=True permits all repositories."
    ),
)
@traced("expand_chunks")
def expand_chunks(
    chunk_ids: list[str],
    card_level: str = "full",
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> ExpandChunksResponse:
    """Batch dereference. Returns a parallel list — one response per id."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return ExpandChunksResponse(
            results=[
                ExpandChunkResponse(chunk_id=chunk_id, found=False, result=None)
                for chunk_id in chunk_ids
            ],
            found_count=0,
            missing_count=len(chunk_ids),
        )
    return expand_chunks_impl(chunk_ids, card_level, app, to_result=_to_result, repo_id=repo_id)


# ---- graph tools (RFC §5 — Trilha A 1.5b) ----------------------------------
# Always registered for tool-discovery stability. When the graphify mirror
# tables are empty (no `graphify-out/graph.json` ever loaded), the tools
# short-circuit with `{graph_loaded: False, message: ...}` so the calling
# agent gets actionable feedback instead of a silent empty list.


def _graph_unavailable() -> dict:
    return {
        "graph_loaded": False,
        "message": (
            "graphify graph not loaded. Run `graphify .` in the repo root to "
            "bootstrap, then `stropha index` to mirror it into SQLite."
        ),
    }


def _graph_repo_unavailable() -> dict:
    return {
        "graph_loaded": False,
        "message": "Current repository could not be resolved; graph query was not run.",
    }


# ---- remember / recall / forget (WIN-223, EPIC 1.9 #15) ---------------------


def _resolve_target_repo_id(app: AppContext, *, repo: str | None = None) -> int | None:
    return server_resolve_target_repo_id(app, repo=repo, log=log)


def _current_head_commit(app: AppContext, *, repo: str | None = None) -> str | None:
    return server_current_head_commit(app, repo=repo)


def _entry_from_record(rec) -> MemoryEntry:  # type: ignore[no-untyped-def]
    return entry_from_record(rec)


@mcp.tool(
    title="Remember a project-scoped fact",
    description=(
        "WIN-223 — write a long-term note about THIS repo so you "
        "don't have to re-explain it on the next session. "
        "Anchor facts that survive across MCP-server restarts: design "
        "decisions ('FSRS scheduler is authoritative'), constraints "
        "('legacy auth module is on the deprecation list'), runbooks "
        "('Java tests run via ./gradlew test --tests'), conventions, "
        "glossary entries — agent picks the ``kind`` bucket. "
        "(kind, key) within a repo is the UPSERT identity: re-issuing "
        "remember() with the same (kind, key) updates the existing "
        "entry, it doesn't duplicate. Use forget() to remove an entry "
        "and recall() to fetch."
    ),
)
@traced("remember")
def remember(
    kind: str,
    key: str,
    value: str,
    *,
    ctx: Context,
    expires_at: str | None = None,
    repo: str | None = None,
) -> RememberResponse:
    """Upsert a project-memory entry.

    ``repo`` (normalized key like ``github.com/org/name`` or a local path)
    anchors the entry to a specific repo in a multi-repo index; defaults to
    the server's configured target repo.
    """
    return remember_impl(
        kind,
        key,
        value,
        _ctx(ctx),
        expires_at=expires_at,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        current_head_commit_fn=_current_head_commit,
        repo=repo,
    )


@mcp.tool(
    title="Recall project-scoped facts",
    description=(
        "WIN-223 — read long-term notes about THIS repo that the agent "
        "(or a previous session) curated via remember(). Filter by "
        "``kind`` to fetch a single category (e.g. all decisions), "
        "and/or by ``key`` for a specific entry. No filters returns "
        "the most-recently updated entries across every kind, capped "
        "by ``limit``. Expired entries are hidden by default; pass "
        "include_expired=True for audit."
    ),
)
@traced("recall")
def recall(
    *,
    ctx: Context,
    kind: str | None = None,
    key: str | None = None,
    limit: int = 50,
    include_expired: bool = False,
    repo: str | None = None,
) -> RecallResponse:
    """Fetch project-memory entries (``repo`` = key/path; default target repo)."""
    return recall_impl(
        _ctx(ctx),
        kind=kind,
        key=key,
        limit=limit,
        include_expired=include_expired,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        repo=repo,
    )


@mcp.tool(
    title="Forget a project-scoped fact",
    description=(
        "WIN-223 — delete a single project_memory entry by (kind, key). "
        "Returns ``removed=False`` silently when the entry didn't "
        "exist; never raises. Pair with recall() to inspect the set "
        "before pruning."
    ),
)
@traced("forget")
def forget(
    kind: str,
    key: str,
    *,
    ctx: Context,
    repo: str | None = None,
) -> ForgetResponse:
    """Delete one project-memory entry (``repo`` = key/path; default target)."""
    return forget_impl(
        kind,
        key,
        _ctx(ctx),
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        repo=repo,
    )


# ---- investigation planning (WIN-232, EPIC 1.10 #23) -----------------------


class InvestigationStepEntry(BaseModel):
    """One step in a returned investigation plan."""

    order: int
    tool: str
    purpose: str
    args: dict = Field(default_factory=dict)


class InvestigationPlanResponse(BaseModel):
    """Outcome of ``plan_investigation`` — a ready-to-execute recipe."""

    task: str
    steps: list[InvestigationStepEntry] = Field(default_factory=list)
    primary_symbols: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@mcp.tool(
    title="Plan investigation route",
    description=(
        "WIN-232 — decompose ``task`` into an ORDERED route over the "
        "graph: entry points → callers (deps) → tests (contract) → "
        "rationale (design docs). Returns a recipe of MCP tool calls "
        "the agent executes in order. The planner does NOT run the "
        "steps itself — it returns a plan; you stay in control of "
        "cost / benefit per step. Use when 'how does X work?' would "
        "otherwise take 3-4 chained calls. Defaults to the current "
        "repository; include_other_repos=True explicitly broadens the plan."
    ),
)
@traced("plan_investigation")
def plan_investigation(
    task: str,
    top_k: int = 5,
    include_callers: bool = True,
    include_tests: bool = True,
    include_rationale: bool = True,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> InvestigationPlanResponse:
    """Return an ordered investigation plan for ``task``.

    Args:
        task: Natural-language investigation target.
        top_k: Maximum primary symbols to plan around.
        include_callers: Include caller-discovery steps.
        include_tests: Include test-discovery steps.
        include_rationale: Include rationale steps when the graph is loaded.
        include_other_repos: Search all indexed repositories. False scopes
            planning to the current repository.
    """
    from .retrieval.investigation import plan_investigation as _plan

    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return InvestigationPlanResponse(
            task=task,
            notes=["Current repository could not be resolved; planning was not run."],
        )
    plan = _plan(
        task,
        search_engine=app.search_engine,
        storage=app.storage,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        include_rationale=include_rationale,
        repo_id=repo_id,
    )
    return InvestigationPlanResponse(
        task=plan.task,
        steps=[
            InvestigationStepEntry(
                order=s.order,
                tool=s.tool,
                purpose=s.purpose,
                args=s.args,
            )
            for s in plan.steps
        ],
        primary_symbols=plan.primary_symbols,
        notes=plan.notes,
    )


# ---- cross-session cache (WIN-230, EPIC 1.9 #22) ---------------------------


@mcp.tool(
    title="Cross-session cache stats",
    description=(
        "WIN-230 — return counters for the Schema v13 cross-session "
        "semantic cache: total_entries, total_hits (cumulative across "
        "lookups), distinct_models. Use to confirm the cache is "
        "actually accumulating useful findings across sessions. The "
        "cache itself is populated by future SearchEngine integration; "
        "this tool inspects the on-disk state."
    ),
)
@traced("cache_stats")
def cache_stats(*, ctx: Context) -> CrossSessionCacheStats:
    """Inspect the cross-session cache for the configured target repo."""
    return cache_stats_impl(
        _ctx(ctx),
        resolve_target_repo_id_fn=_resolve_target_repo_id,
    )


@mcp.tool(
    title="Prune cross-session cache",
    description=(
        "WIN-230 — remove cache entries. Pass ``expired_only=True`` "
        "for TTL cleanup, or ``older_than_iso`` for a cutoff date. "
        "Calling with no filters is a NO-OP (refuses to wipe "
        "everything implicitly)."
    ),
)
@traced("cache_prune")
def cache_prune(
    *,
    ctx: Context,
    expired_only: bool = False,
    older_than_iso: str | None = None,
) -> CrossSessionCachePruneResponse:
    """Bulk-delete cache entries with explicit filters."""
    return cache_prune_impl(
        _ctx(ctx),
        expired_only=expired_only,
        older_than_iso=older_than_iso,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
    )


@mcp.tool(
    title="Read scratchpad",
    description=(
        "WIN-229 — read the durable per-session scratchpad: a "
        "persistent place for working notes about the current task. "
        "Survives MCP-server restarts when STROPHA_CLIENT_ID is set "
        "(via the Schema-v12 sessions.scratchpad column). Use to "
        "stash decisions, intermediate findings, file paths to revisit, "
        "etc. — anything that should outlive a single turn but isn't "
        "yet a stable project_memory fact."
    ),
)
@traced("scratchpad_read")
def scratchpad_read(*, ctx: Context) -> ScratchpadResponse:
    """Return the current scratchpad content."""
    return scratchpad_read_impl(_ctx(ctx))


@mcp.tool(
    title="Write scratchpad",
    description=(
        "WIN-229 — REPLACE the scratchpad content entirely. Use when "
        "the agent is rewriting its notes from scratch (e.g. starting "
        "a new task). For incremental notes, use scratchpad_append. "
        "Persisted to disk via the same write-through path as "
        "WIN-217 dedup + WIN-218 pins (gated on STROPHA_CLIENT_ID)."
    ),
)
@traced("scratchpad_write")
def scratchpad_write(content: str, *, ctx: Context) -> ScratchpadResponse:
    """Overwrite the scratchpad with ``content``."""
    return scratchpad_write_impl(
        content,
        _ctx(ctx),
        persist_session_state_fn=_persist_session_state,
    )


@mcp.tool(
    title="Append to scratchpad",
    description=(
        "WIN-229 — append ``text`` to the existing scratchpad. Adds a "
        "newline separator if the existing content doesn't end with "
        "one. Idiomatic for the agent's running 'what I learned this "
        "turn' notes."
    ),
)
@traced("scratchpad_append")
def scratchpad_append(text: str, *, ctx: Context) -> ScratchpadResponse:
    """Append ``text`` to the scratchpad with newline normalisation."""
    return scratchpad_append_impl(
        text,
        _ctx(ctx),
        persist_session_state_fn=_persist_session_state,
    )


@mcp.tool(
    title="Clear scratchpad",
    description=(
        "WIN-229 — wipe the scratchpad. Returns the empty content "
        "so the agent can verify the clear succeeded. The next "
        "session resume (same STROPHA_CLIENT_ID) starts with an "
        "empty pad."
    ),
)
@traced("scratchpad_clear")
def scratchpad_clear(*, ctx: Context) -> ScratchpadResponse:
    """Empty the scratchpad."""
    return scratchpad_clear_impl(
        _ctx(ctx),
        persist_session_state_fn=_persist_session_state,
    )


# ---- staleness check (WIN-228, EPIC 1.9 #20) -------------------------------


@mcp.tool(
    title="Check memory staleness",
    description=(
        "WIN-228 — diff a memory entry's recorded git SHA against the "
        "current HEAD to detect drift. Use this before acting on a "
        "recalled memory: if the project_memory entry was written "
        "two months and 500 commits ago, the value the agent is "
        "about to follow may be obsolete. Returns ``stale=True`` "
        "when changed files intersect the memory's anchor; "
        "``stale=None`` when we couldn't decide (no provenance "
        "recorded, git failure)."
    ),
)
@traced("check_staleness")
def check_staleness(
    kind: str,
    key: str,
    *,
    ctx: Context,
    anchor_files: list[str] | None = None,
    repo: str | None = None,
) -> StalenessResponse:
    """Determine if the named ``project_memory`` entry is stale.

    ``repo`` (key/path) selects which repo the entry belongs to; the diff runs
    against THAT repo's tree + commit (not the server's target). Defaults to
    the target repo.
    """
    return check_staleness_impl(
        kind,
        key,
        _ctx(ctx),
        anchor_files=anchor_files,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        repo=repo,
    )


@mcp.tool(
    title="Bootstrap repo environment",
    description=(
        "WIN-227 — return a warm-start snapshot of THIS repo: detected "
        "language stacks (Python / Node / Rust / Go / Java / .NET / "
        "Ruby), idiomatic build / test / run commands per stack, "
        "entry-point hints (README, main module, docs roots), and "
        "presence of common directories (tests/, docs/, graphify-out/). "
        "Use this on session open instead of burning 2-4 turns "
        "rediscovering the project basics (paper: TerminalBench-2)."
    ),
)
@traced("bootstrap_environment")
def bootstrap_environment(*, ctx: Context) -> BootstrapResponse:
    """Detect language / build / test / entry-point hints for the
    configured ``target_repo``."""
    return bootstrap_environment_impl(_ctx(ctx))


def _episode_to_entry(rec) -> EpisodeEntry:  # type: ignore[no-untyped-def]
    return episode_to_entry(rec)


@mcp.tool(
    title="Record a task episode",
    description=(
        "WIN-225 — log an episodic-memory entry capturing what the "
        "agent just worked on. Pass a short ``title``, an optional "
        "``summary``, and the ``touched_files`` / ``touched_symbols`` "
        "that were edited or read. Episodes are append-only by design "
        "(the timeline matters); use ``forget_episode`` to delete one. "
        "Pair with ``recall_episodes`` for free-text search and "
        "``co_occurrence`` for 'what do I usually touch alongside X?'."
    ),
)
@traced("record_episode")
def record_episode(
    title: str,
    summary: str = "",
    touched_files: list[str] | None = None,
    touched_symbols: list[str] | None = None,
    *,
    ctx: Context,
    repo: str | None = None,
) -> RecordEpisodeResponse:
    """Record an episodic-memory row (``repo`` = key/path; default target)."""
    return record_episode_impl(
        title,
        summary,
        touched_files,
        touched_symbols,
        _ctx(ctx),
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        current_head_commit_fn=_current_head_commit,
        repo=repo,
    )


@mcp.tool(
    title="Recall task episodes",
    description=(
        "WIN-225 — fetch past episodes for THIS repo. Pass ``query`` "
        "to filter by free-text match in title/summary (case-"
        "insensitive substring); omit it for the most-recent episodes "
        "across all topics. ``since`` (ISO timestamp) filters to "
        "episodes created on/after that time, useful for 'what did I "
        "do this week?'. ``limit`` clamps at 200."
    ),
)
@traced("recall_episodes")
def recall_episodes(
    query: str | None = None,
    limit: int = 10,
    *,
    ctx: Context,
    since: str | None = None,
    repo: str | None = None,
) -> RecallEpisodesResponse:
    """Search past episodes (``repo`` = key/path; default target repo)."""
    return recall_episodes_impl(
        _ctx(ctx),
        query=query,
        limit=limit,
        since=since,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        repo=repo,
    )


@mcp.tool(
    title="Episode co-occurrence",
    description=(
        "WIN-225 — answer 'what do I usually touch alongside ``key``?'. "
        "Scans every episode that touched ``key`` (file path OR symbol) "
        "and ranks the other files / symbols by how often they "
        "co-appeared. Returns two dicts (``files``, ``symbols``) sorted "
        "descending by frequency. Use after picking up an old task: "
        "co_occurrence('src/fsrs/calculator.py') tells you which "
        "migrations / tests / docs the agent usually touched in the "
        "same session."
    ),
)
@traced("co_occurrence")
def co_occurrence(
    key: str,
    *,
    ctx: Context,
    limit: int = 10,
    in_files: bool = True,
    in_symbols: bool = True,
    repo: str | None = None,
) -> CoOccurrenceResponse:
    """Rank co-touched files / symbols (``repo`` = key/path; default target)."""
    return co_occurrence_impl(
        key,
        _ctx(ctx),
        limit=limit,
        in_files=in_files,
        in_symbols=in_symbols,
        resolve_target_repo_id_fn=_resolve_target_repo_id,
        repo=repo,
    )


@mcp.tool(
    title="Find callers of a symbol",
    description=(
        "PREFER THIS for 'who calls X?' or 'what uses X?' queries. "
        "Graph traversal is more precise than text search for call relationships. "
        "Returns callers with file paths and line numbers. "
        "Defaults to the current repository; include_other_repos=True broadens traversal. "
        "Requires graphify graph (returns {graph_loaded: false} if missing)."
    ),
)
@traced("find_callers")
def find_callers(
    symbol: str,
    *,
    ctx: Context,
    depth: int = 1,
    limit: int = 20,
    include_other_repos: bool = False,
) -> dict:
    """Return callers of ``symbol`` (BFS up incoming `calls` edges, depth ≤ 3)."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"symbol": symbol}
    return find_callers_impl(
        symbol,
        app,
        depth=depth,
        limit=limit,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_find_callers_fn=graph_find_callers,
    )


@mcp.tool(
    title="Find related nodes",
    description=(
        "Find all code related to a symbol via any relationship type. "
        "Bidirectional graph traversal — finds both 'X uses Y' and 'Y uses X'. "
        "Filter with relations=['calls','implements','references'] if needed. "
        "Use for exploring dependencies and connections around a symbol. "
        "Defaults to the current repository; include_other_repos=True broadens traversal."
    ),
)
@traced("find_related")
def find_related(
    symbol: str,
    *,
    ctx: Context,
    depth: int = 1,
    limit: int = 20,
    relations: list[str] | None = None,
    include_other_repos: bool = False,
) -> dict:
    """Return any node connected to ``symbol`` via the graph."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"symbol": symbol}
    return find_related_impl(
        symbol,
        app,
        depth=depth,
        limit=limit,
        relations=relations,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_find_related_fn=graph_find_related,
    )


@mcp.tool(
    title="Get community of a symbol",
    description=(
        "Return all nodes in the same community as `symbol_or_community_id`. "
        "Communities are pre-computed by graphify clustering and represent "
        "tight cohesive groups (e.g. 'Hybrid Retrieval & RRF'). Use this to "
        "see everything that participates in a coherent feature/module. "
        "Defaults to the current repository; include_other_repos=True broadens membership."
    ),
)
@traced("get_community")
def get_community(
    symbol_or_community_id: str,
    *,
    ctx: Context,
    limit: int = 50,
    include_other_repos: bool = False,
) -> dict:
    """Return the community membership of a symbol or community id."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"query": symbol_or_community_id}
    return get_community_impl(
        symbol_or_community_id,
        app,
        limit=limit,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_get_community_fn=graph_get_community,
    )


@mcp.tool(
    title="Find rationale for a symbol",
    description=(
        "Return rationale-style nodes (typically docs / ADRs / design "
        "comments) that explain why `symbol` exists or was built this way. "
        "Walks `rationale_for` edges from the graphify graph. Returns empty "
        "when no rationale edges target the symbol. Defaults to the current "
        "repository; include_other_repos=True broadens traversal."
    ),
)
@traced("find_rationale")
def find_rationale(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 10,
    include_other_repos: bool = False,
) -> dict:
    """Return rationale nodes (docs/ADRs) explaining ``symbol``."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"symbol": symbol}
    return find_rationale_impl(
        symbol,
        app,
        limit=limit,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        has_rationale_edges_fn=has_rationale_edges,
        unavailable_fn=_graph_unavailable,
        graph_find_rationale_fn=graph_find_rationale,
    )


@mcp.tool(
    title="Find tests covering a symbol",
    description=(
        "PREFER THIS for 'what tests cover X?' or 'find tests for X'. "
        "Finds test files that exercise a symbol via graph + path heuristics. "
        "Default patterns: test_*, *_test, *.spec.*, *.test.*, /tests/, /test/. "
        "Pass custom test_path_patterns for non-standard test conventions. "
        "Defaults to the current repository; include_other_repos=True broadens traversal."
    ),
)
@traced("find_tests_for")
def find_tests_for(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 20,
    test_path_patterns: list[str] | None = None,
    include_other_repos: bool = False,
) -> dict:
    """Return tests that exercise ``symbol`` (path-pattern filtered)."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"symbol": symbol}
    return find_tests_for_impl(
        symbol,
        app,
        limit=limit,
        test_path_patterns=test_path_patterns,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_find_tests_for_fn=graph_find_tests_for,
    )


@mcp.tool(
    title="Trace a feature through the call graph",
    description=(
        "Trace how a feature flows through the codebase. "
        "Given a description like 'user submits an answer', finds entry points "
        "and follows the call chain to show all participating code. "
        "Great for understanding end-to-end flows and Gherkin scenario traces. "
        "Defaults to the current repository; include_other_repos=True broadens traversal."
    ),
)
@traced("trace_feature")
def trace_feature(
    feature: str,
    *,
    ctx: Context,
    max_paths: int = 5,
    max_depth: int = 6,
    include_other_repos: bool = False,
) -> dict:
    """Walk the call graph from feature-matching entry points."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return _graph_repo_unavailable() | {"feature": feature}
    return trace_feature_impl(
        feature,
        app,
        max_paths=max_paths,
        max_depth=max_depth,
        repo_id=repo_id,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_trace_feature_fn=graph_trace_feature,
    )


@mcp.tool(
    title="Report bad result",
    description=(
        "Flag a previous retrieval as bad — Strong-negative T6 signal for "
        "the Meta-Harness outcome labeler (WIN-162). Use this when a result "
        "from search_code / smart_search_tool / get_symbol / find_* was clearly "
        "wrong: irrelevant, the wrong symbol, stale, a duplicate, too verbose, "
        "or other. Pass the `trace_id` from the offending response — search_code "
        "and smart_search_tool return it as a top-level `trace_id` field (WIN-160); "
        "it matches the JSONL trace record so the labeler resolves the chunks."
    ),
)
@traced("report_bad_result")
def report_bad_result(
    trace_id: str,
    reason: str,
    *,
    ctx: Context,
    detail: str | None = None,
) -> dict:
    """Persist a bad-result report for the Meta-Harness labeler.

    Args:
        trace_id: the trace_id of the offending tool call (from the trace stream).
        reason: one of "irrelevant", "wrong_symbol", "stale", "duplicate",
            "too_verbose", or "other".
        detail: optional free-text explanation.

    Returns ``{"ok": True, "trace_id": ..., "reason": ...}`` on success
    and ``{"ok": False, "error": ...}`` when the report could not be
    persisted (e.g. unknown reason, store disabled).
    """
    from .meta.feedback import FeedbackError

    app = _ctx(ctx)
    store = app.bad_result_store
    if store is None:
        return {
            "ok": False,
            "error": "bad_result_store unavailable on this server "
            "(check mcp.bad_result_store.disabled in logs)",
        }
    try:
        record = store.report(  # type: ignore[attr-defined]
            trace_id=trace_id,
            reason=reason,
            detail=detail,
            session_id=app.session_id or None,
        )
    except FeedbackError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "trace_id": record.trace_id,
        "reason": record.reason,
        "ts": record.ts,
    }


# ---- assemble_context (WIN-209) --------------------------------------------


class AssembleSectionResponse(BaseModel):
    """One section of an ``assemble_context`` response."""

    key: str = Field(description="Section identifier: primary, callers, tests.")
    title: str = Field(description="Human-readable section header.")
    hits: list[SearchResult] = Field(default_factory=list)


class AssembleContextResponse(BaseModel):
    """One-shot context bundle ready for the agent to paste."""

    task: str
    trace_id: str | None = Field(
        default=None,
        description=(
            "Trace id for this call (WIN-160). Pass it to report_bad_result to "
            "flag a weak/duplicate bundle — auto-stamped by @traced; matches the "
            "JSONL trace record."
        ),
    )
    bundle: str = Field(
        description=(
            "Markdown-formatted context bundle: file:line:symbol headers + "
            "fenced code blocks per span, grouped by provenance "
            "(primary / callers / tests). Ready to paste into a follow-up "
            "reasoning step."
        ),
    )
    sections: list[AssembleSectionResponse] = Field(default_factory=list)
    total_tokens: int = Field(
        description="Estimated tokens consumed by the packed snippets.",
    )
    budget_tokens: int = Field(
        description="The token budget the caller passed.",
    )
    spilled_count: int = Field(
        default=0,
        description=(
            "Number of relevant hits that fell off the tail because the "
            "budget was tight. 0 means everything fit."
        ),
    )
    truncated_oversize: bool = Field(
        default=False,
        description=(
            "True when the single top hit exceeds the budget and was "
            "included anyway as an escape hatch."
        ),
    )


@mcp.tool(
    title="Assemble context bundle",
    description=(
        "ONE-SHOT context retrieval (WIN-209). Given a natural-language "
        "task, runs hybrid search + optional graph extensions (callers, "
        "tests), greedy-packs the results under context_budget tokens, "
        "and returns a ready-to-paste markdown bundle. Use this when "
        "you would otherwise chain search_code → find_callers → "
        "find_tests_for — assemble_context collapses those turns into "
        "a single call. Returned ``bundle`` is markdown with file:line "
        "headers + fenced code blocks; ``sections`` is the structured "
        "view if you'd rather work in JSON. Set diff_scope=True for "
        "PR-review workflows: results are filtered to chunks in the "
        "current git diff (WIN-211). Defaults to the current repository; pass repo "
        "to select an exact indexed absolute local path or normalized key; "
        "include_other_repos=True explicitly broadens every retrieval step."
    ),
)
@traced("assemble_context")
def assemble_context(
    task: str,
    context_budget: int = 4000,
    top_k: int = 10,
    include_callers: bool = False,
    include_tests: bool = False,
    include_rationale: bool = False,
    diff_scope: bool = False,
    diff_ref_a: str | None = None,
    diff_ref_b: str | None = None,
    include_other_repos: bool = False,
    *,
    repo: str | None = None,
    ctx: Context,
) -> AssembleContextResponse:
    """Run the WIN-209 assembly pipeline and return a packed bundle.

    Args:
        task: natural-language task description ("how does mastery
            decay work in FSRS").
        context_budget: total token budget across all sections. Must
            be positive. Defaults to 4000 — generous for most agent
            context windows; tighten when you need room for the
            agent's own reasoning afterward.
        top_k: candidate count fetched from the primary search before
            packing. Higher means more candidates to choose from but
            potentially more spilled at packing time.
        include_callers: when True, also pull up-the-graph callers
            for each primary hit. Costs graph queries; off by default.
        include_tests: when True, also pull tests-that-cover for each
            primary hit. Costs graph queries; off by default.
        include_rationale: WIN-219 — when True, also pull
            ``rationale_for`` graph edges (docs / ADRs that explain
            each primary symbol) and pack them into a "Rationale"
            section. Lets the agent see code + tests + docs in ONE
            deduplicated bundle instead of 3 disjoint payloads.
        diff_scope: WIN-211 — when True, filter the primary results
            to only chunks in files touched by the current git diff.
            Use for PR-review workflows where the global top-K is too
            noisy. Diff is computed against the configured target_repo.
        diff_ref_a: starting git ref. None = working tree diff.
        diff_ref_b: ending git ref. None + diff_ref_a set = HEAD.
        include_other_repos: Search every indexed repository. False scopes
            primary retrieval and graph extensions to the current repository.
        repo: Optional indexed absolute local path or normalized repository key.
    """
    from .retrieval.assemble import assemble_context as _assemble

    app = _ctx(ctx)
    if context_budget <= 0:
        # Pydantic field can't enforce min in MCP schema reliably; do
        # the check in code so the agent gets a clean error message.
        return AssembleContextResponse(
            task=task,
            bundle="",
            sections=[],
            total_tokens=0,
            budget_tokens=0,
        )
    repo_selector = (repo or "").strip() or None
    if repo_selector and include_other_repos:
        return AssembleContextResponse(
            task=task,
            bundle="",
            sections=[],
            total_tokens=0,
            budget_tokens=context_budget,
        )
    selected_repo_path = None
    if include_other_repos:
        repo_id = None
    elif repo_selector:
        from .server_memory_tools import _resolve_repo

        repo_id, selected_repo_path = _resolve_repo(app, repo_selector, log=log)
    else:
        repo_id = _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return AssembleContextResponse(
            task=task,
            bundle="",
            sections=[],
            total_tokens=0,
            budget_tokens=context_budget,
        )
    # When diff_scope is requested, the target repo for the diff is
    # the configured ``target_repo`` so the agent doesn't have to
    # remember the path. The library function falls back gracefully
    # when the repo is missing or git fails.
    diff_repo = (selected_repo_path or app.config.target_repo) if diff_scope else None
    result = _assemble(
        task,
        search_engine=app.search_engine,
        context_budget=context_budget,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        include_rationale=include_rationale,
        storage=app.storage,
        diff_scope=diff_scope,
        diff_repo=diff_repo,
        diff_ref_a=diff_ref_a,
        diff_ref_b=diff_ref_b,
        repo_id=repo_id,
    )
    return AssembleContextResponse(
        task=result.task,
        bundle=result.bundle,
        sections=[
            AssembleSectionResponse(
                key=sec.key,
                title=sec.title,
                hits=[_to_result(h) for h in sec.hits],
            )
            for sec in result.sections
        ],
        total_tokens=result.total_tokens,
        budget_tokens=result.budget_tokens,
        spilled_count=result.spilled_count,
        truncated_oversize=result.truncated_oversize,
    )


# ---- execute investigation DAG (WIN-233, EPIC 1.10 #24) --------------------


@mcp.tool(
    title="Execute investigation route server-side",
    description=(
        "WIN-233 — build an investigation plan and EXECUTE it "
        "server-side, returning the distilled bundle in one call. "
        "Where ``plan_investigation`` returns a recipe the agent "
        "walks one step at a time, ``execute_investigation`` saves "
        "the round-trips: stropha runs search → callers → tests → "
        "rationale internally, packs the union under a budget, and "
        "returns the same ``AssembleContextResponse`` shape as "
        "``assemble_context``. Use for 'tell me about X' tasks "
        "where you'd otherwise chain N MCP calls. Defaults to the current "
        "repository; include_other_repos=True broadens both planner and DAG."
    ),
)
@traced("execute_investigation")
def execute_investigation(
    task: str,
    context_budget: int = 4000,
    top_k: int = 5,
    include_callers: bool = True,
    include_tests: bool = True,
    include_rationale: bool = True,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> AssembleContextResponse:
    """Plan + execute an investigation in one server-side call.

    Args:
        task: Natural-language investigation target.
        context_budget: Total token budget for the consolidated bundle.
        top_k: Maximum primary symbols used by the planner.
        include_callers: Include caller traversal in the plan and DAG.
        include_tests: Include test traversal in the plan and DAG.
        include_rationale: Include rationale traversal when available.
        include_other_repos: Search all indexed repositories. False scopes
            both planner and DAG execution to the current repository.
    """
    from .retrieval.dag import execute_plan
    from .retrieval.investigation import plan_investigation as _plan

    app = _ctx(ctx)
    if context_budget <= 0:
        return AssembleContextResponse(
            task=task,
            bundle="",
            sections=[],
            total_tokens=0,
            budget_tokens=0,
        )
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return AssembleContextResponse(
            task=task,
            bundle="",
            sections=[],
            total_tokens=0,
            budget_tokens=context_budget,
        )
    plan = _plan(
        task,
        search_engine=app.search_engine,
        storage=app.storage,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        include_rationale=include_rationale,
        repo_id=repo_id,
    )
    result = execute_plan(
        plan,
        search_engine=app.search_engine,
        storage=app.storage,
        context_budget=context_budget,
        repo_id=repo_id,
    )
    return AssembleContextResponse(
        task=result.task,
        bundle=result.bundle,
        sections=[
            AssembleSectionResponse(
                key=sec.key,
                title=sec.title,
                hits=[_to_result(h) for h in sec.hits],
            )
            for sec in result.sections
        ],
        total_tokens=result.total_tokens,
        budget_tokens=result.budget_tokens,
        spilled_count=result.spilled_count,
        truncated_oversize=result.truncated_oversize,
    )


def _pipeline_snapshot(app: AppContext) -> dict:
    """Introspect every built pipeline stage into a JSON-safe dict.

    Returns a structure shaped for the calling LLM agent's convenience:
    each stage shows what adapter is active, its ``adapter_id`` (the
    drift-detection key — see the CLAUDE.md **Drift detection** invariant), and any sub-adapters
    (retrieval streams, reranker). Caller can use this to answer
    questions like "is the cross-encoder reranker on?", "which embedder
    model is producing vectors right now?", "what graph confidence
    filter is the retrieval using?" — all without reading yaml/env.
    """
    built = app.built  # BuiltStages
    cfg = app.config

    def _stage(stage_obj):
        if stage_obj is None:
            return None
        return {
            "adapter": getattr(stage_obj, "adapter_name", "?"),
            "adapter_id": getattr(stage_obj, "adapter_id", "?"),
        }

    # Retrieval has nested sub-adapters (streams + reranker) — expose them.
    retrieval_info = _stage(built.retrieval)
    if built.retrieval is not None:
        retrieval_config = getattr(built.retrieval, "_config", None)
        if retrieval_config is not None and hasattr(retrieval_config, "model_dump"):
            retrieval_info["config"] = retrieval_config.model_dump(mode="json")  # type: ignore[index]
        streams_info: dict[str, dict] = {}
        for name, stream in getattr(built.retrieval, "_streams", {}).items():
            streams_info[name] = {
                "adapter": getattr(stream, "adapter_name", "?"),
                "adapter_id": getattr(stream, "adapter_id", "?"),
            }
        retrieval_info["streams"] = streams_info  # type: ignore[index]
        reranker = getattr(built.retrieval, "_reranker", None)
        if reranker is not None:
            retrieval_info["reranker"] = {  # type: ignore[index]
                "adapter": getattr(reranker, "adapter_name", "?"),
                "adapter_id": getattr(reranker, "adapter_id", "?"),
                "active": True,
            }
        else:
            retrieval_info["reranker"] = {  # type: ignore[index]
                "adapter": "noop",
                "adapter_id": "reranker:noop",
                "active": False,
            }

    return {
        "walker": _stage(built.walker),
        "chunker": _stage(built.chunker),
        "enricher": _stage(built.enricher),
        "embedder": {
            **(_stage(built.embedder) or {}),
            "model": getattr(built.embedder, "model_name", "?"),
            "dim": getattr(built.embedder, "dim", 0),
        },
        "secondary_embedder": (
            {
                **(_stage(built.secondary_embedder) or {}),
                "model": getattr(built.secondary_embedder, "model_name", "?"),
                "dim": getattr(built.secondary_embedder, "dim", 0),
            }
            if built.secondary_embedder is not None
            else None
        ),
        "storage": {
            **(_stage(built.storage) or {}),
            # Report the ACTIVE backend's real target — the Postgres DSN
            # (password-redacted) when postgres is the storage adapter, the
            # SQLite file path otherwise. Falls back to the configured index
            # path for adapters that don't expose ``storage_target``.
            "path": (
                getattr(built.storage, "storage_target", None) or str(cfg.resolve_index_path())
            ),
        },
        "retrieval": retrieval_info,
    }


def _inference_snapshot(cfg: Config) -> dict:
    """Active LLM inference backend (used by enricher / HyDE / query rewrite)."""
    backend = (cfg.inference.backend or "auto").lower()
    base: dict = {
        "backend": backend,
        "enricher_concurrency": cfg.inference.enricher_concurrency,
    }
    if backend == "lms":
        base.update(
            {
                "model": cfg.inference.lms_model,
                "endpoint": cfg.inference.lms_base_url,
                "api_key_set": cfg.secrets.has_lms,
                "timeout_s": cfg.inference.lms_timeout,
                "max_retries": cfg.inference.lms_max_retries,
                "think": cfg.inference.lms_think,
                "max_connections": cfg.inference.lms_max_connections,
                "max_keepalive_connections": (
                    cfg.inference.lms_max_keepalive_connections
                ),
                "location": "internal (OpenAI-compatible LMS)",
            }
        )
    elif backend == "moa":
        base.update(
            {
                "model": cfg.inference.moa_model,
                "endpoint": cfg.inference.moa_base_url,
                "api_key_set": cfg.secrets.has_moa,
                "timeout_s": cfg.inference.moa_timeout,
                "max_retries": cfg.inference.moa_max_retries,
                "location": "local (MoA Gateway)",
            }
        )
    elif backend == "deepseek":
        base.update(
            {
                "model": cfg.inference.deepseek_model,
                "endpoint": cfg.inference.deepseek_base_url,
                "api_key_set": cfg.secrets.has_deepseek,
                "timeout_s": cfg.inference.deepseek_timeout,
                "max_retries": cfg.inference.deepseek_max_retries,
                "location": "cloud (China-routed)",
            }
        )
    elif backend == "anthropic":
        base.update(
            {
                "model": cfg.inference.anthropic_model,
                "endpoint": cfg.inference.anthropic_base_url,
                "api_key_set": cfg.secrets.has_anthropic,
                "timeout_s": cfg.inference.anthropic_timeout,
                "max_retries": cfg.inference.anthropic_max_retries,
                "anthropic_version": cfg.inference.anthropic_version,
                "location": "cloud (Anthropic Messages API)",
            }
        )
    elif backend == "mlx":
        base.update(
            {
                "model": cfg.inference.mlx_model,
                "location": "local (Apple Silicon GPU)",
            }
        )
    elif backend == "ollama":
        base.update(
            {
                "model": cfg.inference.ollama_model,
                "endpoint": cfg.inference.ollama_host,
                "location": "local (Ollama daemon)",
            }
        )
    return base


def _graph_snapshot(app: AppContext) -> dict:
    """graphify symbol-mirror status: loaded? size? embedded?"""
    storage = app.storage
    cfg = app.config
    summary: dict = {
        "confidence_filter": cfg.graph.confidence,
        "fts_augment": cfg.graph.fts_augment,
        "graphify_out_path": (
            str(cfg.graph.graphify_out)
            if cfg.graph.graphify_out
            else f"{cfg.target_repo}/graphify-out"
        ),
    }
    # Pull graph stats directly from Storage. May be absent on a fresh DB.
    try:
        stats = storage.stats()
        graph_stats = stats.get("graph") if isinstance(stats, dict) else None
        if graph_stats:
            summary["loaded"] = True
            summary["node_count"] = graph_stats.get("nodes", 0)
            # ``_graph_stats``/``GraphifyLoader.stats`` expose the edge count as
            # ``edges_total`` (``edges`` was a stale key that silently read 0).
            summary["edge_count"] = graph_stats.get("edges_total", 0)
            summary["last_loaded_at"] = graph_stats.get("last_loaded_at")
            summary["embedding_model"] = graph_stats.get("embedding_model")
            summary["embedding_dim"] = graph_stats.get("embedding_dim")
        else:
            summary["loaded"] = False
            summary["hint"] = "Run `graphify .` in the target repo to populate."
    except Exception as exc:  # pragma: no cover  defensive
        summary["loaded"] = False
        summary["error"] = str(exc)
    return summary


def _features_snapshot(app: AppContext) -> dict:
    """Every query-time toggle from retrieval_features, with on/off summary."""
    cfg = app.config
    rf = cfg.retrieval_features
    retrieval_config = getattr(app.built.retrieval, "_config", None)

    def _retrieval_value(name: str, fallback: Any) -> Any:
        return getattr(retrieval_config, name, fallback)

    return {
        "hyde": {
            "enabled": rf.hyde_enabled or _retrieval_value("hyde_enabled", False),
            "code_tuned": rf.hyde_code_tuned,
            "purpose": "Generate hypothetical code via LLM, embed THAT for dense search.",
        },
        "query_decomposition": {
            "configured": rf.query_decomposition_enabled,
            "integrated": False,
            "uses_llm": rf.query_decomposition_llm,
            "purpose": "Standalone/legacy implementation; production RetrievalStage does not call it.",
        },
        "query_router_llm": {
            "enabled": rf.query_router_llm,
            "purpose": "LLM classifies query intent and routes to find_callers / find_tests_for / etc.",
        },
        "query_rewrite": {
            "enabled": rf.query_rewrite_enabled or _retrieval_value("query_rewrite_enabled", False),
            "max_terms": _retrieval_value("query_rewrite_max_terms", None)
            or rf.query_rewrite_max_terms,
            "policy": "short natural-language queries only; exact and structural queries bypass it",
            "purpose": "Selectively expand vague natural language into code-style terms.",
        },
        "multi_query": {
            "enabled": rf.multi_query_enabled or _retrieval_value("multi_query_enabled", False),
            "configured_count": _retrieval_value("multi_query_count", rf.multi_query_count),
            "purpose": "Generate paraphrases and fuse results; adapter-local multi_query_count is authoritative.",
        },
        "recursive_retrieval": {
            "enabled": _retrieval_value("recursive_enabled", rf.recursive_enabled),
            "adjacency": _retrieval_value("recursive_adjacency", rf.recursive_adjacency),
            "purpose": "Auto-merge adjacent chunks + promote parent when siblings hit.",
        },
        "diversity_packing": {
            "enabled": rf.diversity_enabled,
            "max_chunks_per_file": rf.diversity_max_chunks_per_file,
            "purpose": "Prefer distinct file/parent-symbol evidence within result budgets.",
        },
        "query_cache": {
            "configured": rf.cache_enabled,
            "integrated": False,
            "size": rf.cache_size,
            "ttl_s": rf.cache_ttl,
            "purpose": "Standalone cache implementation; production retrieval does not call it.",
        },
        "adaptive_weighting": {
            "enabled": _retrieval_value("adaptive_weighting", rf.adaptive_weighting),
            "purpose": "Symbol-shaped queries → boost lexical streams; natural → boost dense.",
        },
    }


def _available_adapters() -> dict[str, list[str]]:
    """Enumerate every adapter registered with the framework.

    Lets the calling agent know which alternatives could be activated
    without having to call a separate ``list_adapters`` endpoint.
    Sorted alphabetically for deterministic output.
    """
    from .pipeline.registry import all_adapters

    reg = all_adapters()
    # all_adapters() returns dict[str, list[str]] already.
    return {stage: sorted(names) for stage, names in reg.items()}


def _features_summary(app: AppContext) -> dict:
    """Cheat-sheet for the calling LLM: one-glance status of every major knob."""
    cfg = app.config
    rf = cfg.retrieval_features
    built = app.built
    features = _features_snapshot(app)

    # Reranker is "active" if a non-noop adapter was built.
    reranker = getattr(built.retrieval, "_reranker", None) if built.retrieval else None
    reranker_active = reranker is not None and getattr(reranker, "adapter_name", "noop") != "noop"

    # Enricher uses an LLM iff it's contextual / ollama / mlx.
    enricher_uses_llm = getattr(built.enricher, "adapter_name", "noop") in {
        "contextual",
        "ollama",
        "mlx",
    }

    # Build a "active LLM features" list for quick scan.
    llm_active: list[str] = []
    if enricher_uses_llm:
        llm_active.append(f"enricher({built.enricher.adapter_name})")
    if features["hyde"]["enabled"]:
        llm_active.append("hyde")
    if features["query_rewrite"]["enabled"]:
        llm_active.append("query_rewrite")
    if rf.query_router_llm:
        llm_active.append("query_router")
    if features["multi_query"]["enabled"]:
        llm_active.append(f"multi_query×{features['multi_query']['configured_count']}")
    if reranker_active and getattr(reranker, "adapter_name", "") == "llm":
        llm_active.append("reranker(llm)")

    return {
        "embedder": f"{built.embedder.adapter_name}:{built.embedder.model_name}@{built.embedder.dim}d",
        "enricher": built.enricher.adapter_name,
        "enricher_uses_llm": enricher_uses_llm,
        "inference_backend": (cfg.inference.backend or "auto"),
        "rerank_active": reranker_active,
        "rerank_model": (getattr(reranker, "adapter_id", None) if reranker_active else None),
        "llm_features_active_at_query_time": llm_active,
        "mcp_auto_watch": cfg.mcp_watch,
        "graph_traversal_available": _graph_snapshot(app).get("loaded", False),
    }


@mcp.tool(
    name="get_config",
    description=(
        "Comprehensive runtime configuration snapshot for the active stropha "
        "MCP server. Returns:\n"
        "- 'features_summary': one-glance cheat sheet (active embedder, "
        "  reranker on/off, LLM features at query time, graph status).\n"
        "- 'pipeline': every built stage (walker, chunker, enricher, embedder, "
        "  storage, retrieval) with its adapter + adapter_id, plus retrieval "
        "  sub-streams and the active reranker.\n"
        "- 'inference': active LLM backend with model, endpoint, timeout, "
        "  retry policy, and enricher concurrency.\n"
        "- 'retrieval_features': every query-time toggle (hyde, recursive, "
        "  cache, multi_query, etc) with enabled flag and one-line purpose.\n"
        "- 'graph': graphify mirror state (loaded? node/edge counts? embedding model?).\n"
        "- 'available_adapters': enumeration of every registered adapter per "
        "  stage so the caller knows which alternatives could be activated.\n"
        "- Plus the legacy env-var snapshot (sources, env_values, env_groups) "
        "  for back-compat with older callers.\n"
        "\n"
        "All paths are stringified, secrets are redacted as '(set)'. Use this "
        "to verify which DB the MCP is reading, which embedder is producing "
        "vectors, which features cost LLM calls per query, and what could be "
        "enabled to improve retrieval quality."
    ),
)
@traced("get_config")
def get_config_tool(*, ctx: Context) -> dict:
    """Return the active configuration for debugging.

    Backwards-compatible: the legacy ``config`` / ``settings`` / ``sources`` /
    ``env_values`` / ``env_groups`` / ``runtime`` / ``embedder`` /
    ``index_stats`` keys are preserved. New keys are additive.
    """
    app = _ctx(ctx)
    stats = app.storage.stats()

    # Legacy fields (preserved for back-compat).
    info = get_config_info()
    info["embedder"] = {
        "model": app.embedder.model_name,
        "dim": app.embedder.dim,
    }
    info["index_stats"] = {
        "total_chunks": stats.get("chunks", 0),
        "total_files": stats.get("files", 0),
        "repos": [
            {"name": r.get("normalized_key"), "chunks": r.get("chunks")}
            for r in stats.get("repos", [])
        ],
    }

    # Comprehensive runtime snapshot for the agent.
    info["features_summary"] = _features_summary(app)
    info["pipeline"] = _pipeline_snapshot(app)
    info["inference"] = _inference_snapshot(app.config)
    info["retrieval_features"] = _features_snapshot(app)
    info["graph"] = _graph_snapshot(app)
    info["available_adapters"] = _available_adapters()

    return info


@mcp.tool(
    name="index_health",
    description=(
        "Index self-check (WIN-348): per-repo coverage% (indexed files / git-"
        "tracked files), last-index lag in seconds, commits behind HEAD, and "
        "config drift (active embedder/enricher vs what last wrote the index). "
        "Use this to decide HOW MUCH TO TRUST the index before relying on "
        "search results — e.g. if a repo is many commits behind HEAD or the "
        "embedder config has drifted, results may be stale and a `stropha "
        "index` run is due. Git-derived fields are null when the repo is not a "
        "git tree or git is unavailable."
    ),
)
@traced("index_health")
def index_health_tool(*, ctx: Context) -> dict:
    """Return freshness / coverage / drift signals for every indexed repo."""
    from .server_health_tools import index_health_impl

    return index_health_impl(_ctx(ctx))


@mcp.resource("stropha://stats")
def index_stats_resource() -> StatsPayload:
    """Index statistics. Exposed as both a Tool and a Resource for discovery."""
    # FastMCP resources cannot use the lifespan context directly today, so we
    # re-derive what we need by routing through the adapter framework (same
    # path as the lifespan). Read-only and called rarely, so the per-call
    # rebuild is acceptable.
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    with built.storage as storage:
        info = storage.stats()
    return StatsPayload(**info)


# ---- suggest_next_tool (WIN-238) ------------------------------------------


class SuggestionEntry(BaseModel):
    """A single suggested next-tool invocation.

    ``args`` is the keyword payload the agent should pass to the
    target tool. ``confidence`` is a soft score in ``[0, 1]`` — higher
    means stropha is more sure this is the right next move.
    """

    tool: str = Field(description="Name of the MCP tool to call next.")
    args: dict = Field(
        default_factory=dict,
        description="Keyword arguments ready to forward to the target tool.",
    )
    reason: str = Field(description="Why this tool was suggested.")
    confidence: float = Field(
        description="Soft score in [0, 1]; higher means more certain.",
    )


class SuggestNextResponse(BaseModel):
    """Response from :func:`suggest_next_tool_mcp`."""

    query: str = Field(default="", description="Echo of the input query.")
    suggestions: list[SuggestionEntry] = Field(
        default_factory=list,
        description="Suggestions sorted by confidence DESC.",
    )
    based_on_hits: int = Field(
        description=(
            "How many recent hits the heuristics inspected. 0 means "
            "the response is based purely on the query shape (e.g. "
            "the agent hasn't searched yet)."
        ),
    )
    graph_loaded: bool = Field(
        description=(
            "Whether the graphify mirror is present. False suppresses "
            "graph-only suggestions like find_callers."
        ),
    )


@mcp.tool(
    title="Suggest the next MCP tool to call",
    description=(
        "WIN-238 — meta-guidance for the agent's loop. Given the most "
        "recent query + chunk_ids, return a ranked list of next-tool "
        "actions ('find_callers on FooClass', 'get_file_outline on "
        "x.py', etc.) with confidence scores. Pure heuristics, no LLM "
        "call. When chunk_ids is omitted, uses the session's seen "
        "chunk_ids (last 10). Never raises — empty list when no "
        "actionable signal. Defaults to the current repository; "
        "include_other_repos=True broadens chunk and graph inspection."
    ),
)
@traced("suggest_next_tool")
def suggest_next_tool_mcp(
    query: str = "",
    chunk_ids: list[str] | None = None,
    max_suggestions: int = 5,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> SuggestNextResponse:
    """Return ranked next-tool suggestions for the agent."""
    from .retrieval.suggest_next import suggest_next_tool as _suggest

    app = _ctx(ctx)
    max_suggestions = max(1, min(int(max_suggestions), 20))
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return SuggestNextResponse(
            query=query,
            suggestions=[],
            based_on_hits=0,
            graph_loaded=False,
        )

    # Resolve which chunks to inspect: explicit list wins; otherwise
    # fall back to the most recent ones the agent has seen.
    ids = list(chunk_ids) if chunk_ids else list(app.seen_chunk_ids)[-10:]

    hits: list = []
    for cid in ids:
        try:
            hit = app.storage.lookup_by_chunk_id(cid)
        except Exception:  # pragma: no cover - defensive
            hit = None
        if hit is not None and chunk_belongs_to_repo(app.storage, cid, repo_id):
            hits.append(hit)

    has_graph = graph_loaded(app.storage, repo_id=repo_id)
    has_rationale = has_rationale_edges(app.storage, repo_id=repo_id) if has_graph else False

    raw = _suggest(
        query,
        hits,
        has_graph=has_graph,
        has_rationale=has_rationale,
        max_suggestions=max_suggestions,
    )

    return SuggestNextResponse(
        query=query,
        suggestions=[
            SuggestionEntry(
                tool=s.tool,
                args=(dict(s.args) | {"include_other_repos": True})
                if include_other_repos
                else dict(s.args),
                reason=s.reason,
                confidence=s.confidence,
            )
            for s in raw
        ],
        based_on_hits=len(hits),
        graph_loaded=has_graph,
    )


# ---- named task recipes (WIN-240) -----------------------------------------


class RecipeStepEntry(BaseModel):
    """One materialised step of a recipe.

    ``args`` are placeholder-substituted and ready to forward to the
    named tool. ``budget_tokens`` is derived from the step's
    declared ``budget_share`` × the recipe's effective budget.
    """

    tool: str = Field(description="Name of the MCP tool to call.")
    args: dict = Field(
        default_factory=dict,
        description="Ready-to-forward keyword arguments.",
    )
    reason: str = Field(description="Why this step is in the recipe.")
    budget_tokens: int = Field(
        description="Suggested token allowance for this step.",
    )


class RecipeSummary(BaseModel):
    """Lightweight recipe summary returned by ``list_task_recipes``."""

    name: str
    description: str
    placeholders: list[str]
    step_count: int
    default_budget: int


class ListRecipesResponse(BaseModel):
    recipes: list[RecipeSummary]


class ApplyRecipeResponse(BaseModel):
    name: str = Field(description="Recipe name (echoes the request).")
    found: bool = Field(
        description="False when ``name`` is not a registered recipe.",
    )
    steps: list[RecipeStepEntry] = Field(
        default_factory=list,
        description="Materialised steps with placeholders resolved.",
    )
    missing_placeholders: list[str] = Field(
        default_factory=list,
        description=(
            "Placeholders the recipe references but the caller didn't "
            "supply. Empty when every slot was filled."
        ),
    )
    budget_tokens: int = Field(
        default=0,
        description="Effective budget used (recipe default or override).",
    )


@mcp.tool(
    title="List named task recipes",
    description=(
        "WIN-240 — enumerate the built-in workflow templates. Each "
        "recipe captures a common task pattern (bug-fix, add-feature, "
        "review, rename-symbol, understand-file) as an ordered "
        "sequence of stropha tool invocations with placeholder args. "
        "Pair with apply_task_recipe to materialise concrete steps."
    ),
)
@traced("list_task_recipes")
def list_task_recipes(*, ctx: Context) -> ListRecipesResponse:
    """Return every registered recipe with metadata only (no steps)."""
    from .retrieval.recipes import list_recipes

    _ = _ctx(ctx)  # signature parity with the rest of the surface
    return ListRecipesResponse(
        recipes=[
            RecipeSummary(
                name=r.name,
                description=r.description,
                placeholders=list(r.placeholders),
                step_count=len(r.steps),
                default_budget=r.default_budget,
            )
            for r in list_recipes()
        ],
    )


@mcp.tool(
    title="Apply a named task recipe",
    description=(
        "WIN-240 — materialise a recipe with the caller's context. "
        "Returns a list of concrete tool invocations (with args + "
        "reason + budget) ready to execute one-by-one or hand to "
        "execute_plan for server-side DAG execution. Missing "
        "placeholders are surfaced in ``missing_placeholders`` "
        "rather than failing the call — caller decides whether to "
        "fill them and retry."
    ),
)
@traced("apply_task_recipe")
def apply_task_recipe(
    name: str,
    context: dict | None = None,
    budget_tokens: int | None = None,
    *,
    ctx: Context,
) -> ApplyRecipeResponse:
    """Look up a recipe and substitute placeholders against ``context``."""
    from .retrieval.recipes import (
        apply_recipe,
        get_recipe,
        required_placeholders,
    )

    _ = _ctx(ctx)
    recipe = get_recipe(name)
    if recipe is None:
        return ApplyRecipeResponse(
            name=name,
            found=False,
            steps=[],
            missing_placeholders=[],
            budget_tokens=0,
        )

    context = dict(context or {})
    missing = sorted(required_placeholders(recipe) - set(context.keys()))

    invocations = apply_recipe(
        recipe,
        context,
        budget_tokens=budget_tokens,
        strict=False,
    )
    effective_budget = (
        recipe.default_budget if budget_tokens is None else max(0, int(budget_tokens))
    )

    return ApplyRecipeResponse(
        name=recipe.name,
        found=True,
        steps=[
            RecipeStepEntry(
                tool=inv.tool,
                args=inv.args,
                reason=inv.reason,
                budget_tokens=inv.budget_tokens,
            )
            for inv in invocations
        ],
        missing_placeholders=missing,
        budget_tokens=effective_budget,
    )


@mcp.tool(
    title="Verify a claim against the graph",
    description=(
        "WIN-244 — sanity-check natural-language assertions like "
        "'FooClass calls BarService' against the graphify mirror. "
        "Supported predicates: calls, called by, uses / references, "
        "implements, is a <kind>, is in <file>, has tests. Returns "
        "tri-state verdict (True / False / null) with evidence and, "
        "when False, the actual edges so the agent can correct its "
        "model without re-querying. Defaults to the current repository; "
        "include_other_repos=True broadens graph resolution. Never raises."
    ),
)
@traced("verify_claim")
def verify_claim_tool(
    claim: str,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> VerdictResponse:
    """Parse + verify a claim against the graph."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return VerdictResponse(
            raw_text=claim,
            parsed=False,
            verified=None,
            reason="Current repository could not be resolved; claim was not verified.",
        )
    return verify_claim_impl(claim, app, graph_loaded_fn=graph_loaded, repo_id=repo_id)


@mcp.tool(
    title="Guard against hallucinated symbols",
    description=(
        "WIN-245 — vet a batch of symbol names against the index "
        "BEFORE the agent commits them to code. Each symbol gets a "
        "verdict (exists/missing) plus near-miss suggestions on "
        "misses, so the agent can pick the correct name without a "
        "second search. Defensive: callable exceptions become notes, "
        "never propagate; empty / duplicate names skipped. Defaults to "
        "the current repository; include_other_repos=True broadens lookup."
    ),
)
@traced("check_symbols")
def check_symbols_tool(
    symbols: list[str],
    max_suggestions: int = 5,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> HallucinationGuardResponse:
    """Run the hallucination guard against the live index."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return HallucinationGuardResponse(
            checks=[],
            present_count=0,
            missing_count=0,
            notes=["Current repository could not be resolved; symbols were not checked."],
        )
    return check_symbols_impl(symbols, app, max_suggestions=max_suggestions, repo_id=repo_id)


@mcp.tool(
    title="Post-edit blast-radius report",
    description=(
        "WIN-246 — given the symbols you just edited, return every "
        "caller / test / related symbol / rationale node that may be "
        "affected, deduplicated across the input list. Use AFTER an "
        "edit to know what else needs verification. Composes the "
        "existing find_callers / find_tests_for / find_related / "
        "find_rationale graph readers under one call. Defaults to the current "
        "repository; include_other_repos=True broadens traversal."
    ),
)
@traced("blast_radius")
def blast_radius_tool(
    symbols: list[str],
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> BlastRadiusResponse:
    """Build a blast-radius report for the changed symbols."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return BlastRadiusResponse(
            changed_symbols=[symbol.strip() for symbol in symbols if symbol.strip()],
            graph_loaded=False,
            summary="Current repository could not be resolved; blast radius was not computed.",
            notes=["graph_loaded=False"],
        )
    return blast_radius_impl(
        symbols,
        app,
        graph_loaded_fn=graph_loaded,
        has_rationale_edges_fn=has_rationale_edges,
        repo_id=repo_id,
    )


@mcp.tool(
    title="Assert a cached snippet is still fresh",
    description=(
        "WIN-247 — vet a snippet the agent cached against the live "
        "index. Pass the chunk_id + the snippet text the agent "
        "remembers; the response tells you whether the chunk still "
        "exists and whether its content has drifted. Defensive — "
        "lookup exceptions surface as fresh=null with a reason. "
        "Use strict=true for byte-exact comparison; default is "
        "whitespace-normalised. Defaults to the current repository; "
        "include_other_repos=True permits any indexed chunk."
    ),
)
@traced("assert_snippet_freshness")
def assert_snippet_freshness_tool(
    chunk_id: str,
    expected_snippet: str,
    strict: bool = False,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> FreshnessResponse:
    """Verify a cached snippet against the live index."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return FreshnessResponse(
            chunk_id=chunk_id,
            fresh=None,
            reason="Current repository could not be resolved; freshness was not checked.",
            expected_snippet=expected_snippet,
        )
    return assert_snippet_freshness_impl(
        chunk_id,
        expected_snippet,
        app,
        strict=bool(strict),
        repo_id=repo_id,
    )


@mcp.tool(
    title="Test-impact analysis",
    description=(
        "WIN-248 — given the changed symbols, compute the minimal "
        "test set that should run. Direct tests via find_tests_for "
        "plus (when transitive_depth>=1) tests covering immediate "
        "callers. Returns a ready-to-paste run_command (pytest "
        "by default; pass runner='go test' / 'npm test' for other "
        "ecosystems). Defaults to the current repository; "
        "include_other_repos=True broadens traversal. Never raises."
    ),
)
@traced("test_impact")
def test_impact_tool(
    symbols: list[str],
    transitive_depth: int = 1,
    runner: str = "pytest",
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> TestImpactResponse:
    """Compute the impacted test set for the changed symbols."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return TestImpactResponse(
            changed_symbols=[symbol.strip() for symbol in symbols if symbol.strip()],
            graph_loaded=False,
            run_command="",
            notes=["graph_loaded=False", "Current repository could not be resolved."],
        )
    return test_impact_impl(
        symbols,
        app,
        transitive_depth=transitive_depth,
        runner=runner,
        graph_loaded_fn=graph_loaded,
        repo_id=repo_id,
    )


# ---- audit_citations (WIN-249) --------------------------------------------


@mcp.tool(
    title="Audit a response for verifiable citations",
    description=(
        "WIN-249 — vet a piece of agent output for citation hygiene. "
        "Splits the text into sentences, extracts [chunk_id] / "
        "[file:line] / [file:start-end] citation tokens, and "
        "resolves each against the index. Returns per-sentence "
        "verdicts plus aggregate counts (cited / uncited / broken). "
        "Use to prevent 'vibes' assertions from shipping. Defaults to "
        "the current repository; include_other_repos=True broadens validation."
    ),
)
@traced("audit_citations")
def audit_citations_tool(
    text: str,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> CitationAuditResponse:
    """Audit the agent's text for verifiable citations."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return CitationAuditResponse(
            claims=[],
            total_claims=0,
            cited_claims=0,
            uncited_claims=0,
            notes=["Current repository could not be resolved; citations were not audited."],
        )
    return audit_citations_impl(text, app, repo_id=repo_id)


# ---- spec_consistency (WIN-250) -------------------------------------------


def _default_spec_loader(repo_root: str) -> dict[str, str]:
    return server_default_spec_loader(repo_root)


@mcp.tool(
    title="Spec-consistency check",
    description=(
        "WIN-250 — cross-reference planned changes against documented "
        "invariants (CLAUDE.md, AGENTS.md, docs/architecture/*.md). "
        "Flags every spec sentence carrying MUST / NEVER / "
        "forward-only / etc. that mentions a changed symbol or file. "
        "Helps avoid silently violating an undocumented contract. "
        "Severity: HIGH (MUST/NEVER) / MEDIUM (should/avoid) / "
        "LOW (note). This tool is repository-local and reads specs only "
        "from the configured current target repository."
    ),
)
@traced("spec_consistency")
def spec_consistency_tool(
    changed_symbols: list[str],
    changed_files: list[str] | None = None,
    *,
    ctx: Context,
) -> SpecConsistencyResponse:
    """Check the changes against documented invariants."""
    app = _ctx(ctx)
    target_repo = app.config.target_repo
    if target_repo is None:
        return SpecConsistencyResponse(
            invariants_found=0,
            notes=["No current target repository is configured; specs were not scanned."],
        )
    return spec_consistency_impl(
        changed_symbols,
        changed_files,
        repo_root=str(target_repo),
        default_spec_loader_fn=_default_spec_loader,
    )


# ---- contract_tests (WIN-251) ----------------------------------------------


@mcp.tool(
    title="Surface contract tests for a symbol",
    description=(
        "WIN-251 — partition the tests for a symbol into contract "
        "tests (public-behaviour guarantees) and other tests "
        "(implementation-internal). Contract tests are flagged by "
        "any of three signals: path pattern (/contracts/, "
        "_spec.py, Spec.java, …), symbol name (contract, interface, "
        "api, behavior), or tag/annotation. The agent uses this to "
        "see WHICH tests must keep passing for the public surface "
        "to stay intact. Defaults to the current repository; "
        "include_other_repos=True broadens graph resolution."
    ),
)
@traced("contract_tests")
def contract_tests_tool(
    symbol: str,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> ContractTestsResponse:
    """Surface contract tests for ``symbol``."""
    app = _ctx(ctx)
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return ContractTestsResponse(
            target_symbol=symbol,
            graph_loaded=False,
            notes=["graph_loaded=False", "Current repository could not be resolved."],
        )
    return contract_tests_impl(symbol, app, graph_loaded_fn=graph_loaded, repo_id=repo_id)


# ---- working_set (WIN-253) -------------------------------------------------


class WorkingSetEntryModel(BaseModel):
    chunk_id: str
    path: str = ""
    symbol: str = ""
    kind: str = ""
    start_line: int = 0
    end_line: int = 0
    snippet: str = ""
    role: str = Field(description="'pinned' or 'recent'.")
    tokens: int = 0


class WorkingSetResponse(BaseModel):
    """Outcome of ``working_set_tool`` / ``working_set_resource``."""

    session_id: str
    entries: list[WorkingSetEntryModel] = Field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    pinned_count: int = 0
    recent_count: int = 0
    dropped_count: int = Field(
        default=0,
        description="Recent chunks dropped because the budget was tight.",
    )
    markdown: str = Field(
        default="",
        description="Rendered markdown bundle suitable for an LLM prompt.",
    )
    notes: list[str] = Field(default_factory=list)


@mcp.tool(
    title="Build the current working set",
    description=(
        "WIN-253 — assemble the session's working set (pinned + "
        "recent chunks) into a markdown bundle ready to plug into "
        "the next LLM prompt. Pinned chunks are always included; "
        "recent chunks are packed under budget_tokens. Defaults to "
        "the current session and current repository; pass session_id "
        "to query another session. include_other_repos=True broadens hydration."
    ),
)
@traced("working_set")
def working_set_tool(
    session_id: str = "",
    budget_tokens: int = 4000,
    max_recent: int = 20,
    include_other_repos: bool = False,
    *,
    ctx: Context,
) -> WorkingSetResponse:
    """Assemble the working set for the current (or given) session."""
    from .retrieval.working_set import build_working_set

    app = _ctx(ctx)
    sid = (session_id or "").strip() or app.session_id
    repo_id = None if include_other_repos else _resolve_target_repo_id(app)
    if repo_id is None and not include_other_repos:
        return WorkingSetResponse(
            session_id=sid,
            budget_tokens=max(0, int(budget_tokens)),
            notes=["Current repository could not be resolved; working set was not built."],
        )

    if session_id:
        # Pull pinned + seen from the sessions table via SessionRecord.
        pinned = []
        seen = []
        try:
            record = app.storage.get_session(session_id)
        except Exception:
            record = None
        if record is not None:
            pinned = list(record.pinned_chunk_ids)
            seen = list(record.seen_chunk_ids)
    else:
        pinned = list(app.pinned_chunk_ids)
        seen = list(app.seen_chunk_ids)

    def _lookup(cid: str):
        try:
            if chunk_belongs_to_repo(app.storage, cid, repo_id):
                return app.storage.lookup_by_chunk_id(cid)
            return None
        except Exception:
            return None

    ws = build_working_set(
        pinned,
        seen,
        lookup_fn=_lookup,
        session_id=sid,
        budget_tokens=max(0, int(budget_tokens)),
        max_recent=max(0, int(max_recent)),
    )

    return WorkingSetResponse(
        session_id=ws.session_id,
        entries=[
            WorkingSetEntryModel(
                chunk_id=e.chunk_id,
                path=e.path,
                symbol=e.symbol,
                kind=e.kind,
                start_line=e.start_line,
                end_line=e.end_line,
                snippet=e.snippet,
                role=e.role,
                tokens=e.tokens,
            )
            for e in ws.entries
        ],
        total_tokens=ws.total_tokens,
        budget_tokens=ws.budget_tokens,
        pinned_count=ws.pinned_count,
        recent_count=ws.recent_count,
        dropped_count=ws.dropped_count,
        markdown=ws.markdown,
        notes=list(ws.notes),
    )


@mcp.resource("stropha://working-set")
def working_set_resource() -> str:
    """Resource view of the most-recently-updated session's working
    set, rendered as markdown.

    FastMCP resources can't read the AppContext, so this opens its
    own Storage handle (same pattern as ``index_stats_resource``)
    and pulls the latest session row from the v8 ``sessions``
    table. Returns a plain markdown string suitable for the
    ``contents`` field of an MCP resource response.
    """
    from .retrieval.working_set import build_working_set

    cfg = get_config()
    target_repo = cfg.target_repo
    if target_repo is None:
        return "# Working set\n\n_current repository unavailable._\n"
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    with built.storage as storage:
        try:
            from .ingest.git_meta import detect

            identity = detect(target_repo)
            repo_id = storage.lookup_repo_id(identity.normalized_key)
        except Exception:
            repo_id = None
        if repo_id is None:
            return "# Working set\n\n_current repository unavailable._\n"
        try:
            record = storage.get_latest_session()
        except Exception:
            record = None
        if record is None:
            return "# Working set\n\n_no session state yet._\n"

        def _lookup(cid: str):
            try:
                if chunk_belongs_to_repo(storage, cid, repo_id):
                    return storage.lookup_by_chunk_id(cid)
                return None
            except Exception:
                return None

        ws = build_working_set(
            list(record.pinned_chunk_ids),
            list(record.seen_chunk_ids),
            lookup_fn=_lookup,
            session_id=record.session_id,
        )
    return ws.markdown


# ---- recipe prompts (WIN-254) ---------------------------------------------


def _render_recipe_prompt(recipe_name: str, context: dict) -> str:
    """Apply the named recipe + render a user-facing prompt body.

    Used by the per-recipe ``@mcp.prompt`` registrations. Each
    prompt fills in the recipe's placeholders from the caller's
    arguments and returns a single user-role message that lists
    the steps with their tool / args / reason / suggested budget.
    """
    from .retrieval.recipes import apply_recipe, get_recipe

    recipe = get_recipe(recipe_name)
    if recipe is None:
        return f"Unknown recipe {recipe_name!r}. List recipes via list_task_recipes."

    invocations = apply_recipe(recipe, context, strict=False)
    lines = [
        f"# Task recipe: **{recipe.name}**",
        "",
        recipe.description,
        "",
        "## Steps",
        "",
    ]
    for idx, inv in enumerate(invocations, start=1):
        args_str = ", ".join(f"{k}={v!r}" for k, v in inv.args.items())
        lines.append(f"{idx}. **{inv.tool}**({args_str})")
        lines.append(f"   - _why_: {inv.reason}")
        lines.append(f"   - _budget_: {inv.budget_tokens} tokens")
        lines.append("")
    lines.append("Run each step with the suggested args; surface")
    lines.append("``working_set_tool`` between steps for a fresh view.")
    return "\n".join(lines)


@mcp.prompt(
    name="bug-fix",
    title="Bug-fix workflow",
    description=(
        "Walk the canonical bug-fix recipe (find the failure site, "
        "map the file, gather tests + callers + rationale)."
    ),
)
def bug_fix_prompt(symptom: str, file: str, symbol: str) -> str:
    """WIN-254 — render the bug-fix recipe with the caller's args."""
    return _render_recipe_prompt(
        "bug-fix",
        {"symptom": symptom, "file": file, "symbol": symbol},
    )


@mcp.prompt(
    name="add-feature",
    title="Add-feature workflow",
    description=(
        "Walk the canonical add-feature recipe (bootstrap, find the "
        "integration point, sibling patterns, tests, rationale)."
    ),
)
def add_feature_prompt(feature_query: str, anchor_symbol: str) -> str:
    """WIN-254 — render the add-feature recipe."""
    return _render_recipe_prompt(
        "add-feature",
        {"feature_query": feature_query, "anchor_symbol": anchor_symbol},
    )


@mcp.prompt(
    name="review",
    title="Code review workflow",
    description=(
        "Walk the canonical review recipe (outline + callers + tests "
        "+ rationale of the changed file/symbol)."
    ),
)
def review_prompt(changed_file: str, changed_symbol: str) -> str:
    """WIN-254 — render the review recipe."""
    return _render_recipe_prompt(
        "review",
        {"changed_file": changed_file, "changed_symbol": changed_symbol},
    )


@mcp.prompt(
    name="rename-symbol",
    title="Rename-symbol workflow",
    description=(
        "Walk the canonical rename-symbol recipe (definition, "
        "callers, tests — the audit needed before renaming)."
    ),
)
def rename_symbol_prompt(symbol: str) -> str:
    """WIN-254 — render the rename-symbol recipe."""
    return _render_recipe_prompt("rename-symbol", {"symbol": symbol})


@mcp.prompt(
    name="understand-file",
    title="Understand-file workflow",
    description=(
        "Walk the canonical understand-file recipe (outline, "
        "references, related symbols, rationale)."
    ),
)
def understand_file_prompt(file: str, primary_symbol: str) -> str:
    """WIN-254 — render the understand-file recipe."""
    return _render_recipe_prompt(
        "understand-file",
        {"file": file, "primary_symbol": primary_symbol},
    )


# ---- entry point -----------------------------------------------------------

_DEFAULT_PORT = 9741


def main() -> None:
    """Run the MCP server over streamable-http.

    stropha-mcp is HTTP-only: a single shared singleton serves every MCP
    client concurrently (one SQLite connection, zero WAL contention).
    Use ``bin/stropha-mcp-ensure`` to start/stop the daemon; clients
    connect via the ``streamable-http`` transport at
    ``http://<host>:<port>/mcp``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="stropha-mcp",
        description="stropha RAG MCP server (streamable-http transport only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"Port to bind (default: {_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    try:
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
