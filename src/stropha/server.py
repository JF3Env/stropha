"""MCP server entry point.

Wires Config → Embedder → Storage → SearchEngine inside a FastMCP lifespan,
and exposes the Phase 1 tool surface from spec §9.2:

- `search_code`         — hybrid retrieval (dense + BM25 + RRF).
- `get_symbol`          — direct symbol lookup.
- `get_file_outline`    — symbolic outline of a single file.
- `list_repos`          — enumerate repositories present in the index.
- `index_stats` resource (stropha://stats).

Every search result carries a ``repo`` field (URL, default branch, HEAD)
so a remote MCP client can ``git clone`` the source. The server speaks
the MCP ``streamable-http`` transport at ``http://<host>:<port>/mcp``;
``bin/stropha-mcp-ensure`` is the canonical daemon launcher and MCP
clients connect to it as a ``remote`` server.

Background file watcher (enabled by default via ``STROPHA_MCP_WATCH=1``):
The server starts a background thread that polls the target repo for file
changes. After a debounce period (default 2s of no further changes), it
triggers a light re-index pass so MCP queries always see recent edits
without requiring a separate ``stropha watch`` command. Disable with
``STROPHA_MCP_WATCH=0`` if you prefer manual indexing or use the hook only.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import Context
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, get_config, get_config_info
from .errors import StrophaError
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
    ctx_from_mcp,
    persist_session_state,
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
        repo_url: Optional git URL to search (e.g., https://github.com/org/repo).
                  The repo is cloned and indexed on first access, then cached.
        refresh: If True with repo_url, re-clone and re-index the remote repository.
        context_budget: Optional token budget for result packing (WIN-208). When
            set, results are greedily packed in score order up to this many
            estimated tokens. May return fewer than top_k hits; the truncated
            tail is reported via the ``spilled_count`` field. Pass when your
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
    from .tools.git_url import is_git_url, search_git_url
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
        search_git_url_fn=search_git_url,
        apply_filters_fn=_apply_filters,
        stropha_error_cls=StrophaError,
        log=log,
    )


@mcp.tool(
    title="Reset session dedup",
    description=(
        "WIN-217 — clear the session's seen-chunk-ids set. Use this when "
        "the agent has finished one logical task and wants a fresh slate "
        "for the next retrieval round, OR when an iterative search keeps "
        "returning empty because everything got dedup'd away. The set "
        "lives in memory only for this MCP process; restarting the "
        "server already clears it. Returns the count of ids dropped."
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
        "twice is a no-op."
    ),
)
@traced("pin_chunks")
def pin_chunks(chunk_ids: list[str], *, ctx: Context) -> PinResponse:
    """Add ``chunk_ids`` to the session's pin set."""
    return pin_chunks_impl(
        chunk_ids,
        _ctx(ctx),
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
        "decide whether to unpin them."
    ),
)
@traced("list_pins")
def list_pins(*, ctx: Context) -> ListPinsResponse:
    """Resolve every pinned chunk_id into a structured response."""
    return list_pins_impl(_ctx(ctx), to_result=_to_result)


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
        "Prefer this over manually choosing tools when the query type is ambiguous."
    ),
)
@traced("smart_search")
def smart_search_tool(
    query: str,
    top_k: int = 10,
    use_llm_router: bool = False,
    *,
    ctx: Context,
) -> SmartSearchResponse:
    """Smart search with automatic query routing.
    
    Args:
        query: Natural-language question or technical terms.
        top_k: 1–30. Default 10.
        use_llm_router: If True, use LLM for ambiguous queries (slower but more accurate).
    """
    return smart_search_tool_impl(
        query,
        top_k,
        use_llm_router,
        _ctx(ctx),
        to_result=_to_result,
        smart_search_fn=smart_search,
        graph_loaded_fn=graph_loaded,
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
        "Returns file path, line numbers, and code snippet for each match."
    ),
)
@traced("get_symbol")
def get_symbol(symbol: str, limit: int = 5, *, ctx: Context) -> list[SearchResult]:
    """Find by symbol name. Returns up to `limit` matches, exact-match first."""
    return get_symbol_impl(symbol, limit, _ctx(ctx), to_result=_to_result)


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
        "then you can Read just lines 45-67 for the specific method you need."
    ),
)
@traced("get_file_outline")
def get_file_outline(path: str, *, ctx: Context) -> list[OutlineEntry]:
    """Return the chunk-level outline of a file path (repo-relative)."""
    return get_file_outline_impl(path, _ctx(ctx))


@mcp.tool(
    title="Expand chunk by id (just-in-time dereference)",
    description=(
        "WIN-216 — dereference a lightweight chunk handle into the full "
        "SearchResult. Use this after a search_code(card_level='line') or "
        "an assemble_context bundle: the agent gets cheap handles first, "
        "then expands only the chunks it actually needs. Mirrors Claude "
        "Code's file-on-demand pattern. Returns ``found=False`` when the "
        "chunk_id is unknown (e.g. the index was rebuilt and the id is "
        "stale) — never raises."
    ),
)
@traced("expand_chunk")
def expand_chunk(
    chunk_id: str,
    card_level: str = "full",
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
    return expand_chunk_impl(chunk_id, card_level, _ctx(ctx), to_result=_to_result)


@mcp.tool(
    title="Expand chunks (batched)",
    description=(
        "WIN-216 batch variant of expand_chunk. Pass a list of chunk_ids "
        "to dereference them in one MCP call. Useful when assemble_context "
        "returned a bundle of line-card handles and the agent decided to "
        "expand several at once. Missing chunk_ids appear in the response "
        "with found=False; the call never partially fails."
    ),
)
@traced("expand_chunks")
def expand_chunks(
    chunk_ids: list[str],
    card_level: str = "full",
    *,
    ctx: Context,
) -> ExpandChunksResponse:
    """Batch dereference. Returns a parallel list — one response per id."""
    return expand_chunks_impl(chunk_ids, card_level, _ctx(ctx), to_result=_to_result)


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
    kind: str, key: str, *, ctx: Context, repo: str | None = None,
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
        "otherwise take 3-4 chained calls."
    ),
)
@traced("plan_investigation")
def plan_investigation(
    task: str,
    top_k: int = 5,
    include_callers: bool = True,
    include_tests: bool = True,
    include_rationale: bool = True,
    *,
    ctx: Context,
) -> InvestigationPlanResponse:
    """Return an ordered investigation plan for ``task``."""
    from .retrieval.investigation import plan_investigation as _plan

    app = _ctx(ctx)
    plan = _plan(
        task,
        search_engine=app.search_engine,
        storage=app.storage,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        include_rationale=include_rationale,
    )
    return InvestigationPlanResponse(
        task=plan.task,
        steps=[
            InvestigationStepEntry(
                order=s.order, tool=s.tool, purpose=s.purpose, args=s.args,
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
) -> dict:
    """Return callers of ``symbol`` (BFS up incoming `calls` edges, depth ≤ 3)."""
    return find_callers_impl(
        symbol,
        _ctx(ctx),
        depth=depth,
        limit=limit,
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
        "Use for exploring dependencies and connections around a symbol."
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
) -> dict:
    """Return any node connected to ``symbol`` via the graph."""
    return find_related_impl(
        symbol,
        _ctx(ctx),
        depth=depth,
        limit=limit,
        relations=relations,
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
        "see everything that participates in a coherent feature/module."
    ),
)
@traced("get_community")
def get_community(
    symbol_or_community_id: str,
    *,
    ctx: Context,
    limit: int = 50,
) -> dict:
    """Return the community membership of a symbol or community id."""
    return get_community_impl(
        symbol_or_community_id,
        _ctx(ctx),
        limit=limit,
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
        "when no rationale edges target the symbol."
    ),
)
@traced("find_rationale")
def find_rationale(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 10,
) -> dict:
    """Return rationale nodes (docs/ADRs) explaining ``symbol``."""
    return find_rationale_impl(
        symbol,
        _ctx(ctx),
        limit=limit,
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
        "Pass custom test_path_patterns for non-standard test conventions."
    ),
)
@traced("find_tests_for")
def find_tests_for(
    symbol: str,
    *,
    ctx: Context,
    limit: int = 20,
    test_path_patterns: list[str] | None = None,
) -> dict:
    """Return tests that exercise ``symbol`` (path-pattern filtered)."""
    return find_tests_for_impl(
        symbol,
        _ctx(ctx),
        limit=limit,
        test_path_patterns=test_path_patterns,
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
        "Great for understanding end-to-end flows and Gherkin scenario traces."
    ),
)
@traced("trace_feature")
def trace_feature(
    feature: str,
    *,
    ctx: Context,
    max_paths: int = 5,
    max_depth: int = 6,
) -> dict:
    """Walk the call graph from feature-matching entry points."""
    return trace_feature_impl(
        feature,
        _ctx(ctx),
        max_paths=max_paths,
        max_depth=max_depth,
        graph_loaded_fn=graph_loaded,
        unavailable_fn=_graph_unavailable,
        graph_trace_feature_fn=graph_trace_feature,
    )


@mcp.tool(
    title="Report bad result",
    description=(
        "Flag a previous retrieval as bad — Strong-negative T6 signal for "
        "the Meta-Harness outcome labeler (WIN-162). Use this when a result "
        "from search_code / smart_search / get_symbol / find_* was clearly "
        "wrong: irrelevant, the wrong symbol, stale, a duplicate, too verbose, "
        "or other. Pass the `trace_id` from the offending response — search_code "
        "and smart_search return it as a top-level `trace_id` field (WIN-160); "
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
        "current git diff (WIN-211)."
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
    *,
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
    # When diff_scope is requested, the target repo for the diff is
    # the configured ``target_repo`` so the agent doesn't have to
    # remember the path. The library function falls back gracefully
    # when the repo is missing or git fails.
    diff_repo = app.config.target_repo if diff_scope else None
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
        "where you'd otherwise chain N MCP calls."
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
    *,
    ctx: Context,
) -> AssembleContextResponse:
    """Plan + execute an investigation in one server-side call."""
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
    plan = _plan(
        task,
        search_engine=app.search_engine,
        storage=app.storage,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        include_rationale=include_rationale,
    )
    result = execute_plan(
        plan,
        search_engine=app.search_engine,
        storage=app.storage,
        context_budget=context_budget,
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
    drift-detection key — see CLAUDE.md §3.1), and any sub-adapters
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
        "storage": {
            **(_stage(built.storage) or {}),
            # Report the ACTIVE backend's real target — the Postgres DSN
            # (password-redacted) when postgres is the storage adapter, the
            # SQLite file path otherwise. Falls back to the configured index
            # path for adapters that don't expose ``storage_target``.
            "path": (
                getattr(built.storage, "storage_target", None)
                or str(cfg.resolve_index_path())
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
    if backend == "deepseek":
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
            summary["edge_count"] = graph_stats.get("edges", 0)
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


def _features_snapshot(cfg: Config) -> dict:
    """Every query-time toggle from retrieval_features, with on/off summary."""
    rf = cfg.retrieval_features
    return {
        "hyde": {
            "enabled": rf.hyde_enabled,
            "code_tuned": rf.hyde_code_tuned,
            "purpose": "Generate hypothetical code via LLM, embed THAT for dense search.",
        },
        "query_decomposition": {
            "enabled": rf.query_decomposition_enabled,
            "uses_llm": rf.query_decomposition_llm,
            "purpose": "Split compound queries ('X and Y') into sub-queries; RRF-fuse results.",
        },
        "query_router_llm": {
            "enabled": rf.query_router_llm,
            "purpose": "LLM classifies query intent and routes to find_callers / find_tests_for / etc.",
        },
        "query_rewrite": {
            "enabled": rf.query_rewrite_enabled,
            "purpose": "LLM expands natural language into code-style terms. Redundant with HyDE for most cases.",
        },
        "multi_query": {
            "enabled": rf.multi_query_enabled,
            "count": rf.multi_query_count,
            "purpose": "Generate N paraphrases of the query, fuse results. Costs N LLM calls per query.",
        },
        "recursive_retrieval": {
            "enabled": rf.recursive_enabled,
            "adjacency": rf.recursive_adjacency,
            "purpose": "Auto-merge adjacent chunks + promote parent when siblings hit.",
        },
        "query_cache": {
            "enabled": rf.cache_enabled,
            "size": rf.cache_size,
            "ttl_s": rf.cache_ttl,
            "purpose": "LRU cache of recent query results.",
        },
        "adaptive_weighting": {
            "enabled": rf.adaptive_weighting,
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

    # Reranker is "active" if a non-noop adapter was built.
    reranker = (
        getattr(built.retrieval, "_reranker", None) if built.retrieval else None
    )
    reranker_active = reranker is not None and getattr(
        reranker, "adapter_name", "noop"
    ) != "noop"

    # Enricher uses an LLM iff it's contextual / ollama / mlx.
    enricher_uses_llm = getattr(built.enricher, "adapter_name", "noop") in {
        "contextual", "ollama", "mlx",
    }

    # Build a "active LLM features" list for quick scan.
    llm_active: list[str] = []
    if enricher_uses_llm:
        llm_active.append(f"enricher({built.enricher.adapter_name})")
    if rf.hyde_enabled:
        llm_active.append("hyde")
    if rf.query_rewrite_enabled:
        llm_active.append("query_rewrite")
    if rf.query_router_llm:
        llm_active.append("query_router")
    if rf.query_decomposition_llm:
        llm_active.append("query_decomposition")
    if rf.multi_query_enabled:
        llm_active.append(f"multi_query×{rf.multi_query_count}")

    return {
        "embedder": f"{built.embedder.adapter_name}:{built.embedder.model_name}@{built.embedder.dim}d",
        "enricher": built.enricher.adapter_name,
        "enricher_uses_llm": enricher_uses_llm,
        "inference_backend": (cfg.inference.backend or "auto"),
        "rerank_active": reranker_active,
        "rerank_model": (
            getattr(reranker, "adapter_id", None) if reranker_active else None
        ),
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
    info["retrieval_features"] = _features_snapshot(app.config)
    info["graph"] = _graph_snapshot(app)
    info["available_adapters"] = _available_adapters()

    return info


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
        "actionable signal."
    ),
)
@traced("suggest_next_tool")
def suggest_next_tool_mcp(
    query: str = "",
    chunk_ids: list[str] | None = None,
    max_suggestions: int = 5,
    *,
    ctx: Context,
) -> SuggestNextResponse:
    """Return ranked next-tool suggestions for the agent."""
    from .retrieval.suggest_next import suggest_next_tool as _suggest

    app = _ctx(ctx)
    max_suggestions = max(1, min(int(max_suggestions), 20))

    # Resolve which chunks to inspect: explicit list wins; otherwise
    # fall back to the most recent ones the agent has seen.
    ids = list(chunk_ids) if chunk_ids else list(app.seen_chunk_ids)[-10:]

    hits: list = []
    for cid in ids:
        try:
            hit = app.storage.lookup_by_chunk_id(cid)
        except Exception:  # pragma: no cover - defensive
            hit = None
        if hit is not None:
            hits.append(hit)

    has_graph = graph_loaded(app.storage)
    has_rationale = has_rationale_edges(app.storage) if has_graph else False

    raw = _suggest(
        query, hits,
        has_graph=has_graph,
        has_rationale=has_rationale,
        max_suggestions=max_suggestions,
    )

    return SuggestNextResponse(
        query=query,
        suggestions=[
            SuggestionEntry(
                tool=s.tool,
                args=dict(s.args),
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
        recipe, context, budget_tokens=budget_tokens, strict=False,
    )
    effective_budget = (
        recipe.default_budget if budget_tokens is None
        else max(0, int(budget_tokens))
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
        "model without re-querying. Never raises."
    ),
)
@traced("verify_claim")
def verify_claim_tool(
    claim: str,
    *,
    ctx: Context,
) -> VerdictResponse:
    """Parse + verify a claim against the graph."""
    return verify_claim_impl(claim, _ctx(ctx), graph_loaded_fn=graph_loaded)


@mcp.tool(
    title="Guard against hallucinated symbols",
    description=(
        "WIN-245 — vet a batch of symbol names against the index "
        "BEFORE the agent commits them to code. Each symbol gets a "
        "verdict (exists/missing) plus near-miss suggestions on "
        "misses, so the agent can pick the correct name without a "
        "second search. Defensive: callable exceptions become notes, "
        "never propagate; empty / duplicate names skipped."
    ),
)
@traced("check_symbols")
def check_symbols_tool(
    symbols: list[str],
    max_suggestions: int = 5,
    *,
    ctx: Context,
) -> HallucinationGuardResponse:
    """Run the hallucination guard against the live index."""
    return check_symbols_impl(symbols, _ctx(ctx), max_suggestions=max_suggestions)


@mcp.tool(
    title="Post-edit blast-radius report",
    description=(
        "WIN-246 — given the symbols you just edited, return every "
        "caller / test / related symbol / rationale node that may be "
        "affected, deduplicated across the input list. Use AFTER an "
        "edit to know what else needs verification. Composes the "
        "existing find_callers / find_tests_for / find_related / "
        "find_rationale graph readers under one call."
    ),
)
@traced("blast_radius")
def blast_radius_tool(
    symbols: list[str],
    *,
    ctx: Context,
) -> BlastRadiusResponse:
    """Build a blast-radius report for the changed symbols."""
    return blast_radius_impl(
        symbols,
        _ctx(ctx),
        graph_loaded_fn=graph_loaded,
        has_rationale_edges_fn=has_rationale_edges,
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
        "whitespace-normalised."
    ),
)
@traced("assert_snippet_freshness")
def assert_snippet_freshness_tool(
    chunk_id: str,
    expected_snippet: str,
    strict: bool = False,
    *,
    ctx: Context,
) -> FreshnessResponse:
    """Verify a cached snippet against the live index."""
    return assert_snippet_freshness_impl(
        chunk_id,
        expected_snippet,
        _ctx(ctx),
        strict=bool(strict),
    )


@mcp.tool(
    title="Test-impact analysis",
    description=(
        "WIN-248 — given the changed symbols, compute the minimal "
        "test set that should run. Direct tests via find_tests_for "
        "plus (when transitive_depth>=1) tests covering immediate "
        "callers. Returns a ready-to-paste run_command (pytest "
        "by default; pass runner='go test' / 'npm test' for other "
        "ecosystems). Never raises."
    ),
)
@traced("test_impact")
def test_impact_tool(
    symbols: list[str],
    transitive_depth: int = 1,
    runner: str = "pytest",
    *,
    ctx: Context,
) -> TestImpactResponse:
    """Compute the impacted test set for the changed symbols."""
    return test_impact_impl(
        symbols,
        _ctx(ctx),
        transitive_depth=transitive_depth,
        runner=runner,
        graph_loaded_fn=graph_loaded,
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
        "Use to prevent 'vibes' assertions from shipping."
    ),
)
@traced("audit_citations")
def audit_citations_tool(
    text: str,
    *,
    ctx: Context,
) -> CitationAuditResponse:
    """Audit the agent's text for verifiable citations."""
    return audit_citations_impl(text, _ctx(ctx))


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
        "LOW (note)."
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
    return spec_consistency_impl(
        changed_symbols,
        changed_files,
        _ctx(ctx),
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
        "to stay intact."
    ),
)
@traced("contract_tests")
def contract_tests_tool(
    symbol: str,
    *,
    ctx: Context,
) -> ContractTestsResponse:
    """Surface contract tests for ``symbol``."""
    return contract_tests_impl(symbol, _ctx(ctx), graph_loaded_fn=graph_loaded)


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
        "the current session; pass session_id to query another."
    ),
)
@traced("working_set")
def working_set_tool(
    session_id: str = "",
    budget_tokens: int = 4000,
    max_recent: int = 20,
    *,
    ctx: Context,
) -> WorkingSetResponse:
    """Assemble the working set for the current (or given) session."""
    from .retrieval.working_set import build_working_set

    app = _ctx(ctx)
    sid = (session_id or "").strip() or app.session_id

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
            return app.storage.lookup_by_chunk_id(cid)
        except Exception:
            return None

    ws = build_working_set(
        pinned, seen,
        lookup_fn=_lookup,
        session_id=sid,
        budget_tokens=max(0, int(budget_tokens)),
        max_recent=max(0, int(max_recent)),
    )

    return WorkingSetResponse(
        session_id=ws.session_id,
        entries=[
            WorkingSetEntryModel(
                chunk_id=e.chunk_id, path=e.path, symbol=e.symbol,
                kind=e.kind, start_line=e.start_line, end_line=e.end_line,
                snippet=e.snippet, role=e.role, tokens=e.tokens,
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

    resolved = load_pipeline_config()
    built = build_stages(resolved)
    with built.storage as storage:
        try:
            record = storage.get_latest_session()
        except Exception:
            record = None
        if record is None:
            return "# Working set\n\n_no session state yet._\n"

        def _lookup(cid: str):
            try:
                return storage.lookup_by_chunk_id(cid)
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
        return (
            f"Unknown recipe {recipe_name!r}. "
            "List recipes via list_task_recipes."
        )

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
