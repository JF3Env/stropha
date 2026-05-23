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
stdio by default; that is the transport Claude Code, Cursor and friends
launch automatically when configured via `.mcp.json`.

Background file watcher (enabled by default via ``STROPHA_MCP_WATCH=1``):
The server starts a background thread that polls the target repo for file
changes. After a debounce period (default 2s of no further changes), it
triggers a light re-index pass so MCP queries always see recent edits
without requiring a separate ``stropha watch`` command. Disable with
``STROPHA_MCP_WATCH=0`` if you prefer manual indexing or use the hook only.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, get_config, get_config_info
from .embeddings.base import Embedder
from .errors import StrophaError
from .logging import configure_logging, get_logger
from .pipeline import build_stages, load_pipeline_config
from .retrieval import SearchEngine
from .retrieval.graph import (
    find_callers as graph_find_callers,
    find_rationale as graph_find_rationale,
    find_related as graph_find_related,
    find_tests_for as graph_find_tests_for,
    get_community as graph_get_community,
    graph_loaded,
    has_rationale_edges,
    trace_feature as graph_trace_feature,
)
from .retrieval.savings import track_search_savings
from .retrieval.smart_search import SmartSearchResult, smart_search
from .meta.decorator import traced
from .retrieval.graph import (
    find_rationale as graph_find_rationale,
)
from .retrieval.graph import (
    find_related as graph_find_related,
)
from .retrieval.graph import (
    get_community as graph_get_community,
)
from .retrieval.graph import (
    graph_loaded,
    has_rationale_edges,
)
from .storage import Storage
from .watch import WatchController

log = get_logger(__name__)


# ---- composition root ------------------------------------------------------

@dataclass
class AppContext:
    """Long-lived dependencies, injected into every tool via the MCP Context.

    ``built`` is the full :class:`BuiltStages` so MCP tools can introspect
    every pipeline stage (walker, chunker, enricher, retrieval, reranker)
    — not just storage + embedder. The ``get_config`` tool uses this to
    render a comprehensive runtime snapshot for the calling agent.

    ``session_id`` (WIN-160) is rotated on every ``mcp.startup`` — one uuid4
    per process, recorded on every trace entry so the Meta-Harness outcome
    labeler can detect engagement between correlated calls (tier T4).
    """

    config: Config
    storage: Storage
    embedder: Embedder
    search_engine: SearchEngine
    built: object  # BuiltStages; typed as object to avoid the heavy import here
    session_id: str = ""
    trace_recorder: object = None  # TraceRecorder | None; typed loose to avoid import cycles
    bad_result_store: object = None  # BadResultStore | None — produces the T6 signal for the labeler


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[AppContext]:
    import uuid as _uuid
    from pathlib import Path as _Path

    from .meta import TraceRecorder as _TraceRecorder

    # Use the centralized config singleton. It handles .env loading with proper
    # precedence (env vars from MCP client override .env file values).
    cfg = get_config()
    configure_logging(cfg.log_level)
    # WIN-160: one session_id per process, generated at mcp.startup. Logged
    # immediately so the operator can correlate this run's events.
    session_id = _uuid.uuid4().hex
    log.info(
        "mcp.startup",
        version=__version__,
        index_path=str(cfg.resolve_index_path()),
        target=str(cfg.target_repo),
        mcp_watch=cfg.mcp_watch,
        session_id=session_id,
    )
    # Route through the adapter framework so pipeline.embedder.adapter from
    # stropha.yaml is honored (mlx, voyage, local, bge-m3, …). Without this
    # the MCP server would diverge from `stropha index` / `stropha search`,
    # which both go through build_stages().
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    embedder = built.embedder
    storage = built.storage  # already open; sqlite-vec adapter subclass of Storage
    engine = SearchEngine(storage, embedder)

    # WIN-160: spin up the Meta-Harness trace recorder. Default location is
    # ``<index_parent>/meta/traces/`` next to ``index.db`` so a single repo's
    # meta state stays self-contained. WIN-161 will formalize the layout.
    traces_dir = (
        _Path(os.environ.get("STROPHA_TRACES_DIR"))
        if os.environ.get("STROPHA_TRACES_DIR")
        else _Path(cfg.resolve_index_path()).parent / "meta" / "traces"
    )
    try:
        trace_recorder: object = _TraceRecorder(
            base_dir=traces_dir, session_id=session_id
        )
    except OSError as exc:
        # Trace recording is observability, not critical path. If we cannot
        # open the directory (read-only fs, permission, …) log and continue.
        log.warning("mcp.trace_recorder.disabled", error=str(exc))
        trace_recorder = None

    # WIN-163: bad-result feedback store. Lives next to traces under the
    # same parent meta/ directory so the labeler can find it without
    # extra plumbing.
    from .meta.feedback import BadResultStore as _BadResultStore
    try:
        bad_result_store: object = _BadResultStore(base_dir=traces_dir.parent)
    except OSError as exc:
        log.warning("mcp.bad_result_store.disabled", error=str(exc))
        bad_result_store = None

    # Start background file watcher if enabled (default: on).
    # This keeps the index fresh as the user edits files, so MCP queries
    # always see recent changes without requiring a separate `stropha watch`.
    watcher: WatchController | None = None
    if cfg.mcp_watch:
        watcher = WatchController(
            repo=cfg.target_repo,
            interval_s=cfg.mcp_watch_interval,
            debounce_s=cfg.mcp_watch_debounce,
            full_refresh=False,  # Light refresh; hook handles graphify
        )
        watcher.start()

    try:
        yield AppContext(
            config=cfg,
            storage=storage,
            embedder=embedder,
            search_engine=engine,
            built=built,
            session_id=session_id,
            trace_recorder=trace_recorder,
            bad_result_store=bad_result_store,
        )
    finally:
        if watcher is not None:
            watcher.stop()
        if trace_recorder is not None:
            try:
                trace_recorder.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — defensive shutdown
                log.warning("mcp.trace_recorder.close_failed", error=str(exc))
        log.info("mcp.shutdown", session_id=session_id)
        storage.close()


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


mcp = FastMCP(
    name="stropha_rag",
    instructions=_INSTRUCTIONS,
    lifespan=_lifespan,
)


# ---- response models --------------------------------------------------------

class RepoInfo(BaseModel):
    """Source-repo identity attached to every search result.

    A client can run `git clone {url}` (when `url` is non-null) to obtain
    the source tree, then check out `default_branch` (or `head_commit`).
    """

    normalized_key: str = Field(
        description="Stable cross-user key (e.g. 'github.com/foo/bar')."
    )
    url: str | None = Field(default=None, description="HTTPS clone URL for remote access.")
    local_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the repo root on the indexer host. "
            "Combined with 'path' gives the full local file path."
        ),
    )
    default_branch: str | None = None
    head_commit: str | None = None


class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Hybrid (RRF) score; higher = more relevant.")
    path: str = Field(description="File path relative to the repo root.")
    start_line: int
    end_line: int
    language: str
    kind: str = Field(description="file | class | method | function | section | scenario | …")
    symbol: str | None = None
    snippet: str = Field(description="Truncated preview. Use Read tool for full content.")
    chunk_id: str
    repo: RepoInfo | None = Field(
        default=None,
        description="Source repository identity (URL, branch, HEAD).",
    )


class FacetCounts(BaseModel):
    """Counts per facet dimension, computed from search results."""

    language: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per language (e.g. {'python': 15, 'java': 8}).",
    )
    kind: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per chunk kind (e.g. {'method': 20, 'class': 5}).",
    )
    repo: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results per repository.",
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_candidates: int = Field(
        description="Total candidates considered before fusion (sum of dense + sparse top-50)."
    )
    query: str
    facets: FacetCounts | None = Field(
        default=None,
        description="Facet counts when include_facets=True. Shows distribution by language, kind, repo.",
    )
    # WIN-208 — token-budget packing observability. Populated only when
    # the caller passed ``context_budget``; left at defaults otherwise so
    # the legacy response shape is unchanged.
    spilled_count: int = Field(
        default=0,
        description=(
            "WIN-208: number of hits that fell off the tail because "
            "context_budget was tight. 0 when no packing was requested."
        ),
    )
    truncated_oversize: bool = Field(
        default=False,
        description=(
            "WIN-208: True when the single top hit exceeds context_budget "
            "and was returned anyway (escape hatch to avoid empty result)."
        ),
    )


class OutlineEntry(BaseModel):
    chunk_id: str
    kind: str
    symbol: str | None
    parent_chunk_id: str | None
    start_line: int
    end_line: int


class RepoSummary(BaseModel):
    """One entry in the `list_repos` response.

    Includes both the git origin URL and local filesystem path for each
    repo, so callers can access files via either method.
    """

    normalized_key: str
    url: str | None = Field(default=None, description="HTTPS clone URL for remote access.")
    local_path: str | None = Field(
        default=None,
        description="Absolute filesystem path to the repo root on the indexer host.",
    )
    default_branch: str | None = None
    head_commit: str | None = None
    files: int
    chunks: int
    last_indexed_at: str | None


class StatsPayload(BaseModel):
    db_path: str
    size_bytes: int
    chunks: int
    files: int
    index_dim: int
    models: list[dict]
    repos: list[RepoSummary] = Field(default_factory=list)
    enrichers: list[dict] = Field(default_factory=list)
    enrichment_cache_size: int = 0
    graph: dict | None = Field(
        default=None,
        description="Graphify mirror summary (None when no graph loaded).",
    )


# ---- helpers ---------------------------------------------------------------

def _ctx(mcp_ctx: Context) -> AppContext:
    """Pull the AppContext out of an MCP Context and bind session_id to logs.

    WIN-160: ``structlog.contextvars.bind_contextvars`` makes ``session_id``
    appear on every structlog event emitted while this request is in flight,
    without each tool having to remember to pass it. The bind is per-thread
    (Python's contextvars semantics), so concurrent MCP calls do not bleed
    each other's session ids.
    """
    import structlog

    app: AppContext = mcp_ctx.request_context.lifespan_context
    if app.session_id:
        structlog.contextvars.bind_contextvars(session_id=app.session_id)
    return app


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
    result = hits
    
    if language:
        lang_set = {lang.lower() for lang in language}
        result = [h for h in result if h.language.lower() in lang_set]
    
    if path_prefix:
        result = [h for h in result if h.rel_path.startswith(path_prefix)]
    
    if kind:
        kind_set = {k.lower() for k in kind}
        result = [h for h in result if h.kind.lower() in kind_set]
    
    if exclude_tests:
        test_patterns = ("test_", "_test.", ".test.", "/tests/", "/test/", "Test.")
        result = [
            h for h in result
            if not any(p in h.rel_path for p in test_patterns)
        ]
    
    return result


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
    """
    import os

    from .tools.git_url import is_git_url, search_git_url

    app = _ctx(ctx)
    top_k = max(1, min(top_k, 30))

    # Enable recursive retrieval for this search if requested
    prev_recursive = os.environ.get("STROPHA_RECURSIVE_RETRIEVAL")
    if recursive:
        os.environ["STROPHA_RECURSIVE_RETRIEVAL"] = "1"

    # Fetch more candidates when filtering to ensure we get enough results
    has_filters = any([language, path_prefix, kind, exclude_tests])
    fetch_k = top_k * 3 if has_filters else top_k

    try:
        # Handle remote git URL search
        if repo_url and is_git_url(repo_url):
            log.info("mcp.search_code.remote", repo_url=repo_url, refresh=refresh)
            hits = search_git_url(query, repo_url, top_k=fetch_k, refresh=refresh)
        else:
            hits = app.search_engine.search(query, top_k=fetch_k)
    except StrophaError as exc:
        log.warning("mcp.search_code.error", error=str(exc))
        return SearchResponse(results=[], total_candidates=0, query=query, facets=None)
    finally:
        # Restore previous env var state
        if recursive:
            if prev_recursive is None:
                os.environ.pop("STROPHA_RECURSIVE_RETRIEVAL", None)
            else:
                os.environ["STROPHA_RECURSIVE_RETRIEVAL"] = prev_recursive

    # Apply filters
    if has_filters:
        hits = _apply_filters(
            hits,
            language=language,
            path_prefix=path_prefix,
            kind=kind,
            exclude_tests=exclude_tests,
        )

    # Compute facets before truncating (on filtered results)
    facets: FacetCounts | None = None
    if include_facets and hits:
        chunk_ids = [h.chunk_id for h in hits]
        facet_data = app.storage.compute_facets(chunk_ids)
        facets = FacetCounts(
            language=facet_data["language"],
            kind=facet_data["kind"],
            repo=facet_data["repo"],
        )

    # Truncate to requested top_k
    hits = hits[:top_k]

    # WIN-210: Extractive compression of each hit's snippet — keep
    # the signature, matched lines, small context window. Applied
    # BEFORE budget packing so the budget reflects compressed cost.
    if compress and hits:
        from .retrieval.compression import compress_hits

        hits = compress_hits(hits, query)

    # WIN-208: Greedy-pack to ``context_budget`` AFTER top_k truncation.
    # Done at the MCP layer (not inside SearchEngine.search) so the
    # facet computation above sees the full top_k slice — facet counts
    # describe what was *available* before packing, not what survived.
    # The CLI search wires the budget via the engine's own packing path
    # since it doesn't compute facets.
    spilled_count = 0
    truncated_oversize = False
    if context_budget is not None and context_budget > 0:
        from .retrieval.packing import pack_to_budget

        pack = pack_to_budget(hits, context_budget)
        hits = pack.packed
        spilled_count = len(pack.spilled)
        truncated_oversize = pack.truncated
        log.info(
            "mcp.search_code.packed",
            budget=context_budget,
            total_tokens=pack.total_tokens,
            packed=len(hits),
            spilled=spilled_count,
            truncated=truncated_oversize,
        )

    # WIN-158: Track token savings so `stropha cost` can report on RAG value.
    # The producer at retrieval/savings.py emits a `search.served` structlog
    # event which the consumer at cost.py:211 already aggregates. Wrapped in
    # try/except so a savings-calc bug can never break a search response —
    # this is an observability hook, not a critical path.
    try:
        track_search_savings(query, hits)
    except Exception:
        # Defensive: never fail a search response on a stats-tracking bug.
        log.debug("search.savings.track_failed", exc_info=True)

    return SearchResponse(
        results=[_to_result(h) for h in hits],
        total_candidates=len(hits),
        query=query,
        facets=facets,
        spilled_count=spilled_count,
        truncated_oversize=truncated_oversize,
    )


# ---- smart search (query routing) ------------------------------------------

class SmartSearchResponse(BaseModel):
    """Response from smart_search with routing metadata."""
    results: list[SearchResult]
    total_candidates: int
    query: str
    intent: str = Field(description="Detected query intent (find_callers, find_tests_for, search_code, etc.)")
    symbol: str | None = Field(default=None, description="Extracted symbol name, if any")
    tool_used: str = Field(description="The actual tool invoked (find_callers, search_code, etc.)")
    confidence: float = Field(description="Routing confidence (0.0-1.0)")
    graph_available: bool = Field(default=True, description="Whether the graph is loaded")


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
    app = _ctx(ctx)
    top_k = max(1, min(top_k, 30))
    
    try:
        result = smart_search(
            query,
            app.storage,
            app.embedder,
            top_k=top_k,
            use_router=True,
            use_llm_router=use_llm_router,
        )
        return SmartSearchResponse(
            results=[_to_result(h) for h in result.hits],
            total_candidates=len(result.hits),
            query=query,
            intent=result.routed.intent.value,
            symbol=result.routed.symbol,
            tool_used=result.tool_used,
            confidence=result.routed.confidence,
            graph_available=graph_loaded(app.storage),
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
            graph_available=graph_loaded(app.storage),
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
    app = _ctx(ctx)
    hits = app.storage.lookup_by_symbol(symbol, limit=max(1, min(limit, 20)))
    return [_to_result(h) for h in hits]


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
    app = _ctx(ctx)
    return [
        RepoSummary(
            normalized_key=r.normalized_key,
            url=r.url,
            local_path=r.local_path,
            default_branch=r.default_branch,
            head_commit=r.head_commit,
            files=r.files,
            chunks=r.chunks,
            last_indexed_at=r.last_indexed_at,
        )
        for r in app.storage.list_repos()
    ]


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
    app = _ctx(ctx)
    rows = app.storage.file_outline(path)
    return [
        OutlineEntry(
            chunk_id=r["chunk_id"],
            kind=r["kind"],
            symbol=r["symbol"],
            parent_chunk_id=r["parent_chunk_id"],
            start_line=int(r["start_line"]),
            end_line=int(r["end_line"]),
        )
        for r in rows
    ]


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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    return graph_find_callers(app.storage, symbol, depth=depth, limit=limit)


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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    return graph_find_related(
        app.storage, symbol, depth=depth, limit=limit,
        relations=tuple(relations) if relations else None,
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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"query": symbol_or_community_id}
    # Accept stringified int as community id
    query: str | int = symbol_or_community_id
    try:
        query = int(symbol_or_community_id)
    except (ValueError, TypeError):
        pass
    return graph_get_community(app.storage, query, limit=limit)


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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    if not has_rationale_edges(app.storage):
        return {
            "symbol": symbol,
            "rationale": [],
            "message": "Graph loaded but no `rationale_for` edges present yet.",
        }
    return graph_find_rationale(app.storage, symbol, limit=limit)


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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"symbol": symbol}
    patterns = (
        tuple(test_path_patterns)
        if test_path_patterns
        else ("test_", "_test", "/tests/", "/test/", ".spec.", ".test.")
    )
    return graph_find_tests_for(
        app.storage, symbol, limit=limit, test_path_patterns=patterns,
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
    app = _ctx(ctx)
    if not graph_loaded(app.storage):
        return _graph_unavailable() | {"feature": feature}
    return graph_trace_feature(
        app.storage, feature, max_paths=max_paths, max_depth=max_depth,
    )


@mcp.tool(
    title="Report bad result",
    description=(
        "Flag a previous retrieval as bad — Strong-negative T6 signal for "
        "the Meta-Harness outcome labeler (WIN-162). Use this when a result "
        "from search_code / smart_search / get_symbol / find_* was clearly "
        "wrong: irrelevant, the wrong symbol, stale, a duplicate, too verbose, "
        "or other. The trace_id comes from the trace stream WIN-160 attaches "
        "to every tool call (also visible in the meta traces directory)."
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
        "view if you'd rather work in JSON."
    ),
)
@traced("assemble_context")
def assemble_context(
    task: str,
    context_budget: int = 4000,
    top_k: int = 10,
    include_callers: bool = False,
    include_tests: bool = False,
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
    result = _assemble(
        task,
        search_engine=app.search_engine,
        context_budget=context_budget,
        top_k=top_k,
        include_callers=include_callers,
        include_tests=include_tests,
        storage=app.storage,
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
            "path": str(cfg.resolve_index_path()),
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


# ---- entry point -----------------------------------------------------------

def main() -> None:
    """Run the MCP server over stdio. Used by `[project.scripts]` console entry."""
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
