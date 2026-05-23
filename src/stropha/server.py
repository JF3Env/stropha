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
from collections import deque
from dataclasses import dataclass, field

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

    ``seen_chunk_ids`` (WIN-217) tracks which chunks have been served to
    the agent in this session, enabling session-scoped dedup on subsequent
    searches. Reset via ``reset_dedup`` MCP tool.

    ``pinned_chunk_ids`` (WIN-218) is the inverse: chunks the agent
    explicitly wants kept across retrievals. Pinning wins over dedup
    (pinned chunks are NEVER filtered out). Mutated via ``pin_chunks``
    / ``unpin_chunks`` MCP tools.

    ``client_id`` (WIN-221) — optional stable identifier the MCP
    client provides via ``STROPHA_CLIENT_ID``. When set, the
    Schema-v8 ``sessions`` row is resumed across server restarts so
    seen + pinned sets survive process boundaries. When unset, each
    process gets a fresh ephemeral session.

    ``recent_queries`` (WIN-224) — ring buffer of the last N
    ``search_code`` queries. Used when the agent passes
    ``conversation_aware=True`` to bias retrieval toward the current
    conversation thread instead of treating each query as amnesic.

    ``scratchpad`` (WIN-229) — durable free-form working notes that
    survive MCP-server restarts via the Schema-v12
    ``sessions.scratchpad`` column. Mutated via the
    ``scratchpad_*`` MCP tools.
    """

    config: Config
    storage: Storage
    embedder: Embedder
    search_engine: SearchEngine
    built: object  # BuiltStages; typed as object to avoid the heavy import here
    session_id: str = ""
    trace_recorder: object = None  # TraceRecorder | None; typed loose to avoid import cycles
    bad_result_store: object = None  # BadResultStore | None — produces the T6 signal for the labeler
    seen_chunk_ids: set[str] = field(default_factory=set)
    pinned_chunk_ids: set[str] = field(default_factory=set)
    client_id: str | None = None
    recent_queries: deque[str] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    scratchpad: str = ""


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

    # WIN-221 / EPIC 1.9: load the prior session row if a client_id is
    # configured and a matching session exists. Otherwise the in-
    # memory sets start empty and the row is created lazily on the
    # first persistent mutation.
    seen_chunk_ids: set[str] = set()
    pinned_chunk_ids: set[str] = set()
    client_id = os.environ.get("STROPHA_CLIENT_ID")
    if client_id:
        try:
            prior_sessions = storage.list_sessions(
                client_id=client_id, limit=1,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("mcp.sessions.list_failed", error=str(exc))
            prior_sessions = []
        if prior_sessions:
            prior = prior_sessions[0]
            seen_chunk_ids = set(prior.seen_chunk_ids)
            pinned_chunk_ids = set(prior.pinned_chunk_ids)
            prior_scratchpad = prior.scratchpad or ""
            # Reuse the same session_id so cross-restart correlation
            # via the trace recorder stays cheap.
            session_id = prior.session_id
            log.info(
                "mcp.sessions.resumed",
                session_id=session_id,
                client_id=client_id,
                seen=len(seen_chunk_ids),
                pinned=len(pinned_chunk_ids),
                scratchpad_chars=len(prior_scratchpad),
            )
        else:
            prior_scratchpad = ""
    else:
        prior_scratchpad = ""

    app_ctx = AppContext(
        config=cfg,
        storage=storage,
        embedder=embedder,
        search_engine=engine,
        built=built,
        session_id=session_id,
        trace_recorder=trace_recorder,
        bad_result_store=bad_result_store,
        seen_chunk_ids=seen_chunk_ids,
        pinned_chunk_ids=pinned_chunk_ids,
        client_id=client_id,
        scratchpad=prior_scratchpad,
    )
    try:
        yield app_ctx
    finally:
        if watcher is not None:
            watcher.stop()
        if trace_recorder is not None:
            try:
                trace_recorder.close()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — defensive shutdown
                log.warning("mcp.trace_recorder.close_failed", error=str(exc))
        # WIN-221: persist the final session state so the next process
        # picks up where we left off. We do this in ``finally`` so an
        # exception during the MCP yield doesn't lose the working set.
        # WIN-229 extends this to flush the scratchpad too — we read
        # from the LIVE app_ctx (not the startup-time prior_scratchpad)
        # so in-flight mutations during the session land on disk.
        try:
            storage.upsert_session(
                session_id=session_id,
                client_id=client_id,
                seen_chunk_ids=app_ctx.seen_chunk_ids,
                pinned_chunk_ids=app_ctx.pinned_chunk_ids,
                scratchpad=app_ctx.scratchpad,
            )
        except Exception as exc:  # noqa: BLE001 — defensive shutdown
            log.warning("mcp.sessions.persist_failed", error=str(exc))
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
    # WIN-214 — per-hit provenance. Tells the agent WHICH retrieval
    # streams contributed to this hit so it can self-audit context.
    # e.g. {"dense": 0.0164, "sparse": 0.0162} means dense + sparse
    # both surfaced the chunk via RRF; a symbol-only hit would have
    # {"symbol": <score>} and signal a literal-match.
    streams: dict[str, float] | None = Field(
        default=None,
        description=(
            "WIN-214: per-stream RRF contributions when the hit came "
            "from a fused retrieval. Keys: dense | sparse | symbol | "
            "graph-vec. None for single-stream short-circuit hits "
            "where no fusion happened."
        ),
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


def _persist_session_state(app: AppContext) -> None:
    """Write-through the current AppContext session state to storage.

    WIN-221 / EPIC 1.9 — every mutation of ``seen_chunk_ids`` or
    ``pinned_chunk_ids`` calls here so a process crash never loses
    more than the last call's worth of state. The persistence is
    GATED on ``client_id`` being set: without a stable identifier
    we can't rejoin the row across restarts, so paying the write
    cost serves no purpose.

    Defensive: a write failure (read-only fs, disk full, …) is
    logged but does not propagate — the agent's MCP call must
    succeed even when persistence is degraded.
    """
    if not app.client_id:
        return
    try:
        app.storage.upsert_session(
            session_id=app.session_id,
            client_id=app.client_id,
            seen_chunk_ids=app.seen_chunk_ids,
            pinned_chunk_ids=app.pinned_chunk_ids,
            scratchpad=app.scratchpad,
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        log.warning(
            "mcp.sessions.persist_failed",
            session_id=app.session_id,
            error=str(exc),
        )


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

    # WIN-224: build the conversation context BEFORE the search so the
    # SearchEngine can blend it into the dense embedding. The current
    # query is NOT included in the context (it goes directly to the
    # engine), only the prior turns. We add the current query to the
    # ring buffer AFTER the search so the next call sees this query
    # as part of its context window.
    conversation_context: str | None = None
    if conversation_aware and app.recent_queries:
        # Join the last N queries with " | " separators so the
        # embedder sees them as related-but-secondary signal.
        conversation_context = " | ".join(app.recent_queries)

    try:
        # Handle remote git URL search
        if repo_url and is_git_url(repo_url):
            log.info("mcp.search_code.remote", repo_url=repo_url, refresh=refresh)
            hits = search_git_url(query, repo_url, top_k=fetch_k, refresh=refresh)
        else:
            hits = app.search_engine.search(
                query, top_k=fetch_k,
                conversation_context=conversation_context,
            )
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

    # WIN-217: session-scoped dedup. Filter out chunks we've already
    # served to this agent's session, then accumulate the survivors
    # into the seen set so subsequent calls skip them too. The
    # ``seen_chunk_ids`` set lives on AppContext and is the MVP backing
    # store; EPIC 1.9 (WIN-221) will swap in the persistent Schema v8
    # sessions table without touching this call site.
    #
    # WIN-218 interaction: pinned chunk_ids are EXEMPT from dedup.
    # The agent's pin signal is strictly stronger than "skip
    # already-seen" — a pinned chunk should still be visible even
    # after it was served once.
    dedup_filtered = 0
    if dedup and hits:
        before = len(hits)
        hits = [
            h for h in hits
            if h.chunk_id in app.pinned_chunk_ids
            or h.chunk_id not in app.seen_chunk_ids
        ]
        dedup_filtered = before - len(hits)

    # WIN-218: Prepend pinned chunks at the top of the result list.
    # Resolves each pin via Storage.lookup_by_chunk_id (WIN-216) and
    # splices into the head of ``hits``, deduped by chunk_id so a
    # search result that's also pinned doesn't appear twice. Pinning
    # is always-on (no opt-in flag) — once the agent has pinned, the
    # signal applies until unpinned.
    if app.pinned_chunk_ids:
        from .retrieval.pinning import prepend_pins, resolve_pins

        resolved = resolve_pins(app.storage, app.pinned_chunk_ids)
        pinned_hits = [r.hit for r in resolved if r.found and r.hit is not None]
        if pinned_hits:
            hits = prepend_pins(pinned_hits, hits)

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

    # WIN-213: Relevance decay (per-rank card schedule) wins over
    # card_level when both are set — the agent explicitly asked for
    # graduated detail per rank, which is more specific than a single
    # uniform level.
    if relevance_decay and hits:
        from .retrieval.cards import apply_relevance_decay

        hits = apply_relevance_decay(hits)
    # WIN-212: Apply card-level rendering BEFORE packing so the budget
    # reflects the smaller per-hit cost. "full" is a no-op pass-through.
    elif card_level != "full" and hits:
        from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

        if card_level not in ALLOWED_LEVELS:
            log.warning(
                "mcp.search_code.bad_card_level",
                level=card_level, allowed=list(ALLOWED_LEVELS),
            )
        else:
            # type narrowed by the membership check above
            hits = apply_level_to_snippets(hits, card_level)  # type: ignore[arg-type]

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

    # WIN-224: record the current query into the ring buffer so the
    # NEXT conversation-aware call can see it as context. We append
    # unconditionally (regardless of ``conversation_aware``) so the
    # operator who toggles the flag on mid-session immediately gets
    # useful context; the deque's maxlen caps memory growth.
    if query and query.strip():
        app.recent_queries.append(query.strip())

    # WIN-217: After every filter/pack pass, accumulate the final
    # response's chunk_ids into the session's seen set so subsequent
    # `dedup=True` calls skip them. We only record when the caller
    # opted in to dedup — opting out should be transparent.
    #
    # WIN-218 interaction: pinned chunks are NOT added to the seen
    # set. They appear in results because of the explicit pin, not
    # because the agent organically asked for them — treating them
    # as "seen" would make a future unpin + dedup'd search drop
    # them, which is the opposite of what the agent wants.
    if dedup and hits:
        new_seen = {
            h.chunk_id for h in hits
            if h.chunk_id not in app.pinned_chunk_ids
        }
        if new_seen:
            app.seen_chunk_ids.update(new_seen)
            # WIN-221: persist to Schema-v8 sessions row when the
            # agent supplied a stable client_id. Without one, the
            # session is ephemeral and persistence would be noise.
            _persist_session_state(app)
        log.info(
            "mcp.search_code.dedup",
            filtered_before_pack=dedup_filtered,
            served=len(hits),
            session_seen_total=len(app.seen_chunk_ids),
            pinned=len(app.pinned_chunk_ids),
        )

    return SearchResponse(
        results=[_to_result(h) for h in hits],
        total_candidates=len(hits),
        query=query,
        facets=facets,
        spilled_count=spilled_count,
        truncated_oversize=truncated_oversize,
    )


# ---- reset_dedup (WIN-217) -------------------------------------------------


class ResetDedupResponse(BaseModel):
    """Outcome of clearing the session's seen-set."""

    cleared: int = Field(
        description="Number of chunk_ids removed from the seen set.",
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
    app = _ctx(ctx)
    count = len(app.seen_chunk_ids)
    app.seen_chunk_ids.clear()
    _persist_session_state(app)
    log.info("mcp.reset_dedup", cleared=count, session_id=app.session_id)
    return ResetDedupResponse(cleared=count)


# ---- pin_chunks / unpin_chunks / list_pins (WIN-218) ------------------------


class PinResponse(BaseModel):
    """Outcome of a pin/unpin mutation."""

    pinned: list[str] = Field(
        description="Chunk ids currently pinned after this operation.",
    )
    added: int = Field(
        default=0,
        description="How many ids the call added to the pin set.",
    )
    removed: int = Field(
        default=0,
        description="How many ids the call removed from the pin set.",
    )


class PinListEntry(BaseModel):
    """One entry in the ``list_pins`` response."""

    chunk_id: str
    found: bool = Field(
        description=(
            "True when the pinned id resolved to a stored chunk. "
            "False usually means the index was rebuilt since the "
            "pin was set."
        ),
    )
    result: SearchResult | None = Field(
        default=None,
        description="Resolved chunk metadata when found=True.",
    )


class ListPinsResponse(BaseModel):
    """Response payload for ``list_pins``."""

    entries: list[PinListEntry]
    found_count: int
    missing_count: int


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
    app = _ctx(ctx)
    before = len(app.pinned_chunk_ids)
    app.pinned_chunk_ids.update(chunk_ids)
    added = len(app.pinned_chunk_ids) - before
    if added:
        _persist_session_state(app)
    log.info(
        "mcp.pin_chunks",
        added=added,
        total=len(app.pinned_chunk_ids),
        session_id=app.session_id,
    )
    return PinResponse(
        pinned=sorted(app.pinned_chunk_ids),
        added=added,
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
    app = _ctx(ctx)
    before = len(app.pinned_chunk_ids)
    app.pinned_chunk_ids.difference_update(chunk_ids)
    removed = before - len(app.pinned_chunk_ids)
    if removed:
        _persist_session_state(app)
    log.info(
        "mcp.unpin_chunks",
        removed=removed,
        total=len(app.pinned_chunk_ids),
        session_id=app.session_id,
    )
    return PinResponse(
        pinned=sorted(app.pinned_chunk_ids),
        removed=removed,
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
    from .retrieval.pinning import resolve_pins

    app = _ctx(ctx)
    resolved = resolve_pins(app.storage, app.pinned_chunk_ids)
    entries: list[PinListEntry] = []
    found = 0
    for r in resolved:
        if r.found and r.hit is not None:
            entries.append(
                PinListEntry(
                    chunk_id=r.chunk_id, found=True,
                    result=_to_result(r.hit),
                )
            )
            found += 1
        else:
            entries.append(
                PinListEntry(chunk_id=r.chunk_id, found=False, result=None)
            )
    return ListPinsResponse(
        entries=entries,
        found_count=found,
        missing_count=len(entries) - found,
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
        # WIN-234: pass the configured router stage so the dispatch
        # uses the operator's chosen policy. Falls back to the
        # legacy ``route_query`` when no router was built.
        configured_router = getattr(app.built, "router", None)
        result = smart_search(
            query,
            app.storage,
            app.embedder,
            top_k=top_k,
            use_router=True,
            use_llm_router=use_llm_router,
            router=configured_router,
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


# ---- expand_chunk(s) — just-in-time dereferencing (WIN-216) ----------------


class ExpandChunkResponse(BaseModel):
    """Single ``expand_chunk`` payload: either the resolved hit, or
    a not-found marker."""

    chunk_id: str
    found: bool = Field(
        description="True when the chunk_id resolved to a stored chunk.",
    )
    result: SearchResult | None = Field(
        default=None,
        description="The resolved chunk as a SearchResult, or null when missing.",
    )


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
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    app = _ctx(ctx)
    # Storage.get_chunk_by_id returns the raw chunk dict; we convert it
    # to a SearchHit-shaped dict via the same path lookup_by_symbol uses.
    hit = app.storage.lookup_by_chunk_id(chunk_id)
    if hit is None:
        return ExpandChunkResponse(chunk_id=chunk_id, found=False, result=None)
    if card_level != "full" and card_level in ALLOWED_LEVELS:
        hits = apply_level_to_snippets([hit], card_level)  # type: ignore[arg-type]
        hit = hits[0]
    return ExpandChunkResponse(
        chunk_id=chunk_id,
        found=True,
        result=_to_result(hit),
    )


class ExpandChunksResponse(BaseModel):
    """Batch ``expand_chunks`` payload."""

    results: list[ExpandChunkResponse]
    found_count: int = Field(
        description="How many chunk_ids resolved successfully.",
    )
    missing_count: int = Field(
        description="How many chunk_ids could not be resolved.",
    )


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
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    app = _ctx(ctx)
    results: list[ExpandChunkResponse] = []
    found = 0
    for cid in chunk_ids:
        hit = app.storage.lookup_by_chunk_id(cid)
        if hit is None:
            results.append(
                ExpandChunkResponse(chunk_id=cid, found=False, result=None)
            )
            continue
        if card_level != "full" and card_level in ALLOWED_LEVELS:
            hits = apply_level_to_snippets([hit], card_level)  # type: ignore[arg-type]
            hit = hits[0]
        results.append(
            ExpandChunkResponse(
                chunk_id=cid, found=True, result=_to_result(hit),
            )
        )
        found += 1
    return ExpandChunksResponse(
        results=results,
        found_count=found,
        missing_count=len(chunk_ids) - found,
    )


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


def _resolve_target_repo_id(app: AppContext) -> int | None:
    """Resolve the configured target repo's ``repos.id`` for memory tools.

    Project memory is repo-scoped (Schema v9 FK). We derive the
    repo_id by detecting the target_repo's normalized_key and looking
    it up. Returns ``None`` when the repo has never been indexed
    (no row to anchor against).
    """
    from .ingest.git_meta import detect as _detect_repo

    if app.config.target_repo is None:
        return None
    try:
        identity = _detect_repo(app.config.target_repo)
    except Exception as exc:  # noqa: BLE001 — defensive
        log.warning(
            "mcp.memory.detect_repo_failed", error=str(exc),
        )
        return None
    return app.storage.lookup_repo_id(identity.normalized_key)


def _current_head_commit(app: AppContext) -> str | None:
    """WIN-228 — fetch the target_repo's HEAD SHA for provenance.

    Returns ``None`` on any git failure or when no target_repo is
    configured. The memory write succeeds either way; provenance is
    a hint, not a contract.
    """
    if app.config.target_repo is None:
        return None
    from .meta.provenance import current_head_commit

    return current_head_commit(app.config.target_repo)


class MemoryEntry(BaseModel):
    """One ``project_memory`` row exposed to MCP clients."""

    kind: str
    key: str
    value: str
    source: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None
    provenance_commit: str | None = Field(
        default=None,
        description=(
            "Git HEAD SHA at the time the entry was last upserted "
            "(WIN-228). Drives the check_staleness tool."
        ),
    )


class RememberResponse(BaseModel):
    """Outcome of a ``remember`` call."""

    ok: bool
    entry: MemoryEntry | None = None
    error: str | None = Field(
        default=None,
        description=(
            "Populated when ok=False — typically 'no indexed repo' "
            "when the agent calls remember before any index pass has "
            "registered the target repo."
        ),
    )


class RecallResponse(BaseModel):
    """Outcome of a ``recall`` call."""

    entries: list[MemoryEntry]
    repo: str | None = Field(
        default=None,
        description="normalized_key of the repo the entries belong to.",
    )


class ForgetResponse(BaseModel):
    """Outcome of a ``forget`` call."""

    removed: bool
    kind: str
    key: str


def _entry_from_record(rec) -> MemoryEntry:  # type: ignore[no-untyped-def]
    return MemoryEntry(
        kind=rec.kind,
        key=rec.key,
        value=rec.value,
        source=rec.source,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        expires_at=rec.expires_at,
        provenance_commit=getattr(rec, "provenance_commit", None),
    )


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
) -> RememberResponse:
    """Upsert a project-memory entry for the configured target repo."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return RememberResponse(
            ok=False,
            error=(
                "no indexed repo for target_repo — run `stropha index` "
                "first so the memory entry has a repo to anchor against"
            ),
        )
    rec = app.storage.remember(
        repo_id=repo_id,
        kind=kind,
        key=key,
        value=value,
        source=app.session_id or None,
        expires_at=expires_at,
        provenance_commit=_current_head_commit(app),
    )
    return RememberResponse(ok=True, entry=_entry_from_record(rec))


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
) -> RecallResponse:
    """Fetch project-memory entries for the configured target repo."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return RecallResponse(entries=[], repo=None)
    rows = app.storage.recall(
        repo_id=repo_id,
        kind=kind,
        key=key,
        limit=max(1, min(limit, 200)),
        include_expired=include_expired,
    )
    # Best-effort surface of the repo's normalized_key so the agent
    # can confirm "this is the right project".
    repo_key: str | None = None
    if rows:
        try:
            row = app.storage._conn.execute(
                "SELECT normalized_key FROM repos WHERE id = ?",
                (repo_id,),
            ).fetchone()
            repo_key = row["normalized_key"] if row else None
        except Exception:
            repo_key = None
    return RecallResponse(
        entries=[_entry_from_record(r) for r in rows],
        repo=repo_key,
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
def forget(kind: str, key: str, *, ctx: Context) -> ForgetResponse:
    """Delete one project-memory entry."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return ForgetResponse(removed=False, kind=kind, key=key)
    removed = app.storage.forget(repo_id=repo_id, kind=kind, key=key)
    return ForgetResponse(removed=removed, kind=kind, key=key)


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


class CrossSessionCacheStats(BaseModel):
    """Cache counters returned by ``cache_stats``."""

    total_entries: int
    total_hits: int
    distinct_models: int
    repo: str | None = None


class CrossSessionCachePruneResponse(BaseModel):
    """Outcome of a ``cache_prune`` call."""

    removed: int


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
    from .retrieval.cross_session_cache import stats

    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return CrossSessionCacheStats(
            total_entries=0, total_hits=0, distinct_models=0,
            repo=None,
        )
    counters = stats(app.storage, repo_id=repo_id)
    return CrossSessionCacheStats(
        total_entries=counters["total_entries"],
        total_hits=counters["total_hits"],
        distinct_models=counters["distinct_models"],
        repo=None,  # could resolve normalized_key if needed
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
    from .retrieval.cross_session_cache import prune

    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return CrossSessionCachePruneResponse(removed=0)
    removed = prune(
        app.storage,
        repo_id=repo_id,
        expired_only=expired_only,
        older_than_iso=older_than_iso,
    )
    return CrossSessionCachePruneResponse(removed=removed)


# ---- scratchpad (WIN-229, EPIC 1.9 #21) ------------------------------------


class ScratchpadResponse(BaseModel):
    """Outcome of any scratchpad mutation. Returns the full
    post-mutation content so the agent can verify the write."""

    content: str
    char_count: int


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
    app = _ctx(ctx)
    return ScratchpadResponse(
        content=app.scratchpad,
        char_count=len(app.scratchpad),
    )


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
    app = _ctx(ctx)
    app.scratchpad = content
    _persist_session_state(app)
    return ScratchpadResponse(
        content=app.scratchpad,
        char_count=len(app.scratchpad),
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
    app = _ctx(ctx)
    if not text:
        # Don't dirty the persistence path for a no-op call.
        return ScratchpadResponse(
            content=app.scratchpad,
            char_count=len(app.scratchpad),
        )
    if app.scratchpad and not app.scratchpad.endswith("\n"):
        app.scratchpad = app.scratchpad + "\n" + text
    else:
        app.scratchpad = app.scratchpad + text
    _persist_session_state(app)
    return ScratchpadResponse(
        content=app.scratchpad,
        char_count=len(app.scratchpad),
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
    app = _ctx(ctx)
    app.scratchpad = ""
    _persist_session_state(app)
    return ScratchpadResponse(content="", char_count=0)


# ---- staleness check (WIN-228, EPIC 1.9 #20) -------------------------------


class StalenessResponse(BaseModel):
    """Result of a ``check_staleness`` call."""

    found: bool = Field(
        description=(
            "True when the requested memory entry exists. False means "
            "no entry under that (kind, key) — caller should treat as "
            "missing, not stale."
        ),
    )
    stale: bool | None = Field(
        default=None,
        description=(
            "True when files changed between the recorded commit and "
            "HEAD intersect the memory's anchor. False when nothing "
            "changed. None means we couldn't decide (missing "
            "provenance, git failure, detached HEAD)."
        ),
    )
    since_commit: str | None = None
    current_commit: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    reason: str | None = Field(
        default=None,
        description=(
            "When stale is None, the textual reason for the unknown "
            "verdict (no provenance, git failure, etc.)."
        ),
    )


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
) -> StalenessResponse:
    """Determine if the named ``project_memory`` entry is stale."""
    from .meta.provenance import staleness_signal

    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return StalenessResponse(
            found=False,
            reason="no indexed repo for target_repo",
        )
    rows = app.storage.recall(repo_id=repo_id, kind=kind, key=key, limit=1)
    if not rows:
        return StalenessResponse(found=False, reason="entry not found")
    entry = rows[0]
    if app.config.target_repo is None:
        return StalenessResponse(
            found=True,
            stale=None,
            since_commit=entry.provenance_commit,
            reason="no target_repo configured",
        )
    signal = staleness_signal(
        app.config.target_repo,
        since_commit=entry.provenance_commit,
        anchor_files=anchor_files,
    )
    return StalenessResponse(
        found=True,
        stale=signal.stale,
        since_commit=signal.since_commit,
        current_commit=signal.current_commit,
        changed_files=signal.changed_files,
        reason=signal.reason,
    )


# ---- bootstrap (WIN-227, EPIC 1.9 #19) -------------------------------------


class DetectedStackEntry(BaseModel):
    """One stack from the bootstrap snapshot exposed to MCP clients."""

    name: str
    language: str
    build_command: str | None = None
    test_command: str | None = None
    run_command: str | None = None
    matched_markers: list[str] = Field(default_factory=list)


class BootstrapResponse(BaseModel):
    """Bootstrap payload — warm-start view of the target repo."""

    repo_path: str
    stacks: list[DetectedStackEntry] = Field(default_factory=list)
    entry_points: dict[str, list[str]] = Field(default_factory=dict)
    has_git: bool = False
    has_tests_dir: bool = False
    has_docs_dir: bool = False
    has_graphify_out: bool = False


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
    from pathlib import Path as _Path

    from .retrieval.bootstrap import detect_bootstrap

    app = _ctx(ctx)
    target = app.config.target_repo
    if target is None:
        target = _Path.cwd()
    snapshot = detect_bootstrap(_Path(target))
    return BootstrapResponse(
        repo_path=snapshot.repo_path,
        stacks=[
            DetectedStackEntry(
                name=s.name,
                language=s.language,
                build_command=s.build_command,
                test_command=s.test_command,
                run_command=s.run_command,
                matched_markers=list(s.matched_markers),
            )
            for s in snapshot.stacks
        ],
        entry_points=snapshot.entry_points,
        has_git=snapshot.has_git,
        has_tests_dir=snapshot.has_tests_dir,
        has_docs_dir=snapshot.has_docs_dir,
        has_graphify_out=snapshot.has_graphify_out,
    )


# ---- episodic memory (WIN-225, EPIC 1.9 #17) -------------------------------


class EpisodeEntry(BaseModel):
    """One ``episodes`` row exposed to MCP clients."""

    id: int
    title: str
    summary: str = ""
    session_id: str | None = None
    touched_files: list[str] = Field(default_factory=list)
    touched_symbols: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    provenance_commit: str | None = Field(
        default=None,
        description=(
            "Git HEAD SHA at the time the episode was recorded "
            "(WIN-228). Drives the check_staleness tool."
        ),
    )


class RecordEpisodeResponse(BaseModel):
    """Outcome of a ``record_episode`` call."""

    ok: bool
    entry: EpisodeEntry | None = None
    error: str | None = None


class RecallEpisodesResponse(BaseModel):
    """Outcome of a ``recall_episodes`` call."""

    entries: list[EpisodeEntry]
    repo: str | None = None


class CoOccurrenceResponse(BaseModel):
    """Outcome of an ``episode_co_occurrence`` call."""

    key: str
    files: dict[str, int] = Field(default_factory=dict)
    symbols: dict[str, int] = Field(default_factory=dict)
    repo: str | None = None


def _episode_to_entry(rec) -> EpisodeEntry:  # type: ignore[no-untyped-def]
    return EpisodeEntry(
        id=rec.id,
        title=rec.title,
        summary=rec.summary,
        session_id=rec.session_id,
        touched_files=rec.touched_files,
        touched_symbols=rec.touched_symbols,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        provenance_commit=getattr(rec, "provenance_commit", None),
    )


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
) -> RecordEpisodeResponse:
    """Record a single episodic-memory row for the target repo."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return RecordEpisodeResponse(
            ok=False,
            error=(
                "no indexed repo for target_repo — run `stropha index` "
                "first so the episode has a repo to anchor against"
            ),
        )
    rec = app.storage.record_episode(
        repo_id=repo_id,
        session_id=app.session_id or None,
        title=title,
        summary=summary,
        touched_files=touched_files or [],
        touched_symbols=touched_symbols or [],
        provenance_commit=_current_head_commit(app),
    )
    return RecordEpisodeResponse(ok=True, entry=_episode_to_entry(rec))


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
) -> RecallEpisodesResponse:
    """Search past episodes by free-text + optional time window."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return RecallEpisodesResponse(entries=[], repo=None)
    rows = app.storage.recall_episodes(
        repo_id=repo_id,
        query=query,
        limit=max(1, min(limit, 200)),
        since_iso=since,
    )
    repo_key: str | None = None
    if rows:
        try:
            r = app.storage._conn.execute(
                "SELECT normalized_key FROM repos WHERE id = ?",
                (repo_id,),
            ).fetchone()
            repo_key = r["normalized_key"] if r else None
        except Exception:
            repo_key = None
    return RecallEpisodesResponse(
        entries=[_episode_to_entry(r) for r in rows],
        repo=repo_key,
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
) -> CoOccurrenceResponse:
    """Rank co-touched files / symbols for the given key."""
    app = _ctx(ctx)
    repo_id = _resolve_target_repo_id(app)
    if repo_id is None:
        return CoOccurrenceResponse(key=key, files={}, symbols={}, repo=None)
    result = app.storage.episode_co_occurrence(
        repo_id=repo_id,
        key=key,
        in_files=in_files,
        in_symbols=in_symbols,
        limit=max(1, min(limit, 200)),
    )
    repo_key: str | None = None
    try:
        r = app.storage._conn.execute(
            "SELECT normalized_key FROM repos WHERE id = ?", (repo_id,),
        ).fetchone()
        repo_key = r["normalized_key"] if r else None
    except Exception:
        repo_key = None
    return CoOccurrenceResponse(
        key=key,
        files=result.get("files", {}),
        symbols=result.get("symbols", {}),
        repo=repo_key,
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
    if chunk_ids:
        ids = list(chunk_ids)
    else:
        ids = list(app.seen_chunk_ids)[-10:]

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


# ---- entry point -----------------------------------------------------------

def main() -> None:
    """Run the MCP server over stdio. Used by `[project.scripts]` console entry."""
    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
