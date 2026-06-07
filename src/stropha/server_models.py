"""Pydantic response models for the MCP server surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RepoInfo(BaseModel):
    """Source-repo identity attached to every search result."""

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
    kind: str = Field(description="file | class | method | function | section | scenario | ...")
    symbol: str | None = None
    snippet: str = Field(description="Truncated preview. Use Read tool for full content.")
    chunk_id: str
    repo: RepoInfo | None = Field(
        default=None,
        description="Source repository identity (URL, branch, HEAD).",
    )
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
    trace_id: str | None = Field(
        default=None,
        description=(
            "Trace id for this call (WIN-160). Pass it to report_bad_result "
            "to flag bad / duplicate / low-signal ranking — that's how the "
            "Meta-Harness feedback loop learns. Auto-stamped by the @traced "
            "decorator; matches the JSONL trace record for this search."
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
    """One entry in the `list_repos` response."""

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


class ResetDedupResponse(BaseModel):
    """Outcome of clearing the session's seen-set."""

    cleared: int = Field(description="Number of chunk_ids removed from the seen set.")


class PinResponse(BaseModel):
    """Outcome of a pin/unpin mutation."""

    pinned: list[str] = Field(description="Chunk ids currently pinned after this operation.")
    added: int = Field(default=0, description="How many ids the call added to the pin set.")
    removed: int = Field(default=0, description="How many ids the call removed from the pin set.")


class PinListEntry(BaseModel):
    """One entry in the ``list_pins`` response."""

    chunk_id: str
    found: bool = Field(
        description=(
            "True when the pinned id resolved to a stored chunk. "
            "False usually means the index was rebuilt since the pin was set."
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
    trace_id: str | None = Field(
        default=None,
        description=(
            "Trace id for this call (WIN-160). Pass it to report_bad_result "
            "to flag bad/duplicate/low-signal ranking. Auto-stamped by @traced."
        ),
    )


class ExpandChunkResponse(BaseModel):
    """Single ``expand_chunk`` payload."""

    chunk_id: str
    found: bool = Field(description="True when the chunk_id resolved to a stored chunk.")
    result: SearchResult | None = Field(
        default=None,
        description="The resolved chunk as a SearchResult, or null when missing.",
    )


class ExpandChunksResponse(BaseModel):
    """Batch ``expand_chunks`` payload."""

    results: list[ExpandChunkResponse]
    found_count: int = Field(description="How many chunk_ids resolved successfully.")
    missing_count: int = Field(description="How many chunk_ids could not be resolved.")


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


class CrossSessionCacheStats(BaseModel):
    """Cache counters returned by ``cache_stats``."""

    total_entries: int
    total_hits: int
    distinct_models: int
    repo: str | None = None


class CrossSessionCachePruneResponse(BaseModel):
    """Outcome of a ``cache_prune`` call."""

    removed: int


class ScratchpadResponse(BaseModel):
    """Outcome of any scratchpad mutation."""

    content: str
    char_count: int


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


class VerdictResponse(BaseModel):
    """Outcome of a ``verify_claim`` call."""

    raw_text: str = Field(description="Echo of the claim as received.")
    parsed: bool = Field(
        description=(
            "Whether the claim matched one of the supported "
            "predicates (calls / called by / uses / references / "
            "implements / is a <kind> / is in <file> / has tests)."
        ),
    )
    subject: str = Field(default="")
    predicate: str = Field(default="")
    object: str = Field(default="")
    verified: bool | None = Field(
        default=None,
        description=(
            "Tri-state: True (graph confirms) / False (graph "
            "contradicts) / null (graph unavailable, unknown "
            "predicate, or subject not resolvable)."
        ),
    )
    reason: str = Field(default="")
    evidence: list[dict] = Field(default_factory=list)
    alternatives: list[dict] = Field(default_factory=list)


class SymbolCheckEntry(BaseModel):
    """Per-symbol outcome from ``check_symbols``."""

    symbol: str = Field(description="Echo of the input name.")
    exists: bool = Field(description="True when the index resolves the symbol.")
    suggestions: list[str] = Field(default_factory=list)
    reason: str = Field(default="")


class HallucinationGuardResponse(BaseModel):
    """Full report from a ``check_symbols`` call."""

    checks: list[SymbolCheckEntry]
    present_count: int = Field(description="How many input symbols resolved in the index.")
    missing_count: int = Field(description="How many input symbols failed to resolve.")
    notes: list[str] = Field(default_factory=list)


class PerSymbolCounts(BaseModel):
    callers: int = 0
    tests: int = 0
    related: int = 0
    rationale: int = 0


class BlastRadiusResponse(BaseModel):
    """Aggregated blast-radius report from ``blast_radius_tool``."""

    changed_symbols: list[str] = Field(default_factory=list)
    callers: list[dict] = Field(default_factory=list)
    tests: list[dict] = Field(default_factory=list)
    related: list[dict] = Field(default_factory=list)
    rationale: list[dict] = Field(default_factory=list)
    per_symbol: dict[str, PerSymbolCounts] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    graph_loaded: bool = Field(description="False when the graphify mirror is unavailable.")


class FreshnessResponse(BaseModel):
    """Outcome of ``assert_snippet_freshness_tool``."""

    chunk_id: str
    fresh: bool | None = Field(default=None)
    reason: str = Field(default="")
    current_snippet: str | None = Field(default=None)
    expected_snippet: str = Field(default="")
    drift_details: list[str] = Field(default_factory=list)


class ImpactedTestEntry(BaseModel):
    path: str
    symbol: str = ""
    via: list[str] = Field(default_factory=list)


class TestImpactResponse(BaseModel):
    """Outcome of ``test_impact_tool``."""

    changed_symbols: list[str] = Field(default_factory=list)
    tests: list[ImpactedTestEntry] = Field(default_factory=list)
    per_symbol: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    run_command: str = Field(default="")
    graph_loaded: bool = Field(description="False when the graphify mirror is unavailable.")


class CitationEntry(BaseModel):
    fragment: str
    kind: str = Field(description="One of 'chunk_id', 'file_line', 'unknown'.")
    chunk_id: str = ""
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    resolved: bool | None = Field(default=None)
    reason: str = ""


class ClaimAuditEntry(BaseModel):
    sentence: str
    citations: list[CitationEntry]
    valid_count: int


class CitationAuditResponse(BaseModel):
    """Outcome of ``audit_citations_tool``."""

    claims: list[ClaimAuditEntry]
    total_claims: int = Field(description="Total sentences extracted.")
    cited_claims: int = Field(description="Sentences carrying at least one citation token.")
    uncited_claims: int = Field(description="Sentences with no citation at all.")
    broken_citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SpecInvariantEntry(BaseModel):
    text: str
    spec_file: str
    line: int
    keyword: str
    severity: str = Field(description="HIGH / MEDIUM / LOW.")


class SpecConflictEntry(BaseModel):
    invariant: SpecInvariantEntry
    matched: str = Field(description="The changed symbol / file the invariant mentions.")
    severity: str
    reason: str


class SpecConsistencyResponse(BaseModel):
    """Outcome of ``spec_consistency_tool``."""

    invariants_found: int
    conflicts: list[SpecConflictEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    spec_files_scanned: list[str] = Field(default_factory=list)


class ContractTestEntry(BaseModel):
    path: str
    symbol: str = ""
    signals: list[str] = Field(default_factory=list)


class ContractTestsResponse(BaseModel):
    """Outcome of ``contract_tests_tool``."""

    target_symbol: str
    contract_tests: list[ContractTestEntry] = Field(default_factory=list)
    other_tests: list[ContractTestEntry] = Field(default_factory=list)
    signal_counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    graph_loaded: bool = Field(description="False when the graphify mirror is unavailable.")
