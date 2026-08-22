# Stropha Agent Exoskeleton

> **Status:** Phases 0-3 implemented. Agent observations and runtime artifacts
> are bounded, ephemeral, and isolated by MCP session.

## Purpose

The exoskeleton is stropha's local evidence layer for coding agents. It keeps a
versioned view of saved source, exposes bounded retrieval and structural tools,
and lets clients choose a compact static MCP surface without creating a
different server process per client.

It does not generate code, ingest unsaved editor buffers, or accept arbitrary
per-session tool registration. Hosts may report editor metadata and runtime
evidence, but Stropha never becomes the agent runtime.

## Source Evidence

The persistent index represents saved working-tree bytes, including uncommitted
edits. `HEAD` is provenance metadata, not proof of indexed content. Phase 1 made
that contract explicit through:

- full raw-byte SHA-256 identities and immutable chunker input;
- adapter-owned atomic replace/delete operations with rollback safety;
- removal of obsolete chunks when files shrink, rechunk, or become empty;
- completion-only repository watermarks and durable graph-dirty state;
- startup watcher reconciliation and retry backoff;
- source-version metadata on every retrieval stream;
- independent primary and Model2Vec vector spaces.

Index-only results carry the indexed file hash. Tools that reread a saved source
slice can additionally classify it as `fresh`, `stale`, or `missing`.

## Compact MCP Surface

### Canonical Manifest

`src/stropha/server_tool_manifest.py` is the only canonical tool inventory. Each
immutable entry records a tool name and its minimum profile. Registration order
must exactly match manifest order; startup fails on missing, stale, reordered,
or duplicate entries.

Every registered tool has an output schema. Suggestion generation filters out
tools hidden by the active profile, and task recipes validate generated
arguments against the referenced tool schemas.

### Static Profiles

Profiles are nested and apply to tools only. Resources and prompts remain
unchanged.

| Profile | Contract |
|---|---|
| `core` | Small retrieval and orientation surface; 13 tools |
| `standard` | `core` plus observations, session ergonomics, investigations, and change intelligence; 40 tools |
| `full` | Complete compatibility surface; 55 tools including `capabilities` |

`core` contains:

`capabilities`, `select_repository`, `search_code`, `get_symbol`, `list_repos`,
`get_file_outline`, `bootstrap_environment`, `find_callers`, `find_related`,
`find_rationale`, `find_tests_for`, `assemble_context`, and `index_health`.

`standard` adds:

`observe_agent_state`, `record_runtime_artifact`, `resolve_agent_instructions`,
`reset_dedup`, `pin_chunks`, `unpin_chunks`, `list_pins`, `smart_search_tool`,
`expand_chunk`,
`expand_chunks`, `plan_investigation`, `get_community`, `trace_feature`,
`assemble_agent_context`, `execute_investigation`, `suggest_next_tool_mcp`, `list_task_recipes`,
`apply_task_recipe`, `verify_claim_tool`, `check_symbols_tool`,
`blast_radius_tool`, `assert_snippet_freshness_tool`, `test_impact_tool`,
`audit_citations_tool`, `spec_consistency_tool`, `contract_tests_tool`, and
`working_set_tool`.

`full` adds:

`remember`, `recall`, `forget`, `cache_stats`, `cache_prune`, `scratchpad_read`,
`scratchpad_write`, `scratchpad_append`, `scratchpad_clear`, `check_staleness`,
`record_episode`, `recall_episodes`, `co_occurrence`, `report_bad_result`, and
`get_config`.

### Discovery And Invocation

`tools/list` returns only manifest entries allowed by the active profile, in
manifest order. `tools/call` applies the same policy and returns an explicit MCP
tool error for hidden names. `capabilities` is always visible and returns the
active profile, accepted profiles, tool count, ordered names, and the fact that
the profile is static.

The `core` schema contract is tested against two budgets: at most 40 KiB for the
serialized `tools/list` response and at least a 50 percent reduction from the
99,451-character pre-profile baseline.

### Daemon Lifecycle

The only setting is `mcp.profile` in `~/.stropha/config.yaml`; accepted values
are `core`, `standard`, and `full`. Omission selects `full` so existing clients
retain the complete pre-profile surface plus `capabilities`.

The launcher reads the profile before starting streamable HTTP. It is shared by
all clients and fixed for the daemon lifetime. There is intentionally no CLI,
environment, client, or session override. A dashboard patch can save a new
value, but reports `restart_required: true` and leaves the active surface
unchanged until `bin/stropha-mcp-ensure` is restarted.

## Ephemeral Agent Observations

`observe_agent_state` accepts a complete `AgentStateSnapshotV1` or, with no
snapshot argument, returns the current session view. Every accepted update must
have a revision greater than the current revision and replaces the prior
snapshot. The contract carries the canonical repository key, active and recent
files, optional line selection, changed paths, referenced runtime artifact IDs,
and an offset-aware observation timestamp.

The validated snapshot repository becomes the default scope for repository-local
tools in that MCP session. An explicit tool repository selector still wins. A
session without a selection or snapshot retains the daemon target as a
compatibility fallback; session scope never changes the daemon-wide watcher or
pipeline target.

`select_repository` provides the same session-local default with a compact schema
available in `core`. It accepts an exact indexed normalized key or absolute local
path and is cleared on reconnect. Standard and full hosts may use the richer
snapshot instead; its repository updates the same selection.

`record_runtime_artifact` accepts diagnostics, test failures, stack frames, and
log excerpts as either `text/plain` or `application/json`. Every artifact has an
immutable ID, a host producer, server-constrained `host_reported` trust, source
locations, redaction status, and an offset-aware expiry. A session may hold at
most 32 KiB of UTF-8 content per artifact and 128 KiB total, and may accept at
most 16 unique artifact IDs over its lifetime. Expired artifact content is
removed on the next observation or record call, but its ID cannot be rebound in
that session.

All paths are canonical repository-relative paths resolved against an indexed
local repository root. Absolute paths, parent traversal, inaccessible roots,
unknown repositories, and symlink escapes are rejected. Snapshots and artifacts
are revalidated on readback so a path cannot become trusted after a symlink
change. They live only in `AppContext`; they are not written to session metadata,
memory, the index, artifact storage, or traces. Session shutdown clears both
collections and artifact-ID tombstones, so a reconnect starts empty and the host
must resend its state.

## Instruction And Change Context

`resolve_agent_instructions` reads saved-worktree `AGENTS.md` and `CLAUDE.md`
files that apply to one repository-relative active path. It checks only the
repository root and each ancestor directory through the active file's directory;
unrelated subtrees are never scanned. Results are ordered from general to
specific, with `AGENTS.md` before `CLAUDE.md` at the same scope. Root sources
apply repository-wide and nested sources apply only within their subtree.

The tool defaults to the session repository and observed active file. Explicit
repository and path selectors win. Observed `changed_paths` are returned only
when the snapshot belongs to the resolved repository, ready for the existing
blast-radius, test-impact, and spec-consistency tools.

Every source carries its repository-relative path, scope depth, SHA-256 over the
complete saved bytes, full and included character counts, and truncation state.
Sources share a deterministic fair character budget (16,000 by default, 32,768
maximum). Any individual source above 256 KiB or applicable source set above 1
MiB or 64 files is rejected before unbounded allocation. Instruction files must
be UTF-8 regular files opened through no-follow directory descriptors beneath
the indexed repository root; Windows validates normalized root and file handles.
Absolute active paths, traversal, inaccessible roots, symlink escapes, and
path-replacement races fail explicitly.

This is evidence resolution, not prompt construction: stropha neither interprets
nor executes instruction content. It does not read unsaved buffers, persist
resolved content, write the index, enter tracing, or alter session observations.
Unsaved overlay ingestion remains a Phase 8 concern; bounded context assembly is
defined in the Phase 5 slice below.

## Bounded Agent Context Assembly

`assemble_agent_context` is the first bounded assembly slice for the agent
exoskeleton. It resolves the applicable saved-worktree instructions first, then
passes that response's canonical repository key explicitly into the existing
`assemble_context` retrieval pipeline. One repository therefore scopes both
sections even if session state changes between the two operations.

Instruction content and retrieved snippets retain independent budgets: 4,000
characters and 4,000 estimated tokens by default, with hard maxima of 32,768
characters and 32,000 tokens. The existing retrieval packer's oversized-first-hit
escape remains unchanged for its current callers; this composition omits that hit
when it exceeds the declared retrieval budget, reports the decision through
`retrieval_oversize_omitted`, and still returns any applicable instructions.
`retrieval.spilled_count` remains reserved for hits that fell off the tail.

The response preserves bounded structured inputs inside a strict outer model and
adds deterministic markdown ordered as applicable instructions followed by
retrieved evidence. It renders structured retrieval hits with content-sized
fences rather than exposing the older raw retrieval bundle in either nested or
combined Markdown.
Instruction sources retain path, scope, SHA-256, and truncation provenance;
retrieval retains chunk provenance, graph status, notes, and its retrieval-only
trace ID. The outer composition is deliberately not traced, so instruction
content never enters the trace stream.

This slice does not interpret instructions or change retrieval ranking/packing.
It excludes runtime artifacts, pinned/recent working-set fusion, unsaved buffers,
LLM routing, persistence, migrations, and index writes. Other documents in this
repository use "Phase 5" for unrelated historical initiatives; this phase number
belongs specifically to the agent-exoskeleton sequence in this document.

## Phase Boundary

Phases delivered so far:

- **Phase 0:** baseline hardening and green repository contracts.
- **Phase 1:** trustworthy saved-source snapshots.
- **Phase 2:** canonical compact MCP profiles and structured tool contracts.
- **Phase 3:** ephemeral agent observations and bounded runtime artifacts.
- **Phase 4:** deterministic saved-worktree instruction and change-context resolution.
- **Phase 5:** bounded instruction and retrieved-evidence assembly.

Runtime-artifact and working-set fusion remain future assembly slices.
Unsaved-buffer overlays remain deferred to Phase 8.

## Non-Negotiable Invariants

- Streamable HTTP is the only MCP transport.
- One daemon owns one warm pipeline/storage runtime.
- Heavy adapters are shared; dedup, pins, scratchpad, recent queries, traces,
  agent observations, and runtime artifacts remain session-local.
- Graph-gated tools explicitly report unavailable graph data.
- Source and index writes remain adapter-owned and forward-only.
- Local-only operation remains the default.

See `CLAUDE.md` for the operational invariant list and
`docs/architecture/stropha-system.md` for the broader system design.
