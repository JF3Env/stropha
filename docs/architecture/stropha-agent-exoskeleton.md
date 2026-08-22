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
| `core` | Small retrieval and orientation surface; 12 tools |
| `standard` | `core` plus observations, session ergonomics, investigations, and change intelligence; 37 tools |
| `full` | Complete compatibility surface; 52 tools including `capabilities` |

`core` contains:

`capabilities`, `search_code`, `get_symbol`, `list_repos`, `get_file_outline`,
`bootstrap_environment`, `find_callers`, `find_related`, `find_rationale`,
`find_tests_for`, `assemble_context`, and `index_health`.

`standard` adds:

`observe_agent_state`, `record_runtime_artifact`, `reset_dedup`, `pin_chunks`,
`unpin_chunks`, `list_pins`, `smart_search_tool`, `expand_chunk`,
`expand_chunks`, `plan_investigation`, `get_community`, `trace_feature`,
`execute_investigation`, `suggest_next_tool_mcp`, `list_task_recipes`,
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

## Phase Boundary

Phases delivered so far:

- **Phase 0:** baseline hardening and green repository contracts.
- **Phase 1:** trustworthy saved-source snapshots.
- **Phase 2:** canonical compact MCP profiles and structured tool contracts.
- **Phase 3:** ephemeral agent observations and bounded runtime artifacts.

Phase 4 may add instruction resolution and change intelligence. Unsaved-buffer
overlays remain deferred to Phase 8.

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
