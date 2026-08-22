# MCP Integration Guide

stropha exposes MCP over streamable HTTP only. A single daemon serves all
clients with one shared pipeline and storage runtime. stdio, SSE, and
per-client subprocess configurations are not supported.

## Quick Start

```bash
git clone https://github.com/JonatasFischer/stropha.git
cd stropha
uv sync --extra local

uv run stropha index --repo /path/to/project
bin/stropha-mcp-ensure
bin/stropha-mcp-ensure --status
```

The default MCP URL is `http://127.0.0.1:9741/mcp`. Process health is available
at `http://127.0.0.1:9741/health`.

The daemon process starts first. The heavy adapter graph is built lazily on the
first MCP or frontend request and remains warm for later sessions. Session
state, including dedup history, pins, and scratchpads, stays isolated. When
`mcp.watch` is enabled, the shared runtime starts one polling watcher and closes
it only when the runtime is replaced or the daemon stops.

## Client Configuration

### OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "stropha_rag": {
      "type": "remote",
      "url": "http://127.0.0.1:9741/mcp",
      "enabled": true
    }
  }
}
```

### Claude, Cursor, And Similar Clients

Use the configuration location documented by the client and point its remote
MCP entry at the shared URL:

```json
{
  "mcpServers": {
    "stropha_rag": {
      "url": "http://127.0.0.1:9741/mcp",
      "transport": "streamable-http"
    }
  }
}
```

Continue and Zed may use a different outer schema, but the transport and URL are
the same. Do not add `command`, `args`, or a per-client `env` block for stropha.

## Tool Selection

The active MCP `tools/list` response is the authoritative live inventory. The
canonical full inventory and minimum profile for each tool are declared in
`src/stropha/server_tool_manifest.py`. Profiles are nested:

- **core (13 tools):** `capabilities`, `select_repository`, `search_code`,
  `get_symbol`, `list_repos`, `get_file_outline`, `bootstrap_environment`,
  `find_callers`, `find_related`, `find_rationale`, `find_tests_for`,
  `assemble_context`, and `index_health`.
- **standard (39 tools):** core plus `observe_agent_state`,
  `record_runtime_artifact`, `resolve_agent_instructions`, `reset_dedup`,
  `pin_chunks`, `unpin_chunks`,
  `list_pins`, `smart_search_tool`, `expand_chunk`, `expand_chunks`,
  `plan_investigation`, `get_community`, `trace_feature`,
  `execute_investigation`, `suggest_next_tool_mcp`, `list_task_recipes`,
  `apply_task_recipe`, `verify_claim_tool`, `check_symbols_tool`,
  `blast_radius_tool`, `assert_snippet_freshness_tool`, `test_impact_tool`,
  `audit_citations_tool`, `spec_consistency_tool`, `contract_tests_tool`, and
  `working_set_tool`.
- **full (54 tools):** standard plus `remember`, `recall`, `forget`,
  `cache_stats`, `cache_prune`, `scratchpad_read`, `scratchpad_write`,
  `scratchpad_append`, `scratchpad_clear`, `check_staleness`, `record_episode`,
  `recall_episodes`, `co_occurrence`, `report_bad_result`, and `get_config`.

`full` is the compatibility default. Profiles filter tools only; resources and
prompts are unchanged. `capabilities` is always visible and reports the active
profile and its ordered tool names.

These are the main public entry points:

| Tool | Minimum profile | Best use |
|---|---|---|
| `capabilities` | core | Inspect the active profile and tool inventory |
| `select_repository` | core | Bind the current session to an indexed repository |
| `observe_agent_state` | standard | Replace or retrieve ephemeral editor state and runtime evidence |
| `record_runtime_artifact` | standard | Attach a bounded diagnostic or failing-test artifact |
| `resolve_agent_instructions` | standard | Resolve bounded saved-worktree instructions and observed changed paths |
| `search_code` | core | Conceptual or free-text code search |
| `smart_search_tool` | standard | Route an ambiguous request to semantic, symbol, or graph retrieval |
| `get_symbol` | core | Look up a known class, function, or method |
| `get_file_outline` | core | Inspect symbols and line ranges before reading a file |
| `find_callers` | core | Find callers through graph edges |
| `find_tests_for` | core | Find tests associated with a symbol |
| `find_related` | core | Traverse related symbols in either direction |
| `get_community` | standard | Inspect a graph community |
| `find_rationale` | core | Find docs or ADRs linked to a symbol |
| `trace_feature` | standard | Trace a feature through graph call paths |
| `assemble_context` | core | Retrieve and pack a bounded context bundle |

Examples:

```text
search_code("how does authentication work", language=["typescript"])
get_symbol("FsrsCalculator")
find_callers("FsrsCalculator", depth=2)
find_tests_for("submitAnswer")
trace_feature("user submits an answer and mastery is updated")
resolve_agent_instructions(path="src/stropha/server.py")
```

Search, symbol, graph, and composed tools default to the current repository.
Pass `include_other_repos=true` only when cross-repository results are intended.
Graph tools return `graph_loaded: false` with guidance when the graph mirror is
not available.

## Ephemeral Host Context

Hosts using `standard` or `full` may call `observe_agent_state` with a complete
snapshot. Revisions must increase and each accepted snapshot replaces the prior
one. Calling the tool without `snapshot` returns that session's current snapshot
and runtime artifacts. `record_runtime_artifact` accepts up to 16 immutable
diagnostic, test-failure, stack-frame, or log-excerpt IDs over a session's
lifetime, with a 32 KiB per-item and 128 KiB active-content limit. Expiry removes
content but does not permit an ID to be rebound before reconnect.

Call `select_repository` before repository-local tools. Standard and full hosts
may instead use `observe_agent_state`; its validated repository updates the same
session default. Explicit repository arguments take precedence. Sessions that do
not select or report a repository retain the daemon target as a legacy fallback.

`resolve_agent_instructions` defaults to the session repository and observed
active file, while explicit `repo` and `path` selectors take precedence. It
returns only applicable saved-worktree `AGENTS.md` and `CLAUDE.md` sources in
root-to-leaf order, plus changed paths from a snapshot for the same repository.
Each source includes full-source SHA-256 provenance and bounded content under a
shared character budget. The tool does not interpret instructions, read unsaved
buffers, persist or trace content, or assemble a final prompt. Secure no-follow
reads reject path-replacement races, sources above 256 KiB, and applicable source
sets above 1 MiB or 64 files.

Repository keys must match an indexed repository. Active, recent, changed,
selection, and artifact source paths must be repository-relative and may not
escape through `..`, absolute paths, or symlinks. Paths are checked again on
readback. Observation content is never persisted or traced. It is cleared when
the MCP session ends, so reconnecting hosts must resend it.

## Configuration And Reloading

All settings and secrets live in `~/.stropha/config.yaml`. Project-level and
`.env` files, and generic environment-variable setting overlays, are not
supported.

Example daemon settings:

```yaml
mcp:
  profile: full
  watch: true
  watch_interval: 1.0
  watch_debounce: 2.0

logging:
  level: INFO
```

`STROPHA_TARGET_REPO` is a runtime argument rather than configuration.
`STROPHA_MCP_HOST` and `STROPHA_MCP_PORT` are operational inputs to
`bin/stropha-mcp-ensure`. Pipeline, storage, model, logging, and API-key settings
belong in the YAML file.

The dashboard at `http://127.0.0.1:9741/dashboard` updates configuration through
`PATCH /api/config`. A successful patch atomically replaces the warm runtime;
new requests use the replacement while existing sessions can finish with their
prior adapters. `mcp.profile` is the exception: profile membership is fixed for
the daemon lifetime. A profile patch is saved and returns
`restart_required: true`, but discovery and invocation keep using the old
profile until the daemon restarts. There is no CLI, environment, or per-session
profile override.

Indexing-time changes still require an index pass. If a primary vector dimension
change conflicts with the existing index, the endpoint returns HTTP 202 and
`pending_rebuild: true`; run:

```bash
uv run stropha index --rebuild
```

The old runtime remains active. A primary dimension rebuild clears all derived
search data for every repository in the shared backend, so include all indexed
repositories in that pass. After the rebuild, restart the daemon or submit the
patch again so it can install a compatible replacement. After editing
`~/.stropha/config.yaml` directly, restart the daemon because the file itself is
not watched for live reload:

```bash
bin/stropha-mcp-ensure --stop
bin/stropha-mcp-ensure
```

## Automatic Index Updates

Install the post-commit hook from the stropha checkout:

```bash
uv run stropha hook install --target /path/to/project
uv run stropha hook status --target /path/to/project
```

The installer honors `core.hooksPath`, so the active file is not necessarily
`.git/hooks/post-commit`. Use `stropha hook status` to see the resolved path and
installed version. Reinstalling updates only the managed block and preserves
other hook content.

The hook detaches a background job and returns control to Git immediately. It
runs normal `stropha index --repo <target>` auto mode; the pipeline owns the
graphify update and graph loading. The next query observes committed index
updates through the shared storage backend without a client restart.

The daemon watcher is separate. It polls saved working-tree files and writes
incremental changes to the persistent index. It is not a RAM overlay, and its
default lightweight refresh does not rebuild graph vectors on every save.

## PostgreSQL

PostgreSQL plus pgvector is a shipped optional storage backend. Install its
dependency and configure it in the user YAML:

```bash
uv sync --extra local --extra postgres
```

```yaml
pipeline:
  storage:
    adapter: postgres
    config:
      dsn: postgresql://user:password@127.0.0.1:5432/stropha
```

The launcher leaves native or externally managed PostgreSQL alone. To ask the
launcher to manage its opt-in local Docker container:

```bash
STROPHA_USE_POSTGRES=1 bin/stropha-mcp-ensure
```

## Troubleshooting

### Connection Refused

```bash
bin/stropha-mcp-ensure --status
curl http://127.0.0.1:9741/health
```

If status reports a live but unresponsive MCP endpoint, use the recovery command
it prints, then inspect `~/.cache/stropha/mcp-server.log`.

### Expected Tool Is Missing

Call `capabilities` to inspect the daemon's active profile and ordered tool
names. If the tool belongs to a larger profile, set `mcp.profile` in
`~/.stropha/config.yaml`, then stop and start `bin/stropha-mcp-ensure`. Re-saving
the client configuration or opening a new MCP session does not change the
daemon-wide profile.

### No Results

```bash
uv run stropha stats
uv run stropha search "simple known symbol"
```

`stropha stats` opens the configured backend. For SQLite, this can initialize an
empty database at a missing path, so check that chunk counts are nonzero and the
expected repository appears; command success alone does not prove a prior index
exists.

### Graph Tools Report `graph_loaded: false`

```bash
graphify .
uv run stropha index
```

### Hook Not Running

```bash
uv run stropha hook status --target /path/to/project
uv run stropha hook install --force --target /path/to/project
```

Use the hook path and log path printed by `status`; do not assume a fixed Git
hook directory or script version.
