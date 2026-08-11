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

The MCP schema is the authoritative live inventory. These are the main public
entry points:

| Tool | Best use |
|---|---|
| `search_code` | Conceptual or free-text code search |
| `smart_search_tool` | Route an ambiguous request to semantic, symbol, or graph retrieval |
| `get_symbol` | Look up a known class, function, or method |
| `get_file_outline` | Inspect symbols and line ranges before reading a file |
| `find_callers` | Find callers through graph edges |
| `find_tests_for` | Find tests associated with a symbol |
| `find_related` | Traverse related symbols in either direction |
| `get_community` | Inspect a graph community |
| `find_rationale` | Find docs or ADRs linked to a symbol |
| `trace_feature` | Trace a feature through graph call paths |
| `assemble_context` | Retrieve and pack a bounded context bundle |

Examples:

```text
search_code("how does authentication work", language=["typescript"])
get_symbol("FsrsCalculator")
find_callers("FsrsCalculator", depth=2)
find_tests_for("submitAnswer")
trace_feature("user submits an answer and mastery is updated")
```

Search, symbol, graph, and composed tools default to the current repository.
Pass `include_other_repos=true` only when cross-repository results are intended.
Graph tools return `graph_loaded: false` with guidance when the graph mirror is
not available.

## Configuration And Reloading

All settings and secrets live in `~/.stropha/config.yaml`. Project-level and
`.env` files, and generic environment-variable setting overlays, are not
supported.

Example daemon settings:

```yaml
mcp:
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
prior adapters.

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
