# MCP Integration Guide

> **stropha** exposes its MCP tools over **HTTP only** (streamable-http
> transport). A single shared singleton serves every MCP client
> concurrently: one process, one warm pipeline/storage runtime, and no
> subprocess spawning. This guide shows how to start the daemon and point
> each client at it.

---

## Quick Start

### 1. Install stropha

```bash
git clone https://github.com/JonatasFischer/stropha.git
cd stropha
uv sync
```

### 2. Index your codebase

```bash
uv run stropha index                                   # current repo
STROPHA_TARGET_REPO=/path/to/project uv run stropha index   # explicit target
uv run stropha search "where is the main entry point"
```

### 3. Start the shared MCP server

```bash
bin/stropha-mcp-ensure          # start (returns when healthy)
bin/stropha-mcp-ensure --status # check
bin/stropha-mcp-ensure --stop   # shut down
```

Defaults:

| | |
|---|---|
| URL | `http://127.0.0.1:9741/mcp` |
| Health probe | `GET /health` |
| PID file | `~/.cache/stropha/mcp-server.pid` |
| Log file | `~/.cache/stropha/mcp-server.log` |

Override host / port with `STROPHA_MCP_HOST` / `STROPHA_MCP_PORT`.

The first MCP session builds the configured adapters and starts the watcher.
Later sessions, including stale-session recovery after a client reconnect,
reuse that warm runtime. Session state remains isolated: dedup history, pins,
scratchpad, recent queries, and trace recorder are created per session. The
watcher and storage connection close only when the daemon shuts down.
Restart the daemon after changing adapter or storage configuration so the warm
runtime is rebuilt from the new settings.

### 4. Configure your MCP client

Copy `.mcp.example.json` into the repo that you want the client to read,
or use one of the client-specific snippets below.

---

## Client-Specific Configuration

All clients connect to the same URL. The schema differs slightly by
client.

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

### Claude Desktop / Claude Code

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

### Cursor

`.cursor/mcp.json` in your project:

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

### Continue (VS Code)

`~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "stropha_rag",
      "url": "http://127.0.0.1:9741/mcp",
      "transport": "streamable-http"
    }
  ]
}
```

### Zed

`~/.config/zed/settings.json`:

```json
{
  "assistant": {
    "mcp_servers": {
      "stropha_rag": {
        "url": "http://127.0.0.1:9741/mcp",
        "transport": "streamable-http"
      }
    }
  }
}
```

### Other MCP-compatible clients

Any client that speaks the MCP streamable-http transport works. Point
it at `http://127.0.0.1:9741/mcp`. **stdio is not supported** — there
is no per-client subprocess.

---

## Available Tools

### Search Tools

| Tool | Description |
|------|-------------|
| `search_code` | Hybrid semantic + lexical search. Use for "where is X?" or "how does Y work?" |
| `smart_search` | Route an ambiguous query to scoped semantic, symbol, or graph retrieval. |
| `plan_investigation` | Propose a scoped entry points → callers → tests → rationale recipe. |
| `assemble_context` | Run scoped semantic retrieval and optional graph extensions into one bundle. |
| `execute_investigation` | Plan and execute the scoped investigation DAG in one call. |
| `get_symbol` | Exact symbol lookup. Cheaper than `search_code` when you know the name. |
| `get_file_outline` | Get all symbols in a file. Plan reads before consuming context. |
| `list_repos` | List all indexed repositories. |

Semantic, symbol, direct graph, and composed search tools default to the current
repository. Set `include_other_repos=true` explicitly to search all indexed
repositories. If the current repository cannot be resolved, scoped tools fail
closed; graph-gated tools report `graph_loaded: false` with an actionable message.

### Graph Traversal Tools

These tools require the graphify mirror to be loaded. Index runs
`graphify update` automatically when the binary is installed.

| Tool | Description |
|------|-------------|
| `find_callers` | Who calls this symbol? BFS up `calls` edges. |
| `find_tests_for` | Find tests that cover a symbol. |
| `find_related` | Find symbols related to this one (any edge type). |
| `get_community` | Get all symbols in the same architectural community. |
| `find_rationale` | Find docs/ADRs explaining why this symbol exists. |
| `trace_feature` | Trace a feature description to all code that implements it. |

### Debug Tools

| Tool | Description |
|------|-------------|
| `get_config` | Show active configuration. Debug which index is being used. |

---

## Example Queries

### Finding Code

```
search_code("where is the FSRS calculator implemented")
search_code("how does authentication work", language=["typescript"])
search_code("API endpoints", path_prefix="src/api/", exclude_tests=True)
```

### Graph Traversal

```
find_callers("FsrsCalculator", depth=2)
find_tests_for("submitAnswer")
trace_feature("user submits an answer and mastery is updated")
get_community("StudyService")
```

### Architecture Understanding

```
search_code("entry point", kind=["class", "function"])
find_related("Database", relations=["implements", "references"])
find_rationale("FsrsCalculator")
```

---

## Environment Variables

### Server (consumed by `bin/stropha-mcp-ensure` and `stropha-mcp`)

| Variable | Default | Description |
|----------|---------|-------------|
| `STROPHA_MCP_HOST` | `127.0.0.1` | Bind address for the daemon |
| `STROPHA_MCP_PORT` | `9741` | Bind port for the daemon |
| `STROPHA_TARGET_REPO` | cwd | Repository the server reads from |
| `STROPHA_INDEX_PATH` | `~/.stropha/index.db` | SQLite index location |
| `STROPHA_LOG_LEVEL` | `WARNING` | Log verbosity for the server |

### Indexing (consumed by `stropha index`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VOYAGE_API_KEY` | unset | Enable Voyage embeddings (optional, local fallback available) |
| `STROPHA_LOG_LEVEL` | `INFO` | Log verbosity |

Query-time toggles (HyDE, recursive merge, multi-query, query cache)
live in `stropha.yaml` under `retrieval_features:` — they no longer
have env-var equivalents. See the project's `stropha.yaml` for the
full schema.

---

## Automatic Index Updates

stropha includes a post-commit hook that updates the index after every
commit:

```bash
cd /path/to/your/project
uv run --directory /path/to/stropha stropha hook install

uv run --directory /path/to/stropha stropha hook status
```

The hook runs in the background (non-blocking) and refreshes both the
SQLite index and the graphify graph. Because the MCP server reads the
same SQLite file, your next query sees the new state immediately — no
client restart required.

---

## Troubleshooting

### "Connection refused" / client can't reach stropha

1. Is the daemon running? `bin/stropha-mcp-ensure --status`
2. Health check: `curl http://127.0.0.1:9741/health`
3. Tail the log: `tail -f ~/.cache/stropha/mcp-server.log`

### "No results" from `search_code`

1. Check the index exists: `ls ~/.stropha/index.db`
2. Verify repo is indexed: `uv run stropha stats`
3. Try a simpler query to test basic functionality

### Graph tools return `graph_loaded: false`

1. Make sure graphify is installed: `which graphify`
2. Re-run index: `uv run stropha index` (runs `graphify update` automatically)
3. Verify graph exists: `ls graphify-out/graph.json`

### Slow indexing

1. Use local embeddings (default) instead of Voyage for large repos
2. Enable incremental indexing (default in hook v4)
3. Consider excluding large generated files via `.strophaignore`

### Hook not running

1. Check hook is installed: `cat .git/hooks/post-commit`
2. Verify permissions: `chmod +x .git/hooks/post-commit`
3. Check logs: `cat ~/.cache/stropha-hook.log`

---

## Advanced Configuration

### YAML Configuration

Create `stropha.yaml` in your repo root:

```yaml
walker:
  adapter: git-ls-files
  config:
    max_file_bytes: 1048576  # 1MB

enricher:
  adapter: contextual  # or: hierarchical, graph-aware, noop
  config:
    model: qwen2.5-coder:1.5b

embedder:
  adapter: local  # or: voyage, bge-m3
  config:
    model: mixedbread-ai/mxbai-embed-large-v1

retrieval:
  adapter: hybrid-rrf
  config:
    top_k: 10

retrieval_features:
  hyde_enabled: true
  multi_query_enabled: true
  recursive_enabled: true
```

### Multi-Repo Indexing

Index multiple repos into a single index:

```bash
# Via manifest file
cat > repos.yaml << EOF
repos:
  - path: ../frontend
  - path: ../backend
  - path: ../shared
EOF

uv run stropha index --manifest repos.yaml
```

Or via command line:

```bash
uv run stropha index --repo ../frontend --repo ../backend --repo ../shared
```

---

## Performance Tips

1. **Keep the daemon alive.** It's a singleton — start once per workstation, every project reads from it.
2. **Use local embeddings** for development (zero latency, zero cost)
3. **Enable query cache** for repeated queries (`retrieval_features.cache_enabled: true`)
4. **Use incremental indexing** (default with hook v4)
5. **Exclude large files** via `.strophaignore`
6. **Use `get_symbol` instead of `search_code`** when you know the exact name
