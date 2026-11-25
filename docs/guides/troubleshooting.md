# Troubleshooting Guide

## Installation

### `No module named 'fastembed'` Or Missing Tree-Sitter Support

The base package does not include the default local embedding and AST chunking
dependencies. Install the local extra:

```bash
uv sync --extra local
```

With pip:

```bash
pip install -e ".[local]"
```

### SQLite Extension Loading Is Unavailable

Use a Python build with SQLite extension loading enabled, such as a Python
installed by uv:

```bash
uv python install 3.12
uv sync --extra local
```

## Configuration

`~/.stropha/config.yaml` is the only settings file. Project-level and `.env`
files, and old per-setting environment controls, do not configure the current
pipeline.

Inspect what stropha resolved:

```bash
uv run stropha pipeline show
uv run stropha pipeline validate
```

`STROPHA_TARGET_REPO` is still accepted as a runtime target argument. Prefer an
explicit `--repo /path/to/project` for commands when diagnosing scope problems.

## Indexing

### No Files Are Indexed

Check the target, ignore rules, and resulting counts:

```bash
git status
uv run stropha index --repo /path/to/project
uv run stropha stats
```

The default file-size limit is 512 KiB. Configure larger intentional source
files in `~/.stropha/config.yaml`:

```yaml
walker:
  max_file_bytes: 1048576

pipeline:
  walker:
    adapter: git-ls-files
    config:
      max_file_bytes: 1048576
```

Exclude generated and minified content with `.strophaignore` rather than raising
the limit indiscriminately.

### Embedding Dimension Mismatch

The configured primary vector dimension does not match the existing fixed-width
tables. Confirm the intended embedder, then rebuild the index:

```bash
uv run stropha pipeline show
uv run stropha index --rebuild --repo /path/to/project [--repo /path/to/another]
```

A dimension change clears all derived search data for the entire shared SQLite
or PostgreSQL index. Include every indexed repository in the rebuild; otherwise
repositories omitted from the command remain unsearchable.

Changing only an enricher normally needs a regular index pass, not a destructive
rebuild; drift detection reprocesses affected chunks.

### Indexing Is Slow

Common causes are an LLM-backed enricher, cloud embeddings, large intentional
files, or a full graph refresh. Inspect the resolved pipeline and start from the
local hierarchical baseline:

```bash
uv run stropha pipeline show
uv run stropha index --enricher hierarchical --embedder local
```

Do not rely on fixed throughput numbers from another machine or repository.

## Search

### No Results For Known Code

```bash
uv run stropha stats
uv run stropha search "KnownSymbol"
```

`stropha stats` opens the configured storage backend. With SQLite, opening a
missing path can initialize an empty database. Check that chunk counts are
nonzero and that the expected repository appears; success alone does not prove
that an earlier index completed.

Also remove restrictive filters and confirm the current target repository.
Within MCP, use `get_symbol("KnownSymbol")` for an exact identifier.

### Search Is Slow

HyDE, query rewriting, and multi-query expansion add inference calls. Disable
them in the user YAML while diagnosing:

```yaml
retrieval_features:
  hyde_enabled: false
  query_rewrite_enabled: false
  multi_query_enabled: false

pipeline:
  retrieval:
    adapter: hybrid-rrf
    config:
      recursive_enabled: false
```

The semantic query-cache implementation is not connected to the production
search path, so enabling `cache_enabled` is not a valid latency fix.

If you edited the YAML directly while the daemon was running, restart it:

```bash
bin/stropha-mcp-ensure --stop
bin/stropha-mcp-ensure
```

## MCP Daemon

### Server Does Not Start

```bash
which uv
bin/stropha-mcp-ensure
bin/stropha-mcp-ensure --status
curl http://127.0.0.1:9741/health
```

The launcher checks both process health and a real MCP initialization request.
If `--status` reports a wedged endpoint, use the stop/start recovery it prints
and inspect `~/.cache/stropha/mcp-server.log`.

For foreground diagnosis, set logging in `~/.stropha/config.yaml`, stop the
shared daemon, and run:

```yaml
logging:
  level: DEBUG
```

```bash
uv run stropha-mcp --host 127.0.0.1 --port 9741
```

### Tools Do Not Appear

Verify that the client uses server name `stropha_rag`, remote URL
`http://127.0.0.1:9741/mcp`, and streamable HTTP. Remove subprocess fields such
as `command`, `args`, and per-process `env`. Restart the client after changing
its MCP configuration.

Call `capabilities` to inspect the daemon's active profile and tool names. The
public router is named `smart_search_tool` and requires the `standard` or `full`
profile. To expose a tool from a larger profile, change `mcp.profile` in
`~/.stropha/config.yaml`, then stop and start the shared daemon. Restarting only
the client does not change the daemon-wide tool surface.

### Configuration Change Is Not Active

- Dashboard `PATCH /api/config` writes the YAML and replaces a compatible warm
  runtime immediately.
- `mcp.profile` is static for the daemon lifetime. A dashboard change returns
  `restart_required: true`; stop and start the daemon to activate it.
- Direct file edits require a daemon restart.
- Walker, chunker, enricher, and graph changes need a subsequent index pass to
  update stored content.
- A primary dimension conflict returns HTTP 202 with `pending_rebuild: true`;
  the old runtime remains active. Run `stropha index --rebuild`, then restart
  the daemon or submit the patch again.

## Watcher And Hooks

### Saved Changes Are Not Appearing

The MCP watcher starts when the shared runtime is first requested, not merely
when the shell launcher process starts. Check `mcp.watch` and the target:

```yaml
mcp:
  watch: true
  watch_interval: 1.0
  watch_debounce: 2.0
```

The watcher consumes native filesystem notifications and updates the persistent
index after a debounce period. `watch_interval` is the maximum event wait before
maintenance and retry checks, not a repository scan interval. Startup and
periodic reconciliation recover missed events. This is not a RAM overlay, and
the default lightweight pass does not refresh graphify data or graph vectors on
each save.

### Hook Is Not Running

```bash
uv run stropha hook status --target /path/to/project
uv run stropha hook install --force --target /path/to/project
```

Use the hook path and log path reported by `status`. The installer honors
`core.hooksPath`, so `.git/hooks/post-commit` may be the wrong file. `status`
also reports the installed script version; avoid hard-coded minimum versions
because the generated hook evolves.

The hook invokes one normal `stropha index --repo <target>` background job. The
pipeline, not the shell hook, runs graphify. The normal auto mode uses
hash-based reconcile; Git-diff incremental mode requires an explicit
`--incremental` CLI request.

If the hook must be bypassed for one operation:

```bash
STROPHA_HOOK_SKIP=1 git commit -m "message"
uv run stropha index --repo /path/to/project
```

## Graph

### `graph_loaded: false`

```bash
graphify .
uv run stropha index
```

Graph-gated tools intentionally return this explicit state instead of a silent
empty result. Ensure `graphify-out/graph.json` belongs to the target repository
and rerun indexing so the pipeline can load it.

## Database Recovery

First identify the configured backend and location:

```bash
uv run stropha pipeline show
uv run stropha stats
```

Do not use SQLite file commands against PostgreSQL, and never delete SQLite
`-wal` or `-shm` files to clear a lock. They can contain committed data not yet
checkpointed into the main file.

### SQLite: Database Is Locked

1. Let active indexing or hook jobs finish.
2. Stop the MCP daemon and any `stropha watch` process.
3. Retry the command.
4. If the lock persists, inspect which process has the configured database open
   before terminating only that known stropha process.

```bash
bin/stropha-mcp-ensure --stop
uv run stropha index --repo /path/to/project
```

Do not remove WAL or shared-memory sidecars while any process can have the
database open.

### SQLite: Suspected Corruption

Stop every stropha process using the database, then run SQLite's integrity check
against the path shown by `pipeline show`:

```bash
bin/stropha-mcp-ensure --stop
sqlite3 /path/from/pipeline-show 'PRAGMA integrity_check;'
```

If the result is not `ok`, preserve the main database and any `-wal` and `-shm`
sidecars together before attempting recovery. Because the index is derived from
source, the simplest safe recovery is usually to move that complete stopped
database set to a backup location, run `stropha index --rebuild`, and keep the
backup until the rebuilt index is verified. Do not remove an entire `.stropha`
directory: configuration and secrets live under `~/.stropha`, and the configured
index may be elsewhere.

### PostgreSQL

For the PostgreSQL adapter, first verify server connectivity, credentials,
database availability, and the pgvector extension. Use PostgreSQL backup and
restore tools for physical or logical recovery. Once the server is healthy,
`stropha index --rebuild --repo /path/to/project` can reconstruct the derived
index rows for the selected repository. Do not delete Docker volumes or database
schemas as a generic lock or corruption fix.

If the launcher manages the opt-in local container, include
`STROPHA_USE_POSTGRES=1` when checking its launcher status. Externally managed
PostgreSQL is never started or stopped by the launcher.

## Enrichers

### Contextual Enrichment Fails

The `contextual` enricher uses `inference.backend`; it is not inherently an
Ollama adapter. Inspect both selections:

```bash
uv run stropha pipeline show
uv run stropha pipeline validate
```

If `inference.backend: ollama` is selected, check the configured Ollama host and
model. If the pipeline enricher itself is `ollama`, that separate adapter calls
Ollama directly and uses its own adapter configuration.

Failed contextual generation falls back to raw content and is not cached as a
successful enrichment. Fix the backend and rerun indexing to retry those chunks.

## Reporting Issues

File issues at <https://github.com/JonatasFischer/stropha/issues>.

Include:

1. `stropha --version`
2. OS and Python version
3. `stropha pipeline show` with secrets removed
4. Full error text and reproduction steps
5. Relevant daemon or hook logs with credentials and source secrets redacted
