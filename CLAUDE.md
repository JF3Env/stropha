# CLAUDE.md — stropha

Operational guide for LLM coding agents working on this repository.

## What this project is

**stropha** is a local-first RAG system for codebases, exposed via MCP.
Agents call `search_code(...)` and get file/line/snippet results. Structural
tools (`find_callers`, `find_tests_for`, etc.) traverse the graphify symbol graph.

## How to work in this repo

1. **Use RAG tools before reading files.** `search_code`, `get_symbol`,
   `get_file_outline` answer "where is X?" cheaper than `Read`.
2. **Use the graph before grep.** `find_callers`, `find_related`,
   `find_tests_for` are more precise than text search for structural queries.
3. **Do not manually run `stropha index`.** The post-commit hook handles it.
4. **After editing code:** `uv run pytest -q` MUST stay green.
5. **Check invariants below before editing.**

## How to run

```bash
uv sync --extra local --extra dev          # one-time setup
uv run stropha index                       # index target repo
uv run stropha search "query"              # hybrid search
uv run stropha stats                       # index metadata
uv run pytest -q                           # run tests
uv run stropha eval --top-k 10             # golden-set evaluation
bin/stropha-mcp-ensure                     # start the shared MCP daemon (HTTP)
bin/stropha-mcp-ensure --status            # check it
bin/stropha-mcp-ensure --stop              # shut it down
```

MCP clients connect to `http://127.0.0.1:9741/mcp` (override host/port
via `STROPHA_MCP_HOST` / `STROPHA_MCP_PORT`). Health probe: `GET /health`.

## Invariants (do NOT break)

1. **HTTP-only MCP transport** -- `stropha-mcp` is served exclusively
   over `streamable-http` at `http://<host>:<port>/mcp`. A single shared
   singleton (started by `bin/stropha-mcp-ensure`) serves every client and
   owns one warm pipeline/storage runtime for the daemon lifetime. MCP
   sessions share those heavy adapters but never their dedup, pins, scratchpad,
   recent-query, or trace state.
   stdio and SSE are explicitly NOT supported -- never re-introduce
   `--transport stdio`, subprocess-spawning client configs, or
   per-client env blocks.
2. **Static MCP profiles** -- `src/stropha/server_tool_manifest.py` is the
   canonical tool inventory. `mcp.profile` is daemon-wide, accepts only `core`,
   `standard`, or `full`, and defaults to `full`. Enforce it in both discovery
   and invocation. Profile changes require a daemon restart; never add CLI,
   environment, or per-session overrides. `capabilities` stays visible in every
   profile.
3. **Drift detection** -- every config flag that changes adapter output
   MUST be baked into `adapter_id`. The pipeline uses
   `(content_hash, embedding_model, enricher_id)` as freshness key.
4. **Forward-only schema migrations** -- every storage adapter owns an
   idempotent, forward-only migration path. SQLite's `Storage._migrate()` uses
   `IF NOT EXISTS` / `_add_column_if_missing`; PostgreSQL provides the equivalent
   adapter migration. Never DROP.
5. **`Pipeline.run()` owns graphify lifecycle** -- only call site for
   `graphify update` -> `GraphifyLoader` -> `GraphVecLoader` -> `augment_fts`.
   Do not reorder.
6. **Graph-gated tools never return silent empty** -- when `graph_nodes`
   is empty, return `{"graph_loaded": false, "message": ...}`.
7. **Hook <= 100 ms before fork** -- post-commit hook exits foreground
   instantly. All real work in detached background.
8. **Local-only by default** -- no mandatory cloud dependency.
   Voyage/Anthropic/OpenAI stay optional.
9. **SQLite FTS5 writes are explicit** -- the `sqlite-vec` adapter's
   `fts_chunks` table is contentless-with-external-content. Replace rows with
   `DELETE` followed by explicit
   `INSERT INTO fts_chunks(rowid, content, rel_path)`; never `UPDATE` it.
   PostgreSQL uses its own `chunks.fts` tsvector path and is not subject to this
   SQLite-specific rule.
10. **Search-time toggles don't require re-index** -- HyDE and recursive
   merge are query-path only.
11. **Primary and secondary vector spaces stay independent** -- the primary
    embedder owns `chunks.embedding` / `vec_chunks`; the optional 256-d
    Model2Vec lane uses its own tables and drift identity. Enabling or changing
    the secondary lane must never rebuild the primary vector space.

## Code conventions

- Python 3.12+, type hints on public functions, pydantic for data.
- No global state. Pass `Config`/`Storage` explicitly.
- Errors: raise `stropha.errors.StrophaError` subclasses, never bare `Exception`.
- Logging: `structlog`, include `request_id`/`chunk_id`, never log secrets.
- Line length 100. `ruff` + `mypy` clean.

## Hygiene rules

1. Never invent symbols or APIs -- read `.venv/` source if unsure.
2. Never bypass the adapter framework -- use `@register_adapter` + protocol.
3. Never write to the index outside the active storage adapter implementation
   (`storage/sqlite.py` for `sqlite-vec`, `adapters/storage/postgres.py` for
   PostgreSQL).
4. Never silently empty the `find_*` tools.
5. Never index secrets.
6. Spec drift = update `docs/architecture/` specs in the same commit.
7. The hook is non-negotiably non-blocking.

## Key architecture

Configuration lives in **`~/.stropha/config.yaml`** — single source of
truth. No project-level `stropha.yaml`, no `.env` overlay, no
`STROPHA_*_*` env vars for settings. The only env vars still honoured
are process/runtime arguments for target selection, the shared MCP launcher,
session/tracing identity, hook escape hatches, and opt-in local PostgreSQL
container management; see `src/stropha/config/__init__.py` `_ENV_REGISTRY` for
the canonical list. API keys (`voyage_api_key`, `anthropic_api_key`,
`deepseek_api_key`, `lms_api_key`, `moa_api_key`) live under `secrets:` in the
YAML. Use the always-visible `capabilities` MCP tool for the active tool surface;
`get_config` provides the broader live runtime state only in the `full` profile.

Adapters register via `@register_adapter(stage=..., name=...)` across the core
pipeline stages and nested stages such as language chunkers, retrieval streams,
rerankers, and routers. Run `stropha adapters list` or `stropha pipeline show`
for the live inventory; do not copy hard-coded totals into current docs.

Schema is at v14 (SQLite/PostgreSQL). v14 adds the independent 256-d
Model2Vec chunk-vector space. All migrations are forward-only and idempotent.
Run `stropha stats` for current schema + chunk counts.

## Useful pointers

| Resource | What |
|---|---|
| `docs/architecture/stropha-system.md` | Master design doc |
| `docs/architecture/stropha-agent-exoskeleton.md` | Source-evidence and compact MCP architecture |
| `docs/architecture/stropha-pipeline-adapters.md` | Adapter framework ADR |
| `docs/architecture/stropha-graphify-integration.md` | Graphify integration RFC |
| `src/stropha/server_tool_manifest.py` | Canonical MCP tool inventory and profile membership |
| `graphify-out/` (untracked) | Local graphify cache + structural map; regenerated by post-commit hook. Never commit. |
| `README.md` | Public pitch + install |
