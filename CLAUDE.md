# CLAUDE.md — stropha

Operational guide for LLM coding agents (Claude Code, OpenCode, etc.) working on this repository.

## How to resume (start here in a new session)

1. **Read this file end-to-end.** It is the live state. Sections §2 (snapshot) and §3 (invariants) are reference; the chronological history at §6 is for "why is this here?" questions.
2. **Use the graph before grep.** `graphify-out/GRAPH_REPORT.md` has the god nodes; `graphify query "<question>"` traverses the EXTRACTED + INFERRED edges. The MCP server `stropha_rag` exposes 11 tools (see §2.2) for symbol / structural queries.
3. **Use the RAG before reading files.** The MCP tools `mcp_Stropha_search_code`, `mcp_Stropha_get_symbol`, `mcp_Stropha_get_file_outline` answer 90% of "where is X?" questions cheaper than a `Read`.
4. **The hook auto-refreshes on commit.** `.git/hooks/post-commit` v=3 runs `graphify update` → `stropha index` in detached background after every commit. Do NOT manually run `stropha index` unless you are debugging the hook itself. Logs: `~/.cache/stropha-hook.log`.
5. **Before editing code, check §3 invariants.** Drift detection, schema migration discipline, and the local-only policy are non-negotiable.
6. **After editing code:** `uv run pytest -q` MUST stay green (1457 tests at last bump). The hook regenerates the graph; you don't have to.

For the public pitch + install instructions, see `README.md`. For design rationale, see the specs in `docs/architecture/`.

## Project purpose

**stropha** is a local-first Retrieval-Augmented Generation system for arbitrary codebases, exposed to LLM clients via the **Model Context Protocol (MCP)**.

Goal: when an agent like Claude Code is editing a target codebase, it should be able to call `search_code("how does mastery work?")` and get back precise file/line/snippet results without burning context window on directory walks. Plus a small set of structural tools (`find_callers`, `find_tests_for`, `find_related`, `get_community`, `find_rationale`, `trace_feature`) that traverse the graphify symbol graph in pure SQL.

The original development target is the sibling **Mimoria** repository at `../mimoria`, but stropha is repo-agnostic — point `STROPHA_TARGET_REPO` anywhere, or use `stropha hook install --target <repo> --project-dir <where-stropha-lives>` to index a different repo cross-process.

## 2. Current state snapshot

> The tables below mirror what `uv run stropha adapters list` / `uv run stropha hook status` / `uv run pytest --collect-only` actually print at HEAD. Refresh them when adding/removing components.

### 2.1 SQLite schema versions

| Version | Added | Migration shape |
|---|---|---|
| v1 | `chunks`, `vec_chunks`, `fts_chunks`, `meta` | initial |
| v2 | `repos` + `chunks.repo_id` | per-chunk repo identity (Phase 4 foundation) |
| v3 | `chunks.embedding_text` + `chunks.enricher_id` + `enrichments` cache | drift detection via enricher_id |
| v4 | `graph_nodes` + `graph_edges` + `graph_meta` | graphify mirror (RFC §1.5a) |
| v5 | `graph_nodes.embedding` + `embedding_model` + `embedding_dim` | graph-vec retrieval stream (Trilha A L3) |
| v6 | `files` table (repo_id, rel_path, mtime, size_bytes, content_hash, last_enricher_id, last_embedder_model) | file-level dirty cache for incremental indexing (Phase A) |
| v7 | `graph_nodes.repo_id` + `graph_edges.repo_id` | multi-repo graphify support (distributed monorepo architecture) |

All migrations are forward-only and idempotent (`_add_column_if_missing` + `CREATE TABLE IF NOT EXISTS`). Legacy NULL `enricher_id` is treated as `'noop'` so v1/v2 dbs upgrade without full re-embed.

**Multi-repo graphify support (v7)**: Multiple repos with their own `graphify-out/graph.json` can share a single stropha index. Node IDs are prefixed with `{repo_id}:` to avoid collisions (e.g., repo_id=3, node "FooClass" becomes "3:FooClass"). Graph tools (find_callers, find_related, etc.) work cross-repo by default. The GraphifyLoader tracks per-repo staleness via `graph_meta` keys like `last_loaded_mtime:3`.

### 2.2 MCP tools (16) — server name `stropha_rag`

| Tool | Purpose | Graph required |
|---|---|---|
| `search_code` | Hybrid semantic + lexical search (4 streams + RRF, optional reranker + filters: `language`, `path_prefix`, `kind`, `exclude_tests`, `recursive`) | no |
| `smart_search` | Intelligent search with automatic query routing — classifies intent and dispatches to best tool | auto |
| `get_symbol` | Exact symbol lookup, cheaper than `search_code` when name is known | no |
| `get_file_outline` | Symbolic outline of one file — plan a `Read` before consuming a whole file | no |
| `list_repos` | Enumerate repos present in the index | no |
| `get_config` | Show active configuration (index path, target repo, embedding model, env sources) — debug tool | no |
| `find_callers` | Who calls `symbol`? BFS up `calls` edges, EXTRACTED only by default | yes |
| `find_tests_for` | Tests covering `symbol`. Path-pattern filter (`test_*`, `*.spec.*`, …) overrideable | yes |
| `find_related` | Symmetric BFS over any edge type (optional `relations` filter) | yes |
| `get_community` | Members of a precomputed graphify cluster | yes |
| `find_rationale` | Docs / ADRs explaining `symbol` via `rationale_for` edges | yes |
| `trace_feature` | DFS along `calls` edges from token-overlap entry points (Gherkin → step → method) | yes |
| `report_bad_result` | Flag a previous retrieval as bad (Strong-negative T6 signal for the Meta-Harness labeler, WIN-163) | no |
| `assemble_context` | One-shot context bundle (WIN-209). Hybrid search + optional caller/test graph extensions, packed under a token budget, rendered as a ready-to-paste markdown bundle | no |
| `expand_chunk` | Just-in-time dereference (WIN-216): resolve a single chunk_id to its full SearchResult, optionally re-rendered at line/paragraph level | no |
| `expand_chunks` | Batch variant of `expand_chunk` — dereference many chunk_ids in one call (missing ids surface with `found=False`, never partial-fails) | no |

Graph-gated tools return `{"graph_loaded": false, "message": …}` when the mirror is empty — never silent empty list.

**Query Router** (`smart_search`): Pattern-based intent classification routes queries like "what calls X" → `find_callers`, "tests for X" → `find_tests_for`. Falls back to `search_code` for conceptual queries. Optional LLM classification via `STROPHA_QUERY_ROUTER_LLM=1` for ambiguous cases.

### 2.3 Adapters registered (auto-loaded from `stropha.adapters`)

| Stage | Adapters | `adapter_id` shape |
|---|---|---|
| walker | `git-ls-files`, `filesystem`, `nested-git` | `<name>:max=<bytes>[:depth=<n>]` |
| chunker | `tree-sitter-dispatch` (+ 5 language sub-adapters: `ast-generic`, `file-level`, `heading-split`, `sfc-split`, `regex-feature-scenario`) | `tree-sitter-dispatch:<hash>` |
| enricher | `noop`, `hierarchical`, `graph-aware`, `ollama`, `mlx`, `contextual` | `<name>:<flag-letters>` (drift on flag flip) |
| embedder | `local` (fastembed), `voyage`, `bge-m3`, `model2vec` | `<name>:<model>:<dim>` |
| storage | `sqlite-vec` | `sqlite-vec:dim=<n>` |
| retrieval | `hybrid-rrf` (4 streams fused + optional reranker) | `hybrid-rrf:k=60:streams=<hash>:reranker=<id>` |
| retrieval-stream | `vec-cosine`, `fts5-bm25`, `like-tokens`, `graph-vec` | `<name>:k=<n>[:min=<sim>]` |
| reranker | `noop` (default), `cross-encoder` (BAAI/bge-reranker-base), `code-aware` | `reranker:<name>:<model>` |

The chunker's language sub-adapters are themselves an adapter stage (`language-chunker`) — `stropha adapters list --stage language-chunker` shows them. The `hybrid-rrf` retrieval adapter digests its stream composition into its `adapter_id`, so disabling/swapping any sub-stream forces a fresh id (drift hooks pick it up).

### 2.4 CLI commands

| Command | Purpose |
|---|---|
| `stropha index [--repo … / --manifest …] [--rebuild] [--vacuum] [--full / --incremental] [--since <sha>] [--enricher … --embedder …]` | Run the indexing pipeline (incremental by default when checkpoint exists). Add `--vacuum` to reclaim disk space after `--rebuild`. |
| `stropha search "<query>" [<repo_url>] [--top-k N] [--refresh]` | Hybrid retrieval, prints top-K. Accepts git URL for remote repos. |
| `stropha init [<target>] [--mcp-client]` | Initialize project with .stropha/, MCP config, and first index |
| `stropha stats` | Index metadata: chunks, models, repos, graphify mirror status |
| `stropha vacuum` | Reclaim unused disk space (runs SQLite VACUUM + WAL checkpoint). Use after `index --rebuild` to actually shrink the .db file. |
| `stropha pipeline {show,validate}` | Inspect / probe the resolved adapter composition |
| `stropha adapters list [--stage …]` | Enumerate registered adapters |
| `stropha eval [--top-k --tag --json --golden]` | Run golden dataset, report Recall@K + MRR (exits non-zero when <0.85) |
| `stropha watch [--interval --debounce --full-refresh]` | File-watcher soft index |
| `stropha cost [--log-path --json]` | Aggregate hook log into per-repo / per-adapter dashboard |
| `stropha hook {install,uninstall,status}` | Manage the post-commit hook (cross-repo via `--project-dir / --index-path / --log-path`) |
| `stropha glossary {add,remove,list,import,export,stats}` | Manage domain glossary (term definitions for better conceptual search) |

### 2.5 Configuration (YAML-first)

**BREAKING CHANGE (v7.0):** Query-time features are now configured via `stropha.yaml`, not environment variables. The following ENV vars are **removed**:
- `STROPHA_HYDE_ENABLED`, `STROPHA_HYDE_CODE_TUNED`, `STROPHA_HYDE_MODEL`
- `STROPHA_QUERY_ROUTER_LLM`, `STROPHA_QUERY_DECOMPOSITION_ENABLED`, `STROPHA_QUERY_DECOMPOSITION_LLM`
- `STROPHA_QUERY_REWRITE_ENABLED`, `STROPHA_MULTI_QUERY_ENABLED`, `STROPHA_MULTI_QUERY_COUNT`
- `STROPHA_RECURSIVE_RETRIEVAL`, `STROPHA_RECURSIVE_ADJACENCY`
- `STROPHA_QUERY_CACHE_ENABLED`, `STROPHA_QUERY_CACHE_SIZE`, `STROPHA_QUERY_CACHE_TTL`
- `STROPHA_MLX_MODEL`, `STROPHA_OLLAMA_MODEL`, `STROPHA_GRAPH_CONFIDENCE`, `STROPHA_GRAPH_FTS_AUGMENT`

#### stropha.yaml sections

```yaml
# Query-time retrieval features (all disabled by default)
retrieval_features:
  hyde_enabled: false           # Hypothetical document embeddings
  hyde_code_tuned: true         # Use code-specific HyDE prompts
  query_decomposition_enabled: false
  query_decomposition_llm: false
  query_router_llm: false
  query_rewrite_enabled: false
  multi_query_enabled: false
  multi_query_count: 3
  recursive_enabled: false      # Parent + adjacent-line auto-merge
  recursive_adjacency: 5
  cache_enabled: false          # Semantic query cache
  cache_size: 500
  cache_ttl: 3600

# LLM inference backend
inference:
  backend: null                 # "mlx", "ollama", "deepseek", "anthropic", or null for auto-detect
  mlx_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
  ollama_model: qwen2.5-coder:1.5b
  ollama_host: http://localhost:11434  # Also reads OLLAMA_HOST env
  # DeepSeek (cloud, opt-in — requires DEEPSEEK_API_KEY env var).
  # Auto-detection NEVER selects deepseek (local-first invariant §3.6);
  # set `backend: deepseek` explicitly to use it.
  deepseek_model: deepseek-chat       # or deepseek-reasoner
  deepseek_base_url: https://api.deepseek.com/v1  # also reads DEEPSEEK_BASE_URL env
  # Anthropic (cloud, opt-in — requires ANTHROPIC_API_KEY env var).
  # Same local-first invariant: never auto-selected. Set `backend: anthropic` explicitly.
  # Default model claude-haiku-4-5 is fast + cheap (good for per-chunk enrichment).
  anthropic_model: claude-haiku-4-5   # or claude-sonnet-4-5, claude-opus-4-7
  anthropic_base_url: https://api.anthropic.com/v1
  anthropic_version: "2023-06-01"     # Anthropic-Version header (pinned to last GA)
  # Indexing-time throughput controls (Phase 5):
  enricher_concurrency: 1             # in-flight LLM requests (ThreadPool, default 1=serial)
  enricher_batch_size: 1              # chunks packed per LLM call (default 1=per-chunk; only `contextual` enricher implements batching)

# Graph integration
graph:
  confidence: EXTRACTED         # EXTRACTED, INFERRED, or comma-separated
  fts_augment: true
```

#### Environment variables (still supported)

| Var | Default | Effect |
|---|---|---|
| `STROPHA_TARGET_REPO` | cwd | Default repo for `index` / MCP server |
| `STROPHA_INDEX_PATH` | `.stropha/index.db` | SQLite index path |
| `STROPHA_LOG_LEVEL` | `INFO` | structlog level |
| `VOYAGE_API_KEY` | unset | Enables the `voyage` embedder |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon (kept for ecosystem compat) |
| `DEEPSEEK_API_KEY` | unset | Required when `inference.backend: deepseek` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | Override for proxies / self-hosted gateways |
| `ANTHROPIC_API_KEY` | unset | Required when `inference.backend: anthropic` |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Override for proxies / self-hosted gateways |
| `STROPHA_GRAPHIFY_OUT` | `<repo>/graphify-out` | Override path to graphify output |
| `STROPHA_HOOK_SKIP=1` | unset | Bypass the post-commit hook |
| `STROPHA_HOOK_TIMEOUT` | `600` | Hook bg-process wall-clock (seconds) |
| `STROPHA_HOOK_LOG` | `~/.cache/stropha-hook.log` | Hook log path |
| `STROPHA_HOOK_UV` | discovered | Override `uv` binary path |
| `STROPHA_HOOK_GRAPHIFY` | discovered | Override `graphify` binary path |
| `STROPHA_HOOK_NO_GRAPHIFY=1` | unset | Skip the graphify step inside the hook |
| `STROPHA_HOOK_PROJECT_DIR` | baked / `$TOPLEVEL` | `uv run --directory` target (cross-repo) |
| `STROPHA_HOOK_INDEX_PATH` | baked / `.env` | Per-repo `STROPHA_INDEX_PATH` for the hook |
| `STROPHA_RENAME_THRESHOLD` | `80` | Git rename detection similarity threshold (0-100). Higher = stricter match. |
| `STROPHA_MCP_WATCH` | `1` | Enable auto-reindex file watcher in MCP server (debounced) |
| `STROPHA_MCP_WATCH_INTERVAL` | `1.0` | Polling interval in seconds for MCP watch |
| `STROPHA_MCP_WATCH_DEBOUNCE` | `2.0` | Debounce window in seconds (waits for quiet period before reindex) |
| `STROPHA_ADAPTIVE_WEIGHTING` | `1` | Enable adaptive query weighting in hybrid-rrf (symbol queries boost lexical, natural queries boost dense) |
| `STROPHA_GIT_URL_CACHE` | `~/.cache/stropha/repos` | Cache directory for git URL clones |
| `STROPHA_GIT_URL_TTL` | `86400` | Cache TTL for git URL repos in seconds (default 24h) |

Cross-repo hooks (v=3, v=4) bake `PROJECT_DIR_DEFAULT` / `INDEX_PATH_DEFAULT` / `LOG_DEFAULT` directly into the generated script — see `stropha hook install --help`. Env vars still override. Hook v=4 uses `--incremental` for git-diff aware ingestion.

### 2.6 Test inventory (1457 unit tests, ~22s)

Per file: highlights — `test_anthropic_backend` 38 (was 34, +4 retry-after / 429 tests) · `test_contextual_enricher` 31 (was 19, +12 batch / status / truncation-tolerance tests) · `test_pipeline_parallel_enrich` 16 (was 8, +8 batch / status tests) · `test_savings` 22 (was 17, +5 WIN-158 wiring) · `test_trace_recorder` 32 (NEW, WIN-159) · `test_traced_decorator` 35 (NEW, WIN-160) · `test_search_streams` 5 (NEW, WIN-160 stream attribution) · `test_artifact_store` 55 (NEW, WIN-161) · `test_outcome_labeling` 56 (NEW, WIN-162) · `test_feedback` 26 (NEW, WIN-163). Other files unchanged from the prior baseline.

## 3. Key invariants (do NOT break)

1. **Drift detection** — every config flag that changes adapter output MUST be baked into the adapter's `adapter_id` digest. The pipeline uses `(content_hash, embedding_model, enricher_id)` as the chunk freshness key; if you add a config knob, extend `adapter_id` so flipping it triggers re-process. Reference: `src/stropha/adapters/enricher/hierarchical.py::adapter_id`.
2. **Forward-only schema migrations** — `Storage._migrate()` uses `IF NOT EXISTS` / `_add_column_if_missing`. Never DROP without bumping `SCHEMA_VERSION`. The migration is run on every `Storage()` open, so it MUST be idempotent.
3. **`Pipeline.run()` owns graph + FTS post-processing order** — `Pipeline.run()` is the only call site for `GraphifyLoader → GraphVecLoader → Storage.augment_fts_with_graph()`. The order matters (structural → embedding → FTS) and you must not reorder it. Reference: `src/stropha/pipeline/pipeline.py::_refresh_graphify_mirror`.
4. **Graph-gated MCP tools never return silent empty** — when `graph_nodes` is empty, the `find_*` tools and `trace_feature` MUST return `{"graph_loaded": false, "message": …}`. The actionable message is part of the contract — agents rely on it to decide whether to call `graphify .`.
5. **Hook ≤ 100 ms before fork** — the post-commit hook MUST exit the foreground in under 100 ms so `git commit` returns instantly. All real work runs in a detached `nohup` + `flock`-guarded background. The hook itself never blocks the commit, never makes a network call before forking.
6. **Local-only by default** — no new mandatory cloud dependency. Voyage / Anthropic / OpenAI integrations stay optional (`[mlx]` / `VOYAGE_API_KEY` style). The default install on a fresh box does indexing + search + hook + graph traversal with zero network calls (after fastembed downloads weights once).
7. **`fts_chunks` is contentless-with-external-content** — every row inserted via explicit `INSERT INTO fts_chunks(rowid, content, rel_path) VALUES (?, ?, ?)`. The `_fts_text()` helper builds the document (content + identifier split + path tokens + symbol). `Storage.augment_fts_with_graph()` follows the same pattern: DELETE + INSERT, never UPDATE.
8. **Search-time toggles do not require re-index** — HyDE (`retrieval_features.hyde_enabled`) and recursive merge (`retrieval_features.recursive_enabled`) live in the query path only. Flipping them on/off costs zero — no re-embed, no drift detection trigger.

## Authoritative spec

**`docs/architecture/stropha-system.md`** is the master design doc (the 1.0 version that mapped the solution space). It's preserved as historical context; for "what's true today", read §2 above.

The companion specs:
- `docs/architecture/stropha-pipeline-adapters.md` — ADR for the Stage/adapter framework (Phase 1–4 shipped).
- `docs/architecture/stropha-graphify-integration.md` — RFC for graphify mirror + post-commit hook + L2/L3 augmentation (Fase 1.5a/b/c/e/f shipped, status `Implemented`).
- `docs/architecture/stropha-meta-harness.md` — RFC: implementation plan for a Meta-Harness over stropha (an outer loop optimizing the adapter search space), per Lee et al. (arXiv:2603.28052). Status `Proposed` — nothing shipped. Companion analysis: `docs/comparisons/stropha-vs-meta-harness.md`.

Each spec carries its own `Status` line. Update both when behaviour changes.

## 6. Chronological history (how we got here)

The sections below are the cumulative shipping log. New work appends to the bottom. For "what is true today" read §2 (snapshot); these sections answer "why is this code shaped this way?".

### Phase 0 / 1 — MVP MCP

Phase 0 (spike) ✓ — kept for reference:
- [x] `uv` project scaffold, walker, dummy chunker, embedder abstraction,
      sqlite-vec storage, CLI `index` / `search` / `stats`.

Phase 1 ✓:
- [x] Tree-sitter AST chunking for Java, TypeScript, JavaScript, Python,
      Rust, Go, Kotlin (via `tree-sitter-language-pack.process()`).
- [x] Custom chunkers for Vue SFC, Markdown (heading split), Gherkin.
- [x] Class/interface skeleton chunks: emit a meso-level chunk per
      container with qualified name + member list so BM25 matches the
      type by identifier.
- [x] Hybrid search (spec §6.1): **four streams** fused via RRF —
      dense (sqlite-vec) + sparse (FTS5 BM25, with CamelCase splitting +
      path/symbol tokens augmenting the FTS document) + symbol-token
      lookup (query routing per spec §6.3.5) + graph-vec (graphify embeddings).
- [x] MCP server (`stropha-mcp`) over stdio. Tools: `search_code`,
      `get_symbol`, `get_file_outline`, `list_repos`.
      Resource: `stropha://stats`.
- [x] `.mcp.example.json` template for Claude Code / Cursor integration.
- [x] Chunk-level freshness skip — re-running `index` on a stable repo
      is near-instant (no API calls, no DB writes).
- [x] Local embedder default: `mixedbread-ai/mxbai-embed-large-v1`
      (1024-dim, top open MTEB at this scale, ONNX-stable on macOS aarch64).
- [x] **Per-chunk repository identity** (schema v2): every chunk records
      the git repo it came from (`normalized_key`, clone URL, default
      branch, HEAD commit at index time). Returned in every `SearchHit`
      so MCP clients can `git clone <url>` to fetch the source. Schema
      auto-migrates from v1 with sanity-checked backfill.

Phase 4 (multi-repo) ✓:
- [x] `stropha index --repo A --repo B ...` walks each repo sequentially
      sharing the same Storage and Embedder.
- [x] `chunk_id` is namespaced by repo (`make_chunk_id(..., repo_key=...)`)
      so identical files in distinct repos do not collide on the global
      `chunks.chunk_id` UNIQUE constraint.
- [x] `IndexStats` aggregates per-repo counters (`stats.repos: list[RepoStats]`)
      while keeping single-repo back-compat accessors.
- [x] `--rebuild` clears chunks but preserves the `repos` table — identities
      survive rebuilds so FK references stay stable across runs.

Pipeline-adapters Phase 1 ✓ (per `docs/architecture/stropha-pipeline-adapters.md`):
- [x] `Stage` protocol + `StageContext`/`StageHealth` + `@register_adapter`
      decorator with auto-import discovery (`stropha.pipeline`,
      `stropha.adapters`).
- [x] `EmbedderStage` and `EnricherStage` protocols (`stropha.stages.*`).
      Legacy `stropha.embeddings.Embedder` Protocol kept as the minimal
      back-compat subset.
- [x] Migrated `LocalEmbedder` + `VoyageEmbedder` to
      `stropha.adapters.embedder.*` with `@register_adapter`. Old import
      paths (`stropha.embeddings.local`, `…voyage`) still resolve via
      shims.
- [x] New enricher adapters: `noop` (default — identity) and
      `hierarchical` (prepends parent skeleton).
- [x] Pydantic-validated config: YAML (`./stropha.yaml`,
      `~/.stropha/config.yaml`) + env-var overlay
      (`STROPHA_<STAGE>__<KEY>__<SUBKEY>`) + CLI flags
      (`--enricher`, `--embedder`). Legacy env vars
      (`STROPHA_LOCAL_EMBED_MODEL`, `STROPHA_VOYAGE_EMBED_MODEL`,
      `STROPHA_VOYAGE_EMBED_DIM`) keep working as aliases routed
      strictly by the resolved adapter.
- [x] Schema v3: `chunks.embedding_text` + `chunks.enricher_id` +
      `enrichments(content_hash, enricher_id)` cache table.
      Migration v1→v2→v3 is idempotent; legacy NULL `enricher_id`
      is treated as `'noop'` so upgrades do NOT trigger full re-embed.
- [x] Adapter-drift detection (ADR-004): switching the active enricher
      (e.g. `--enricher hierarchical`) automatically re-enriches +
      re-embeds the drifted chunks — no `--rebuild` needed. Per-chunk
      cache via `enrichments` makes repeat enrichment near-free.
- [x] New CLI: `stropha pipeline show` (resolved composition + health),
      `stropha pipeline validate` (probe), `stropha adapters list`.
- [x] 30 new unit tests (`test_pipeline_framework.py`,
      `test_enricher_adapters.py`, `test_pipeline_drift.py`) on top of
      the 53 pre-existing → 83 total, all green.

Pipeline-adapters Phase 2 ✓:
- [x] `WalkerStage` / `StorageStage` / `RetrievalStage` protocols.
- [x] `adapters/walker/git_ls_files.py`, `adapters/storage/sqlite_vec.py`
      (subclass of `Storage` so the full read/write surface is inherited),
      `adapters/retrieval/hybrid_rrf.py`.
- [x] `build_stages()` extended to all 5 stages with cross-stage
      injection (storage gets embedder.dim; retrieval gets storage +
      embedder).
- [x] `Pipeline` class uses walker + storage adapters; `cli.py` `index`
      / `search` / `stats` route through the builder; `pipeline show`
      drops the "(legacy)" rows for migrated stages.
- [x] Legacy env aliases extended: `STROPHA_INDEX_PATH` →
      `pipeline.storage.config.path`; `STROPHA_MAX_FILE_BYTES` →
      `pipeline.walker.config.max_file_bytes`.
- [x] 14 new unit tests (`test_phase2_adapters.py`).

Pipeline-adapters Phase 3 ✓:
- [x] `ChunkerStage` + `LanguageChunkerStage` protocols.
- [x] `adapters/chunker/tree_sitter_dispatch.py` — dispatcher adapter
      with sub-pipeline (language → sub-adapter) exposed via YAML
      `chunker.config.languages.*`.
- [x] Per-language sub-adapters under `adapters/chunker/languages/`:
      `ast-generic`, `heading-split`, `sfc-split`,
      `regex-feature-scenario`, `file-level`. All wrap the existing
      classes under `ingest/chunkers/` to preserve test coverage.
- [x] `Pipeline` builds + uses the chunker adapter; output is
      byte-identical (same `chunk_id`s) to the legacy `Chunker` for
      every supported language.
- [x] `pipeline show` / `pipeline validate` cover all 6 stages — no
      more "(legacy)" rows.
- [x] 11 new unit tests (`test_phase3_chunker.py`).

Pipeline-adapters Phase 4 ✓:
- [x] `RetrievalStreamStage` protocol; sub-adapters under
      `adapters/retrieval/streams/`: `vec-cosine`, `fts5-bm25`,
      `like-tokens`.
- [x] `hybrid-rrf` refactored to read `config.streams.{name: {adapter,
      config}}` and instantiate registered streams. Setting a stream to
      `null` disables it; omitted streams inherit the legacy default.
- [x] Performance: `hybrid-rrf.search` skips `embed_query` when no dense
      stream is enabled.
- [x] `adapter_id` of `hybrid-rrf` digests the stream composition so
      changing any sub-adapter forces a fresh `adapter_id` (cache /
      drift hooks pick it up).
- [x] 12 new unit tests (`test_phase4_retrieval_streams.py`). Total
      suite: **120 tests, all green**.

Graphify integration (per `docs/architecture/stropha-graphify-integration.md`):
- [x] **Fase 1.5a — Graph loader & SQLite schema v4**:
      `graph_nodes` / `graph_edges` / `graph_meta` tables, `GraphifyLoader`
      with idempotent + transactional load, mtime-based staleness, env
      var `STROPHA_GRAPH_CONFIDENCE` filter, `Storage.stats()['graph']`
      reporting. Auto-loaded by `Pipeline.run()` when graph is stale.
- [x] **Fase 1.5b — MCP tools** `find_callers`, `find_related`,
      `get_community`, `find_rationale` (all in
      `src/stropha/retrieval/graph.py`). Symbol resolution via exact →
      dotted-suffix → substring fallback. Returns
      `{graph_loaded: false, ...}` when the mirror is empty so the LLM
      gets actionable feedback. `find_rationale` only meaningful when
      `rationale_for` edges exist (graceful empty otherwise).
- [x] **Fase 1.5c — Hook installer CLI**: `stropha hook install /
      uninstall / status` with atomic write between markers, version
      detection (`v=2`), `core.hooksPath` honoured per RFC §6.3,
      coexistence warning when graphify-hook present.
- [x] **L2 augmentation — graph-aware enricher**:
      `adapters/enricher/graph_aware.py` prepends matching community
      label + node label to every chunk's `embedding_text`. Builder
      late-injects `Storage` so the enricher can query the graph mirror.
      `adapter_id` digests every flag → drift detection auto-rebuilds.
- [x] **MCP server name** changed to `stropha_rag` (FastMCP +
      `opencode.json`) so LLM clients understand the server is a RAG.
- [x] 71 new unit tests (`test_graphify_loader.py` 19,
      `test_graph_tools.py` 22, `test_hook_install.py` 17,
      `test_graph_aware_enricher.py` 13).

Phase 2 evaluation harness (per spec §16):
- [x] `src/stropha/eval/harness.py`: golden JSONL loader,
      `run_eval()` returning `EvalReport` with `Recall@K` + `MRR` +
      per-tag breakdowns. Tolerates both real `SearchHit` and mocked
      hits (so retrieval stack can be stubbed in tests).
- [x] `tests/eval/golden.jsonl`: ~30 baseline cases tagged by feature
      area (retrieval, storage, chunker, …).
- [x] CLI `stropha eval` exits non-zero when `Recall@K < 0.85` (CI hook).
- [x] 12 new unit tests (`test_eval_harness.py`).

Pipeline-adapters Phase 5 (in progress, local-only focus):
- [x] `enricher/ollama` — local LLM one-line summarisation, pure stdlib
      HTTP, fails gracefully (returns raw content on any error). 14
      mocked tests cover success, all failure paths, and health probes.
      Ollama installed locally via `brew install ollama` + `qwen2.5-coder:1.5b`
      pulled (~1 GB, sub-second per chunk on Apple Silicon).
- [x] `enricher/mlx` — native Apple Silicon inference via `mlx-lm` (lazy
      import, optional `[mlx]` extra). Lazy model load (`load()` called
      once, then cached across enrich calls). Falls back to raw content
      when mlx-lm missing or load fails. 15 mocked tests.
- [x] `walker/filesystem` — non-git directory walker, skips a default
      list of cache dirs (`.venv`, `node_modules`, `__pycache__`, …).
- [x] `walker/nested-git` — discovers nested `.git/` directories under
      an umbrella root for monorepos / vendored deps. Output rebased to
      the umbrella root. 13 walker tests.
- [x] `inference/anthropic_backend` — Anthropic Messages API backend.
      Opt-in (never auto-selected per CLAUDE.md §3.6). Used transparently
      by the existing `contextual` enricher / HyDE / query rewrite when
      `inference.backend: anthropic` is set. Default model
      `claude-haiku-4-5` (fast + cheap for per-chunk enrichment).
      34 mocked tests cover success, multi-block responses, 4xx vs 5xx
      retry policy, exponential backoff, malformed JSON, x-api-key
      header, and graceful fallback when ANTHROPIC_API_KEY is missing.
- [ ] `enricher/openai` (cloud — skipped per local-only directive).
- [ ] Storage variants (`qdrant`, `pgvector`, `lancedb`).
- [ ] Retrieval `hybrid-rrf-rerank` (Voyage rerank-2.5 — cloud, skipped).

Trilha A L3 — graph node embeddings (4th RRF stream):
- [x] Schema v5: `graph_nodes.embedding` BLOB + `embedding_model` +
      `embedding_dim` columns.
- [x] `GraphVecLoader` (idempotent: skips nodes whose stored embedding
      model matches the active embedder; re-embeds on model change).
- [x] `retrieval-stream/graph-vec` — brute-force cosine over packed
      float32 BLOBs (sub-ms for ≤50K nodes), hydrates SearchHits via
      chunk lookup (rel_path + line containment). Auto-runs from
      `Pipeline.run()` after the structural mirror reload.
- [x] 16 unit tests (`test_graph_vec.py`).

Phase 3 — `trace_feature` tool:
- [x] `trace_feature(feature, max_paths, max_depth)` walks outbound
      `calls` edges DFS from token-overlap entry points. Cycle-safe.
      Surfaces full chain (entry → step → method → ...). Wired as 9th
      MCP tool (`stropha_rag.trace_feature`). 4 unit tests.

Phase 4 — declarative multi-repo manifest:
- [x] `stropha index --manifest repos.yaml` accepts a YAML
      `{repos: [{path, enabled}]}` declaration. Relative paths resolved
      from the manifest's directory; `~` expanded; `enabled: false`
      skipped. Mutually exclusive with `--repo`. 12 unit tests.

Trilha A — completing the RFC (post-1.5):
- [x] **`find_tests_for`** (10th MCP tool). Path-pattern heuristic catches
      `test_*`, `*_test`, `*.spec.*`, `*.test.*`, `/tests/`, `/test/`;
      override via param. 4 new tests in `test_graph_tools.py`.
- [x] **`Storage.augment_fts_with_graph()`** — RFC §1.5e retroactive L2.
      Runs after every index pass (toggle `STROPHA_GRAPH_FTS_AUGMENT=1`,
      default on). Idempotent: re-running produces same FTS5 state. 8
      tests in `test_fts_augment.py`.
- [x] **Doc hygiene**: RFC bumped from `Proposed` → `Implemented`; all
      §9 phase tables flipped from `TODO` → `done` (32 cells);
      system spec §16 Phase 2/3/4 checkboxes updated to reflect reality.

Phase 3 — Sofisticação (local-only):
- [x] **HyDE query rewrite** (`STROPHA_HYDE_ENABLED=1`). Routes the
      query through Ollama (`qwen2.5-coder:1.5b` default) and embeds
      the hypothetical doc on the dense stream only. BM25 + symbol
      lanes keep the literal query. Fails gracefully → raw query
      fallback. 6 tests in `test_hyde_and_recursive.py`.
- [x] **Recursive retrieval / auto-merging** (`STROPHA_RECURSIVE_RETRIEVAL=1`).
      Two passes: (1) parent promotion when 2+ siblings of the same
      `parent_chunk_id` hit; (2) adjacency merge when chunks on the
      same file are within `STROPHA_RECURSIVE_ADJACENCY=5` lines.
      Cycle-safe, score-preserving. 10 tests.
- [x] **File-watcher soft index** (`stropha watch`). stdlib polling,
      debounce 2s, honours .gitignore. 8 tests in
      `test_watch_and_bge_m3.py`.
- [x] **Cost dashboard** (`stropha cost`). Aggregates hook log +
      structlog into per-repo / per-adapter / per-graph tables. JSON
      or rich.Table output. 11 tests in `test_cost.py`.

Phase 4 — Escala:
- [x] **`bge-m3` embedder adapter** — Pre-configured local fastembed
      backend pinned to `BAAI/bge-m3`, the recommended multilingual
      local fallback per spec §15.

Incremental indexing (Phase A/B/C):
- [x] **Schema v6 — file-level dirty cache** — `files` table stores
      `(mtime, size_bytes, content_hash, enricher_id, embedder_model)` per
      file for O(1) freshness check. 10x faster no-op index runs.
- [x] **Git-diff aware ingestion** — `GitDiffWalker` + `FileDelta` model.
      Hook v=4 passes `--incremental`. Only touched files visit chunker.
- [x] **Rename-resilient chunk_id** — `Storage.rename_chunks()` recomputes
      chunk_id + FTS5 rows. Zero re-embed cost for renames.
- [x] **GraphifyLoader diff-load** — deltas only, not full DELETE+INSERT.

Anthropic 429 / batched enrichment (rate-limit hardening):
- [x] **Fix 1 — 429 retry with retry-after** —
      `AnthropicBackend.generate` no longer treats HTTP 429 as permanent.
      Reads the `retry-after` header (RFC 7231 seconds form), capped at
      60s to defeat misconfigured proxies. Falls back to exponential
      backoff when the header is absent. Other 4xx (401, 400, …) remain
      permanent. The previous TODO comment that admitted "treating it
      as permanent" is gone. Reference:
      `src/stropha/inference/anthropic_backend.py::generate`.
- [x] **Fix 2 — failed enrichments don't poison the cache** —
      `ContextualEnricher.enrich_with_status(chunk, ctx) -> tuple[str,
      bool]` returns `success=False` when the LLM call comes back empty.
      Helper
      `stropha.stages.enricher.enrich_with_status(enricher, chunk, ctx)`
      dispatches to the override when present, defaults to `True` for
      legacy enrichers (noop, hierarchical, graph-aware) which never
      fail. `Pipeline._enrich_chunks` returns `list[tuple[str, bool]]`;
      the call site skips `Storage.put_enrichment` when success=False,
      so the chunk is re-attempted on the next index run instead of
      being cached as "already enriched". New stat counter
      `RepoStats.chunks_enriched_failed`.
- [x] **Fix 3 — batched enrichment** —
      `ContextualEnricher.enrich_batch(chunks, contexts)` packs N chunks
      into a single Messages API call by asking the model to return a
      JSON array of descriptions (one per chunk, ID-matched). Tolerates
      markdown fences, trailing commas, partial responses,
      hallucinated extra IDs, **and truncated arrays** (the response was
      cut off mid-string by ``max_tokens`` — observed in the wild against
      DeepSeek with batch_size=10). The truncation-tolerant pass takes
      everything from the opening ``[`` to the last complete ``}``,
      synthesises a closing ``]``, and parses — recovering N-1 entries
      when the model ran out of tokens on the last one. Every failure
      mode falls back to per-chunk `(content, False)` so Fix 2's
      cache-skip logic keeps retries on the table. Combines with
      `enricher_concurrency`: N parallel batches of M chunks = N×M chunks
      per round-trip cycle. Batched ``max_tokens`` budget is
      ``(per_chunk_tokens + 32) × N + 256`` to absorb DeepSeek's verbose
      preamble + code-fence wrappers without hitting the truncation path.
      Knobs: `inference.enricher_batch_size: int = 1` (YAML) /
      `--batch-size N` (CLI). The pipeline auto-falls back to per-chunk
      dispatch for enrichers without `enrich_batch` (noop, hierarchical,
      ollama, mlx, graph-aware).

**Total suite: 1457 tests, all green. 5 enrichers, 4 retrieval streams,
4 walkers, 3 embedders, 4 inference backends (mlx, ollama, deepseek, anthropic),
12 MCP tools, 8 CLI commands.**

### Exit criteria status

- **Phase 0** exit: `stropha search "where is the FSRS calculator"` returns the right file in top 3 — ✓.
- **Phase 1** floor: hybrid search returns the right chunk in top 3 for symbol + conceptual queries — ✓.
- **Phase 2** golden-set floor (Recall@10 ≥ 0.85): ✓ on the shipped 30-query set; full 50-query target deferred (incremental).

### Pending work (curated, local-only)

> Cloud-only items (Voyage rerank-2.5, openai backend, qdrant / pgvector, Web UI, OAuth) are explicitly deferred per the local-only directive. **Anthropic** is shipped as an opt-in inference backend (see Phase 5 above) — usable transparently by the existing `contextual` enricher when `inference.backend: anthropic` is set.

#### Retrieval Quality Roadmap (State-of-the-Art)

Current benchmark (mimoria golden set, 43 queries): symbol-lookup 80%, conceptual 80%, multi-hop 20% (needs graph), natural-language 20%, compound 100%.

**Priority 1 — Query Intelligence (highest ROI, ~3 days total):**

| Feature | Effort | Expected Gain | Status |
|---------|--------|---------------|--------|
| Query routing to graph tools | 2d | +40% multi-hop | **done** |
| Query decomposition + sub-query fusion | 1d | +25% natural-language | **done** |

Query routing classifies intent ("what calls X" → `find_callers`, "tests for X" → `find_tests_for`) and dispatches to the appropriate tool. Implemented as `smart_search` MCP tool with pattern-based classification + optional LLM fallback (`retrieval_features.query_router_llm: true`). Query decomposition splits complex queries into atomic sub-queries, retrieves each, and fuses via RRF. Enable with `retrieval_features.query_decomposition_enabled: true` or use LLM for ambiguous cases with `retrieval_features.query_decomposition_llm: true`.

**Priority 2 — Contextual Retrieval (Anthropic method, ~3 days):**

| Feature | Effort | Expected Gain | Status |
|---------|--------|---------------|--------|
| Contextual enricher (full implementation) | 3d | +25% conceptual | **done** |

Prepend LLM-generated context to each chunk before embedding. The context describes what the chunk does and how it fits in the file. Generated once at index time, stored in `embedding_text`. Use `--enricher contextual` to enable. Anthropic reported 49% reduction in retrieval failures.

**Priority 3 — HyDE Improvements (~1 day):**

| Feature | Effort | Expected Gain | Status |
|---------|--------|---------------|--------|
| Code-tuned HyDE prompts | 0.5d | +15% conceptual | **done** |
| Larger model support (deepseek-coder-v2, codestral) | 0.5d | +10% conceptual | pending |

Code-tuned HyDE prompts automatically select the best prompt template based on query type (function, class, how, where, test, error). Enable with `STROPHA_HYDE_ENABLED=1` (code-tuned prompts are on by default). Disable code-tuned prompts with `STROPHA_HYDE_CODE_TUNED=0`.

**NOTE (v7.0 breaking change):** These ENV vars are deprecated. Use `stropha.yaml` instead:
```yaml
retrieval_features:
  hyde_enabled: true
  hyde_code_tuned: true
```

**Priority 4 — Late Chunking (~1 week):**

| Feature | Effort | Expected Gain | Status |
|---------|--------|---------------|--------|
| Late chunking with contextual embeddings | 5d | +20% conceptual | pending |

Jina AI's technique: embed the full document first, then pool token embeddings per chunk boundary. Each chunk embedding retains awareness of siblings. Requires long-context embedder (jina-embeddings-v3, 8K tokens).

**Priority 5 — Advanced Retrieval (~2 weeks):**

| Feature | Effort | Expected Gain | Status |
|---------|--------|---------------|--------|
| ColBERT late interaction | 10d | +20% all queries | pending |
| SPLADE learned sparse retrieval | 5d | +12% conceptual | pending |

ColBERT stores per-token embeddings, computes MaxSim at query time. SPLADE produces sparse vectors with learned term expansion. Both require new retrieval infrastructure.

#### Infrastructure & Tooling

**High ROI, small effort:**
- E2E test do graph load (~0.5d) — fixture under `tests/eval/` with a real `graphify-out/` snapshot.
- +30 golden queries for mimoria (~0.5d) — balanced across symbol/conceptual/multi-hop/natural-language.
- Benchmark harness improvements (~0.5d) — add Precision@K, NDCG, per-query latency tracking.

**Medium effort:**
- LanceDB storage adapter (~1d) — embedded alternative to sqlite-vec; same `StorageStage` protocol.
- Indexação on-demand de dependências externas (~1d) — `stropha index --include-deps <name>` walks `node_modules/<name>` / vendored deps.

**Deferred (infra dependency or out of scope):**
- OpenTelemetry tracing → Langfuse — needs Jaeger/Langfuse decision; `structlog` + `stropha cost` cover the immediate need.
- RAGAS-style answer-quality evaluation — requires an LLM judge; local Recall@K + MRR cover the regression-guard use case.

## How to run

```bash
# One-time setup
uv sync

# Copy and edit env
cp .env.example .env
# (optional) set VOYAGE_API_KEY for best quality; otherwise local fallback is used.

# Index the target repo (defaults to current working dir; override via STROPHA_TARGET_REPO)
uv run stropha index

# Search
uv run stropha search "how does mastery work"

# Stats
uv run stropha stats
```

## Code conventions

- **Python 3.12+**, type hints mandatory on public functions, `pydantic` for structured data.
- **Async by default** in IO-bound paths (embedding API, file reads). CPU-bound stays sync.
- **No global state.** Pass `Config` / `Storage` explicitly. Single composition root in `cli.py`.
- **Errors are structured.** Raise `stropha.errors.StrophaError` subclasses; never bare `Exception`.
- **Logging via `structlog`.** Always include `request_id` / `chunk_id` when relevant; never log raw secrets.
- **File references in messages.** Use `path:line` format so editors can jump (e.g. `walker.py:42`).
- Line length 100. `ruff` + `mypy` clean in `dev` extra.

## Architecture map (current — keep in sync with §2)

Grouped by concern. Mirrors the actual filesystem layout under `src/stropha/`. The §2 snapshot lists adapter names + adapter_id shapes; this table maps **files** to **responsibilities**.

### Composition / entry points

| Module | Responsibility |
|---|---|
| `cli.py` | Typer composition root — wires every CLI subcommand (`index`, `search`, `stats`, `pipeline`, `adapters`, `eval`, `watch`, `cost`, `hook`). |
| `server.py` | FastMCP entry (`stropha-mcp`). Registers the 11 MCP tools + the `stropha://stats` resource. Lifespan opens Storage + builds SearchEngine. |
| `config.py` | Pydantic settings; legacy env-var aliases routed to the pipeline config tree. |
| `errors.py` | `StrophaError` hierarchy — every operational path raises a subclass. |
| `logging.py` | `structlog` config. JSON logs in non-TTY contexts. |
| `models.py` | Pydantic data: `Chunk`, `SourceFile`, `SearchHit`, `RepoRef`, `RepoStats`. |

#### File and Repository Identification

Every file in the index is identified by its **git repository** (not just the filesystem folder). The repository identity includes:

- **`normalized_key`**: Stable cross-user key (e.g., `github.com/user/repo`). For local-only repos: `local:<path>`. For non-git dirs: `path:<path>`.
- **`url`**: HTTPS clone URL for remote access (e.g., `https://github.com/user/repo.git`). `None` for local-only repos.
- **`local_path`**: Absolute filesystem path to the repo root on the indexer host (e.g., `C:\sources\myproject`).
- **`default_branch`**: Default branch name (e.g., `main`).
- **`head_commit`**: HEAD SHA at index time.

File paths (`rel_path` in `Chunk`, `SearchHit`, etc.) are **always relative to the git root**, not absolute filesystem paths. To get the full local path:
```
full_path = repo.local_path + "/" + hit.rel_path
```

This design ensures:
1. Files are uniquely identified across multiple indexed repos
2. Clients can access files via git clone (using `url`) or direct filesystem access (using `local_path`)
3. The same logical repo accessed via different URLs (SSH vs HTTPS) collapses to one identity

### Pipeline / adapter framework (`pipeline/`, `stages/`, `adapters/`)

| Module | Responsibility |
|---|---|
| `pipeline/base.py` | `Stage` protocol, `StageContext`, `StageHealth` types. |
| `pipeline/registry.py` | `@register_adapter` decorator + lookup. Auto-imports `stropha.adapters` on first use. |
| `pipeline/config.py` | YAML + env + CLI cascade → resolved config dict. Legacy aliases live here. |
| `pipeline/builder.py` | `build_stages(resolved)` → `BuiltStages` (one of each stage). Late-injects Storage into graph-aware enricher. |
| `pipeline/pipeline.py` | `Pipeline.run(rebuild)` — orchestrator. **Owns the post-index order**: graphify_loader → graph_vec_loader → augment_fts. |
| `stages/{walker,chunker,enricher,embedder,storage,retrieval,retrieval_stream}.py` | One Protocol per stage. Adapters import + implement. |
| `adapters/walker/{git_ls_files,filesystem,nested_git}.py` | 3 walker adapters. |
| `adapters/chunker/tree_sitter_dispatch.py` + `adapters/chunker/languages/*.py` | Chunker dispatcher + 5 language sub-adapters (`ast-generic`, `file-level`, `heading-split`, `sfc-split`, `regex-feature-scenario`). |
| `adapters/enricher/{noop,hierarchical,graph_aware,ollama,mlx}.py` | 5 enrichers. `graph_aware` reads `graph_nodes` (Storage late-injected). `ollama` + `mlx` are LLM-backed with graceful fallback. |
| `adapters/embedder/{local,voyage,bge_m3}.py` | 3 embedders. `bge_m3` is a pre-configured subclass of `local`. |
| `adapters/storage/sqlite_vec.py` | Subclass of the legacy `Storage` so the full read/write surface is inherited. |
| `adapters/retrieval/hybrid_rrf.py` + `adapters/retrieval/streams/*.py` | Hybrid coordinator + 4 stream sub-adapters (`vec-cosine`, `fts5-bm25`, `like-tokens`, `graph-vec`). |

### Ingestion (`ingest/`)

| Module | Responsibility |
|---|---|
| `ingest/walker.py` | Legacy walker (still wrapped by `git-ls-files` adapter). `.strophaignore` + binary/size filters. |
| `ingest/git_meta.py` | Repo identity: normalized URL, default branch, HEAD. Strips auth tokens. |
| `ingest/chunker.py` + `ingest/chunkers/*.py` | Legacy chunker family (wrapped by language sub-adapters). |
| `ingest/pipeline.py` | Legacy `IndexPipeline` — kept for back-compat tests; production path uses `pipeline/pipeline.py`. |
| `ingest/graphify_loader.py` | Mirror `graphify-out/graph.json` into `graph_nodes` / `graph_edges` / `graph_meta`. Idempotent + transactional. |
| `ingest/graph_vec_loader.py` | Embed `graph_nodes.label` via active embedder. Skip-fresh by `embedding_model`. |
| `ingest/manifest.py` | YAML manifest loader for `stropha index --manifest repos.yaml`. |

### Storage / retrieval (`storage/`, `retrieval/`)

| Module | Responsibility |
|---|---|
| `storage/sqlite.py` | SQLite + sqlite-vec + FTS5. Schema v1→v5. `_fts_text()` builds the FTS5 document. `augment_fts_with_graph()` does retroactive L2. |
| `retrieval/rrf.py` | RRF fuse (k=60). |
| `retrieval/search.py` | `SearchEngine` — composes 4 streams from `Storage` + optional HyDE rewrite + optional recursive merge. Used by CLI; MCP uses `hybrid-rrf` adapter (same logic). |
| `retrieval/graph.py` | Pure-SQL graph traversal — backs all 6 graph MCP tools. |
| `retrieval/hyde.py` | Hypothetical doc rewrite via Ollama (toggle `STROPHA_HYDE_ENABLED`). |
| `retrieval/recursive.py` | Parent promotion + adjacency merge (toggle `STROPHA_RECURSIVE_RETRIEVAL`). |

### Supporting (`eval/`, `tools/`, root)

| Module | Responsibility |
|---|---|
| `eval/harness.py` | Golden JSONL loader + Recall@K + MRR. CLI: `stropha eval`. |
| `tools/hook_install.py` | Post-commit hook installer (template at `_render_hook_block`, atomic write, `core.hooksPath` honoured). |
| `watch.py` | `stropha watch` — stdlib polling, debounce. |
| `cost.py` | `stropha cost` — log aggregator. |
| `embeddings/{base,local,voyage}.py` | Back-compat shims. Real implementations live under `adapters/embedder/`. |

## Embedding strategy

Three adapters behind `EmbedderStage` (`stropha.stages.embedder.EmbedderStage`):

1. **`voyage`** (`adapters/embedder/voyage.py`) — Voyage `voyage-4-large` @ 512 dims (Matryoshka) by default; `voyage-code-3` remains available as a code-SOTA opt-in (system spec §4.1). Activates when `VOYAGE_API_KEY` is set. Cloud only. **Token-aware sub-batching**: respects Voyage's 120K-tokens-per-batch ceiling by splitting incoming pipeline batches into sub-batches sized against `max_tokens_per_batch` (default 96K) using a `chars_per_token` heuristic (default 3.5). Both knobs are intentionally absent from `adapter_id` — they shape requests, not vectors, so toggling them never triggers re-embed.
2. **`local`** (`adapters/embedder/local.py`) — fastembed (ONNX). Default model `mixedbread-ai/mxbai-embed-large-v1` @ 1024 dims. Top open-source English MTEB at this scale, ONNX-stable on macOS aarch64. Zero cost, zero network after the first download. Configurable via the adapter config's `model:` field or the legacy `STROPHA_LOCAL_EMBED_MODEL` env alias.
3. **`bge-m3`** (`adapters/embedder/bge_m3.py`) — Pre-configured local fastembed pinned to `BAAI/bge-m3` (multilingual). Subclass of `LocalEmbedder` with `adapter_name="bge-m3"` so the drift detector treats it as a distinct adapter even when the underlying model overlaps with `local`'s config.

> **Avoid `jinaai/jina-embeddings-v2-base-code`** as a local model on macOS aarch64. It exhibits an ONNX-runtime instability that hangs or crashes the process on the second consecutive embed call. See `stropha-graphify-integration.md` ADR-008.

Every adapter writes `(embedding_model, embedding_dim)` into each chunk row. Switching providers does **not** invalidate the index; rows from a different model are ignored at query time until re-index. `stropha stats` prints a warning when mixed models are detected.

## Testing

```bash
uv run pytest -q                 # 1457 unit tests, ~22s
uv run pytest -m e2e             # end-to-end (skipped by default; needs fixture)
uv run pytest --collect-only -q  # full inventory by test id

uv run stropha eval --top-k 10   # golden dataset harness (Recall@K + MRR)
uv run stropha eval --json       # CI-friendly JSON (exits non-zero when Recall@K < 0.85)
```

The golden dataset is at `tests/eval/golden.jsonl` (~30 queries; see §2.6 for the per-file unit-test breakdown).

## Hygiene rules for agents

1. **Never invent symbols or APIs.** If unsure whether `voyageai` has `embed_async`, read the installed source under `.venv/`.
2. **Never bypass the adapter framework.** New embedders, walkers, chunkers, enrichers, storage backends, retrieval streams MUST register via `@register_adapter(stage=..., name=...)` and implement the corresponding protocol.
3. **Never write to the index outside `storage/sqlite.py`.** All schema migrations go through `Storage._migrate()` — bump `SCHEMA_VERSION` and add a forward-only step. `_add_column_if_missing` is the helper to use.
4. **Never silently empty the find_* tools.** When the graphify mirror is empty, return `{"graph_loaded": false, "message": …}` with an actionable suggestion. Reference: `src/stropha/server.py::_graph_unavailable`.
5. **Never index secrets.** The walker excludes `.env*`; if you add a new file type, confirm gitleaks-style scanning is in place before merge.
6. **Spec drift = update the spec in the same commit.** All three specs in `docs/architecture/` carry a `Status` line. Update both the line and the relevant section when behaviour changes.
7. **The hook is non-negotiably non-blocking.** `.git/hooks/post-commit` exits in under 100 ms (just enough to fork + disown). All real work runs in `nohup` + `flock` background. Anything else regresses commit UX.

## Useful pointers

| Resource | Purpose |
|---|---|
| `docs/architecture/stropha-system.md` | Master design doc (v1.0 — solution space mapping) |
| `docs/architecture/stropha-pipeline-adapters.md` | ADR for the Stage / adapter framework (status `Implemented`) |
| `docs/architecture/stropha-graphify-integration.md` | RFC for graphify mirror + post-commit hook (status `Implemented`) |
| `docs/architecture/stropha-meta-harness.md` | RFC — Meta-Harness implementation plan (status `Proposed`; 7 phases, ~3–4wk MVP) |
| `docs/comparisons/stropha-vs-meta-harness.md` | Concept analysis — stropha as a harness + 118 improvement options |
| `https://arxiv.org/abs/2603.28052` | Meta-Harness paper (Lee et al., 2026) — the source for both docs above |
| `graphify-out/GRAPH_REPORT.md` | Live structural map — god nodes + community labels (regenerated on every commit by the hook) |
| `graphify-out/graph.json` | Raw graph (1500+ nodes — backs the MCP `find_*` tools) |
| `~/.cache/stropha-hook.log` | Hook execution log; `~/.cache/stropha-hook-*.log` for cross-repo installs |
| `AGENTS.md` | OpenCode-specific graphify rules |
| `README.md` | Public pitch + install + mermaid diagrams |
| `https://modelcontextprotocol.io/specification` | MCP protocol reference |
| `https://github.com/asg017/sqlite-vec` | sqlite-vec docs |
| `https://docs.voyageai.com` | Voyage embedder docs |
