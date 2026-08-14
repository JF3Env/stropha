# Performance Tuning Guide

Performance depends on repository shape, hardware, models, storage, and enabled
query features. This guide avoids fixed benchmark claims and focuses on controls
that are present in the current implementation.

## Establish A Baseline

Start with the default local pipeline:

```bash
uv sync --extra local
uv run stropha pipeline show
uv run stropha pipeline validate
uv run stropha index
uv run stropha search "a representative query"
```

The no-key default uses the local embedder, the hierarchical enricher, three RRF
streams, and no reranker. Measure this baseline on a representative golden set
before enabling features that add model calls.

For automated tuning, keep selection and validation separate. `stropha meta run`
pins the source, builds or reuses isolated candidate indexes, evaluates the
active configuration plus proposal rounds on `search`, and freezes the Pareto
shortlist. `stropha meta validate <run-id>` then opens holdout for that baseline
and shortlist. Promotion is blocked unless the live baseline and candidate have
comparable search and holdout evidence and the frozen runtime identity still
matches.

```bash
uv run stropha meta run --iterations 5 --candidates 3 --parallel 2
uv run stropha meta validate <run-id>
uv run stropha meta promote <candidate-id> --run-id <run-id>  # review diff
uv run stropha meta promote <candidate-id> --run-id <run-id> --apply
```

Run-level retrieval, indexing, proposer, verification, and artifact usage is in
the meta directory's `runs/<run-id>/cost.json`. `stropha cost` below is a
separate dashboard over structured hook logs.

## Configuration

Tune settings only in `~/.stropha/config.yaml`. Environment-variable controls
for embedders, query features, file size, index path, and log level are not
supported.

```yaml
logging:
  level: INFO

walker:
  max_file_bytes: 524288

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

Query-time features apply to new requests without re-indexing. When the daemon
is running, use the dashboard's `PATCH /api/config` path for live replacement or
restart the daemon after editing the YAML directly.

## Large Repositories

### Exclude Generated Content

Use `.strophaignore` for generated, vendored, or minified content:

```text
**/node_modules/**
**/dist/**
**/build/**
**/*.min.js
**/*.bundle.js
**/vendor/**
**/__pycache__/**
```

### File Size Limit

The default maximum file size is 524,288 bytes, or 512 KiB. Increase it only for
source files that should genuinely be indexed:

```yaml
walker:
  max_file_bytes: 1048576

pipeline:
  walker:
    adapter: git-ls-files
    config:
      max_file_bytes: 1048576
```

The top-level value supplies the general walker setting, while the pipeline
adapter block makes the selected adapter's value explicit.

### Select Repositories Explicitly

```bash
uv run stropha index \
  --repo ./packages/frontend \
  --repo ./packages/backend \
  --repo ./packages/shared
```

For a reusable list:

```yaml
# repos.yaml
repos:
  - path: packages/frontend
  - path: packages/backend
  - path: packages/shared
  - path: packages/deprecated
    enabled: false
```

```bash
uv run stropha index --manifest repos.yaml
```

## Embedders

### Local

The local adapter is the normal no-key default and currently uses
`mixedbread-ai/mxbai-embed-large-v1` unless configured otherwise. Model weights
download on first use.

Hardware-specific adapters have different optional dependencies. Inspect the
live registry and schema rather than relying on a copied adapter inventory:

```bash
uv run stropha adapters list --stage embedder
uv run stropha pipeline show
```

### Voyage

Voyage is an optional cloud adapter, not stropha's overall default. Install the
extra, put the API key under `secrets.voyage_api_key`, and select the adapter in
`~/.stropha/config.yaml`. The adapter's current model default is
`voyage-4-large` with a 512-dimensional output.

```yaml
pipeline:
  embedder:
    adapter: voyage
    config:
      model: voyage-4-large
      dim: 512

secrets:
  voyage_api_key: <key>
```

Changing the primary model or dimension changes the vector space. Run an index
pass; if stropha reports a dimension conflict, use `stropha index --rebuild`.
A dimension rebuild clears all derived search data across the shared backend,
so include every repository that must remain searchable.

## Enrichers

| Enricher | Behavior | Main cost |
|---|---|---|
| `noop` | Embeds raw chunk content | Lowest indexing work |
| `hierarchical` | Adds deterministic parent context | Small local CPU cost |
| `graph-aware` | Adds graph labels and parent context | Requires a loaded graph |
| `contextual` | Generates semantic context through `inference.backend` | One or batched LLM calls during indexing |
| `ollama` | Calls Ollama directly for a concise summary | Local Ollama inference |

The `contextual` and `ollama` adapters are distinct. Select `contextual` when
you want unified backend selection; select `ollama` only when the direct Ollama
adapter's behavior is intended.

Generated enrichment is cached by content and adapter identity. Changing an
enricher causes affected chunks to be enriched and embedded again on the next
index pass.

## Query-Time Features

HyDE, query rewriting, and multi-query expansion invoke the configured inference
backend and can dominate interactive latency. Recursive merging performs extra
retrieval work but no LLM call. Keep features disabled until a representative
evaluation shows that their additional recall is useful.

```yaml
retrieval_features:
  hyde_enabled: true
  query_rewrite_enabled: false
  multi_query_enabled: false

pipeline:
  retrieval:
    adapter: hybrid-rrf
    config:
      recursive_enabled: true
```

The semantic query-cache classes are not integrated into the production search
path. Do not enable `cache_enabled` expecting lower query latency, and do not use
cache hit-rate claims for capacity planning.

## Reranking

No reranker is the current default and the recommended starting point. Rerankers
can improve some datasets while demoting exact code matches on others.

The local `cross-encoder` adapter defaults to `BAAI/bge-reranker-base` and
requires the `local` extra. The `llm` adapter uses the configured inference
backend. Both add work after RRF fusion.

```yaml
pipeline:
  retrieval:
    adapter: hybrid-rrf
    config:
      candidate_k: 50
      reranker:
        adapter: cross-encoder
        config:
          model: BAAI/bge-reranker-base
```

Treat this as an experiment, not a universal recommendation. Compare against
the no-reranker baseline with the target repository's golden set.

## Retrieval Streams And Memory

The default retrieval uses dense, BM25, and symbol streams. `graph-vec` and
`model2vec-cosine` are opt-in, allowing up to five configured streams.

The graph-vector stream materializes float32 node embeddings. The raw matrix
size is:

```text
node_count * embedding_dimension * 4 bytes
```

For example, 50,000 nodes at 512 dimensions require about 98 MiB for the raw
matrix, before array, metadata, storage-driver, and process overhead. The old
estimate of roughly 50 bytes per node omitted the embedding dimension and was
not usable for sizing.

Model2Vec uses separate 256-dimensional tables and does not rebuild or replace
the primary vectors.

## Watcher And Hook

The MCP daemon starts a polling watcher by default after its shared runtime is
first built. It debounces saves and writes changed working-tree files to the
persistent index. This is not an in-memory overlay. The lightweight refresh does
not update graphify data or graph vectors on every save.

The post-commit hook runs detached and uses normal index auto mode. Check its
resolved location and installed version with:

```bash
uv run stropha hook status
```

Re-running `stropha hook install --force` updates the managed block. During a
large rebase, `STROPHA_HOOK_SKIP=1` remains an explicit hook-process escape
hatch; run `stropha index` afterward.

## Storage And Stats

`stropha stats` opens the configured backend and reports backend location,
database size, chunks, files, embedding identities, enrichers, and repositories:

```bash
uv run stropha stats
```

For SQLite, opening a missing path can initialize an empty database. Verify
nonzero counts and the expected repository instead of treating command success
or file existence as proof of a completed index.

SQLite WAL mode is managed by stropha. Do not delete `-wal` or `-shm` files to
improve performance or clear locks. PostgreSQL sizing and maintenance belong to
PostgreSQL tooling; do not apply SQLite file instructions to that backend.

Use `stropha vacuum` only for SQLite space reclamation after large deletions.

## Profiling

Set logging in the user YAML, restart after a direct edit, and inspect structured
stage timings:

```yaml
logging:
  level: DEBUG
```

```bash
uv run stropha cost
uv run stropha cost --json
```

Measure indexing and query behavior on the actual repository. Hardware labels
and model names alone are not enough to support portable latency or memory
claims.

## Detached Benchmark Campaigns

The bundled comprehensive campaign runs broad discovery on the development
`search` split, repeats stochastic quality stages, and finishes with focused
validation on the frozen `holdout` split in a detached process:

```bash
uv run stropha bench campaign start
uv run stropha bench campaign status
```

The launcher pins the current commit in a detached worktree, so later edits to
the main checkout do not invalidate a multi-day run. Results, logs, PID/state,
per-stage CSV/JSON reports, the final summary, and a human-readable `report.md`
are stored under
`.stropha/bench/campaigns/comprehensive-quality-v3/<commit>-<config>/`. The report
includes winner stability, confidence intervals, quality/cost metrics, and the
strongest and weakest query tags. The configured experiment policy still
controls storage and adapter blacklists; PostgreSQL campaigns use canonical
index caches plus disposable per-cell clones.

An interrupted or failed campaign resumes successful cells when `start` is run
again at the same commit. Stop the worker and its active benchmark with:

```bash
uv run stropha bench campaign stop
```

Campaign stages may override `split`, declare `axes` to vary only selected
parameters, and set `repetitions` plus `min_repetitions` for confidence-driven
independent runs. Repeated groups stop early only when the modal winner's 95%
Wilson interval is above a 50% win probability. Holdout stages must be last and
use separately frozen shortlist matrices; discovery matrices never run on
holdout. The report carries a holdout qualification because previously exposed
cases cannot provide an unbiased estimate of generalization.
