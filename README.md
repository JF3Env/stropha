# stropha

> Local-first retrieval over source code, exposed to LLM clients through the
> [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

[![Python](https://img.shields.io/badge/python-3.12+-blue)](#requirements)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

stropha combines AST-aware chunking, three retrieval streams by default,
optional graph and Model2Vec streams, and structural graph tools such as
`find_callers` and `find_tests_for`. The default install and query path is
local. Cloud embedders and inference backends are opt-in.

## Quick Start

```bash
git clone https://github.com/JonatasFischer/stropha.git
cd stropha

# The local extra is required for the default local embedder and AST chunking.
uv sync --extra local

# Index and search the current repository.
uv run stropha index
uv run stropha search "where does authentication live"
uv run stropha stats
```

To index another repository, pass it explicitly:

```bash
uv run stropha index --repo /path/to/project
```

## Installation

### Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) or `pip`
- A C compiler only on platforms where `sqlite-vec` has to be built from source

### From a checkout

```bash
uv sync --extra local
uv run stropha --help
```

With pip:

```bash
pip install -e ".[local]"
stropha --help
```

Installing the base package without the `local` extra is not enough for the
default local embedding and tree-sitter pipeline.

### Optional Extras

| Extra | Purpose |
|---|---|
| `voyage` | Voyage cloud embeddings |
| `model2vec` | Optional independent 256-dimensional retrieval lane |
| `mlx` | In-process LLM inference on Apple Silicon |
| `mlx-embed` | MLX embedding models on Apple Silicon |
| `postgres` | PostgreSQL and pgvector storage |

For example:

```bash
uv sync --extra local --extra voyage
```

Graph traversal also requires
[`graphify`](https://github.com/safishamsi/graphify):

```bash
pipx install graphifyy
graphify .
uv run stropha index
```

## How It Works

### Indexing

The pipeline walks files, creates structural chunks, enriches their embedding
text, embeds them, and writes the configured storage backend. Each stage is an
adapter selected in `~/.stropha/config.yaml`.

Chunks are tracked by content, embedder, and enricher identity. A normal
`stropha index` skips fresh content and processes changes. Use `--rebuild` when
the command reports that the active vector space is incompatible with the
stored one. A primary dimension change clears derived index data for the entire
shared backend, so re-index every repository stored there.

The default maximum source-file size is 512 KiB. Larger files are skipped; tune
`walker.max_file_bytes` in the user configuration if they are intentional.

### Retrieval

The default `hybrid-rrf` pipeline runs three streams and fuses them with
Reciprocal Rank Fusion:

1. Dense vector similarity over code chunks
2. FTS5 BM25 lexical search
3. Identifier-oriented symbol matching

Up to five streams can participate when explicitly configured:

- `graph-vec` searches graphify node embeddings.
- `model2vec-cosine` searches the independent 256-dimensional Model2Vec lane.

The optional streams do not alter the primary embedding space. `graph-vec`
requires a loaded graph with node embeddings, and Model2Vec requires the
`model2vec` extra plus a secondary embedder configuration.

No reranker is enabled by default. The local cross-encoder and inference-backed
LLM rerankers are opt-in and should be evaluated on the target repository before
being retained.

### Graph Tools

`Pipeline.run()` owns the graph lifecycle. When graphify is installed, indexing
runs its update, loads the graph mirror, loads graph vectors, and augments FTS in
that order.

Graph-backed tools include `find_callers`, `find_tests_for`, `find_related`,
`get_community`, `find_rationale`, and `trace_feature`. If no graph is loaded,
they return `graph_loaded: false` with an actionable message rather than an
empty success response.

## MCP Integration

stropha is HTTP-only. One shared streamable-HTTP daemon serves all MCP clients
at `http://127.0.0.1:9741/mcp` by default; stdio and per-client subprocess
configurations are not supported.

### Start the daemon

```bash
bin/stropha-mcp-ensure
bin/stropha-mcp-ensure --status
bin/stropha-mcp-ensure --stop
```

The launcher waits for both `/health` and an MCP protocol response. The heavy
pipeline runtime is built lazily on the first MCP or frontend request and then
shared. The file watcher starts with that runtime when `mcp.watch` is enabled,
which is the default, and stops with the daemon.

`mcp.profile` selects one daemon-wide static tool surface: `core`, `standard`,
or `full`. `full` is the compatibility default. `core` keeps only the primary
retrieval and orientation tools, while `standard` adds investigation and
change-intelligence workflows. Call `capabilities` from any profile to inspect
the active tool names. Every client connected to the daemon sees the same
profile, and changing it requires a daemon restart.

The launcher can optionally manage a local pgvector Docker container when
started with `STROPHA_USE_POSTGRES=1`. Without that explicit launcher flag it
does not start or stop PostgreSQL; a configured PostgreSQL adapter connects to
the supplied DSN as-is.

Open `http://127.0.0.1:9741/dashboard` for the local dashboard. Supporting
routes include:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health |
| `GET /api/status` | Active runtime and index status |
| `GET /api/config` | Redacted user configuration |
| `GET /api/config/schema` | Editable configuration schema |
| `PATCH /api/config` | Write configuration; hot-swap compatible runtimes or report a required restart/rebuild |
| `POST /api/query` | Answer a question from retrieved evidence |

`PATCH /api/config` applies query-time and compatible adapter changes without a
daemon restart. The exception is `mcp.profile`: the new value is saved, but the
active tool surface remains unchanged and the response reports
`restart_required: true`. Changes that alter indexed content take effect fully
after the next index pass. A primary embedding dimension mismatch returns HTTP
202 with `pending_rebuild: true`; the previous runtime remains active. Run
`stropha index --rebuild`, then restart the daemon or submit the patch again to
activate the rebuilt vector space. Dimension rebuilds clear all derived search
data for every repository in the shared backend, not only the selected
repository. Editing the YAML directly does not trigger live reload, so restart
the daemon after direct edits.

### Configure a client

OpenCode:

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

Clients using the common `mcpServers` shape:

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

Useful public tools include:

| Tool | Minimum profile | Use |
|---|---|---|
| `capabilities` | core | Inspect the active profile and tool inventory |
| `search_code` | core | Semantic and lexical code search |
| `smart_search_tool` | standard | Route an ambiguous question to the appropriate search or graph tool |
| `get_symbol` | core | Exact symbol lookup |
| `get_file_outline` | core | Inspect file structure before reading it |
| `find_callers` | core | Find callers through graph edges |
| `find_tests_for` | core | Find tests associated with a symbol |
| `assemble_context` | core | Retrieve and pack a bounded context bundle |
| `execute_investigation` | standard | Run a planned search, callers, tests, and rationale investigation |

Tools are scoped to the current repository by default. Set
`include_other_repos=true` only for an intentional cross-repository query.

See [the MCP integration guide](docs/guides/mcp-integration.md) for more client
examples and operational details.

## Configuration

`~/.stropha/config.yaml` is the only configuration file. There is no
project-level or `.env` overlay and no generic environment-variable overlay for
settings. Explicit CLI flags can override settings for one command.

`STROPHA_TARGET_REPO` remains a runtime target argument. `STROPHA_HOOK_*`
variables are escape hatches for the detached hook process. Launcher-only
variables such as `STROPHA_MCP_HOST`, `STROPHA_MCP_PORT`, and
`STROPHA_USE_POSTGRES` control the shell launcher, not pipeline configuration.

Example:

```yaml
paths:
  index: ~/.stropha/index.db

walker:
  max_file_bytes: 524288

pipeline:
  enricher:
    adapter: hierarchical
    config: {}
  embedder:
    adapter: local
    config:
      model: mixedbread-ai/mxbai-embed-large-v1
  retrieval:
    adapter: hybrid-rrf
    config:
      adaptive_weighting: true
      multi_query_count: 3
      recursive_enabled: false
      recursive_adjacency: 5
      reranker: null

retrieval_features:
  hyde_enabled: false
  multi_query_enabled: false

git:
  clone_depth: 1
  clone_timeout: 300

mcp:
  profile: full
  watch: true
  watch_interval: 1.0
  watch_debounce: 2.0

logging:
  level: INFO
```

`mcp.profile` accepts `core`, `standard`, or `full`. It has no CLI or generic
environment-variable override. Restart `bin/stropha-mcp-ensure` after changing
it, including when the dashboard saved the new value.

API keys belong under `secrets:` in this file. Restrict it to the current user:

```bash
chmod 600 ~/.stropha/config.yaml
```

Machine-local experiment policy also lives only in `~/.stropha/config.yaml`.
For native PostgreSQL evaluation with local adapter exclusions:

```yaml
experiments:
  storage: postgres
  postgres:
    admin_dsn: postgresql://ROLE:PASSWORD@HOST:PORT/stropha_bench_admin
    database_prefix: stropha_bench
    template_database: stropha_bench_template
  blacklist:
    adapters:
      embedder: [voyage]
      walker: [git-diff]
```

`stropha bench run` and `stropha meta run` both use this storage selection and
blacklist. SQLite remains the default for backward compatibility. PostgreSQL
mode instead evaluates the native pgvector and PostgreSQL BM25 paths and avoids
sqlite-vec KNN candidate limits, making it the appropriate mode when those
production-backend behaviors are what the experiment must measure.

The PostgreSQL `admin_dsn` must name a database beginning with
`<database_prefix>_`; the example therefore uses `stropha_bench_admin`. Use a
dedicated, non-production admin database and role, and grant that role
`CREATEDB` plus membership in `pg_signal_backend`. The latter lets guarded
`DROP DATABASE ... FORCE` cleanup terminate PostgreSQL maintenance sessions in
an exact, marked evaluation database. The credentials stay only in
`~/.stropha/config.yaml`: experiment
provenance records a policy fingerprint and non-secret cluster identity, never
the DSN or password. The production DSN and production database are never
mutated.

The template database must be owned by the evaluation role and have the
`vector` extension installed by a PostgreSQL administrator. Stropha clones it
before opening storage, so the evaluation role does not need superuser rights.

PostgreSQL experiments build deterministic, marked canonical cache databases
and create a disposable template clone for each benchmark cell or Meta-Harness
candidate. Canonical caches persist for reuse across runs; work clones are
dropped after use. Destructive cleanup is restricted to an exact generated name
under the fixed prefix with a valid Stropha marker and the expected owner. There
is no implicit bulk cleanup.

The blacklist is local policy and does not modify the checked-in search-space
manifest. In addition, `git-diff` is always excluded from full-snapshot
experiments because it is an incremental-only walker, even when it is not listed
in the blacklist. Meta holdout validation fails closed unless the frozen storage
backend, policy fingerprint, and PostgreSQL cluster identity still match the
search run.

The no-key default embedder is local. If `secrets.voyage_api_key` is present and
the pipeline embedder adapter is omitted, auto-selection chooses Voyage. The
Voyage adapter currently defaults to `voyage-4-large` with 512-dimensional
output; Voyage is not the overall default.

The `contextual` enricher uses the unified `inference.backend` selection and can
run through MLX, Ollama, Anthropic, DeepSeek, LMS, or MoA as configured. The
separate `ollama` enricher calls Ollama directly with its own model and endpoint
configuration. LMS has no built-in endpoint: set `inference.lms_base_url` (any
OpenAI-compatible API root) and `secrets.lms_api_key`; without them the LMS
backend reports itself unavailable.

Inspect the resolved configuration rather than relying on a copied inventory:

```bash
uv run stropha pipeline show
uv run stropha pipeline validate
uv run stropha adapters list
```

## Post-Commit Automation

Install the hook in the target repository:

```bash
uv run stropha hook install
uv run stropha hook status
```

The installer honors Git's `core.hooksPath`; do not assume the file is at
`.git/hooks/post-commit`. `stropha hook status` reports the resolved hook path
and installed script version. Re-running `stropha hook install --force` updates
the managed block while preserving unrelated hook content.

The foreground hook detaches immediately. Its background job runs one normal
`stropha index --repo <target>` in hash-based reconcile mode and owns graphify
processing. Git-diff incremental mode requires an explicit `--incremental` CLI
request. The same detached job can run evaluation tracking and outcome
backfill. Meta iterations are disabled unless explicitly installed or enabled.
Concurrent hook jobs are guarded by a portable directory lock.

For a hook in another repository, use generic paths rather than machine-specific
ones:

```bash
uv run stropha hook install \
  --target /path/to/project \
  --project-dir /path/to/stropha \
  --log-path ~/.cache/stropha-hook-project.log

uv run stropha hook status --target /path/to/project
```

## CLI

The stable entry points are:

```text
stropha index
stropha search "QUERY"
stropha stats
stropha pipeline show
stropha pipeline validate
stropha adapters list
stropha hook install|status|uninstall
stropha watch
stropha eval
stropha cost
stropha bench
stropha meta run|validate|status|report|promote|trace|snapshot|prune|outcome|stream-weights
```

Run `stropha <command> --help` for the live option set. `stropha stats` opens the
configured storage backend and reports its location, size, chunk and file
counts, embedding identities, enrichers, and indexed repositories. For SQLite,
opening an absent path may initialize an empty database, so a successful stats
command is not proof that indexing previously completed; check for nonzero
counts and the expected repository.

## Development

```bash
uv sync --extra local --extra dev
uv run pytest -q
uv run ruff check src tests
uv run mypy src/stropha
uv build
```

The evaluation harness uses the repository's JSONL golden set:

```bash
uv run stropha eval --top-k 10
```

Evaluation defaults to the configured target repository, matching MCP search
scope. It reports query success, entity recall, MRR, NDCG, MAP, latency,
pipeline/index provenance, and reranker fallback diagnostics. Use
`--include-other-repos` only for an explicit cross-repository benchmark and
`--require-clean-index` when non-comparable index drift must fail CI.

### Meta-Harness

The Meta-Harness searches stropha's registered configuration space for retrieval
settings that improve quality, context efficiency, or latency on the target
repository's golden set. It evaluates the real production pipeline rather than
synthetic scores, maintains a Pareto frontier instead of forcing every objective
into one winner, and leaves the live configuration unchanged during search and
validation.

Evaluation is split deliberately:

- `meta run` repeatedly evaluates candidates on the visible `search` split.
- `meta validate` evaluates the frozen baseline and shortlist on hidden
  `holdout` cases to detect overfitting.
- `meta promote` prints a reviewable diff by default and changes
  `~/.stropha/config.yaml` only with an eligible validation and explicit
  `--apply`.

Each run pins committed `HEAD`; uncommitted source changes are excluded. It uses
the machine-local `experiments` storage and blacklist described above, never the
production index. SQLite uses isolated writable file clones; PostgreSQL uses
disposable template-cloned databases, so candidates can be evaluated
concurrently without corrupting shared retrieval state.

First confirm the configured repository and pipeline:

```bash
uv run stropha pipeline show
uv run stropha pipeline validate
```

Start with a small reproducible bandit smoke run:

```bash
uv run stropha meta run \
  --iterations 2 \
  --candidates 2 \
  --parallel 2 \
  --seed 42 \
  --proposer bandit
```

The result prints a `<run-id>`. Inspect the archive and candidate leaderboard:

```bash
uv run stropha meta status
uv run stropha meta report
```

When the frozen shortlist is ready, validate it against holdout once:

```bash
uv run stropha meta validate <run-id>
```

Choose a `<candidate-id>` from the report or shortlist and review its proposed
configuration diff. This command does not write the configuration:

```bash
uv run stropha meta promote <candidate-id> --run-id <run-id>
```

Apply only after reviewing the diff:

```bash
uv run stropha meta promote <candidate-id> \
  --run-id <run-id> \
  --apply
```

After the smoke run, a reasonable larger local run is:

```bash
uv run stropha meta run \
  --iterations 5 \
  --candidates 3 \
  --parallel 2 \
  --seed 42 \
  --proposer bandit
```

Resume an interrupted bandit run by repeating all frozen options exactly and
adding `--resume`. Changing the seed, golden set, budgets, parallelism, source,
runtime configuration, manifest, registry, graph identity, or cache format is
rejected:

```bash
uv run stropha meta run \
  --iterations 2 \
  --candidates 2 \
  --parallel 2 \
  --seed 42 \
  --proposer bandit \
  --resume <run-id>
```

With the default `~/.stropha/index.db`, run artifacts are stored under
`~/.stropha/meta/runs/<run-id>/`:

```text
provenance.json
checkpoint.json
report.json
shortlist.json
validation.json
cost.json
```

`validation.json` appears only after explicit validation. The top-level
`stropha cost` command is a separate hook-log dashboard; Meta-Harness retrieval,
indexing, proposer, verification, and artifact usage is recorded in the run's
`cost.json`.

Do not repeatedly validate the same run while tuning against its results. Use
the `search` split for iteration and open holdout only for a frozen shortlist.
Exact resume is currently bandit-only. Generated-code proposer modes
(`local-code` and `cloud-code`) are explicit opt-ins, restricted to one sandboxed
reranker adapter per candidate, and always review-only.

Promotion with `--apply` fails closed unless the candidate has eligible
schema-v2 validation and its candidate hash, source snapshot, runtime
configuration, manifest, registry, graph identity, and cache format still match
the frozen run.

The active-stack benchmark matrix is available at
`src/stropha/bench/active_matrix.yaml`.

Public documentation intentionally does not publish fixed latency, memory,
quality, test-count, or dataset-size claims. Measure the active configuration on
the target repository instead.

## Roadmap

Current and planned work is tracked in [ROADMAP.md](ROADMAP.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
