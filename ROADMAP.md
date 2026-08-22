# Stropha Roadmap

This roadmap distinguishes usable features from scaffolding and future work.
It intentionally avoids fixed dates, benchmark percentages, test counts, and
adapter inventories that become stale quickly.

## Shipped

### Core Retrieval

- AST-aware chunking with deterministic enrichment and drift detection
- Three default RRF streams: dense, BM25, and symbol matching
- Optional graph-vector and Model2Vec streams, for up to five configured streams
- Query-time HyDE, query rewriting, multi-query expansion, and recursive merging
- Optional rerankers, with no reranker enabled by default
- Multi-repository indexing and repository-scoped MCP tools

### Graph Intelligence

- Graphify lifecycle integrated into `Pipeline.run()`
- SQLite and PostgreSQL graph mirrors
- Callers, tests, related-symbol, community, rationale, and feature-trace tools
- Graph-gated responses that report `graph_loaded: false` instead of silently
  returning an empty result

### Enrichment And Embeddings

- Local embeddings as the no-key default
- Optional Voyage embeddings; the current Voyage adapter model default is
  `voyage-4-large`
- Hierarchical and graph-aware deterministic enrichers
- Direct Ollama and MLX enrichers
- Contextual enricher using the unified inference backend

The contextual enricher is shipped. It is not Ollama-only: it uses
`inference.backend`, while the separate `ollama` enricher calls Ollama directly.
No repository-independent recall improvement is claimed.

### MCP Surface

- One canonical tool manifest with structured output schemas
- Nested daemon-wide `core`, `standard`, and `full` profiles
- `full` compatibility default and always-visible `capabilities` inspection
- Profile enforcement during both discovery and invocation
- Static profile selection through `~/.stropha/config.yaml`; changes require a
  daemon restart

### Storage And Operations

- SQLite plus sqlite-vec as the default storage backend
- PostgreSQL plus pgvector as an optional shipped backend
- HTTP-only shared MCP daemon with a lazily built warm runtime
- Local dashboard and atomic `PATCH /api/config` runtime replacement
- Detached post-commit hook using hash-based reconcile indexing
- Polling file watcher for saved working-tree changes

The watcher is shipped and updates the persistent configured index. A separate
RAM overlay that merges uncommitted content at query time has not shipped.

### Glossary: Partial

Manual glossary management is shipped:

- Add, remove, list, import, export, and inspect terms through the CLI
- Index glossary entries as searchable `kind=glossary` chunks
- Discover supported glossary YAML files

Automatic LLM extraction, category-aware statistics, and dedicated conceptual
ranking boosts remain planned. The glossary should therefore be described as
partial rather than complete.

## Implemented But Not Integrated

### Semantic Query Cache

An in-memory semantic-cache implementation, configuration model, and tests
exist, but the production retrieval path does not call it. Enabling
`retrieval_features.cache_enabled` does not currently accelerate normal search.
Integration, invalidation guarantees, and end-to-end measurement are still
required before this can be presented as a shipped performance feature.

## Planned

### Near Term

- Integrate or remove the semantic query-cache path, with repository and index
  invalidation semantics
- Complete glossary auto-extraction and ranking behavior
- Improve watcher handling for graph refreshes without requiring every save to
  run a full graph lifecycle
- Add a true working-tree overlay if persistent incremental writes are
  insufficient for editing workflows
- Expand backend-aware backup, restore, and operational diagnostics

### Retrieval Research

- Evaluate sparse-neural retrieval such as SPLADE against BM25
- Evaluate late chunking and multi-vector retrieval on representative codebases
- Improve query routing and learned stream weighting with reproducible golden
  sets
- Keep rerankers opt-in unless repository-specific evaluation demonstrates a
  benefit

### Observability And Distribution

- OpenTelemetry tracing
- On-demand indexing of external dependencies
- Distributed or federated cross-repository retrieval where a single storage
  backend is not sufficient

## Explicit Non-Goals

- Building an IDE or editor
- Requiring a cloud service for the default path
- Publishing precise quality or latency claims without a reproducible benchmark
- Treating implementation scaffolding as a shipped user-facing feature

## Status Rules

- **Shipped** means the production CLI, daemon, or MCP path uses the feature.
- **Partial** means a useful public subset exists but stated acceptance criteria
  remain incomplete.
- **Implemented but not integrated** means code exists without a production
  call path.
- **Planned** means no stable public behavior should be assumed.
