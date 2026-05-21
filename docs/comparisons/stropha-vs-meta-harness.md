# stropha vs. the "meta-harness" — concept comparison + improvement map

**Status:** Conceptual analysis · 2026-05-21 (rev. 2 — re-anchored definition)
**Scope:** stropha @ HEAD (the *concept*, not a line-by-line audit) vs. the concept of a **meta-harness**.
**Goal:** position stropha against the meta-harness idea, show how much of it stropha *already* owns, and enumerate **115 realistic, state-of-the-art improvement options** that close the gap — without breaking the §3 invariants (drift detection, forward-only migrations, local-only default, non-blocking hook).

**Working definition (from the project owner):**
> **A meta-harness is the ability to improve context — so an agent receives better *information* and better *guidance*.**

**Sources verified for this document:**
- `CLAUDE.md` §2 (snapshot), §3 (invariants), §6 (history)
- `src/stropha/server.py` (`grep @mcp.tool` → 9 tool decorators), `src/stropha/adapters/**` (registry tree)
- `src/stropha/retrieval/*` (smart_search, query_router, query_decomposition, recursive, hyde, cache, savings)
- `ROADMAP.md`, `docs/comparisons/stropha-vs-semble.md`, `docs/architecture/stropha-system.md`

> This is an *analysis*, not a spec. Nothing here is shipped. The §3 invariants and the local-only directive bound every option below; cloud-only items are flagged.

---

## 1. Definitions — a retriever vs. a context-improvement engine

### 1.1 What stropha is (the concept)

stropha is a **local-first retrieval substrate for codebases**, exposed to LLM clients over MCP. Its concept is: *take an arbitrary repo, build a structural + semantic model of it, and answer "where is X / who calls X / what explains X" cheaply, so the agent does not burn its context window on directory walks.*

It is a **context *retrieval* engine**. It is invoked with a query; it returns the relevant material; it is done. It does not shape that material, does not advise the agent on it, and does not remember it.

### 1.2 What a "meta-harness" is

> **A meta-harness is the ability to improve context — so an agent receives better *information* and better *guidance*.**

> **Terminology note.** "Meta-harness" is *not* standardized industry jargon. The established terms are **agent harness** / **scaffold** (the deterministic code around the model — popularized in eval work, e.g. METR and SWE-bench, where "the harness matters as much as the model") and, for the concept defined above, **context engineering** — the term the market actually converged on (~2025) for deliberately curating what enters an agent's context window. "Meta-harness" is used in this document as the project's *working term*; read it as *context engineering elevated to a dedicated system*. Where the industry does attach "meta-" to a harness, it usually signals **self-improvement** (a harness that tunes itself — cf. self-improving / "Gödel-machine"-style agents), which this document treats as Floor 3 (§7).

The "meta" here is not about orchestrating harnesses or owning the agent loop. It is about one job, done deliberately: **context improvement**. A plain harness *assembles* context — it globs a few files, dumps a search result, appends history. A meta-harness *improves* it: it treats the context window as a product to be optimized, along two axes.

- **Better information** — the facts in the window are not merely relevant, they are *shaped*: distilled, compressed, deduplicated, fused across sources, fit to a token budget, stripped of noise, annotated with provenance. The objective is *maximum useful signal per token*.
- **Better guidance** — the window does not only contain facts, it carries *direction*: what to do next, what to avoid, what is risky, which constraints apply, what has already been tried, which claim is verified and which is a guess.

A harness answers *"what context exists?"*. A meta-harness answers *"what is the **most useful possible** context for this agent, this task, right now — and what should I tell the agent **about** it?"*.

### 1.3 The thesis

> **stropha is a context *retrieval* engine. A meta-harness is a context *improvement* engine. stropha already nails the expensive prerequisite — finding the right material — but it returns that material nearly raw, and it informs without guiding.**

Under this definition the gap is not *altitude* (orchestration). It is **transformation and direction**:

1. stropha **returns**; it does not **shape** — chunks come back near-raw: no token budget, no compression, no distillation, no cross-source fusion, no dedup. `search_code` gives you K results, not "the best context that fits in N tokens".
2. stropha **informs**; it does not **guide** — it reports what exists; it never tells the agent what to do, what to avoid, what is risky, what to look at next, or which spec constrains the change.
3. stropha is **reactive**; a meta-harness is **anticipatory** — it answers the query asked, instead of pre-empting the next information need or the task's likely failure mode.
4. stropha is **amnesiac** — it cannot improve context using prior turns, prior mistakes, or guidance already given.
5. stropha **cannot tell whether it helped** — there is no feedback loop, so "better context" is never actually defined or measured.

Finding the right material is the costly part, and stropha has it. The 115 options in §5 are about the cheaper-but-neglected rest: shaping that material into better *information*, and wrapping it in *guidance*.

---

## 2. Side-by-side: stropha (today) vs. a meta-harness

Compared on the dimensions that the definition makes load-bearing — *information quality* and *guidance*.

| Dimension | stropha @ HEAD | A meta-harness |
|---|---|---|
| Core job | Retrieve relevant material | Improve context — better information + guidance |
| Unit returned | K chunks / hits / graph nodes | A budget-fit, shaped, annotated **working set** |
| Information shaping | None — chunks returned near-raw | Distilled, compressed, deduplicated, fused, noise-stripped |
| Token-budget awareness | None — `top_k`, not "fits in N tokens" | Packs to the budget; trades rank-N detail for rank-1 depth |
| Guidance to the agent | None — reports what exists | Says what to do, avoid, verify, look at next |
| Anticipation | Reactive — answers the query asked | Pre-empts the next need and the likely failure mode |
| Provenance / trust signals | Minimal (path/line) | Every span tagged: source, why selected, verified vs. guessed |
| Use of the past | Stateless per call (`query_cache` = exact-match only) | Memory: improves context from prior turns + prior mistakes |
| Definition of "better" | Implicit — `eval` is a manual tripwire | Explicit + measured — a closed observe→tune loop |
| World model | graphify mirror (`calls`/`import`/`rationale`, communities) | Same raw material — but mined for *guidance*, not just navigation |
| Verification | None — never checks the agent's claims | Flags hallucinated symbols, verifies claims, scopes blast radius |
| Freshness | Incremental index on commit (hook v=4) | Same — freshness is a precondition of correct *information* |
| Extensibility | Strong: 6-stage adapter framework + drift detection | Same — the substrate is genuinely shared |

**Read of the table:** stropha is *strong* exactly where a meta-harness needs a foundation — sourcing, world model, freshness, extensibility — and *absent* on every row that is about *improving* the context once sourced: shaping, budget, guidance, anticipation, memory, a measured definition of "better".

---

## 3. Where stropha is *already* a meta-harness foundation

A context-improvement engine needs seven organs. Stropha owns three outright, holds one in partial form, and lacks three. Crucially, the three it owns are the *expensive* ones.

| Organ | stropha's existing analogue | Maturity |
|---|---|---|
| **Information sourcing** (find the right material) | 4-stream hybrid retrieval + RRF + graph tools — the hard, costly organ | **Production** |
| **World model** (raw material for guidance) | graphify mirror: `graph_nodes`/`graph_edges`/`graph_meta`, communities, god nodes, `rationale_for`, multi-repo (v7) | **Production** |
| **Freshness sensing** (stale context = wrong information) | `adapter_id` drift detection + `files` dirty cache + git-diff walker + commit hook | **Production** |
| **A definition of "better"** | `eval/harness.py` (Recall@K, MRR, golden set) + `cost.py` + `savings.py` | **Partial** — the *instrument* exists; the *closed loop* does not |
| **Information shaping** (budget, compression, distillation, fusion) | — (`recursive` merges adjacent chunks; that is the closest, and it is off by default) | **Missing** |
| **Guidance generation** (what to do / avoid / verify next) | — (`smart_search` routes *one* query; it never advises the agent) | **Missing** |
| **Memory** (improve context from the past) | — (`query_cache` is exact-match, not memory) | **Missing** |

So: three production organs, one partial, three missing. The honest read is that stropha is *further* from a meta-harness than a loose reading would suggest — shaping, guidance, and memory are genuinely absent. The *encouraging* read is that the three it owns (sourcing, world model, freshness) are the ones that take years to get right; the three it lacks are mostly additive and cheaper. stropha has the foundation; it has not yet built the two floors above it.

---

## 4. The gap, split by the two axes of the definition

**Better information — stropha returns, it does not shape.** Every hit comes back as raw chunk text. There is no token budget, so a planning loop that calls `search_code` repeatedly pays full freight each time. There is no compression — boilerplate (imports, getters, license headers) is returned at the same cost as the one matching line. There is no cross-source fusion — a `find_callers` result and a `search_code` result arrive as two disconnected payloads the agent must reconcile itself. There is no dedup — re-retrieving across turns re-spends the same tokens. The information is *relevant*; it is not *optimized*.

**Better guidance — stropha informs, it does not direct.** stropha will tell the agent that `FsrsCalculator.next_interval` exists and where. It will not tell the agent: this symbol has 14 callers so the change is high-blast-radius; these three tests cover it; ADR-008 constrains its behavior; the agent's claim that it "calls `decay()`" is false per the graph; the symbol the agent just named does not exist at all; last session a similar edit also required a migration. All of that is *latent in stropha's own graph and history* — it is simply never surfaced as direction.

The 115 options below are organized so each one is clearly an **information** improvement, a **guidance** improvement, or an **enabler** that makes the other two possible (memory, feedback, delivery, measurement).

---

## 5. 115 improvement options

Continuously numbered 1–115. **Effort:** S (<1d) · M (1–4d) · L (>1w). **Impact** is on agent task success, not micro-benchmarks. Each group is tagged by which axis of the definition it serves — **[INFO]** better information · **[GUIDE]** better guidance · **[ENABLE]** enabler that makes INFO/GUIDE improvable. Every option respects §3 (forward-only migrations, drift detection, local-only default, non-blocking hook); cloud-dependent items are flagged `[cloud]` and stay optional.

### A. Context shaping — turn "return chunks" into "return a working set" · **[INFO]**

The heart of "better information": the same facts, shaped for maximum signal per token.

| # | Option | What it does (grounded in stropha) | Effort | Impact |
|---|---|---|---|---|
| 1 | Token-budget result packing | Add `context_budget: int` to `search_code`; pack hits until the budget is hit instead of a fixed `top_k`. Knapsack over (score, token-cost). | S | High |
| 2 | `assemble_context` MCP tool | New 13th tool: given a task description, return a ready-to-paste bundle — entry-point files, called symbols, covering tests, ADR rationale — under a budget. The headline verb of context improvement. | L | High |
| 3 | Extractive chunk compression | Before returning, drop boilerplate (imports, getters, headers) and keep signature + the lines that matched. Halves returned tokens at equal recall. | M | High |
| 4 | Diff-scoped context | For "review this PR" tasks, return only changed lines + their semantic neighborhood (callers/tests), via the existing `git-diff` walker. | M | High |
| 5 | Hierarchical result cards | Each hit carries a 1-line / 1-paragraph / full view; the client expands on demand instead of paying full cost up front. | M | Med |
| 6 | Relevance-decayed compression | Top-1 returned in full, rank-10 as signature-only. Compression strength scales with rank. | S | Med |
| 7 | Provenance tags on every span | Each returned span states *why* it was selected (stream, RRF score, graph edge path) so the agent can weigh and self-audit its context. | S | Med |
| 8 | Anti-retrieval / negative context | Surface near-miss chunks the agent should *not* confuse with the answer (e.g. the deprecated twin). | M | Med |
| 9 | Just-in-time retrieval handles | Return lightweight references (id + outline) the agent dereferences only when needed — mirrors Claude Code's file-on-demand model. | M | High |
| 10 | Session-scoped dedup | Don't re-return a chunk already delivered in this session (needs §B state). Pure token savings. | S | Med |
| 11 | Context pinning | Let the harness pin chunks so re-retrieval keeps them stable across turns — avoids context churn. | M | Med |
| 12 | Cross-source fusion | Merge `search_code` + `find_callers` + `find_rationale` results into one coherent, deduplicated working set instead of three disjoint payloads. | M | High |

### B. Session memory & statefulness — improve context using the past · **[ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 13 | Schema v8: `sessions` table | Forward-only migration adding session identity keyed by MCP client/session id. Foundation for everything stateful. | M | High |
| 14 | Persistent project memory | Store distilled facts/decisions per repo ("the FSRS scheduler is authoritative; don't touch X"), retrievable by future sessions. | M | High |
| 15 | `remember` / `recall` MCP tools | Explicit agent-driven write/read of memory — the agent curates its own long-term notes. | M | High |
| 16 | Conversation-aware retrieval | Accept the last N turns as latent query context to bias retrieval toward the current thread (anticipation). | M | High |
| 17 | Episodic task memory | "Last time you touched FSRS code you also had to update the migration." Mined from session + git history. | L | High |
| 18 | Memory consolidation (sleep-time) | Background job compacts episodic entries into semantic memory — runs in the existing hook's detached background. | L | Med |
| 19 | Working-set warm start | On session open for a branch, pre-load the likely-relevant set from prior sessions on that branch. | M | Med |
| 20 | Memory provenance + TTL | Every memory carries a source commit; drift detection (§3.1) marks it stale automatically. | S | Med |
| 21 | Writable scratchpad resource | An MCP resource the agent uses as durable per-repo notes between turns. | M | Med |
| 22 | Cross-session semantic cache | Promote `query_cache` from exact-match to a shared, semantic, session-spanning cache. | M | Med |

### C. Guidance, planning & next-step steering — direct, don't just inform · **[GUIDE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 23 | `plan_investigation` tool | Decompose a task into an ordered retrieval plan over the graph: entry points → dependencies → tests → rationale. Hands the agent a route, not just facts. | L | High |
| 24 | Server-side retrieval DAG | Multi-step retrieval where step N's results seed step N+1 — an agentic retrieval loop run inside stropha, returning only the distilled result. | L | High |
| 25 | `router` as a first-class stage | Promote query routing to a pluggable adapter stage so routing policy is swappable and benchmarkable like every other stage. | M | Med |
| 26 | Confidence-gated escalation | Cheap static lookup first; escalate to LLM-routed / graph traversal only when classifier confidence is low. Cost-shaped. | M | High |
| 27 | Sub-query budget allocator | When `query_decomposition` fans out, allocate the token budget across sub-queries by estimated importance. | M | Med |
| 28 | Parallel fan-out with early-exit | Stop spending on slow streams once the RRF top-K is provably stable. | M | Med |
| 29 | `suggest_next_tool` | Given current results, recommend the next graph tool to call — meta-guidance that steers the agent's loop. | M | High |
| 30 | Local refinement sub-agent | A small *local* LLM loop (Ollama/MLX backend, already present) that critiques and re-queries before returning. | L | Med |
| 31 | Named task recipes | Templates ("bug-fix", "add-feature", "review") that fix a tool sequence + budget profile — guidance encoded as a workflow. | M | Med |
| 32 | Cost-budgeted multi-hop | Cap graph traversal by a *cost budget* (fan-out × depth × token estimate), not a fixed depth. | M | Med |
| 33 | Speculative retrieval prefetch | Predict the agent's next 2–3 queries from the task and pre-compute during idle time. | L | Med |

### D. The feedback loop — define and measure "better context" · **[ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 34 | Usage feedback logging | Log which returned chunks the agent actually cited/edited (correlate with the subsequent `git diff`). The label source for everything below. | M | High |
| 35 | Learned RRF weights | Fit per-repo stream weights from the feedback log instead of the fixed k=60 / equal weighting. | M | High |
| 36 | Auto-eval on every commit | Run the golden set in the hook's detached background; track Recall@K over time; alert on regression. Stays non-blocking (§3.5). | M | High |
| 37 | Golden-set auto-expansion | Mine new golden queries from real sessions: (query, the chunk that got edited) is a free labeled pair. | M | High |
| 38 | Shadow A/B of adapter configs | Run two pipeline compositions on shadow traffic; compare on the feedback metric; the `adapter_id` digest already isolates them. | L | Med |
| 39 | Counterfactual stream logging | Log what each *disabled* stream *would* have returned, to estimate each stream's marginal value. | M | Med |
| 40 | Regression bisection | When Recall@K drops, bisect over commits/config changes (drift detection makes the changes addressable). | M | Med |
| 41 | Per-tool success metrics | Track find_callers / trace_feature hit-rate; surface dead tools the harness should stop offering. | S | Med |
| 42 | `report_bad_result` tool | An MCP tool the agent calls on a miss; entries become hard negatives for reranker training. | S | Med |
| 43 | Self-tuning chunk granularity | Measure retrieval success per chunk size and adapt per language — close the loop on chunking, the dominant quality lever. | L | Med |
| 44 | Local reranker distillation | Periodically distill the `code-aware` reranker from accumulated feedback via lightweight local fine-tune (LoRA). Local-only. | L | Med |

### E. World-model depth — richer facts and guidance material · **[INFO + GUIDE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 45 | Data-flow / def-use edges | Beyond `calls`: track variable def-use so "what reads this field" resolves. | L | High |
| 46 | Type-hierarchy edges | `implements` / `extends` / `overrides` edges for OO navigation. | M | High |
| 47 | Cross-repo dependency edges | Link a monorepo's packages by their import graph (schema v7 already has `repo_id` on edges). | M | Med |
| 48 | Co-change / logical-coupling edges | Mine git history: "these files change together". A live coupling signal grep cannot see — and a guidance signal ("you'll probably need to touch X too"). | M | High |
| 49 | Ownership edges | Map symbols → authors/teams via git blame for "who knows about X". | M | Med |
| 50 | Real test-coverage edges | Ingest lcov / coverage.py output so `find_tests_for` uses actual coverage, not path heuristics. | M | High |
| 51 | Runtime edges from traces | Optionally ingest an OTEL/local trace to add "actually-called-at-runtime" edges. | L | Med |
| 52 | Issue / PR edges | Link symbols to GitHub issues/PRs that touched them — `find_rationale`, extended to live history. | M | Med |
| 53 | Structural graph embeddings | node2vec / GraphSAGE embeddings so the graph-vec stream ranks by *structural* similarity, not just label text. | L | Med |
| 54 | GraphRAG community summaries | LLM-generate a summary per graphify community; retrieve summaries for "how does subsystem X work" (local LLM). | L | High |
| 55 | Global vs. local query modes | Global mode = community summaries (whole-system questions); local mode = node neighborhood. | M | High |
| 56 | Architectural-drift detection | Diff community structure over time; flag when a module's shape changes sharply — surfaced to the agent as guidance. | M | Med |
| 57 | God-node alert tool | Expose graphify's god nodes as a first-class refactoring/risk signal. | S | Low |
| 58 | Spec ↔ code consistency edges | Link ADR sections to the symbols they constrain; flag when code drifts from `docs/architecture/`. | M | Med |

### F. Retrieval quality — relevance is the floor of good information · **[INFO]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 59 | Model2Vec static-embedding stream | A near-free dense lane (static table lookup, ~1 ms) for the planning-loop fast path. The `model2vec` adapter file already exists — wire a retrieval stream around it. | M | High |
| 60 | Late chunking (Jina) | Embed the whole document, then pool token embeddings per chunk boundary — chunk embeddings stay sibling-aware. Needs a long-context embedder. | L | High |
| 61 | ColBERT late-interaction stream | Per-token embeddings + MaxSim at query time, behind the `retrieval-stream` adapter stage. | L | High |
| 62 | SPLADE learned-sparse stream | Learned term expansion; complements BM25. Already a deferred ROADMAP item — revive behind the stream adapter. | L | Med |
| 63 | Contextual BM25 | Pair the existing `contextual` enricher with contextual sparse indexing — Anthropic's *full* contextual-retrieval method. | M | High |
| 64 | Binary / int8 vector quantization | Quantize stored embeddings (4–32× smaller index); rerank survivors at full precision. Fixes the float32-only footprint noted in stropha-vs-semble §2.1. | M | High |
| 65 | Two-stage retrieve | Coarse quantized ANN → exact rerank. Cuts both latency and the 195 MB-per-50k-chunks footprint. | M | High |
| 66 | HNSW index | Replace brute-force cosine with sub-linear ANN as repos grow past ~100k chunks. | M | Med |
| 67 | Matryoshka-aware storage | Store one embedding; query at a truncated dimension for a cheap coarse pass, then refine at full dim. | M | Med |
| 68 | Reranker on by default | Ship the ROADMAP's `mxbai-rerank` cross-encoder enabled for small candidate sets — the `reranker` stage already exists. | M | High |
| 69 | Code-aware rerank signals | Definition boost, identifier stemming, noise penalty for `test_*`/`legacy/` — semble parity, into the existing `code_aware` reranker. | M | High |
| 70 | Adaptive per-query stream weighting | Symbol queries boost the lexical lane, NL queries boost dense. `STROPHA_ADAPTIVE_WEIGHTING` is partial — make the weights query-conditioned, not binary. | M | Med |
| 71 | Multi-vector chunks | Separate embeddings for signature / docstring / body; match the most relevant facet of the chunk. | L | Med |
| 72 | Propositional indexing | Index atomic factual propositions extracted from docs/comments, not just raw chunks. | L | Med |
| 73 | Repo-local embedder fine-tune | Contrastive fine-tune of the local embedder on the repo's own confusable identifier pairs. Local-only, optional. | L | Med |

### G. Liveness — fresh context is the only correct information · **[INFO]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 74 | LSP-backed precise edges | Consume a language server for exact go-to-def/refs instead of graphify heuristics — eliminates symbol-resolution false edges. | L | High |
| 75 | Editor-buffer ingestion | An LSP/MCP channel so unsaved buffer edits update the index, not just commits. | L | High |
| 76 | Streaming re-embed queue | Debounced background re-embed of just-touched symbols (`stropha watch` already polls — make it symbol-granular). | M | Med |
| 77 | Branch-aware index overlays | Per-branch overlays so switching branches doesn't thrash the index. | L | Med |
| 78 | Stale-result watermark | Every result carries the index's HEAD SHA so the agent knows if it is reading old code. | S | Med |
| 79 | Predictive prefetch on file-open | When the agent opens file F, pre-warm retrieval for F's graph neighbors. | M | Med |
| 80 | Index health self-check tool | An MCP tool returning coverage %, drift %, last-index lag — the agent learns to trust or distrust the index. | S | Med |
| 81 | Worktree-aware indexing | Isolate index state per git worktree so parallel agent sessions don't collide. | M | Med |

### H. Verification & grounding — the sharpest form of guidance · **[GUIDE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 82 | `verify_claim` tool | Given an agent assertion ("X calls Y"), check it against the graph; return true/false + evidence edges. | M | High |
| 83 | Hallucinated-symbol guard | Flag when an agent-named symbol does not exist in the index — catches invented APIs before they ship. | S | High |
| 84 | Post-edit blast-radius report | After an edit, return the callers/tests now possibly broken (graph + coverage edges). | M | High |
| 85 | Snippet-freshness assertion | Verify a snippet the agent quoted still matches the current file content. | S | Med |
| 86 | Test-impact analysis tool | Given a diff, return the minimal set of tests to run — graph `calls` edges + real coverage edges (#50). | M | High |
| 87 | Citation enforcement | Return chunks with stable line anchors so the agent's output cites verifiable ranges, not vibes. | S | Med |
| 88 | Spec-consistency check | On a change, surface the ADR/spec section it must stay consistent with (extends `find_rationale` + #58). | M | Med |
| 89 | Contract-test surfacing | For an API symbol, surface its contract tests so the agent edits behavior-aware. | M | Med |

### I. Protocol surface — deliver the improved context well · **[ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 90 | Working set as an MCP resource | Expose the assembled context (#2) as a streamed resource, not only a tool call — lower per-turn overhead. | M | Med |
| 91 | MCP `prompts` for task recipes | Ship bug-fix / review / add-feature prompts as MCP prompts, pre-wired to the right tool sequence (#31). | M | Med |
| 92 | MCP `sampling` for enrichment | Let stropha request a completion from the *client's* model for HyDE/enrichment — no separate LLM, zero extra cost. | M | High |
| 93 | Streaming tool results | Stream hits as they fuse so the agent reads before reranking finishes — cuts perceived latency. | M | Med |
| 94 | Harness-agnostic context API | A stable JSON contract so Cursor / Zed / Continue all get the identical improved-context service. | M | High |
| 95 | A2A endpoint | Expose retrieval over emerging agent-to-agent protocols, not only MCP. | L | Low |
| 96 | Multi-tenant index server | One stropha process serving many repos/agents with per-repo isolation. | L | Med |
| 97 | Shared context bus | Multiple agent sessions on one repo share a retrieval cache + memory. | L | Med |

### J. Observability, cost & governance — accountable context improvement · **[ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 98 | OpenTelemetry retrieval DAG tracing | Per-stage spans across the pipeline. Deferred in ROADMAP — now justified by the multi-step retrieval in §C. | M | Med |
| 99 | Per-session cost & savings ledger | Extend `savings.py` into a real per-agent-run dashboard: tokens saved, compute spent, value delivered. | M | Med |
| 100 | `explain_ranking` tool | Given a query + chunk, return why it ranked where it did (per-stream contributions, RRF math). | M | Med |
| 101 | Per-session budget enforcement | Hard token/compute caps with graceful degradation (drop reranker, drop dense lane) instead of silent overrun. | M | Med |
| 102 | Secret-egress guard | Scan returned chunks for secrets before they leave the process — `.strophaignore` excludes files, this guards *content*. | S | High |
| 103 | Retrieval audit log | Record everything the agent saw, for reproducibility and post-incident review. | S | Med |
| 104 | Deterministic replay | Snapshot index state + query so a session can be replayed exactly. | L | Med |
| 105 | Path-scoped access policy | Pluggable policy adapter: which paths a given agent identity may retrieve. | M | Med |

### K. Evaluation maturity — measure what "better context" means · **[ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 106 | NDCG / MAP / Precision@K | Add ranked-quality metrics to `eval/harness.py` — Recall@K + MRR alone hide ranking quality. ROADMAP item. | S | Med |
| 107 | Per-query latency + P95 | Track latency in the eval harness — the §1.3 spec target (p95 ≤ 500 ms) is currently unmeasured in CI. | S | Med |
| 108 | Public benchmark harness | Run on SWE-bench-retrieval / CodeSearchNet; publish NDCG@10 — stropha currently has no public benchmark. | L | High |
| 109 | Synthetic golden generation | LLM-generate query/answer pairs from the graph for cold-start eval on a fresh repo. | M | Med |
| 110 | End-to-end retrieval-conditioned eval | Measure whether better context actually raises *agent task success* (SWE-bench-style), not just recall. | L | High |
| 111 | Context-efficiency metric | Score "right answer per token returned" — the meta-harness's true objective function. | M | High |

### L. Frontier — the most ambitious, still realistic · **[INFO + GUIDE + ENABLE]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 112 | RL-learned retrieval policy | Treat the router (#25) as an RL agent optimizing end-task success from the feedback log (#34). Local training. | L | High |
| 113 | Sleep-time compute | Between sessions, the index reasons in background: builds community summaries, mines memories, refines negatives — uses the existing detached hook process. | L | High |
| 114 | Self-describing tools | stropha generates its own MCP tool descriptions from usage stats so the harness picks the right tool — the harness improving its own guidance. | M | Med |
| 115 | World-model checkpointing | Version the graph so an agent can ask "what did this code look like 50 commits ago" — time-travel over the world model. | L | Med |

---

## 6. If you only do ten

The highest-leverage moves toward the definition — five that improve **information**, five that add **guidance**.

**Better information:**
1. **#1 token-budget packing** — the smallest change that shifts the concept from "K chunks" to "shaped working set".
2. **#3 extractive compression** — doubles useful signal per token at equal recall.
3. **#2 `assemble_context`** — the headline verb of context improvement; what an agent actually wants.
4. **#12 cross-source fusion** — one coherent working set instead of three disjoint payloads the agent must reconcile.
5. **#54/#55 GraphRAG community summaries + global mode** — converts the graph from navigation into *understanding*.

**Better guidance:**
6. **#29 `suggest_next_tool`** — the cheapest way to start steering the agent's loop, not just feeding it.
7. **#84 post-edit blast-radius report** — turns the graph into "here is what your change just put at risk".
8. **#82/#83 verify_claim + hallucinated-symbol guard** — the sharpest guidance: telling the agent when it is wrong.
9. **#88 spec-consistency check** — surfaces the ADR/constraint a change must respect.
10. **#23 `plan_investigation`** — hands the agent a route through the task, not just facts about it.

**Enablers that make the above improvable:** #13 `sessions` table, #34 usage feedback logging, #36 auto-eval on commit. Without these, "better" is never defined or remembered — schedule them alongside the ten.

---

## 7. The evolution path — three floors above the foundation

stropha does not become a meta-harness in one jump. It builds upward from the foundation it already has:

- **Floor 0 — Retriever (today).** Finds the right material, returns it raw. The expensive foundation: sourcing, world model, freshness. *Done.*
- **Floor 1 — Information shaper.** Owns the *quality* of the context: budget packing, compression, distillation, cross-source fusion, dedup, provenance. (§A, §F, §G) — *the cheapest, highest-ROI floor.*
- **Floor 2 — Guide.** Adds *direction* to the context: next-step suggestions, plans, blast-radius, verification, spec constraints, anticipation. (§C, §H, parts of §E) — *the floor that makes it a context-**improvement** engine, not just a context-**shaping** one.*
- **Floor 3 — Self-improving meta-harness.** Memory + a closed feedback loop, so "better information" and "better guidance" are *learned and measured*, not hand-tuned. (§B, §D, §K, §L) — *the destination implied by the word "meta".*

Each floor is independently shippable and independently valuable — stropha is a better RAG backend at Floor 1 even if it never reaches Floor 3. None of the floors requires breaking a §3 invariant: memory needs a forward-only v8 migration, the feedback loop runs in the already-detached hook process, and every LLM-powered option uses the existing local Ollama/MLX backends or MCP sampling (#92), keeping the local-only default intact.

**Bottom line:** under the definition "a meta-harness improves context for better information and guidance", stropha today is a context *retriever* — it has solved the hard, expensive prerequisite of finding the right material, but it returns that material raw and it informs without directing. The graph, the drift detector, and the eval instrument are exactly the organs a context-improvement engine is built from. Add information shaping, add guidance, add a memory-backed feedback loop, and the same codebase *is* a meta-harness.
