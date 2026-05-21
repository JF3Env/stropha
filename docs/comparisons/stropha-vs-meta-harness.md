# stropha vs. the "meta-harness" — concept comparison + improvement map

**Status:** Conceptual analysis · 2026-05-21
**Scope:** stropha @ HEAD (the *concept*, not a line-by-line audit) vs. the concept of an agentic **meta-harness**.
**Goal:** position stropha against the meta-harness idea, show how much meta-harness substrate stropha *already* owns, and enumerate **115 realistic, state-of-the-art improvement options** that move it up the stack — without breaking the §3 invariants (drift detection, forward-only migrations, local-only default, non-blocking hook).

**Sources verified for this document:**
- `CLAUDE.md` §2 (snapshot), §3 (invariants), §6 (history)
- `src/stropha/server.py` (`grep @mcp.tool` → 9 tool decorators), `src/stropha/adapters/**` (registry tree)
- `src/stropha/retrieval/*` (smart_search, query_router, query_decomposition, recursive, hyde, cache, savings)
- `ROADMAP.md`, `docs/comparisons/stropha-vs-semble.md`, `docs/architecture/stropha-system.md`

> This is an *analysis*, not a spec. Nothing here is shipped. The §3 invariants and the local-only directive bound every option below; cloud-only items are flagged.

---

## 1. Definitions — two concepts, two altitudes

### 1.1 What stropha is (the concept)

stropha is a **local-first retrieval substrate for codebases**, exposed to LLM clients over MCP. Its concept is: *take an arbitrary repo, build a structural + semantic model of it, and answer "where is X / who calls X / what explains X" cheaply, so the agent does not burn its context window on directory walks.*

It is a **component of an agent harness** — specifically the *retrieval organ*. It is invoked; it does not invoke. It is stateless per call. It returns chunks; it does not decide what the agent does with them.

### 1.2 What a "meta-harness" is (the concept)

An **agent harness** is the deterministic software scaffold around an LLM that turns a stateless model into an agent: the agent loop, tool definitions and dispatch, context assembly, memory, sub-agent orchestration, verification, permission gating. Claude Code is a harness. Cursor's agent is a harness. The Claude Agent SDK is a harness toolkit.

A **meta-harness** sits one altitude higher. It is the harness *concerned with the harness itself*. Four coherent readings, all in play:

1. **Harness-of-harnesses / orchestrator** — composes, selects, and routes between specialized sub-agents; owns the "which agent, which tool, which context" decision.
2. **Self-improving harness** — observes runs, evaluates them, and tunes the harness (prompts, tool descriptions, retrieval weights, context budgets). It treats the harness as a *learnable artifact*.
3. **Context-engineering substrate** — manages the *entire context lifecycle* of an agent: what enters the window, in what order, when to compact, what to evict, what to persist as memory.
4. **Harness-agnostic protocol layer** — sits beneath any specific harness (Claude Code, Cursor, Zed) and offers a uniform context / retrieval / memory / eval service.

A meta-harness's defining traits: it is **stateful** (sessions, memory), **active** (it orchestrates, it does not just answer), **accountable** (it measures and improves itself), and **harness-agnostic** (it is the substrate, not the UI).

### 1.3 The thesis

> **stropha is one altitude below a meta-harness — and is, structurally, one stage away from becoming one.**

stropha already owns an unusually large fraction of the meta-harness substrate (see §3). What it lacks is not infrastructure; it is **altitude**: it returns chunks instead of *managing a working set*, it is stateless instead of *session-aware*, it is invoked instead of *orchestrating*, and it is hand-tuned instead of *self-tuning*. The 115 options in §5 are the climb.

---

## 2. Side-by-side: stropha (today) vs. a meta-harness

| Dimension | stropha @ HEAD | A meta-harness |
|---|---|---|
| Altitude | Retrieval *component* of a harness | The harness's orchestration + context brain |
| Primary verb | **Answers** ("where is X?") | **Orchestrates** ("solve this; here is the plan, the context, the budget") |
| Unit returned | Chunks / hits / graph nodes | A budget-aware, task-scoped **working set** |
| State | Stateless per MCP call (`query_cache` is exact-match only) | Stateful: sessions, episodic + semantic memory |
| Context lifecycle | Not owned — the client decides what to keep | Owned end-to-end: admit, order, compress, compact, evict, persist |
| Loop ownership | None — stropha is invoked | Owns the agent loop / sub-agent fan-out |
| Routing | `smart_search` classifies intent → one tool | Routes across tools *and* sub-agents *and* models, by cost/confidence |
| Self-improvement | `eval` golden set is a manual regression guard | A closed feedback loop: observe → evaluate → tune → ship |
| World model | graphify mirror (`calls`/`import`/`rationale` edges, communities) | A live, multi-relation world model that orchestration plans over |
| Verification | None — stropha never checks the agent's claims | Closes the loop: verifies claims, flags hallucinated symbols, scopes tests |
| Cost awareness | `cost` dashboard + `savings.py` (token accounting) | Per-session budget enforcement + degradation policy |
| Observability | structlog + `stropha cost`; OTEL deferred | Full traced retrieval DAG, per-session ledgers, deterministic replay |
| Protocol surface | MCP tools + one `stropha://stats` resource | Tools + resources + prompts + sampling + A2A; harness-agnostic |
| Extensibility | Strong: 6-stage adapter framework, drift detection | Same — but the *composition itself* is a tuned search space |
| Liveness | Incremental index on commit (hook v=4) | Tracks unsaved buffers, branches, runtime traces |

**Read of the table:** stropha is *strong* on the rows that are infrastructure (extensibility, world model, cost accounting, liveness) and *absent* on the rows that are altitude (state, loop, orchestration, self-improvement, verification). That asymmetry is the opportunity.

---

## 3. Where stropha is *already* meta-harness substrate

This is the non-obvious part. A meta-harness needs seven organs; stropha already has five of them in a form that is unusually close to fit-for-purpose:

| Meta-harness organ | stropha's existing analogue | Maturity |
|---|---|---|
| Pluggable strategy framework | 6-stage adapter framework + `@register_adapter` + registry | **Production** — a meta-harness *needs* swappable strategies; stropha already enforces them via Protocols |
| World model | graphify mirror: `graph_nodes`/`graph_edges`/`graph_meta`, communities, god nodes, `rationale_for` | **Production** — multi-relation, multi-repo (schema v7) |
| Intent classification | `query_router` + `smart_search` + `query_classify` | **Production** — pattern-based + optional LLM fallback |
| Staleness / drift sensing | `adapter_id` drift detection + `files` dirty cache + git-diff walker | **Production** — a meta-harness must know when its model is stale; stropha does |
| Feedback instrument | `eval/harness.py` (Recall@K, MRR, golden set) + `cost.py` + `savings.py` | **Partial** — the *instrument* exists; the *loop* (observe→tune) does not |
| Session memory | — | **Missing** — `query_cache` is exact-match, not memory |
| Orchestration / loop | — (`query_decomposition` fans out *one* query; it does not orchestrate) | **Missing** |

Two of seven are missing; the other five are real. That is why the thesis holds: stropha is not "far" from a meta-harness — it is **two organs short**, and both (memory, orchestration) are additive, not rewrites.

---

## 4. The gap, in one paragraph

stropha returns *the right chunks*. A meta-harness returns *the right context, in the right shape, within a budget, remembered across turns, with a plan for what to retrieve next, and a measurement of whether it helped.* The gap is: **(a)** no notion of a token budget — `search_code` returns K results, not "as much as fits in N tokens"; **(b)** no statefulness — every call is amnesiac; **(c)** no orchestration — stropha is a leaf, never a root; **(d)** no closed feedback loop — `eval` is a tripwire, not a learner; **(e)** no verification — stropha never checks what the agent did with what it returned. The 115 options below close exactly those five gaps, plus deepen the substrate that already exists.

---

## 5. 115 improvement options

Continuously numbered 1–115. **Effort:** S (<1d) · M (1–4d) · L (>1w). **Impact** is on agent task success, not micro-benchmarks. Every option respects §3 (forward-only migrations, drift detection, local-only default, non-blocking hook); cloud-dependent items are flagged `[cloud]` and stay optional.

### A. Context engineering — turn "return chunks" into "return a working set" (the core leap)

| # | Option | What it does (grounded in stropha) | Effort | Impact |
|---|---|---|---|---|
| 1 | Token-budget result packing | Add `context_budget: int` to `search_code`; pack hits until the budget is hit instead of a fixed `top_k`. Knapsack over (score, token-cost). | S | High |
| 2 | `assemble_context` MCP tool | New 13th tool: given a task description, return a ready-to-paste bundle — entry-point files, called symbols, covering tests, ADR rationale — under a budget. The meta-harness's headline verb. | L | High |
| 3 | Extractive chunk compression | Before returning, drop boilerplate (imports, getters) and keep signature + the lines that matched. Halves returned tokens at equal recall. | M | High |
| 4 | Diff-scoped context | For "review this PR" tasks, return only changed lines + their semantic neighborhood (callers/tests), via the existing `git-diff` walker. | M | High |
| 5 | Hierarchical result cards | Each hit carries a 1-line / 1-paragraph / full view; the client expands on demand instead of paying full cost up front. | M | Med |
| 6 | Relevance-decayed compression | Top-1 returned in full, rank-10 as signature-only. Compression strength scales with rank. | S | Med |
| 7 | Provenance tags on every span | Each returned span states *why* it was selected (stream, RRF score, graph edge path) so the agent can self-audit its context. | S | Med |
| 8 | Anti-retrieval / negative context | Surface near-miss chunks the agent should *not* confuse with the answer (e.g. the deprecated twin). | M | Med |
| 9 | Just-in-time retrieval handles | Return lightweight references (id + outline) the agent dereferences only when needed — mirrors Claude Code's file-on-demand model. | M | High |
| 10 | Session-scoped dedup | Don't re-return a chunk already delivered in this session (needs §B state). Pure context savings. | S | Med |
| 11 | Context pinning | Let the harness pin chunks so re-retrieval keeps them stable across turns — avoids context churn. | M | Med |
| 12 | Budget-efficiency scoring | Rank candidate packings by "answer-coverage per token", not raw score sum. | M | Med |

### B. Session memory & statefulness — stop being amnesiac

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 13 | Schema v8: `sessions` table | Forward-only migration adding session identity keyed by MCP client/session id. Foundation for everything in §B. | M | High |
| 14 | Persistent project memory | Store distilled facts/decisions per repo ("the FSRS scheduler is authoritative; don't touch X"), retrievable by future sessions. | M | High |
| 15 | `remember` / `recall` MCP tools | Explicit agent-driven write/read of memory — the agent curates its own long-term notes. | M | High |
| 16 | Conversation-aware retrieval | Accept the last N turns as latent query context to bias retrieval toward the current thread. | M | High |
| 17 | Episodic task memory | "Last time you touched FSRS code you also had to update the migration." Mined from session + git history. | L | High |
| 18 | Memory consolidation (sleep-time) | Background job compacts episodic entries into semantic memory — runs in the existing hook's detached background. | L | Med |
| 19 | Working-set warm start | On session open for a branch, pre-load the likely-relevant set from prior sessions on that branch. | M | Med |
| 20 | Memory provenance + TTL | Every memory carries a source commit; drift detection (§3.1) marks it stale automatically. | S | Med |
| 21 | Writable scratchpad resource | An MCP resource the agent uses as durable per-repo notes between turns. | M | Med |
| 22 | Cross-session retrieval cache | Promote `query_cache` from exact-match to a shared, semantic, session-spanning cache. | M | Med |

### C. Orchestration & sub-agent routing — become a root, not a leaf

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 23 | `plan_investigation` tool | Decompose a task into an ordered retrieval plan over the graph: entry points → dependencies → tests → rationale. | L | High |
| 24 | Server-side retrieval DAG | Multi-step retrieval where step N's results seed step N+1 — an agentic retrieval loop run inside stropha, returning only the distilled result. | L | High |
| 25 | `router` as a first-class stage | Promote query routing to a pluggable adapter stage so routing policy is swappable and benchmarkable like every other stage. | M | Med |
| 26 | Confidence-gated escalation | Cheap static lookup first; escalate to LLM-routed / graph traversal only when classifier confidence is low. Cost-shaped. | M | High |
| 27 | Sub-query budget allocator | When `query_decomposition` fans out, allocate the token budget across sub-queries by estimated importance. | M | Med |
| 28 | Parallel fan-out with early-exit | Stop spending on slow streams once the RRF top-K is provably stable. | M | Med |
| 29 | `suggest_next_tool` | Given current results, recommend the next graph tool to call — meta-guidance that steers the agent's loop. | M | High |
| 30 | Server-side refinement sub-agent | A small *local* LLM loop that critiques and re-queries before returning (local-only: Ollama/MLX backend already present). | L | Med |
| 31 | Named task recipes | Templates ("bug-fix", "add-feature", "review") that fix a tool sequence + budget profile. | M | Med |
| 32 | Cost-budgeted multi-hop | Cap graph traversal by a *cost budget* (fan-out × depth × token estimate), not a fixed depth. | M | Med |
| 33 | Speculative retrieval prefetch | Predict the agent's next 2–3 queries from the task and pre-compute during idle time. | L | Med |

### D. The feedback loop — treat the harness as learnable

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 34 | Usage feedback logging | Log which returned chunks the agent actually cited/edited (correlate with subsequent `git diff`). The label source for everything below. | M | High |
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

### E. World-model depth — the graph as the meta-harness's model of reality

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 45 | Data-flow / def-use edges | Beyond `calls`: track variable def-use so "what reads this field" resolves. | L | High |
| 46 | Type-hierarchy edges | `implements` / `extends` / `overrides` edges for OO navigation. | M | High |
| 47 | Cross-repo dependency edges | Link a monorepo's packages by their import graph (schema v7 already has `repo_id` on edges). | M | Med |
| 48 | Co-change / logical-coupling edges | Mine git history: "these files change together". A live coupling signal grep cannot see. | M | High |
| 49 | Ownership edges | Map symbols → authors/teams via git blame for "who knows about X". | M | Med |
| 50 | Real test-coverage edges | Ingest lcov / coverage.py output so `find_tests_for` uses actual coverage, not path heuristics. | M | High |
| 51 | Runtime edges from traces | Optionally ingest an OTEL trace to add "actually-called-at-runtime" edges. `[cloud-optional]` — local trace files work too. | L | Med |
| 52 | Issue / PR edges | Link symbols to GitHub issues/PRs that touched them — `find_rationale`, extended to live history. | M | Med |
| 53 | Structural graph embeddings | node2vec / GraphSAGE embeddings so the graph-vec stream ranks by *structural* similarity, not just label text. | L | Med |
| 54 | GraphRAG community summaries | LLM-generate a summary per graphify community; retrieve summaries for "how does subsystem X work" (local LLM). | L | High |
| 55 | Global vs. local query modes | Global mode = community summaries (whole-system questions); local mode = node neighborhood. | M | High |
| 56 | Architectural-drift detection | Diff community structure over time; flag when a module's shape changes sharply — surfaced to the agent. | M | Med |
| 57 | God-node alert tool | Expose graphify's god nodes as a first-class refactoring/risk signal. | S | Low |
| 58 | Spec ↔ code consistency edges | Link ADR sections to the symbols they constrain; flag when code drifts from `docs/architecture/`. | M | Med |

### F. Retrieval quality — state-of-the-art techniques

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

### G. Liveness — track reality cheaply

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

### H. Verification & grounding — close the loop on correctness

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

### I. Protocol surface — be the substrate, not just a tool

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 90 | Working set as an MCP resource | Expose the assembled context (#2) as a streamed resource, not only a tool call — lower per-turn overhead. | M | Med |
| 91 | MCP `prompts` for task recipes | Ship bug-fix / review / add-feature prompts as MCP prompts, pre-wired to the right tool sequence (#31). | M | Med |
| 92 | MCP `sampling` for enrichment | Let stropha request a completion from the *client's* model for HyDE/enrichment — no separate LLM, zero extra cost. | M | High |
| 93 | Streaming tool results | Stream hits as they fuse so the agent reads before reranking finishes — cuts perceived latency. | M | Med |
| 94 | Harness-agnostic context API | A stable JSON contract so Cursor / Zed / Continue all get the identical working-set service. | M | High |
| 95 | A2A endpoint | Expose retrieval over emerging agent-to-agent protocols, not only MCP — future-proofs the substrate role. | L | Low |
| 96 | Multi-tenant index server | One stropha process serving many repos/agents with per-repo isolation — the substrate runs once, not per-client. | L | Med |
| 97 | Shared context bus | Multiple agent sessions on one repo share a retrieval cache + memory — the meta-harness for a *team* of agents. | L | Med |

### J. Observability, cost & governance — be accountable

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 98 | OpenTelemetry retrieval DAG tracing | Per-stage spans across the pipeline. Deferred in ROADMAP — now justified by the orchestration in §C. | M | Med |
| 99 | Per-session cost & savings ledger | Extend `savings.py` into a real per-agent-run dashboard: tokens saved, compute spent, value delivered. | M | Med |
| 100 | `explain_ranking` tool | Given a query + chunk, return why it ranked where it did (per-stream contributions, RRF math). | M | Med |
| 101 | Per-session budget enforcement | Hard token/compute caps with graceful degradation (drop reranker, drop dense lane) instead of silent overrun. | M | Med |
| 102 | Secret-egress guard | Scan returned chunks for secrets before they leave the process — `.strophaignore` excludes files, this guards *content*. | S | High |
| 103 | Retrieval audit log | Record everything the agent saw, for reproducibility and post-incident review. | S | Med |
| 104 | Deterministic replay | Snapshot index state + query so a session can be replayed exactly — debugging + regression capture. | L | Med |
| 105 | Path-scoped access policy | Pluggable policy adapter: which paths a given agent identity may retrieve. | M | Med |

### K. Evaluation maturity — measure what actually matters

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 106 | NDCG / MAP / Precision@K | Add ranked-quality metrics to `eval/harness.py` — Recall@K + MRR alone hide ranking quality. ROADMAP item. | S | Med |
| 107 | Per-query latency + P95 | Track latency in the eval harness, not just relevance — the §1.3 spec target (p95 ≤ 500 ms) is currently unmeasured in CI. | S | Med |
| 108 | Public benchmark harness | Run on SWE-bench-retrieval / CodeSearchNet; publish NDCG@10 — stropha-vs-semble §1 notes stropha has no public benchmark. | L | High |
| 109 | Synthetic golden generation | LLM-generate query/answer pairs from the graph for cold-start eval on a fresh repo. | M | Med |
| 110 | End-to-end retrieval-conditioned eval | Measure whether better retrieval actually raises *agent task success* (SWE-bench-style), not just recall. | L | High |
| 111 | Context-efficiency metric | Score "right answer per token returned" — the meta-harness's true objective function. | M | High |

### L. Frontier — the most ambitious, still realistic

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 112 | RL-learned retrieval policy | Treat the router (#25) as an RL agent optimizing end-task success from the feedback log (#34). Local training. | L | High |
| 113 | Sleep-time compute | Between sessions, the index reasons in background: builds community summaries, mines memories, refines negatives — uses the existing detached hook process. | L | High |
| 114 | Self-describing tools | stropha generates its own MCP tool descriptions from usage stats so the harness picks the right tool — the harness tuning itself. | M | Med |
| 115 | World-model checkpointing | Version the graph so an agent can ask "what did this code look like 50 commits ago" — time-travel over the world model. | L | Med |

---

## 6. If you only do ten

The highest leverage toward the meta-harness, in order. Each unlocks others.

1. **#13 `sessions` table** — the keystone; everything stateful depends on it.
2. **#1 token-budget packing** — the smallest change that shifts the concept from "K chunks" to "a working set".
3. **#2 `assemble_context`** — the meta-harness's headline verb; what an agent actually wants.
4. **#34 usage feedback logging** — the label source; no learning loop exists without it.
5. **#36 auto-eval on commit** — turns `eval` from a tripwire into a live signal, non-blocking per §3.5.
6. **#68 + #69 reranker on by default + code-aware signals** — closes the documented quality gap vs. semble.
7. **#64 vector quantization** — fixes the float32 footprint blocker for large repos.
8. **#82/#83 verification tools** — the cheapest way to add the "close the loop" trait a meta-harness defines.
9. **#54/#55 GraphRAG community summaries + global mode** — converts the graph from navigation into *understanding*.
10. **#92 MCP sampling** — gives stropha LLM power for HyDE/enrichment with zero new cost or cloud dependency.

---

## 7. The evolution path — four altitudes

stropha does not become a meta-harness in one jump. It climbs:

- **Altitude 0 — Retriever (today).** Stateless, returns chunks. *Done.*
- **Altitude 1 — Context engine.** Owns the working set: budget packing, compression, JIT handles, dedup. (§A) — *the cheapest, highest-ROI rung.*
- **Altitude 2 — Stateful context engine.** Sessions, memory, feedback loop, verification. (§B, §D, §H) — *the rung that makes it a "meta" anything.*
- **Altitude 3 — Meta-harness.** Orchestration, sub-agent routing, self-tuning, harness-agnostic substrate. (§C, §I, §L) — *the destination.*

Each altitude is independently shippable and independently valuable — stropha is a better RAG backend at Altitude 1 even if it never reaches Altitude 3. None of the rungs requires breaking a §3 invariant: memory needs a forward-only v8 migration, the feedback loop runs in the already-detached hook process, and every LLM-powered option uses the existing local Ollama/MLX backends or MCP sampling (#92), keeping the local-only default intact.

**Bottom line:** stropha is not a meta-harness, and the gap is real — but it is a gap of *altitude*, not of *foundation*. The adapter framework, the symbol graph, the drift detector, the eval instrument, and the cost ledger are exactly the organs a meta-harness is built from. Add session memory and an orchestration loop, and the same codebase *is* one.
