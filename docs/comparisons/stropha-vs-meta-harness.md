# stropha as a harness — and the Meta-Harness that optimizes it

**Status:** Conceptual analysis · 2026-05-21 (rev. 3 — realigned to the published Meta-Harness vocabulary)
**Scope:** stropha @ HEAD (the *concept*, not a line-by-line audit), positioned against the **harness / Meta-Harness** distinction from the literature.
**Goal:** locate stropha precisely in that vocabulary, show what it would take to make it a full *harness* and then a *Meta-Harness target*, and enumerate **118 realistic, state-of-the-art improvement options** — without breaking the §3 invariants (drift detection, forward-only migrations, local-only default, non-blocking hook).

**Primary source:**
> Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn. *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052 [cs.AI], submitted 2026-03-30. Stanford · KRAFTON · MIT. CC BY 4.0.
> [abstract](https://arxiv.org/abs/2603.28052) · [HTML](https://arxiv.org/html/2603.28052v1) · [PDF](https://arxiv.org/pdf/2603.28052) · [DOI](https://doi.org/10.48550/arXiv.2603.28052)
> Summarized from the arXiv HTML; metadata confirmed against the abstract page (fetched 2026-05-21). The paper postdates the assistant's knowledge cutoff — read it directly before relying on a specific figure.
> The implementation plan derived from this analysis is `docs/architecture/stropha-meta-harness.md`.

**Other sources verified:**
- `CLAUDE.md` §2 (snapshot), §3 (invariants), §6 (history)
- `src/stropha/server.py` (`grep @mcp.tool` → 9 tool decorators), `src/stropha/adapters/**` (registry tree)
- `src/stropha/retrieval/*`, `src/stropha/eval/harness.py`, `src/stropha/cost.py`
- `ROADMAP.md`, `docs/comparisons/stropha-vs-semble.md`, `docs/architecture/stropha-system.md`

> This is an *analysis*, not a spec. Nothing here is shipped. The §3 invariants and the local-only directive bound every option below; cloud-dependent items are flagged.

---

## 1. The vocabulary — harness, context engineering, Meta-Harness

Earlier revisions of this document used "meta-harness" loosely. The term now has a precise published definition, and this revision adopts it.

### 1.1 The three terms (Lee et al., 2026)

- **Harness** — *"a stateful program that wraps a language model and determines what context the model sees at each step."* The harness is not the model; it is the code around it that decides what to store, retrieve, and present. The paper's framing: *"the performance of LLM systems depends not only on model weights, but also on their harness… yet harnesses are still designed largely by hand."*
- **Context engineering** — the practice of refining the code around an LLM to improve system performance. It is what good harness design *is*.
- **Meta-Harness** — *"an outer-loop system that searches over harness code for LLM applications."* It automates context engineering: an agentic *proposer* (the paper uses Claude Code with Opus-4.6) reads a filesystem of every prior candidate's **code, scores, and full execution traces**, proposes new harnesses, evaluates them, and stores the results. The proposer is steered by a "minimal domain-specific skill" (a Markdown operational guide). Algorithm 1 returns a **Pareto frontier** of harnesses, not a single best. The paper's key empirical claim: full-history access beats compressed-feedback optimizers decisively (Table 3 ablation on text classification: full Meta-Harness 50.0 median accuracy vs scores-only 34.6 vs scores+summary 34.9), and the discovered harnesses beat hand-engineered baselines across three domains (text classification, math retrieval, agentic coding).

### 1.2 Where stropha sits

> **stropha today is not a harness in Lee et al.'s sense — it is a harness *component*: the context-sourcing organ.**

A harness is *stateful* and decides *all* the context the model sees *at each step*. stropha is stateless per MCP call, it is *invoked* rather than wrapping the model, and it sources *material* without shaping it, sequencing it, or budgeting it. It is the part of a harness that answers "where is X?" — a critical part, and the expensive one, but one organ.

This gives a clean two-stage climb:

1. **stropha → a full harness.** Add statefulness, information shaping, and guidance so stropha *determines* an agent's context rather than merely supplying candidates for it. (§A–§I below.)
2. **stropha → a Meta-Harness target.** Make that harness *automatically optimizable*: emit code + scores + traces into an artifact store, treat `eval` as a real fitness function, and run an agentic proposer over the adapter search space. (§J–§L below.)

### 1.3 The thesis

> **stropha is the context-sourcing component of a harness. It already nails the costly prerequisite — finding the right material — and, unusually, it is *already structured as a search space*. The climb is to make it a full harness, then to let a Meta-Harness search that space.**

The interesting observation is the second clause. The paper notes that automation wins *"once a search space becomes accessible through improved coding agents."* For most LLM apps, harness code is an undifferentiated blob. stropha's harness component is **already a declarative search space**: a 6-stage adapter framework with registered, swappable adapters; YAML-composed pipelines; `adapter_id` digests that make every point in the space addressable and cache-coherent; drift detection that makes changes safe to apply incrementally; an `eval` golden set; a `cost` meter. stropha is not just a harness component — it is an *unusually Meta-Harness-ready* one. It is missing the artifact store and the proposer loop, not the search space.

---

## 2. Side-by-side: component → harness → Meta-Harness

| Dimension | stropha today — harness *component* | A full harness (Lee et al.) | A Meta-Harness (Lee et al.) |
|---|---|---|---|
| Job | Source relevant material on demand | Determine every piece of context the model sees, each step | Search over harness code to optimize it end-to-end |
| State | Stateless per MCP call (`query_cache` = exact-match) | Stateful across a session | Stateful across *runs* — a full-history artifact store |
| Output | K raw chunks / graph nodes | A shaped, budgeted, guided working set | Better *harnesses* (code + config) |
| Information shaping | None — chunks returned near-raw | Distilled, compressed, deduplicated, fused, budget-fit | n/a — it tunes the shaper |
| Guidance | None — reports what exists | What to do / avoid / verify / look at next | n/a — it tunes the guide |
| Designed by | Hand (adapters, `stropha.yaml`) | Hand — "designed largely by hand" (the paper's problem) | An agentic proposer (the paper uses Claude Code) |
| Feedback used | `eval` golden set = manual tripwire | Consumes feedback to adapt within a session | Consumes code + scores + execution traces of all prior candidates |
| Optimization target | Recall@K / MRR (proxy metrics) | Task success within a session | Held-out task success, across models |
| Search space | A 6-stage adapter framework — *already declarative* | The harness's own code | The harness's code + config (stropha's adapter space) |
| Reached via | — (this is stropha today) | §A–§I of this document | §J–§L of this document |

**Read of the table:** the columns are a ladder, and stropha already stands on the first rung holding the materials for the third — its adapter framework is the search space a Meta-Harness needs, decades ahead of the typical "blob of harness code" the paper has to cope with.

---

## 3. Where stropha already fits the picture

### 3.1 stropha as a harness — organs present and missing

| Organ of a harness | stropha's analogue | Status |
|---|---|---|
| Context sourcing | 4-stream hybrid retrieval + RRF + graph tools — the hard, costly organ | **Production** |
| World model | graphify mirror: `graph_nodes`/`graph_edges`/`graph_meta`, communities, god nodes, `rationale_for`, multi-repo (v7) | **Production** |
| Freshness | `adapter_id` drift detection + `files` dirty cache + git-diff walker + commit hook | **Production** |
| Statefulness | — (`query_cache` is exact-match, not session state) | **Missing** |
| Information shaping | — (`recursive` merge is the closest; off by default) | **Missing** |
| Guidance | — (`smart_search` routes one query; never advises the agent) | **Missing** |

Three production organs, three missing — and the three present (sourcing, world model, freshness) are the ones that take years to get right.

### 3.2 stropha as a Meta-Harness *target* — preconditions

This is stropha's non-obvious strength. A Meta-Harness needs five things to optimize a harness; stropha already supplies three.

| Precondition the Meta-Harness needs | stropha's analogue | Status |
|---|---|---|
| An accessible search space | 6-stage adapter framework, registered adapters, YAML compositions | **Strong** — declarative, enumerable |
| Addressable, cache-coherent changes | `adapter_id` digest + drift detection (§3 invariant #1) | **Strong** — every config point is identified |
| A fitness function | `eval/harness.py` — Recall@K, MRR, golden set | **Partial** — exists, but is a proxy metric |
| A full-history artifact store (code + scores + traces) | — (structlog + `cost.py` log fragments; nothing unified) | **Missing** |
| An agentic proposer | — (but the post-commit hook + a Claude Code agent is half the wiring) | **Missing** |

The paper's headline finding is that *full-history* access (raw code + scores + traces) beats compressed feedback. stropha's gap is precisely there: it has the search space and the addressing, but no artifact store and no proposer. §J–§L close exactly that.

---

## 4. The two gaps

**The harness gap — stropha sources, it does not determine context.** Every hit returns as raw chunk text. There is no token budget (`search_code` gives K results, not "the best context that fits in N tokens" — note the paper's discovered harness wins on text classification by **+7.7 accuracy points AND uses ~4× fewer context tokens** simultaneously, not as a trade-off). No compression: boilerplate costs the same as the matching line. No cross-source fusion: `find_callers` and `search_code` arrive as disjoint payloads. No statefulness: every call is amnesiac. And no guidance — stropha reports that `FsrsCalculator.next_interval` exists; it never says the symbol has 14 callers, that ADR-008 constrains it, that three tests cover it, or that the agent's claim it "calls `decay()`" is false per the graph. All of that is latent in stropha's own graph and is simply never surfaced.

**The Meta-Harness gap — stropha is hand-tuned, never searched.** Its adapter composition, RRF `k=60`, stream weights, chunk sizes, and enricher choice are all set by hand — exactly the "designed largely by hand" condition the paper identifies as the opportunity. stropha has the search space and the fitness function but no loop closing over them: no artifact store of what was tried, no proposer reading that history, no held-out generalization check.

The 118 options below are tagged by which gap they close: **[Harness]** options make stropha a full harness; **[Meta-Harness]** options build the outer optimization loop.

---

## 5. 118 improvement options

Continuously numbered 1–118. **Effort:** S (<1d) · M (1–4d) · L (>1w). **Impact** is on agent task success, not micro-benchmarks. Each group is tagged **[Harness]** (makes stropha a better context-determining harness) or **[Meta-Harness]** (builds the automated optimization loop). Every option respects §3 (forward-only migrations, drift detection, local-only default, non-blocking hook); cloud-dependent items are flagged `[cloud]` and stay optional.

### A. Context shaping — turn "return chunks" into "determine a working set" · **[Harness — information]**

| # | Option | What it does (grounded in stropha) | Effort | Impact |
|---|---|---|---|---|
| 1 | Token-budget result packing | Add `context_budget: int` to `search_code`; pack hits until the budget is hit instead of a fixed `top_k`. Knapsack over (score, token-cost). The paper's harnesses win partly on token efficiency. | S | High |
| 2 | `assemble_context` MCP tool | New 13th tool: given a task, return a ready-to-paste bundle — entry-point files, called symbols, covering tests, ADR rationale — under a budget. The verb of a context-determining harness. | L | High |
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

### B. Statefulness & session memory — the property that *makes* it a harness · **[Harness — statefulness]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 13 | Schema v8: `sessions` table | Forward-only migration adding session identity keyed by MCP client/session id. A harness is *stateful by definition*; this is the keystone. | M | High |
| 14 | Persistent project memory | Store distilled facts/decisions per repo ("the FSRS scheduler is authoritative; don't touch X"), retrievable by future sessions. | M | High |
| 15 | `remember` / `recall` MCP tools | Explicit agent-driven write/read of memory — the agent curates its own long-term notes. | M | High |
| 16 | Conversation-aware retrieval | Accept the last N turns as latent query context to bias retrieval toward the current thread. | M | High |
| 17 | Episodic task memory | "Last time you touched FSRS code you also had to update the migration." Mined from session + git history. | L | High |
| 18 | Memory consolidation | Background job compacts episodic entries into semantic memory — runs in the existing hook's detached background. | L | Med |
| 19 | Environment-bootstrap warm start | On session open, gather a repo snapshot (languages, build tools, test runner, entry points) before the first agent turn — the paper's TerminalBench-2 trick that eliminates 2–4 exploratory turns. | M | High |
| 20 | Memory provenance + TTL | Every memory carries a source commit; drift detection (§3.1) marks it stale automatically. | S | Med |
| 21 | Writable scratchpad resource | An MCP resource the agent uses as durable per-repo notes between turns. | M | Med |
| 22 | Cross-session semantic cache | Promote `query_cache` from exact-match to a shared, semantic, session-spanning cache. | M | Med |

### C. Guidance & next-step steering — direct, don't just inform · **[Harness — guidance]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 23 | `plan_investigation` tool | Decompose a task into an ordered retrieval plan over the graph: entry points → dependencies → tests → rationale. Hands the agent a route, not just facts. | L | High |
| 24 | Server-side retrieval DAG | Multi-step retrieval where step N's results seed step N+1 — an agentic retrieval loop run inside stropha, returning only the distilled result. | L | High |
| 25 | `router` as a first-class stage | Promote query routing to a pluggable adapter stage so routing policy is swappable and benchmarkable — and a point in the Meta-Harness search space. | M | Med |
| 26 | Confidence-gated escalation | Cheap static lookup first; escalate to LLM-routed / graph traversal only when classifier confidence is low. Cost-shaped. | M | High |
| 27 | Sub-query budget allocator | When `query_decomposition` fans out, allocate the token budget across sub-queries by estimated importance. | M | Med |
| 28 | Parallel fan-out with early-exit | Stop spending on slow streams once the RRF top-K is provably stable. | M | Med |
| 29 | `suggest_next_tool` | Given current results, recommend the next graph tool to call — meta-guidance that steers the agent's loop. | M | High |
| 30 | Local refinement sub-agent | A small *local* LLM loop (Ollama/MLX backend, already present) that critiques and re-queries before returning. | L | Med |
| 31 | Named task recipes | Templates ("bug-fix", "add-feature", "review") that fix a tool sequence + budget profile — guidance encoded as a workflow. | M | Med |
| 32 | Cost-budgeted multi-hop | Cap graph traversal by a *cost budget* (fan-out × depth × token estimate), not a fixed depth. | M | Med |
| 33 | Speculative retrieval prefetch | Predict the agent's next 2–3 queries from the task and pre-compute during idle time. | L | Med |

### D. Feedback signals — instrument the harness so it can be optimized · **[Meta-Harness — signal]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 34 | Tiered usage-feedback labeling | Log which returned chunks the agent actually used — **not just edited**. `git diff` overlap (T1) catches direct edits; the existing graphify symbol graph bridges to **T2 — chunks whose defined symbols appear in changed code** (the read-the-class-to-use-it case); same-session MCP follow-up calls give T4 engagement; explicit `report_bad_result` (#41) gives T6 negative. Absence of all signals is T7-neutral, not "miss". The label source for the whole loop — see `stropha-meta-harness.md` §Phase 0 for the tier table. | M | High |
| 35 | Learned RRF weights | Outcome backfill now fits regularized per-repo stream multipliers from tiered feedback and `results[].streams` attribution. Atomic profiles hot-reload into hybrid RRF, combine with query-adaptive weights, and safely fall back when evidence is absent or invalid. This replaces globally equal weighting with the first narrow meta-optimization while retaining `k=60` rank smoothing. | M | High |
| 36 | Auto-eval on every commit | Run the golden set in the hook's detached background; track Recall@K over time; alert on regression. Stays non-blocking (§3.5). | M | High |
| 37 | Golden-set auto-expansion | Mine new golden queries from real sessions: (query, the chunk that got edited) is a free labeled pair. | M | High |
| 38 | Counterfactual stream logging | Log what each *disabled* stream *would* have returned, to estimate each stream's marginal value. | M | Med |
| 39 | Regression bisection | When Recall@K drops, bisect over commits/config changes (drift detection makes the changes addressable). | M | Med |
| 40 | Per-tool success metrics | Track find_callers / trace_feature hit-rate; surface dead tools the harness should stop offering. | S | Med |
| 41 | `report_bad_result` tool | An MCP tool the agent calls on a miss; entries become hard negatives for reranker training. | S | Med |
| 42 | Self-tuning chunk granularity | Measure retrieval success per chunk size and adapt per language — close the loop on chunking, the dominant quality lever. | L | Med |
| 43 | Local reranker distillation | Periodically distill the `code-aware` reranker from accumulated feedback via lightweight local fine-tune (LoRA). Local-only. | L | Med |
| 44 | Token-efficiency signal | Log returned-tokens-per-solved-task from outcome backfill. `session_id` identifies a task, positive T1–T4 evidence marks it solved, and all observed-task tokens stay in the numerator so failed attempts are penalized. The paper optimizes explicitly for token frugality, so it must be a first-class signal. | S | Med |

### E. World-model depth — richer facts and guidance material · **[Harness — information + guidance]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 45 | Data-flow / def-use edges | Beyond `calls`: track variable def-use so "what reads this field" resolves. | L | High |
| 46 | Type-hierarchy edges | `implements` / `extends` / `overrides` edges for OO navigation. | M | High |
| 47 | Cross-repo dependency edges | Link a monorepo's packages by their import graph (schema v7 already has `repo_id` on edges). | M | Med |
| 48 | Co-change / logical-coupling edges | Mine git history: "these files change together" — a coupling signal grep cannot see, and a guidance signal ("you'll likely need X too"). | M | High |
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

### F. Retrieval quality — relevance is the floor of good context · **[Harness — information]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 59 | Model2Vec static-embedding stream | A near-free dense lane (static table lookup, ~1 ms) for the planning-loop fast path. The `model2vec` adapter file already exists — wire a retrieval stream around it. | M | High |
| 60 | Late chunking (Jina) | Embed the whole document, then pool token embeddings per chunk boundary — chunk embeddings stay sibling-aware. Needs a long-context embedder. | L | High |
| 61 | ColBERT late-interaction stream | Per-token embeddings + MaxSim at query time, behind the `retrieval-stream` adapter stage. | L | High |
| 62 | SPLADE learned-sparse stream | Learned term expansion; complements BM25. A deferred ROADMAP item — revive behind the stream adapter. | L | Med |
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

### G. Liveness — fresh context is the only correct context · **[Harness — information]**

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

### H. Verification & grounding — the sharpest form of guidance · **[Harness — guidance]**

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

### I. Protocol surface — deliver the determined context well · **[Harness — delivery]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 90 | Working set as an MCP resource | Expose the assembled context (#2) as a streamed resource, not only a tool call — lower per-turn overhead. | M | Med |
| 91 | MCP `prompts` for task recipes | Ship bug-fix / review / add-feature prompts as MCP prompts, pre-wired to the right tool sequence (#31). | M | Med |
| 92 | MCP `sampling` for enrichment | Let stropha request a completion from the *client's* model for HyDE/enrichment — no separate LLM, zero extra cost. | M | High |
| 93 | Streaming tool results | Stream hits as they fuse so the agent reads before reranking finishes — cuts perceived latency. | M | Med |
| 94 | Harness-agnostic context API | A stable JSON contract so Cursor / Zed / Continue all get the identical context service. | M | High |
| 95 | A2A endpoint | Expose retrieval over emerging agent-to-agent protocols, not only MCP. | L | Low |
| 96 | Multi-tenant index server | One stropha process serving many repos/agents with per-repo isolation. | L | Med |
| 97 | Shared context bus | Multiple agent sessions on one repo share a retrieval cache + memory. | L | Med |

### J. Observability & the artifact substrate — what the Meta-Harness proposer reads · **[Meta-Harness — substrate]**

The paper's central finding: a proposer with **full-history access** (code + scores + execution traces) decisively beats one fed compressed feedback. This group builds that substrate.

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 98 | OpenTelemetry retrieval DAG tracing | Per-stage spans across the pipeline — the raw execution traces a Meta-Harness proposer inspects. Deferred in ROADMAP; now load-bearing. | M | High |
| 99 | Per-session cost & savings ledger | Extend `savings.py` into a per-agent-run dashboard: tokens saved, compute spent, value delivered. | M | Med |
| 100 | `explain_ranking` tool | Given a query + chunk, return why it ranked where it did (per-stream contributions, RRF math). | M | Med |
| 101 | Per-session budget enforcement | Hard token/compute caps with graceful degradation (drop reranker, drop dense lane) instead of silent overrun. | M | Med |
| 102 | Secret-egress guard | Scan returned chunks for secrets before they leave the process — `.strophaignore` excludes files; this guards *content*. | S | High |
| 103 | Retrieval audit log | Record everything the agent saw, every call — the trace history the proposer mines. | S | High |
| 104 | Deterministic replay | Snapshot index state + query so a candidate harness can be re-evaluated exactly — required for fair Meta-Harness scoring. | L | High |
| 105 | Path-scoped access policy | Pluggable policy adapter: which paths a given agent identity may retrieve. | M | Med |

### K. Evaluation — the Meta-Harness fitness function · **[Meta-Harness — fitness]**

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 106 | NDCG / MAP / Precision@K | Add ranked-quality metrics to `eval/harness.py` — Recall@K + MRR alone hide ranking quality. ROADMAP item. | S | Med |
| 107 | Per-query latency + P95 | Track latency in the eval harness — the spec target (p95 ≤ 500 ms) is currently unmeasured in CI. | S | Med |
| 108 | Public benchmark harness | Run on SWE-bench-retrieval / CodeSearchNet; publish NDCG@10 — stropha currently has no public benchmark. | L | High |
| 109 | Synthetic golden generation | LLM-generate query/answer pairs from the graph for cold-start eval on a fresh repo. | M | Med |
| 110 | End-to-end retrieval-conditioned eval | Measure whether a harness config actually raises *agent task success* (SWE-bench-style) — the only fitness signal a Meta-Harness should trust. | L | High |
| 111 | Context-efficiency metric | Score "right answer per token returned" — combined with #44, the paper's true objective (accuracy *and* token frugality). | M | High |

### L. The Meta-Harness — the outer optimization loop itself · **[Meta-Harness — the loop]**

This is the destination: stropha's harness, optimized end-to-end by an agentic outer loop, exactly as Lee et al. describe.

| # | Option | What it does | Effort | Impact |
|---|---|---|---|---|
| 112 | Adapter-composition artifact store | Persist every harness candidate to the filesystem: the resolved `stropha.yaml`, its `adapter_id` digests, eval scores, and full execution traces. The paper's "full-history access" design — and stropha's main missing Meta-Harness organ. | M | High |
| 113 | Agentic proposer loop | A Claude Code / local-agent loop that reads the artifact store (#112) via grep/cat, proposes new adapter compositions — and eventually new adapter *code* — evaluates them with `eval` (#110), and appends results. **This is the Meta-Harness over stropha.** Wireable into the existing post-commit hook background. | L | High |
| 114 | Search-set / held-out eval split | Split the golden set: the proposer optimizes on the search set; generalization is measured on a held-out set — the paper's protocol against overfitting the harness. | S | High |
| 115 | Bandit / RL search over the adapter space | Once the artifact store has history, replace pure agentic proposal with a bandit/RL policy over the (declarative, addressable) adapter space — cheaper, faster convergence. Local training. | L | High |
| 116 | Sleep-time compute | Between sessions, the index reasons in background: builds community summaries, mines memories, refines hard negatives, runs proposer iterations — uses the existing detached hook process. | L | High |
| 117 | Self-describing tools | stropha generates its own MCP tool descriptions from usage stats — the harness tuning its own interface, a micro-Meta-Harness. | M | Med |
| 118 | World-model checkpointing | Version the graph so a candidate harness can be evaluated against a fixed repo snapshot — reproducibility for the Meta-Harness, and time-travel ("what did this look like 50 commits ago") for the agent. | L | Med |

---

## 6. If you only do twelve

Six to make stropha a **harness**, six to make it **Meta-Harness-ready**.

**To become a full harness (Floor 1):**
1. **#13 `sessions` table** — statefulness is the definition of a harness; the keystone.
2. **#1 token-budget packing** — the smallest change that shifts "K chunks" to "a determined working set".
3. **#2 `assemble_context`** — the headline verb of a context-determining harness.
4. **#3 + #12 compression + cross-source fusion** — doubles useful signal per token; one working set, not three payloads.
5. **#29 `suggest_next_tool`** — the cheapest way to start *guiding* the agent, not just informing it.
6. **#82/#83 verify_claim + hallucinated-symbol guard** — the sharpest guidance: telling the agent when it is wrong.

**To become Meta-Harness-ready (Floors 2–3):**
7. **#34 usage feedback logging** — the label source; nothing downstream works without it.
8. **#103 + #98 audit log + OTEL traces** — the execution-trace history the proposer reads (the paper: full history beats compressed feedback).
9. **#112 artifact store** — stropha's single biggest missing Meta-Harness organ.
10. **#110 end-to-end task-success eval** — the only fitness signal worth optimizing.
11. **#114 search-set / held-out split** — so discovered configs generalize instead of overfit.
12. **#113 agentic proposer loop** — the Meta-Harness itself; everything above exists to make this safe and cheap.

---

## 7. The path — four floors

- **Floor 0 — Harness component (today).** Sources context, stateless, invoked. The expensive foundation — sourcing, world model, freshness — is *done*.
- **Floor 1 — Harness.** Stateful; shapes and guides; *determines* what context the agent sees. (§A, §B, §C, §E, §F, §G, §H, §I)
- **Floor 2 — Meta-Harness-ready.** Emits code + scores + execution traces into an artifact store; `eval` is a real, task-conditioned fitness function. (§D, §J, §K)
- **Floor 3 — Meta-Harness.** The outer loop runs: an agentic proposer searches stropha's declarative adapter space, end-to-end optimized against held-out task success. (§L)

Each floor is independently shippable and independently valuable — stropha is a far better RAG backend at Floor 1 even if Floor 3 never runs. None of the floors breaks a §3 invariant: statefulness needs a forward-only v8 migration, the proposer loop runs in the already-detached hook process, and every LLM-powered option uses the local Ollama/MLX backends or MCP sampling (#92), keeping the local-only default intact.

**Bottom line.** In the vocabulary of Lee et al. (2026), stropha today is the *context-sourcing component of a harness* — it has solved the hard prerequisite of finding the right material. Floors 1–2 turn it into a full *harness*: stateful, shaping, guiding, instrumented. Floor 3 is the *Meta-Harness* — and stropha is an unusually ready target for one, because its adapter framework already is the declarative, addressable search space that, the paper argues, is the precondition for automated context engineering to beat hand-tuning. The graph, the drift detector, the `adapter_id` digests, and the `eval` golden set are not just harness organs — they are the scaffolding a Meta-Harness optimizes over. stropha does not have to *become* something foreign to host one; it has to finish being itself.

> *Footnote.* The paper's proposer is Claude Code (Opus 4.6). This document was written by Claude Code (Opus 4.7) reading stropha's own code, graph, and `eval` harness, and proposing 118 changes to it — which is, by Lee et al.'s definition, iteration zero of a Meta-Harness over stropha, run by hand.
