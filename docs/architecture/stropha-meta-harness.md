# Meta-Harness — implementation plan for stropha

> **Status:** Proposed (plan — nothing shipped).
> **Version:** 1.0 (RFC)
> **Date:** 2026-05-21
> **Audience:** maintainers of `stropha` and LLM agents operating it.
> **Companion documents:**
> - `docs/comparisons/stropha-vs-meta-harness.md` — the concept analysis + 118 improvement options this plan operationalizes.
> - `docs/architecture/stropha-pipeline-adapters.md` — the adapter framework that *is* the search space.
> - `CLAUDE.md §3` — the invariants this plan must not break.
> **Primary source:** Lee, Nair, Zhang, Lee, Khattab, Finn. *Meta-Harness: End-to-End Optimization of Model Harnesses.* arXiv:2603.28052 (2026).

---

## 0. TL;DR

A **Meta-Harness** is an outer loop that searches over a harness's code/config to optimize it end-to-end, using an agentic *proposer* with full-history access to every prior candidate's code, scores, and execution traces. stropha already owns the one rare prerequisite — a declarative, addressable search space (the adapter framework). It is missing the entire optimizer apparatus.

This plan builds that apparatus in **7 phases**. Phases 0–4 (≈3–4 weeks) deliver a **working, fully local Meta-Harness** that searches stropha's configuration space. Phases 5–6 (≈3 more weeks) add code-level search and continuous operation. No phase breaks a `CLAUDE.md §3` invariant; Phases 0–4 require **no schema migration** (the artifact store is filesystem-first, matching the paper and keeping it grep-able by an agentic proposer).

---

## 1. Goal & scope

**Goal.** Implement, on top of stropha, a Meta-Harness in the sense of Lee et al.: an outer loop `propose → validate → evaluate → store → repeat` that discovers harness configurations (and, in v1, harness code) which outperform the hand-tuned `stropha.yaml`.

**In scope.** The optimizer apparatus: artifact store, execution traces, evaluator with a real fitness function, reproducible evaluation, search-space manifest, the loop, two proposers (local bandit + agentic), promotion, sleep-time operation.

**Out of scope.** Making stropha a richer *harness* (statefulness, context shaping, guidance — `stropha-vs-meta-harness.md` §A–§C). That is a parallel track (§9): it is not a blocker for the loop, it only widens the search space.

**Non-negotiable constraints (`CLAUDE.md §3`).**
- Forward-only, idempotent schema migrations — Phases 0–4 add none; if Phase 5+ needs SQL it is a forward-only `v8`.
- Local-only default — the loop, store, evaluator, and a bandit proposer are 100% local. The *agentic* proposer needs a strong coding model and is therefore **opt-in**, gated like `VOYAGE_API_KEY`.
- Non-blocking post-commit hook — sleep-time operation (Phase 6) runs only in the existing detached `nohup`+`flock` background.
- Never bypass the adapter framework — config search varies `stropha.yaml`; code search (Phase 5) adds adapters via `@register_adapter`, guarded by `pytest`.

---

## 2. Recap — what exists vs. what is missing

From `stropha-vs-meta-harness.md` §3, the Meta-Harness decomposes into 12 pieces:

| # | Piece | Status today | Phase |
|---|---|---|---|
| 1 | A harness as the optimization target | Partial — stropha is a harness *component* | §9 (parallel) |
| 2 | An accessible search space | **Strong** — adapter framework + `stropha.yaml` | Phase 3 (formalize) |
| 3 | An agentic proposer | Missing | Phase 4 |
| 4 | Artifact store — candidate code/config | Missing | Phase 0 |
| 5 | Artifact store — candidate scores | Partial — `eval` produces them, unstored | Phase 0 + 1 |
| 6 | Artifact store — execution traces | **Missing — the largest gap** | Phase 0 |
| 7 | Evaluator over a search set | Partial — `eval/harness.py`, manual, proxy metric | Phase 1 |
| 8 | Search-set / held-out split | Missing | Phase 1 |
| 9 | Candidate validation | Partial — `stropha pipeline validate` | Phase 1 |
| 10 | The outer loop (Algorithm 1) | Missing | Phase 4 |
| 11 | Reproducible evaluation conditions | Missing | Phase 2 |
| 12 | Isolation for many candidates | Partial — `--index-path` per run | Phase 2 |

---

## 3. Architecture

A new package `src/stropha/meta/` holds the optimizer. It *consumes* the existing pipeline (`pipeline/builder.py`, `eval/harness.py`) but never modifies it — the harness under optimization stays exactly the production code path.

```
                    ┌──────────────────────────────────────────┐
                    │             META-HARNESS LOOP             │
                    │      stropha/meta/loop.py — Algorithm 1    │
                    │   propose → validate → evaluate → store    │
                    └──────────────────────────────────────────┘
              read store │            ▲ write          │ evaluate
                         ▼            │                ▼
   ┌────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
   │    PROPOSER    │  │    ARTIFACT STORE    │  │     EVALUATOR       │
   │ bandit (local) │─▶│  .stropha/meta/      │◀─│ build pipeline from │
   │ agentic (opt-in)│  │   candidates/<id>/   │  │ candidate config +  │
   │  → k configs   │  │     config.yaml      │  │ run fitness on the  │
   └────────────────┘  │     scores.json      │  │ pinned snapshot     │
          ▲            │     traces/*.jsonl   │  └─────────────────────┘
          │            │     meta.json        │            │
   search-space        │   index.jsonl        │            ▼
   manifest (Phase 3)  └──────────────────────┘  ┌─────────────────────┐
                                                  │   SNAPSHOT (Ph. 2)  │
                                                  │ pinned source @SHA  │
                                                  │ + cached indexes    │
                                                  └─────────────────────┘
```

**Module layout (`src/stropha/meta/`):**

| Module | Responsibility | Phase |
|---|---|---|
| `trace.py` | `TraceRecorder` — append-only execution traces of every retrieval call | 0 |
| `store.py` | `ArtifactStore` — read/write `.stropha/meta/` candidate tree | 0 |
| `outcome.py` | Post-hoc label: correlate returned chunks with subsequent `git diff` | 0 |
| `fitness.py` | Composite fitness metric (retrieval quality + token-efficiency + latency) | 1 |
| `evaluator.py` | `evaluate_candidate(config, split) → ScoreCard`; validation | 1 |
| `snapshot.py` | Pinned source worktree + per-indexing-config index cache | 2 |
| `search_space.py` | Machine-readable search-space manifest loader + sampler | 3 |
| `loop.py` | `MetaHarness` orchestrator — Algorithm 1 | 4 |
| `proposer/bandit.py` | Local proposer — Thompson/UCB over the manifest | 4 |
| `proposer/agentic.py` | Opt-in proposer — drives a coding agent over the store | 4 |
| `proposer/code.py` | v1 proposer — proposes new adapter *code* | 5 |
| `promote.py` | Human-gated promotion of a discovered config | 6 |

**CLI — a new `stropha meta` command group** (`cli.py`):

| Command | Purpose |
|---|---|
| `stropha meta snapshot [--commit SHA]` | Freeze a source snapshot for reproducible evaluation |
| `stropha meta run [--iterations N] [--candidates k] [--proposer bandit\|agent] [--budget …]` | Run the loop |
| `stropha meta status` | Artifact store summary + current best candidate |
| `stropha meta report [--json]` | Candidate leaderboard, search vs. held-out scores |
| `stropha meta promote <candidate_id>` | Emit a `stropha.yaml` diff for human review (never auto-applies) |
| `stropha meta trace {tail,stats}` | Inspect the execution-trace log |

---

## 4. Phased plan

Every phase has the same shape: **deliverables · files touched · CLI/schema · tests · effort · exit criteria.** Effort: S (<1d) · M (1–4d) · L (>1w).

### Phase 0 — Artifact store + execution traces · effort **L (~5–7d)**

The foundation. The paper's central finding is that *full-history* access (code + scores + traces) beats compressed feedback decisively; this phase builds that history. It is also independently valuable — the trace log alone is a usage-feedback dataset.

**Deliverables**
- `meta/trace.py`: `TraceRecorder` writes one JSONL line per retrieval call. Record fields: `trace_id`, `session_id`, `ts`, `tool`, `query`, `params`, `candidate_id`, `results[]` (`chunk_id`, `rel_path`, `lines`, `score`, `streams`, `rank`), `latency_ms`, `returned_tokens`.
- `meta/store.py`: `ArtifactStore` over `.stropha/meta/` — directory-per-candidate (grep-able, matching the paper's filesystem design):
  ```
  .stropha/meta/
    candidates/<candidate_id>/
      config.yaml      # the resolved harness config
      meta.json        # adapter_id digests, parent id, iteration, ts
      scores.json      # written by Phase 1
      traces/*.jsonl   # execution traces from this candidate's eval runs
    index.jsonl        # flat append-only log of all candidates
    runs/<run_id>/     # per-loop-run logs
  ```
- `meta/outcome.py`: a post-hoc job (invoked from the post-commit hook background) that reads recent traces, diffs the commit, and back-fills each trace with an `outcome` (`edited_files`, `cited_chunk_ids`, derived `label`). This is the supervision signal.
- Wire `TraceRecorder` into `server.py` (all MCP tools) and `retrieval/search.py` (CLI path). Recording is best-effort and never on the latency-critical path — fire-and-forget append.

**Files** new `meta/{trace,store,outcome}.py`; edits to `server.py`, `retrieval/search.py`, `tools/hook_install.py` (add the outcome back-fill step to the hook template).
**Schema** none — filesystem-only.
**Tests** `tests/unit/test_meta_trace.py`, `test_meta_store.py`, `test_meta_outcome.py` (~25 tests): trace round-trip, store layout, outcome correlation against a synthetic diff.
**Exit criteria** every `search_code` call appends a trace; `stropha meta trace stats` reports call counts; outcome back-fill labels ≥1 trace against a real commit; trace recording adds <2 ms p95 to a search.

### Phase 1 — Evaluator: fitness function + held-out split · effort **M (~3–4d)**

Turn `eval` from a manual tripwire into a callable fitness function the loop optimizes.

**Deliverables**
- Split the golden set: tag every entry in `tests/eval/golden.jsonl` with `split: search|holdout` (≈70/30). The proposer only ever sees `search`; generalization is reported on `holdout` (the paper's anti-overfitting protocol).
- `meta/fitness.py`: a composite `fitness(scorecard) → float`:
  `fitness = w₁·NDCG@10 + w₂·Recall@10 + w₃·context_efficiency − w₄·latency_penalty`
  where `context_efficiency = answer_tokens / returned_tokens`. Weights live in `meta/search_space.yaml`. NDCG / Precision@K added to `eval/harness.py` (option #106).
- `meta/evaluator.py`: `evaluate_candidate(config, split) → ScoreCard`. It (1) **validates** — builds stages via `pipeline/builder.py` and probes health, equivalent to `stropha pipeline validate`, rejecting candidates that fail; (2) builds/loads the index for that config (Phase 2); (3) runs the golden set on the requested split; (4) returns a structured `ScoreCard` (per-metric, per-tag, latency, token cost).

**Files** new `meta/{fitness,evaluator}.py`; edits to `eval/harness.py` (NDCG/Precision@K, split awareness), `tests/eval/golden.jsonl`.
**Schema** none.
**Tests** `test_meta_fitness.py`, `test_meta_evaluator.py` (~20 tests): fitness monotonicity, validation rejects a broken config, scorecard shape, split isolation.
**Exit criteria** `evaluate_candidate(load("stropha.yaml"), "search")` returns a `ScoreCard`; an intentionally broken config is rejected, not scored; `eval` reports NDCG.

### Phase 2 — Reproducibility: pinned snapshots + index cache · effort **M (~3–4d)**

Candidates must be compared under identical conditions, and re-indexing per candidate must not be ruinous.

**Deliverables**
- `meta/snapshot.py`: `stropha meta snapshot` creates a pinned git worktree at a chosen SHA under `.stropha/meta/snapshot/src/`. All candidate evaluations run against this frozen tree (option #118).
- **Index cache** — the key efficiency lever. An index is keyed by a hash of only the *indexing-relevant* `adapter_id`s (walker + chunker + enricher + embedder + storage). Candidates differing only in retrieval/reranker config **share** a cached index; candidates that change indexing trigger one rebuild against the snapshot, cached at `.stropha/meta/snapshot/indexes/<index_hash>/index.db`. Without this, every candidate pays a full re-embed.
- Per-candidate isolation: each evaluation gets its own `STROPHA_INDEX_PATH` (a copy-on-write clone of the cached index) so concurrent candidates never corrupt each other (option #12).

**Files** new `meta/snapshot.py`; edits to `evaluator.py` (consume the cache), `cli.py` (`meta snapshot`).
**Schema** none.
**Tests** `test_meta_snapshot.py` (~12 tests): snapshot pinning, index-cache hit/miss by config hash, isolation under two concurrent evals.
**Exit criteria** two evaluations of the same config on the same snapshot produce byte-identical scorecards; two retrieval-only variants share one cached index (one rebuild, not two).

### Phase 3 — Search-space manifest · effort **S–M (~2d)**

Make the search space explicit and machine-readable — it bounds the proposer and feeds the bandit.

**Deliverables**
- `meta/search_space.yaml`: a declarative spec — per pipeline stage, which adapters are eligible and which config knobs vary, with ranges/choices. Example: `retrieval.streams.vec-cosine.k ∈ [10,100]`, `retrieval.reranker ∈ {noop, cross-encoder, code-aware}`, `chunker.languages.python ∈ {ast-generic, file-level}`.
- `meta/search_space.py`: loads the manifest, validates a candidate config against it, and exposes `sample()` (random / Latin-hypercube) + `enumerate_dimensions()` for the bandit.
- The manifest is generated *from the adapter registry* where possible (`registry.py` already knows every registered adapter) so it cannot drift from reality.

**Files** new `meta/search_space.{py,yaml}`.
**Schema** none.
**Tests** `test_meta_search_space.py` (~14 tests): manifest validates the shipped `stropha.yaml`, rejects an out-of-range knob, `sample()` only produces valid configs.
**Exit criteria** the current `stropha.yaml` validates against the manifest; `sample()` yields 100 valid, distinct candidate configs.

### Phase 4 — The outer loop · effort **L (~8–12d)**

Algorithm 1, with two proposers. **This is the Meta-Harness.**

**Deliverables**
- `meta/loop.py`: `MetaHarness.run(iterations, candidates_per_iter, proposer, budget)` — for each iteration: ask the proposer for `k` candidate configs → validate + evaluate each on the `search` split → write `config.yaml`/`meta.json`/`scores.json`/`traces/` into the store → repeat. Tracks the running best; honors a wall-clock / token budget (option #101, #32).
- `meta/proposer/bandit.py` — **the local-only default.** Iterations 1–2 sample the manifest (Latin-hypercube); thereafter Thompson sampling / UCB over per-dimension reward estimates from stored scorecards. Zero LLM, zero network — keeps a fully local Meta-Harness path (§3 invariant 6).
- `meta/proposer/agentic.py` — **opt-in.** Drives a coding agent (Claude Code via its CLI, or an Anthropic API call) with a prompt that points at `.stropha/meta/` and `search_space.yaml`, includes the N best and N worst prior candidates with configs + score deltas + trace excerpts, and asks for `k` new `stropha.yaml` candidates *with a stated hypothesis*. Mirrors the paper's "form causal hypotheses" design and its grep/cat filesystem access. Gated behind a config flag + credentials; absent credentials, the loop silently uses the bandit.
- CLI `stropha meta run`; final step evaluates the best candidate on the **held-out** split and prints search-vs-held-out (overfit check).

**Files** new `meta/loop.py`, `meta/proposer/{__init__,bandit,agentic}.py`; edits to `cli.py`, `config.py` (a `meta:` config section).
**Schema** none.
**Tests** `test_meta_loop.py`, `test_meta_proposer_bandit.py`, `test_meta_proposer_agentic.py` (~35 tests; the agentic proposer is tested with a mocked agent). Loop terminates on budget, store is consistent after a crash mid-iteration, bandit improves mean fitness on a synthetic landscape.
**Exit criteria** `stropha meta run --iterations 5 --candidates 4 --proposer bandit` completes fully locally and produces a candidate with `search`-split fitness ≥ the baseline `stropha.yaml`; `--proposer agent` works when credentials are present and falls back cleanly when not.

### Phase 5 — Code search (v1) · effort **L (~10–15d)**

Let the proposer modify harness *code*, not just config — the paper's full "search over harness code".

**Deliverables**
- `meta/proposer/code.py`: the agentic proposer may also write **new adapter files** into `candidates/<id>/adapters/`, registered via `@register_adapter` (never editing production adapters).
- The evaluator loads candidate adapters in an isolated import namespace and **guards every code candidate with `uv run pytest -q`** — a candidate whose code breaks the suite is rejected before scoring (extends Phase 1 validation).
- Sandboxing: code candidates run in a subprocess with the candidate's `adapters/` on `sys.path`, so a faulty proposed adapter cannot corrupt the loop process.

**Files** new `meta/proposer/code.py`; edits to `evaluator.py` (subprocess isolation, pytest gate), `registry.py` (namespaced candidate registration).
**Schema** none (a `v8` would only be needed if candidates introduce storage changes — out of scope for v1).
**Tests** `test_meta_code_search.py` (~18 tests): a proposed adapter that fails tests is rejected; a valid proposed adapter is scored; the loop process survives a candidate that raises on import.
**Exit criteria** the loop accepts a hand-written valid candidate adapter and scores it; rejects one that fails `pytest`; production `adapters/` is never modified.

### Phase 6 — Promotion + sleep-time operation · effort **M (~3–4d)**

Close the loop into continuous, safe operation.

**Deliverables**
- `meta/promote.py` + `stropha meta promote <id>`: emits a **diff** of `stropha.yaml` (or a branch + commit) for human review. It never auto-applies — promoting a harness change is a shared-state action and stays human-gated, consistent with the §3 spirit on risky operations.
- Sleep-time operation (option #116): an optional hook step runs a *bounded* number of bandit-proposer iterations in the **existing detached post-commit background** (`flock`-guarded, budget-capped). The hook foreground still exits in <100 ms (§3 invariant 5). Off by default; enabled via `stropha hook install --meta`.
- `stropha meta report` leaderboard; regression alert when the live `stropha.yaml` falls behind a discovered candidate by a margin on held-out.

**Files** new `meta/promote.py`; edits to `cli.py`, `tools/hook_install.py`, `cost.py` (meta-loop cost rows).
**Schema** none.
**Tests** `test_meta_promote.py` (~12 tests): promote emits a valid diff and never writes `stropha.yaml` directly; sleep-time step respects the budget and the 100 ms foreground bound.
**Exit criteria** `stropha meta promote` produces a reviewable diff; the hook with `--meta` runs iterations in background without slowing `git commit`.

---

## 5. The MVP milestone

**Phases 0 + 1 + 2 + 3 + 4 (bandit proposer only)** = a complete, fully local, fully offline Meta-Harness that searches stropha's configuration space and discovers configs beating the hand-tuned default. ≈**3–4 weeks**. It needs no cloud, no schema change, and breaks no invariant. Everything after — the agentic proposer, code search, sleep-time — is strict upside on that base.

---

## 6. Sequencing & dependencies

```
Phase 0 ──┬─▶ Phase 1 ──▶ Phase 2 ──┐
          │                          ├─▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
          └─▶ Phase 3 ───────────────┘
```

Phase 0 is the hard dependency for everything. Phases 1+2 and Phase 3 can proceed in parallel after 0. Phase 4 needs 1, 2, 3. Phases 5 and 6 are independent upside on 4 and may be reordered.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Proxy-metric overfitting** — optimizing Recall@K yields harnesses good at Recall@K, not at helping agents. | The composite fitness (§Phase 1) already blends NDCG + token-efficiency + latency. The held-out split (Phase 1) catches overfit. Roadmap: a retrieval-conditioned end-to-end task eval (`stropha-vs-meta-harness.md` #110) as the true fitness once trace volume allows. |
| **Re-indexing cost per candidate** dominates loop wall-clock. | The index cache (Phase 2) keyed by indexing-`adapter_id`s; bias the search-space manifest so cheap retrieval-only knobs are explored first. |
| **Local-only tension** — a good agentic proposer needs a strong model. | The bandit proposer is the local default and the MVP path; the agentic proposer is opt-in and gated like `VOYAGE_API_KEY`. The loop, store, and evaluator are always 100% local. |
| **A proposed code candidate breaks the process** (Phase 5). | Subprocess isolation + `pytest` gate before scoring; production `adapters/` never touched. |
| **Trace volume / disk growth.** | Traces are JSONL with rotation; `meta/store.py` prunes candidates below a score percentile after N iterations, keeping the best + worst (the paper shows worst candidates are informative to the proposer). |
| **Auto-applying a regressed harness.** | Promotion is human-gated (Phase 6) — the loop proposes, a human merges. |

---

## 8. Parallel track — harness enrichment (optional)

This plan optimizes stropha *as it is*. The richer stropha is *as a harness* — statefulness, context shaping, guidance (`stropha-vs-meta-harness.md` §A/§B/§C) — the more surface a Meta-Harness has to optimize. Those options are **not blockers** for this plan; each one shipped simply becomes one or more new dimensions in the Phase 3 search-space manifest. Recommended ordering: land the MVP (§5) first to get the loop running, then let the loop itself help evaluate harness-enrichment changes.

---

## 9. Effort summary

| Phase | Deliverable | Effort | Schema | Cloud |
|---|---|---|---|---|
| 0 | Artifact store + execution traces | L (~5–7d) | none | no |
| 1 | Evaluator: fitness + held-out split | M (~3–4d) | none | no |
| 2 | Reproducible snapshots + index cache | M (~3–4d) | none | no |
| 3 | Search-space manifest | S–M (~2d) | none | no |
| 4 | The outer loop (bandit + agentic proposer) | L (~8–12d) | none | opt-in (agentic only) |
| 5 | Code search (v1) | L (~10–15d) | none | opt-in |
| 6 | Promotion + sleep-time operation | M (~3–4d) | none | no |

**MVP (Phases 0–4, bandit): ≈3–4 weeks, fully local. Full plan: ≈7–9 weeks.**

---

## 10. Open questions

1. **Fitness ground truth.** The composite metric is a proxy. Is a retrieval-conditioned task eval (an agent solving golden tasks with the candidate harness) worth the cost as the loop's true objective, or does the held-out composite suffice? Decide before Phase 4 weights are tuned.
2. **Agentic proposer host.** Claude Code CLI as a subprocess vs. a direct Anthropic API loop — the former reuses the user's existing harness and auth; the latter is simpler to sandbox. Prototype both in Phase 4.
3. **Promotion autonomy.** Should sleep-time operation ever open a PR automatically (human reviews the PR), or only ever leave a local diff? This plan assumes the latter; revisit after Phase 6 lands.
