# Meta-Harness — implementation plan for stropha

> **Status:** Historical RFC with completed v1 delivery. The phased body records the
> 2026-05 plan; it is not a current claim that "nothing shipped". The audited
> delivery matrix below is authoritative for implementation status.
> **Version:** 1.0 (RFC)
> **Date:** 2026-05-21
> **Delivery audit:** 2026-08-13. Current operational rows were reconciled with
> the live source tree on that date.
> **Audience:** maintainers of `stropha` and LLM agents operating it.
> **Companion documents:**
> - `docs/comparisons/stropha-vs-meta-harness.md` — the concept analysis + 118 improvement options this plan operationalizes.
> - `docs/architecture/stropha-pipeline-adapters.md` — the adapter framework that *is* the search space.
> - `CLAUDE.md` **Invariants (do NOT break)** — the constraints this plan must not break.
> **Primary source:** Lee et al., *Meta-Harness: End-to-End Optimization of Model Harnesses*, arXiv:2603.28052 [cs.AI], 2026 — see §11 for the full citation, links, and the implementation-relevant findings.

---

## 0. TL;DR

A **Meta-Harness** is an outer loop that searches over a harness's code/config to optimize it end-to-end, using an agentic *proposer* with full-history access to every prior candidate's code, scores, and execution traces. At the time of the RFC, stropha owned the declarative adapter search space but not the optimizer apparatus; the delivery matrix below records the subsequent partial implementation.

This RFC proposed that apparatus in **7 phases**. Its schedule and future-tense
exit criteria are retained as planning history. See the matrix below for what is
actually wired at the audited commit.

### 0.1 Delivery matrix

| Phase | Audited status | Delivered evidence | Current gap |
|---|---|---|---|
| 0 — traces/store/outcomes | **Delivered** | `meta/{trace,store,outcome,backfill,feedback}.py`; traced MCP calls, explicit bad-result feedback, and auditable per-hit outcome evidence | Label quality still depends on post-hoc signals and calibration. |
| 1 — evaluator/fitness | **Delivered and wired** | `meta/{fitness,evaluator,gate,frontier,validation}.py`; candidate-specific outcome scoring, signed scalar reporting, search-only optimization, and explicit shortlist holdout validation | End-to-end task success remains a future fitness dimension. |
| 2 — reproducibility | **Delivered** | `meta/{snapshot,index_cache,candidate_index}.py`; every run pins a source snapshot, builds normal-pipeline cache entries, and evaluates isolated writable SQLite clones or native PostgreSQL template clones | Cache reuse still depends on exact indexing, graph, backend, policy, cluster, and cache-format identity. |
| 3 — search space | **Delivered** | bundled manifest + `meta/search_space.py` validation/sampling | The manifest remains curated; registry metadata cannot infer every constraint. |
| 4 — outer loop | **Delivered** | batched loop, Pareto frontier, bandit and agentic proposers, bounded parallel evaluation, durable checkpoints, exact bandit resume, candidate traces, shortlist, and cost artifacts | Agentic proposer resume is intentionally unsupported because exact model state cannot be restored. |
| 5 — code search | **Delivered with a bounded scope** | `local-code`/`cloud-code` proposers, static validation, pytest gate, and sandboxed reranker proxy | v1 accepts one generated reranker file per candidate; code candidates remain review-only. |
| 6 — promotion/operation | **Delivered with a bounded scope** | status/report/promote/prune commands; explicit holdout validation; fail-closed schema-v2 promotion; atomic deep merge into canonical `~/.stropha/config.yaml`; hook-triggered bounded bandit runs | Hook-triggered optimization still depends on its configured runtime and budget. |

**Current limitations:** exact resume is available only for the bandit proposer;
generated code is limited to one sandboxed reranker and cannot be auto-applied;
monetary run cost remains unknown unless exact provider usage and pricing are
available. The historical plan body below keeps the original estimates and
broader ambitions for traceability.

---

## 1. Goal & scope

**Goal.** Implement, on top of stropha, a Meta-Harness in the sense of Lee et al.: an outer loop `propose → validate → evaluate → store → repeat` that discovers harness configurations (and, in v1, harness code) which outperform the hand-tuned `stropha.yaml`.

**In scope.** The optimizer apparatus: artifact store, execution traces, evaluator with a real fitness function, reproducible evaluation, search-space manifest, the loop, two proposers (local bandit + agentic), promotion, sleep-time operation.

**Out of scope.** Making stropha a richer *harness* (statefulness, context shaping, guidance — `stropha-vs-meta-harness.md` §A–§C). That is a parallel track (§9): it is not a blocker for the loop, it only widens the search space.

**Non-negotiable constraints** (`CLAUDE.md`, section **Invariants (do NOT break)**).
- Forward-only, idempotent schema migrations — Phases 0–4 add none; if Phase 5+ needs SQL it is a forward-only `v8`.
- Local-only default — the loop, store, evaluator, and a bandit proposer are 100% local. The *agentic* proposer needs a strong coding model and is therefore **opt-in**, gated by an explicitly configured inference backend and its YAML secret when required.
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
| `stropha meta snapshot [--ref REF] [--repo PATH] [--list]` | Create or list pinned source worktrees |
| `stropha meta run [--iterations N] [--candidates K] [--parallel N] [--resume RUN_ID]` | Search on a pinned snapshot and write a frozen search shortlist |
| `stropha meta validate RUN_ID` | Evaluate the frozen baseline and shortlist on holdout |
| `stropha meta status` | Artifact store summary + current best candidate |
| `stropha meta report [--json]` | Candidate leaderboard and promoted-baseline alerts |
| `stropha meta promote CANDIDATE_ID [--run-id RUN_ID] [--apply]` | Print a diff by default; apply only eligible validated config candidates |
| `stropha meta trace CANDIDATE_ID` | List one candidate's evaluation trace files |
| `stropha meta prune` | Apply the artifact-retention policy |

---

## 3.1 Completion contract

This section is the implementation contract for completing the original scope.
It supersedes future-tense details in the historical phase descriptions below.
The behavior in this section is shipped and covered by the Meta-Harness tests.

### Run lifecycle

`stropha meta run` is a search-only optimization command. It freezes one source
snapshot, evaluates the active configuration as the initial seed, then proposes
`K` candidates per round. `--iterations N` is the number of proposal rounds; the
legacy loop attempt budget remains internal. Holdout is not shown to proposers
and is opened only by explicit validation. The original RFC's suggestion of 2–3
additional hand-picked seeds remains historical guidance, not a v1 requirement:

```text
meta run [--candidates K] [--parallel N] [--resume RUN_ID]
  -> snapshot -> baseline(search) -> seeds -> batched search -> shortlist

meta validate RUN_ID
  -> baseline(holdout) + frozen shortlist(holdout) -> validation.json

meta promote CANDIDATE_ID --run-id RUN_ID --apply
  -> recheck identity + validation gate -> atomic config merge
```

Every batch is deterministic under the run seed: proposals are generated and
registered in slot order against the same pre-batch archive; evaluation may run
in parallel, but archive insertion, hooks, checkpoint writes, and output remain
slot ordered. Failed proposals, gates, timeouts, and evaluations consume budget.

### Snapshot and index isolation

- A run pins `HEAD` into one detached worktree. Dirty source changes are not
  included. Every candidate in the run uses that same snapshot SHA.
- `stropha meta run` and `stropha bench run` both apply the machine-local
  `experiments.storage` selection and `experiments.blacklist.adapters` policy
  from `~/.stropha/config.yaml`. SQLite remains the backward-compatible default;
  selecting PostgreSQL measures native pgvector and PostgreSQL BM25 behavior and
  avoids sqlite-vec KNN candidate limits.
- Evaluation storage is always isolated and is never the production index.
  SQLite uses writable file clones. PostgreSQL uses deterministic, marked
  canonical cache databases and a disposable `CREATE DATABASE ... TEMPLATE ...`
  work clone for each Meta candidate or benchmark cell.
- The immutable cache identity is
  `(repo_key, snapshot_sha, indexing_hash, graph_identity, storage_backend,
  experiment_policy, postgres_cluster, cache_format)`.
- A cache miss runs the normal `Pipeline.run()` against the snapshot; that method
  remains the sole owner of graphify loading and FTS augmentation order.
- SQLite candidates are reflink/full-copy clones of a closed cache database;
  writable hardlinks and symlinks are forbidden. PostgreSQL candidates are
  template clones of a canonical cache database whose connections are closed
  before publication.
- Retrieval-only candidates share one cache entry. Indexing-stage candidates
  produce one entry per distinct indexing identity.
- Canonical cache publication is atomic and metadata is verified on every read.
- PostgreSQL canonical cache databases persist across runs for reuse. Disposable
  staging/work databases are cleaned up by exact generated name only after
  fixed-prefix, Stropha-marker, and owner checks. The admin and production
  databases are protected, and no command performs implicit bulk cleanup.

For PostgreSQL, configure a dedicated non-production admin database and role:

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

The admin database name must begin with `<database_prefix>_`, and the role must
have `CREATEDB`. Credentials remain only in `~/.stropha/config.yaml`; persisted
provenance contains the policy fingerprint and non-secret cluster coordinates,
not the DSN or password. The production DSN and production database are never
mutated. Holdout validation fails closed unless the frozen storage backend,
policy fingerprint, and PostgreSQL cluster identity still match the search run.
The dedicated template database is owned by the evaluation role and has
`vector` installed by an administrator; derived databases clone it without
granting the evaluation role superuser privileges.

### Code-search boundary

Phase 5 v1 accepts only one generated **reranker** adapter per candidate. This is
an intentional reduction from arbitrary harness code:

- proposer modes are `local-code` and `cloud-code`, both explicit opt-ins;
- exactly one file `adapters/reranker/<name>.py`, at most 64 KiB;
- exactly one literal `@register_adapter(stage="reranker", name="<name>")`;
- the candidate's `reranker` setting must name that registration;
- static AST checks reject prohibited stages, imports, calls, and path mismatch;
- candidate-aware tests and import/protocol checks run before scoring;
- reranking executes in a fresh constrained subprocess receiving only serialized
  query/hit data, never storage or production pipeline objects;
- candidate files are never copied into `src/stropha/adapters/` and promotion
  remains review-only for code candidates.

The subprocess is defense in depth, not a hostile multi-tenant security
boundary. It uses a sanitized environment, temporary HOME, candidate cwd,
process-group termination, bounded output, wall/CPU/address-space/file/process
limits where supported, and no inherited credentials. Network denial is applied
only where the host OS provides a reliable sandbox facility.

### Proposal history and learning

Proposals have a typed shape: configuration, optional code artifact, hypothesis,
evidence candidate/trace IDs, expected objective deltas, confidence, and usage.
After evaluation, the store records actual parent-relative deltas and a verdict.

Agentic proposers receive a token-budgeted history bundle containing:

1. a compact ledger for every candidate available to the run;
2. full records for the baseline, current Pareto frontier, recent candidates,
   representative failures, and objective-space-diverse examples;
3. selected candidate code, gate/sandbox diagnostics, hypotheses, and traces;
4. explicit included/spilled IDs and token estimates.

Sections are serialized atomically and never substring-truncated. Candidate
evaluation traces are written beneath `candidates/<id>/traces/` and carry the
candidate/run IDs. Holdout evidence is excluded from proposer history.

The local bandit learns categorical, boolean, and enumerated integer arms using
all signed objectives. It samples an objective preference per proposal and adds
frontier-crowding pressure so exploration does not collapse to quality alone.

### Manifest preflight

The search-space manifest is validated against the complete adapter registry.
Every registered stage/adapter must be eligible or explicitly excluded with a
reason. Every feature has one runtime config binding. Invalid ranges, steps,
defaults, duplicate names, dangling constraints, active adapters absent from the
manifest, and CLI/evaluator dimension disagreement fail before any candidate is
registered. The manifest and registry hashes are persisted in provenance. The
machine-local blacklist filters the preflighted space without changing the
checked-in `search_space.yaml`. Independently of that blacklist, `git-diff` is
always excluded because an incremental-only walker cannot build a full-snapshot
experiment index.

### Artifacts and resume

```text
meta/
  snapshots/worktrees/<sha>/
  index-cache/<cache-key>/{index.db,meta.json}  # SQLite backend
  # PostgreSQL caches are canonical marked databases under database_prefix.
  candidates/<candidate-id>/
    config.yaml
    meta.json                 # run/round/slot/hypothesis/status
    scores.json
    gate.json
    sandbox.json
    hypothesis-verdict.json
    adapters/                 # code candidates only
    traces/*.jsonl
  runs/<run-id>/
    provenance.json
    checkpoint.json
    report.json
    cost.json
    shortlist.json
    validation.json           # written only by meta validate
```

`checkpoint.json` is atomically replaced after the seed and every batch. It records
archive/frontier IDs, registered candidate states, next round/slot, proposer RNG
state, budgets, and identity. `--resume` requires exact repository, snapshot,
golden, manifest, registry, options, storage backend, experiment policy,
PostgreSQL cluster identity when applicable, and index-format compatibility.
Scored candidates are never rerun; registered/evaluating candidates are
recovered once.

### Budgets and costs

Run limits include wall clock, rounds, candidate attempts/evaluations, proposer
tokens, per-candidate timeout, and maximum parallel candidates. Search and
explicit holdout validation have separate usage. `cost.json` records proposer
calls/fallbacks, estimated and provider-reported tokens, indexing/cache work,
retrieval queries/tokens/time, candidate states, verification duration, and
artifact bytes. Monetary cost stays `null` unless exact provider usage and a
configured price are available.

### Baseline and promotion gate

The live configuration is evaluated alongside candidates under identical search
and holdout identities. Promotion requires a schema-v2 validation artifact with:

- completed comparable search and holdout evidence for baseline and candidate;
- a search improvement on the declared primary metric;
- no material holdout regression in Success, Recall, MRR, NDCG, latency,
  returned tokens, per-tag metrics, or per-case rank under configured thresholds;
- all named verification checks passing;
- current config/source/index/runtime identity matching validation;
- the storage backend, experiment-policy fingerprint, and PostgreSQL cluster
  identity matching the frozen search run.

Legacy scalar-only baselines are retained as `legacy_unverified`; they never
authorize promotion and are replaced only after same-identity revalidation.

---

## 4. Phased plan

Every phase has the same shape: **deliverables · files touched · CLI/schema · tests · effort · exit criteria.** Effort: S (<1d) · M (1–4d) · L (>1w).

### Phase 0 — Artifact store + execution traces · effort **L (~5–7d)**

The foundation. The paper's central finding is that *full-history* access (code + scores + traces) beats compressed feedback decisively; this phase builds that history. It is also independently valuable — the trace log alone is a usage-feedback dataset.

**Deliverables**
- `meta/trace.py`: `TraceRecorder` writes one JSONL line per retrieval call. Record fields: `trace_id`, `session_id`, `ts`, `tool`, `query`, `params`, `candidate_id`, `results[]` (`chunk_id`, `rel_path`, `lines`, `score`, `streams`, `rank`), `latency_ms`, `returned_tokens`. **`session_id` is the unifying key** — it ties a `search_code` to subsequent `get_symbol` / `find_callers` / `get_file_outline` calls inside the same agent session, so engagement can be reconstructed post-hoc (see `meta/outcome.py`).
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
- `meta/outcome.py`: a post-hoc job (invoked from the post-commit hook background) that reads recent traces and back-fills each result with a layered **outcome label**. **The naïve "did the AI edit a file we returned?" rule is insufficient** — read-only retrievals (definition lookup, interface reading, type lookup, documentation, config schema reading, test-contract reading, pure-explanation queries) are *useful* but produce no `git diff` overlap. Treating them as misses would train the tuner to deprioritize exactly the retrievals AI agents most rely on for correct-on-first-try code. The supervision signal is therefore tiered:

  | Tier | Condition | Label contribution |
  |---|---|---|
  | T1 — direct edit | The returned chunk's file is in the commit's `git diff`. | Strong positive |
  | T2 — definition-of-used-symbol | The chunk **defines** a symbol whose name **appears in** code edited by the commit (resolved via the existing graphify symbol graph: `calls`, `imports`, `references` edges). Catches the "AI read the class to use it correctly" case. | Strong positive |
  | T3 — graph-neighborhood | The chunk is one or two hops in the call/import graph from any edited file. | Weak positive |
  | T4 — follow-up engagement | Same `session_id` shows subsequent `get_symbol` / `find_callers` / `get_file_outline` / `find_tests_for` calls for paths or symbols in the returned chunks. | Medium positive |
  | T5 — same-session re-query | Within the same `session_id`, the agent issued a similar query within N seconds (token-overlap ≥ threshold). | Weak negative |
  | T6 — explicit miss report | The agent called the new `report_bad_result` MCP tool (stretch goal, option #41 of the companion analysis) flagging the result as unhelpful. | Strong negative |
  | T7 — no signal | No edit, no engagement, no re-query, no explicit report. | **Neutral / unknown — NOT a miss.** Absence of evidence is not evidence of absence. |

  The label is a small JSON document attached to each trace (`outcome.tiers: [...]`, `outcome.score: float`, `outcome.evidence: {edited_files, symbols_in_diff, follow_up_calls, …}`). The `outcome.score` is what fitness consumes; the evidence dict is what the agentic proposer reads.

  **Token efficiency (WIN-391):** `session_id` is also the task identity. A
  task is considered solved when any trace in its session receives positive
  T1–T4 evidence. Outcome backfill records total returned tokens across all
  observed tasks divided by solved tasks; unsuccessful attempts remain in the
  numerator so failures are not free. With no solved tasks the ratio is `null`,
  never a misleading zero.

  **Learned stream weights (WIN-380):** the same backfill joins each labeled
  hit to its `streams` attribution and repository identity. It fits a
  zero-prior, regularized mean tier score per stream, maps scores to positive
  multipliers, and normalizes each repository profile to mean 1. Profiles are
  written atomically under `meta/stream-weights/`; hybrid RRF hot-reloads them
  and multiplies them by query-adaptive weights. Missing, malformed, or
  unattributed feedback leaves retrieval behavior unchanged.

- Wire `TraceRecorder` into `server.py` (all MCP tools, not just `search_code` — engagement tiers depend on capturing `get_symbol`/`find_callers`/etc.) and `retrieval/search.py` (CLI path). Recording is best-effort and never on the latency-critical path — fire-and-forget append.
- **Stretch goal — `report_bad_result` MCP tool**: a 13th tool the agent calls when a search clearly misfired (option #41 in the companion analysis). Single call, structured payload (`trace_id`, `reason: enum`), zero retrieval work. This is the only explicit negative signal source and is the cheapest improvement to label quality after T1+T2 land.

**Files** new `meta/{trace,store,outcome}.py`; edits to `server.py` (all 12 tools wrapped + optional 13th tool), `retrieval/search.py`, `tools/hook_install.py` (add the outcome back-fill step to the hook template). `meta/outcome.py` consumes the graphify symbol graph already available via `Storage.graph_*` tables — no new graph code.
**Schema** none — filesystem-only.
**Tests** `tests/unit/test_meta_trace.py`, `test_meta_store.py`, `test_meta_outcome.py` (~30 tests): trace round-trip, store layout, **per-tier label correctness** (synthetic diff + synthetic graph + synthetic engagement), neutral-vs-negative discrimination, end-to-end backfill against a real commit.
**Exit criteria** every retrieval call appends a trace; `stropha meta trace stats` reports call counts; outcome back-fill produces tiered labels against a real commit (at least one T1, one T2, and one T7-neutral case observable); trace recording adds <2 ms p95 to a search; T2 (definition-of-used-symbol) actually fires — measured by enabling a stropha-on-stropha session and verifying ≥1 read-only chunk gets labeled positive that the old rule would have missed.

### Phase 1 — Evaluator: fitness function + held-out split · effort **M (~3–4d)**

Turn `eval` from a manual tripwire into a callable fitness function the loop optimizes.

**Deliverables**
- Split the golden set: tag every entry in `tests/eval/golden.jsonl` with `split: search|holdout` (≈70/30). The proposer only ever sees `search`; generalization is reported on `holdout` (the paper's anti-overfitting protocol).
- `meta/fitness.py`: a **multi-objective** scorer. The paper (§3, Algorithm 1 line 13) returns a Pareto frontier, not a scalar — "When multiple objectives are relevant (e.g., accuracy and context cost), we evaluate candidates under Pareto dominance and report the resulting frontier." Our implementation respects this:
  - The primary objectives, exposed as a tuple `ScoreCard.objectives = (quality, context_efficiency, latency)`:
    - `quality = w_q1·NDCG@10 + w_q2·Recall@10 + w_q3·outcome_score` (the "is the retrieval good?" axis — weighted blend of relevance metrics AND the tiered outcome label from Phase 0)
    - `context_efficiency = answer_tokens / returned_tokens` (the "does the AI need to read less?" axis — the paper's headline win co-axis)
    - `latency = p95 search latency` (the "is it fast enough?" axis — minimised)
  - A **scalar reduction** `fitness(scorecard) → float = quality + w_ce·context_efficiency − w_lat·latency` is provided for the bandit proposer (which is scalar by necessity) and for the regression-alert in Phase 6. **The agentic proposers + the final Phase 4 output + `stropha meta promote` consume the Pareto frontier directly, not the scalar.**
  - **Critical constraint on `outcome_score`**: must NOT be derived solely from T1 (direct edit). The tiered labeling exists precisely to weight read-only retrievals (T2 definition-of-used-symbol, T4 follow-up engagement) at parity with edit-driven ones. A composite that collapses to T1-overlap retrains the tuner to prefer chunks-about-to-be-edited over definitions-to-be-read; that inversion is the failure mode this design is built to prevent.
  - Weights live in `meta/search_space.yaml` and are themselves a (frozen-after-calibration) search dimension. Initial values: `w_q1=w_q2=0.4, w_q3=0.2, w_ce=0.5, w_lat=0.0` (latency disabled until Phase 0 traces accumulate enough p95 data).
- **NDCG / Precision@K**: ~~added to `eval/harness.py` (option #106)~~ already implemented at `eval/harness.py:47-92,182,189,220,227` with graded-relevance support — this line of the original RFC was factually wrong (see `stropha-meta-harness-evaluation.md` §4.1). Phase 1's actual remaining metric work is: render NDCG/Precision in the by-tag CLI table (`cli.py:947-965` ignores them today), add MAP, add per-query latency capture.
- `meta/evaluator.py`: `evaluate_candidate(config, split) → ScoreCard`. It (1) **validates** — builds stages via `pipeline/builder.py` and probes health, equivalent to `stropha pipeline validate`, rejecting candidates that fail; (2) builds/loads the index for that config (Phase 2); (3) runs the golden set on the requested split; (4) returns a structured `ScoreCard` (per-metric, per-tag, latency, token cost, mean outcome_score per tier).

**Files** new `meta/{fitness,evaluator}.py`; edits to `eval/harness.py` (MAP + per-query latency + by-tag NDCG/Precision rendering — NDCG/Precision themselves already exist), `tests/eval/golden.jsonl` (`split:` tagging), `cli.py` (eval rendering).
**Schema** none.
**Tests** `test_meta_fitness.py`, `test_meta_evaluator.py` (~22 tests): fitness monotonicity, **read-only retrieval is rewarded** (synthetic: a config that retrieves definitions consulted-but-not-edited beats one that retrieves edited-but-redundant chunks), validation rejects a broken config, scorecard shape, split isolation, weight-sensitivity sanity.
**Exit criteria** `evaluate_candidate(load("stropha.yaml"), "search")` returns a `ScoreCard`; an intentionally broken config is rejected, not scored; eval reports MAP and per-tag NDCG/Precision; the read-only-retrieval test from above passes (proving the fitness rewards comprehension, not just edit-overlap).

### Phase 2 — Reproducibility: pinned snapshots + index cache · effort **M (~3–4d)**

Candidates must be compared under identical conditions, and re-indexing per candidate must not be ruinous.

**Deliverables**
- `meta/snapshot.py`: `stropha meta snapshot` creates a pinned git worktree at a chosen SHA under `.stropha/meta/snapshot/src/`. All candidate evaluations run against this frozen tree (option #118).
- **Index cache** — the key efficiency lever. An index is keyed by the
  indexing-relevant adapter identities plus the frozen source, graph, storage,
  experiment-policy, cluster, and cache-format identity. Candidates differing
  only in retrieval/reranker config **share** a canonical cache; candidates that
  change indexing trigger one rebuild against the snapshot. SQLite caches live
  under the Meta artifact tree, while PostgreSQL caches are marked canonical
  databases that persist for reuse. Without this, every candidate pays a full
  re-embed.
- Per-candidate isolation: SQLite evaluations receive a writable copy-on-write
  file clone; PostgreSQL evaluations receive a disposable database cloned from
  the canonical cache template. Concurrent candidates never mutate each other or
  production storage (option #12).

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

### Phase 4 — The outer loop · effort **L (~10–14d)**

Algorithm 1, with three proposers. **This is the Meta-Harness.**

**Deliverables**
- `meta/loop.py`: `MetaHarness.run(iterations, candidates_per_iter, proposer, budget)` — Algorithm 1 from the paper. **First, initialize the population `ℋ`** with the live `stropha.yaml` plus 2-3 hand-picked variants (e.g., reranker-on, recursive-merge-on, contextual-enricher-on); evaluate each, log into the store. **Then** for each iteration: ask the proposer for `k` candidate configs → validate + evaluate each on the `search` split → write `config.yaml`/`meta.json`/`scores.json`/`traces/` into the store → update the Pareto frontier → repeat. Tracks both the scalar running best (for the bandit) AND the Pareto frontier of `(fitness, context_efficiency)` (for the agentic proposers); honors a wall-clock / token budget (option #101, #32). **Typical `k` per the paper: 2-3.**
- **`meta/proposer/skill.md`**: a Markdown operational skill for the agentic proposers, written in the Claude Code "agent skill" format. Per the paper (Appendix D tip #1, the strongest lever on whether the loop works): describes the directory layout, the CLI commands available, the output format, the forbidden actions (don't edit `src/stropha/adapters/`, don't write `stropha.yaml` directly, don't egress credentials), and the objectives. **Must NOT prescribe a diagnosis procedure** — the proposer's reasoning is left free. Plan 3-5 short shakedown evolution runs purely to debug this skill before the full Phase 4 launch.
- `meta/proposer/bandit.py` — **the zero-LLM control.** Iterations 1-2 sample the manifest (numpy random; LHS deferred); thereafter Thompson sampling / UCB over per-dimension reward estimates from stored scorecards. Zero LLM, zero network — proves the loop infrastructure works and provides the "did search structure help?" floor. Per the paper's Table 4, Best-of-N (pure random sampling) reached 34.0 median accuracy where Meta-Harness reached 50.0 — the bandit lives somewhere in between; **the bandit is not expected to capture the paper's headline gain.** Its value is debuggability and being fully local.
- `meta/proposer/inference_helper.py` — **the recommended local-default agentic proposer.** Uses the existing `stropha.inference.generate()` helper (MLX/Ollama) to drive a small coding model (e.g. `mlx-community/Qwen2.5-Coder-7B-Instruct` or `32B`) through a grep/cat reading loop over `.stropha/meta/`. Reads `skill.md` + the Pareto frontier + a sampled subset of prior candidates' code+scores+traces, proposes `k` new configs *with a stated hypothesis*. Local-only, no credentials needed. **Caveat (per the paper Section 5):** Meta-Harness was validated with Claude Opus 4.6; the paper explicitly notes "a broader study of how the effect varies across proposer agents remains for future work". A local 7B-32B model may not reach Opus 4.6's "median 82 files per iteration / 20+ candidates per step" diagnostic depth, so this proposer's gain may be capped well below the paper's 50%-vs-34.6% ablation result.
- `meta/proposer/agentic.py` — **opt-in cloud escalation.** Drives Claude Code via its CLI (paper's design) OR an Anthropic API call. Same skill, same store access, same output contract. Gated behind a config flag + credentials; absent credentials, the loop silently uses `inference_helper`, and absent BOTH MLX/Ollama and credentials, falls back to the bandit.
- CLI `stropha meta run`; final step evaluates the **Pareto frontier** (not just the running-best candidate) on the **held-out** split and prints search-vs-held-out for every frontier point (overfit check across the trade-off curve, mirroring the paper's Figure 3).

**Files** new `meta/loop.py`, `meta/proposer/{__init__,bandit,inference_helper,agentic,skill.md}.py`; edits to `cli.py`, `config/models.py` (a `MetaConfig` dataclass).
**Schema** none.
**Tests** `test_meta_loop.py`, `test_meta_proposer_bandit.py`, `test_meta_proposer_inference_helper.py`, `test_meta_proposer_agentic.py` (~40 tests; the LLM-backed proposers are tested with a mocked `inference.generate` / subprocess). Loop terminates on budget, store is consistent after a crash mid-iteration, bandit improves mean fitness on a synthetic landscape, initial seed population is evaluated before iteration 1, **Pareto frontier is correctly maintained as Pareto-dominated candidates are added/superseded.**
**Exit criteria** `stropha meta run --iterations 5 --candidates 3 --proposer bandit` completes fully locally and produces a Pareto frontier with at least one point that dominates the baseline `stropha.yaml` on the `search` split; `--proposer inference-helper` works locally without credentials; `--proposer agent` works when credentials are present and falls back cleanly to `inference-helper` then bandit when not. **The held-out split shows no point on the frontier regressing materially vs the baseline** (sanity check, not a strict pass/fail).

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
- Sleep-time operation (option #116): an optional hook step runs a *bounded* number of bandit-proposer iterations in the **existing detached post-commit background** (`flock`-guarded, budget-capped). The hook foreground still exits in <100 ms (`CLAUDE.md` invariant **Hook <= 100 ms before fork**). Off by default; enabled via `stropha hook install --meta`.
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
| **Proxy-metric overfitting** — optimizing Recall@K yields harnesses good at Recall@K, not at helping agents. | The composite fitness (§Phase 1) blends NDCG + token-efficiency + tiered outcome_score + latency. The held-out split (Phase 1) catches overfit. Roadmap: a retrieval-conditioned end-to-end task eval (`stropha-vs-meta-harness.md` #110) as the true fitness once trace volume allows. |
| **Edit-only labeling would train the wrong tuner** — a label that flags chunks as "useful" only when their file ends up in `git diff` systematically underweights definition lookups, interface reading, type lookups, documentation, config schema reads, test-contract reads, and pure-explanation queries — likely the majority of real retrievals. The tuner would learn to deprioritize exactly the read-only retrievals AI agents most rely on for correct-on-first-try code. | The tiered labeling in §Phase 0 (T1 edit + T2 definition-of-used-symbol + T3 graph-neighborhood + T4 follow-up engagement + T5/T6 negative signals + T7 neutral-unknown) explicitly treats absence of edit as **neutral, not negative**. T2 uses the existing graphify symbol graph to bridge "edited line X" → "definitions consulted to write X correctly". Fitness weight `w₄·outcome_score` consumes the *tiered* score, never raw T1 overlap. The Phase 1 read-only-retrieval test gates the design against regression. |
| **Re-indexing cost per candidate** dominates loop wall-clock. | The index cache (Phase 2) keyed by indexing-`adapter_id`s; bias the search-space manifest so cheap retrieval-only knobs are explored first. |
| **Local-only tension** — a good agentic proposer needs a strong model. | The bandit proposer is the local default and the MVP path; the agentic proposer is opt-in and gated by explicit inference configuration. The loop, store, and evaluator are always 100% local. |
| **A proposed code candidate breaks the process** (Phase 5). | Subprocess isolation + `pytest` gate before scoring; production `adapters/` never touched. |
| **Trace volume / disk growth.** Paper §1 reports a single evaluation can produce up to **10,000,000 tokens** of diagnostic information (≈30-40 MB at 3-4 bytes/token). At our scale a search-code trace is closer to 5-10 KB, but a full Phase 4 run captures ~60 candidates × the trace tail × the per-eval golden run, which at the upper bound is single-digit GB per run. | Traces are JSONL with rotation. Concrete pruning rule: keep top-10 + bottom-10 + last-50 (the paper Appendix D notes worst candidates are informative). Periodic compaction (monthly default) drops trace bodies older than 90 days while keeping `config.yaml` + `scores.json` + `meta.json` indefinitely. `stropha cost` reports per-run disk footprint. |
| **Proposer capability cap.** Paper §5 explicitly notes: "our experiments demonstrate that harness search can work with one particularly strong coding-agent proposer (Claude Code); a broader study of how the effect varies across proposer agents remains for future work." A local 7B-32B model may not reach Opus 4.6's "median 82 files per iteration / 20+ candidates per step" diagnostic depth. The `inference-helper` proposer mode (Phase 4) is therefore an extrapolation beyond the paper's validated regime. | Treat `inference-helper` as a v1 prototype, not a guaranteed reproduction of the paper's gains. Benchmark it against Best-of-N (Table 4 control, 34.0 median vs Meta-Harness 50.0) to measure how much of the structural gain it actually captures. If the gap is small, escalating to the cloud `agentic` proposer becomes a documented, evidence-backed cost decision rather than a leap of faith. |
| **Auto-applying a regressed harness.** | Promotion is human-gated (Phase 6) — the loop proposes (a Pareto frontier), a human picks an operating point and merges. |

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
2. **Labeling-design calibration.** The tiered outcome scheme (§Phase 0) has free parameters: per-tier weights, the token-overlap threshold for T5 same-session re-query detection, the graph-hop radius for T3, the time window for T4 engagement attribution. Initial values are educated guesses. Plan: collect ~1000 traces with all tiers populated, hand-audit a stratified sample (~50), tune per-tier weights to maximize hand-audit agreement, then freeze for Phase 4. **This is the highest-leverage tuning decision in the whole project** — the tuner can only be as good as its label, and the paper's win (50% vs 35%) came from a domain with clean ground truth that code search lacks.
3. **Agentic proposer host.** Claude Code CLI as a subprocess vs. a direct Anthropic API loop vs. the existing local `stropha.inference.generate()` (MLX/Ollama) helper. The local helper has zero precedent risk and preserves the local-only default — recommended as the v1 path (the `inference-helper` proposer), with cloud CLI/API as opt-in v2. See `stropha-meta-harness-evaluation.md` §4.3 + §8.
4. **Does the local agentic proposer reach the agentic-Meta-Harness regime, or just Best-of-N+structure?** The paper (§5) explicitly defers this to future work; we are running ahead of the literature. Measurement protocol: run the bandit, `inference-helper`, and (when available) the cloud `agentic` proposers on the same seed + the same iteration budget, and compare against Best-of-N (Table 4 control). The gap between `inference-helper` and Best-of-N tells us how much of the paper's "agentic gain" a local model captures. If <50% of the cloud-vs-Best-of-N gap, the local mode is meaningful but not transformative.
5. **Promotion autonomy.** Should sleep-time operation ever open a PR automatically (human reviews the PR), or only ever leave a local diff? This plan assumes the latter; revisit after Phase 6 lands. The `stropha meta promote` command surfaces the **Pareto frontier** for the human to choose from (mirroring the paper's Figure 3 accuracy-context curve), not just the running-best scalar.

---

## 11. References & background — the Meta-Harness paper

### 11.1 The paper

> Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn.
> **Meta-Harness: End-to-End Optimization of Model Harnesses.**
> arXiv:2603.28052 [cs.AI], submitted 2026-03-30. Stanford University · KRAFTON · MIT. License: CC BY 4.0.

| Link | URL |
|---|---|
| Abstract | https://arxiv.org/abs/2603.28052 |
| Full text (HTML) | https://arxiv.org/html/2603.28052v1 |
| PDF | https://arxiv.org/pdf/2603.28052 |
| DOI | https://doi.org/10.48550/arXiv.2603.28052 |

```bibtex
@misc{lee2026metaharness,
  title         = {Meta-Harness: End-to-End Optimization of Model Harnesses},
  author        = {Lee, Yoonho and Nair, Roshen and Zhang, Qizheng and
                   Lee, Kangwook and Khattab, Omar and Finn, Chelsea},
  year          = {2026},
  eprint        = {2603.28052},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2603.28052},
  url           = {https://arxiv.org/abs/2603.28052}
}
```

> **Sourcing caveat (rev 1).** This paper postdates the knowledge cutoff of the assistant that drafted this plan. Every claim attributed to it here was extracted from the arXiv HTML and abstract page, fetched 2026-05-21.
> **Update (rev 3, 2026-05-22):** The paper has since been read in full. The findings table below has been corrected against the source (text classification gain, Pareto frontier as the default output, initial seed population, the proposer-skill mechanism, the 10M-token-per-evaluation order of magnitude, the proposer-capability caveat in Section 5). A full alignment audit lives in `stropha-meta-harness-evaluation.md §8`.

### 11.2 Findings that drive this plan

Each phase below traces back to a concrete result in the paper. Read this table as "why the plan is shaped this way".

| Paper finding | Source | Why it matters for stropha | Drives |
|---|---|---|---|
| **Algorithm 1 — explicit seed init**: `Initialize ℋ with valid baseline harnesses → Evaluate seed → for t=1..N: proposer queries 𝒟 → proposes k → validate → evaluate → store → return Pareto frontier` | §3, Alg. 1 | The outer loop always starts from seeded baselines (zero-shot/few-shot/ACE/MCE in the paper's experiments; live `stropha.yaml` + 2-3 hand-picked variants for us). The **return value is a Pareto frontier**, not a single best. | Phase 4 |
| **Full-history filesystem access** — the proposer reads the *source code*, *scores*, and *complete execution traces* (prompts, tool calls, model outputs, state updates) of every prior candidate | §1 + §3 | The artifact store must hold raw traces, not summaries or scalars | Phase 0 |
| **Ablation (Table 3)**: full Meta-Harness 50.0 median / 56.7 best accuracy vs scores-only 34.6/41.3 vs scores+summary 34.9/38.7 | §4.1, Table 3 | Compressed feedback is decisively worse — this justifies paying the cost of trace capture | Phase 0 |
| **Proposer = a coding agent** with a "**minimal domain-specific skill**" (Markdown operational guide). The paper uses Claude Code with Opus-4.6. Skill describes "where to write new harnesses, how to inspect previous harnesses and their execution traces, and what files it can and cannot modify." Per Appendix D tip #1, **the skill's quality is "the strongest lever on whether the loop works"** — but it should NOT prescribe a diagnosis procedure. | §3, App. D | The store is a grep-able filesystem tree, not a database blob; the proposer is an agent steered by a skill, not a raw LLM call | Phase 0 layout · Phase 4 `proposer/skill.md` |
| **File-access statistics (App. A.1, Table 8)**: median 82 files per iteration (range 69–99) on the TerminalBench-2 run — **41% prior harness source code, 40% execution traces, 6% scores, 13% other**. Section 1 cites "over 20 prior candidates per step" referenced; App. A.1 reports the file count directly. | §1, App. A.1 | The store layout must let the proposer pull traces and source code at ~equal volume; the on-disk format must be regex-friendly | Phase 0 layout |
| **+7.7 points AND ~4× fewer context tokens** on online text classification: Meta-Harness reaches 48.6% vs ACE 40.9% (7.7-point gain) AND uses 11.4K context tokens vs ACE 50.8K (4.46× cheaper). Both improvements simultaneously, not a trade-off. | §4.1, Table 2 | Token efficiency *and* accuracy are co-optimizable; the fitness function should preserve both as a **Pareto trade-off**, not collapse them. | Phase 1 fitness (Pareto) |
| **Pareto frontier as default output** — Algorithm 1 line 13 returns the Pareto frontier, not a scalar best. "When multiple objectives are relevant (e.g., accuracy and context cost), we evaluate candidates under Pareto dominance and report the resulting frontier." | §3, Alg. 1 line 13 | The bandit is scalar by necessity, but the agentic proposer + the final output + the promotion step must surface the frontier so the user can choose the operating point. | Phase 4 loop · Phase 6 promote |
| **Best-of-N is a meaningful control baseline** (Table 4): Best-of-N (independent samples from the seed, no search structure) reaches 34.0 median, Meta-Harness 50.0 — the 16-point gap is the gain attributable to the loop's structure (proposer reading prior candidates) | §4.1, Table 4 | Phase 4 should benchmark each proposer against Best-of-N; the bandit lives between Best-of-N and the full agentic proposer | Phase 4 |
| The loop "evaluates **validated** harnesses" — Algorithm 1 line 11 explicitly gates on `H passes interface validation` before evaluation | §3, Alg. 1 line 11 | A validation gate precedes evaluation | Phase 1 |
| Results transfer across **five held-out models** for retrieval-augmented math (+4.7 points average over no-retriever baseline) | §4.2, Table 6 | Generalization must be measured on a held-out split kept away from the proposer | Phase 1 split |
| **TerminalBench-2 win = environment bootstrapping**: discovered harness injects a one-shot environment snapshot (OS, languages, package managers, /app contents) into the initial prompt, "eliminating the 2–4 exploratory turns that agents typically spend discovering what tools and files are available". Total implementation: ~80 lines on top of Terminus-KIRA. | App. B.3 | A repo/environment snapshot as warm-start is a high-value harness feature; informs the §8 parallel track (option #19) | §8 parallel track |
| **Causal reasoning over prior failures (App. A.2)** — at iteration 3, the proposer noticed two prior failures shared a prompt rewrite, hypothesized it was the confound, and tested an isolated structural change. Full trajectory at App. A.2. | App. A.2 | Justifies trace capture: without prior-candidate traces the proposer cannot form these hypotheses | Phase 0 trace schema |
| **Evaluation is the bottleneck** — "Since evaluation is the main computational bottleneck, we give each method the same budget of proposal harness evaluations." Single eval can produce up to **10,000,000 tokens** of diagnostic information. | §4.1 + §1 | Disk growth and pruning rules must account for ≤10M tokens per eval; the index cache (Phase 2) is the dominant efficiency lever | Phase 0 disk · Phase 2 cache |
| **Typical run: ~60 harnesses over 20 iterations** (≈3 candidates per iteration; text classification ran 2/iter, math ran ~2.7/iter) | §3 + §4.1 + §4.2 | `meta run --candidates k` default = 2-3, not higher | Phase 4 defaults |
| **Practical implementation tips (App. D)**: (1) write a good skill — debug with 3-5 short shakedown runs; (2) start from a baseline + a search set hard for it (~50 evals; 50-100 examples); (3) log in regex-friendly format; (4) build a small CLI for the artifact store (Pareto frontier, top-k, code diffs); (5) lightweight validation before full eval; (6) automate evaluation outside the proposer | App. D | Operational checklist for Phase 4 launch | Phase 4 deliverables |
| **Proposer capability caveat** — paper Section 5 explicitly notes: "we evaluate on three diverse domains, our experiments demonstrate that harness search can work with one particularly strong coding-agent proposer (Claude Code); a broader study of how the effect varies across proposer agents remains for future work." | §5 | A local 7B-32B model may not reach Opus 4.6's diagnostic depth; the `inference-helper` proposer mode's ceiling is unknown | §7 risks · Phase 4 |
| **Code-space search is inspectable** — "Overfitting in code space is also more inspectable: brittle if-chains or hard-coded class mappings are visible on inspection in a way that weight-space overfitting is not." | §5 | Additional safety argument for Phase 5 (code search): a regressed candidate's source is grep-able | Phase 5 safety |
| Headline result: discovered harnesses **beat hand-engineered baselines** across three domains | abstract + §4 | The thesis — automated context engineering can outperform the hand-tuned `stropha.yaml` | the whole plan |

### 11.3 Internal references (where to start coding)

| Resource | Role in this plan |
|---|---|
| `docs/comparisons/stropha-vs-meta-harness.md` | Companion concept analysis — the 118 options and the 12-piece gap table this plan operationalizes |
| `docs/architecture/stropha-pipeline-adapters.md` | The adapter framework = the search space (Phase 3) |
| `src/stropha/eval/harness.py` | The evaluator to extend with NDCG + the search/held-out split (Phase 1) |
| `src/stropha/pipeline/builder.py` | `build_stages(resolved)` — how the evaluator instantiates a candidate harness |
| `src/stropha/pipeline/registry.py` | The adapter registry — the source from which the search-space manifest is generated (Phase 3) |
| `src/stropha/pipeline/config.py` | YAML + env + CLI config cascade — the shape of a candidate `config.yaml` |
| `src/stropha/server.py`, `src/stropha/retrieval/search.py` | Where `TraceRecorder` is wired in (Phase 0) |
| `src/stropha/tools/hook_install.py` | The post-commit hook template — extended for outcome back-fill (Phase 0) and sleep-time operation (Phase 6) |
| `CLAUDE.md` **Invariants (do NOT break)** | The invariants every phase must respect |

### 11.4 Prior art

| Reference | Relevance |
|---|---|
| DSPy — https://github.com/stanfordnlp/dspy | Automated optimization of LM pipelines/prompts; Omar Khattab (a Meta-Harness co-author) is its lead author. The closest existing system to "treat the harness as a tunable artifact" — useful prior art for the proposer and the search-space manifest. |
| Model Context Protocol — https://modelcontextprotocol.io/specification | The protocol stropha's harness is exposed over; traces (Phase 0) are captured at the MCP tool boundary. |
