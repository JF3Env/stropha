# Stropha Output Quality Implementation Plan

> **Status:** In progress
> **Created:** 2026-08-12
> **Scope:** Retrieval quality, context quality, and the evidence used to tune them
> **Target pipeline:** PostgreSQL, Qwen3 MLX embeddings, contextual enrichment,
> hybrid RRF, adaptive/learned weights, recursive retrieval, and LLM reranking

## 1. Objective

Improve the usefulness of context returned by Stropha to MCP clients. The work
targets five properties:

1. **Correctness:** relevant files and symbols are returned for the current repository.
2. **Coverage:** multi-file and multi-symbol questions retrieve all important entities.
3. **Ranking:** the best evidence appears early and survives the context budget.
4. **Efficiency:** duplicate or low-value context does not consume tokens or latency.
5. **Reproducibility:** every promoted change is supported by a comparable benchmark.

This plan improves the retrieval product, not final prose generation. Better
retrieval should make downstream agents more accurate while requiring less context.

## 2. Current Baseline

The baseline recorded on 2026-08-12 is:

| Item | Current state |
|---|---|
| Primary embedder | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`, 2560 dimensions |
| Enricher | Contextual, LMS-backed, batch size 8 |
| Retrieval | Dense + sparse + symbol streams fused by RRF |
| Query weighting | Adaptive and per-repository learned weights enabled |
| Recursive retrieval | Enabled, adjacency 5 |
| Reranker | LMS LLM reranker, 20 candidates |
| Router | `noop` in the active pipeline; explicit smart-search routing exists |
| Golden set | 54 cases, 2 graded multi-entity cases |
| Last scoped eval before the final golden additions | Success@10 0.8462, MRR 0.7122, NDCG@10 0.7289 |
| Reranker reliability | 30 fallbacks from 52 attempts, 57.69% fallback rate |
| Index state | Two commits behind at observation time and non-comparable enricher identity |

The clean-index baseline archived after Phase 0 at commit `e47f79d9` contains 54
cases: Success@10 0.8519, entity Recall@10 0.8426, MRR 0.6997, NDCG@10 0.7331,
and 26 reranker fallbacks from 54 attempts (48.15%).

### 2.1 Confirmed identity defect

`ContextualEnricher.adapter_id` includes `:batch=<n>` in
`src/stropha/adapters/enricher/contextual.py`. `Pipeline.__init__` then appends
`:batch=<n>` again when constructing `_effective_enricher_id` in
`src/stropha/pipeline/pipeline.py`. The indexed identity consequently ends in
`:batch=8:batch=8`, while pipeline preflight compares it with the adapter's
single-suffix identity.

This must be fixed before any quality comparison. Reindexing without fixing the
identity construction would reproduce the mismatch.

## 3. Success Criteria

Use the current clean-index baseline as the reference. If baseline collection
changes because the golden set grows, compare only runs with identical golden
hash, repository scope, pipeline identity, and index identity.

### 3.1 Required promotion gates

| Metric | Required gate |
|---|---|
| Index comparability | `--require-clean-index` passes |
| Success@10 | No decrease greater than 1 percentage point on holdout |
| Entity Recall@10 | No decrease greater than 1 percentage point on holdout |
| NDCG@10 | At least +2 percentage points for ranking-focused changes |
| MRR | No regression greater than 1 percentage point |
| Reranker fallback rate | Below 5% before LLM reranking is considered reliable |
| Returned tokens per case | No increase unless entity recall improves proportionally |
| Latency p95 | No increase greater than 25% without an explicit quality tradeoff |
| Structural routing accuracy | At least 95% on deterministic routing cases |
| Golden integrity | Unique IDs, expected entities present, search/holdout split valid |

### 3.2 Decision rule

Do not promote a candidate because it wins one aggregate metric. A candidate is
eligible only if it:

- Passes all required gates.
- Improves either NDCG, entity recall, or solved-task outcome score.
- Has no material per-tag holdout regression.
- Has a documented latency and token-cost tradeoff.
- Reproduces in at least two clean runs when an LLM is in the query path.

## 4. Non-Negotiable Constraints

Every phase must preserve the repository invariants in `CLAUDE.md`:

- Keep the MCP transport HTTP-only and preserve the shared singleton daemon.
- Include every output-changing indexing setting in `adapter_id`.
- Keep migrations idempotent and forward-only; no schema change is planned here.
- Leave graphify lifecycle ownership in `Pipeline.run()`.
- Never make graph-gated tools silently return empty when no graph is loaded.
- Keep the post-commit hook under 100 ms before fork.
- Preserve local-only operation by default; LMS and other cloud services stay optional.
- Keep SQLite FTS5 writes explicit and do not change storage outside adapters.
- Keep query-time toggles independent from index freshness.
- Keep primary and Model2Vec vector spaces independent.

Additional execution constraints:

- Scope production evaluation to the current repository unless cross-repository
  behavior is explicitly under test.
- Never compare runs with different golden hashes or drifted index identities.
- Use the search split for selection and the holdout split only for validation.
- Change one conceptual variable per experiment batch whenever practical.

## 5. Delivery Sequence

The phases are ordered dependencies. Phase 0 and Phase 1 are mandatory before
retrieval tuning. Later phases can be split into separate pull requests, but their
promotion gates remain cumulative.

| Phase | Deliverable | Estimate | Depends on |
|---|---|---:|---|
| 0 | Clean and stable index identity | 1-2 days | None |
| 1 | Reliable reranker and complete diagnostics | 2-4 days | Phase 0 |
| 2 | Comparable benchmark and promotion workflow | 1-2 days | Phases 0-1 |
| 3 | Representative graded golden dataset | 2-4 days | Phase 2 |
| 4 | Feedback-driven stream learning | 2-4 days | Phase 3 |
| 5 | Structural query routing | 2-3 days | Phase 3 |
| 6 | Selective query expansion | 2-4 days | Phases 2-3 |
| 7 | Diversity-aware context packing | 3-5 days | Phases 2-3 |
| 8 | Optional retrieval lanes and long-term optimization | Ongoing | Phases 4-7 |

## 6. Phase 0: Clean and Stable Index Identity

### Goal

Make index freshness deterministic and collect a strict, comparable baseline.

### Implementation

1. Define one owner for effective enricher identity.
2. Prefer the smallest fix: make `Pipeline._effective_enricher_id` equal to
   `self._enricher.adapter_id` because the contextual adapter already includes
   every output-changing field, including effective batch size.
3. Verify non-batched enrichers still have stable IDs and do not gain an inactive
   batch setting.
4. Ensure a contextual batch-size change changes the adapter ID exactly once.
5. Keep `active_enricher_id` writes in `Pipeline.run()` unchanged except for the
   corrected identity value.
6. Improve index-health diagnostics if needed so duplicate suffixes are displayed
   clearly and the recommended action is unambiguous.
7. Commit the fix and let the detached post-commit hook rebuild stale entries.
8. Restart the MCP singleton only after indexing completes and verify health.

### Files

- `src/stropha/pipeline/pipeline.py`
- `src/stropha/adapters/enricher/contextual.py` only if ownership cannot be fixed
  exclusively in the pipeline
- `src/stropha/server_health_tools.py` for diagnostic wording, if required
- `tests/unit/test_pipeline_drift.py`
- `tests/unit/test_pipeline_parallel_enrich.py`
- `tests/unit/test_contextual_enricher.py`

### Tests

- Contextual identity contains exactly one `:batch=8` suffix.
- Changing batch size changes the effective identity.
- Changing concurrency alone does not change enrichment output identity.
- A non-batched enricher has no synthetic batch suffix.
- Clean-index preflight passes after metadata is refreshed.
- Drift preflight fails for a genuinely changed prompt, backend, temperature, or
  batch size.

### Verification

```bash
uv run pytest -q tests/unit/test_pipeline_drift.py \
  tests/unit/test_pipeline_parallel_enrich.py \
  tests/unit/test_contextual_enricher.py
uv run stropha stats
uv run stropha eval --top-k 10 --json --require-clean-index
```

Call the MCP `index_health` tool for the full coverage, lag, commits-behind, and
adapter-drift report. `stropha stats` is the CLI metadata check; there is no
equivalent `stropha index-health` CLI command. Do not bypass the post-commit
indexing workflow merely to make preflight pass.

### Exit criteria

- Active and indexed enricher IDs are byte-identical.
- Stropha is indexed at current `HEAD`.
- The strict eval completes and its JSON report is archived as the baseline.
- Baseline includes provenance, diagnostics, returned tokens, latency, and all
  ranking metrics.

### Rollback

Revert the identity ownership change if it causes unaffected enrichers to reindex.
Do not edit stored metadata manually to conceal drift.

## 7. Phase 1: Reliable Reranking

### Goal

Reduce malformed or empty LLM reranker output from 57.69% to below 5%, and make
fallback behavior explicit and measurable.

### Step 1: Diagnose failure classes

Extend reranker diagnostics beyond aggregate fallback count:

- `empty_response`
- `invalid_json`
- `invalid_ids`
- `backend_error`
- `timeout`
- `repair_attempts`
- `repair_successes`
- primary and repair latency

Never log candidate source or secrets at warning level. Log request ID, reason,
candidate count, and backend identity.

### Step 2: Make generation task-specific

The generic `InferenceBackend.generate` API currently exposes only prompt,
`max_tokens`, temperature, model, and backend. Add optional generation controls
only if the LMS API and other adapters can support them without changing existing
callers:

- Structured JSON response format.
- Thinking/reasoning disabled for ranking-only requests.
- Per-call timeout if it can be implemented consistently.

Keep defaults identical for enrichment and other inference users. Reranker-only
settings belong in `LlmRerankerConfig` and must be included in its `adapter_id`.

### Step 3: Harden parsing and repair

1. Accept a JSON array or a small object such as `{"order": [3, 1, 2]}`.
2. Reject booleans, duplicates, zero, negative IDs, and IDs above candidate count.
3. Preserve partial valid order by appending missing candidates.
4. On malformed output, make at most one bounded repair request using only the
   malformed response and valid ID range.
5. If repair fails, use a configured deterministic fallback.

### Step 4: Benchmark fallback policy

Compare:

- Original RRF order.
- Existing code-aware reranker.
- Cross-encoder reranker.

Prefer the highest-quality local fallback that stays inside latency and dependency
budgets. Do not silently instantiate a heavyweight model unless the adapter is
explicitly configured and health-checked.

### Files

- `src/stropha/adapters/retrieval/reranker/llm.py`
- `src/stropha/inference/backend.py`
- `src/stropha/inference/lms_backend.py`
- Other inference adapters only when extending the shared protocol requires it
- `src/stropha/adapters/retrieval/hybrid_rrf.py`
- `tests/unit/test_llm_reranker.py`
- LMS/backend unit tests associated with `src/stropha/inference/`

### Tests

- Structured valid output reranks correctly.
- Markdown-wrapped output remains accepted for compatibility.
- Empty, malformed, duplicate, boolean, negative, and out-of-range IDs are
  classified correctly.
- One repair attempt occurs, never an unbounded retry loop.
- Repair success and repair failure update diagnostics correctly.
- Backend exceptions preserve deterministic fallback output.
- Reset diagnostics clears every counter.
- `adapter_id` changes when prompt, output mode, repair policy, or fallback changes.
- Existing inference callers retain their current behavior by default.

### Verification

```bash
uv run pytest -q tests/unit/test_llm_reranker.py
uv run stropha eval --top-k 10 --json --require-clean-index
```

Run the strict eval twice because the reranker uses an external LMS service.

### Exit criteria

- Fallback rate is below 5% across two full clean eval runs.
- NDCG@10 is not worse than the no-reranker baseline.
- Reranker p95 latency is recorded and within the accepted budget.
- A backend outage returns deterministic RRF or configured fallback results.

Phase 1 reliability validation on 2026-08-12 produced zero fallbacks in both
54-case clean-index runs. Both runs reported Success@10 0.8889, entity Recall@10
0.8796, MRR 0.7577, and NDCG@10 0.7837. Reranker p95 latency was 1052.325 ms and
1042.691 ms. Reports are archived as
`.stropha/bench/output-quality-phase1-run1.json` and
`.stropha/bench/output-quality-phase1-run2.json` locally.

The fallback benchmark reused the same PostgreSQL index, golden hash, repository
scope, and query-time settings via `stropha eval --reranker`. Results were:

| Policy | Success@10 | MRR | NDCG@10 | p95 latency |
|---|---:|---:|---:|---:|
| Original RRF | 0.7963 | 0.6188 | 0.6592 | 352.428 ms |
| Code-aware | 0.8148 | 0.6330 | 0.6698 | 371.399 ms |
| Cross-encoder | 0.8333 | 0.4137 | 0.5124 | 932.303 ms |

The cross-encoder also incurred a 28.4-second cold first query. Code-aware is the
selected default fallback because it improves every quality metric over original
RRF for a 19 ms p95 increase and has no model dependency. The final active LLM
run with code-aware armed as fallback retained zero fallbacks and NDCG@10 0.7837.

## 8. Phase 2: Comparable Benchmark and Promotion Workflow

### Goal

Turn `src/stropha/bench/active_matrix.yaml` into the authoritative query-time
experiment and promotion workflow.

### Implementation

1. Verify the matrix dry-run still resolves 192 query variants over one indexing
   signature.
2. Include all metrics needed for decisions in each benchmark record:
   - Success@K
   - Entity Recall@K
   - MRR
   - NDCG@K
   - MAP@K
   - Precision@K
   - Returned tokens
   - Latency p50/p95/p99
   - Reranker attempts, fallbacks, fallback rate, and latency
3. Include golden hash, source commit, repository key, index identity, pipeline IDs,
   and matrix cell in every result record.
4. Refuse or mark non-comparable any cell with index identity drift.
5. Add a quality-first report mode that ranks by NDCG but displays guardrail
   metrics beside it.
6. Add or document a shortlist workflow: broad matrix on search, top candidates on
   holdout, then repeated finalists.
7. Keep indexing dimensions pinned until snapshot/index-cache composition can build
   and isolate compatible indexes.

### Suggested experiment sequence

Do not begin with all 192 cells if the reranker is expensive:

1. Smoke test: `--limit 4`.
2. Reranker and candidate pool: 4 cells.
3. Recursive settings: adjacency 3, 5, and 10 against on/off.
4. Adaptive and learned weights.
5. RRF `k` values 40, 60, and 80.
6. Full matrix only after all individual axes run successfully.

### Files

- `src/stropha/bench/active_matrix.yaml`
- `src/stropha/bench/runner.py`
- `src/stropha/bench/report.py`
- `src/stropha/cli.py`
- `tests/unit/test_bench.py`
- Add focused runner/report tests if `test_bench.py` becomes too broad

### Commands

```bash
uv run stropha bench run \
  --matrix src/stropha/bench/active_matrix.yaml \
  --out .stropha/bench/output-quality.jsonl \
  --dry-run

uv run stropha bench run \
  --matrix src/stropha/bench/active_matrix.yaml \
  --out .stropha/bench/output-quality.jsonl \
  --limit 4

uv run stropha bench report .stropha/bench/output-quality.jsonl \
  --metric ndcg@10
```

### Exit criteria

- Every successful record has complete provenance and diagnostics.
- Resume is deterministic and does not mix incompatible runs.
- The baseline cell can be identified without copying a generated cell ID by hand.
- At least one candidate improves NDCG without violating the promotion gates.

The first Phase 2 slice completed on 2026-08-12. The active matrix still resolves
192 query-time cells over one indexing signature. Benchmark records now include
the complete eval summary, reranker diagnostics, pipeline and index identities,
golden hash, repository key, source commit plus dirty-diff hash, matrix identity,
and a shared run identity. Resume and reporting reject missing or mixed run
identities, and `bench report --matrix` resolves the declared baseline without a
copied cell ID. A four-cell smoke run completed with zero failures and deterministic
resume. Staged axis experiments and holdout validation remain open.

The first production-identity staged experiment compared reranker on/off with
`candidate_k` 20/50 while pinning all other axes to baseline. LLM reranking
improved NDCG@10 from 0.6685 to 0.7783 and MRR from 0.6082 to 0.7537 with zero
fallbacks. Increasing `candidate_k` from 20 to 50 changed no quality or token
metric in either reranker state and slightly increased LLM p95 latency from
886.702 ms to 897.111 ms. Keep `candidate_k=20`; do not promote 50.

The recursive stage compared off with adjacency 3/5/10. Enabling recursion
improved NDCG@10 from 0.7767 to 0.7783 and reduced returned tokens from 49,845
to about 38,645 (22.5%). Adjacency 3/5/10 produced identical quality; adjacency
3 had the lowest p95 latency (879.629 ms) and enters the shortlist. Keep
recursion enabled; retain production adjacency 5 until repeated/holdout validation.

The weighting stage compared adaptive and learned weighting on/off. Disabling
both produced the strongest quality-first search result: Success@10 0.9259,
Recall@10 0.9167, NDCG@10 0.7886, MRR 0.7515, and 35,947 returned tokens. Versus
the production baseline this is +0.037 Recall, +0.0103 NDCG, -0.0022 MRR, and
7.0% fewer tokens. Bootstrap intervals remain wide on 54 cases, so this enters
the holdout shortlist but does not change production defaults yet.

The RRF stage compared `k` 40/60/80 with other baseline axes pinned. `k=40`
was strongest on search: Success@10 0.9074, Recall@10 0.8889, MRR 0.7749,
NDCG@10 0.7982, and 39,103 returned tokens. `k=60` scored NDCG 0.7783 and
`k=80` scored 0.7851. Add `k=40` to the holdout shortlist; keep production at
60 until the shortlist is repeated and holdout-validated.

Post-stage review found that `bench run` loaded all 54 golden cases despite the
37 search / 17 holdout labels. The results above are exploratory and cannot be
used for promotion because they exposed holdout during tuning. Benchmark runs
now require an isolated `search` or `holdout` split (default `search`) and bake
the split into provenance and `run_id`. Repeat the shortlist on search, freeze
the candidates, and evaluate only those candidates on holdout.

Benchmark index files are also keyed by the source commit plus tracked changes
and non-ignored untracked content, matching the walker's source scope. Reuse
requires a benchmark-owned completion marker written only after
`Pipeline.run()` returns. An interrupted source-matched build resumes through
hash reconciliation but cannot be evaluated or silently reused while partial.

The isolated search rerun retained `rrf_k=40` (NDCG 0.8100, Recall 0.9054,
MRR 0.7876) and replaced the earlier weights-off candidate with adaptive
weighting off while learned weights remain enabled (NDCG 0.8050, Recall 0.9324,
MRR 0.7712). Adjacency 3/5/10 remained quality-identical and adjacency 3 was no
longer faster, so adjacency stays at 5. The interaction check added adaptive
off with `rrf_k=80` as the strongest search finalist (NDCG 0.8150, Recall
0.9324, MRR 0.7847). `phase2_shortlist_matrix.yaml` freezes these three
candidates plus the production baseline before the one-time holdout run.

Final source-matched repetition used source fingerprint `997ca02809a0` and the
frozen four-cell shortlist. Two executions reproduced every aggregate quality
and token metric exactly. `rrf_k=40` was numerically strongest on both splits,
with zero fallbacks, lower holdout token use, and no per-case holdout rank
regression. These numbers are not promotion evidence, however: the earlier
exploratory all-case run exposed the same 17 holdout cases before the shortlist
was frozen. Repeating an exposed holdout does not restore its independence.

Production therefore remained at `rrf_k=60` pending a new, never-inspected
holdout. Exact exploratory metrics remain in source- and split-identified local
JSONL artifacts under `.stropha/bench/`; they may guide new test authoring but
must not select or promote a candidate.

The first Phase 3 promotion cohort was defined as a checked-in 30-case catalog with
three source-reviewed, graded cases in each category. `stropha bench batch`
validates it against the exposed golden, freezes canonical JSONL before retrieval,
executes only the declared `rrf_k=60` baseline and `rrf_k=40` candidate, applies
aggregate/per-tag/per-case gates, runs an allowlisted verification profile, and
optionally updates only `rrf_k` through explicit `--promote`. The first cohort is
frozen as `phase3-unseen-holdout-v1` with SHA-256
`d8f9b7622e27799ffc2338020d4032157fad9a11c7268535af55569252a446d1`.
This pilot does not satisfy the 120-case Phase 3 completion target.

Post-run review invalidated the cohort's result because the checked-in answer
catalog was itself indexable during benchmark index construction. The observed
metrics are not promotion evidence and must not be interpreted as accepting or
rejecting `rrf_k=40`. The batch specification is marked invalidated, production
remains at `rrf_k=60`, and this exposed cohort must not be rerun. Future batches
validate that their catalog is excluded through `.strophaignore` before freezing
or retrieval.

The replacement `phase3-unseen-holdout-v2` cohort was frozen without retrieval,
then invalidated during pre-commit review because expected paths and symbols were
not positionally paired. It was never executed. The corrected catalog uses the
new `phase3-unseen-holdout-v3` name, then pre-commit review found that graded
IDCG counted paired path and symbol aliases twice. V3 was never executed. The
evaluator now derives ideal relevance once per positional entity, and the cohort
uses the new `phase3-unseen-holdout-v4` protocol name. V4 was frozen without
execution at SHA-256
`2100642667cce8be364c33c8e1d360ad16a4dbe115d043530a20482ca507ee58`.
After commit `7b71c6a`, V4 was executed once. The `rrf_k=40` candidate improved
NDCG@10 from `0.8419` to `0.8451` (`+0.0032`), below the required `+0.02`, so it
was not eligible for promotion. Every other aggregate, safety, per-tag,
per-case, and verification gate passed. Production remains at `rrf_k=60`; V4
is now exposed and must not be rerun or used for tuning.

## 9. Phase 3: Representative Graded Golden Dataset

### Goal

Make evaluation representative enough that tuning improves real agent work instead
of overfitting 54 mostly binary cases.

### Dataset design

Target 120-150 cases before enabling automated promotion:

| Category | Target cases | Required judgments |
|---|---:|---|
| Exact class/function lookup | 15 | Definition path and symbol |
| Conceptual implementation search | 20 | Multiple implementation entities |
| Callers and usage | 15 | Entry symbol plus callers |
| Tests and contracts | 15 | Source symbol plus relevant tests |
| End-to-end feature trace | 15 | Entry point, service, storage, tests |
| Configuration and operations | 10 | Config owner, launcher, docs |
| Documentation and rationale | 10 | ADR/spec plus implementation |
| Ambiguous/similar symbols | 10 | Correct repository and symbol |
| Negative/noisy cases | 10 | Relevant entities among plausible distractors |
| Multi-repository explicit queries | 5 | Repository-specific expected entities |

### Case authoring rules

- Every case has a stable unique ID.
- Every case names at least one expected path or symbol.
- Pair path and symbol entries by entity position.
- Use relevance 3 for necessary evidence, 2 for useful supporting evidence, and 1
  for optional context.
- Prefer two or more entities for feature and workflow questions.
- Add tags for query category, language, graph requirement, and ambiguity.
- Keep approximately 70% search and 30% holdout.
- Do not move a failing holdout case into search merely to tune it.
- Review expected entities after refactors; stale goldens are not retrieval failures.

### Failure-driven additions

For every confirmed poor production result:

1. Save the query and trace ID.
2. Determine the expected entities through code review, not retrieval output.
3. Add a search case if it represents a known recurring failure being fixed.
4. Add a holdout case if it tests generalization of the same failure family.
5. Record why distractor results were plausible.

### Files

- `tests/eval/golden.jsonl`
- `src/stropha/eval/harness.py` only for missing validation or reporting support
- `tests/unit/test_eval_harness.py`
- Optional authoring guide: `tests/eval/README.md`

### Tests

- Production golden file parses in a unit test.
- IDs are unique.
- Every case has expected entities.
- Every relevance key maps to an expected entity.
- Relevance is within the accepted scale.
- Required category tags and split ratios are reported.
- Entity pairing and graded IDCG behavior have direct tests.

### Exit criteria

- At least 120 reviewed cases.
- At least half the cases are multi-entity or graded.
- Search and holdout both contain every major query category.
- Baseline scores are archived with the expanded golden hash.

The representative dataset was expanded to 124 source-reviewed cases on
2026-08-13. It contains 88 search and 36 holdout cases, 72 graded or multi-entity
cases, and both splits cover all ten major categories. The answer catalog is
excluded from indexing and has SHA-256
`c61c5fef05bd3f2f0f67e3da425d3bd46d79c2bca9b779aa135187c4c5e01b1b`.
The post-commit refresh removed the catalog from retrieval and resolved the
active contextual-enricher identity drift at commit `c22a556`. The comparable production baseline
was archived in `tests/eval/baseline-c61c5fef.json`: Success@10 `0.9032`,
Recall@10 `0.8199`, MRR `0.8224`, and NDCG@10 `0.7810` at `rrf_k=60`.

## 10. Phase 4: Feedback-Driven Stream Learning

### Goal

Convert explicit bad-result reports and post-commit outcomes into trustworthy,
per-repository retrieval-weight updates.

### Implementation

1. Preserve `report_bad_result` as the explicit negative-feedback entry point.
2. Resolve each report to the exact trace and returned chunks using
   `resolve_bad_chunks`.
3. Map reasons to bounded labels:
   - `irrelevant`: strong negative
   - `wrong_symbol`: strong negative
   - `stale`: exclude from ranking learning and trigger freshness investigation
   - `duplicate`: negative for diversity, not necessarily for the contributing stream
   - `too_verbose`: negative for packing, not necessarily relevance
4. Merge explicit labels with `run_backfill` outcome labels without counting the
   same task twice.
5. Require a minimum observation count before writing a learned profile.
6. Keep the current regularization toward equal weights for small samples.
7. Write profile provenance: repository key, observation count, label source counts,
   creation time, and training commit range.
8. Add a report command showing the active profile and evidence supporting it.
9. Evaluate the proposed profile against baseline before activation.

### Files

- `src/stropha/meta/feedback.py`
- `src/stropha/meta/backfill.py`
- `src/stropha/meta/stream_weights.py`
- `src/stropha/adapters/retrieval/hybrid_rrf.py`
- `src/stropha/cli.py`
- `tests/unit/test_feedback.py`
- `tests/unit/test_backfill.py`
- `tests/unit/test_stream_weights.py`

### Tests

- Every feedback reason maps to the intended learning channel.
- Stale results do not train stream relevance weights.
- Duplicate and verbosity reports affect only their intended optimization stage.
- Cross-repository labels never update another repository's profile.
- Duplicate traces and tasks are deduplicated.
- Small samples cannot produce extreme weights.
- Profile fingerprints change with data or algorithm changes.
- Corrupt profiles fail closed to adaptive/default weights.

### Exit criteria

- At least 50 attributable observations exist for the target repository before
  automatic use.
- Learned weights beat default/adaptive-only weights on search and do not regress
  holdout.
- Active weight provenance is visible through config diagnostics.

The first Phase 4 evidence-contract slice completed on 2026-08-13. Feedback
reasons now map to isolated relevance, freshness, diversity, packing, or review
channels; only relevance reports may train stream weights. Training examples are
deduplicated by task and chunk with explicit relevance feedback taking precedence
over post-commit outcome labels. Backfill writes schema-v2 candidates with unique
example counts, source counts, generation time, and commit range to
`stream-weight-candidates/`. Hybrid retrieval loads only separately activated
profiles with at least 50 attributable examples and benchmark evidence showing a
search NDCG gain with no holdout NDCG regression. Invalid, legacy, low-sample, or
unvalidated profiles fail closed. `stropha meta stream-weights report` exposes
active and blocked evidence. `stropha meta stream-weights benchmark <repo>` now
builds default/adaptive-only and isolated candidate retrieval stages over the
same clean index with downstream reranking disabled so the learned fusion
multipliers are the only changed ranking input. It evaluates both golden splits, records the golden hash, split
sizes, retrieval and index identities, and NDCG metrics, then activates only when
search improves without holdout regression. No schema-v2 candidate has yet met
the evidence floor and passed this benchmark in the running environment. The
first deterministic run used 87 search and 37 holdout cases at index commit
`1ff91b6`, comparing `learned=off` with candidate fingerprint `b0b96d1d`.
Search NDCG changed from 0.571292 to 0.569699 (-0.001594) and holdout NDCG from
0.608799 to 0.608636 (-0.000163), so the candidate remained blocked.

## 11. Phase 5: Structural Query Routing

### Goal

Use graph traversal for structural questions instead of asking semantic retrieval
to infer exact relationships.

### Implementation

1. Establish a routing benchmark with expected intent and symbol extraction.
2. Start with `PatternRouter`, not the LLM router.
3. Route high-confidence intents:
   - Who calls/uses X to `find_callers`.
   - Tests for X to `find_tests_for`.
   - Related to X to `find_related`.
   - Trace feature X to `trace_feature`.
   - Exact definition queries to symbol lookup.
4. Keep ambiguous conceptual queries on `search_code`.
5. If the graph is not loaded, return the required explicit graph-unavailable
   response or route to an explicitly documented semantic fallback. Never return a
   silent empty graph result.
6. Use `EscalatingRouter` or LLM fallback only after deterministic routing accuracy
   is measured.
7. Surface route choice, confidence, and fallback reason in smart-search metadata.
8. Consider making `pattern` the configured router only after MCP contract tests and
   holdout routing tests pass.

### Files

- `src/stropha/adapters/router/pattern.py`
- `src/stropha/adapters/router/escalating.py`
- `src/stropha/retrieval/query_router.py`
- `src/stropha/retrieval/smart_search.py`
- `src/stropha/server_discovery_tools.py`
- Pipeline configuration documentation
- `tests/unit/test_router_stage.py`
- Smart-search and server discovery tool tests
- `tests/eval/golden.jsonl` routing-tagged cases

### Tests

- Intent and symbol extraction for callers, tests, related, exact symbol, and trace.
- Ambiguous queries remain semantic search.
- Graph-unavailable response is explicit.
- Repository ID is preserved through every routed tool.
- Router exceptions fail safely to semantic search with metadata.
- Routing does not broaden to other repositories by default.

### Exit criteria

- At least 95% routing accuracy on deterministic routing cases.
- Structural holdout Success@K and NDCG improve or remain equal.
- No silent empty graph-gated response.
- Added routing latency is negligible for pattern-routed queries.

## 12. Phase 6: Selective Query Expansion

### Goal

Increase recall for vague conceptual queries without globally reducing precision or
adding unnecessary LLM latency.

### Experiment order

1. Query rewrite only.
2. Multi-query with count 2, then count 3.
3. HyDE for behavior descriptions with unknown identifiers.
4. Combinations only if individual features show complementary gains.

### Selection policy

Use deterministic query classification first:

- Exact identifiers and symbol-shaped queries: no expansion.
- Callers/tests/related queries: graph routing, no expansion.
- Short vague natural-language queries: query rewrite candidate.
- Broad conceptual or architecture queries: multi-query candidate.
- Behavior descriptions with no obvious code terms: HyDE candidate.

Make expansion policy explicit in configuration and diagnostics. Search-time toggles
must not alter adapter freshness or require reindexing.

### Files

- `src/stropha/adapters/retrieval/hybrid_rrf.py`
- Existing query rewrite, multi-query, and HyDE modules under `src/stropha/retrieval/`
- `src/stropha/retrieval/query_classify.py`
- `src/stropha/meta/search_space.yaml`
- `src/stropha/bench/active_matrix.yaml` or a dedicated expansion matrix
- Existing feature-specific unit tests plus benchmark tests

### Metrics

- Per-tag entity Recall@10 and NDCG@10.
- Query-generation success/fallback rate.
- Added inference calls and tokens.
- End-to-end p95 latency.
- Duplicate candidate rate before and after fusion.

### Exit criteria

- Expansion improves its target tags by at least 3 percentage points in entity
  Recall@10 or NDCG@10.
- Aggregate holdout does not regress beyond promotion gates.
- Expansion is disabled for exact and structural queries.
- Added latency has a documented upper bound.

## 13. Phase 7: Diversity-Aware Context Packing

### Goal

Return broader useful evidence under a fixed token budget instead of multiple
redundant chunks from the same file or parent symbol.

### Step 1: Measure redundancy

Add report fields for:

- Unique files returned.
- Unique symbols returned.
- Duplicate entity count.
- Maximum chunks from one file.
- Packed versus spilled relevant entities.
- Relevant entities per 1,000 returned tokens.

### Step 2: Add a bounded diversification pass

Insert diversification after final ranking/recursive merge and before token packing.
Start with a deterministic policy rather than MMR:

1. Preserve the top-ranked hit.
2. Prefer unseen `(rel_path, parent symbol)` groups.
3. Cap chunks per file for the first pass.
4. Fill remaining positions in original rank order.
5. Do not discard unique expected entities merely because they share a file.

Keep rank provenance and expose whether a hit was deferred by diversity policy.

### Step 3: Improve budget packing

`pack_to_budget` currently stops on the first overflow to preserve rank monotonicity.
Benchmark against a second-pass fill strategy that skips one oversized lower-ranked
hit and admits smaller later hits. Preserve the existing first-hit-oversize escape.

Do not change default packing until benchmark cases demonstrate better entity
coverage per token without material ranking loss.

### Step 4: Improve assembled context

For complex tasks, benchmark automatic inclusion of callers, tests, and rationale
based on routed intent. Keep section budgets explicit so graph extensions cannot
starve primary evidence.

### Files

- New small module under `src/stropha/retrieval/` only if diversification is reused
  by both search and assembly
- `src/stropha/retrieval/search.py`
- `src/stropha/retrieval/packing.py`
- `src/stropha/retrieval/assemble.py`
- Search response diagnostics models if fields are public
- `tests/unit/test_packing.py`
- `tests/unit/test_assemble.py`
- Search engine tests
- `tests/unit/test_eval_harness.py` for efficiency metrics

### Tests

- Top result remains stable.
- Distinct files are preferred over duplicate chunks at equal relevance.
- Multiple necessary symbols in one file are preserved.
- Recursive parent/child duplicates do not consume multiple diversity slots.
- Packing never exceeds budget except the documented first-hit escape.
- Second-pass fill is deterministic.
- Repository boundaries remain intact.
- Assembled section budgets preserve primary evidence.

### Exit criteria

- Duplicate entity rate decreases by at least 30%.
- Relevant entities per 1,000 tokens improves by at least 15%.
- Entity Recall@10 and NDCG@10 stay within promotion gates.
- No increase in broken or incomplete code snippets without a measured benefit.

## 14. Phase 8: Optional and Long-Term Experiments

Run these only after the preceding measurement loop is stable.

### 14.1 Model2Vec secondary lane

- Benchmark it as an independent retrieval stream.
- Verify enabling or changing it never rebuilds primary vectors.
- Measure marginal recall, latency, and memory usage.

### 14.2 Graph-vector stream

- Populate graph embeddings through the existing graph lifecycle.
- Keep graph-vector identity independent from chunk embeddings.
- Benchmark relationship-heavy queries only before broad activation.

### 14.3 Snapshot/index-cache composition

- Implement isolated index snapshots before allowing Meta-Harness indexing-stage
  candidates.
- Key cache entries by source SHA and full indexing identity.
- Use copy-on-write or independent database instances for concurrent evaluations.

### 14.4 Task-level answer evaluation

- Add a small agent-task suite that measures whether retrieved context leads to a
  correct patch or explanation.
- Keep retrieval metrics as diagnostics, not the sole optimization target.
- Include read-only comprehension tasks so edit overlap does not become the proxy
  for usefulness.

## 15. Cross-Phase Testing Strategy

Every implementation pull request must run:

```bash
uv run ruff check src tests
uv run mypy src/stropha
uv run pytest -q
git diff --check
```

Quality-affecting pull requests must additionally run:

```bash
uv run stropha eval --top-k 10 --json --require-clean-index
```

Run benchmark commands only against an index whose active and stored identities
match. Archive benchmark JSONL outside Git unless a small fixture is intentionally
added for report tests.

### Required test layers

| Layer | Purpose |
|---|---|
| Unit | Parsing, routing, metrics, identity, packing, and fallback contracts |
| Integration | Pipeline build, storage scope, clean-index preflight, graph fallback |
| Golden eval | Ranking and coverage against representative queries |
| Matrix benchmark | Query-time configuration comparison |
| Operational smoke | Shared MCP restart, `/health`, one semantic and one graph query |

## 16. Observability Requirements

The following fields should be available per eval or trace before automated tuning:

- Query category and selected route.
- Repository key and ID.
- Index and pipeline identities.
- Candidate count by retrieval stream.
- Effective stream weights and their provenance.
- RRF and reranker rank changes.
- Reranker outcome, failure class, repair result, and latency.
- Expansion type, generated query count, and fallback state.
- Recursive merge and diversity actions.
- Returned and spilled token counts.
- Unique file, symbol, and entity counts.
- Golden match details during evaluation.

Diagnostics must not log secrets or full proprietary snippets by default.

## 17. Rollout and Rollback

Use this rollout sequence for every behavior change:

1. Add diagnostics and tests without changing default behavior.
2. Run a clean baseline.
3. Enable the change only in a benchmark cell or explicit config override.
4. Validate search split.
5. Validate holdout once for finalists.
6. Repeat finalists if an LLM participates.
7. Promote through `~/.stropha/config.yaml` only after gates pass.
8. Restart the shared MCP daemon.
9. Run operational smoke queries and monitor diagnostics.

Rollback is configuration-first for query-time features. Keep the previous winning
configuration recorded so reranker, expansion, routing, recursive retrieval, or
diversity can be disabled without reindexing. Indexing identity changes require a
compatible reindex and must never be hidden by editing metadata.

## 18. Implementation Checklist

### Phase 0

- [x] Remove duplicate batch ownership from effective enricher identity.
- [x] Add identity regression tests.
- [x] Let the post-commit hook refresh the index.
- [x] Verify index health and current `HEAD`.
- [x] Archive strict baseline JSON.

### Phase 1

- [x] Add reranker failure-class diagnostics.
- [x] Add task-specific structured generation controls.
- [x] Add one bounded repair attempt.
- [x] Benchmark deterministic fallback policies.
- [x] Reach less than 5% fallback rate in two runs.

### Phase 2

- [x] Add complete metrics and provenance to benchmark records.
- [x] Add comparability checks.
- [x] Run matrix smoke test.
- [x] Run staged axis experiments.
- [x] Produce a repeated shortlist; defer promotion because the holdout was exposed.

### Phase 3

- [x] Define category quotas and authoring guide.
- [x] Freeze a replacement unseen promotion cohort with answer catalogs excluded.
- [x] Add reproducible batch gates, verification, and explicit promotion tooling.
- [x] Expand to at least 120 reviewed cases.
- [x] Make at least half multi-entity or graded.
- [x] Validate search/holdout category coverage.
- [x] Archive expanded-golden baseline.

### Phase 4

- [x] Map feedback reasons to learning channels.
- [x] Deduplicate explicit and post-commit labels.
- [x] Add profile evidence and reporting.
- [x] Enforce minimum observations and regularization.
- [x] Validate learned weights on holdout before activation.

### Phase 5

- [ ] Build routing benchmark.
- [ ] Enable deterministic structural routing in experiments.
- [ ] Preserve explicit graph-unavailable responses.
- [ ] Surface route diagnostics.
- [ ] Reach at least 95% deterministic routing accuracy.

### Phase 6

- [ ] Benchmark rewrite, multi-query, and HyDE separately.
- [ ] Add query-category activation policy.
- [ ] Record generation diagnostics and costs.
- [ ] Validate target-tag improvements on holdout.

### Phase 7

- [ ] Add redundancy and efficiency metrics.
- [ ] Implement deterministic diversity pass behind config.
- [ ] Benchmark second-pass budget fill.
- [ ] Add section budgets to assembled context if beneficial.
- [ ] Meet duplicate-rate and entities-per-token gates.

### Phase 8

- [ ] Benchmark Model2Vec independently.
- [ ] Evaluate graph-vector retrieval when embeddings are available.
- [ ] Add snapshot/index-cache isolation before indexing-stage search.
- [ ] Add task-level answer evaluation.

## 19. First Execution Slice

The first implementation session should remain deliberately small:

1. Fix duplicate effective enricher identity ownership.
2. Add direct identity tests.
3. Run the required unit and full test suites.
4. Commit and allow the hook to refresh the index.
5. Verify `index_health` reports no drift and zero commits behind.
6. Run strict eval and archive the report.
7. Stop and review the baseline before changing reranking.

This slice establishes trustworthy evidence. No retrieval knob should be changed in
the same commit.
