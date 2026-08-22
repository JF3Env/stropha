# Golden Dataset Authoring

Golden data is generated through checked-in batch specifications. Do not hand-edit
materialized files under `.stropha/bench/batches/`.

The production golden contains 124 source-reviewed cases: 87 search and 37
holdout, with 72 graded or multi-entity cases and every major category represented
in both splits. SHA-256:
`bde07d1e9525a34f69c1c7fc91b14ac440761c6a404cc8c70f16398d321f6832`.
`tests/eval/golden.jsonl` is an answer catalog and must remain excluded from
indexing.

`tests/eval/routing.jsonl` is the frozen Phase 5 structural-routing holdout. Run
it once, after its defining commit has been indexed, with:

```bash
uv run stropha bench routing
```

The command compares semantic-only search with deterministic `PatternRouter`
dispatch over the same repository-scoped runtime. It fails closed on routing
accuracy below 95%, graph fallback, Success@K or NDCG regression, lost baseline
hits, router p95 above 0.1 ms, source changes, index commit drift, or adapter
identity drift. The one-time clean run at commit `245b3c3` passed and is archived
in `tests/eval/routing-baseline-997e15e9.json`. The catalog is now exposed and
must be replaced rather than tuned or rerun.

`tests/eval/baseline-c61c5fef.json` remains the comparable historical baseline
for the superseded `c61c5fef...` catalog. The current hash corrects the working-set
case after removal of the request-context-free resource and needs a new baseline
after this checkout is indexed. Do not relabel the historical metrics. That
checked-in all-case report is distinct from the Meta-Harness promotion baseline,
which records same-run search and holdout evidence under
`<meta>/runs/<run-id>/validation.json`.

## Workflow

`phase3-unseen-holdout-v1` is retained as an invalidated historical specification
because its answer catalog was indexable during its run. Do not rerun it.

`phase3-unseen-holdout-v2` is also invalidated and was never executed: pre-commit
review found that expected paths and symbols were not positionally paired.

`phase3-unseen-holdout-v3` is invalidated and was never executed: review found
that graded IDCG counted paired path and symbol aliases twice.

`phase3-unseen-holdout-v4` is the corrected replacement. Its 30 cases were
authored from source review, have no ID or normalized-query overlap with the
54-case golden or invalidated v1 catalog, and are excluded from indexing. It was
executed once after commit `7b71c6a` and is now exposed; do not rerun or tune
against it. Every future unseen cohort must include the v4 catalog in
`exposed_catalogs`.

Frozen v4 cohort SHA-256:
`2100642667cce8be364c33c8e1d360ad16a4dbe115d043530a20482ca507ee58`.

V4 did not promote `rrf_k=40`. NDCG@10 improved from `0.8419` to `0.8451`
(`+0.0032`) against the required `+0.02`; all other gates and verification
steps passed. Production remains at `rrf_k=60`.

The next unseen protocol freezes the three cells in
`src/stropha/bench/phase4_quality_candidates_matrix.yaml`: the deployed
contextual baseline, hierarchical indexing, and contextual retrieval with
`rrf_k=40`. Its future catalog must contain about 200 newly source-reviewed
cases, list both v1 and v4 under `exposed_catalogs`, and be prepared before any
retrieval run. Do not execute the matrix against an incomplete or unreviewed
catalog.

Frozen Phase 4 V1 cohort SHA-256:
`7676003c755d980d765ef69b24b528b365af85626bb0ccf20d2bc81a569c35c9`.

```bash
uv run stropha bench batch \
  path/to/new-unseen-batch.yaml \
  --prepare-only

uv run stropha bench batch \
  path/to/new-unseen-batch.yaml
```

The first command validates and freezes the catalog without executing retrieval.
The execution command runs a frozen baseline/candidate matrix, applies promotion
gates, runs the named verification profile, and writes `decision.json`.

Reusing a batch name with changed cohort data is rejected. Create a new batch name
for every new unseen holdout. Ordinary execution never changes user configuration.
`--promote` is explicit and succeeds only when every gate and verification step
passes.

For Meta-Harness optimization, the search split may be used repeatedly, but the
holdout split is opened only during explicit validation of a frozen shortlist:

```text
stropha meta run ...
stropha meta validate <run-id>
stropha meta promote <candidate-id> --run-id <run-id> --apply
```

The live baseline and every shortlisted candidate must use the same snapshot,
golden hash, index identity, repository scope, and top-K. Holdout failures are
blocking, never neutral. Validation evidence is excluded from proposer history.
`meta run` evaluates only `search`; it cannot produce or overwrite
`validation.json`. Promotion is diff-only unless `--apply` is supplied, and the
apply path rechecks the frozen validation and current runtime identity.

## Case Contract

- Use a stable unique ID and a natural query authored from source review.
- Provide existing expected paths and symbols present in those paths.
- Every relevance key must name an expected path or symbol; scores are 1-3.
- Add exactly one category tag declared by the batch implementation.
- Holdout catalogs must not reuse IDs or normalized queries from exposed goldens.
- Do not tune against a holdout report or rerun an exposed unseen cohort.
- Rotate holdout cases by creating a new frozen batch; never mutate an executed
  catalog in place.
- Prefer multi-entity graded judgments for workflow and conceptual queries.
