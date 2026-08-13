# Golden Dataset Authoring

Golden data is generated through checked-in batch specifications. Do not hand-edit
materialized files under `.stropha/bench/batches/`.

The production golden contains 124 source-reviewed cases: 88 search and 36
holdout, with 72 graded or multi-entity cases and every major category represented
in both splits. SHA-256:
`c61c5fef05bd3f2f0f67e3da425d3bd46d79c2bca9b779aa135187c4c5e01b1b`.
`tests/eval/golden.jsonl` is an answer catalog and must remain excluded from
indexing.

The comparable production baseline for this hash is archived in
`tests/eval/baseline-c61c5fef.json` with exact pipeline and index provenance.

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

## Case Contract

- Use a stable unique ID and a natural query authored from source review.
- Provide existing expected paths and symbols present in those paths.
- Every relevance key must name an expected path or symbol; scores are 1-3.
- Add exactly one category tag declared by the batch implementation.
- Holdout catalogs must not reuse IDs or normalized queries from exposed goldens.
- Prefer multi-entity graded judgments for workflow and conceptual queries.
