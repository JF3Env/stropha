# Golden Dataset Authoring

Golden data is generated through checked-in batch specifications. Do not hand-edit
materialized files under `.stropha/bench/batches/`.

## Workflow

`phase3-unseen-holdout-v1` is retained as an invalidated historical specification
because its answer catalog was indexable during its run. Do not rerun it. Create a
new batch name and entirely new unseen cases for the next evaluation.

```bash
uv run stropha bench batch \
  path/to/new-unseen-batch.yaml \
  --prepare-only

uv run stropha bench batch \
  path/to/new-unseen-batch.yaml
```

The first command validates and freezes the catalog without executing retrieval.
The second command runs the frozen baseline/candidate matrix, applies promotion
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
