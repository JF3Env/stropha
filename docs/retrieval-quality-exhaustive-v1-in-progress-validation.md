# Retrieval Quality Exhaustive V1: Interrupted Validation

Status: **INTERRUPTED - NOT APPROVED FOR PROMOTION**

This document records provisional evidence from the intentionally stopped
`retrieval-quality-exhaustive-v1` benchmark campaign. Its artifacts remain
preserved, but the incomplete findings cannot change production defaults.

## Validation Snapshot

Analysis snapshot time: 2026-08-17 04:49 UTC
Stop-state confirmation: 2026-08-17 07:37 UTC

- Campaign: `retrieval-quality-exhaustive-v1`
- Source commit: `81e60a2af43c86a53f85d6821e25d334c1b0debf`
- Plan SHA-256: `1aab31ba305b0245e94eca52b2864446689dd414fb4fef61182acb3cc2a9bc96`
- Configuration SHA-256: `8292cd8693cba05f4626b0059d7a8dafe4e51feaaf27ab337e6a1247d607e406`
- Campaign state: `interrupted` (`running: false`)
- Completed cells at stop: 1,508 / 6,216 (24.3%)
- Approximate completed queries at stop: 131,196 / 540,792
- Recorded cell failures: 0
- Artifact root:
  `.stropha/bench/campaigns/retrieval-quality-exhaustive-v1/81e60a2af43c-8292cd8693cb-1aab31ba305b/`

The metrics below were calculated from the earlier 1,468-cell analysis snapshot
and were not recomputed after the final 40 cells. That analyzed slice contains
only `candidate_k=20` and `adaptive_weighting=false`, including all 1,152 planned
`hyde=false` cells in that slice and 316 matched `hyde=true` cells. Candidate
size 50, adaptive weighting, and most HyDE combinations remain untested. The
declared campaign baseline has therefore not completed.

## Analysis Method

The findings below use matched comparisons. Each treatment cell is compared
with a completed cell that differs only in the axis being measured. Raw means
across all completed cells are not used to infer feature effects because matrix
constraints produce unequal groups and the campaign execution order is not
random.

These are deterministic measurements over the current 87-case golden set, not
independent repetitions. Win rates describe the fraction of matched
configuration pairs with a positive NDCG change; they are not confidence
intervals or generalization guarantees.

## Provisional Retrieval Findings

### Stream composition

Dense retrieval is the essential lane in the completed slice.

| Treatment relative to dense + sparse + symbol | Matched pairs | NDCG delta | p95 latency delta | NDCG win rate |
|---|---:|---:|---:|---:|
| Disable symbol | 287 | -0.0055 | -78 ms | 39.7% |
| Disable sparse | 287 | -0.0307 | -190 ms | 10.5% |
| Disable dense | 192 | -0.1455 | -366 ms | 0.0% |
| Dense only | 159 | +0.0193 | -344 ms | 52.2% |
| Sparse only | 128 | -0.5101 | -542 ms | 0.0% |
| Symbol only | 128 | -0.2040 | -745 ms | 0.0% |

Dense-only currently outperforms all-stream fusion while also reducing tail
latency. This does not establish that sparse and symbol retrieval are useless.
The interaction suggests that equal-weight fusion may introduce noise and that
adaptive weighting remains an important pending test.

### Query-time features

| Feature enabled | Matched pairs | NDCG delta | p95 latency delta | NDCG win rate |
|---|---:|---:|---:|---:|
| Recursive merging | 732 | +0.0117 | +93 ms | 95.4% |
| Multi-query | 480 | +0.0053 | +1,537 ms | 76.7% |
| Query rewrite | 732 | -0.0037 | +884 ms | 36.9% |
| HyDE | 316 | +0.0004 | +1,373 ms | 50.3% |

Recursive merging is the only completed query-time feature with a clear
quality/cost case. It also removes approximately 7,065 returned tokens per
cell and reduces duplicate-entity rate by 19.4 percentage points on average.

HyDE has no measurable quality benefit in the completed matched slice.
Query rewriting is a small quality regression. Multi-query is directionally
positive but its current tail-latency cost is disproportionate to its quality
gain.

RRF values 40, 60, and 80 remain practically equivalent. Their matched NDCG
differences are below 0.003.

### Rerankers

| Reranker relative to none | Matched pairs | NDCG delta | p95 latency delta | NDCG win rate |
|---|---:|---:|---:|---:|
| LLM | 180 | +0.1451 | +3,841 ms | 100.0% |
| Jina v2 base multilingual | 184 | +0.0633 | +1,591 ms | 89.1% |
| BGE reranker base | 184 | +0.0276 | +1,395 ms | 73.9% |
| MiniLM L12 | 184 | +0.0006 | +394 ms | 52.2% |
| MiniLM L6 | 184 | -0.0047 | +201 ms | 51.6% |
| Code-aware | 184 | -0.0102 | -9 ms | 27.7% |
| Jina v1 turbo English | 184 | -0.0500 | +226 ms | 23.9% |

The LLM reranker is the strongest ranking intervention in the completed slice.
Jina v2 is the strongest completed non-LLM alternative. Code-aware and Jina v1
should not be promoted based on current evidence.

## Current Quality/Latency Frontier

All three current leaders use dense-only retrieval, recursive merging, no query
rewrite, `candidate_k=20`, and `adaptive_weighting=false`.

| Profile | NDCG@10 | MRR | Success@10 | Recall@10 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Fast: multi-query, no reranker | 0.7308 | 0.7509 | 0.9655 | 0.8774 | 147 ms | 685 ms |
| Balanced: LLM, no expansion | 0.8085 | 0.8287 | 0.9540 | 0.8908 | 1,632 ms | 2,045 ms |
| Maximum observed: multi-query + LLM | 0.8142 | 0.8339 | 0.9540 | 0.8908 | 1,523 ms | 7,790 ms |

The balanced profile loses only 0.0057 NDCG relative to the maximum observed
profile while reducing p95 latency by approximately 74%. The large quality
change is from LLM reranking; multi-query adds little quality on top of it.

## Multi-Query Serialization Finding

The campaign source implementation performed avoidable serial work:

1. `HybridRrfRetrieval._search_multi_query()` looped over paraphrases
   sequentially.
2. Every paraphrase called `_search_single_query()`, which applied the configured
   reranker and recursive merge.
3. The fused result was reranked and recursively merged again.
4. Stream searches inside each single-query execution were also sequential.
5. When HyDE and multi-query were both enabled, HyDE generation and dense
   embedding occurred inside each paraphrase execution.

For the maximum-NDCG cell, 10 queries activated multi-query and produced 40
paraphrase searches. The cell made 127 LLM reranker calls across 87 evaluation
queries, compared with 87 calls in the balanced non-multi-query cell. All 127
primary reranker calls succeeded, with no repairs or fallbacks. The additional
latency is therefore not explained by malformed output or retry behavior.

The follow-up implementation now retrieves candidates without per-paraphrase
reranking or recursive merging, fuses those candidates, and applies both stages
once. PostgreSQL uses a bounded independent read pool, and the coordinator runs
safe stream or paraphrase reads concurrently while preserving configured order,
repository scope, deterministic exceptions, and one final post-fusion rerank.
SQLite, graph-vector, and Model2Vec paths remain serial where safety is not yet
declared.

A non-destructive production-index probe on 2026-08-17 checked out two distinct
PostgreSQL backend sessions from a two-connection pool. A three-stream hybrid
query then opened three pooled sessions, returned 10 hits, reported one bounded
parallel stream fan-out, and left every pool connection available afterward.
This validates pool wiring and lifecycle, not retrieval-quality promotion.

## LMS and OpenCode Configuration Finding

The benchmark and OpenCode use the same configured LMS model and endpoint, but
they do not use the same generation profile.

| Setting | OpenCode LMS model profile | Benchmark LLM reranker |
|---|---:|---:|
| Temperature | 1.0 | 0.0 |
| Top-p | 0.95 | Not passed |
| Top-k sampling | 40 | Not passed |
| Presence penalty | 1.0 | Not passed |
| Thinking | Not explicitly disabled | Disabled |
| Declared/per-call output limit | 32,768 model ceiling | 2,048 per request |
| Request timeout | 1,800 seconds | 300 seconds |
| Response mode | General generation | Strict JSON schema |

At the analysis snapshot, `LmsBackend.generate()` accepted and sent
`temperature`, `max_tokens`, `think`, timeout, and optional JSON schema, but it
could not express top-p, top-k sampling, or presence penalty and used
synchronous `urllib` requests. The follow-up implementation adds those optional
sampling controls and a reusable bounded `httpx` connection pool. Defaults stay
unchanged, and reranker sampling policy remains part of its adapter identity.

The OpenCode profile is known-good for agentic code generation, but it is not
automatically optimal for deterministic listwise ranking. The benchmark's
strict JSON mode produced valid output for all calls in the current leaders,
so sampling changes require direct measurement rather than assumption.

## Required Follow-Up Validation

- [x] Stop the suboptimal campaign and preserve its frozen artifacts.
- [ ] Recompute matched effects after `candidate_k=50`, adaptive weighting, and
  all remaining HyDE cells complete.
- [ ] Confirm the declared baseline and final ranking report are generated.
- [x] Implement a candidate-retrieval path that skips per-paraphrase reranking.
- [x] Batch or concurrently execute paraphrase retrieval with explicit bounded
  concurrency and verified adapter/storage safety.
- [x] Ensure multi-query performs one final reranker invocation per user query.
- [x] Avoid repeated identical HyDE generation within one multi-query request.
- [x] Add persistent bounded LMS HTTP connection pooling.
- [ ] Measure connection, generation, fusion, and reranking latency separately.
- [x] Add explicitly configured LMS sampling controls for top-p, top-k, and
  presence penalty where the endpoint supports them.
- [x] Include every output-affecting sampling control in backend/reranker
  identity so configuration drift invalidates stale artifacts.
- [ ] Compare the current deterministic reranker profile with the exact
  OpenCode sampling profile.
- [ ] Compare an OpenCode-derived profile that retains strict JSON and disables
  thinking for ranking.
- [ ] Repeat stochastic profiles enough times to estimate variance.
- [ ] Re-run the fast, balanced, and maximum-quality candidates on an unseen
  holdout before promotion.
- [ ] Verify that quality gains persist by query category and are not dominated
  by a small subset of the 87 golden cases.

## Current Decision

No production configuration is approved by this in-progress validation.

The leading hypothesis is that dense-only retrieval with recursive merging and
a single LLM rerank offers the best quality-focused tradeoff. The leading
latency optimization is to remove per-paraphrase reranking before attempting
more aggressive concurrency. HyDE and query rewrite should remain disabled
unless later campaign cells overturn the current matched evidence.
