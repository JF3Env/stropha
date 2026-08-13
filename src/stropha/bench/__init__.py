"""Structured configuration-matrix benchmark for stropha retrieval.

Unlike the Meta-Harness (``stropha meta``), which *searches* for a good
config via a bandit/agentic proposer, this package *enumerates* a declared
matrix of configurations and scores every cell against the golden set, so
you can compare many configurations head-to-head with full data.

Two properties make a days-long sweep tractable and trustworthy:

- **Index reuse.** Axes are split into *indexing* (embedder / enricher /
  chunker — change the stored vectors) and *query-time* (reranker / HyDE /
  recursive / candidate_k — reuse the same index). The runner builds each
  unique index once and sweeps every query-time variant against it.
- **Resumable + guarded.** Results append to a JSONL store keyed by a stable
  cell id, so an interrupted multi-day run resumes where it left off. Every
  index is verified complete (vecs == chunks) and, for non-noop enrichers,
  actually enriched before any cell is scored.
"""

from __future__ import annotations

from .batch import BatchError, prepare_batch, run_batch
from .matrix import Cell, Matrix, MatrixError, load_matrix
from .report import BenchReport, load_results
from .routing import (
    RoutingBenchmark,
    RoutingBenchmarkError,
    RoutingCase,
    benchmark_routing,
    load_routing_cases,
)

__all__ = [
    "BatchError",
    "BenchReport",
    "Cell",
    "Matrix",
    "MatrixError",
    "RoutingBenchmark",
    "RoutingBenchmarkError",
    "RoutingCase",
    "benchmark_routing",
    "load_matrix",
    "load_results",
    "load_routing_cases",
    "prepare_batch",
    "run_batch",
]
