"""Evaluation harness — golden dataset + ranked retrieval metrics.

The harness runs deterministic offline metrics against a JSONL golden
dataset. Success@K records whether any expected entity appears, while
Recall@K records the fraction of distinct judged entities retrieved.

Why: lets us guard regressions when swapping adapters (drift detection,
new enricher, new retrieval stream) without paying for LLM-judge runs.
RAGAS-style answer-level evaluation (faithfulness, answer relevance) is
intentionally out of scope here — see §16.4 for the optional integration.
"""

from .harness import (
    EvalCase,
    EvalReport,
    EvalResult,
    filter_by_split,
    load_golden,
    run_eval,
    split_counts,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "filter_by_split",
    "load_golden",
    "run_eval",
    "split_counts",
]
