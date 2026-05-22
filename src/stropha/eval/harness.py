"""Golden-dataset evaluation runner.

Schema (JSONL, one case per line):

    {
      "id": "stropha-001",
      "query": "where is the FSRS calculator",
      "expected_paths": ["src/stropha/fsrs.py"],
      "expected_symbols": ["FsrsCalculator"],
      "relevance": [3, 1, 0],  # optional: graded relevance for NDCG
      "tags": ["symbol-lookup"]
    }

A hit counts as relevant when any of:
  - ``hit.path`` matches one of ``expected_paths`` (exact or suffix)
  - ``hit.symbol`` matches one of ``expected_symbols`` (case-insensitive)

Metrics computed:
  - Recall@K = (cases with ≥1 relevant hit in top-K) / total
  - MRR = mean(1 / rank of first relevant hit, 0 when none in top-K)
  - NDCG@K = Normalized Discounted Cumulative Gain (for graded relevance)
  - Precision@K = (relevant hits in top-K) / K

The runner does NOT depend on any LLM and never re-embeds. It uses
whatever ``SearchEngine`` the caller built (so swapping an adapter and
re-running is a one-line change).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..logging import get_logger
from ..retrieval.search import SearchEngine

log = get_logger(__name__)


# --------------------------------------------------------------------- NDCG


def _dcg(relevances: list[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain.
    
    DCG = Σ (2^rel_i - 1) / log2(i + 1) for i in 1..k
    
    Args:
        relevances: List of relevance scores (higher = more relevant).
        k: Cutoff rank (None = use all).
        
    Returns:
        DCG score.
    """
    if k is not None:
        relevances = relevances[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        # Using the more common formula: (2^rel - 1) / log2(i + 1)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def _ndcg(relevances: list[float], k: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain.
    
    NDCG = DCG / IDCG where IDCG is DCG with ideal ranking.
    
    Args:
        relevances: List of relevance scores at each rank position.
        k: Cutoff rank (None = use all).
        
    Returns:
        NDCG score in [0, 1].
    """
    dcg = _dcg(relevances, k)
    if dcg == 0:
        return 0.0
    
    # Ideal DCG: sort relevances in descending order
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


@dataclass(frozen=True)
class EvalCase:
    """One test case loaded from the golden file."""

    id: str
    query: str
    expected_paths: tuple[str, ...] = ()
    expected_symbols: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    # Graded relevance for NDCG (optional). If provided, should be a dict
    # mapping path/symbol to relevance score (0-3). If not provided, binary
    # relevance (1 for match, 0 for non-match) is used.
    relevance_map: dict[str, float] = field(default_factory=dict)
    # WIN-168 anti-overfit split. ``"search"`` is the subset the
    # Meta-Harness proposer can optimise against. ``"holdout"`` is reserved
    # for generalisation measurement and must NOT be visible to the
    # proposer. Legacy cases without the field default to ``"search"`` so
    # existing eval runs keep working unchanged.
    split: Literal["search", "holdout"] = "search"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalCase:
        # Build relevance map from various formats
        relevance_map: dict[str, float] = {}
        if "relevance" in d:
            rel = d["relevance"]
            if isinstance(rel, dict):
                # Explicit mapping: {"src/foo.py": 3, "FooClass": 2}
                relevance_map = {str(k): float(v) for k, v in rel.items()}
            elif isinstance(rel, list):
                # List format: assign to expected_paths/symbols in order
                expected = list(d.get("expected_paths") or []) + list(d.get("expected_symbols") or [])
                for i, r in enumerate(rel):
                    if i < len(expected):
                        relevance_map[expected[i]] = float(r)

        split_raw = str(d.get("split", "search")).lower()
        if split_raw not in ("search", "holdout"):
            raise ValueError(
                f"invalid split {split_raw!r} for case {d.get('id')!r}; "
                f"expected 'search' or 'holdout'"
            )

        return cls(
            id=str(d.get("id") or d.get("query", "")[:40]),
            query=str(d["query"]),
            expected_paths=tuple(d.get("expected_paths") or ()),
            expected_symbols=tuple(d.get("expected_symbols") or ()),
            tags=tuple(d.get("tags") or ()),
            relevance_map=relevance_map,
            split=split_raw,  # type: ignore[arg-type]  # validated above
        )
    
    def get_relevance(self, path: str | None, symbol: str | None) -> float:
        """Get relevance score for a path/symbol match.
        
        Returns 1.0 (binary relevant) if no graded relevance is defined.
        """
        if not self.relevance_map:
            return 1.0  # Binary relevance
        
        # Check path
        if path:
            for key, score in self.relevance_map.items():
                if path == key or path.endswith("/" + key) or path.endswith(key):
                    return score
        
        # Check symbol
        if symbol:
            symbol_lower = symbol.lower()
            for key, score in self.relevance_map.items():
                key_lower = key.lower()
                if symbol_lower == key_lower or symbol_lower.endswith("." + key_lower):
                    return score
        
        return 1.0  # Default to binary relevance


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running one :class:`EvalCase`."""

    case: EvalCase
    rank_of_first_hit: int | None  # 1-based, None if not found in top-K
    top_k: int
    matched_path: str | None = None
    matched_symbol: str | None = None
    # Relevance scores at each rank position (for NDCG)
    relevances: tuple[float, ...] = ()
    # Number of relevant hits in top-K (for Precision@K)
    relevant_count: int = 0
    # WIN-170: per-query latency capture. ``latency_ms`` is the wall-clock
    # cost of the single retrieval call this result describes. Captured by
    # ``run_eval`` so latency is first-class fitness, not an invisible-in-CI
    # afterthought.
    latency_ms: float = 0.0

    @property
    def hit(self) -> bool:
        return self.rank_of_first_hit is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank_of_first_hit if self.rank_of_first_hit else 0.0

    @property
    def ndcg(self) -> float:
        """NDCG for this single case."""
        if not self.relevances:
            return 0.0
        return _ndcg(list(self.relevances), self.top_k)

    @property
    def precision(self) -> float:
        """Precision@K for this case."""
        return self.relevant_count / self.top_k if self.top_k > 0 else 0.0

    @property
    def average_precision(self) -> float:
        """Average Precision (AP) for this single case.

        AP = mean of precision-at-each-relevant-rank within the top-K.
        Returns 0.0 when no relevant hit is in the top-K. Used by
        :attr:`EvalReport.map_at_k` (mean AP = MAP).
        """
        if not self.relevances:
            return 0.0
        n_relevant = sum(1 for r in self.relevances if r > 0)
        if n_relevant == 0:
            return 0.0
        ap_sum = 0.0
        running_relevant = 0
        for i, rel in enumerate(self.relevances, start=1):
            if rel > 0:
                running_relevant += 1
                ap_sum += running_relevant / i
        return ap_sum / n_relevant


@dataclass
class EvalReport:
    """Aggregate metrics across an evaluation run."""

    results: list[EvalResult] = field(default_factory=list)
    top_k: int = 10

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def recall_at_k(self) -> float:
        return self.n_hits / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / self.n
    
    @property
    def ndcg_at_k(self) -> float:
        """Mean NDCG@K across all cases."""
        if not self.results:
            return 0.0
        return sum(r.ndcg for r in self.results) / self.n
    
    @property
    def precision_at_k(self) -> float:
        """Mean Precision@K across all cases."""
        if not self.results:
            return 0.0
        return sum(r.precision for r in self.results) / self.n

    @property
    def map_at_k(self) -> float:
        """Mean Average Precision @K — the first-class quality metric
        the WIN-169 fitness function consumes for retrieval quality.

        Robust to single-rank reranking jitter in a way recall@K is not:
        moving a relevant hit from rank 4 to rank 1 doesn't change
        recall but lifts MAP, which is exactly what the proposer should
        optimise toward.
        """
        if not self.results:
            return 0.0
        return sum(r.average_precision for r in self.results) / self.n

    @property
    def latency_p50_ms(self) -> float:
        """Median per-query latency across all cases (ms)."""
        return self._latency_percentile(50)

    @property
    def latency_p95_ms(self) -> float:
        """p95 per-query latency (ms). The fitness function (WIN-169)
        consumes this directly as the ``latency_ms`` dimension."""
        return self._latency_percentile(95)

    @property
    def latency_p99_ms(self) -> float:
        return self._latency_percentile(99)

    def _latency_percentile(self, pct: int) -> float:
        if not self.results:
            return 0.0
        samples = sorted(r.latency_ms for r in self.results)
        idx = max(0, min(len(samples) - 1, int(len(samples) * pct / 100)))
        return samples[idx]

    def by_tag(self) -> dict[str, EvalReport]:
        """Slice results per tag for diagnostic breakdowns."""
        buckets: dict[str, EvalReport] = {}
        for r in self.results:
            for t in r.case.tags or ("(untagged)",):
                buckets.setdefault(t, EvalReport(top_k=self.top_k)).results.append(r)
        return buckets

    def by_split(self) -> dict[str, EvalReport]:
        """WIN-168 — slice results per split (search vs holdout) so the
        evaluator can report both numbers without having to re-run."""
        buckets: dict[str, EvalReport] = {}
        for r in self.results:
            key = r.case.split
            buckets.setdefault(key, EvalReport(top_k=self.top_k)).results.append(r)
        return buckets

    def summary(self) -> dict[str, Any]:
        return {
            "cases": self.n,
            "hits": self.n_hits,
            f"recall@{self.top_k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.top_k}": round(self.ndcg_at_k, 4),
            f"precision@{self.top_k}": round(self.precision_at_k, 4),
            f"map@{self.top_k}": round(self.map_at_k, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
        }


# --------------------------------------------------------------------- io


def load_golden(path: Path) -> list[EvalCase]:
    """Parse a JSONL file into :class:`EvalCase` instances.

    Empty / comment lines (``#``-prefixed) are skipped. Invalid lines raise
    a ``ValueError`` indicating the line number — fast feedback when
    hand-editing the golden set.
    """
    cases: list[EvalCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict) or "query" not in payload:
            raise ValueError(f"{path}:{lineno}: missing required `query` key")
        try:
            cases.append(EvalCase.from_dict(payload))
        except ValueError as exc:
            # Re-raise with line-number context so misformed split values
            # (or any future field validation) surface where the user can
            # find them.
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return cases


def filter_by_split(
    cases: Iterable[EvalCase],
    split: Literal["search", "holdout"],
) -> list[EvalCase]:
    """Return only the cases tagged with ``split``.

    The Meta-Harness proposer should only ever see the ``"search"`` subset
    — anything else is overfit waiting to happen. ``"holdout"`` is the
    locked anti-overfit measure consumed by the evaluator at promotion
    time (Phase 4).
    """
    return [c for c in cases if c.split == split]


def split_counts(cases: Iterable[EvalCase]) -> dict[str, int]:
    """How many cases fall into each split. Sanity check for golden hygiene."""
    counts = {"search": 0, "holdout": 0}
    for c in cases:
        counts[c.split] = counts.get(c.split, 0) + 1
    return counts


# --------------------------------------------------------------------- runner


def _path_matches(actual: str, expected: Iterable[str]) -> str | None:
    """``actual`` matches when it equals or ends with one of ``expected``."""
    a = actual.strip()
    for e in expected:
        if not e:
            continue
        e = e.strip()
        if a == e or a.endswith("/" + e) or a.endswith(e):
            return e
    return None


def _symbol_matches(actual: str | None, expected: Iterable[str]) -> str | None:
    if not actual:
        return None
    a = actual.lower().strip()
    for e in expected:
        if not e:
            continue
        if a == e.lower().strip() or a.endswith("." + e.lower().strip()):
            return e
    return None


def run_eval(
    cases: Iterable[EvalCase],
    engine: SearchEngine,
    *,
    top_k: int = 10,
) -> EvalReport:
    """Execute every case against ``engine`` and aggregate metrics."""
    import time as _time

    report = EvalReport(top_k=top_k)
    for case in cases:
        # WIN-170: capture per-query latency. perf_counter() is monotonic
        # and high-resolution; the cost (~50 ns per call) is irrelevant
        # next to a multi-millisecond search.
        start = _time.perf_counter()
        hits = engine.search(case.query, top_k=top_k)
        latency_ms = (_time.perf_counter() - start) * 1000.0

        # Track relevance at each rank for NDCG
        relevances: list[float] = []
        relevant_count = 0
        first_hit_rank: int | None = None
        first_matched_path: str | None = None
        first_matched_symbol: str | None = None

        for rank, h in enumerate(hits, 1):
            # Tolerate both real SearchHit (.rel_path) and tests' fake hits (.path).
            actual_path = getattr(h, "rel_path", None) or getattr(h, "path", "")
            actual_symbol = getattr(h, "symbol", None)

            mp = _path_matches(actual_path, case.expected_paths)
            ms = _symbol_matches(actual_symbol, case.expected_symbols)

            if mp or ms:
                # This hit is relevant
                rel = case.get_relevance(actual_path, actual_symbol)
                relevances.append(rel)
                relevant_count += 1

                if first_hit_rank is None:
                    first_hit_rank = rank
                    first_matched_path = mp
                    first_matched_symbol = ms
            else:
                # Not relevant
                relevances.append(0.0)

        # Pad with zeros if fewer hits than top_k
        while len(relevances) < top_k:
            relevances.append(0.0)

        result = EvalResult(
            case=case,
            rank_of_first_hit=first_hit_rank,
            top_k=top_k,
            matched_path=first_matched_path,
            matched_symbol=first_matched_symbol,
            relevances=tuple(relevances),
            relevant_count=relevant_count,
            latency_ms=latency_ms,
        )
        report.results.append(result)
        log.info(
            "eval.case.done",
            case=case.id,
            hit=result.hit,
            rank=first_hit_rank,
            ndcg=round(result.ndcg, 4),
            precision=round(result.precision, 4),
            latency_ms=round(latency_ms, 3),
            tags=list(case.tags),
        )
    log.info("eval.run.summary", **report.summary())
    return report
