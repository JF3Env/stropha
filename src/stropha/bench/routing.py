"""Deterministic structural-routing holdout benchmark."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from ..adapters.router.noop import NoopRouter
from ..adapters.router.pattern import PatternRouter
from ..errors import StrophaError
from ..eval import EvalCase, EvalReport, run_eval
from ..retrieval.query_router import QueryIntent
from ..retrieval.smart_search import SmartSearchResult, smart_search


class RoutingBenchmarkError(StrophaError):
    """The routing benchmark input or runtime is not eligible."""


@dataclass(frozen=True)
class RoutingCase:
    """One retrieval judgment plus its expected routing contract."""

    eval_case: EvalCase
    expected_intent: QueryIntent
    expected_route_symbol: str | None
    expected_tool: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RoutingCase:
        try:
            expected_intent = QueryIntent(str(raw["expected_intent"]))
            expected_tool = str(raw["expected_tool"])
        except (KeyError, ValueError) as exc:
            raise RoutingBenchmarkError(
                f"routing case {raw.get('id')!r} has an invalid routing contract"
            ) from exc
        case = EvalCase.from_dict(raw)
        if case.split != "holdout":
            raise RoutingBenchmarkError(f"routing case {case.id!r} must use split='holdout'")
        if "structural-routing" not in case.tags:
            raise RoutingBenchmarkError(
                f"routing case {case.id!r} must have the 'structural-routing' tag"
            )
        symbol = raw.get("expected_route_symbol")
        return cls(
            eval_case=case,
            expected_intent=expected_intent,
            expected_route_symbol=str(symbol) if symbol is not None else None,
            expected_tool=expected_tool,
        )


@dataclass(frozen=True)
class RoutingObservation:
    """Actual deterministic route selected for one holdout query."""

    case_id: str
    expected_intent: str
    actual_intent: str
    expected_symbol: str | None
    actual_symbol: str | None
    expected_tool: str
    actual_tool: str
    confidence: float
    reason: str

    @property
    def correct(self) -> bool:
        return (
            self.actual_intent == self.expected_intent
            and self.actual_symbol == self.expected_symbol
            and self.actual_tool == self.expected_tool
        )

    @property
    def graph_fallback(self) -> bool:
        return "graph not loaded" in self.reason.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_intent": self.expected_intent,
            "actual_intent": self.actual_intent,
            "expected_symbol": self.expected_symbol,
            "actual_symbol": self.actual_symbol,
            "expected_tool": self.expected_tool,
            "actual_tool": self.actual_tool,
            "confidence": self.confidence,
            "reason": self.reason,
            "correct": self.correct,
            "graph_fallback": self.graph_fallback,
        }


@dataclass(frozen=True)
class RoutingBenchmark:
    """Baseline/candidate retrieval comparison and promotion verdict."""

    baseline: EvalReport
    candidate: EvalReport
    observations: tuple[RoutingObservation, ...]
    router_latency_p50_ms: float
    router_latency_p95_ms: float
    route_accuracy_threshold: float = 0.95
    router_latency_limit_ms: float = 0.1

    @property
    def routing_accuracy(self) -> float:
        if not self.observations:
            return 0.0
        return sum(observation.correct for observation in self.observations) / len(
            self.observations
        )

    @property
    def success_delta(self) -> float:
        return self.candidate.success_at_k - self.baseline.success_at_k

    @property
    def ndcg_delta(self) -> float:
        return self.candidate.ndcg_at_k - self.baseline.ndcg_at_k

    @property
    def lost_baseline_hits(self) -> tuple[str, ...]:
        baseline_hits = {result.case.id for result in self.baseline.results if result.hit}
        candidate_hits = {result.case.id for result in self.candidate.results if result.hit}
        return tuple(sorted(baseline_hits - candidate_hits))

    @property
    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.routing_accuracy < self.route_accuracy_threshold:
            failures.append(
                f"routing accuracy {self.routing_accuracy:.3f} is below "
                f"{self.route_accuracy_threshold:.3f}"
            )
        if any(observation.graph_fallback for observation in self.observations):
            failures.append("one or more structural routes fell back because the graph was unavailable")
        if self.success_delta < 0:
            failures.append(f"Success@K regressed by {self.success_delta:.6f}")
        if self.ndcg_delta < 0:
            failures.append(f"NDCG@K regressed by {self.ndcg_delta:.6f}")
        if self.lost_baseline_hits:
            failures.append("candidate lost baseline hits: " + ", ".join(self.lost_baseline_hits))
        if self.router_latency_p95_ms > self.router_latency_limit_ms:
            failures.append(
                f"router p95 {self.router_latency_p95_ms:.6f} ms exceeds "
                f"{self.router_latency_limit_ms:.6f} ms"
            )
        return tuple(failures)

    @property
    def eligible(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "failures": list(self.failures),
            "cases": len(self.observations),
            "top_k": self.candidate.top_k,
            "routing_accuracy": self.routing_accuracy,
            "baseline": _report_payload(self.baseline),
            "candidate": _report_payload(self.candidate),
            "delta": {
                "success_at_k": self.success_delta,
                "ndcg_at_k": self.ndcg_delta,
            },
            "router_latency": {
                "p50_ms": self.router_latency_p50_ms,
                "p95_ms": self.router_latency_p95_ms,
                "limit_ms": self.router_latency_limit_ms,
            },
            "lost_baseline_hits": list(self.lost_baseline_hits),
            "routes": [observation.to_dict() for observation in self.observations],
        }


class _RoutedEvalEngine:
    def __init__(
        self,
        *,
        storage: Any,
        embedder: Any,
        search_engine: Any,
        router: Any,
        cases_by_query: dict[str, RoutingCase],
        smart_search_fn: Callable[..., SmartSearchResult],
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._search_engine = search_engine
        self._router = router
        self._cases_by_query = cases_by_query
        self._smart_search = smart_search_fn
        self.observations: list[RoutingObservation] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        repo_id: int | None = None,
    ) -> list[Any]:
        result = self._smart_search(
            query,
            self._storage,
            self._embedder,
            top_k=top_k,
            router=self._router,
            search_engine=self._search_engine,
            repo_id=repo_id,
        )
        case = self._cases_by_query[query]
        self.observations.append(
            RoutingObservation(
                case_id=case.eval_case.id,
                expected_intent=case.expected_intent.value,
                actual_intent=result.routed.intent.value,
                expected_symbol=case.expected_route_symbol,
                actual_symbol=result.routed.symbol,
                expected_tool=case.expected_tool,
                actual_tool=result.tool_used,
                confidence=result.routed.confidence,
                reason=result.routed.reason,
            )
        )
        return result.hits


def load_routing_cases(path: Path) -> list[RoutingCase]:
    """Load and validate the frozen structural-routing JSONL catalog."""
    cases: list[RoutingCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise RoutingBenchmarkError("routing case must be a JSON object")
            case = RoutingCase.from_dict(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RoutingBenchmarkError(f"{path}:{line_number}: {exc}") from exc
        if case.eval_case.id in seen_ids:
            raise RoutingBenchmarkError(f"duplicate routing case id {case.eval_case.id!r}")
        if case.eval_case.query in seen_queries:
            raise RoutingBenchmarkError(f"duplicate routing query {case.eval_case.query!r}")
        seen_ids.add(case.eval_case.id)
        seen_queries.add(case.eval_case.query)
        cases.append(case)
    if not cases:
        raise RoutingBenchmarkError(f"routing catalog is empty: {path}")
    return cases


def benchmark_routing(
    *,
    cases: Sequence[RoutingCase],
    storage: Any,
    embedder: Any,
    search_engine: Any,
    top_k: int = 10,
    repo_id: int | None = None,
    provenance: dict[str, Any] | None = None,
    route_iterations: int = 10_000,
    router_latency_limit_ms: float = 0.1,
    smart_search_fn: Callable[..., SmartSearchResult] = smart_search,
) -> RoutingBenchmark:
    """Compare semantic-only and deterministic structural routing on one runtime."""
    if not cases:
        raise RoutingBenchmarkError("at least one routing case is required")
    if route_iterations < 1:
        raise RoutingBenchmarkError("route_iterations must be positive")
    cases_by_query = {case.eval_case.query: case for case in cases}
    if len(cases_by_query) != len(cases):
        raise RoutingBenchmarkError("routing case queries must be unique")

    baseline_engine = _RoutedEvalEngine(
        storage=storage,
        embedder=embedder,
        search_engine=search_engine,
        router=NoopRouter(),
        cases_by_query=cases_by_query,
        smart_search_fn=smart_search_fn,
    )
    candidate_router = PatternRouter()
    candidate_engine = _RoutedEvalEngine(
        storage=storage,
        embedder=embedder,
        search_engine=search_engine,
        router=candidate_router,
        cases_by_query=cases_by_query,
        smart_search_fn=smart_search_fn,
    )
    eval_cases = [case.eval_case for case in cases]
    baseline = run_eval(
        eval_cases,
        baseline_engine,  # type: ignore[arg-type]
        top_k=top_k,
        repo_id=repo_id,
        provenance=dict(provenance or {}) | {"router": "noop"},
    )
    candidate = run_eval(
        eval_cases,
        candidate_engine,  # type: ignore[arg-type]
        top_k=top_k,
        repo_id=repo_id,
        provenance=dict(provenance or {}) | {"router": candidate_router.adapter_id},
    )

    queries = tuple(cases_by_query)
    for index in range(min(1_000, route_iterations)):
        candidate_router.route(queries[index % len(queries)])
    samples_ns: list[int] = []
    for index in range(route_iterations):
        start = perf_counter_ns()
        candidate_router.route(queries[index % len(queries)])
        samples_ns.append(perf_counter_ns() - start)
    samples_ns.sort()

    return RoutingBenchmark(
        baseline=baseline,
        candidate=candidate,
        observations=tuple(candidate_engine.observations),
        router_latency_p50_ms=_percentile(samples_ns, 50) / 1_000_000,
        router_latency_p95_ms=_percentile(samples_ns, 95) / 1_000_000,
        router_latency_limit_ms=router_latency_limit_ms,
    )


def _percentile(samples: Sequence[int], percentile: int) -> int:
    index = min(len(samples) - 1, int(len(samples) * percentile / 100))
    return samples[max(0, index)]


def _report_payload(report: EvalReport) -> dict[str, Any]:
    return {
        "router": report.provenance.get("router"),
        "success_at_k": report.success_at_k,
        "ndcg_at_k": report.ndcg_at_k,
        "latency_p50_ms": report.latency_p50_ms,
        "latency_p95_ms": report.latency_p95_ms,
        "results": [
            {
                "id": result.case.id,
                "hit": result.hit,
                "rank": result.rank_of_first_hit,
                "ndcg": result.ndcg,
                "latency_ms": result.latency_ms,
            }
            for result in report.results
        ],
    }
