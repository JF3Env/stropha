"""Tests for the deterministic structural-routing holdout benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stropha.adapters.router.pattern import PatternRouter
from stropha.bench.routing import (
    RoutingBenchmarkError,
    RoutingCase,
    benchmark_routing,
    load_routing_cases,
)
from stropha.models import SearchHit
from stropha.retrieval.query_router import QueryIntent
from stropha.retrieval.smart_search import SmartSearchResult


def _case(
    case_id: str,
    query: str,
    expected_path: str,
    intent: QueryIntent,
    symbol: str,
    tool: str,
) -> RoutingCase:
    return RoutingCase.from_dict(
        {
            "id": case_id,
            "query": query,
            "expected_paths": [expected_path],
            "expected_symbols": [symbol],
            "expected_intent": intent.value,
            "expected_route_symbol": symbol,
            "expected_tool": tool,
            "tags": ["structural-routing", "test"],
            "split": "holdout",
        }
    )


def _hit(path: str, symbol: str) -> SearchHit:
    return SearchHit(
        rank=1,
        score=1.0,
        rel_path=path,
        language="python",
        kind="function",
        symbol=symbol,
        start_line=1,
        end_line=1,
        snippet=symbol,
        chunk_id=f"chunk:{symbol}",
    )


def test_load_routing_cases_requires_frozen_holdout_contract(tmp_path: Path) -> None:
    path = tmp_path / "routing.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "query": "callers of Foo",
                "expected_paths": ["foo.py"],
                "expected_intent": "find_callers",
                "expected_tool": "find_callers",
                "tags": ["structural-routing"],
                "split": "search",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RoutingBenchmarkError, match="split='holdout'"):
        load_routing_cases(path)


def test_checked_in_routing_catalog_matches_deterministic_router() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = root / "tests/eval/routing.jsonl"
    cases = load_routing_cases(catalog)
    router = PatternRouter()

    assert len(cases) >= 12
    for case in cases:
        routed = router.route(case.eval_case.query)
        assert routed.intent == case.expected_intent, case.eval_case.id
        assert routed.symbol == case.expected_route_symbol, case.eval_case.id

    evidence = json.loads(
        (root / "tests/eval/routing-baseline-997e15e9.json").read_text(encoding="utf-8")
    )
    assert evidence["provenance"]["golden_sha256"] == hashlib.sha256(
        catalog.read_bytes()
    ).hexdigest()
    assert evidence["eligible"] is True


def test_benchmark_accepts_non_regressing_deterministic_routes() -> None:
    cases = [
        _case(
            "callers",
            "callers of Foo",
            "caller.py",
            QueryIntent.FIND_CALLERS,
            "Foo",
            "find_callers",
        ),
        _case(
            "symbol",
            "definition of Bar",
            "bar.py",
            QueryIntent.GET_SYMBOL,
            "Bar",
            "get_symbol",
        ),
    ]

    def fake_smart_search(
        query: str,
        _storage: Any,
        _embedder: Any,
        *,
        router: Any,
        **_kwargs: Any,
    ) -> SmartSearchResult:
        routed = router.route(query)
        if routed.intent == QueryIntent.FIND_CALLERS:
            tool = "find_callers"
            hits = [_hit("caller.py", "Foo")]
        elif routed.intent == QueryIntent.GET_SYMBOL:
            tool = "get_symbol"
            hits = [_hit("bar.py", "Bar")]
        else:
            tool = "search_code"
            hits = [_hit("caller.py", "Foo"), _hit("bar.py", "Bar")]
        return SmartSearchResult(hits=hits, routed=routed, tool_used=tool)

    result = benchmark_routing(
        cases=cases,
        storage=SimpleNamespace(),
        embedder=SimpleNamespace(),
        search_engine=SimpleNamespace(),
        route_iterations=20,
        router_latency_limit_ms=10.0,
        smart_search_fn=fake_smart_search,
    )

    assert result.eligible is True
    assert result.routing_accuracy == 1.0
    assert result.success_delta == 0.0
    assert result.ndcg_delta > 0.0
    assert result.lost_baseline_hits == ()


def test_benchmark_rejects_route_mismatch_and_lost_baseline_hit() -> None:
    cases = [
        _case(
            "callers",
            "callers of Foo",
            "caller.py",
            QueryIntent.FIND_CALLERS,
            "Foo",
            "find_callers",
        )
    ]

    def fake_smart_search(
        query: str,
        _storage: Any,
        _embedder: Any,
        *,
        router: Any,
        **_kwargs: Any,
    ) -> SmartSearchResult:
        routed = router.route(query)
        if routed.intent == QueryIntent.SEARCH_CODE:
            return SmartSearchResult(
                hits=[_hit("caller.py", "Foo")],
                routed=routed,
                tool_used="search_code",
            )
        routed = routed.__class__(
            intent=QueryIntent.SEARCH_CODE,
            symbol="Foo",
            original_query=query,
            confidence=0.4,
            reason="graph not loaded, fallback from find_callers",
        )
        return SmartSearchResult(hits=[], routed=routed, tool_used="search_code")

    result = benchmark_routing(
        cases=cases,
        storage=SimpleNamespace(),
        embedder=SimpleNamespace(),
        search_engine=SimpleNamespace(),
        route_iterations=10,
        router_latency_limit_ms=10.0,
        smart_search_fn=fake_smart_search,
    )

    assert result.eligible is False
    assert result.routing_accuracy == 0.0
    assert result.lost_baseline_hits == ("callers",)
    assert any("graph was unavailable" in failure for failure in result.failures)
    assert any("Success@K regressed" in failure for failure in result.failures)
