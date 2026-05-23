"""Unit tests for cost-budgeted multi-hop traversal
(WIN-241, EPIC 1.10 #32)."""

from __future__ import annotations

import pytest

from stropha.retrieval.cost_traversal import (
    TraversalBudget,
    TraversalResult,
    VisitedNode,
    cost_budgeted_bfs,
)


# ---------------------------------------------------------------------------
# Test graph helpers
# ---------------------------------------------------------------------------


def _make_neighbors(graph: dict[str, list[str]]) -> "function":
    """Return a neighbors_fn over the given adjacency dict."""
    def _fn(node: str) -> list[str]:
        return list(graph.get(node, ()))
    return _fn  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_budget_defaults(self) -> None:
        b = TraversalBudget()
        assert b.max_cost is None
        assert b.max_depth is None
        assert b.max_visits is None

    def test_visited_node_defaults(self) -> None:
        v = VisitedNode(node="x")
        assert v.depth == 0
        assert v.cost == 0.0

    def test_traversal_result_defaults(self) -> None:
        r = TraversalResult()
        assert r.visited == []
        assert r.total_cost == 0.0
        assert r.max_depth_reached == 0
        assert r.halted_reason == "exhausted"

    def test_frozen(self) -> None:
        b = TraversalBudget()
        with pytest.raises((AttributeError, Exception)):
            b.max_cost = 100.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Basic BFS
# ---------------------------------------------------------------------------


class TestBasicBFS:
    def test_single_node_no_neighbors(self) -> None:
        result = cost_budgeted_bfs(
            "root", lambda n: [],
            budget=TraversalBudget(),
        )
        assert len(result.visited) == 1
        assert result.visited[0].node == "root"
        assert result.visited[0].depth == 0
        assert result.halted_reason == "exhausted"

    def test_simple_chain(self) -> None:
        graph = {"a": ["b"], "b": ["c"], "c": []}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(),
        )
        nodes = [v.node for v in result.visited]
        assert nodes == ["a", "b", "c"]
        assert result.max_depth_reached == 2

    def test_branching_graph(self) -> None:
        graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(),
        )
        # BFS order: a, b, c, d (d visited once via b, deduped from c)
        nodes = [v.node for v in result.visited]
        assert nodes == ["a", "b", "c", "d"]

    def test_cycle_safe(self) -> None:
        graph = {"a": ["b"], "b": ["a"]}  # cycle
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(),
        )
        nodes = [v.node for v in result.visited]
        assert nodes == ["a", "b"]


# ---------------------------------------------------------------------------
# Cost cap
# ---------------------------------------------------------------------------


class TestCostBudget:
    def test_unit_cost_caps_visits(self) -> None:
        graph = {f"n{i}": [f"n{i+1}"] for i in range(10)}
        graph["n10"] = []
        result = cost_budgeted_bfs(
            "n0", _make_neighbors(graph),
            budget=TraversalBudget(max_cost=3.0),
        )
        # Seed + 2 neighbors = 3 nodes, then the 4th would exceed.
        assert len(result.visited) == 3
        assert result.halted_reason == "budget_exhausted"
        assert result.total_cost == 3.0

    def test_seed_busts_budget_returns_empty(self) -> None:
        result = cost_budgeted_bfs(
            "expensive", lambda n: [],
            budget=TraversalBudget(max_cost=0.5),
            cost_fn=lambda _: 1.0,
        )
        assert result.visited == []
        assert result.halted_reason == "budget_exhausted"
        assert result.total_cost == 0.0

    def test_token_cost_function(self) -> None:
        """Cost function based on node 'token count' — verifies
        cumulative budget tracking with non-unit costs."""
        sizes = {"a": 10, "b": 20, "c": 5, "d": 100}
        graph = {"a": ["b", "c"], "b": ["d"], "c": [], "d": []}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(max_cost=40.0),
            cost_fn=lambda n: float(sizes.get(n, 0)),
        )
        # a (10) + b (20) + c (5) = 35; d (100) would push to 135 > 40 → skip.
        nodes = [v.node for v in result.visited]
        assert set(nodes) == {"a", "b", "c"}
        assert result.total_cost == 35.0
        assert result.halted_reason == "budget_exhausted"

    def test_cheaper_sibling_fits_after_expensive_skipped(self) -> None:
        """When the budget rejects one neighbor at a depth, cheaper
        siblings at the same depth should still be considered."""
        # a -> [b (1000), c (1)]. Budget 100; a costs 10.
        # We want c to be visited even though b is rejected.
        sizes = {"a": 10, "b": 1000, "c": 1}
        graph = {"a": ["b", "c"], "b": [], "c": []}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(max_cost=100.0),
            cost_fn=lambda n: float(sizes.get(n, 0)),
        )
        nodes = {v.node for v in result.visited}
        assert "c" in nodes
        assert "b" not in nodes


# ---------------------------------------------------------------------------
# Depth cap
# ---------------------------------------------------------------------------


class TestDepthCap:
    def test_depth_one_returns_seed_plus_direct(self) -> None:
        graph = {"a": ["b", "c"], "b": ["d"], "c": ["e"]}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(max_depth=1),
        )
        nodes = {v.node for v in result.visited}
        assert nodes == {"a", "b", "c"}
        assert result.halted_reason == "depth_exhausted"

    def test_depth_zero_returns_seed_only(self) -> None:
        graph = {"a": ["b"]}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(max_depth=0),
        )
        assert [v.node for v in result.visited] == ["a"]
        assert result.halted_reason == "depth_exhausted"


# ---------------------------------------------------------------------------
# Visit cap
# ---------------------------------------------------------------------------


class TestVisitCap:
    def test_visit_cap_truncates(self) -> None:
        graph = {f"n{i}": [f"n{i+1}"] for i in range(10)}
        graph["n10"] = []
        result = cost_budgeted_bfs(
            "n0", _make_neighbors(graph),
            budget=TraversalBudget(max_visits=4),
        )
        assert len(result.visited) == 4
        assert result.halted_reason == "visit_limit"

    def test_visit_cap_one_returns_seed_only(self) -> None:
        graph = {"a": ["b"]}
        result = cost_budgeted_bfs(
            "a", _make_neighbors(graph),
            budget=TraversalBudget(max_visits=1),
        )
        assert [v.node for v in result.visited] == ["a"]
        assert result.halted_reason == "visit_limit"

    def test_visit_cap_zero_returns_empty(self) -> None:
        result = cost_budgeted_bfs(
            "a", lambda n: [],
            budget=TraversalBudget(max_visits=0),
        )
        assert result.visited == []
        assert result.halted_reason == "visit_limit"


# ---------------------------------------------------------------------------
# Combined caps
# ---------------------------------------------------------------------------


class TestCombinedCaps:
    def test_first_cap_wins(self) -> None:
        graph = {f"n{i}": [f"n{i+1}"] for i in range(10)}
        graph["n10"] = []
        # max_cost=3 will trigger first (visit_limit=5 wouldn't have).
        result = cost_budgeted_bfs(
            "n0", _make_neighbors(graph),
            budget=TraversalBudget(max_cost=3.0, max_visits=5),
        )
        assert result.halted_reason == "budget_exhausted"
        assert len(result.visited) == 3


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


class TestDefensive:
    def test_neighbors_fn_exception_treated_as_deadend(self) -> None:
        def explosive_neighbors(node: str) -> list[str]:
            if node == "b":
                raise RuntimeError("kaboom")
            return ["b"] if node == "a" else []

        result = cost_budgeted_bfs(
            "a", explosive_neighbors,
            budget=TraversalBudget(),
        )
        # a + b visited; b's neighbors raise → treated as dead-end.
        nodes = [v.node for v in result.visited]
        assert nodes == ["a", "b"]
        # The traversal completes normally rather than propagating.
        assert result.halted_reason == "exhausted"

    def test_custom_key_fn(self) -> None:
        """Custom key_fn allows dicts as nodes by extracting their id."""
        nodes_db = {
            1: {"id": 1, "next": [2, 3]},
            2: {"id": 2, "next": [3]},  # cycle target via different path
            3: {"id": 3, "next": []},
        }
        def neighbors_fn(node: dict) -> list[dict]:
            return [nodes_db[nid] for nid in node["next"]]

        result = cost_budgeted_bfs(
            nodes_db[1], neighbors_fn,
            budget=TraversalBudget(),
            key_fn=lambda n: n["id"],
        )
        ids = [v.node["id"] for v in result.visited]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import cost_traversal as mod
        assert set(mod.__all__) == {
            "TraversalBudget",
            "TraversalResult",
            "VisitedNode",
            "cost_budgeted_bfs",
        }
