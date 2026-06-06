"""Cost-budgeted multi-hop graph traversal (WIN-241, EPIC 1.10 #32).

Today the graph tools (``find_callers``, ``find_related``,
``trace_feature``, …) cap traversal by ``depth`` and ``limit``.
Both are blunt instruments: ``depth=2`` on a sparse graph returns
two nodes; on a fan-out hub it returns hundreds. The agent that
asked the question can't predict the TCO from those parameters.

WIN-241 adds a cost-budgeted alternative. The traversal stops when
the cumulative cost (sum of a per-node cost function) exceeds a
budget — so the agent dials in *spend*, not *depth*. Useful defaults:

- ``cost_fn=lambda n: 1`` → budget is "visit at most N nodes".
- ``cost_fn=lambda n: token_count(n.content)`` → budget is "consume
  at most B tokens" (mirrors the WIN-208 packer's currency).

The traversal is a generic BFS over a caller-supplied
``neighbors_fn`` so the primitive doesn't depend on the storage
schema. The matching consumer (``find_callers_budgeted``, future
work) wires it into the SQL graph reader.

Halt reasons
------------

- ``budget_exhausted`` — adding the next node would push cumulative
  cost over ``max_cost``.
- ``depth_exhausted`` — ``max_depth`` reached.
- ``visit_limit`` — ``max_visits`` reached.
- ``exhausted`` — the BFS frontier emptied (graph too small for the
  budget).

This module is a **standalone primitive** matching the WIN-230 /
WIN-236 / WIN-237 / WIN-238 / WIN-239 / WIN-240 pattern. No
auto-integration into the graph tools yet.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

NodeT = TypeVar("NodeT")


@dataclass(frozen=True)
class TraversalBudget:
    """Hard caps for :func:`cost_budgeted_bfs`.

    Any cap that is ``None`` is treated as "no limit" for that
    dimension. The traversal stops as soon as ANY cap is reached.
    """

    max_cost: float | None = None
    max_depth: int | None = None
    max_visits: int | None = None


@dataclass(frozen=True)
class VisitedNode:
    """One node visited by :func:`cost_budgeted_bfs`.

    ``node`` is the opaque caller-supplied object. ``depth`` is the
    BFS distance from ``start``. ``cost`` is what the cost function
    returned for this node (the operator can audit which node spent
    the most budget).
    """

    node: Any
    depth: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class TraversalResult:
    """Aggregate of one :func:`cost_budgeted_bfs` invocation.

    ``visited`` is the ordered list of nodes consumed (BFS order
    by default). ``total_cost`` sums each visited node's cost.
    ``halted_reason`` documents WHY the traversal ended — agents
    rely on this to decide whether to spend more budget on a
    follow-up call.
    """

    visited: list[VisitedNode] = field(default_factory=list)
    total_cost: float = 0.0
    max_depth_reached: int = 0
    halted_reason: str = "exhausted"


def _unit_cost(_node: Any) -> float:
    return 1.0


def _identity_key(node: Any) -> Any:
    return node


def cost_budgeted_bfs[NodeT](
    start: NodeT,
    neighbors_fn: Callable[[NodeT], Iterable[NodeT]],
    *,
    budget: TraversalBudget,
    cost_fn: Callable[[NodeT], float] = _unit_cost,
    key_fn: Callable[[NodeT], Any] = _identity_key,
) -> TraversalResult:
    """BFS from ``start`` until any cap in ``budget`` is reached.

    Args:
        start: the seed node. Its own cost is charged to the budget
            before any expansion happens.
        neighbors_fn: callable returning an iterable of nodes
            directly reachable from a given node. Cycles in the
            graph are tolerated — visited nodes are deduplicated by
            ``key_fn``.
        budget: caps; any ``None`` field disables that dimension.
        cost_fn: per-node cost — defaults to a unit cost so the
            budget collapses to a node-count limit. Pass
            ``token_count`` (or similar) when budgeting in tokens.
        key_fn: returns a hashable identity for a node. Defaults to
            the node itself — useful when nodes are already hashable
            (strings / ints). Pass ``lambda n: n["id"]`` for dicts.

    Returns:
        :class:`TraversalResult` whose ``halted_reason`` tells the
        caller why the BFS ended.
    """
    visited: list[VisitedNode] = []
    seen: set[Any] = set()
    total_cost = 0.0
    max_depth_reached = 0
    halted_reason = "exhausted"

    start_key = key_fn(start)
    start_cost = float(cost_fn(start))

    if budget.max_cost is not None and start_cost > budget.max_cost:
        # Even the seed busts the budget — return empty with
        # explicit halt reason so the caller doesn't mistake an
        # empty result for "the graph is empty".
        return TraversalResult(
            visited=[],
            total_cost=0.0,
            max_depth_reached=0,
            halted_reason="budget_exhausted",
        )

    if budget.max_visits is not None and budget.max_visits <= 0:
        return TraversalResult(
            visited=[],
            total_cost=0.0,
            max_depth_reached=0,
            halted_reason="visit_limit",
        )

    visited.append(VisitedNode(node=start, depth=0, cost=start_cost))
    seen.add(start_key)
    total_cost += start_cost

    if budget.max_visits is not None and len(visited) >= budget.max_visits:
        return TraversalResult(
            visited=visited,
            total_cost=total_cost,
            max_depth_reached=0,
            halted_reason="visit_limit",
        )

    queue: deque[tuple[NodeT, int]] = deque([(start, 0)])

    while queue:
        node, depth = queue.popleft()
        next_depth = depth + 1

        if budget.max_depth is not None and next_depth > budget.max_depth:
            halted_reason = "depth_exhausted"
            continue

        try:
            neighbors = list(neighbors_fn(node))
        except Exception:
            # Defensive: a graph reader that throws on a missing
            # node shouldn't crash the traversal. Treat the node as
            # a dead-end.
            continue

        for neighbor in neighbors:
            neighbor_key = key_fn(neighbor)
            if neighbor_key in seen:
                continue

            neighbor_cost = float(cost_fn(neighbor))
            prospective_total = total_cost + neighbor_cost
            if (
                budget.max_cost is not None
                and prospective_total > budget.max_cost
            ):
                halted_reason = "budget_exhausted"
                # Don't add this neighbor, but keep exploring siblings
                # at this depth — a cheaper sibling may still fit.
                continue

            seen.add(neighbor_key)
            visited.append(VisitedNode(
                node=neighbor, depth=next_depth, cost=neighbor_cost,
            ))
            total_cost = prospective_total
            max_depth_reached = max(max_depth_reached, next_depth)

            if (
                budget.max_visits is not None
                and len(visited) >= budget.max_visits
            ):
                return TraversalResult(
                    visited=visited,
                    total_cost=total_cost,
                    max_depth_reached=max_depth_reached,
                    halted_reason="visit_limit",
                )

            queue.append((neighbor, next_depth))

    return TraversalResult(
        visited=visited,
        total_cost=total_cost,
        max_depth_reached=max_depth_reached,
        halted_reason=halted_reason,
    )


__all__ = [
    "TraversalBudget",
    "TraversalResult",
    "VisitedNode",
    "cost_budgeted_bfs",
]
