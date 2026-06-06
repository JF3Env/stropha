"""Sub-query budget allocator (WIN-236, EPIC 1.10 #27).

When :func:`query_decomposition.decompose_query` splits a compound
question into N atomic sub-queries, each sub-query produces its own
result set. The WIN-208 packer caps the FINAL union under a total
context_budget, but it has no way to influence how that budget is
spread across the sub-queries BEFORE search.

WIN-236 adds a thin allocation layer between decomposition and
packing: given a list of sub-queries + a total budget + an
optional weight function, return per-sub-query budgets that sum to
the total. Sub-queries the agent / Meta-Harness deems more
important get a larger slice.

Strategies
----------

- ``uniform`` (default): split equally. Simplest, surprises no one.
- ``weighted``: caller passes explicit weights; budgets are
  proportional. Weights are normalised to sum 1 before allocation.
- ``length`` (heuristic): weight by sub-query token length —
  longer queries are usually more complex and benefit from a
  bigger candidate pool.

A floor (``min_per_query``) guarantees every sub-query gets at
least that many tokens — useful when a long-query bias would
otherwise starve a one-word sub-query of any room.

Rounding
--------

We allocate integer token budgets that SUM EXACTLY to ``total``.
The remainder from integer division goes to the highest-weighted
sub-query, so the operator never sees "1999 ≠ 2000" off-by-one
mismatches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

AllocationStrategy = Literal["uniform", "weighted", "length"]


@dataclass(frozen=True)
class BudgetAllocation:
    """Result of :func:`allocate_budget`.

    ``per_query`` is a list parallel to the input ``sub_queries``;
    each entry is the integer token budget for that sub-query.
    ``strategy`` echoes the strategy used so logs / debug output
    surface what happened.
    """

    per_query: list[int] = field(default_factory=list)
    total: int = 0
    strategy: str = "uniform"
    notes: list[str] = field(default_factory=list)


def _length_weights(sub_queries: list[str]) -> list[float]:
    """Weight sub-queries by character length.

    Longer queries usually carry more substance; a one-word
    sub-query rarely deserves the same budget as a 12-word
    technical question. We use ``max(1, len(q))`` so an empty
    query still gets a non-zero share.
    """
    return [float(max(1, len(q))) for q in sub_queries]


def allocate_budget(
    sub_queries: list[str],
    total: int,
    *,
    strategy: AllocationStrategy = "uniform",
    weights: list[float] | None = None,
    min_per_query: int = 0,
    weight_fn: Callable[[list[str]], list[float]] | None = None,
) -> BudgetAllocation:
    """Distribute ``total`` tokens across ``sub_queries``.

    Args:
        sub_queries: the parallel list of sub-queries from
            decomposition. Must not be empty.
        total: total budget to distribute. ``total <= 0`` returns
            an empty allocation.
        strategy: one of ``"uniform"`` / ``"weighted"`` /
            ``"length"``. ``"weighted"`` requires the ``weights``
            argument.
        weights: per-sub-query weights. Required when
            ``strategy="weighted"``. Lengths must match
            ``sub_queries``; negative values are clamped to 0.
        min_per_query: floor for each sub-query's allocation.
            When the floor × count > total, the floor wins and
            the remainder shrinks the largest allocation first.
        weight_fn: when set + ``strategy="length"``, override the
            built-in length weights. Lets the Meta-Harness inject
            its own estimator without touching this module.

    Returns:
        :class:`BudgetAllocation` whose ``per_query`` sums to
        exactly ``total`` (when the floor doesn't force overflow).
    """
    if not sub_queries:
        return BudgetAllocation(
            per_query=[], total=0, strategy=strategy,
            notes=["empty sub_queries — nothing to allocate"],
        )
    if total <= 0:
        return BudgetAllocation(
            per_query=[0] * len(sub_queries),
            total=0,
            strategy=strategy,
            notes=["total <= 0 — zero-budget allocation"],
        )

    n = len(sub_queries)

    if strategy == "uniform":
        weights_resolved = [1.0] * n
    elif strategy == "weighted":
        if weights is None:
            raise ValueError(
                "strategy='weighted' requires the weights argument"
            )
        if len(weights) != n:
            raise ValueError(
                f"weights length {len(weights)} != "
                f"sub_queries length {n}"
            )
        # Clamp negatives to 0; allows callers to pass raw scores.
        weights_resolved = [max(0.0, w) for w in weights]
    elif strategy == "length":
        if weight_fn is None:
            weights_resolved = _length_weights(sub_queries)
        else:
            weights_resolved = weight_fn(sub_queries)
            if len(weights_resolved) != n:
                raise ValueError(
                    "weight_fn returned wrong number of weights"
                )
            weights_resolved = [max(0.0, w) for w in weights_resolved]
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    # Handle all-zero weights by collapsing to uniform.
    weight_sum = sum(weights_resolved)
    notes: list[str] = []
    if weight_sum <= 0:
        weights_resolved = [1.0] * n
        weight_sum = float(n)
        notes.append("all weights ≤ 0; fell back to uniform")

    # Proportional integer allocation. We compute floats first,
    # floor them, then distribute the remainder to the highest-
    # weighted sub-queries so the sum equals ``total`` exactly.
    raw = [(w / weight_sum) * total for w in weights_resolved]
    floored = [int(x) for x in raw]
    remainder = total - sum(floored)
    if remainder > 0:
        # Distribute the leftover units to the entries with the
        # largest fractional component first (standard largest-
        # remainder method).
        fractions = [
            (i, raw[i] - floored[i]) for i in range(n)
        ]
        fractions.sort(key=lambda t: t[1], reverse=True)
        for i, _frac in fractions[:remainder]:
            floored[i] += 1

    # Enforce the floor — borrow from the largest allocation first.
    if min_per_query > 0:
        deficits = [
            max(0, min_per_query - x) for x in floored
        ]
        total_deficit = sum(deficits)
        if total_deficit > 0:
            # Add the deficit so each is at least min_per_query.
            for i, d in enumerate(deficits):
                floored[i] += d
            # Remove from the largest allocations (skipping the
            # ones that were just lifted).
            surplus = total_deficit
            # Sort indices by current allocation DESC; skip those
            # that just hit the floor.
            ordered = sorted(
                range(n),
                key=lambda i: floored[i],
                reverse=True,
            )
            for i in ordered:
                if surplus <= 0:
                    break
                if floored[i] <= min_per_query:
                    continue
                takeable = floored[i] - min_per_query
                take = min(takeable, surplus)
                floored[i] -= take
                surplus -= take
            if surplus > 0:
                notes.append(
                    f"min_per_query={min_per_query} × {n} > total={total}; "
                    "floor forces overflow"
                )

    return BudgetAllocation(
        per_query=floored,
        total=sum(floored),
        strategy=strategy,
        notes=notes,
    )


__all__ = [
    "AllocationStrategy",
    "BudgetAllocation",
    "allocate_budget",
]
