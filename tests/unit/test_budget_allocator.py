"""Unit tests for the sub-query budget allocator (WIN-236, EPIC 1.10 #27).

Coverage target: every branch of ``allocate_budget`` plus the
``_length_weights`` helper. Pure-Python, no fixtures, no IO.
"""

from __future__ import annotations

import pytest

from stropha.retrieval.budget_allocator import (
    BudgetAllocation,
    _length_weights,
    allocate_budget,
)


# ---------------------------------------------------------------------------
# BudgetAllocation dataclass
# ---------------------------------------------------------------------------


class TestBudgetAllocationDataclass:
    def test_defaults(self) -> None:
        alloc = BudgetAllocation()
        assert alloc.per_query == []
        assert alloc.total == 0
        assert alloc.strategy == "uniform"
        assert alloc.notes == []

    def test_frozen(self) -> None:
        alloc = BudgetAllocation(per_query=[1, 2], total=3, strategy="weighted")
        with pytest.raises((AttributeError, Exception)):
            alloc.total = 99  # type: ignore[misc]

    def test_default_factories_are_independent(self) -> None:
        """Both ``per_query`` and ``notes`` use ``field(default_factory=list)``
        so two instances must not share mutable state."""
        a = BudgetAllocation()
        b = BudgetAllocation()
        assert a.per_query is not b.per_query
        assert a.notes is not b.notes


# ---------------------------------------------------------------------------
# _length_weights helper
# ---------------------------------------------------------------------------


class TestLengthWeights:
    def test_proportional_to_length(self) -> None:
        weights = _length_weights(["abc", "abcdef"])
        assert weights == [3.0, 6.0]

    def test_empty_string_gets_floor(self) -> None:
        """Per the docstring, an empty query still gets a non-zero share."""
        assert _length_weights([""]) == [1.0]

    def test_mixed(self) -> None:
        weights = _length_weights(["", "hi", "a much longer one"])
        assert weights == [1.0, 2.0, float(len("a much longer one"))]


# ---------------------------------------------------------------------------
# allocate_budget — empty / zero-budget edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sub_queries(self) -> None:
        alloc = allocate_budget([], 1000)
        assert alloc.per_query == []
        assert alloc.total == 0
        assert alloc.strategy == "uniform"
        assert any("empty sub_queries" in note for note in alloc.notes)

    def test_empty_sub_queries_echoes_strategy(self) -> None:
        alloc = allocate_budget([], 1000, strategy="length")
        # The empty branch still echoes the requested strategy so the
        # operator can see what was asked even when nothing was done.
        assert alloc.strategy == "length"

    def test_total_zero(self) -> None:
        alloc = allocate_budget(["q1", "q2"], 0)
        assert alloc.per_query == [0, 0]
        assert alloc.total == 0
        assert any("total <= 0" in note for note in alloc.notes)

    def test_total_negative(self) -> None:
        alloc = allocate_budget(["q1", "q2", "q3"], -5)
        assert alloc.per_query == [0, 0, 0]
        assert alloc.total == 0
        assert any("total <= 0" in note for note in alloc.notes)


# ---------------------------------------------------------------------------
# uniform strategy
# ---------------------------------------------------------------------------


class TestUniformStrategy:
    def test_even_split(self) -> None:
        alloc = allocate_budget(["a", "b", "c", "d"], 400)
        assert alloc.per_query == [100, 100, 100, 100]
        assert alloc.total == 400
        assert alloc.strategy == "uniform"
        assert alloc.notes == []

    def test_uneven_split_distributes_remainder(self) -> None:
        # 10 / 3 = 3 with remainder 1 → largest fraction wins.
        alloc = allocate_budget(["a", "b", "c"], 10)
        assert sum(alloc.per_query) == 10
        # Each entry has the same fraction (0.33), so the first slot
        # picked by the sort gets the +1 in stable order.
        assert sorted(alloc.per_query) == [3, 3, 4]

    def test_single_query(self) -> None:
        alloc = allocate_budget(["only"], 500)
        assert alloc.per_query == [500]
        assert alloc.total == 500


# ---------------------------------------------------------------------------
# weighted strategy
# ---------------------------------------------------------------------------


class TestWeightedStrategy:
    def test_proportional_split(self) -> None:
        alloc = allocate_budget(
            ["a", "b"], 1000, strategy="weighted", weights=[3.0, 1.0]
        )
        assert alloc.per_query == [750, 250]
        assert alloc.total == 1000
        assert alloc.strategy == "weighted"

    def test_missing_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights argument"):
            allocate_budget(["a", "b"], 100, strategy="weighted")

    def test_wrong_length_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            allocate_budget(
                ["a", "b", "c"], 100, strategy="weighted", weights=[1.0, 2.0]
            )

    def test_negative_weights_clamped(self) -> None:
        # -5 clamps to 0, leaving [0, 1, 1] → split 100 across the
        # last two queries.
        alloc = allocate_budget(
            ["a", "b", "c"], 100,
            strategy="weighted",
            weights=[-5.0, 1.0, 1.0],
        )
        assert alloc.per_query[0] == 0
        assert alloc.per_query[1] + alloc.per_query[2] == 100

    def test_all_zero_weights_falls_back_to_uniform(self) -> None:
        alloc = allocate_budget(
            ["a", "b"], 100, strategy="weighted", weights=[0.0, 0.0]
        )
        assert alloc.per_query == [50, 50]
        assert any("fell back to uniform" in note for note in alloc.notes)

    def test_all_negative_weights_falls_back_to_uniform(self) -> None:
        """Negatives clamp to 0; if every weight clamps, fall back."""
        alloc = allocate_budget(
            ["a", "b"], 100, strategy="weighted", weights=[-1.0, -2.0]
        )
        assert alloc.per_query == [50, 50]
        assert any("fell back to uniform" in note for note in alloc.notes)


# ---------------------------------------------------------------------------
# length strategy
# ---------------------------------------------------------------------------


class TestLengthStrategy:
    def test_longer_queries_get_more(self) -> None:
        alloc = allocate_budget(
            ["short", "this query is significantly longer"],
            1000,
            strategy="length",
        )
        # Longer query must receive a strictly larger budget.
        assert alloc.per_query[1] > alloc.per_query[0]
        assert sum(alloc.per_query) == 1000

    def test_custom_weight_fn(self) -> None:
        def equal_thirds(_: list[str]) -> list[float]:
            return [1.0, 1.0, 1.0]

        alloc = allocate_budget(
            ["a", "b", "c"], 300,
            strategy="length",
            weight_fn=equal_thirds,
        )
        assert alloc.per_query == [100, 100, 100]

    def test_custom_weight_fn_wrong_length_raises(self) -> None:
        def bad_fn(_: list[str]) -> list[float]:
            return [1.0]

        with pytest.raises(ValueError, match="wrong number of weights"):
            allocate_budget(
                ["a", "b"], 100, strategy="length", weight_fn=bad_fn,
            )

    def test_custom_weight_fn_negatives_clamped(self) -> None:
        def neg_fn(_: list[str]) -> list[float]:
            return [-1.0, 1.0]

        alloc = allocate_budget(
            ["a", "b"], 100, strategy="length", weight_fn=neg_fn,
        )
        assert alloc.per_query == [0, 100]


# ---------------------------------------------------------------------------
# unknown strategy
# ---------------------------------------------------------------------------


class TestUnknownStrategy:
    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            allocate_budget(
                ["a"], 100, strategy="bogus",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Largest-remainder distribution
# ---------------------------------------------------------------------------


class TestRemainderDistribution:
    def test_remainder_zero_no_redistribution(self) -> None:
        """When the proportional floats are already exact integers,
        the largest-remainder branch shouldn't fire."""
        alloc = allocate_budget(["a", "b"], 100, strategy="weighted",
                                weights=[1.0, 1.0])
        assert alloc.per_query == [50, 50]
        # Each entry was exactly 50.0 → fraction 0.0 → no remainder.

    def test_remainder_goes_to_largest_fraction(self) -> None:
        # weights produce fractions: 7/3=2.33, 7/3=2.33, 7/3=2.33; remainder 1.
        alloc = allocate_budget(
            ["a", "b", "c"], 7, strategy="weighted",
            weights=[1.0, 1.0, 1.0],
        )
        # Sum equals total.
        assert sum(alloc.per_query) == 7

    def test_distinct_fractions(self) -> None:
        # 100 split [1,2,3]/6 → 16.67, 33.33, 50.00 → floors [16,33,50]=99
        # remainder 1 goes to the entry with largest fraction (33.33 == 0.33,
        # 16.67 == 0.67, 50.00 == 0.00) → entry 0 wins.
        alloc = allocate_budget(
            ["a", "b", "c"], 100, strategy="weighted",
            weights=[1.0, 2.0, 3.0],
        )
        assert sum(alloc.per_query) == 100
        assert alloc.per_query == [17, 33, 50]


# ---------------------------------------------------------------------------
# min_per_query floor enforcement
# ---------------------------------------------------------------------------


class TestMinPerQuery:
    def test_floor_not_triggered(self) -> None:
        # Every entry already ≥ 10, no borrowing happens.
        alloc = allocate_budget(["a", "b"], 1000, min_per_query=10)
        assert alloc.per_query == [500, 500]
        assert alloc.notes == []

    def test_floor_borrows_from_largest(self) -> None:
        # weighted [10, 1, 1] across total=100:
        # raw=[83.33, 8.33, 8.33], floored=[83, 8, 8], rem=1 → [83, 9, 8].
        # min=15 → deficits=[0, 6, 7], total_deficit=13.
        # After lifting: [83, 15, 15]. Borrow 13 from largest (83).
        # Final: [70, 15, 15] → sum 100.
        alloc = allocate_budget(
            ["a", "b", "c"], 100,
            strategy="weighted", weights=[10.0, 1.0, 1.0],
            min_per_query=15,
        )
        assert min(alloc.per_query) >= 15
        assert sum(alloc.per_query) == 100
        assert alloc.per_query[0] == 70
        assert alloc.per_query[1] == 15 or alloc.per_query[2] == 15

    def test_floor_forces_overflow(self) -> None:
        # min=5, n=3, total=10 → impossible to keep sum ≤ total.
        alloc = allocate_budget(
            ["a", "b", "c"], 10, min_per_query=5,
        )
        assert alloc.per_query == [5, 5, 5]
        assert alloc.total == 15  # floor wins, total inflates.
        assert any("floor forces overflow" in note for note in alloc.notes)

    def test_floor_break_on_zero_surplus(self) -> None:
        """When the borrowing loop satisfies surplus mid-iteration,
        the ``break`` on ``surplus <= 0`` must fire (covers that
        line)."""
        # weighted [100, 1] across 200: raw=[198, 1.98]. floored=[198, 1].
        # rem=1 → fractions=[(0, 0.01), (1, 0.98)] → entry 1 gets +1.
        # → [198, 2]. min=10 → deficits=[0, 8]. Lift: [198, 10].
        # ordered DESC: [0, 1]. i=0: takeable=188, take=min(188,8)=8.
        # floored[0]=190, surplus=0. break fires.
        alloc = allocate_budget(
            ["a", "b"], 200,
            strategy="weighted", weights=[100.0, 1.0],
            min_per_query=10,
        )
        assert sum(alloc.per_query) == 200
        assert min(alloc.per_query) == 10

    def test_floor_skips_freshly_lifted_entries(self) -> None:
        """Entries lifted to the floor must be skipped by the
        ``floored[i] <= min_per_query`` guard so we don't oscillate."""
        # 4 entries, [10, 1, 1, 1] across 100: raw=[76.9, 7.7, 7.7, 7.7],
        # floored=[76, 7, 7, 7] → rem=3, distribute: [76, 8, 8, 8].
        # min=20 → deficits=[0, 12, 12, 12], total_deficit=36.
        # Lift: [76, 20, 20, 20]. Take 36 from the largest (76).
        # → [40, 20, 20, 20] → sum 100.
        alloc = allocate_budget(
            ["a", "b", "c", "d"], 100,
            strategy="weighted", weights=[10.0, 1.0, 1.0, 1.0],
            min_per_query=20,
        )
        assert alloc.per_query == [40, 20, 20, 20]
        assert sum(alloc.per_query) == 100


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import budget_allocator as mod
        assert set(mod.__all__) == {
            "AllocationStrategy",
            "BudgetAllocation",
            "allocate_budget",
        }
