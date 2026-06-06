"""Unit tests for ``stropha.retrieval.packing`` (WIN-208).

Token-budget result packing turns "give me top_k chunks" into "give
me as much as fits in N tokens". Tests cover the greedy-first-fit
algorithm + the spill / truncation / first-hit-oversize edge cases.
"""

from __future__ import annotations

import pytest

from stropha.models import SearchHit
from stropha.retrieval.packing import (
    pack_to_budget,
)


def _hit(content: str, rank: int = 0, score: float = 1.0) -> SearchHit:
    """Build a minimal SearchHit whose snippet drives the token cost."""
    return SearchHit(
        chunk_id=f"chunk-{rank}",
        rel_path=f"file_{rank}.py",
        start_line=1,
        end_line=10,
        score=score,
        rank=rank,
        snippet=content,
        language="python",
        kind="function",
        symbol=None,
    )


def _fixed_estimator(target_per_word: int = 1):
    """Build a deterministic estimator: token count = word count."""
    return lambda s: len(s.split())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPackToBudgetHappyPath:
    def test_all_hits_fit_under_budget(self) -> None:
        hits = [_hit("a b c", rank=i) for i in range(3)]  # 3 tokens each
        result = pack_to_budget(hits, 100, estimator=_fixed_estimator())
        assert len(result.packed) == 3
        assert result.spilled == []
        assert result.total_tokens == 9
        assert result.truncated is False

    def test_packs_until_budget_then_stops(self) -> None:
        """Greedy first-fit: 5 hits at 3 tokens each, budget 9 → 3 fit."""
        hits = [_hit("a b c", rank=i) for i in range(5)]
        result = pack_to_budget(hits, 9, estimator=_fixed_estimator())
        assert len(result.packed) == 3
        assert len(result.spilled) == 2
        assert result.total_tokens == 9
        assert result.budget_tokens == 9

    def test_preserves_score_order(self) -> None:
        """The packer must not re-sort — input rank order is the
        score-ranked goodness signal."""
        hits = [
            _hit("aaa", rank=0, score=0.9),
            _hit("bbb", rank=1, score=0.7),
            _hit("ccc", rank=2, score=0.5),
        ]
        result = pack_to_budget(hits, 100, estimator=_fixed_estimator())
        assert [h.rank for h in result.packed] == [0, 1, 2]

    def test_score_order_preserved_under_partial_pack(self) -> None:
        """Same invariant under partial packing — we drop the tail,
        not random items."""
        hits = [
            _hit("a", rank=0, score=0.9),    # 1 token
            _hit("b c", rank=1, score=0.7),  # 2 tokens
            _hit("d", rank=2, score=0.5),    # 1 token
        ]
        result = pack_to_budget(hits, 2, estimator=_fixed_estimator())
        # Budget=2: hit 0 (1 token) fits; hit 1 (2 tokens) would push
        # to 3 — exceeds → stop. Hit 2 NEVER tried (monotonic).
        assert [h.rank for h in result.packed] == [0]
        # Spilled retains the rest, including the never-attempted tail.
        assert [h.rank for h in result.spilled] == [1, 2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPackToBudgetEdges:
    def test_empty_hits_returns_empty_pack(self) -> None:
        result = pack_to_budget([], 100, estimator=_fixed_estimator())
        assert result.packed == []
        assert result.spilled == []
        assert result.total_tokens == 0

    def test_zero_budget_returns_empty_pack_no_escape(self) -> None:
        """A zero budget is a legitimate 'do nothing' request — no
        escape hatch. All hits spill."""
        hits = [_hit("x", rank=i) for i in range(3)]
        result = pack_to_budget(hits, 0, estimator=_fixed_estimator())
        assert result.packed == []
        assert len(result.spilled) == 3
        assert result.total_tokens == 0
        assert result.truncated is False

    def test_negative_budget_returns_empty_pack(self) -> None:
        """Negative budgets are caller error but must not crash; treat
        as zero-budget."""
        hits = [_hit("x", rank=0)]
        result = pack_to_budget(hits, -100, estimator=_fixed_estimator())
        assert result.packed == []
        assert result.budget_tokens == 0

    def test_first_hit_oversize_returned_anyway_with_truncated_flag(
        self,
    ) -> None:
        """Escape hatch: if the highest-rank hit alone exceeds the
        budget, include it so the caller sees something. Mark
        truncated=True so the UI can warn."""
        hits = [
            _hit("aaa bbb ccc ddd eee", rank=0),  # 5 tokens, budget 3
            _hit("f", rank=1),                     # would fit
        ]
        result = pack_to_budget(hits, 3, estimator=_fixed_estimator())
        assert len(result.packed) == 1
        assert result.packed[0].rank == 0
        assert result.truncated is True
        # Spilled retains the rest of the input order.
        assert [h.rank for h in result.spilled] == [1]
        # total_tokens accurately reports the cost of what we packed,
        # even though it exceeds the budget.
        assert result.total_tokens == 5

    def test_first_hit_exactly_at_budget_packs_normally(self) -> None:
        """Boundary: cost == budget is in (≤ is the strict comparison)."""
        hits = [_hit("a b c", rank=0)]
        result = pack_to_budget(hits, 3, estimator=_fixed_estimator())
        assert len(result.packed) == 1
        assert result.truncated is False

    def test_overflow_stops_does_not_skip_to_smaller_later_hits(
        self,
    ) -> None:
        """Monotonic ordering: once a hit overflows, STOP. Do not
        rummage in the tail for a smaller hit that would fit — that
        would re-order results and break the agent's attention."""
        hits = [
            _hit("a b c", rank=0),       # 3 tokens
            _hit("d e f g h", rank=1),    # 5 tokens — would overflow
            _hit("i", rank=2),            # 1 token — would fit, but skipped
        ]
        result = pack_to_budget(hits, 4, estimator=_fixed_estimator())
        # Budget 4: hit 0 packs (total 3), hit 1 would push to 8 → stop.
        # Hit 2 is NOT considered.
        assert [h.rank for h in result.packed] == [0]
        assert [h.rank for h in result.spilled] == [1, 2]

    def test_empty_snippet_costs_zero(self) -> None:
        """A SearchHit with empty snippet costs 0 tokens — packs for free."""
        hits = [_hit("", rank=i) for i in range(5)]
        result = pack_to_budget(hits, 1, estimator=_fixed_estimator())
        assert len(result.packed) == 5
        assert result.total_tokens == 0

    def test_negative_estimator_clamped_to_zero(self) -> None:
        """A misbehaving estimator returning a negative number can't
        re-credit the budget — clamp at 0."""
        hits = [_hit("anything", rank=0)]
        result = pack_to_budget(
            hits, 1, estimator=lambda _: -999,
        )
        assert len(result.packed) == 1
        assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# Estimator parameterisation
# ---------------------------------------------------------------------------


class TestPackToBudgetEstimator:
    def test_default_estimator_used_when_not_passed(self) -> None:
        """Default ``estimate_content_tokens`` is the savings module's
        word+punctuation heuristic. We verify the no-arg call works."""
        hits = [_hit("hello world foo bar", rank=0)]
        result = pack_to_budget(hits, 1000)  # default estimator
        assert len(result.packed) == 1
        # estimate_content_tokens for "hello world foo bar" is ~5
        # (4 words × 1.2 multiplier, rounded). Just assert it's >0
        # so we can verify the default actually ran.
        assert result.total_tokens > 0

    def test_custom_estimator_drives_packing(self) -> None:
        """The packer must use the passed estimator, not the default."""
        hits = [_hit("anything", rank=0)]
        # Pretend each hit is exactly 50 tokens.
        result = pack_to_budget(hits, 100, estimator=lambda _: 50)
        assert len(result.packed) == 1
        assert result.total_tokens == 50


# ---------------------------------------------------------------------------
# PackingResult dataclass
# ---------------------------------------------------------------------------


class TestPackingResultDataclass:
    def test_frozen_dataclass_rejects_mutation(self) -> None:
        """Immutable contract — callers must not patch the result."""
        result = pack_to_budget([], 100, estimator=_fixed_estimator())
        with pytest.raises((AttributeError, TypeError)):
            result.packed = []  # type: ignore[misc]

    def test_default_fields(self) -> None:
        """PackingResult populated by `pack_to_budget` always has
        every field, even on the empty/empty/zero path."""
        result = pack_to_budget([], 100, estimator=_fixed_estimator())
        assert isinstance(result.packed, list)
        assert isinstance(result.spilled, list)
        assert isinstance(result.total_tokens, int)
        assert isinstance(result.budget_tokens, int)
        assert isinstance(result.truncated, bool)


# ---------------------------------------------------------------------------
# Integration with SearchEngine
# ---------------------------------------------------------------------------


class TestSearchEngineContextBudget:
    """SearchEngine.search exposes the budget via a kwarg.

    These integration tests mock the embedder + storage so we can
    drive the engine without indexing real data — the goal is to
    prove the budget kwarg flows into pack_to_budget at the right
    place in the pipeline (AFTER recursive merge).
    """

    def _make_engine(self, hit_snippets: list[str]):
        from unittest.mock import MagicMock

        from stropha.retrieval.search import SearchEngine

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage = MagicMock()
        # Dense returns the canned hits in score order; sparse/symbol
        # return empty so the single-stream short-circuit fires (no
        # RRF fusion needed for the test).
        storage.search_dense.return_value = [
            _hit(snippet, rank=i, score=1.0 - i * 0.01)
            for i, snippet in enumerate(hit_snippets)
        ]
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        return SearchEngine(storage, embedder)

    def test_search_without_budget_returns_full_top_k(self) -> None:
        eng = self._make_engine(["a b c"] * 5)
        results = eng.search("foo", top_k=5)
        assert len(results) == 5

    def test_search_with_budget_packs(self) -> None:
        """5 hits × 3 tokens each, budget 9 → only 3 fit."""
        eng = self._make_engine(["a b c"] * 5)
        results = eng.search("foo", top_k=5, context_budget=9)
        # With default estimator, "a b c" = ~4 tokens (3 words × 1.2);
        # depending on rounding, the pack may have 2-3 entries. The
        # contract is "less than 5 and never overflow"; assert that.
        assert len(results) < 5
        assert len(results) >= 1

    def test_search_with_zero_budget_passes_through(self) -> None:
        """``context_budget=0`` is a no-op — the packer is bypassed.
        Confirms _maybe_pack short-circuits on non-positive budgets."""
        eng = self._make_engine(["a b c"] * 5)
        results = eng.search("foo", top_k=5, context_budget=0)
        assert len(results) == 5
