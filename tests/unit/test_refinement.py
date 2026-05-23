"""Unit tests for the local refinement sub-agent (WIN-239, EPIC 1.10 #30)."""

from __future__ import annotations

from typing import Any

import pytest

from stropha.retrieval.refinement import (
    DOMINANCE_THRESHOLD,
    RefinementResult,
    RefinementStep,
    SPARSE_THRESHOLD,
    _dominant_file,
    _has_code_anchor,
    _has_tests,
    _is_test_path,
    _merge_unique,
    _safe_search,
    _split_compound,
    refine_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Hit:
    def __init__(
        self,
        chunk_id: str = "c1",
        symbol: str | None = None,
        kind: str | None = None,
        rel_path: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.symbol = symbol
        self.kind = kind
        self.rel_path = rel_path


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_step_defaults(self) -> None:
        s = RefinementStep(gap="x", query="q")
        assert s.added_chunk_ids == []
        assert s.reason == ""

    def test_result_defaults(self) -> None:
        r = RefinementResult()
        assert r.hits == []
        assert r.steps == []
        assert r.notes == []

    def test_step_frozen(self) -> None:
        s = RefinementStep(gap="x", query="q")
        with pytest.raises((AttributeError, Exception)):
            s.gap = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Path / kind helpers
# ---------------------------------------------------------------------------


class TestIsTestPath:
    @pytest.mark.parametrize("path", [
        "tests/test_foo.py",
        "src/foo_test.go",
        "x.spec.ts",
        "y.test.js",
        "module/tests/case.py",
        "module/test/case.py",
        "src/TEST_BIG.py",  # case-insensitive
    ])
    def test_test_paths(self, path: str) -> None:
        assert _is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "src/foo.py",
        "lib/bar.go",
        "",
        "specifications.md",  # 'spec' is part of a word, not '.spec.'
    ])
    def test_non_test_paths(self, path: str) -> None:
        assert _is_test_path(path) is False


class TestHasTests:
    def test_empty(self) -> None:
        assert _has_tests([]) is False

    def test_one_test_hit(self) -> None:
        assert _has_tests(
            [_Hit(rel_path="src/foo.py"), _Hit(rel_path="tests/test_foo.py")]
        ) is True

    def test_no_test_hits(self) -> None:
        assert _has_tests([_Hit(rel_path="src/foo.py")]) is False


class TestHasCodeAnchor:
    def test_returns_first_code_symbol(self) -> None:
        hits = [
            _Hit(kind="doc", symbol="DocHeading"),
            _Hit(kind="class", symbol="FooClass"),
            _Hit(kind="function", symbol="bar"),
        ]
        assert _has_code_anchor(hits) == "FooClass"

    def test_none_when_no_code_hit(self) -> None:
        assert _has_code_anchor([_Hit(kind="doc", symbol="x")]) is None

    def test_none_when_code_hit_lacks_symbol(self) -> None:
        assert _has_code_anchor([_Hit(kind="class", symbol=None)]) is None


class TestDominantFile:
    def test_empty(self) -> None:
        assert _dominant_file([]) is None

    def test_no_paths(self) -> None:
        assert _dominant_file([_Hit(rel_path=None)]) is None

    def test_single_file_dominance(self) -> None:
        hits = [
            _Hit(chunk_id=str(i), rel_path="src/x.py") for i in range(4)
        ]
        assert _dominant_file(hits) == "src/x.py"

    def test_below_threshold(self) -> None:
        # 2 in x.py, 2 in y.py — 0.5 < 0.75 threshold.
        hits = [
            _Hit(chunk_id="a", rel_path="src/x.py"),
            _Hit(chunk_id="b", rel_path="src/x.py"),
            _Hit(chunk_id="c", rel_path="src/y.py"),
            _Hit(chunk_id="d", rel_path="src/y.py"),
        ]
        assert _dominant_file(hits) is None


# ---------------------------------------------------------------------------
# _safe_search
# ---------------------------------------------------------------------------


class TestSafeSearch:
    def test_normal_call(self) -> None:
        out, err = _safe_search(lambda q: [_Hit(chunk_id="ok")], "q")
        assert err is None
        assert len(out) == 1

    def test_returns_none_treated_as_empty(self) -> None:
        out, err = _safe_search(lambda q: None, "q")  # type: ignore[return-value]
        assert err is None
        assert out == []

    def test_exception_captured_as_note(self) -> None:
        def boom(q: str) -> list[Any]:
            raise RuntimeError("nope")

        out, err = _safe_search(boom, "q")
        assert out == []
        assert err is not None
        assert "nope" in err


# ---------------------------------------------------------------------------
# _split_compound
# ---------------------------------------------------------------------------


class TestSplitCompound:
    def test_and_splits(self) -> None:
        assert _split_compound("foo and bar") == ["foo", "bar"]

    def test_multiple_tokens(self) -> None:
        out = _split_compound("a and b then c plus d")
        assert out == ["a", "b", "c", "d"]

    def test_ampersand(self) -> None:
        assert _split_compound("x & y") == ["x", "y"]

    def test_no_splitter_returns_original(self) -> None:
        assert _split_compound("plain query") == ["plain query"]

    def test_empty_query_returns_original(self) -> None:
        assert _split_compound("") == [""]

    def test_dedupes_repeated_clauses(self) -> None:
        # "foo and foo" splits to two "foo" clauses → dedupe to one.
        assert _split_compound("foo and foo") == ["foo"]

    def test_case_insensitive_match_preserves_case(self) -> None:
        out = _split_compound("Alpha AND Beta")
        assert out == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# _merge_unique
# ---------------------------------------------------------------------------


class TestMergeUnique:
    def test_appends_new_ids(self) -> None:
        target: list[Any] = [_Hit(chunk_id="a")]
        seen = {"a"}
        added = _merge_unique(target, [_Hit(chunk_id="b"), _Hit(chunk_id="c")], seen)
        assert added == ["b", "c"]
        assert len(target) == 3
        assert seen == {"a", "b", "c"}

    def test_skips_seen_ids(self) -> None:
        target: list[Any] = [_Hit(chunk_id="a")]
        seen = {"a"}
        added = _merge_unique(target, [_Hit(chunk_id="a"), _Hit(chunk_id="b")], seen)
        assert added == ["b"]

    def test_skips_empty_ids(self) -> None:
        target: list[Any] = []
        seen: set[str] = set()
        added = _merge_unique(target, [_Hit(chunk_id="")], seen)
        assert added == []
        assert target == []


# ---------------------------------------------------------------------------
# refine_results — gap detection
# ---------------------------------------------------------------------------


def _make_search_fn(
    by_query: dict[str, list[Any]],
) -> Any:
    def _fn(q: str) -> list[Any]:
        return by_query.get(q, [])
    return _fn


class TestRefineResultsGaps:
    def test_max_steps_zero_returns_originals(self) -> None:
        hits = [_Hit(chunk_id="a")]
        result = refine_results(
            "q", hits, search_fn=lambda q: [], max_steps=0,
        )
        assert result.hits == hits
        assert result.steps == []
        assert any("max_steps" in n for n in result.notes)

    def test_sparse_triggers_hyde_retry(self) -> None:
        search_fn = _make_search_fn({
            "q (HyDE)": [_Hit(chunk_id="hyde-1"), _Hit(chunk_id="hyde-2")],
        })
        result = refine_results(
            "q", [_Hit(chunk_id="a")],
            search_fn=search_fn,
        )
        assert any(s.gap == "sparse" for s in result.steps)
        sparse_step = next(s for s in result.steps if s.gap == "sparse")
        assert sparse_step.added_chunk_ids == ["hyde-1", "hyde-2"]

    def test_sparse_skipped_when_above_threshold(self) -> None:
        hits = [_Hit(chunk_id=str(i)) for i in range(SPARSE_THRESHOLD)]
        result = refine_results(
            "q", hits, search_fn=lambda q: [],
        )
        assert all(s.gap != "sparse" for s in result.steps)

    def test_sparse_skipped_when_query_empty(self) -> None:
        result = refine_results(
            "", [_Hit(chunk_id="a")], search_fn=lambda q: [],
        )
        assert all(s.gap != "sparse" for s in result.steps)

    def test_no_tests_triggers_tests_for_retry(self) -> None:
        anchor_hits = [_Hit(
            chunk_id="c1", symbol="FooClass", kind="class",
            rel_path="src/foo.py",
        )]
        search_fn = _make_search_fn({
            "tests for FooClass": [_Hit(
                chunk_id="t1", rel_path="tests/test_foo.py",
            )],
        })
        result = refine_results(
            "FooClass", anchor_hits, search_fn=search_fn,
            max_steps=5,  # avoid sparse from consuming all the budget
        )
        gaps = [s.gap for s in result.steps]
        assert "no_tests" in gaps

    def test_no_tests_gap_label_without_graph(self) -> None:
        anchor_hits = [
            _Hit(chunk_id="a", symbol="Foo", kind="class", rel_path="src/a.py"),
            _Hit(chunk_id="b", symbol="Foo", kind="class", rel_path="src/b.py"),
            _Hit(chunk_id="c", symbol="Foo", kind="class", rel_path="src/c.py"),
        ]
        result = refine_results(
            "Foo", anchor_hits,
            search_fn=lambda q: [],
            has_graph=False,
        )
        gaps = [s.gap for s in result.steps]
        assert "no_tests_no_graph" in gaps

    def test_tests_already_present_no_retry(self) -> None:
        # Use `test_` prefix so _is_test_path recognises it.
        hits = [
            _Hit(chunk_id="src", symbol="Foo", kind="class", rel_path="src/foo.py"),
            _Hit(chunk_id="test", rel_path="tests/test_foo.py"),
            _Hit(chunk_id="x", rel_path="src/x.py"),
        ]
        result = refine_results("Foo", hits, search_fn=lambda q: [])
        assert all(s.gap not in {"no_tests", "no_tests_no_graph"}
                   for s in result.steps)

    def test_single_file_dominance_triggers_references(self) -> None:
        hits = [_Hit(
            chunk_id=str(i), symbol="Foo", kind="class",
            rel_path="src/foo.py",
        ) for i in range(4)]
        # `test_t.py` matches `test_` so _has_tests returns True
        # and the no_tests branch is suppressed.
        search_fn = _make_search_fn({
            "tests for Foo": [_Hit(chunk_id="t", rel_path="tests/test_t.py")],
            "references to Foo": [_Hit(chunk_id="ref-1")],
        })
        result = refine_results(
            "Foo", hits, search_fn=search_fn, max_steps=5,
        )
        gaps = [s.gap for s in result.steps]
        assert "single_file_dominance" in gaps

    def test_compound_unresolved_runs_per_clause(self) -> None:
        # Many hits + a real test file → no sparse, no dominance,
        # no no_tests; only compound should fire.
        hits = [
            _Hit(chunk_id=str(i), symbol="Foo", kind="class",
                 rel_path=f"src/{i}.py")
            for i in range(5)
        ] + [_Hit(chunk_id="t", rel_path="tests/test_foo.py")]
        search_fn = _make_search_fn({
            "foo": [_Hit(chunk_id="foo-hit")],
            "bar": [_Hit(chunk_id="bar-hit")],
        })
        result = refine_results(
            "foo and bar", hits, search_fn=search_fn, max_steps=5,
        )
        compound_gaps = [s for s in result.steps
                         if s.gap == "compound_unresolved"]
        # One per clause.
        assert len(compound_gaps) == 2
        queries = {s.query for s in compound_gaps}
        assert queries == {"foo", "bar"}

    def test_max_steps_cap_honoured(self) -> None:
        hits = [_Hit(chunk_id="a")]
        search_fn = _make_search_fn({
            "q (HyDE)": [_Hit(chunk_id="h")],
        })
        # Only one step allowed; sparse consumes it; compound branch
        # never executes even though "and" is present.
        result = refine_results(
            "x and y", hits, search_fn=search_fn, max_steps=1,
        )
        assert len(result.steps) == 1

    def test_search_fn_exception_captured(self) -> None:
        def boom(q: str) -> list[Any]:
            raise RuntimeError("network down")

        result = refine_results(
            "q", [_Hit(chunk_id="a")], search_fn=boom,
        )
        # The sparse step still records (zero adds) and a note explains.
        assert any("network down" in n for n in result.notes)

    def test_compound_step_cap_mid_clause(self) -> None:
        """When max_steps is reached part-way through the compound
        clause loop, the inner break must fire."""
        # 5 class hits + a real test file → only compound branch fires.
        hits = [_Hit(chunk_id=str(i), symbol="Foo", kind="class",
                     rel_path=f"src/{i}.py") for i in range(5)
                ] + [_Hit(chunk_id="t", rel_path="tests/test_t.py")]
        search_fn = _make_search_fn({"a": [], "b": [], "c": []})
        # max_steps=2 → 3 clauses, only 2 should run.
        result = refine_results(
            "a and b and c", hits, search_fn=search_fn, max_steps=2,
        )
        compound = [s for s in result.steps if s.gap == "compound_unresolved"]
        assert len(compound) == 2

    def test_exception_in_no_tests_branch_captured(self) -> None:
        """search_fn raising on the no_tests re-query must surface as
        a note (covers the deeper notes-append branch)."""
        def selective_boom(q: str) -> list[Any]:
            if q == "tests for Foo":
                raise RuntimeError("no_tests boom")
            return []

        # 3+ hits to skip sparse; code anchor present; no test file.
        hits = [
            _Hit(chunk_id="a", symbol="Foo", kind="class", rel_path="src/a.py"),
            _Hit(chunk_id="b", symbol="Foo", kind="class", rel_path="src/b.py"),
            _Hit(chunk_id="c", symbol="Foo", kind="class", rel_path="src/c.py"),
        ]
        result = refine_results("Foo", hits, search_fn=selective_boom)
        assert any("no_tests boom" in n for n in result.notes)

    def test_exception_in_dominance_branch_captured(self) -> None:
        def selective_boom(q: str) -> list[Any]:
            if q == "references to Foo":
                raise RuntimeError("dom boom")
            if q == "tests for Foo":
                # Return a test hit so no_tests branch is satisfied
                # and dominance can still fire.
                return [_Hit(chunk_id="t", rel_path="tests/test_t.py")]
            return []

        hits = [_Hit(
            chunk_id=str(i), symbol="Foo", kind="class",
            rel_path="src/foo.py",
        ) for i in range(4)]
        result = refine_results(
            "Foo", hits, search_fn=selective_boom, max_steps=5,
        )
        assert any("dom boom" in n for n in result.notes)

    def test_exception_in_compound_branch_captured(self) -> None:
        def selective_boom(q: str) -> list[Any]:
            if q == "foo":
                raise RuntimeError("compound boom")
            return []

        hits = [
            _Hit(chunk_id=str(i), symbol="Foo", kind="class",
                 rel_path=f"src/{i}.py")
            for i in range(5)
        ] + [_Hit(chunk_id="t", rel_path="tests/test_t.py")]
        result = refine_results(
            "foo and bar", hits, search_fn=selective_boom, max_steps=5,
        )
        assert any("compound boom" in n for n in result.notes)


# ---------------------------------------------------------------------------
# refine_results — merging behaviour
# ---------------------------------------------------------------------------


class TestMerging:
    def test_originals_preserved_first(self) -> None:
        originals = [_Hit(chunk_id="orig-1"), _Hit(chunk_id="orig-2")]
        search_fn = _make_search_fn({
            "q (HyDE)": [_Hit(chunk_id="new-1")],
        })
        result = refine_results("q", originals, search_fn=search_fn)
        # Order: originals first, then HyDE add.
        ids = [_h.chunk_id for _h in result.hits]
        assert ids[0:2] == ["orig-1", "orig-2"]
        assert "new-1" in ids[2:]

    def test_duplicates_not_added(self) -> None:
        originals = [_Hit(chunk_id="dup")]
        search_fn = _make_search_fn({
            "q (HyDE)": [_Hit(chunk_id="dup")],
        })
        result = refine_results("q", originals, search_fn=search_fn)
        ids = [_h.chunk_id for _h in result.hits]
        assert ids.count("dup") == 1


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import refinement as mod
        assert set(mod.__all__) == {
            "DOMINANCE_THRESHOLD",
            "RefinementResult",
            "RefinementStep",
            "SPARSE_THRESHOLD",
            "refine_results",
        }
