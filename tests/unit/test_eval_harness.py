"""Unit tests for ``stropha.eval.harness`` — golden loader, Recall@K, MRR, NDCG.

We mock the SearchEngine to avoid touching real embedders/storage. The
harness must work with anything that exposes ``search(query, top_k) ->
list[SearchHit]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from stropha.eval import EvalCase, EvalResult, load_golden, run_eval
from stropha.eval.harness import _dcg, _ndcg

# --------------------------------------------------------------------------- mock


@dataclass
class _FakeHit:
    path: str
    symbol: str | None
    score: float = 1.0


class _FakeEngine:
    """Returns canned hits per query string. Works as a SearchEngine stub."""

    def __init__(self, by_query: dict[str, list[_FakeHit]]) -> None:
        self._by_query = by_query

    def search(self, query: str, *, top_k: int = 10) -> list[_FakeHit]:
        return self._by_query.get(query, [])[:top_k]


# --------------------------------------------------------------------------- load_golden


def test_load_golden_skips_blank_and_comments(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(
        "\n"
        "# header comment\n"
        '{"id":"a","query":"foo","expected_paths":["x.py"]}\n'
        "\n"
        '{"id":"b","query":"bar","expected_symbols":["Bar"]}\n',
        encoding="utf-8",
    )
    cases = load_golden(f)
    assert len(cases) == 2
    assert cases[0].id == "a"
    assert cases[0].expected_paths == ("x.py",)
    assert cases[1].expected_symbols == ("Bar",)


def test_load_golden_invalid_json_raises_with_lineno(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(
        '{"id":"good","query":"x"}\n'
        '{ this is broken }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2:"):
        load_golden(f)


def test_load_golden_missing_query_raises(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text('{"id":"x","expected_paths":["a"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required `query`"):
        load_golden(f)


# --------------------------------------------------------------------------- run_eval


def test_recall_when_path_matches_at_rank_1() -> None:
    cases = [EvalCase(id="c1", query="foo", expected_paths=("a.py",))]
    engine = _FakeEngine({"foo": [_FakeHit("a.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n == 1
    assert report.n_hits == 1
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0


def test_recall_zero_when_no_hit() -> None:
    cases = [EvalCase(id="c1", query="foo", expected_paths=("a.py",))]
    engine = _FakeEngine({"foo": [_FakeHit("z.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_path_match_is_suffix_aware() -> None:
    """``expected_paths=['fsrs.py']`` matches ``src/stropha/fsrs.py``."""
    cases = [EvalCase(id="c1", query="q", expected_paths=("fsrs.py",))]
    engine = _FakeEngine({"q": [_FakeHit("src/stropha/fsrs.py", None)]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n_hits == 1
    assert report.results[0].matched_path == "fsrs.py"


def test_symbol_match_dotted_suffix() -> None:
    """``expected_symbols=['login']`` matches ``AuthService.login``."""
    cases = [EvalCase(id="c1", query="q", expected_symbols=("login",))]
    engine = _FakeEngine({"q": [_FakeHit("a.py", "AuthService.login")]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.n_hits == 1
    assert report.results[0].matched_symbol == "login"


def test_mrr_is_first_relevant_rank() -> None:
    cases = [EvalCase(id="c1", query="q", expected_paths=("target.py",))]
    engine = _FakeEngine({"q": [
        _FakeHit("a.py", None),
        _FakeHit("b.py", None),
        _FakeHit("target.py", None),  # rank 3
    ]})
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.results[0].rank_of_first_hit == 3
    assert report.mrr == pytest.approx(1.0 / 3)


def test_top_k_cuts_off() -> None:
    cases = [EvalCase(id="c1", query="q", expected_paths=("target.py",))]
    engine = _FakeEngine({"q": [
        _FakeHit("a.py", None), _FakeHit("b.py", None), _FakeHit("target.py", None),
    ]})
    report = run_eval(cases, engine, top_k=2)  # type: ignore[arg-type]
    assert report.recall_at_k == 0.0


def test_aggregate_recall_across_cases() -> None:
    cases = [
        EvalCase(id="c1", query="q1", expected_paths=("a.py",)),
        EvalCase(id="c2", query="q2", expected_paths=("b.py",)),
        EvalCase(id="c3", query="q3", expected_paths=("c.py",)),
    ]
    engine = _FakeEngine({
        "q1": [_FakeHit("a.py", None)],   # hit
        "q2": [_FakeHit("z.py", None)],   # miss
        "q3": [_FakeHit("c.py", None)],   # hit
    })
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    assert report.recall_at_k == pytest.approx(2 / 3)


def test_by_tag_buckets() -> None:
    cases = [
        EvalCase(id="c1", query="q1", expected_paths=("a.py",), tags=("retrieval",)),
        EvalCase(id="c2", query="q2", expected_paths=("b.py",), tags=("retrieval",)),
        EvalCase(id="c3", query="q3", expected_paths=("c.py",), tags=("storage",)),
    ]
    engine = _FakeEngine({
        "q1": [_FakeHit("a.py", None)],
        "q2": [_FakeHit("z.py", None)],
        "q3": [_FakeHit("c.py", None)],
    })
    report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
    by_tag = report.by_tag()
    assert by_tag["retrieval"].recall_at_k == 0.5
    assert by_tag["storage"].recall_at_k == 1.0


# --------------------------------------------------------------------------- end-to-end with the shipped golden file


def test_shipped_golden_file_parses() -> None:
    """The default golden set must always parse cleanly."""
    p = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"
    if not p.is_file():
        pytest.skip("golden file not present")
    cases = load_golden(p)
    assert len(cases) >= 20  # we ship ~30
    # Every case has at least one expected_paths or expected_symbols entry
    for c in cases:
        assert c.expected_paths or c.expected_symbols, f"case {c.id} has no expectations"


# --------------------------------------------------------------------------- NDCG tests


class TestDCG:
    """Tests for DCG computation."""

    def test_dcg_perfect(self):
        # Perfect ranking: highest relevance first
        relevances = [3.0, 2.0, 1.0, 0.0]
        dcg = _dcg(relevances)
        # (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) = 7/1 + 3/1.58 + 1/2 = 7 + 1.89 + 0.5 = 9.39
        assert dcg > 9.0

    def test_dcg_zeros(self):
        relevances = [0.0, 0.0, 0.0]
        assert _dcg(relevances) == 0.0

    def test_dcg_with_cutoff(self):
        relevances = [3.0, 2.0, 1.0, 0.0]
        dcg_k2 = _dcg(relevances, k=2)
        dcg_full = _dcg(relevances)
        assert dcg_k2 < dcg_full  # Less gain with fewer positions


class TestNDCG:
    """Tests for NDCG computation."""

    def test_ndcg_perfect_ranking(self):
        # Perfect ranking: relevances already sorted descending
        relevances = [3.0, 2.0, 1.0, 0.0]
        ndcg = _ndcg(relevances)
        assert ndcg == pytest.approx(1.0)  # Perfect score

    def test_ndcg_worst_ranking(self):
        # Worst ranking: relevances in ascending order
        relevances = [0.0, 1.0, 2.0, 3.0]
        ndcg = _ndcg(relevances)
        assert ndcg < 1.0  # Not perfect

    def test_ndcg_all_zeros(self):
        relevances = [0.0, 0.0, 0.0]
        assert _ndcg(relevances) == 0.0

    def test_ndcg_single_relevant(self):
        relevances = [1.0, 0.0, 0.0]
        ndcg = _ndcg(relevances)
        assert ndcg == pytest.approx(1.0)  # Only one relevant item, at top

    def test_ndcg_with_cutoff(self):
        relevances = [0.0, 0.0, 3.0, 0.0]  # Relevant item at rank 3
        ndcg_k2 = _ndcg(relevances, k=2)
        ndcg_k4 = _ndcg(relevances, k=4)
        assert ndcg_k2 == 0.0  # No relevant in top 2
        assert ndcg_k4 > 0.0  # Relevant item found


class TestEvalCaseRelevance:
    """Tests for graded relevance in EvalCase."""

    def test_from_dict_with_relevance_dict(self):
        d = {
            "id": "test",
            "query": "foo",
            "expected_paths": ["a.py"],
            "relevance": {"a.py": 3, "b.py": 1},
        }
        case = EvalCase.from_dict(d)
        assert case.relevance_map == {"a.py": 3.0, "b.py": 1.0}

    def test_from_dict_with_relevance_list(self):
        d = {
            "id": "test",
            "query": "foo",
            "expected_paths": ["a.py", "b.py"],
            "relevance": [3, 1],
        }
        case = EvalCase.from_dict(d)
        assert case.relevance_map["a.py"] == 3.0
        assert case.relevance_map["b.py"] == 1.0

    def test_get_relevance_with_map(self):
        case = EvalCase(
            id="test",
            query="foo",
            expected_paths=("a.py",),
            relevance_map={"a.py": 3.0},
        )
        assert case.get_relevance("a.py", None) == 3.0
        assert case.get_relevance("b.py", None) == 1.0  # Default

    def test_get_relevance_binary(self):
        case = EvalCase(
            id="test",
            query="foo",
            expected_paths=("a.py",),
        )
        # No relevance_map, so binary (1.0)
        assert case.get_relevance("a.py", None) == 1.0


class TestEvalResultNDCG:
    """Tests for NDCG in EvalResult."""

    def test_result_ndcg_property(self):
        case = EvalCase(id="t", query="q", expected_paths=("a.py",))
        result = EvalResult(
            case=case,
            rank_of_first_hit=1,
            top_k=5,
            relevances=(1.0, 0.0, 0.0, 0.0, 0.0),
        )
        assert result.ndcg == pytest.approx(1.0)

    def test_result_precision_property(self):
        case = EvalCase(id="t", query="q", expected_paths=("a.py", "b.py"))
        result = EvalResult(
            case=case,
            rank_of_first_hit=1,
            top_k=5,
            relevant_count=2,
        )
        assert result.precision == pytest.approx(0.4)  # 2/5


class TestEvalReportNDCG:
    """Tests for NDCG and Precision in EvalReport."""

    def test_report_ndcg_at_k(self):
        cases = [
            EvalCase(id="c1", query="q1", expected_paths=("a.py",)),
            EvalCase(id="c2", query="q2", expected_paths=("b.py",)),
        ]
        engine = _FakeEngine({
            "q1": [_FakeHit("a.py", None)],  # hit at rank 1
            "q2": [_FakeHit("x.py", None), _FakeHit("b.py", None)],  # hit at rank 2
        })
        report = run_eval(cases, engine, top_k=5)  # type: ignore[arg-type]
        
        # First case: perfect NDCG (relevant at rank 1)
        # Second case: lower NDCG (relevant at rank 2)
        assert report.ndcg_at_k > 0.5

    def test_report_precision_at_k(self):
        cases = [
            EvalCase(id="c1", query="q1", expected_paths=("a.py", "b.py")),
        ]
        engine = _FakeEngine({
            "q1": [_FakeHit("a.py", None), _FakeHit("b.py", None), _FakeHit("c.py", None)],
        })
        report = run_eval(cases, engine, top_k=5)  # type: ignore[arg-type]
        
        # 2 relevant out of 5 = 0.4
        assert report.precision_at_k == pytest.approx(0.4)

    def test_summary_includes_new_metrics(self):
        cases = [
            EvalCase(id="c1", query="q1", expected_paths=("a.py",)),
        ]
        engine = _FakeEngine({
            "q1": [_FakeHit("a.py", None)],
        })
        report = run_eval(cases, engine, top_k=10)  # type: ignore[arg-type]
        
        summary = report.summary()
        assert "ndcg@10" in summary
        assert "precision@10" in summary


# ---------------------------------------------------------------------------
# WIN-168 — split: search | holdout
# ---------------------------------------------------------------------------


class TestEvalSplit:
    """Anti-overfit split tagging on EvalCase + the filter/count helpers."""

    def test_legacy_case_defaults_to_search(self):
        """A golden line without a split field must NOT be silently
        treated as holdout — the proposer would lose half its training set."""
        case = EvalCase.from_dict({"query": "q1"})
        assert case.split == "search"

    def test_split_field_round_trip(self):
        case = EvalCase.from_dict({"query": "q1", "split": "holdout"})
        assert case.split == "holdout"

    def test_split_normalised_case_insensitive(self):
        case = EvalCase.from_dict({"query": "q1", "split": "HOLDOUT"})
        assert case.split == "holdout"

    def test_invalid_split_raises(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="invalid split"):
            EvalCase.from_dict({"query": "q1", "split": "train"})

    def test_filter_by_split_returns_only_matching(self):
        from stropha.eval import filter_by_split

        cases = [
            EvalCase(id="s1", query="q1", split="search"),
            EvalCase(id="h1", query="q2", split="holdout"),
            EvalCase(id="s2", query="q3", split="search"),
        ]
        assert [c.id for c in filter_by_split(cases, "search")] == ["s1", "s2"]
        assert [c.id for c in filter_by_split(cases, "holdout")] == ["h1"]

    def test_split_counts_reports_both_buckets(self):
        from stropha.eval import split_counts

        cases = [
            EvalCase(id="s1", query="q1", split="search"),
            EvalCase(id="s2", query="q2", split="search"),
            EvalCase(id="h1", query="q3", split="holdout"),
        ]
        counts = split_counts(cases)
        assert counts == {"search": 2, "holdout": 1}

    def test_load_golden_propagates_invalid_split_with_lineno(self, tmp_path):
        import pytest as _pytest

        from stropha.eval import load_golden

        f = tmp_path / "bad.jsonl"
        f.write_text(
            '{"id": "ok", "query": "ok"}\n'
            '{"id": "bad", "query": "q", "split": "train"}\n'
        )
        with _pytest.raises(ValueError, match=r"bad\.jsonl:2:.*invalid split"):
            load_golden(f)

    def test_real_golden_file_is_validly_split(self):
        """End-to-end: the shipped tests/eval/golden.jsonl loads and the
        70/30 split target holds (allowing for some slack)."""
        from pathlib import Path as _Path

        from stropha.eval import load_golden, split_counts

        golden = _Path(__file__).resolve().parent.parent.parent / "tests" / "eval" / "golden.jsonl"
        cases = load_golden(golden)
        counts = split_counts(cases)
        total = counts["search"] + counts["holdout"]
        assert total >= 25, f"expected >=25 golden cases, got {total}"
        # Holdout share between 20% and 40% — wide tolerance.
        holdout_pct = counts["holdout"] / total
        assert 0.20 <= holdout_pct <= 0.40, (
            f"holdout share {holdout_pct:.0%} outside [20%, 40%] "
            f"(counts={counts})"
        )
