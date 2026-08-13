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


class _RepoScopedEngine(_FakeEngine):
    def __init__(self, by_query: dict[str, list[_FakeHit]]) -> None:
        super().__init__(by_query)
        self.repo_ids: list[int | None] = []

    def search(
        self, query: str, *, top_k: int = 10, repo_id: int | None = None
    ) -> list[_FakeHit]:
        self.repo_ids.append(repo_id)
        return super().search(query, top_k=top_k)


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


def test_load_golden_rejects_duplicate_ids(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text(
        '{"id":"same","query":"a","expected_paths":["a.py"]}\n'
        '{"id":"same","query":"b","expected_paths":["b.py"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
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


def test_run_eval_passes_repository_scope() -> None:
    engine = _RepoScopedEngine({"q": [_FakeHit("a.py", None)]})
    report = run_eval(
        [EvalCase(id="c", query="q", expected_paths=("a.py",))],
        engine,  # type: ignore[arg-type]
        repo_id=7,
    )
    assert report.success_at_k == 1.0
    assert engine.repo_ids == [7]


def test_duplicate_chunks_count_as_one_relevant_entity() -> None:
    case = EvalCase(id="c", query="q", expected_paths=("a.py", "b.py"))
    engine = _FakeEngine({
        "q": [
            _FakeHit("a.py", "one"),
            _FakeHit("a.py", "two"),
            _FakeHit("b.py", "three"),
        ]
    })
    report = run_eval([case], engine, top_k=3)  # type: ignore[arg-type]
    result = report.results[0]
    assert result.relevant_count == 2
    assert report.recall_at_k == 1.0
    assert result.precision == pytest.approx(2 / 3)


def test_success_and_entity_recall_are_distinct() -> None:
    case = EvalCase(id="c", query="q", expected_paths=("a.py", "b.py"))
    engine = _FakeEngine({"q": [_FakeHit("a.py", None)]})
    report = run_eval([case], engine, top_k=2)  # type: ignore[arg-type]
    assert report.success_at_k == 1.0
    assert report.recall_at_k == 0.5


def test_extra_symbol_entity_can_share_first_entities_path() -> None:
    case = EvalCase(
        id="c",
        query="q",
        expected_paths=("a.py",),
        expected_symbols=("first", "second"),
    )
    engine = _FakeEngine({
        "q": [
            _FakeHit("a.py", "first"),
            _FakeHit("a.py", "second"),
        ]
    })
    report = run_eval([case], engine, top_k=2)  # type: ignore[arg-type]
    assert report.results[0].relevant_count == 2
    assert report.recall_at_k == 1.0


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
    assert len(cases) >= 20  # we ship ~50
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

    def test_ideal_relevance_counts_paired_path_symbol_once(self):
        case = EvalCase(
            id="test",
            query="foo",
            expected_paths=("a.py", "b.py"),
            expected_symbols=("A", "B"),
            relevance_map={"a.py": 3.0, "A": 3.0, "b.py": 2.0, "B": 2.0},
        )

        assert case.ideal_relevances == (3.0, 2.0)

    def test_ideal_relevance_defaults_only_unjudged_entities(self):
        case = EvalCase(
            id="test",
            query="foo",
            expected_paths=("a.py", "b.py"),
            expected_symbols=("A", "B"),
            relevance_map={"a.py": 0.0},
        )

        assert case.ideal_relevances == (1.0, 0.0)

    def test_paired_graded_entities_can_reach_perfect_ndcg(self):
        case = EvalCase(
            id="test",
            query="foo",
            expected_paths=("a.py", "b.py"),
            expected_symbols=("A", "B"),
            relevance_map={"a.py": 3.0, "A": 3.0, "b.py": 2.0, "B": 2.0},
        )
        result = EvalResult(
            case=case,
            rank_of_first_hit=1,
            top_k=2,
            relevances=(3.0, 2.0),
            relevant_count=2,
        )

        assert result.ndcg == pytest.approx(1.0)


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


# ---------------------------------------------------------------------------
# WIN-170 — per-query latency capture + MAP + by_split
# ---------------------------------------------------------------------------


class TestPerQueryLatency:
    def test_run_eval_captures_latency_per_case(self):
        import time as _time

        def slow_engine_search(self, query, top_k):
            _time.sleep(0.005)
            return []

        class _SlowEngine:
            search = slow_engine_search

        cases = [
            EvalCase(id="c1", query="q1"),
            EvalCase(id="c2", query="q2"),
        ]
        report = run_eval(cases, _SlowEngine(), top_k=5)  # type: ignore[arg-type]
        # Each result has latency_ms; both ≥ ~5ms (the sleep), < 500ms (slop).
        for r in report.results:
            assert r.latency_ms >= 4.0
            assert r.latency_ms < 500.0

    def test_eval_result_default_latency_is_zero(self):
        """Direct EvalResult construction without latency keeps the zero
        default — back-compat with synthetic tests that don't time."""
        r = EvalResult(case=EvalCase(id="x", query="q"), rank_of_first_hit=1, top_k=10)
        assert r.latency_ms == 0.0


class TestLatencyPercentiles:
    def _result_with_latency(self, latency_ms: float) -> EvalResult:
        return EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=1,
            top_k=10,
            latency_ms=latency_ms,
        )

    def test_empty_report_zero_percentiles(self):
        from stropha.eval import EvalReport

        report = EvalReport(top_k=10)
        assert report.latency_p50_ms == 0.0
        assert report.latency_p95_ms == 0.0
        assert report.latency_p99_ms == 0.0

    def test_percentiles_match_sorted_position(self):
        from stropha.eval import EvalReport

        report = EvalReport(top_k=10)
        report.results = [self._result_with_latency(float(i)) for i in range(1, 101)]
        # 100 samples, p50 = 50th value (sorted), p95 = 95th, p99 = 99th.
        # _latency_percentile uses int(N*pct/100), so p50=index 50 → value 51.
        assert report.latency_p50_ms in (50.0, 51.0)
        assert report.latency_p95_ms in (95.0, 96.0)
        assert report.latency_p99_ms in (99.0, 100.0)

    def test_percentile_clamps_for_tiny_sample(self):
        """1 sample: every percentile returns that sample."""
        from stropha.eval import EvalReport

        report = EvalReport(top_k=10)
        report.results = [self._result_with_latency(12.5)]
        assert report.latency_p50_ms == 12.5
        assert report.latency_p95_ms == 12.5
        assert report.latency_p99_ms == 12.5


class TestAveragePrecisionAndMap:
    def test_ap_zero_for_no_relevances(self):
        r = EvalResult(case=EvalCase(id="x", query="q"), rank_of_first_hit=None, top_k=10)
        assert r.average_precision == 0.0

    def test_ap_zero_when_top_k_all_zeroes(self):
        r = EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=None,
            top_k=5,
            relevances=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        assert r.average_precision == 0.0

    def test_ap_perfect_first_rank(self):
        """Single relevant at rank 1 → AP = 1.0."""
        r = EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=1,
            top_k=5,
            relevances=(1.0, 0.0, 0.0, 0.0, 0.0),
        )
        assert r.average_precision == pytest.approx(1.0)

    def test_ap_relevant_at_second_rank(self):
        """Single relevant at rank 2 → AP = 1/2 = 0.5."""
        r = EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=2,
            top_k=5,
            relevances=(0.0, 1.0, 0.0, 0.0, 0.0),
        )
        assert r.average_precision == pytest.approx(0.5)

    def test_ap_multiple_relevant(self):
        """Relevant at ranks 1 and 3 → AP = (1/1 + 2/3) / 2 ≈ 0.833."""
        r = EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=1,
            top_k=5,
            relevances=(1.0, 0.0, 1.0, 0.0, 0.0),
        )
        assert r.average_precision == pytest.approx((1.0 + 2/3) / 2)

    def test_map_at_k_is_mean_ap(self):
        from stropha.eval import EvalReport

        report = EvalReport(top_k=5)
        report.results = [
            EvalResult(
                case=EvalCase(id="a", query="qa"),
                rank_of_first_hit=1,
                top_k=5,
                relevances=(1.0, 0.0, 0.0, 0.0, 0.0),
            ),  # AP = 1.0
            EvalResult(
                case=EvalCase(id="b", query="qb"),
                rank_of_first_hit=2,
                top_k=5,
                relevances=(0.0, 1.0, 0.0, 0.0, 0.0),
            ),  # AP = 0.5
        ]
        assert report.map_at_k == pytest.approx(0.75)

    def test_map_at_k_empty_report(self):
        from stropha.eval import EvalReport

        assert EvalReport(top_k=5).map_at_k == 0.0


class TestSummaryAndBySplit:
    def test_summary_includes_map_and_latency_percentiles(self):
        cases = [EvalCase(id="c1", query="q1", expected_paths=("a.py",))]
        engine = _FakeEngine({"q1": [_FakeHit("a.py", None)]})
        report = run_eval(cases, engine, top_k=5)  # type: ignore[arg-type]
        s = report.summary()
        for key in ("map@5", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
            assert key in s

    def test_by_split_groups_results(self):
        from stropha.eval import EvalReport

        report = EvalReport(top_k=5)
        report.results = [
            EvalResult(
                case=EvalCase(id="s1", query="q1", split="search"),
                rank_of_first_hit=1, top_k=5,
            ),
            EvalResult(
                case=EvalCase(id="h1", query="q2", split="holdout"),
                rank_of_first_hit=1, top_k=5,
            ),
            EvalResult(
                case=EvalCase(id="s2", query="q3", split="search"),
                rank_of_first_hit=2, top_k=5,
            ),
        ]
        buckets = report.by_split()
        assert set(buckets) == {"search", "holdout"}
        assert buckets["search"].n == 2
        assert buckets["holdout"].n == 1


# ---------------------------------------------------------------------------
# Pre-existing coverage gaps closed under WIN-170 (push harness.py to 100%)
# ---------------------------------------------------------------------------


class TestPreExistingGapsClosedForCompleteness:
    """Targeted tests that fill defensive branches the prior eval suite
    happened to skip. Closed in WIN-170 since the same module gained MAP
    + latency surface and coverage was being pushed to 100% anyway."""

    def test_eval_result_ndcg_zero_when_no_relevances(self):
        r = EvalResult(
            case=EvalCase(id="x", query="q"),
            rank_of_first_hit=None,
            top_k=5,
            relevances=(),
        )
        assert r.ndcg == 0.0

    def test_empty_report_mrr_zero(self):
        from stropha.eval import EvalReport

        assert EvalReport(top_k=5).mrr == 0.0

    def test_empty_report_ndcg_at_k_zero(self):
        from stropha.eval import EvalReport

        assert EvalReport(top_k=5).ndcg_at_k == 0.0

    def test_empty_report_precision_at_k_zero(self):
        from stropha.eval import EvalReport

        assert EvalReport(top_k=5).precision_at_k == 0.0

    def test_get_relevance_with_graded_map(self):
        case = EvalCase.from_dict({
            "id": "g1",
            "query": "graded",
            "expected_paths": ["src/foo.py"],
            "relevance": {"src/foo.py": 3.0, "FooClass": 2.0},
        })
        assert case.get_relevance("src/foo.py", None) == 3.0
        assert case.get_relevance(None, "FooClass") == 2.0
        # Unknown path/symbol falls through to 1.0 (binary fallback —
        # the relevance lookup does not penalise misses, only graded
        # matches get a score >1; misses outside the map still count
        # as binary relevant).
        assert case.get_relevance("other.py", "Other") == 1.0

    def test_path_matches_skips_empty_expected(self):
        """``_path_matches`` MUST skip empty entries in the expected list
        instead of treating them as a wildcard."""
        from stropha.eval.harness import _path_matches

        # Empty strings in the expected list are silently skipped.
        assert _path_matches("src/foo.py", ["", "src/foo.py"]) == "src/foo.py"
        assert _path_matches("src/foo.py", [""]) is None

    def test_symbol_matches_skips_empty_expected(self):
        from stropha.eval.harness import _symbol_matches

        assert _symbol_matches("Foo", ["", "Foo"]) == "Foo"
        assert _symbol_matches("Foo", [""]) is None

    def test_ndcg_zero_idcg_branch(self):
        """``_ndcg`` returns 0 when the ideal DCG would be 0 (all
        relevances are zero) — guards against div-by-zero."""
        from stropha.eval.harness import _ndcg

        assert _ndcg([0.0, 0.0, 0.0], 3) == 0.0
        assert _ndcg([], 5) == 0.0
