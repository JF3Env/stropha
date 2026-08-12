"""Unit tests for the Meta-Harness candidate evaluator (WIN-171).

Covers every branch of ``evaluate_candidate``:
- search vs holdout vs all split filtering
- empty-cases error path
- ScoreCard composition with + without outcome labels
- explicit labels precedence over outcomes_dir
- outcomes_dir loader (head_sha specific + glob fallback + missing dir)
- repo_size_bytes feeds context_efficiency baseline
- evaluate_from_golden convenience wrapper
- EvaluatorError surface
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.eval import EvalCase
from stropha.meta import (
    EvaluatorError,
    ScoreCard,
    Tier,
    TierLabel,
    evaluate_candidate,
    evaluate_from_golden,
    query_candidate_config,
)
from stropha.meta.evaluator import (
    _baseline_tokens_from_repo,
    _load_labels_from_outcomes,
    _returned_tokens_from_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeHit:
    def __init__(self, rel_path: str, symbol: str | None = None) -> None:
        self.rel_path = rel_path
        self.symbol = symbol


class _FakeEngine:
    """Stub engine that returns canned hits per query."""

    def __init__(self, mapping: dict[str, list[_FakeHit]] | None = None) -> None:
        self.mapping = mapping or {}

    def search(
        self, query: str, *, top_k: int = 10, repo_id: int | None = None
    ) -> list[_FakeHit]:
        return list(self.mapping.get(query, []))[:top_k]


def _cases() -> list[EvalCase]:
    return [
        EvalCase(
            id="s1", query="q-search-1",
            expected_paths=("src/a.py",),
            split="search",
        ),
        EvalCase(
            id="s2", query="q-search-2",
            expected_paths=("src/b.py",),
            split="search",
        ),
        EvalCase(
            id="h1", query="q-holdout-1",
            expected_paths=("src/c.py",),
            split="holdout",
        ),
    ]


def _engine() -> _FakeEngine:
    return _FakeEngine({
        "q-search-1":  [_FakeHit("src/a.py")],
        "q-search-2":  [_FakeHit("src/b.py")],
        "q-holdout-1": [_FakeHit("src/c.py")],
    })


# ---------------------------------------------------------------------------
# evaluate_candidate
# ---------------------------------------------------------------------------


class TestEvaluateCandidate:
    def test_search_split_returns_scorecard(self) -> None:
        card = evaluate_candidate(engine=_engine(), cases=_cases(), split="search")
        assert isinstance(card, ScoreCard)
        assert card.n_queries == 2  # search subset only
        assert card.n_hits == 2
        assert card.quality > 0

    def test_holdout_split_filters_correctly(self) -> None:
        card = evaluate_candidate(engine=_engine(), cases=_cases(), split="holdout")
        assert card.n_queries == 1
        assert card.n_hits == 1

    def test_all_split_includes_everything(self) -> None:
        card = evaluate_candidate(engine=_engine(), cases=_cases(), split="all")
        assert card.n_queries == 3

    def test_empty_split_raises(self) -> None:
        cases = [EvalCase(id="only", query="q", split="search")]
        with pytest.raises(EvaluatorError, match="no cases"):
            evaluate_candidate(engine=_engine(), cases=cases, split="holdout")

    def test_explicit_labels_take_precedence(self, tmp_path: Path) -> None:
        """If both ``labels`` and ``outcomes_dir`` are passed, ``labels``
        wins — the proposer's hand-crafted scoring beats the historical
        backfill."""
        explicit = [
            TierLabel(chunk_id="c1", tier=Tier.T1, score=1.0, rationale="x"),
        ]
        # Even with an outcomes_dir present, the explicit labels are used.
        outcomes = tmp_path / "outcomes"
        outcomes.mkdir()
        (outcomes / "abc.json").write_text(json.dumps({
            "tier_counts": {"T7": 100},  # would otherwise drag score to 0
        }))
        card = evaluate_candidate(
            engine=_engine(), cases=_cases(), split="search",
            labels=explicit, outcomes_dir=outcomes,
        )
        # outcome_score = mean of explicit labels = 1.0
        assert card.outcome_score == 1.0

    def test_outcomes_dir_hydrates_labels(self, tmp_path: Path) -> None:
        outcomes = tmp_path / "outcomes"
        outcomes.mkdir()
        (outcomes / "head1.json").write_text(json.dumps({
            "tier_counts": {"T1": 2, "T7": 2},
        }))
        card = evaluate_candidate(
            engine=_engine(), cases=_cases(), split="search",
            outcomes_dir=outcomes,
        )
        # T1 score 1.0 x 2 + T7 score 0.0 x 2 → aggregate 0.5
        assert card.outcome_score == pytest.approx(0.5)
        assert card.tier_counts["T1"] == 2
        assert card.tier_counts["T7"] == 2

    def test_outcomes_dir_specific_head_sha(self, tmp_path: Path) -> None:
        outcomes = tmp_path / "outcomes"
        outcomes.mkdir()
        (outcomes / "wanted.json").write_text(json.dumps({
            "tier_counts": {"T1": 1},
        }))
        (outcomes / "ignored.json").write_text(json.dumps({
            "tier_counts": {"T6": 10},  # would crater the score
        }))
        card = evaluate_candidate(
            engine=_engine(), cases=_cases(), split="search",
            outcomes_dir=outcomes, head_sha="wanted",
        )
        # Only the head_sha=wanted file was hydrated → T1 only → score 1.0
        assert card.outcome_score == 1.0

    def test_repo_size_feeds_baseline(self) -> None:
        card = evaluate_candidate(
            engine=_engine(), cases=_cases(), split="all",
            repo_size_bytes=10_000_000,  # 10 MB → ~250k baseline tokens
        )
        # Hits returned: 3 x ~50 tokens = 150 returned, baseline 250000 → ~1.0
        assert card.context_efficiency > 0.99

    def test_no_repo_size_zero_efficiency(self) -> None:
        card = evaluate_candidate(engine=_engine(), cases=_cases(), split="search")
        assert card.context_efficiency == 0.0

    def test_repository_scope_is_forwarded(self) -> None:
        class _RecordingEngine(_FakeEngine):
            def __init__(self, mapping: dict[str, list[_FakeHit]]) -> None:
                super().__init__(mapping)
                self.repo_ids: list[int | None] = []

            def search(
                self, query: str, *, top_k: int = 10, repo_id: int | None = None
            ) -> list[_FakeHit]:
                self.repo_ids.append(repo_id)
                return super().search(query, top_k=top_k, repo_id=repo_id)

        engine = _RecordingEngine({"q-search-1": [_FakeHit("src/a.py")]})
        evaluate_candidate(engine=engine, cases=_cases(), split="search", repo_id=9)
        assert engine.repo_ids == [9, 9]


class TestQueryCandidateConfig:
    def test_applies_query_fields_without_changing_indexing_stages(self) -> None:
        base = {
            "embedder": {"adapter": "mlx"},
            "enricher": {"adapter": "contextual"},
            "retrieval": {"adapter": "hybrid-rrf", "config": {}},
        }
        resolved = query_candidate_config(base, {
            "reranker": "llm",
            "candidate_k": 40,
            "rrf_k": 80,
            "recursive": True,
        })
        assert resolved["embedder"] == base["embedder"]
        assert resolved["enricher"] == base["enricher"]
        assert resolved["retrieval"]["config"]["reranker"] == {"adapter": "llm"}
        assert resolved["retrieval"]["config"]["candidate_k"] == 40
        assert resolved["retrieval"]["config"]["rrf_k"] == 80
        assert resolved["retrieval"]["config"]["recursive_enabled"] is True

    def test_rejects_indexing_dimension(self) -> None:
        with pytest.raises(EvaluatorError, match="indexing or unsupported"):
            query_candidate_config(
                {"retrieval": {"config": {}}},
                {"embedder": "voyage"},
            )


# ---------------------------------------------------------------------------
# evaluate_from_golden
# ---------------------------------------------------------------------------


class TestEvaluateFromGolden:
    def test_loads_then_evaluates(self, tmp_path: Path) -> None:
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            json.dumps({"id": "g1", "query": "q1", "expected_paths": ["src/a.py"], "split": "search"}) + "\n"
        )
        card = evaluate_from_golden(
            engine=_FakeEngine({"q1": [_FakeHit("src/a.py")]}),
            golden_path=golden,
            split="search",
        )
        assert card.n_queries == 1
        assert card.n_hits == 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestBaselineTokens:
    def test_none_returns_zero(self) -> None:
        assert _baseline_tokens_from_repo(None) == 0

    def test_zero_returns_zero(self) -> None:
        assert _baseline_tokens_from_repo(0) == 0

    def test_negative_returns_zero(self) -> None:
        assert _baseline_tokens_from_repo(-100) == 0

    def test_positive_estimates_proportionally(self) -> None:
        # 10 MB → 10% = 1 MB → /4 bytes = 250k tokens.
        assert _baseline_tokens_from_repo(10_000_000) == 250_000


class TestReturnedTokensFromReport:
    def test_returns_zero_for_empty_report(self) -> None:
        from stropha.eval import EvalReport

        assert _returned_tokens_from_report(EvalReport(top_k=10)) == 0

    def test_sums_50_per_relevant_hit(self) -> None:
        from stropha.eval import EvalReport, EvalResult

        report = EvalReport(top_k=5)
        report.results = [
            EvalResult(
                case=EvalCase(id="x", query="q"),
                rank_of_first_hit=1, top_k=5,
                relevances=(1.0, 0.0, 1.0, 0.0, 0.0),  # 2 relevant
            ),
        ]
        assert _returned_tokens_from_report(report) == 100  # 2 x 50

    def test_skips_zero_relevances(self) -> None:
        from stropha.eval import EvalReport, EvalResult

        report = EvalReport(top_k=5)
        report.results = [
            EvalResult(
                case=EvalCase(id="x", query="q"),
                rank_of_first_hit=None, top_k=5,
                relevances=(0.0, 0.0, 0.0),
            ),
        ]
        assert _returned_tokens_from_report(report) == 0


class TestLoadLabelsFromOutcomes:
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert _load_labels_from_outcomes(tmp_path / "missing") == []

    def test_loads_all_when_head_sha_omitted(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"tier_counts": {"T1": 1}}))
        (d / "b.json").write_text(json.dumps({"tier_counts": {"T7": 1}}))
        labels = _load_labels_from_outcomes(d)
        tiers = {label.tier.value for label in labels}
        assert tiers == {"T1", "T7"}

    def test_filters_to_head_sha(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "wanted.json").write_text(json.dumps({"tier_counts": {"T1": 1}}))
        (d / "ignored.json").write_text(json.dumps({"tier_counts": {"T6": 1}}))
        labels = _load_labels_from_outcomes(d, head_sha="wanted")
        assert all(label.tier == Tier.T1 for label in labels)

    def test_specific_head_sha_missing_returns_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        # Only ask for a file that does not exist.
        assert _load_labels_from_outcomes(d, head_sha="nope") == []

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "bad.json").write_text("garbage{{")
        (d / "good.json").write_text(json.dumps({"tier_counts": {"T1": 1}}))
        labels = _load_labels_from_outcomes(d)
        assert len(labels) == 1
        assert labels[0].tier == Tier.T1

    def test_unknown_tier_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "x.json").write_text(json.dumps({
            "tier_counts": {"T1": 1, "T99": 5},
        }))
        labels = _load_labels_from_outcomes(d)
        # Only T1 hydrated; T99 silently skipped.
        assert len(labels) == 1
        assert labels[0].tier == Tier.T1

    def test_empty_tier_counts(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "x.json").write_text(json.dumps({"tier_counts": {}}))
        assert _load_labels_from_outcomes(d) == []

    def test_missing_tier_counts_key(self, tmp_path: Path) -> None:
        d = tmp_path / "outcomes"
        d.mkdir()
        (d / "x.json").write_text(json.dumps({}))
        assert _load_labels_from_outcomes(d) == []


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports_evaluator_surface(self) -> None:
        from stropha import meta
        assert hasattr(meta, "evaluate_candidate")
        assert hasattr(meta, "evaluate_from_golden")
        assert hasattr(meta, "EvaluatorError")
