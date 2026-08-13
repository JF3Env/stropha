"""Unit tests for ``stropha.meta.baseline`` (WIN-204).

Covers the production-baseline tracker: read/write/compare/regressions.
The CLI integration with ``meta report`` lives in
``test_meta_cli.py::TestMetaReportBaseline``; this file proves the
library surface is correct independently of Typer wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.meta.baseline import (
    BASELINE_FILENAME,
    BASELINE_SCHEMA_VERSION,
    BaselineError,
    BaselineRecord,
    BaselineValidationProvenance,
    Comparison,
    compare_candidates,
    read_baseline,
    regressions,
    write_baseline,
    write_verified_baseline,
)
from stropha.meta.store import CandidateRecord


def _make_candidate(
    cid: str,
    scores: dict | None = None,
    label: str | None = None,
) -> CandidateRecord:
    """Build a minimal CandidateRecord with optional scores."""
    return CandidateRecord(
        id=cid,
        parent_id=None,
        created_at="2026-05-23T00:00:00+00:00",
        label=label or cid,
        config={"v": cid},
        scores=scores or {},
    )


# ---------------------------------------------------------------------------
# read_baseline
# ---------------------------------------------------------------------------


class TestReadBaseline:
    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """A missing file is the 'never promoted' state, not an error."""
        assert read_baseline(tmp_path) is None

    def test_loads_valid_baseline(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(
                {
                    "candidate_id": "abc",
                    "promoted_at": "2026-05-23T12:00:00+00:00",
                    "fitness": 0.85,
                    "label": "best-so-far",
                    "meta": {"reason": "manual"},
                }
            ),
            encoding="utf-8",
        )
        rec = read_baseline(tmp_path)
        assert rec is not None
        assert rec.candidate_id == "abc"
        assert rec.fitness == pytest.approx(0.85)
        assert rec.label == "best-so-far"
        assert rec.meta == {"reason": "manual"}
        assert rec.schema_version == 1
        assert rec.verification_status == "legacy_unverified"
        assert rec.is_verified is False

    def test_loads_verified_schema_v2_baseline(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "verification_status": "verified",
                    "candidate_id": "abc",
                    "fitness": 0.85,
                    "validation_provenance": {
                        "schema_version": 2,
                        "run_id": "run-1",
                        "artifact_name": "validation.json",
                        "baseline_id": "old-live",
                        "candidate_id": "abc",
                        "validated_identity": {"runtime": "r1"},
                        "verification_checks": {"pytest": True},
                    },
                }
            ),
            encoding="utf-8",
        )

        rec = read_baseline(tmp_path)

        assert rec is not None
        assert rec.schema_version == BASELINE_SCHEMA_VERSION
        assert rec.is_verified is True

    def test_rejects_verified_baseline_without_provenance(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "verification_status": "verified",
                    "candidate_id": "abc",
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(BaselineError, match="requires validation provenance"):
            read_baseline(tmp_path)

    def test_rejects_mismatched_validation_provenance(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "verification_status": "verified",
                    "candidate_id": "abc",
                    "validation_provenance": {
                        "schema_version": 2,
                        "run_id": "run-1",
                        "artifact_name": "validation.json",
                        "baseline_id": "old-live",
                        "candidate_id": "different",
                        "validated_identity": {"runtime": "r1"},
                        "verification_checks": {},
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(BaselineError, match="candidate_id does not match"):
            read_baseline(tmp_path)

    def test_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps({"schema_version": 3, "candidate_id": "abc"}),
            encoding="utf-8",
        )

        with pytest.raises(BaselineError, match="unsupported schema_version"):
            read_baseline(tmp_path)

    def test_loads_baseline_with_null_fitness(self, tmp_path: Path) -> None:
        """A baseline promoted without scores has fitness=null —
        comparisons fall back gracefully."""
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(
                {
                    "candidate_id": "abc",
                    "fitness": None,
                }
            ),
            encoding="utf-8",
        )
        rec = read_baseline(tmp_path)
        assert rec is not None
        assert rec.fitness is None

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text("not valid {{ json", encoding="utf-8")
        with pytest.raises(BaselineError, match="not valid JSON"):
            read_baseline(tmp_path)

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps(["just", "a", "list"]),
            encoding="utf-8",
        )
        with pytest.raises(BaselineError, match="must be a JSON object"):
            read_baseline(tmp_path)

    def test_missing_candidate_id_raises(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps({"fitness": 0.5}),
            encoding="utf-8",
        )
        with pytest.raises(BaselineError, match="missing required field"):
            read_baseline(tmp_path)

    def test_non_numeric_fitness_raises(self, tmp_path: Path) -> None:
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps({"candidate_id": "x", "fitness": "high"}),
            encoding="utf-8",
        )
        with pytest.raises(BaselineError, match="must be a number"):
            read_baseline(tmp_path)

    def test_missing_promoted_at_synthesises_now(self, tmp_path: Path) -> None:
        """Legacy / hand-edited baselines may lack promoted_at — fill
        with a fresh UTC timestamp instead of failing."""
        (tmp_path / BASELINE_FILENAME).write_text(
            json.dumps({"candidate_id": "x", "fitness": 0.5}),
            encoding="utf-8",
        )
        rec = read_baseline(tmp_path)
        assert rec is not None
        # ISO-8601-ish — at least has the YYYY prefix and a T.
        assert rec.promoted_at.startswith("20")
        assert "T" in rec.promoted_at


# ---------------------------------------------------------------------------
# write_baseline
# ---------------------------------------------------------------------------


class TestWriteBaseline:
    def test_persists_candidate_metadata(self, tmp_path: Path) -> None:
        cand = _make_candidate(
            "c1",
            scores={"quality": 0.8, "latency_ms": 50.0},
            label="winner",
        )
        rec = write_baseline(tmp_path, candidate=cand)
        assert rec.candidate_id == "c1"
        assert rec.label == "winner"
        assert rec.fitness == pytest.approx(0.8 - 0.05)
        assert rec.schema_version == BASELINE_SCHEMA_VERSION
        assert rec.verification_status == "unverified"
        # File persisted.
        assert (tmp_path / BASELINE_FILENAME).is_file()

    def test_no_fitness_when_candidate_unscored(self, tmp_path: Path) -> None:
        """Promoting blind (no scores) is allowed; fitness stays None."""
        cand = _make_candidate("c1")
        rec = write_baseline(tmp_path, candidate=cand)
        assert rec.fitness is None
        # JSON round-trips with null fitness.
        payload = json.loads((tmp_path / BASELINE_FILENAME).read_text(encoding="utf-8"))
        assert payload["fitness"] is None

    def test_creates_meta_dir_if_missing(self, tmp_path: Path) -> None:
        deep = tmp_path / "nested" / "meta"
        cand = _make_candidate("c1", scores={"quality": 1.0})
        write_baseline(deep, candidate=cand)
        assert (deep / BASELINE_FILENAME).is_file()

    def test_atomic_replace_no_tmp_leftover(self, tmp_path: Path) -> None:
        cand = _make_candidate("c1", scores={"quality": 1.0})
        write_baseline(tmp_path, candidate=cand)
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_overwrites_existing_baseline(self, tmp_path: Path) -> None:
        cand_a = _make_candidate("a", scores={"quality": 0.5})
        cand_b = _make_candidate("b", scores={"quality": 0.9})
        write_baseline(tmp_path, candidate=cand_a)
        write_baseline(tmp_path, candidate=cand_b)
        rec = read_baseline(tmp_path)
        assert rec is not None
        assert rec.candidate_id == "b"

    def test_extra_meta_passed_through(self, tmp_path: Path) -> None:
        cand = _make_candidate("c1", scores={"quality": 1.0})
        rec = write_baseline(
            tmp_path,
            candidate=cand,
            extra_meta={"approved_by": "op@example.com"},
        )
        assert rec.meta == {"approved_by": "op@example.com"}

    def test_promoted_at_override(self, tmp_path: Path) -> None:
        cand = _make_candidate("c1", scores={"quality": 1.0})
        rec = write_baseline(
            tmp_path,
            candidate=cand,
            promoted_at="2026-01-01T00:00:00+00:00",
        )
        assert rec.promoted_at == "2026-01-01T00:00:00+00:00"

    def test_verified_write_records_validation_provenance(self, tmp_path: Path) -> None:
        from stropha.meta.validation import (
            EvaluationEvidence,
            EvaluationMetrics,
            build_validation_artifact,
        )

        metrics = EvaluationMetrics(
            success_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg_at_k=0.8,
            latency_p95_ms=10.0,
            returned_tokens=100,
        )
        identity = {"runtime": "r1"}
        evidence = EvaluationEvidence(
            search=metrics,
            holdout=metrics,
            search_identity={"golden": "search"},
            holdout_identity={"golden": "holdout"},
            runtime_identity=identity,
        )
        candidate_evidence = evidence.model_copy(
            update={"search": metrics.model_copy(update={"ndcg_at_k": 0.9})}
        )
        validation = build_validation_artifact(
            baseline_record=BaselineRecord(
                candidate_id="old-live",
                promoted_at="2026-08-13T00:00:00+00:00",
                verification_status="verified",
                validation_provenance=BaselineValidationProvenance(
                    run_id="run-0",
                    artifact_name="validation.json",
                    baseline_id="previous-live",
                    candidate_id="old-live",
                    validated_identity=identity,
                    verification_checks={"pytest": True},
                ),
            ),
            candidate_id="candidate",
            baseline=evidence,
            candidate=candidate_evidence,
            verification_checks={"pytest": True},
            current_identity=identity,
        )
        candidate = _make_candidate("candidate", scores={"quality": 0.9})

        record = write_verified_baseline(
            tmp_path,
            candidate=candidate,
            validation=validation,
            run_id="run-1",
        )

        assert record.is_verified is True
        assert record.validation_provenance == BaselineValidationProvenance(
            run_id="run-1",
            artifact_name="validation.json",
            baseline_id="old-live",
            candidate_id="candidate",
            validated_identity=identity,
            verification_checks={"pytest": True},
        )
        assert read_baseline(tmp_path) == record

    def test_verified_write_rejects_ineligible_validation(self, tmp_path: Path) -> None:
        from stropha.meta.validation import (
            EvaluationEvidence,
            EvaluationMetrics,
            build_validation_artifact,
        )

        metrics = EvaluationMetrics(
            success_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg_at_k=0.8,
            latency_p95_ms=10.0,
            returned_tokens=100,
        )
        evidence = EvaluationEvidence(search=metrics, holdout=metrics)
        validation = build_validation_artifact(
            baseline_record=BaselineRecord(candidate_id="old", promoted_at="now"),
            candidate_id="candidate",
            baseline=evidence,
            candidate=evidence,
        )

        with pytest.raises(BaselineError, match="ineligible"):
            write_verified_baseline(
                tmp_path,
                candidate=_make_candidate("candidate"),
                validation=validation,
                run_id="run-1",
            )
        assert not (tmp_path / BASELINE_FILENAME).exists()


# ---------------------------------------------------------------------------
# compare_candidates
# ---------------------------------------------------------------------------


class TestCompareCandidates:
    def test_delta_against_baseline_fitness(self) -> None:
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        candidates = [
            _make_candidate("a", scores={"quality": 0.8}),
            _make_candidate("b", scores={"quality": 0.5}),  # the baseline
            _make_candidate("c", scores={"quality": 0.4}),
        ]
        cmps = compare_candidates(candidates, baseline)
        deltas = {c.candidate.id: c.delta for c in cmps}
        assert deltas["a"] == pytest.approx(0.3)
        assert deltas["b"] == 0.0  # baseline row
        assert deltas["c"] == pytest.approx(-0.1)

    def test_baseline_row_flagged_is_baseline(self) -> None:
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        candidates = [_make_candidate("b", scores={"quality": 0.7})]
        cmps = compare_candidates(candidates, baseline)
        assert len(cmps) == 1
        # Even though the in-archive fitness drifted upward (0.7 vs the
        # baseline-time 0.5), the row is flagged as baseline with
        # delta=0 — the promoted snapshot is authoritative.
        assert cmps[0].is_baseline is True
        assert cmps[0].delta == 0.0

    def test_unscored_candidates_skipped(self) -> None:
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        candidates = [
            _make_candidate("scored", scores={"quality": 0.8}),
            _make_candidate("blank"),  # no scores → fitness == -inf
        ]
        cmps = compare_candidates(candidates, baseline)
        assert len(cmps) == 1
        assert cmps[0].candidate.id == "scored"

    def test_baseline_without_fitness_disables_comparisons(self) -> None:
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=None,
        )
        candidates = [_make_candidate("a", scores={"quality": 0.5})]
        cmps = compare_candidates(candidates, baseline)
        assert cmps == []

    def test_empty_candidates_list(self) -> None:
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        assert compare_candidates([], baseline) == []


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def _make_cmps(self) -> list[Comparison]:
        """Build a comparison list with mixed deltas."""
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        candidates = [
            _make_candidate("better", scores={"quality": 0.9}),
            _make_candidate("worse", scores={"quality": 0.4}),
            _make_candidate("b", scores={"quality": 0.5}),  # baseline
            _make_candidate("tied", scores={"quality": 0.5}),
            _make_candidate("slightly-better", scores={"quality": 0.501}),
        ]
        return compare_candidates(candidates, baseline)

    def test_flags_positive_deltas_only(self) -> None:
        flagged = regressions(self._make_cmps())
        ids = {c.candidate.id for c in flagged}
        # "tied" has delta=0 → not flagged at threshold=0 (strict >).
        # "slightly-better" has positive delta → flagged.
        assert ids == {"better", "slightly-better"}

    def test_threshold_suppresses_noise(self) -> None:
        flagged = regressions(self._make_cmps(), threshold=0.1)
        ids = {c.candidate.id for c in flagged}
        # Only "better" (0.4 delta) exceeds 0.1 threshold.
        assert ids == {"better"}

    def test_sorted_by_delta_descending(self) -> None:
        flagged = regressions(self._make_cmps())
        # Largest delta first.
        assert flagged[0].candidate.id == "better"
        assert flagged[-1].candidate.id == "slightly-better"

    def test_baseline_never_flagged(self) -> None:
        """Even if the baseline's in-archive fitness drifted upward,
        compare_candidates clamps its delta to 0 — so regressions
        cannot return the baseline row by construction."""
        # Force a high in-archive fitness for the baseline candidate.
        baseline = BaselineRecord(
            candidate_id="b",
            promoted_at="2026-01-01T00:00:00+00:00",
            fitness=0.5,
        )
        candidates = [_make_candidate("b", scores={"quality": 0.9})]
        cmps = compare_candidates(candidates, baseline)
        flagged = regressions(cmps)
        assert flagged == []

    def test_empty_comparisons_returns_empty(self) -> None:
        assert regressions([]) == []


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


class TestPublicReexports:
    def test_symbols_exported_from_stropha_meta(self) -> None:
        from stropha import meta

        assert meta.BaselineError is BaselineError
        assert meta.BaselineRecord is BaselineRecord
        assert meta.Comparison is Comparison
        assert meta.read_baseline is read_baseline
        assert meta.write_baseline is write_baseline
        assert meta.compare_candidates is compare_candidates
        assert meta.regressions is regressions
        assert meta.BASELINE_FILENAME == BASELINE_FILENAME
        assert meta.BASELINE_SCHEMA_VERSION == BASELINE_SCHEMA_VERSION
