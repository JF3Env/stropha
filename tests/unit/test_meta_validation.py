"""Tests for schema-v2 evaluation validation and promotion gating."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stropha.eval import EvalCase, EvalReport, EvalResult
from stropha.meta.baseline import BaselineRecord, BaselineValidationProvenance
from stropha.meta.validation import (
    BaselineValidationArtifact,
    EvaluationEvidence,
    EvaluationMetrics,
    QualityMetrics,
    RunValidationArtifact,
    ValidationThresholds,
    build_validation_artifact,
    evaluate_validation_gate,
)


def _metrics(
    *,
    success: float = 0.8,
    recall: float = 0.8,
    mrr: float = 0.7,
    ndcg: float = 0.7,
    latency: float = 100.0,
    tokens: int = 1_000,
    tag_ndcg: float = 0.7,
    rank: int | None = 2,
) -> EvaluationMetrics:
    return EvaluationMetrics(
        success_at_k=success,
        recall_at_k=recall,
        mrr=mrr,
        ndcg_at_k=ndcg,
        latency_p95_ms=latency,
        returned_tokens=tokens,
        per_tag={
            "retrieval": QualityMetrics(
                success_at_k=success,
                recall_at_k=recall,
                mrr=mrr,
                ndcg_at_k=tag_ndcg,
            )
        },
        per_case_rank={"case-1": rank},
    )


def _evidence(search: EvaluationMetrics, holdout: EvaluationMetrics) -> EvaluationEvidence:
    return EvaluationEvidence(
        search=search,
        holdout=holdout,
        search_identity={"golden": "search-v1", "source": "abc"},
        holdout_identity={"golden": "holdout-v1", "source": "abc"},
        runtime_identity={"config": "live", "index": "idx", "runtime": "r1"},
    )


def _baseline(*, verified: bool = True) -> BaselineRecord:
    return BaselineRecord(
        candidate_id="live",
        promoted_at="2026-08-13T00:00:00+00:00",
        verification_status="verified" if verified else "legacy_unverified",
        schema_version=2 if verified else 1,
        validation_provenance=(
            BaselineValidationProvenance(
                run_id="run-0",
                artifact_name="validation.json",
                baseline_id="previous-live",
                candidate_id="live",
                validated_identity={"runtime": "r1"},
                verification_checks={},
            )
            if verified
            else None
        ),
    )


def test_models_reject_invalid_metrics_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="success_at_k"):
        _metrics(success=1.1)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationMetrics.model_validate(
            {
                "success": 0.8,
                "recall": 0.8,
                "mrr": 0.7,
                "ndcg": 0.7,
                "latency_ms": 100,
                "returned_tokens": 10,
                "unknown": True,
            }
        )


def test_models_accept_eval_summary_metric_names() -> None:
    metrics = EvaluationMetrics.model_validate(
        {
            "success@10": 0.8,
            "recall@10": 0.75,
            "mrr": 0.7,
            "ndcg@10": 0.72,
            "latency_ms": 100,
            "returned_tokens": 10,
        }
    )

    assert metrics.success_at_k == 0.8
    assert metrics.ndcg_at_k == 0.72


def test_eval_report_conversion_is_lossless_and_includes_breakdowns() -> None:
    report = EvalReport(
        top_k=3,
        results=[
            EvalResult(
                case=EvalCase(id="a", query="a", tags=("retrieval",)),
                rank_of_first_hit=3,
                top_k=3,
                relevances=(0.0, 0.0, 1.0),
                relevant_count=1,
                total_relevant=1,
                latency_ms=1.23456,
                returned_tokens=17,
            ),
            EvalResult(
                case=EvalCase(id="b", query="b", tags=("storage",)),
                rank_of_first_hit=None,
                top_k=3,
                latency_ms=9.87654,
                returned_tokens=23,
            ),
        ],
    )

    metrics = EvaluationMetrics.from_eval_report(report)

    assert metrics.success_at_k == report.success_at_k
    assert metrics.recall_at_k == report.recall_at_k
    assert metrics.mrr == report.mrr
    assert metrics.ndcg_at_k == report.ndcg_at_k
    assert metrics.latency_p95_ms == report.latency_p95_ms
    assert metrics.returned_tokens == 40
    assert metrics.per_case_rank == {"a": 3, "b": None}
    assert metrics.per_tag["retrieval"].mrr == report.by_tag()["retrieval"].mrr
    assert metrics.mrr != report.summary()["mrr"]


def test_candidate_beating_live_baseline_on_comparable_evidence_is_eligible() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = _evidence(
        _metrics(ndcg=0.75),
        _metrics(success=0.795, recall=0.795, mrr=0.695, ndcg=0.695),
    )

    artifact = build_validation_artifact(
        baseline_record=_baseline(),
        candidate_id="candidate",
        baseline=live,
        candidate=candidate,
        verification_checks={"pytest": True, "pipeline": True},
        current_identity=live.runtime_identity,
    )

    assert artifact.schema_version == 2
    assert artifact.gate.eligible is True
    assert artifact.gate.blocked_reasons == []
    assert artifact.to_dict()["candidate_id"] == "candidate"


def test_run_validation_artifact_enforces_shared_baseline_and_candidate_ids() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = _evidence(_metrics(ndcg=0.75), _metrics())
    validation = build_validation_artifact(
        baseline_record=_baseline(),
        candidate_id="candidate",
        baseline=live,
        candidate=candidate,
        current_identity=live.runtime_identity,
    )
    baseline = BaselineValidationArtifact(
        baseline_id="live",
        evidence=live,
        validated_identity=live.runtime_identity,
    )

    envelope = RunValidationArtifact(
        run_id="run-1",
        baseline=baseline,
        candidates={"candidate": validation},
    )

    assert envelope.schema_version == 2
    assert envelope.to_dict()["candidates"]["candidate"]["schema_version"] == 2
    with pytest.raises(ValidationError, match="candidate keys"):
        RunValidationArtifact(
            run_id="run-1",
            baseline=baseline,
            candidates={"wrong": validation},
        )
    with pytest.raises(ValidationError, match="envelope baseline"):
        RunValidationArtifact(
            run_id="run-1",
            baseline=baseline.model_copy(update={"baseline_id": "other"}),
            candidates={"candidate": validation},
        )


def test_legacy_baseline_and_missing_current_identity_block_promotion() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = _evidence(_metrics(ndcg=0.8), _metrics())

    gate = evaluate_validation_gate(
        baseline_record=_baseline(verified=False),
        baseline=live,
        candidate=candidate,
    )

    assert gate.eligible is False
    assert {reason.code for reason in gate.blocked_reasons} >= {
        "baseline_unverified",
        "current_identity_missing",
    }


def test_gate_reports_all_holdout_regressions_and_failed_checks() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = _evidence(
        _metrics(ndcg=0.8),
        _metrics(
            success=0.7,
            recall=0.7,
            mrr=0.6,
            ndcg=0.6,
            latency=130,
            tokens=1_300,
            tag_ndcg=0.6,
            rank=None,
        ),
    )

    gate = evaluate_validation_gate(
        baseline_record=_baseline(),
        baseline=live,
        candidate=candidate,
        verification_checks={"pytest": False},
        current_identity=live.runtime_identity,
    )

    codes = [reason.code for reason in gate.blocked_reasons]
    assert gate.eligible is False
    assert codes.count("holdout_metric_regression") == 4
    assert "holdout_latency_regression" in codes
    assert "holdout_returned_tokens_regression" in codes
    assert "holdout_tag_regression" in codes
    assert "holdout_case_rank_regression" in codes
    assert "verification_failed" in codes


def test_configured_thresholds_allow_bounded_holdout_regression() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = _evidence(
        _metrics(ndcg=0.71),
        _metrics(success=0.75, recall=0.75, mrr=0.65, ndcg=0.65),
    )
    thresholds = ValidationThresholds(
        min_search_improvement=0.005,
        max_holdout_regression={
            "success_at_k": 0.1,
            "recall_at_k": 0.1,
            "mrr": 0.1,
            "ndcg_at_k": 0.1,
        },
        max_per_tag_regression=0.1,
    )

    gate = evaluate_validation_gate(
        baseline_record=_baseline(),
        baseline=live,
        candidate=candidate,
        thresholds=thresholds,
        current_identity=live.runtime_identity,
    )

    assert gate.eligible is True


def test_incomplete_and_non_comparable_evidence_are_explicitly_blocked() -> None:
    live = _evidence(_metrics(), _metrics())
    candidate = EvaluationEvidence(
        search=_metrics(ndcg=0.8),
        search_identity={"golden": "different"},
        runtime_identity=live.runtime_identity,
    )

    gate = evaluate_validation_gate(
        baseline_record=_baseline(),
        baseline=live,
        candidate=candidate,
        current_identity=live.runtime_identity,
    )

    assert {reason.code for reason in gate.blocked_reasons} >= {
        "candidate_holdout_incomplete",
        "search_identity_mismatch",
        "holdout_identity_mismatch",
    }
