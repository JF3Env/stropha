"""Schema-v2 evaluation evidence and candidate-vs-live promotion gating."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from .baseline import BaselineRecord

if TYPE_CHECKING:
    from ..eval.harness import EvalReport

VALIDATION_SCHEMA_VERSION = 2

_QUALITY_METRICS = ("success_at_k", "recall_at_k", "mrr", "ndcg_at_k")
_METRIC_ALIASES = {
    "success": "success_at_k",
    "recall": "recall_at_k",
    "ndcg": "ndcg_at_k",
    "latency_ms": "latency_p95_ms",
}


class ValidationModel(BaseModel):
    """Strict base for durable validation artifacts."""

    model_config = {"extra": "forbid"}


class QualityMetrics(ValidationModel):
    """Retrieval quality metrics shared by aggregate and per-tag evidence."""

    success_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float

    @model_validator(mode="before")
    @classmethod
    def _normalize_metric_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for source, canonical in _METRIC_ALIASES.items():
            if source in normalized and canonical not in normalized:
                normalized[canonical] = normalized.pop(source)
        for key in tuple(normalized):
            target = next(
                (
                    name
                    for prefix, name in (
                        ("success@", "success_at_k"),
                        ("recall@", "recall_at_k"),
                        ("ndcg@", "ndcg_at_k"),
                    )
                    if key.startswith(prefix)
                ),
                None,
            )
            if target is not None and target not in normalized:
                normalized[target] = normalized.pop(key)
        return normalized

    @model_validator(mode="after")
    def _validate_quality(self) -> QualityMetrics:
        for name in _QUALITY_METRICS:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        return self

    def value(self, name: str) -> float:
        canonical = _METRIC_ALIASES.get(name, name)
        if canonical not in _QUALITY_METRICS:
            raise ValueError(f"unsupported quality metric {name!r}")
        return float(getattr(self, canonical))


class EvaluationMetrics(QualityMetrics):
    """Comparable aggregate, per-tag, and per-case evaluation measurements."""

    latency_p95_ms: float
    returned_tokens: int
    per_tag: dict[str, QualityMetrics] = Field(default_factory=dict)
    per_case_rank: dict[str, int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_costs_and_ranks(self) -> EvaluationMetrics:
        if not math.isfinite(self.latency_p95_ms) or self.latency_p95_ms < 0:
            raise ValueError("latency_p95_ms must be finite and non-negative")
        if self.returned_tokens < 0:
            raise ValueError("returned_tokens must be non-negative")
        invalid_ranks = [
            case_id for case_id, rank in self.per_case_rank.items() if rank is not None and rank < 1
        ]
        if invalid_ranks:
            raise ValueError("per_case_rank values must be positive integers or null")
        return self

    @classmethod
    def from_eval_report(cls, report: EvalReport) -> Self:
        """Convert an eval report without routing metrics through rounded summaries."""

        def quality(source: EvalReport) -> QualityMetrics:
            return QualityMetrics(
                success_at_k=source.success_at_k,
                recall_at_k=source.recall_at_k,
                mrr=source.mrr,
                ndcg_at_k=source.ndcg_at_k,
            )

        return cls(
            **quality(report).model_dump(),
            latency_p95_ms=report.latency_p95_ms,
            returned_tokens=sum(result.returned_tokens for result in report.results),
            per_tag={tag: quality(tag_report) for tag, tag_report in report.by_tag().items()},
            per_case_rank={
                result.case.id: result.rank_of_first_hit for result in report.results
            },
        )

    @classmethod
    def from_report(cls, report: EvalReport) -> Self:
        """Alias for :meth:`from_eval_report`."""

        return cls.from_eval_report(report)


class EvaluationEvidence(ValidationModel):
    """One subject's evidence for both frozen evaluation splits."""

    search: EvaluationMetrics | None = None
    holdout: EvaluationMetrics | None = None
    search_identity: dict[str, Any] = Field(default_factory=dict)
    holdout_identity: dict[str, Any] = Field(default_factory=dict)
    runtime_identity: dict[str, Any] = Field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.search is not None and self.holdout is not None


class ValidationThresholds(ValidationModel):
    """Configurable tolerances used by :func:`evaluate_validation_gate`."""

    primary_search_metric: str = "ndcg_at_k"
    min_search_improvement: float = 0.0
    max_holdout_regression: dict[str, float] = Field(
        default_factory=lambda: {
            "success_at_k": 0.01,
            "recall_at_k": 0.01,
            "mrr": 0.01,
            "ndcg_at_k": 0.01,
        }
    )
    max_latency_increase_ratio: float = 0.10
    max_latency_increase_ms: float = 0.0
    max_returned_tokens_increase_ratio: float = 0.10
    max_returned_tokens_increase: int = 0
    max_per_tag_regression: float = 0.02
    max_per_case_rank_regression: int = 0

    @model_validator(mode="after")
    def _validate_thresholds(self) -> ValidationThresholds:
        QualityMetrics(
            success_at_k=0.0,
            recall_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
        ).value(self.primary_search_metric)
        numeric = {
            "min_search_improvement": self.min_search_improvement,
            "max_latency_increase_ratio": self.max_latency_increase_ratio,
            "max_latency_increase_ms": self.max_latency_increase_ms,
            "max_returned_tokens_increase_ratio": self.max_returned_tokens_increase_ratio,
            "max_per_tag_regression": self.max_per_tag_regression,
        }
        numeric.update(
            {
                f"max_holdout_regression.{name}": value
                for name, value in self.max_holdout_regression.items()
            }
        )
        for name, value in numeric.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in self.max_holdout_regression:
            QualityMetrics(
                success_at_k=0.0,
                recall_at_k=0.0,
                mrr=0.0,
                ndcg_at_k=0.0,
            ).value(name)
        if self.max_returned_tokens_increase < 0:
            raise ValueError("max_returned_tokens_increase must be non-negative")
        if self.max_per_case_rank_regression < 0:
            raise ValueError("max_per_case_rank_regression must be non-negative")
        return self


class BlockedReason(ValidationModel):
    """Machine-readable reason that a candidate cannot be promoted."""

    code: str
    message: str
    split: Literal["search", "holdout"] | None = None
    metric: str | None = None
    baseline_value: float | int | None = None
    candidate_value: float | int | None = None
    threshold: float | int | None = None


class ValidationGate(ValidationModel):
    """Promotion decision with all failures retained, rather than fail-fast."""

    eligible: bool
    blocked_reasons: list[BlockedReason] = Field(default_factory=list)


class EvaluationValidationArtifact(ValidationModel):
    """Durable schema-v2 validation artifact for one candidate."""

    schema_version: Literal[2] = 2
    baseline_id: str
    candidate_id: str
    baseline: EvaluationEvidence
    candidate: EvaluationEvidence
    thresholds: ValidationThresholds = Field(default_factory=ValidationThresholds)
    verification_checks: dict[str, bool] = Field(default_factory=dict)
    validated_identity: dict[str, Any] = Field(default_factory=dict)
    gate: ValidationGate

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BaselineValidationArtifact(ValidationModel):
    """Baseline evidence shared by every candidate in one validation run."""

    schema_version: Literal[2] = 2
    baseline_id: str
    evidence: EvaluationEvidence
    validated_identity: dict[str, Any]

    @model_validator(mode="after")
    def _validate_baseline(self) -> BaselineValidationArtifact:
        if not self.baseline_id:
            raise ValueError("baseline_id must be non-empty")
        if not self.evidence.complete:
            raise ValueError("baseline evidence must include search and holdout metrics")
        if not self.validated_identity:
            raise ValueError("validated_identity must be non-empty")
        if self.evidence.runtime_identity != self.validated_identity:
            raise ValueError("validated_identity must match the baseline runtime identity")
        return self


class RunValidationArtifact(ValidationModel):
    """Strict schema-v2 envelope for one baseline and its candidate validations."""

    schema_version: Literal[2] = 2
    run_id: str
    baseline: BaselineValidationArtifact
    candidates: dict[str, EvaluationValidationArtifact]

    @model_validator(mode="after")
    def _validate_envelope(self) -> RunValidationArtifact:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.candidates:
            raise ValueError("candidates must contain at least one validation artifact")
        for candidate_id, artifact in self.candidates.items():
            if not candidate_id or candidate_id != artifact.candidate_id:
                raise ValueError("candidate keys must match artifact candidate_id values")
            if artifact.baseline_id != self.baseline.baseline_id:
                raise ValueError("candidate artifacts must reference the envelope baseline")
            if artifact.baseline != self.baseline.evidence:
                raise ValueError("candidate artifacts must use the envelope baseline evidence")
            if artifact.validated_identity != self.baseline.validated_identity:
                raise ValueError("candidate artifacts must use the envelope validated identity")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


RunValidationEnvelope = RunValidationArtifact


def _reason(
    code: str,
    message: str,
    *,
    split: Literal["search", "holdout"] | None = None,
    metric: str | None = None,
    baseline_value: float | int | None = None,
    candidate_value: float | int | None = None,
    threshold: float | int | None = None,
) -> BlockedReason:
    return BlockedReason(
        code=code,
        message=message,
        split=split,
        metric=metric,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        threshold=threshold,
    )


def _allowed_increase(baseline: float, ratio: float, absolute: float) -> float:
    return max(absolute, baseline * ratio)


def evaluate_validation_gate(
    *,
    baseline_record: BaselineRecord,
    baseline: EvaluationEvidence,
    candidate: EvaluationEvidence,
    thresholds: ValidationThresholds | None = None,
    verification_checks: dict[str, bool] | None = None,
    current_identity: dict[str, Any] | None = None,
) -> ValidationGate:
    """Compare a candidate with the live baseline and return every blocker."""

    limits = thresholds or ValidationThresholds()
    checks = verification_checks or {}
    reasons: list[BlockedReason] = []

    if not baseline_record.is_verified:
        reasons.append(
            _reason(
                "baseline_unverified",
                f"live baseline is {baseline_record.verification_status}",
            )
        )

    for subject_name, evidence in (("baseline", baseline), ("candidate", candidate)):
        if evidence.search is None:
            reasons.append(
                _reason(
                    f"{subject_name}_search_incomplete",
                    f"{subject_name} search evaluation is incomplete",
                    split="search",
                )
            )
        if evidence.holdout is None:
            reasons.append(
                _reason(
                    f"{subject_name}_holdout_incomplete",
                    f"{subject_name} holdout evaluation is incomplete",
                    split="holdout",
                )
            )

    if not baseline.search_identity or baseline.search_identity != candidate.search_identity:
        reasons.append(
            _reason(
                "search_identity_mismatch",
                "baseline and candidate search evidence are not comparable",
                split="search",
            )
        )
    if not baseline.holdout_identity or baseline.holdout_identity != candidate.holdout_identity:
        reasons.append(
            _reason(
                "holdout_identity_mismatch",
                "baseline and candidate holdout evidence are not comparable",
                split="holdout",
            )
        )
    if not baseline.runtime_identity or baseline.runtime_identity != candidate.runtime_identity:
        reasons.append(
            _reason(
                "runtime_identity_mismatch",
                "baseline and candidate runtime identities differ",
            )
        )
    if current_identity is None:
        reasons.append(
            _reason(
                "current_identity_missing",
                "current runtime identity was not supplied for validation",
            )
        )
    elif current_identity != baseline.runtime_identity:
        reasons.append(
            _reason(
                "current_identity_mismatch",
                "current runtime identity differs from the validated identity",
            )
        )

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    for name in failed_checks:
        reasons.append(
            _reason(
                "verification_failed",
                f"verification check {name!r} failed",
            )
        )

    if baseline.search is not None and candidate.search is not None:
        metric = limits.primary_search_metric
        baseline_value = baseline.search.value(metric)
        candidate_value = candidate.search.value(metric)
        improvement = candidate_value - baseline_value
        insufficient = improvement < limits.min_search_improvement
        if limits.min_search_improvement == 0.0:
            insufficient = improvement <= 0.0
        if insufficient:
            reasons.append(
                _reason(
                    "search_primary_not_improved",
                    f"candidate did not improve search {metric}",
                    split="search",
                    metric=metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    threshold=limits.min_search_improvement,
                )
            )

    if baseline.holdout is not None and candidate.holdout is not None:
        live = baseline.holdout
        arm = candidate.holdout
        for metric, maximum in limits.max_holdout_regression.items():
            live_value = live.value(metric)
            arm_value = arm.value(metric)
            if live_value - arm_value > maximum:
                reasons.append(
                    _reason(
                        "holdout_metric_regression",
                        f"candidate holdout {metric} regressed beyond threshold",
                        split="holdout",
                        metric=metric,
                        baseline_value=live_value,
                        candidate_value=arm_value,
                        threshold=maximum,
                    )
                )

        latency_limit = _allowed_increase(
            live.latency_p95_ms,
            limits.max_latency_increase_ratio,
            limits.max_latency_increase_ms,
        )
        if arm.latency_p95_ms - live.latency_p95_ms > latency_limit:
            reasons.append(
                _reason(
                    "holdout_latency_regression",
                    "candidate holdout latency regressed beyond threshold",
                    split="holdout",
                    metric="latency_p95_ms",
                    baseline_value=live.latency_p95_ms,
                    candidate_value=arm.latency_p95_ms,
                    threshold=latency_limit,
                )
            )

        token_limit = _allowed_increase(
            float(live.returned_tokens),
            limits.max_returned_tokens_increase_ratio,
            float(limits.max_returned_tokens_increase),
        )
        if arm.returned_tokens - live.returned_tokens > token_limit:
            reasons.append(
                _reason(
                    "holdout_returned_tokens_regression",
                    "candidate holdout returned tokens regressed beyond threshold",
                    split="holdout",
                    metric="returned_tokens",
                    baseline_value=live.returned_tokens,
                    candidate_value=arm.returned_tokens,
                    threshold=token_limit,
                )
            )

        for tag, live_tag in live.per_tag.items():
            arm_tag = arm.per_tag.get(tag)
            if arm_tag is None:
                reasons.append(
                    _reason(
                        "holdout_tag_missing",
                        f"candidate holdout evidence is missing tag {tag!r}",
                        split="holdout",
                        metric=tag,
                    )
                )
                continue
            for metric in _QUALITY_METRICS:
                live_value = live_tag.value(metric)
                arm_value = arm_tag.value(metric)
                if live_value - arm_value > limits.max_per_tag_regression:
                    reasons.append(
                        _reason(
                            "holdout_tag_regression",
                            f"candidate holdout tag {tag!r} {metric} regressed",
                            split="holdout",
                            metric=f"{tag}.{metric}",
                            baseline_value=live_value,
                            candidate_value=arm_value,
                            threshold=limits.max_per_tag_regression,
                        )
                    )

        for case_id, live_rank in live.per_case_rank.items():
            arm_rank = arm.per_case_rank.get(case_id)
            if live_rank is None:
                continue
            if arm_rank is None or arm_rank - live_rank > limits.max_per_case_rank_regression:
                reasons.append(
                    _reason(
                        "holdout_case_rank_regression",
                        f"candidate holdout rank regressed for case {case_id!r}",
                        split="holdout",
                        metric=case_id,
                        baseline_value=live_rank,
                        candidate_value=arm_rank,
                        threshold=limits.max_per_case_rank_regression,
                    )
                )

    return ValidationGate(eligible=not reasons, blocked_reasons=reasons)


def build_validation_artifact(
    *,
    baseline_record: BaselineRecord,
    candidate_id: str,
    baseline: EvaluationEvidence,
    candidate: EvaluationEvidence,
    thresholds: ValidationThresholds | None = None,
    verification_checks: dict[str, bool] | None = None,
    current_identity: dict[str, Any] | None = None,
) -> EvaluationValidationArtifact:
    """Evaluate the gate and package its inputs as a schema-v2 artifact."""

    limits = thresholds or ValidationThresholds()
    checks = dict(verification_checks or {})
    gate = evaluate_validation_gate(
        baseline_record=baseline_record,
        baseline=baseline,
        candidate=candidate,
        thresholds=limits,
        verification_checks=checks,
        current_identity=current_identity,
    )
    return EvaluationValidationArtifact(
        baseline_id=baseline_record.candidate_id,
        candidate_id=candidate_id,
        baseline=baseline,
        candidate=candidate,
        thresholds=limits,
        verification_checks=checks,
        validated_identity=dict(baseline.runtime_identity),
        gate=gate,
    )


__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "BaselineValidationArtifact",
    "BlockedReason",
    "EvaluationEvidence",
    "EvaluationMetrics",
    "EvaluationValidationArtifact",
    "QualityMetrics",
    "RunValidationArtifact",
    "RunValidationEnvelope",
    "ValidationGate",
    "ValidationThresholds",
    "build_validation_artifact",
    "evaluate_validation_gate",
]
