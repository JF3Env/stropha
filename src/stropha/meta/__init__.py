"""Meta-Harness apparatus — outer optimization loop over the adapter space.

See ``docs/architecture/stropha-meta-harness.md`` for the master design doc.
This package is built up phase-by-phase per the RFC:

- Phase 0 (this iteration): :mod:`trace` — append-only JSONL execution
  traces of every retrieval call. The substrate the Meta-Harness needs to
  reconstruct the full history (code + scores + traces) the paper requires.
- Phase 0+:  :mod:`store` — artifact store (candidates / scores / traces),
  :mod:`outcome` — tiered post-hoc labeling.
- Phase 1+: evaluator, snapshot, loop, proposers (pending).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .candidate_index import (
        CandidateIndex,
        CandidateIndexError,
        CandidateIndexIdentity,
        CandidateIndexProvider,
    )

from .baseline import (
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
from .code_candidate import (
    CodeCandidateError,
    CodeCandidateSpec,
    validate_code_candidate,
    validate_code_candidate_directory,
    validate_code_candidate_source,
)
from .evaluator import (
    CandidateEvaluationResult,
    EvaluationContext,
    EvaluationRecorder,
    EvaluatorError,
    evaluate_candidate,
    evaluate_from_golden,
    query_candidate_config,
)
from .feedback import (
    BadResult,
    BadResultStore,
    FeedbackError,
    learning_channel,
    resolve_bad_chunks,
)
from .fitness import (
    DIMENSION_SIGNS,
    ScoreCard,
    build_scorecard,
    outcome_score_decomposition,
)
from .fitness import pareto_frontier as scorecard_pareto_frontier
from .frontier import (
    FrontierEntry,
    FrontierReport,
    FrontierTracker,
    HoldoutEvaluatorFn,
    evaluate_frontier_on_holdout,
)
from .gate import (
    DEFAULT_PYTEST_ARGS,
    DEFAULT_TIMEOUT_S,
    GateError,
    GateResult,
    PytestGate,
)
from .index_cache import CacheEntry, IndexCache, IndexCacheError
from .loop import (
    CandidateCheckpoint,
    CheckpointCallback,
    EvaluatorFn,
    GateFn,
    Hypothesis,
    LoopCheckpoint,
    LoopError,
    LoopResult,
    Proposal,
    ProposerFn,
    RegisteredEvaluatorFn,
    run_loop,
)
from .outcome import (
    TIER_SCORE,
    LabelingContext,
    Tier,
    TierLabel,
    aggregate_score,
    label_hit,
    label_hits,
    tier_summary,
)
from .promote import (
    PromotionError,
    PromotionPlan,
    apply_plan,
    build_plan,
    render_summary,
)
from .prune import (
    DEFAULT_TRACE_MAX_AGE_DAYS,
    PrunePlan,
    disk_usage,
    prune,
)
from .reranker_proxy import SandboxedRerankerConfig, SandboxedRerankerProxy
from .run_cost import (
    RUN_COST_SCHEMA_VERSION,
    AggregatedMeasurement,
    CostMeasurement,
    RunCostArtifact,
    RunCostBreakdown,
    RunCostSample,
    aggregate_measurements,
    aggregate_run_cost,
)
from .sandbox import DEFAULT_TIMEOUT_S as SANDBOX_DEFAULT_TIMEOUT_S
from .sandbox import (
    RUNNER_MODULE,
    SandboxError,
    SandboxResult,
    SubprocessSandbox,
)
from .search_space import (
    DEFAULT_MANIFEST,
    Constraint,
    Dimension,
    ResolvedCandidateConfig,
    SearchSpace,
    SearchSpaceError,
    discover_gaps,
    load_search_space,
    resolve_candidate_config,
)
from .snapshot import (
    Snapshot,
    SnapshotError,
    create_snapshot,
    list_snapshots,
    prune_snapshots,
    resolve_sha,
)
from .store import ArtifactStore, ArtifactStoreError, CandidateRecord
from .trace import TraceRecord, TraceRecorder
from .validation import (
    VALIDATION_SCHEMA_VERSION,
    BaselineValidationArtifact,
    BlockedReason,
    EvaluationEvidence,
    EvaluationMetrics,
    EvaluationValidationArtifact,
    QualityMetrics,
    RunValidationArtifact,
    RunValidationEnvelope,
    ValidationGate,
    ValidationThresholds,
    build_validation_artifact,
    evaluate_validation_gate,
)

_LAZY_CANDIDATE_INDEX_EXPORTS = {
    "CANDIDATE_INDEX_CACHE_FORMAT",
    "CandidateIndex",
    "CandidateIndexError",
    "CandidateIndexIdentity",
    "CandidateIndexProvider",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_CANDIDATE_INDEX_EXPORTS:
        from . import candidate_index

        return getattr(candidate_index, name)
    raise AttributeError(name)

__all__ = [
    "BASELINE_FILENAME",
    "BASELINE_SCHEMA_VERSION",
    "CANDIDATE_INDEX_CACHE_FORMAT",
    "DEFAULT_MANIFEST",
    "DEFAULT_PYTEST_ARGS",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_TRACE_MAX_AGE_DAYS",
    "DIMENSION_SIGNS",
    "RUNNER_MODULE",
    "RUN_COST_SCHEMA_VERSION",
    "SANDBOX_DEFAULT_TIMEOUT_S",
    "TIER_SCORE",
    "VALIDATION_SCHEMA_VERSION",
    "AggregatedMeasurement",
    "ArtifactStore",
    "ArtifactStoreError",
    "BadResult",
    "BadResultStore",
    "BaselineError",
    "BaselineRecord",
    "BaselineValidationArtifact",
    "BaselineValidationProvenance",
    "BlockedReason",
    "CacheEntry",
    "CandidateCheckpoint",
    "CandidateEvaluationResult",
    "CandidateIndex",
    "CandidateIndexError",
    "CandidateIndexIdentity",
    "CandidateIndexProvider",
    "CandidateRecord",
    "CheckpointCallback",
    "CodeCandidateError",
    "CodeCandidateSpec",
    "Comparison",
    "Constraint",
    "CostMeasurement",
    "Dimension",
    "EvaluationContext",
    "EvaluationEvidence",
    "EvaluationMetrics",
    "EvaluationRecorder",
    "EvaluationValidationArtifact",
    "EvaluatorError",
    "EvaluatorFn",
    "FeedbackError",
    "FrontierEntry",
    "FrontierReport",
    "FrontierTracker",
    "GateError",
    "GateFn",
    "GateResult",
    "HoldoutEvaluatorFn",
    "Hypothesis",
    "IndexCache",
    "IndexCacheError",
    "LabelingContext",
    "LoopCheckpoint",
    "LoopError",
    "LoopResult",
    "PromotionError",
    "PromotionPlan",
    "Proposal",
    "ProposerFn",
    "PrunePlan",
    "PytestGate",
    "QualityMetrics",
    "RegisteredEvaluatorFn",
    "ResolvedCandidateConfig",
    "RunCostArtifact",
    "RunCostBreakdown",
    "RunCostSample",
    "RunValidationArtifact",
    "RunValidationEnvelope",
    "SandboxError",
    "SandboxResult",
    "SandboxedRerankerConfig",
    "SandboxedRerankerProxy",
    "ScoreCard",
    "SearchSpace",
    "SearchSpaceError",
    "Snapshot",
    "SnapshotError",
    "SubprocessSandbox",
    "Tier",
    "TierLabel",
    "TraceRecord",
    "TraceRecorder",
    "ValidationGate",
    "ValidationThresholds",
    "aggregate_measurements",
    "aggregate_run_cost",
    "aggregate_score",
    "apply_plan",
    "build_plan",
    "build_scorecard",
    "build_validation_artifact",
    "compare_candidates",
    "create_snapshot",
    "discover_gaps",
    "disk_usage",
    "evaluate_candidate",
    "evaluate_from_golden",
    "evaluate_frontier_on_holdout",
    "evaluate_validation_gate",
    "label_hit",
    "label_hits",
    "learning_channel",
    "list_snapshots",
    "load_search_space",
    "outcome_score_decomposition",
    "prune",
    "prune_snapshots",
    "query_candidate_config",
    "read_baseline",
    "regressions",
    "render_summary",
    "resolve_bad_chunks",
    "resolve_candidate_config",
    "resolve_sha",
    "run_loop",
    "scorecard_pareto_frontier",
    "tier_summary",
    "validate_code_candidate",
    "validate_code_candidate_directory",
    "validate_code_candidate_source",
    "write_baseline",
    "write_verified_baseline",
]
