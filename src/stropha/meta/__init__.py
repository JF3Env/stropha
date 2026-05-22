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

from .evaluator import (
    EvaluatorError,
    evaluate_candidate,
    evaluate_from_golden,
)
from .feedback import (
    BadResult,
    BadResultStore,
    FeedbackError,
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
from .index_cache import CacheEntry, IndexCache, IndexCacheError
from .loop import EvaluatorFn, LoopError, LoopResult, ProposerFn, run_loop
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
from .prune import (
    DEFAULT_TRACE_MAX_AGE_DAYS,
    PrunePlan,
    disk_usage,
    prune,
)
from .search_space import (
    DEFAULT_MANIFEST,
    Constraint,
    Dimension,
    SearchSpace,
    SearchSpaceError,
    discover_gaps,
    load_search_space,
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

__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_TRACE_MAX_AGE_DAYS",
    "DIMENSION_SIGNS",
    "TIER_SCORE",
    "ArtifactStore",
    "ArtifactStoreError",
    "BadResult",
    "BadResultStore",
    "CacheEntry",
    "CandidateRecord",
    "Constraint",
    "Dimension",
    "EvaluatorError",
    "EvaluatorFn",
    "FeedbackError",
    "FrontierEntry",
    "FrontierReport",
    "FrontierTracker",
    "HoldoutEvaluatorFn",
    "IndexCache",
    "IndexCacheError",
    "LabelingContext",
    "LoopError",
    "LoopResult",
    "ProposerFn",
    "PrunePlan",
    "ScoreCard",
    "SearchSpace",
    "SearchSpaceError",
    "Snapshot",
    "SnapshotError",
    "Tier",
    "TierLabel",
    "TraceRecord",
    "TraceRecorder",
    "aggregate_score",
    "build_scorecard",
    "create_snapshot",
    "discover_gaps",
    "disk_usage",
    "evaluate_candidate",
    "evaluate_from_golden",
    "evaluate_frontier_on_holdout",
    "label_hit",
    "label_hits",
    "list_snapshots",
    "load_search_space",
    "outcome_score_decomposition",
    "prune",
    "prune_snapshots",
    "resolve_bad_chunks",
    "resolve_sha",
    "run_loop",
    "scorecard_pareto_frontier",
    "tier_summary",
]
