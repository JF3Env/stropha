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
from .store import ArtifactStore, ArtifactStoreError, CandidateRecord
from .trace import TraceRecord, TraceRecorder

__all__ = [
    "DEFAULT_TRACE_MAX_AGE_DAYS",
    "DIMENSION_SIGNS",
    "TIER_SCORE",
    "ArtifactStore",
    "ArtifactStoreError",
    "BadResult",
    "BadResultStore",
    "CandidateRecord",
    "FeedbackError",
    "LabelingContext",
    "PrunePlan",
    "ScoreCard",
    "Tier",
    "TierLabel",
    "TraceRecord",
    "TraceRecorder",
    "aggregate_score",
    "build_scorecard",
    "disk_usage",
    "label_hit",
    "label_hits",
    "outcome_score_decomposition",
    "prune",
    "resolve_bad_chunks",
    "scorecard_pareto_frontier",
    "tier_summary",
]
