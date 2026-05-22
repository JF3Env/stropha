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
from .store import ArtifactStore, ArtifactStoreError, CandidateRecord
from .trace import TraceRecord, TraceRecorder

__all__ = [
    "TIER_SCORE",
    "ArtifactStore",
    "ArtifactStoreError",
    "CandidateRecord",
    "LabelingContext",
    "Tier",
    "TierLabel",
    "TraceRecord",
    "TraceRecorder",
    "aggregate_score",
    "label_hit",
    "label_hits",
    "tier_summary",
]
