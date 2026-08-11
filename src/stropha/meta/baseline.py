"""Production-baseline tracking + regression alerts (WIN-204).

The Meta-Harness produces candidates; the operator promotes one into the live
unified config via ``stropha meta promote --apply``. After that point
the operator's question becomes:

    "Is the bandit finding something better than what I just promoted?"

This module is the answer. ``meta promote --apply`` writes a baseline
record (``meta_dir/baseline.json``) capturing the candidate id, its
fitness at promotion time, and the timestamp. ``meta report`` reads the
record and renders a regression-alert section that flags every
candidate whose fitness beats the baseline by more than a configurable
margin.

Why a side-car file instead of "is_baseline" on CandidateRecord:

- Promotion is monotonic-by-time, not by candidate identity — the same
  candidate id should not gain a flag retroactively just because it
  matches the live config.
- The baseline lives across meta-loop runs; it survives ``meta prune``
  even when the underlying candidate has been compacted.
- A JSON side-car is greppable from shell, scriptable, and trivially
  testable. It does not require schema bumps in the SQLite index.

A baseline reads as "the configuration the operator believed in at
time T". Comparisons against it are honest signals to the operator;
they are not used by any automated promotion (still human-gated per
§3 spirit / WIN-202).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .store import CandidateRecord

log = get_logger(__name__)


BASELINE_FILENAME = "baseline.json"


class BaselineError(RuntimeError):
    """Raised on malformed baseline records (parse / schema failures)."""


@dataclass(frozen=True)
class BaselineRecord:
    """The currently-promoted candidate's metadata.

    ``fitness`` is captured at promotion time so future report runs can
    compute deltas without having to re-evaluate the promoted candidate.
    A ``None`` fitness means the candidate had no scores at promotion
    time (rare — promotion is normally preceded by a meta loop pass).
    """

    candidate_id: str
    promoted_at: str
    fitness: float | None = None
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "promoted_at": self.promoted_at,
            "fitness": self.fitness,
            "label": self.label,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Comparison:
    """One candidate's standing relative to the baseline.

    ``delta`` is ``candidate.fitness - baseline.fitness``. Positive
    means the candidate is better than the live config; that's the
    signal the regression-alerts section flags.
    """

    candidate: CandidateRecord
    candidate_fitness: float
    baseline_fitness: float
    delta: float
    is_baseline: bool = False
    """``True`` when the candidate is the currently-promoted one (delta
    is 0.0 by definition). Lets the UI annotate the baseline row
    differently from "neutral, no comparison ran"."""


def _baseline_path(meta_dir: Path) -> Path:
    return meta_dir / BASELINE_FILENAME


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_baseline(meta_dir: Path) -> BaselineRecord | None:
    """Load the baseline record. Returns ``None`` when none exists.

    Raises :class:`BaselineError` only when the file exists but is
    malformed — a missing file is a legitimate "never promoted" state,
    not an error.
    """
    path = _baseline_path(meta_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BaselineError(
            f"baseline at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BaselineError(
            f"baseline at {path} must be a JSON object at the top level, "
            f"got {type(payload).__name__}"
        )
    if "candidate_id" not in payload:
        raise BaselineError(
            f"baseline at {path} missing required field 'candidate_id'"
        )
    fitness = payload.get("fitness")
    # Tolerate floats / ints / None; reject anything else so a stale
    # human-edited file can't poison the comparison.
    if fitness is not None and not isinstance(fitness, (int, float)):
        raise BaselineError(
            f"baseline at {path}: 'fitness' must be a number or null, "
            f"got {type(fitness).__name__}"
        )
    return BaselineRecord(
        candidate_id=str(payload["candidate_id"]),
        promoted_at=str(payload.get("promoted_at") or _now_utc_iso()),
        fitness=float(fitness) if fitness is not None else None,
        label=payload.get("label"),
        meta=payload.get("meta") or {},
    )


def write_baseline(
    meta_dir: Path,
    *,
    candidate: CandidateRecord,
    promoted_at: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> BaselineRecord:
    """Persist the baseline record after a successful promotion.

    Atomic write via temp file + rename — a SIGKILL mid-write leaves
    either the previous baseline or the new one, never a half-written
    JSON blob.

    The fitness stored is whatever the candidate carries at promotion
    time. ``meta_dir`` is created if missing.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)
    fitness = candidate.fitness()
    record = BaselineRecord(
        candidate_id=candidate.id,
        promoted_at=promoted_at or _now_utc_iso(),
        # CandidateRecord.fitness() returns -inf when no scores; persist
        # None instead so the JSON stays portable (infinity isn't valid
        # JSON unless we set allow_nan).
        fitness=fitness if fitness != float("-inf") else None,
        label=candidate.label,
        meta=extra_meta or {},
    )
    path = _baseline_path(meta_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    log.info(
        "baseline.written",
        candidate=record.candidate_id,
        fitness=record.fitness,
        meta_dir=str(meta_dir),
    )
    return record


def compare_candidates(
    candidates: list[CandidateRecord],
    baseline: BaselineRecord,
) -> list[Comparison]:
    """Build a Comparison entry per candidate vs the baseline.

    Candidates with no fitness (no scores) are skipped silently — they
    cannot be compared. The baseline record itself, when present in the
    candidate list, gets a special row with ``is_baseline=True`` and
    delta=0.0 even if its in-archive fitness drifted slightly (e.g.
    scores rewritten by a re-evaluation since promotion); the stored
    baseline fitness is authoritative for the "what got promoted" view.
    """
    # ``float("-inf")`` sentinel when the operator promoted blind (no fitness on
    # the baseline). Surfacing comparisons still works; the delta is 0 for
    # everyone so the UI can fall back to "no baseline scoring available".
    baseline_fitness = float("-inf") if baseline.fitness is None else baseline.fitness

    out: list[Comparison] = []
    for cand in candidates:
        fit = cand.fitness()
        if fit == float("-inf"):
            continue
        is_baseline = cand.id == baseline.candidate_id
        delta = 0.0 if is_baseline else fit - baseline_fitness
        out.append(
            Comparison(
                candidate=cand,
                candidate_fitness=fit,
                baseline_fitness=baseline_fitness,
                delta=delta,
                is_baseline=is_baseline,
            )
        )
    return out


def regressions(
    comparisons: list[Comparison],
    *,
    threshold: float = 0.0,
) -> list[Comparison]:
    """Filter to candidates whose fitness beats the baseline.

    ``threshold`` is the minimum positive delta to flag — set it above
    zero to suppress noise (e.g. floating-point ties or 0.001-quality
    bumps that aren't worth a promotion). Default is 0.0 which catches
    every strict improvement.

    The baseline candidate itself is never flagged (delta=0 by
    construction). Returns sorted by delta descending so the biggest
    wins appear first.
    """
    flagged = [
        c for c in comparisons
        if not c.is_baseline and c.delta > threshold
    ]
    flagged.sort(key=lambda c: c.delta, reverse=True)
    return flagged


__all__ = [
    "BASELINE_FILENAME",
    "BaselineError",
    "BaselineRecord",
    "Comparison",
    "compare_candidates",
    "read_baseline",
    "regressions",
    "write_baseline",
]
