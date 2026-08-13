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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..logging import get_logger
from .fitness import FITNESS_VERSION
from .store import CandidateRecord

if TYPE_CHECKING:
    from .validation import EvaluationValidationArtifact

log = get_logger(__name__)


BASELINE_FILENAME = "baseline.json"
BASELINE_SCHEMA_VERSION = 2
_VERIFICATION_STATUSES = {"verified", "unverified", "legacy_unverified"}


class BaselineError(RuntimeError):
    """Raised on malformed baseline records (parse / schema failures)."""


class BaselineValidationProvenance(BaseModel):
    """Durable pointer and identity evidence for a verified baseline."""

    model_config = {"extra": "forbid"}

    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1)
    artifact_name: str = Field(pattern=r"^[^/\\]+\.json$")
    baseline_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    validated_identity: dict[str, Any] = Field(min_length=1)
    verification_checks: dict[str, bool]


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
    fitness_version: int = FITNESS_VERSION
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BASELINE_SCHEMA_VERSION
    verification_status: str = "unverified"
    validation_provenance: BaselineValidationProvenance | None = None

    @property
    def is_verified(self) -> bool:
        return (
            self.schema_version == BASELINE_SCHEMA_VERSION
            and self.verification_status == "verified"
            and self.validation_provenance is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "promoted_at": self.promoted_at,
            "fitness": self.fitness,
            "fitness_version": self.fitness_version,
            "label": self.label,
            "meta": self.meta,
            "schema_version": self.schema_version,
            "verification_status": self.verification_status,
            "validation_provenance": (
                self.validation_provenance.model_dump(mode="json")
                if self.validation_provenance is not None
                else None
            ),
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
        raise BaselineError(f"baseline at {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineError(
            f"baseline at {path} must be a JSON object at the top level, "
            f"got {type(payload).__name__}"
        )
    if "candidate_id" not in payload:
        raise BaselineError(f"baseline at {path} missing required field 'candidate_id'")
    fitness = payload.get("fitness")
    fitness_version = payload.get("fitness_version")
    raw_schema_version = payload.get("schema_version")
    if raw_schema_version is None:
        schema_version = 1
        verification_status = "legacy_unverified"
    else:
        if raw_schema_version != BASELINE_SCHEMA_VERSION:
            raise BaselineError(
                f"baseline at {path}: unsupported schema_version {raw_schema_version!r}"
            )
        schema_version = BASELINE_SCHEMA_VERSION
        verification_status = str(payload.get("verification_status") or "unverified")
        if verification_status not in _VERIFICATION_STATUSES:
            raise BaselineError(
                f"baseline at {path}: invalid verification_status {verification_status!r}"
            )
    validation_provenance = None
    raw_provenance = payload.get("validation_provenance")
    if raw_provenance is not None:
        try:
            validation_provenance = BaselineValidationProvenance.model_validate(raw_provenance)
        except ValidationError as exc:
            raise BaselineError(f"baseline at {path}: invalid validation provenance: {exc}") from exc
    if verification_status == "verified" and validation_provenance is None:
        raise BaselineError(f"baseline at {path}: verified baseline requires validation provenance")
    if (
        validation_provenance is not None
        and validation_provenance.candidate_id != str(payload["candidate_id"])
    ):
        raise BaselineError(
            f"baseline at {path}: validation provenance candidate_id does not match baseline"
        )
    # Tolerate floats / ints / None; reject anything else so a stale
    # human-edited file can't poison the comparison.
    if fitness is not None and not isinstance(fitness, (int, float)):
        raise BaselineError(
            f"baseline at {path}: 'fitness' must be a number or null, got {type(fitness).__name__}"
        )
    return BaselineRecord(
        candidate_id=str(payload["candidate_id"]),
        promoted_at=str(payload.get("promoted_at") or _now_utc_iso()),
        fitness=float(fitness) if fitness is not None else None,
        fitness_version=(int(fitness_version) if fitness_version == FITNESS_VERSION else 0),
        label=payload.get("label"),
        meta=payload.get("meta") or {},
        schema_version=schema_version,
        verification_status=verification_status,
        validation_provenance=validation_provenance,
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


def write_verified_baseline(
    meta_dir: Path,
    *,
    candidate: CandidateRecord,
    validation: EvaluationValidationArtifact,
    run_id: str,
    artifact_name: str = "validation.json",
    promoted_at: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> BaselineRecord:
    """Persist an eligible candidate as verified with explicit validation provenance."""

    from .validation import EvaluationValidationArtifact

    if not isinstance(validation, EvaluationValidationArtifact):
        raise BaselineError("validation must be an EvaluationValidationArtifact")
    if validation.candidate_id != candidate.id:
        raise BaselineError("validation candidate_id does not match the promoted candidate")
    if not validation.gate.eligible:
        raise BaselineError("cannot write a verified baseline from an ineligible validation")
    try:
        provenance = BaselineValidationProvenance(
            run_id=run_id,
            artifact_name=artifact_name,
            baseline_id=validation.baseline_id,
            candidate_id=validation.candidate_id,
            validated_identity=validation.validated_identity,
            verification_checks=validation.verification_checks,
        )
    except ValidationError as exc:
        raise BaselineError(f"invalid validation provenance: {exc}") from exc

    meta_dir.mkdir(parents=True, exist_ok=True)
    fitness = candidate.fitness()
    record = BaselineRecord(
        candidate_id=candidate.id,
        promoted_at=promoted_at or _now_utc_iso(),
        fitness=fitness if fitness != float("-inf") else None,
        label=candidate.label,
        meta=extra_meta or {},
        verification_status="verified",
        validation_provenance=provenance,
    )
    path = _baseline_path(meta_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    log.info(
        "baseline.written",
        candidate=record.candidate_id,
        fitness=record.fitness,
        verification_status=record.verification_status,
        validation_run_id=run_id,
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
    if baseline.fitness is None or baseline.fitness_version != FITNESS_VERSION:
        return []
    baseline_fitness = baseline.fitness

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
    flagged = [c for c in comparisons if not c.is_baseline and c.delta > threshold]
    flagged.sort(key=lambda c: c.delta, reverse=True)
    return flagged


__all__ = [
    "BASELINE_FILENAME",
    "BASELINE_SCHEMA_VERSION",
    "BaselineError",
    "BaselineRecord",
    "BaselineValidationProvenance",
    "Comparison",
    "compare_candidates",
    "read_baseline",
    "regressions",
    "write_baseline",
    "write_verified_baseline",
]
