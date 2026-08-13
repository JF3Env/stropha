"""Per-run cost measurements and deterministic aggregation primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

RUN_COST_SCHEMA_VERSION = 1
CostScope = Literal["search", "holdout_validation", "verification", "other"]


class CostModel(BaseModel):
    model_config = {"extra": "forbid"}


class CostMeasurement(CostModel):
    """One quantity with provider-reported and/or estimated values.

    ``exact`` wins when both fields are present. Keeping both values in the
    artifact makes estimation error auditable without double-counting it.
    """

    exact: float | None = None
    estimated: float | None = None

    @model_validator(mode="after")
    def _validate_values(self) -> CostMeasurement:
        for name in ("exact", "estimated"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        return self

    @property
    def available(self) -> float | None:
        return self.exact if self.exact is not None else self.estimated

    @property
    def is_exact(self) -> bool:
        return self.exact is not None


class AggregatedMeasurement(CostModel):
    """Aggregate that separates exact values from estimate fallbacks."""

    exact: float = 0.0
    estimated: float = 0.0
    total: float = 0.0
    exact_samples: int = 0
    estimated_samples: int = 0
    missing_samples: int = 0

    @property
    def fully_exact(self) -> bool:
        return self.exact_samples > 0 and self.estimated_samples == 0 and self.missing_samples == 0


class RunCostSample(CostModel):
    """Incremental usage captured by one run operation or worker."""

    scope: CostScope = "other"
    proposer_calls: int = 0
    proposer_fallbacks: int = 0
    proposer_tokens: CostMeasurement = Field(default_factory=CostMeasurement)
    indexing_files: int = 0
    indexing_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retrieval_queries: int = 0
    retrieval_tokens: CostMeasurement = Field(default_factory=CostMeasurement)
    retrieval_duration_ms: CostMeasurement = Field(default_factory=CostMeasurement)
    candidate_states: dict[str, int] = Field(default_factory=dict)
    verification_duration_ms: CostMeasurement = Field(default_factory=CostMeasurement)
    artifact_bytes: int = 0
    monetary_cost_usd: CostMeasurement = Field(default_factory=CostMeasurement)

    @model_validator(mode="after")
    def _validate_counts(self) -> RunCostSample:
        counts = {
            "proposer_calls": self.proposer_calls,
            "proposer_fallbacks": self.proposer_fallbacks,
            "indexing_files": self.indexing_files,
            "indexing_chunks": self.indexing_chunks,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retrieval_queries": self.retrieval_queries,
            "artifact_bytes": self.artifact_bytes,
            **{f"candidate_states.{key}": value for key, value in self.candidate_states.items()},
        }
        for name, value in counts.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        return self


class RunCostBreakdown(CostModel):
    """Aggregated usage for a run or one run scope."""

    proposer_calls: int = 0
    proposer_fallbacks: int = 0
    proposer_tokens: AggregatedMeasurement = Field(default_factory=AggregatedMeasurement)
    indexing_files: int = 0
    indexing_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retrieval_queries: int = 0
    retrieval_tokens: AggregatedMeasurement = Field(default_factory=AggregatedMeasurement)
    retrieval_duration_ms: AggregatedMeasurement = Field(default_factory=AggregatedMeasurement)
    candidate_states: dict[str, int] = Field(default_factory=dict)
    verification_duration_ms: AggregatedMeasurement = Field(default_factory=AggregatedMeasurement)
    artifact_bytes: int = 0
    monetary_cost_usd: float | None = None
    monetary_cost_status: Literal["exact", "unavailable"] = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        for field_name in (
            "proposer_tokens",
            "retrieval_tokens",
            "retrieval_duration_ms",
            "verification_duration_ms",
        ):
            payload[field_name]["fully_exact"] = getattr(self, field_name).fully_exact
        return payload


class RunCostArtifact(RunCostBreakdown):
    """Aggregated ``cost.json`` payload for a single run."""

    schema_version: Literal[1] = 1
    run_id: str
    by_scope: dict[CostScope, RunCostBreakdown] = Field(default_factory=dict)


def aggregate_measurements(
    measurements: Iterable[CostMeasurement],
) -> AggregatedMeasurement:
    """Use exact values when available and estimates only as fallbacks."""

    exact = 0.0
    estimated = 0.0
    exact_samples = 0
    estimated_samples = 0
    missing_samples = 0
    for measurement in measurements:
        if measurement.exact is not None:
            exact += measurement.exact
            exact_samples += 1
        elif measurement.estimated is not None:
            estimated += measurement.estimated
            estimated_samples += 1
        else:
            missing_samples += 1
    return AggregatedMeasurement(
        exact=exact,
        estimated=estimated,
        total=exact + estimated,
        exact_samples=exact_samples,
        estimated_samples=estimated_samples,
        missing_samples=missing_samples,
    )


def _aggregate_breakdown(rows: list[RunCostSample]) -> RunCostBreakdown:
    candidate_states: dict[str, int] = {}
    for row in rows:
        for state, count in row.candidate_states.items():
            candidate_states[state] = candidate_states.get(state, 0) + count

    billable_rows = [
        row
        for row in rows
        if row.proposer_calls > 0
        or row.proposer_tokens.available not in (None, 0.0)
        or row.monetary_cost_usd.available not in (None, 0.0)
    ]
    exact_money = bool(billable_rows) and all(
        row.proposer_tokens.exact is not None and row.monetary_cost_usd.exact is not None
        for row in billable_rows
    )
    monetary_cost = (
        sum(float(row.monetary_cost_usd.exact or 0.0) for row in billable_rows)
        if exact_money
        else None
    )

    return RunCostBreakdown(
        proposer_calls=sum(row.proposer_calls for row in rows),
        proposer_fallbacks=sum(row.proposer_fallbacks for row in rows),
        proposer_tokens=aggregate_measurements(row.proposer_tokens for row in rows),
        indexing_files=sum(row.indexing_files for row in rows),
        indexing_chunks=sum(row.indexing_chunks for row in rows),
        cache_hits=sum(row.cache_hits for row in rows),
        cache_misses=sum(row.cache_misses for row in rows),
        retrieval_queries=sum(row.retrieval_queries for row in rows),
        retrieval_tokens=aggregate_measurements(row.retrieval_tokens for row in rows),
        retrieval_duration_ms=aggregate_measurements(row.retrieval_duration_ms for row in rows),
        candidate_states=candidate_states,
        verification_duration_ms=aggregate_measurements(
            row.verification_duration_ms for row in rows
        ),
        artifact_bytes=sum(row.artifact_bytes for row in rows),
        monetary_cost_usd=monetary_cost,
        monetary_cost_status="exact" if exact_money else "unavailable",
    )


def aggregate_run_cost(
    run_id: str,
    samples: Iterable[RunCostSample],
) -> RunCostArtifact:
    """Aggregate samples and retain separate search/validation usage."""

    rows = list(samples)
    total = _aggregate_breakdown(rows)
    scopes: dict[CostScope, RunCostBreakdown] = {}
    for scope in ("search", "holdout_validation", "verification", "other"):
        scoped_rows = [row for row in rows if row.scope == scope]
        if scoped_rows:
            scopes[scope] = _aggregate_breakdown(scoped_rows)
    return RunCostArtifact(
        **total.model_dump(),
        run_id=run_id,
        by_scope=scopes,
    )


__all__ = [
    "RUN_COST_SCHEMA_VERSION",
    "AggregatedMeasurement",
    "CostMeasurement",
    "RunCostArtifact",
    "RunCostBreakdown",
    "RunCostSample",
    "aggregate_measurements",
    "aggregate_run_cost",
]
