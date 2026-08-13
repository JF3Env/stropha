"""Tests for per-run exact/estimated cost primitives."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stropha.meta.run_cost import (
    CostMeasurement,
    RunCostSample,
    aggregate_measurements,
    aggregate_run_cost,
)


def test_measurement_prefers_exact_without_double_counting_estimate() -> None:
    aggregate = aggregate_measurements(
        [
            CostMeasurement(exact=100, estimated=120),
            CostMeasurement(estimated=50),
            CostMeasurement(),
        ]
    )

    assert aggregate.exact == 100
    assert aggregate.estimated == 50
    assert aggregate.total == 150
    assert aggregate.exact_samples == 1
    assert aggregate.estimated_samples == 1
    assert aggregate.missing_samples == 1
    assert aggregate.fully_exact is False


def test_cost_models_reject_negative_or_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        CostMeasurement(estimated=-1)
    with pytest.raises(ValidationError, match="non-negative"):
        RunCostSample(candidate_states={"failed": -1})


def test_run_aggregation_sums_counters_states_and_usage() -> None:
    artifact = aggregate_run_cost(
        "run-1",
        [
            RunCostSample(
                scope="search",
                proposer_calls=1,
                proposer_tokens=CostMeasurement(exact=100, estimated=110),
                indexing_files=10,
                cache_hits=1,
                retrieval_queries=5,
                retrieval_tokens=CostMeasurement(estimated=500),
                retrieval_duration_ms=CostMeasurement(exact=250),
                candidate_states={"scored": 1},
                artifact_bytes=1_000,
            ),
            RunCostSample(
                scope="holdout_validation",
                proposer_calls=2,
                proposer_fallbacks=1,
                proposer_tokens=CostMeasurement(estimated=75),
                indexing_chunks=20,
                cache_misses=1,
                retrieval_queries=3,
                retrieval_tokens=CostMeasurement(exact=300),
                candidate_states={"scored": 1, "failed": 1},
                verification_duration_ms=CostMeasurement(exact=50),
                artifact_bytes=500,
            ),
        ],
    )

    assert artifact.proposer_calls == 3
    assert artifact.proposer_fallbacks == 1
    assert artifact.proposer_tokens.total == 175
    assert artifact.retrieval_tokens.total == 800
    assert artifact.candidate_states == {"scored": 2, "failed": 1}
    assert artifact.artifact_bytes == 1_500
    assert artifact.monetary_cost_usd is None
    assert artifact.monetary_cost_status == "unavailable"
    assert artifact.to_dict()["proposer_tokens"]["fully_exact"] is False
    assert artifact.by_scope["search"].retrieval_queries == 5
    assert artifact.by_scope["holdout_validation"].retrieval_queries == 3


def test_monetary_cost_requires_exact_cost_for_every_billable_sample() -> None:
    exact = aggregate_run_cost(
        "exact",
        [
            RunCostSample(
                proposer_tokens=CostMeasurement(exact=100),
                monetary_cost_usd=CostMeasurement(exact=0.01),
            ),
            RunCostSample(
                proposer_tokens=CostMeasurement(exact=200),
                monetary_cost_usd=CostMeasurement(exact=0.02),
            ),
        ],
    )
    mixed = aggregate_run_cost(
        "mixed",
        [
            RunCostSample(
                proposer_tokens=CostMeasurement(exact=100),
                monetary_cost_usd=CostMeasurement(exact=0.01),
            ),
            RunCostSample(
                proposer_tokens=CostMeasurement(estimated=200),
                monetary_cost_usd=CostMeasurement(estimated=0.02),
            ),
        ],
    )

    assert exact.monetary_cost_usd == pytest.approx(0.03)
    assert exact.monetary_cost_status == "exact"
    assert mixed.monetary_cost_usd is None
    assert mixed.monetary_cost_status == "unavailable"


def test_empty_run_has_unknown_monetary_cost() -> None:
    artifact = aggregate_run_cost("empty", [])

    assert artifact.monetary_cost_usd is None
    assert artifact.monetary_cost_status == "unavailable"
    assert artifact.proposer_tokens.fully_exact is False
