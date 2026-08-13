"""Tests for learned per-repository RRF stream weights (WIN-380)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from stropha.meta.outcome import Tier, TierLabel
from stropha.meta.stream_weights import (
    MIN_ACTIVATION_EXAMPLES,
    StreamWeightProfile,
    StreamWeightProfiles,
    fit_stream_weights,
    validate_stream_weight_profile,
    write_stream_weight_profile,
)


def _hit(repo: str, streams: dict[str, float] | None) -> dict:
    return {
        "chunk_id": f"{repo}:{len(streams or {})}",
        "repo": {"normalized_key": repo},
        "streams": streams,
    }


def _label(score: float) -> TierLabel:
    tier = Tier.T1 if score > 0 else Tier.T6 if score < 0 else Tier.T7
    return TierLabel(
        chunk_id="chunk",
        tier=tier,
        score=score,
        rationale="test",
    )


def test_fit_stream_weights_rewards_useful_streams_with_regularization() -> None:
    profile = fit_stream_weights(
        [
            _hit("repo-a", {"dense": 0.02}),
            _hit("repo-a", {"dense": 0.01, "sparse": 0.01}),
            _hit("repo-a", {"sparse": 0.02}),
        ],
        [_label(1.0), _label(1.0), _label(-1.0)],
        repo_key="repo-a",
    )

    assert profile is not None
    assert profile.observations == {"dense": 2, "sparse": 2}
    assert profile.weights["dense"] > 1.0
    assert profile.weights["sparse"] < 1.0
    assert sum(profile.weights.values()) / 2 == pytest.approx(1.0)
    assert profile.weights["dense"] < 1.1


def test_fit_stream_weights_ignores_other_repos_and_unattributed_hits() -> None:
    assert fit_stream_weights(
        [_hit("repo-b", {"dense": 0.02}), _hit("repo-a", None)],
        [_label(1.0), _label(1.0)],
        repo_key="repo-a",
    ) is None


def test_fit_stream_weights_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_stream_weights([_hit("repo-a", {"dense": 1.0})], [], repo_key="repo-a")


def test_profile_round_trip_and_hot_reload(tmp_path: Path) -> None:
    profiles = StreamWeightProfiles(tmp_path)
    assert profiles.weights() == {}

    profile = StreamWeightProfile(
        repo="github.com/acme/app",
        weights={"dense": 1.2, "sparse": 0.8},
        observations={"dense": 50, "sparse": 50},
        mean_scores={"dense": 0.2, "sparse": -0.1},
        generated_at="2026-07-24T00:00:00+00:00",
        examples=MIN_ACTIVATION_EXAMPLES,
        source_counts={"outcome": MIN_ACTIVATION_EXAMPLES},
        since_sha="a",
        head_sha="b",
        validated=True,
        validation={"search_ndcg_delta": 0.02, "holdout_ndcg_delta": 0.0},
    )
    path = write_stream_weight_profile(tmp_path, profile)

    assert path.is_file()
    assert profiles.weights() == {
        "github.com/acme/app": {"dense": 1.2, "sparse": 0.8},
    }
    assert profiles.fingerprint() != "none"


def test_loader_ignores_invalid_profiles(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text(json.dumps({"version": 1}), encoding="utf-8")

    assert StreamWeightProfiles(tmp_path).weights() == {}


def test_loader_rejects_inconsistent_evidence_counts(tmp_path: Path) -> None:
    raw = StreamWeightProfile(
        repo="repo-a",
        weights={"dense": 1.0},
        observations={"dense": 50},
        mean_scores={"dense": 0.1},
        generated_at="2026-07-24T00:00:00+00:00",
        examples=50,
        source_counts={"outcome": 49},
        since_sha="a",
        head_sha="b",
        validated=True,
        validation={"search_ndcg_delta": 0.02, "holdout_ndcg_delta": 0.0},
    )
    (tmp_path / "bad.json").write_text(json.dumps(raw.to_dict()), encoding="utf-8")

    assert StreamWeightProfiles(tmp_path).weights() == {}


def test_loader_blocks_low_sample_and_unvalidated_profiles(tmp_path: Path) -> None:
    profile = fit_stream_weights(
        [_hit("repo-a", {"dense": 1.0})],
        [_label(1.0)],
        repo_key="repo-a",
        since_sha="a",
        head_sha="b",
    )
    assert profile is not None
    write_stream_weight_profile(tmp_path, profile)

    profiles = StreamWeightProfiles(tmp_path)
    assert profiles.weights() == {}
    report = profiles.report()[0]
    assert report["status"] == "blocked"
    assert len(report["blocked_reasons"]) == 2


def test_validation_does_not_copy_low_sample_candidate_to_active(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    active = tmp_path / "active"
    profile = fit_stream_weights(
        [_hit("repo-a", {"dense": 1.0})],
        [_label(1.0)],
        repo_key="repo-a",
        since_sha="a",
        head_sha="b",
    )
    assert profile is not None
    write_stream_weight_profile(candidates, profile)

    updated = validate_stream_weight_profile(
        candidates,
        "repo-a",
        search_ndcg_delta=0.02,
        holdout_ndcg_delta=0.0,
        active_directory=active,
    )

    assert updated.validated is True
    assert not active.exists()


def test_validation_rejects_non_finite_evidence_without_rewriting_candidate(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates"
    profile = fit_stream_weights(
        [_hit("repo-a", {"dense": 1.0})],
        [_label(1.0)],
        repo_key="repo-a",
        since_sha="a",
        head_sha="b",
    )
    assert profile is not None
    path = write_stream_weight_profile(candidates, profile)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        validate_stream_weight_profile(
            candidates,
            "repo-a",
            search_ndcg_delta=float("nan"),
            holdout_ndcg_delta=0.0,
        )

    assert path.read_text(encoding="utf-8") == before


def test_validation_activates_only_non_regressing_profile(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    active = tmp_path / "active"
    hits = [_hit("repo-a", {"dense": 1.0}) for _ in range(MIN_ACTIVATION_EXAMPLES)]
    profile = fit_stream_weights(
        hits,
        [_label(1.0)] * len(hits),
        repo_key="repo-a",
        since_sha="a",
        head_sha="b",
    )
    assert profile is not None
    write_stream_weight_profile(candidates, profile)

    rejected = validate_stream_weight_profile(
        candidates,
        "repo-a",
        search_ndcg_delta=0.02,
        holdout_ndcg_delta=-0.001,
        active_directory=active,
    )
    assert rejected.validated is False
    assert StreamWeightProfiles(active).weights() == {}

    accepted = validate_stream_weight_profile(
        candidates,
        "repo-a",
        search_ndcg_delta=0.02,
        holdout_ndcg_delta=0.0,
        active_directory=active,
    )
    assert accepted.validated is True
    assert StreamWeightProfiles(active).weights()["repo-a"] == accepted.weights


def test_rejected_candidate_does_not_replace_active_profile(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    active = tmp_path / "active"
    base = StreamWeightProfile(
        repo="repo-a",
        weights={"dense": 1.2, "sparse": 0.8},
        observations={"dense": 50, "sparse": 50},
        mean_scores={"dense": 0.2, "sparse": -0.2},
        generated_at="2026-07-24T00:00:00+00:00",
        examples=50,
        source_counts={"outcome": 50},
        since_sha="a",
        head_sha="b",
        validated=True,
        validation={"search_ndcg_delta": 0.02, "holdout_ndcg_delta": 0.0},
    )
    write_stream_weight_profile(active, base)
    write_stream_weight_profile(candidates, replace(base, validated=False, validation={}))

    rejected = validate_stream_weight_profile(
        candidates,
        "repo-a",
        search_ndcg_delta=0.01,
        holdout_ndcg_delta=-0.01,
        active_directory=active,
    )

    assert rejected.validated is False
    assert StreamWeightProfiles(active).weights()["repo-a"] == base.weights
