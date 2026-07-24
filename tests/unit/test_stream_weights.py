"""Tests for learned per-repository RRF stream weights (WIN-380)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.meta.outcome import Tier, TierLabel
from stropha.meta.stream_weights import (
    StreamWeightProfile,
    StreamWeightProfiles,
    fit_stream_weights,
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
        observations={"dense": 12, "sparse": 9},
        mean_scores={"dense": 0.2, "sparse": -0.1},
        generated_at="2026-07-24T00:00:00+00:00",
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
