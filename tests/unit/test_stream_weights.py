"""Tests for learned per-repository RRF stream weights (WIN-380)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from stropha.eval import EvalCase
from stropha.meta.outcome import Tier, TierLabel
from stropha.meta.stream_weights import (
    MIN_ACTIVATION_EXAMPLES,
    StreamWeightBenchmarkLockError,
    StreamWeightProfile,
    StreamWeightProfiles,
    benchmark_stream_weight_profile,
    fit_stream_weights,
    stream_weight_benchmark_lock,
    validate_stream_weight_profile,
    write_stream_weight_profile,
)


class _EvalHit:
    def __init__(self, path: str) -> None:
        self.rel_path = path
        self.symbol = None
        self.snippet = ""


class _EvalEngine:
    def __init__(self, paths: dict[str, list[str]]) -> None:
        self.paths = paths

    def search(self, query: str, *, top_k: int = 10, repo_id: int | None = None) -> list[_EvalHit]:
        return [_EvalHit(path) for path in self.paths[query]][:top_k]


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
    assert (
        fit_stream_weights(
            [_hit("repo-b", {"dense": 0.02}), _hit("repo-a", None)],
            [_label(1.0), _label(1.0)],
            repo_key="repo-a",
        )
        is None
    )


def test_fit_stream_weights_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_stream_weights([_hit("repo-a", {"dense": 1.0})], [], repo_key="repo-a")


def test_benchmark_compares_search_and_holdout_ndcg() -> None:
    cases = [
        EvalCase(id="s", query="search", expected_paths=("src/right.py",), split="search"),
        EvalCase(id="h", query="holdout", expected_paths=("src/holdout.py",), split="holdout"),
    ]
    baseline = _EvalEngine(
        {
            "search": ["src/wrong.py", "src/right.py"],
            "holdout": ["src/holdout.py"],
        }
    )
    candidate = _EvalEngine(
        {
            "search": ["src/right.py", "src/wrong.py"],
            "holdout": ["src/holdout.py"],
        }
    )

    result = benchmark_stream_weight_profile(
        baseline_engine=baseline,
        candidate_engine=candidate,
        cases=cases,
        repo_id=1,
    )

    assert result.search_ndcg_delta > 0
    assert result.holdout_ndcg_delta == 0
    assert result.search_cases == 1
    assert result.holdout_cases == 1


def test_benchmark_requires_both_splits() -> None:
    cases = [EvalCase(id="s", query="search", expected_paths=("src/a.py",), split="search")]

    with pytest.raises(ValueError, match="search and holdout"):
        benchmark_stream_weight_profile(
            baseline_engine=_EvalEngine({"search": ["src/a.py"]}),
            candidate_engine=_EvalEngine({"search": ["src/a.py"]}),
            cases=cases,
        )


def test_benchmark_lock_rejects_same_repository(tmp_path: Path) -> None:
    locks = tmp_path / "locks"

    with (
        stream_weight_benchmark_lock(locks, "repo-a"),
        pytest.raises(StreamWeightBenchmarkLockError, match="already running"),
        stream_weight_benchmark_lock(locks, "repo-a"),
    ):
        pass


def test_benchmark_lock_is_scoped_per_repository(tmp_path: Path) -> None:
    locks = tmp_path / "locks"

    with (
        stream_weight_benchmark_lock(locks, "repo-a"),
        stream_weight_benchmark_lock(locks, "repo-b"),
    ):
        pass


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


def test_rejected_candidate_deactivates_previous_profile(tmp_path: Path) -> None:
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
    assert StreamWeightProfiles(active).weights() == {}


def test_rejected_candidate_only_deactivates_matching_repository(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    active = tmp_path / "active"
    matching = StreamWeightProfile(
        repo="repo-a",
        weights={"dense": 1.2, "sparse": 0.8},
        observations={"dense": 50, "sparse": 50},
        mean_scores={"dense": 0.2, "sparse": -0.2},
        generated_at="2026-07-24T00:00:00+00:00",
        examples=MIN_ACTIVATION_EXAMPLES,
        source_counts={"outcome": MIN_ACTIVATION_EXAMPLES},
        since_sha="a",
        head_sha="b",
    )
    other = replace(matching, repo="repo-b")
    write_stream_weight_profile(candidates, matching)
    write_stream_weight_profile(
        active,
        replace(
            matching,
            validated=True,
            validation={
                "search_ndcg_delta": 0.02,
                "holdout_ndcg_delta": 0.0,
            },
        ),
    )
    write_stream_weight_profile(
        active,
        replace(
            other,
            validated=True,
            validation={
                "search_ndcg_delta": 0.02,
                "holdout_ndcg_delta": 0.0,
            },
        ),
    )

    validate_stream_weight_profile(
        candidates,
        matching.repo,
        search_ndcg_delta=0.01,
        holdout_ndcg_delta=-0.01,
        active_directory=active,
    )

    assert StreamWeightProfiles(active).weights() == {"repo-b": other.weights}
