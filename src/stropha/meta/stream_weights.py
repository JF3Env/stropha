"""Learn and serve per-repository RRF stream weights (WIN-380)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging import get_logger
from .outcome import TierLabel

log = get_logger(__name__)

PROFILE_VERSION = 2
MIN_ACTIVATION_EXAMPLES = 50
DEFAULT_PRIOR_OBSERVATIONS = 10.0
DEFAULT_LEARNING_RATE = 0.5


@dataclass(frozen=True)
class StreamWeightProfile:
    """Regularized stream weights learned for one repository."""

    repo: str
    weights: dict[str, float]
    observations: dict[str, int]
    mean_scores: dict[str, float]
    generated_at: str
    examples: int
    source_counts: dict[str, int]
    since_sha: str
    head_sha: str
    validated: bool = False
    validation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROFILE_VERSION,
            "repo": self.repo,
            "weights": self.weights,
            "observations": self.observations,
            "mean_scores": self.mean_scores,
            "generated_at": self.generated_at,
            "examples": self.examples,
            "source_counts": self.source_counts,
            "commit_range": {"since": self.since_sha, "head": self.head_sha},
            "validated": self.validated,
            "validation": self.validation,
        }


def _hit_repo_key(hit: dict[str, Any]) -> str | None:
    repo = hit.get("repo")
    if isinstance(repo, dict):
        key = repo.get("normalized_key")
        return str(key) if key else None
    key = getattr(repo, "normalized_key", None)
    return str(key) if key else None


def fit_stream_weights(
    hits: Sequence[dict[str, Any]],
    labels: Sequence[TierLabel],
    *,
    repo_key: str,
    prior_observations: float = DEFAULT_PRIOR_OBSERVATIONS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    source_counts: dict[str, int] | None = None,
    since_sha: str = "unknown",
    head_sha: str = "unknown",
) -> StreamWeightProfile | None:
    """Fit normalized stream multipliers from attributed outcome labels.

    Each stream that contributed a hit receives that hit's tier score. A
    zero-score prior keeps small samples close to equal weighting, while the
    exponential transform guarantees positive RRF multipliers. The final
    weights have arithmetic mean 1.0 so repositories remain comparable.
    """
    if len(hits) != len(labels):
        raise ValueError("hits and labels must have the same length")
    if prior_observations < 0:
        raise ValueError("prior_observations must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    score_sums: dict[str, float] = {}
    observations: dict[str, int] = {}
    examples = 0
    for hit, label in zip(hits, labels, strict=True):
        if _hit_repo_key(hit) != repo_key:
            continue
        streams = hit.get("streams")
        if not isinstance(streams, dict):
            continue
        attributed = False
        for stream, contribution in streams.items():
            if not isinstance(stream, str) or not stream:
                continue
            try:
                contributed = float(contribution) > 0
            except (TypeError, ValueError):
                continue
            if not contributed:
                continue
            score_sums[stream] = score_sums.get(stream, 0.0) + label.score
            observations[stream] = observations.get(stream, 0) + 1
            attributed = True
        if attributed:
            examples += 1

    if not observations:
        return None

    mean_scores = {
        stream: score_sums[stream] / (count + prior_observations)
        for stream, count in observations.items()
    }
    raw_weights = {
        stream: math.exp(learning_rate * score)
        for stream, score in mean_scores.items()
    }
    normalizer = sum(raw_weights.values()) / len(raw_weights)
    weights = {
        stream: weight / normalizer
        for stream, weight in raw_weights.items()
    }
    return StreamWeightProfile(
        repo=repo_key,
        weights=weights,
        observations=observations,
        mean_scores=mean_scores,
        generated_at=datetime.now(tz=UTC).isoformat(),
        examples=examples,
        source_counts=dict(source_counts or {"outcome": examples}),
        since_sha=since_sha,
        head_sha=head_sha,
    )


def profile_path(directory: Path, repo_key: str) -> Path:
    digest = hashlib.sha256(repo_key.encode("utf-8")).hexdigest()[:16]
    return directory / f"{digest}.json"


def write_stream_weight_profile(
    directory: Path,
    profile: StreamWeightProfile,
) -> Path:
    """Atomically persist one repository profile and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    target = profile_path(directory, profile.repo)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = json.dumps(profile.to_dict(), indent=2, sort_keys=True)
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _parse_profile(raw: Any, *, require_active: bool = True) -> StreamWeightProfile | None:
    if not isinstance(raw, dict) or raw.get("version") != PROFILE_VERSION:
        return None
    repo = raw.get("repo")
    weights = raw.get("weights")
    observations = raw.get("observations")
    mean_scores = raw.get("mean_scores")
    generated_at = raw.get("generated_at")
    examples = raw.get("examples")
    source_counts = raw.get("source_counts")
    commit_range = raw.get("commit_range")
    validated = raw.get("validated")
    validation = raw.get("validation")
    if not isinstance(repo, str) or not repo or not isinstance(weights, dict):
        return None
    if not isinstance(observations, dict) or not isinstance(mean_scores, dict):
        return None
    if not isinstance(generated_at, str):
        return None
    if not isinstance(source_counts, dict) or not isinstance(commit_range, dict):
        return None
    if not isinstance(validated, bool) or not isinstance(validation, dict):
        return None

    parsed_weights: dict[str, float] = {}
    parsed_observations: dict[str, int] = {}
    parsed_scores: dict[str, float] = {}
    parsed_sources: dict[str, int] = {}
    parsed_validation: dict[str, float] = {}
    try:
        if examples is None:
            return None
        example_count = int(examples)
        if example_count < 0:
            return None
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        for stream, value in weights.items():
            weight = float(value)
            if not isinstance(stream, str) or not stream or not math.isfinite(weight) or weight <= 0:
                return None
            parsed_weights[stream] = weight
        for stream, value in observations.items():
            count = int(value)
            if not isinstance(stream, str) or count < 0:
                return None
            parsed_observations[stream] = count
        for stream, value in mean_scores.items():
            score = float(value)
            if not isinstance(stream, str) or not math.isfinite(score):
                return None
            parsed_scores[stream] = score
        for source, value in source_counts.items():
            count = int(value)
            if not isinstance(source, str) or not source or count < 0:
                return None
            parsed_sources[source] = count
        for metric, value in validation.items():
            score = float(value)
            if not isinstance(metric, str) or not metric or not math.isfinite(score):
                return None
            parsed_validation[metric] = score
    except (TypeError, ValueError):
        return None
    since_sha = commit_range.get("since")
    head_sha = commit_range.get("head")
    if not parsed_weights or set(parsed_weights) != set(parsed_observations) or set(
        parsed_weights
    ) != set(parsed_scores):
        return None
    if not isinstance(since_sha, str) or not since_sha:
        return None
    if not isinstance(head_sha, str) or not head_sha:
        return None
    if sum(parsed_sources.values()) != example_count:
        return None
    if validated and (
        parsed_validation.get("search_ndcg_delta", 0.0) <= 0
        or parsed_validation.get("holdout_ndcg_delta", -1.0) < 0
    ):
        return None
    if require_active and (not validated or example_count < MIN_ACTIVATION_EXAMPLES):
        return None
    return StreamWeightProfile(
        repo=repo,
        weights=parsed_weights,
        observations=parsed_observations,
        mean_scores=parsed_scores,
        generated_at=generated_at,
        examples=example_count,
        source_counts=parsed_sources,
        since_sha=since_sha,
        head_sha=head_sha,
        validated=validated,
        validation=parsed_validation,
    )


def validate_stream_weight_profile(
    candidate_directory: Path,
    repo_key: str,
    *,
    search_ndcg_delta: float,
    holdout_ndcg_delta: float,
    active_directory: Path | None = None,
) -> StreamWeightProfile:
    """Record benchmark evidence; activate only on search gain and no holdout regression."""
    try:
        search_delta = float(search_ndcg_delta)
        holdout_delta = float(holdout_ndcg_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark deltas must be finite numbers") from exc
    if not math.isfinite(search_delta) or not math.isfinite(holdout_delta):
        raise ValueError("benchmark deltas must be finite numbers")
    path = profile_path(candidate_directory, repo_key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load stream-weight profile for {repo_key!r}") from exc
    profile = _parse_profile(raw, require_active=False)
    if profile is None or profile.repo != repo_key:
        raise ValueError(f"invalid stream-weight profile for {repo_key!r}")
    validation = {
        "search_ndcg_delta": search_delta,
        "holdout_ndcg_delta": holdout_delta,
    }
    eligible = search_delta > 0 and holdout_delta >= 0
    updated = replace(profile, validated=eligible, validation=validation)
    write_stream_weight_profile(candidate_directory, updated)
    if eligible and profile.examples >= MIN_ACTIVATION_EXAMPLES:
        write_stream_weight_profile(active_directory or candidate_directory, updated)
    return updated


class StreamWeightProfiles:
    """Cached profile directory that reloads after atomic profile writes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._directory_mtime_ns: int | None = None
        self._profiles: dict[str, StreamWeightProfile] = {}

    def weights(self) -> dict[str, dict[str, float]]:
        self._refresh()
        return {
            repo: dict(profile.weights)
            for repo, profile in self._profiles.items()
        }

    def fingerprint(self) -> str:
        profiles = self.weights()
        if not profiles:
            return "none"
        canonical = json.dumps(profiles, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    def report(self, *, active: bool = False) -> list[dict[str, Any]]:
        """Describe valid candidates and why each is active or blocked."""
        rows: list[dict[str, Any]] = []
        if not self.directory.is_dir():
            return rows
        for path in sorted(self.directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rows.append({"path": str(path), "status": "invalid"})
                continue
            profile = _parse_profile(raw, require_active=False)
            if profile is None:
                rows.append({"path": str(path), "status": "invalid"})
                continue
            reasons = []
            if profile.examples < MIN_ACTIVATION_EXAMPLES:
                reasons.append(
                    f"requires {MIN_ACTIVATION_EXAMPLES} examples; has {profile.examples}"
                )
            if not profile.validated:
                reasons.append("search/holdout validation has not passed")
            rows.append({
                **profile.to_dict(),
                "path": str(path),
                "status": ("active" if active else "eligible") if not reasons else "blocked",
                "blocked_reasons": reasons,
            })
        return rows

    def _refresh(self) -> None:
        try:
            mtime_ns = self.directory.stat().st_mtime_ns
        except OSError:
            self._directory_mtime_ns = None
            self._profiles = {}
            return
        if mtime_ns == self._directory_mtime_ns:
            return

        loaded: dict[str, StreamWeightProfile] = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("stream_weights.invalid_profile", path=str(path), error=str(exc))
                continue
            profile = _parse_profile(raw)
            if profile is None:
                log.warning("stream_weights.invalid_profile", path=str(path))
                continue
            loaded[profile.repo] = profile
        self._profiles = loaded
        self._directory_mtime_ns = mtime_ns


__all__ = [
    "MIN_ACTIVATION_EXAMPLES",
    "StreamWeightProfile",
    "StreamWeightProfiles",
    "fit_stream_weights",
    "profile_path",
    "validate_stream_weight_profile",
    "write_stream_weight_profile",
]
