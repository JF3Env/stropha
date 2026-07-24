"""Reciprocal Rank Fusion (Cormack et al., 2009 — spec §6.1).

Formula: score(d) = Σ_i w_i / (k + rank_i(d))   where k=60 by default.

Extended with optional per-stream weights for adaptive query weighting.
When weights are provided, each stream's contribution is multiplied by its weight.

Why RRF beats weighted-sum here:
- BM25 and cosine live in different scales; normalizing across queries is
  brittle. RRF only consumes ranks.
- Robust to outliers in either list.
- No hyperparameters worth tuning (k=60 is the published default).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..models import SearchHit

DEFAULT_K = 60


def rrf_fuse(
    *ranked_lists: Iterable[SearchHit],
    k: int = DEFAULT_K,
    top_k: int = 10,
    weights: list[float] | None = None,
    stream_names: list[str] | None = None,
    repo_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> list[SearchHit]:
    """Merge any number of ranked lists into a single ranked list.

    Identity is by `chunk_id`. The returned `SearchHit` keeps the metadata of
    the first occurrence of each chunk (typically the dense hit, since dense
    is usually passed first), with its `score` replaced by the RRF score.

    Args:
        *ranked_lists: One or more ranked lists of SearchHit objects.
        k: RRF smoothing constant (default 60).
        top_k: Number of results to return.
        weights: Optional per-stream weights. If provided, must have same
            length as ranked_lists. Each stream's contribution is multiplied
            by its weight. Default weights are 1.0.
        stream_names: Optional labels (one per ranked list, e.g.
            ``["dense", "sparse", "symbol"]``). When provided, each returned
            ``SearchHit`` carries a ``streams`` dict mapping the labels of
            the streams it appeared in to their per-stream RRF contribution.
            Consumed by the Meta-Harness trace recorder (WIN-160).
        repo_weights: Optional per-repository stream multipliers learned from
            feedback. They multiply the query-level ``weights`` for each hit.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match number of ranked lists ({len(ranked_lists)})"
        )
    if stream_names is not None and len(stream_names) != len(ranked_lists):
        raise ValueError(
            f"stream_names length ({len(stream_names)}) must match number of ranked lists ({len(ranked_lists)})"
        )
    if repo_weights and stream_names is None:
        raise ValueError("repo_weights requires stream_names")

    scores: dict[str, float] = {}
    seen: dict[str, SearchHit] = {}
    # Per-chunk breakdown — populated only when stream_names was provided.
    breakdown: dict[str, dict[str, float]] = {}
    for idx, (weight, lst) in enumerate(zip(weights, ranked_lists, strict=False)):
        label = stream_names[idx] if stream_names is not None else None
        for rank, hit in enumerate(lst, start=1):
            effective_weight = weight
            if repo_weights and label is not None and hit.repo is not None:
                effective_weight *= repo_weights.get(
                    hit.repo.normalized_key, {}
                ).get(label, 1.0)
            contribution = effective_weight / (k + rank)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + contribution
            if hit.chunk_id not in seen:
                seen[hit.chunk_id] = hit
            if label is not None:
                breakdown.setdefault(hit.chunk_id, {})[label] = contribution

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[SearchHit] = []
    for new_rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1):
        base = seen[chunk_id]
        update: dict[str, object] = {"rank": new_rank, "score": score}
        if stream_names is not None:
            update["streams"] = breakdown.get(chunk_id, {})
        out.append(base.model_copy(update=update))
    return out


def rrf_fuse_named(
    ranked_lists: Mapping[str, Iterable[SearchHit]],
    *,
    k: int = DEFAULT_K,
    top_k: int = 10,
    weights: dict[str, float] | None = None,
    repo_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> list[SearchHit]:
    """Merge named ranked lists with optional per-stream weights.

    Stream names are preserved on each returned hit's ``streams`` field,
    so callers can see which stream(s) contributed and by how much.

    Args:
        ranked_lists: Dict mapping stream names to ranked lists.
        k: RRF smoothing constant (default 60).
        top_k: Number of results to return.
        weights: Optional dict mapping stream names to weights.
            Missing streams default to weight 1.0.
        repo_weights: Optional per-repository stream multipliers. Missing
            repositories and streams default to 1.0.
    """
    if weights is None:
        weights = {}

    # Convert to positional format
    names = list(ranked_lists.keys())
    lists = [ranked_lists[name] for name in names]
    weight_list = [weights.get(name, 1.0) for name in names]

    return rrf_fuse(
        *lists,
        k=k,
        top_k=top_k,
        weights=weight_list,
        stream_names=names,
        repo_weights=repo_weights,
    )
