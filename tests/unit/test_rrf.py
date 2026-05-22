"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from stropha.models import SearchHit
from stropha.retrieval.rrf import rrf_fuse, rrf_fuse_named


def _hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0,
        rel_path="x.py",
        language="python",
        kind="method",
        start_line=1,
        end_line=2,
        snippet="…",
        chunk_id=chunk_id,
    )


def test_rrf_prefers_chunks_in_both_lists() -> None:
    dense = [_hit("a", 1), _hit("b", 2), _hit("c", 3)]
    sparse = [_hit("b", 1), _hit("a", 2), _hit("d", 3)]
    fused = rrf_fuse(dense, sparse, k=60, top_k=4)
    # `a` is rank 1 in dense + rank 2 in sparse  → 1/61 + 1/62
    # `b` is rank 2 in dense + rank 1 in sparse  → 1/62 + 1/61  (same as a)
    # `c` only in dense, `d` only in sparse.
    ids = [h.chunk_id for h in fused]
    assert ids[0] in {"a", "b"}
    assert ids[1] in {"a", "b"}
    assert set(ids[:2]) == {"a", "b"}
    assert "c" in ids and "d" in ids


def test_rrf_handles_empty_list() -> None:
    dense = [_hit("a", 1), _hit("b", 2)]
    fused = rrf_fuse(dense, [], k=60, top_k=5)
    assert [h.chunk_id for h in fused] == ["a", "b"]


def test_rrf_assigns_consecutive_ranks() -> None:
    dense = [_hit(c, i + 1) for i, c in enumerate(["a", "b", "c", "d"])]
    sparse = [_hit(c, i + 1) for i, c in enumerate(["e", "f", "g"])]
    fused = rrf_fuse(dense, sparse, k=60, top_k=10)
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))


def test_rrf_score_monotonic() -> None:
    dense = [_hit("a", 1), _hit("b", 5)]
    fused = rrf_fuse(dense, [], k=60, top_k=2)
    assert fused[0].score > fused[1].score


# ---------------------------------------------------------------------------
# WIN-160: stream-name attribution
# ---------------------------------------------------------------------------


class TestStreamsAttribution:
    """``stream_names`` populates SearchHit.streams with the per-stream RRF
    contribution. Consumed by the trace recorder."""

    def test_streams_field_default_none(self) -> None:
        """Without stream_names, hits carry streams=None (back-compat)."""
        dense = [_hit("a", 1), _hit("b", 2)]
        sparse = [_hit("a", 2), _hit("c", 1)]
        fused = rrf_fuse(dense, sparse)
        assert all(h.streams is None for h in fused)

    def test_streams_populated_when_labelled(self) -> None:
        dense = [_hit("a", 1), _hit("b", 2)]
        sparse = [_hit("a", 2), _hit("c", 1)]
        fused = rrf_fuse(dense, sparse, stream_names=["dense", "sparse"])
        by_id = {h.chunk_id: h.streams for h in fused}

        # a appeared in BOTH lists — its streams dict carries both keys.
        assert set(by_id["a"]) == {"dense", "sparse"}
        # b only in dense; c only in sparse.
        assert set(by_id["b"]) == {"dense"}
        assert set(by_id["c"]) == {"sparse"}

    def test_stream_contributions_match_rrf_formula(self) -> None:
        # RRF uses enumerate(...) for the rank, NOT SearchHit.rank.
        # Position-1 of dense and position-2 of sparse for chunk `a`.
        dense = [_hit("a", 99)]  # only item → enumerate rank = 1
        sparse = [_hit("z", 99), _hit("a", 99)]  # a is 2nd → enumerate rank = 2
        fused = rrf_fuse(dense, sparse, k=60, stream_names=["dense", "sparse"])
        a = next(h for h in fused if h.chunk_id == "a")
        streams = a.streams
        assert streams is not None
        # RRF formula: 1 / (k + rank). Dense: 1/61, sparse: 1/62.
        assert streams["dense"] == pytest.approx(1 / 61)
        assert streams["sparse"] == pytest.approx(1 / 62)

    def test_stream_names_length_must_match(self) -> None:
        dense = [_hit("a", 1)]
        sparse = [_hit("b", 1)]
        with pytest.raises(ValueError, match="stream_names length"):
            rrf_fuse(dense, sparse, stream_names=["only-one-label"])

    def test_rrf_fuse_named_propagates_labels(self) -> None:
        """The ergonomic named-dict variant must thread labels through."""
        fused = rrf_fuse_named(
            {
                "dense": [_hit("a", 1), _hit("b", 2)],
                "sparse": [_hit("a", 2), _hit("c", 1)],
            }
        )
        a = next(h for h in fused if h.chunk_id == "a")
        assert a.streams is not None
        assert set(a.streams) == {"dense", "sparse"}

    def test_rrf_fuse_named_honours_weights(self) -> None:
        """Weights scale a stream's contribution; rank order respects them."""
        fused = rrf_fuse_named(
            {
                "dense": [_hit("a", 1)],
                "sparse": [_hit("b", 1)],
            },
            weights={"dense": 10.0, "sparse": 1.0},
        )
        # `a` should outrank `b` because dense weight is 10x.
        assert fused[0].chunk_id == "a"
        # And its dense contribution is 10/61 vs sparse 1/61.
        assert fused[0].streams is not None
        assert fused[0].streams["dense"] == pytest.approx(10 / 61)

    def test_weights_length_mismatch_raises(self) -> None:
        dense = [_hit("a", 1)]
        sparse = [_hit("b", 1)]
        with pytest.raises(ValueError, match="weights length"):
            rrf_fuse(dense, sparse, weights=[1.0])

    def test_unknown_stream_in_weights_defaults_to_one(self) -> None:
        """``rrf_fuse_named`` fills weight=1.0 for streams the caller did
        not mention in ``weights``."""
        fused = rrf_fuse_named(
            {"dense": [_hit("a", 1)], "sparse": [_hit("b", 1)]},
            weights={"dense": 5.0},  # sparse not listed
        )
        # Sparse weight defaulted to 1.0.
        b = next(h for h in fused if h.chunk_id == "b")
        assert b.streams is not None
        assert b.streams["sparse"] == pytest.approx(1 / 61)
