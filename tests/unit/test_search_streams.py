"""Unit tests for stream attribution in SearchEngine._search_single (WIN-160).

The refactor at retrieval/search.py:108-128 must preserve which retrieval
stream each fused hit came from. These tests stub Storage + Embedder so
the unit under test is only the fusion + labelling logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from stropha.models import SearchHit
from stropha.retrieval.search import SearchEngine


def _hit(chunk_id: str, rank: int, score: float = 1.0) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=score,
        rel_path=f"{chunk_id}.py",
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=5,
        snippet=f"def {chunk_id}(): pass",
        chunk_id=chunk_id,
    )


def _engine(
    dense: list[SearchHit],
    sparse: list[SearchHit],
    symbol: list[SearchHit],
) -> SearchEngine:
    """Build a SearchEngine whose Storage returns canned per-stream lists."""
    storage = MagicMock()
    storage.search_dense.return_value = dense
    storage.search_bm25.return_value = sparse
    storage.search_symbol_tokens.return_value = symbol
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8
    # Bypass the recursive-merge step by stubbing what `merge_hits` calls into.
    # `recursive.merge_hits` is a pass-through when recursive retrieval
    # is unset, which is the case in unit tests.
    return SearchEngine(storage=storage, embedder=embedder)


class TestStreamAttribution:
    def test_all_three_streams_label_each_hit(self) -> None:
        engine = _engine(
            dense=[_hit("a", 1), _hit("b", 2)],
            sparse=[_hit("a", 2), _hit("c", 1)],
            symbol=[_hit("a", 3)],
        )
        out = engine.search("anything")
        by_id = {h.chunk_id: h.streams for h in out}

        # `a` appeared in all three streams — keys must be all three labels.
        assert set(by_id["a"]) == {"dense", "sparse", "symbol"}
        assert set(by_id["b"]) == {"dense"}
        assert set(by_id["c"]) == {"sparse"}

    def test_single_stream_short_circuit_still_labels(self) -> None:
        """When only one stream returns results, the short-circuit path must
        still synthesise a ``streams`` field so downstream consumers do not
        need to handle a None case."""
        engine = _engine(
            dense=[_hit("a", 1), _hit("b", 2)],
            sparse=[],
            symbol=[],
        )
        out = engine.search("anything")
        for h in out:
            assert h.streams is not None
            assert set(h.streams) == {"dense"}

    def test_empty_streams_returns_empty(self) -> None:
        engine = _engine(dense=[], sparse=[], symbol=[])
        assert engine.search("anything") == []

    def test_empty_query_returns_empty(self) -> None:
        engine = _engine(
            dense=[_hit("a", 1)],
            sparse=[_hit("b", 1)],
            symbol=[],
        )
        assert engine.search("   ") == []

    def test_stream_contribution_is_a_float(self) -> None:
        """``streams[label]`` is a numeric contribution, not just a boolean."""
        engine = _engine(
            dense=[_hit("a", 1)],
            sparse=[_hit("a", 1)],
            symbol=[],
        )
        out = engine.search("anything")
        s = out[0].streams
        assert s is not None
        assert isinstance(s["dense"], float)
        assert isinstance(s["sparse"], float)
        assert s["dense"] > 0
        assert s["sparse"] > 0
