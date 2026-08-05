"""Tests for the inference-backed listwise reranker."""

from __future__ import annotations

from unittest.mock import patch

from stropha.adapters.retrieval.reranker.llm import LlmReranker
from stropha.models import SearchHit
from stropha.pipeline.registry import lookup_adapter


def _hit(rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0 / rank,
        rel_path=f"src/{rank}.py",
        language="python",
        kind="function",
        symbol=f"symbol_{rank}",
        start_line=1,
        end_line=2,
        snippet=f"def symbol_{rank}(): pass",
        chunk_id=f"chunk-{rank}",
    )


def test_registered() -> None:
    assert lookup_adapter("reranker", "llm") is LlmReranker


def test_reranks_from_json_order_and_updates_rank() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    with patch("stropha.inference.generate", return_value="[3, 1, 2]"):
        result = LlmReranker().rerank("query", hits, top_k=3)
    assert [hit.chunk_id for hit in result] == ["chunk-3", "chunk-1", "chunk-2"]
    assert [hit.rank for hit in result] == [1, 2, 3]


def test_partial_order_appends_missing_candidates() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    with patch("stropha.inference.generate", return_value="```json\n[2]\n```"):
        result = LlmReranker().rerank("query", hits, top_k=3)
    assert [hit.chunk_id for hit in result] == ["chunk-2", "chunk-1", "chunk-3"]


def test_invalid_response_preserves_original_order() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    with patch("stropha.inference.generate", return_value="not json"):
        result = LlmReranker().rerank("query", hits, top_k=2)
    assert result == hits[:2]


def test_parser_drops_invalid_duplicate_and_boolean_ids() -> None:
    assert LlmReranker._parse_order("[2, 2, true, 9, 1]", 3) == [2, 1]
