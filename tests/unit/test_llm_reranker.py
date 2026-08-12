"""Tests for the inference-backed listwise reranker."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from stropha.adapters.retrieval.reranker.llm import (
    LlmReranker,
    LlmRerankerConfig,
)
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


def _backend() -> SimpleNamespace:
    return SimpleNamespace(identity="test:model=one", name="test")


def test_registered() -> None:
    assert lookup_adapter("reranker", "llm") is LlmReranker


def test_reranks_from_structured_order_and_requests_ranking_controls() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", return_value='{"order": [3, 1, 2]}') as generate,
    ):
        result = LlmReranker().rerank("query", hits, top_k=3)

    assert [hit.chunk_id for hit in result] == ["chunk-3", "chunk-1", "chunk-2"]
    assert [hit.rank for hit in result] == [1, 2, 3]
    assert generate.call_args.kwargs["json_schema"]["required"] == ["order"]
    assert generate.call_args.kwargs["think"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```json\n[2]\n```", [2]),
        ('prefix\n{"order": [2]}\nsuffix', [2]),
    ],
)
def test_markdown_wrapped_output_and_partial_order_remain_supported(
    raw: str,
    expected: list[int],
) -> None:
    assert LlmReranker._parse_order(raw, 3) == expected

    hits = [_hit(1), _hit(2), _hit(3)]
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", return_value=raw),
    ):
        result = LlmReranker().rerank("query", hits, top_k=3)
    assert [hit.chunk_id for hit in result] == ["chunk-2", "chunk-1", "chunk-3"]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (None, "empty_response"),
        ("not json", "invalid_json"),
        ('{"other": [1]}', "invalid_json"),
        ('{"order": []}', "invalid_ids"),
        ('{"order": [2, 2]}', "invalid_ids"),
        ('{"order": [true, 1]}', "invalid_ids"),
        ('{"order": [-1, 1]}', "invalid_ids"),
        ('{"order": [4, 1]}', "invalid_ids"),
    ],
)
def test_invalid_response_classes(raw: str | None, reason: str) -> None:
    assert LlmReranker._parse_response(raw, 3) == ([], reason)


def test_parser_skips_invalid_fragment_before_valid_order() -> None:
    raw = 'example: [] answer: {"order": [2, 1]}'
    assert LlmReranker._parse_response(raw, 3) == ([2, 1], None)


def test_parser_does_not_treat_nested_array_as_legacy_order() -> None:
    raw = '{"scores": [2, 1]}'
    assert LlmReranker._parse_response(raw, 3) == ([], "invalid_json")


def test_parser_does_not_recover_nested_array_from_malformed_object() -> None:
    raw = '{scores: [2, 1]}'
    assert LlmReranker._parse_response(raw, 3) == ([], "invalid_json")


def test_parser_prefers_final_explicit_order_object() -> None:
    raw = 'example: {"order": [1, 2]} answer: {"order": [2, 1]}'
    assert LlmReranker._parse_response(raw, 3) == ([2, 1], None)


def test_parser_prefers_final_legacy_array() -> None:
    raw = "example: [1, 2] answer: [2, 1]"
    assert LlmReranker._parse_response(raw, 3) == ([2, 1], None)


def test_parser_allows_legacy_array_after_non_json_braces() -> None:
    raw = "note {format} answer: [2, 1]"
    assert LlmReranker._parse_response(raw, 3) == ([2, 1], None)


def test_non_ranking_object_does_not_suppress_legacy_array() -> None:
    raw = 'metadata: {"scores": [1, 2]} answer: [2, 1]'
    assert LlmReranker._parse_response(raw, 3) == ([2, 1], None)


def test_parser_does_not_escape_malformed_object_string() -> None:
    raw = '{note: "}", scores: [2, 1]}'
    assert LlmReranker._parse_response(raw, 3) == ([], "invalid_json")


def test_parser_does_not_escape_single_quoted_malformed_object_string() -> None:
    raw = "{note: '}', scores: [2, 1]}"
    assert LlmReranker._parse_response(raw, 3) == ([], "invalid_json")


def test_one_repair_attempt_succeeds() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    reranker = LlmReranker()
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch(
            "stropha.inference.generate",
            side_effect=["order: 3, 1", '{"order": [3, 1]}'],
        ) as generate,
    ):
        result = reranker.rerank("query", hits, top_k=3)

    assert generate.call_count == 2
    assert [hit.chunk_id for hit in result] == ["chunk-3", "chunk-1", "chunk-2"]
    diagnostics = reranker.diagnostics()
    assert diagnostics["invalid_json"] == 1
    assert diagnostics["repair_attempts"] == 1
    assert diagnostics["repair_successes"] == 1
    assert diagnostics["fallbacks"] == 0


def test_failed_repair_falls_back_without_retrying() -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    reranker = LlmReranker(LlmRerankerConfig(fallback="original"))
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", return_value="not json") as generate,
    ):
        result = reranker.rerank("query", hits, top_k=2)

    assert generate.call_count == 2
    assert result == hits[:2]
    diagnostics = reranker.diagnostics()
    assert diagnostics["invalid_json"] == 2
    assert diagnostics["repair_attempts"] == 1
    assert diagnostics["repair_successes"] == 0
    assert diagnostics["fallbacks"] == 1


def test_default_fallback_uses_code_aware_reranker() -> None:
    hits = [_hit(1), _hit(2)]
    expected = [hits[1]]
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", side_effect=RuntimeError("offline")),
        patch(
            "stropha.adapters.retrieval.reranker.code_aware.CodeAwareReranker.rerank",
            return_value=expected,
        ) as fallback,
    ):
        result = LlmReranker().rerank("query", hits, top_k=1)

    assert result == expected
    fallback.assert_called_once_with("query", hits, top_k=1)


@pytest.mark.parametrize(
    ("error", "reason"),
    [(RuntimeError("offline"), "backend_error"), (TimeoutError(), "timeout")],
)
def test_backend_errors_use_configured_deterministic_fallback(
    error: Exception,
    reason: str,
) -> None:
    hits = [_hit(1), _hit(2), _hit(3)]
    reranker = LlmReranker(LlmRerankerConfig(fallback="original"))
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", side_effect=error) as generate,
    ):
        result = reranker.rerank("query", hits, top_k=2)

    assert generate.call_count == 1
    assert result == hits[:2]
    diagnostics = reranker.diagnostics()
    assert diagnostics[reason] == 1
    assert diagnostics["repair_attempts"] == 0
    assert diagnostics["fallbacks"] == 1


def test_diagnostics_reset_clears_all_counters_and_latencies() -> None:
    reranker = LlmReranker()
    with (
        patch("stropha.inference.get_backend", return_value=_backend()),
        patch("stropha.inference.generate", return_value='{"order": [2, 1]}'),
    ):
        reranker.rerank("query", [_hit(1), _hit(2)])
    assert reranker.diagnostics()["successes"] == 1

    reranker.reset_diagnostics()

    assert reranker.diagnostics() == {
        "attempts": 0,
        "successes": 0,
        "fallbacks": 0,
        "empty_response": 0,
        "invalid_json": 0,
        "invalid_ids": 0,
        "backend_error": 0,
        "timeout": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "primary_latency_ms": 0,
        "primary_latency_p95_ms": 0.0,
        "repair_latency_ms": 0,
        "repair_latency_p95_ms": 0.0,
        "latency_ms": 0,
    }


def test_adapter_id_changes_with_reranker_policy() -> None:
    with patch("stropha.inference.get_backend", return_value=_backend()):
        base = LlmReranker().adapter_id
        assert LlmReranker(LlmRerankerConfig(output_mode="text")).adapter_id != base
        assert LlmReranker(LlmRerankerConfig(repair_enabled=False)).adapter_id != base
        assert LlmReranker(LlmRerankerConfig(fallback="original")).adapter_id != base
