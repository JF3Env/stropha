"""Unit tests for ``stropha.adapters.enricher.contextual.ContextualEnricher``.

The contextual enricher generates semantic descriptions for chunks to improve
embedding quality. Like the ollama enricher, it MUST never block indexing —
every inference failure path falls back to raw content.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from stropha.adapters.enricher.contextual import (
    ContextualEnricher,
    ContextualEnricherConfig,
)
from stropha.models import Chunk
from stropha.pipeline.base import StageContext


def _make_chunk(
    content: str = "def submit_answer(self, exercise_id, answer): pass",
    symbol: str = "submit_answer",
    rel_path: str = "src/services/study.py",
    language: str = "python",
) -> Chunk:
    return Chunk(
        chunk_id="sha-1",
        rel_path=rel_path,
        language=language,
        kind="function",
        symbol=symbol,
        parent_chunk_id=None,
        start_line=1,
        end_line=5,
        content=content,
        content_hash="hash-1",
    )


# --------------------------------------------------------------------------- adapter_id


def test_adapter_id_includes_temperature_and_prompt_hash() -> None:
    e = ContextualEnricher(
        ContextualEnricherConfig(temperature=0.3)
    )
    # adapter_id format: contextual:t=<temp>:p=<prompt_hash>
    assert e.adapter_id.startswith("contextual:t=0.3:p=")


def test_adapter_id_changes_with_prompt_edit() -> None:
    e1 = ContextualEnricher(
        ContextualEnricherConfig(prompt_template="A {content} {language} {rel_path} {symbol}")
    )
    e2 = ContextualEnricher(
        ContextualEnricherConfig(prompt_template="B {content} {language} {rel_path} {symbol}")
    )
    assert e1.adapter_id != e2.adapter_id


def test_adapter_name_is_contextual() -> None:
    assert ContextualEnricher().adapter_name == "contextual"


# --------------------------------------------------------------------------- registry


def test_registered_in_registry() -> None:
    """`stropha adapters list --stage enricher` must surface it."""
    from stropha.pipeline.registry import lookup_adapter

    cls = lookup_adapter("enricher", "contextual")
    assert cls is ContextualEnricher


# --------------------------------------------------------------------------- enrich (success)


def test_enrich_prepends_context_on_success() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "This method handles answer submission in the study service, "
            "updating user progress and triggering FSRS calculations."
        )
        out = enricher.enrich(chunk, StageContext())

    assert out.startswith("[Context: This method handles answer submission")
    assert chunk.content in out


def test_enrich_includes_context_prefix_format() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "This is a test description."
        out = enricher.enrich(chunk, StageContext())

    assert "[Context:" in out
    assert "]\n\n" in out  # Context ends with ] and double newline


# --------------------------------------------------------------------------- enrich (failures)


def test_enrich_returns_raw_on_failure() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None  # Simulate failure
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


def test_enrich_returns_raw_on_empty_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""  # Empty response
        out = enricher.enrich(chunk, StageContext())

    assert out == chunk.content


# --------------------------------------------------------------------------- failure status (Fix 2: no cache poisoning)


def test_enrich_with_status_reports_success_on_happy_path() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "This is a description."
        text, success = enricher.enrich_with_status(chunk, StageContext())

    assert success is True
    assert text.startswith("[Context: ")
    assert chunk.content in text


def test_enrich_with_status_reports_failure_when_llm_returns_none() -> None:
    """LLM returned None → enricher MUST signal failure so pipeline skips caching."""
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = None
        text, success = enricher.enrich_with_status(chunk, StageContext())

    assert success is False
    assert text == chunk.content  # raw fallback


def test_enrich_with_status_reports_failure_on_empty_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = ""
        text, success = enricher.enrich_with_status(chunk, StageContext())

    assert success is False
    assert text == chunk.content


# --------------------------------------------------------------------------- response cleaning


def test_strips_code_fences_from_response() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "```\nThis is the description.\n```"
        out = enricher.enrich(chunk, StageContext())

    assert "```" not in out
    assert "[Context: This is the description.]" in out


def test_takes_first_paragraph_if_multiple() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "First paragraph.\n\nSecond paragraph with more detail."
        out = enricher.enrich(chunk, StageContext())

    assert "First paragraph." in out
    assert "Second paragraph" not in out


def test_truncates_long_descriptions() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_description_chars=100)  # Minimum allowed
    )
    chunk = _make_chunk()
    with patch("stropha.inference.generate") as mock_gen:
        long_response = "This is a very long description that exceeds the limit and keeps going on and on. " * 10
        mock_gen.return_value = long_response
        out = enricher.enrich(chunk, StageContext())

    # Check it's truncated with ellipsis
    context_part = out.split("]\n\n")[0]
    # "[Context: " is 10 chars, so description should be ~90 chars + "..."
    assert len(context_part) < 120
    assert "..." in context_part


# --------------------------------------------------------------------------- health


def test_health_ready_when_backend_available() -> None:
    enricher = ContextualEnricher()
    mock_backend = MagicMock()
    mock_backend.name = "mlx"
    mock_backend.health_check.return_value = (True, "Model loaded")
    with patch("stropha.inference.get_backend", return_value=mock_backend):
        health = enricher.health()

    assert health.status == "ready"
    assert "contextual enricher" in health.message
    assert "mlx" in health.message


def test_health_warning_when_backend_unhealthy() -> None:
    enricher = ContextualEnricher()
    mock_backend = MagicMock()
    mock_backend.name = "ollama"
    mock_backend.health_check.return_value = (False, "Ollama unreachable")
    with patch("stropha.inference.get_backend", return_value=mock_backend):
        health = enricher.health()

    assert health.status == "warning"
    assert "unreachable" in health.message.lower()


# --------------------------------------------------------------------------- config


def test_prompt_includes_chunk_metadata() -> None:
    enricher = ContextualEnricher()
    chunk = _make_chunk(
        content="class UserService: ...",
        symbol="UserService",
        rel_path="src/services/user.py",
        language="python",
    )

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "Test description."
        enricher.enrich(chunk, StageContext())

        # Check the prompt was built with metadata
        call_args = mock_gen.call_args
        prompt = call_args[0][0]  # First positional argument

        assert "UserService" in prompt
        assert "src/services/user.py" in prompt
        assert "python" in prompt


def test_content_truncation() -> None:
    enricher = ContextualEnricher(
        ContextualEnricherConfig(max_content_chars=500)  # Minimum allowed
    )
    long_content = "x" * 2000  # Much longer than 500
    chunk = _make_chunk(content=long_content)

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "Description."
        enricher.enrich(chunk, StageContext())

        # Check content was truncated in prompt
        call_args = mock_gen.call_args
        prompt = call_args[0][0]  # First positional argument

        # Content should be truncated to 500 chars (plus prompt template overhead)
        assert "x" * 500 in prompt
        assert "x" * 600 not in prompt  # Truncated


# --------------------------------------------------------------------------- enrich_batch (Fix 3)


def _make_indexed_chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c-{i}",
        rel_path=f"f{i}.py",
        language="python",
        kind="function",
        symbol=f"sym{i}",
        parent_chunk_id=None,
        start_line=1,
        end_line=2,
        content=f"def f{i}(): pass",
        content_hash=f"h-{i}",
    )


def test_enrich_batch_packs_chunks_into_one_llm_call() -> None:
    """3 chunks → exactly 1 generate() call (not 3)."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(3)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            '[{"id":1,"description":"This is one."},'
            '{"id":2,"description":"This is two."},'
            '{"id":3,"description":"This is three."}]'
        )
        results = enricher.enrich_batch(chunks, contexts)

    assert mock_gen.call_count == 1
    assert len(results) == 3
    assert all(success for _, success in results)
    assert "This is one." in results[0][0]
    assert "This is two." in results[1][0]
    assert "This is three." in results[2][0]


def test_enrich_batch_prompt_lists_all_chunks_numbered() -> None:
    """The batched prompt must contain every chunk's id + metadata."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(3)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = '[{"id":1,"description":"This..."},{"id":2,"description":"This..."},{"id":3,"description":"This..."}]'
        enricher.enrich_batch(chunks, contexts)

    prompt = mock_gen.call_args[0][0]
    assert "CHUNK 1" in prompt
    assert "CHUNK 2" in prompt
    assert "CHUNK 3" in prompt
    assert "STRICT JSON" in prompt
    for chunk in chunks:
        assert chunk.rel_path in prompt
        assert chunk.symbol in prompt


def test_enrich_batch_whole_call_fails_falls_back_per_chunk() -> None:
    """generate() returns None (LLM totally down) → fall back to per-chunk
    enrichment. Each chunk gets a fresh request budget, and even when those
    also fail the chunks come back as (raw_content, False) so the pipeline
    skips caching them."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(4)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        # Batched call returns None (entire batch fails) → per-chunk fallback
        # is then invoked. Per-chunk calls also return None to model a true
        # LLM outage; results must surface failure for every chunk.
        mock_gen.return_value = None
        results = enricher.enrich_batch(chunks, contexts)

    # 1 batched call + 4 per-chunk fallbacks = 5 total calls.
    assert mock_gen.call_count == 5
    assert len(results) == 4
    assert all(not success for _, success in results)
    for chunk, (text, _) in zip(chunks, results):
        assert text == chunk.content  # raw fallback


def test_enrich_batch_per_chunk_fallback_recovers_when_individual_calls_work() -> None:
    """If batched call fails but per-chunk requests succeed, the fallback
    recovers every chunk — same final result as if we'd never batched."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(3)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    # First call (batched) returns None; subsequent per-chunk calls succeed.
    responses = [None, "This is one.", "This is two.", "This is three."]
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.side_effect = responses
        results = enricher.enrich_batch(chunks, contexts)

    assert mock_gen.call_count == 4  # 1 batch + 3 per-chunk
    assert [success for _, success in results] == [True, True, True]
    assert "This is one." in results[0][0]
    assert "This is two." in results[1][0]
    assert "This is three." in results[2][0]


def test_enrich_batch_handles_malformed_json() -> None:
    """LLM returns prose instead of JSON → fallback to per-chunk."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(2)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        # Batch returns prose (parse fails) → per-chunk fallback also returns
        # prose. Each per-chunk path runs its own clean() pipeline; since the
        # "description" doesn't start with "This" the cleaned output is still
        # accepted (no semantic validation here) — the test focuses on the
        # control flow, not the description quality.
        mock_gen.return_value = "I'm sorry, I cannot do that."
        results = enricher.enrich_batch(chunks, contexts)

    # 1 batched + 2 per-chunk = 3 calls.
    assert mock_gen.call_count == 3
    # Per-chunk path treats "I'm sorry, I cannot do that." as a valid
    # description (it's non-empty after cleaning), so success=True. That's
    # fine — Fix 2 only protects against EMPTY responses; a hostile-but-
    # parseable response is the user's problem (and a different bug class).
    # The important assertion: control flow reached the per-chunk fallback.
    assert all(text for text, _ in results)


def test_enrich_batch_partial_response_marks_missing_chunks_failed() -> None:
    """LLM returned 2 of 4 expected entries → those 2 succeed, the rest fail."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(4)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        # Only IDs 1 and 3 came back. IDs 2 and 4 must fall back to raw.
        mock_gen.return_value = (
            '[{"id":1,"description":"This is one."},'
            '{"id":3,"description":"This is three."}]'
        )
        results = enricher.enrich_batch(chunks, contexts)

    statuses = [success for _, success in results]
    assert statuses == [True, False, True, False]
    assert "This is one." in results[0][0]
    assert results[1][0] == chunks[1].content
    assert "This is three." in results[2][0]
    assert results[3][0] == chunks[3].content


def test_enrich_batch_tolerates_markdown_fences_around_json() -> None:
    """Some models wrap JSON in ```json ... ``` — the parser must strip them."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(2)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "Sure, here you go:\n"
            "```json\n"
            '[{"id":1,"description":"This is one."},'
            '{"id":2,"description":"This is two."}]\n'
            "```\n"
        )
        results = enricher.enrich_batch(chunks, contexts)

    assert all(success for _, success in results)


def test_enrich_batch_tolerates_trailing_commas() -> None:
    """A common LLM tic. The repair pass strips them before retrying parse."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(2)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            '[{"id":1,"description":"This is one.",},'
            '{"id":2,"description":"This is two.",},]'
        )
        results = enricher.enrich_batch(chunks, contexts)

    assert all(success for _, success in results)


def test_enrich_batch_ignores_hallucinated_extra_ids() -> None:
    """Model fabricated an id=99 entry. Out-of-range IDs are dropped silently."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(2)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            '[{"id":1,"description":"This is one."},'
            '{"id":2,"description":"This is two."},'
            '{"id":99,"description":"Hallucinated."}]'
        )
        results = enricher.enrich_batch(chunks, contexts)

    assert len(results) == 2
    assert all(success for _, success in results)


def test_enrich_batch_empty_input_returns_empty_list() -> None:
    enricher = ContextualEnricher()
    assert enricher.enrich_batch([], []) == []


def test_enrich_batch_single_chunk_bypasses_batch_path() -> None:
    """N=1 takes the per-chunk path (no JSON envelope overhead)."""
    enricher = ContextualEnricher()
    chunk = _make_indexed_chunk(1)

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = "This is the description."
        results = enricher.enrich_batch([chunk], [StageContext(repo_key="t")])

    # Should match the per-chunk path output, NOT a JSON-shaped response.
    assert len(results) == 1
    text, success = results[0]
    assert success is True
    assert text.startswith("[Context:")
    assert chunk.content in text

    # Prompt sent should be the per-chunk template, not the batched header.
    prompt = mock_gen.call_args[0][0]
    assert "STRICT JSON" not in prompt


def test_enrich_batch_salvages_truncated_array_response() -> None:
    """The LLM ran out of max_tokens mid-string — no closing ``]``.

    The parser must still recover every entry it can. Observed in the wild
    against DeepSeek with batch_size=10: the response gets cut mid-string
    on entry 8, so entries 1-7 are complete but the array has no terminator.
    """
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(10)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        # Truncated mid-description for entry 8; entries 1-7 are intact.
        mock_gen.return_value = (
            "```json\n"
            '[{"id":1,"description":"This is one."},'
            '{"id":2,"description":"This is two."},'
            '{"id":3,"description":"This is three."},'
            '{"id":4,"description":"This is four."},'
            '{"id":5,"description":"This is five."},'
            '{"id":6,"description":"This is six."},'
            '{"id":7,"description":"This is seven."},'
            '{"id":8,"description":"This is eight but the model ran out of toke'
            # No closing }, no closing ], no closing ```. Mid-string truncation.
        )
        results = enricher.enrich_batch(chunks, contexts)

    statuses = [success for _, success in results]
    # Entries 1-7 succeed (indices 0-6). Entries 8-10 (indices 7-9) fail.
    assert statuses == [True] * 7 + [False] * 3
    # Spot-check a recovered entry and a failed one.
    assert "This is one." in results[0][0]
    assert "This is seven." in results[6][0]
    assert results[7][0] == chunks[7].content  # truncated entry → raw fallback


def test_enrich_batch_truncated_at_first_entry_falls_back_per_chunk() -> None:
    """When the batched response has no recoverable entries (truncated
    mid-first-entry), the per-chunk fallback is invoked. Each per-chunk
    call uses its own ``max_tokens`` budget so the single-description
    response fits — turning a hard-failed batch into a successful run."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(3)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    responses = [
        # Batched call: truncated mid-first-description, no complete entry.
        '```json\n[{"id":1,"description":"This is one',
        # Per-chunk fallback: each call now succeeds because each has its
        # own fresh max_tokens budget for a single short description.
        "This is one.",
        "This is two.",
        "This is three.",
    ]
    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.side_effect = responses
        results = enricher.enrich_batch(chunks, contexts)

    assert mock_gen.call_count == 4  # 1 batched + 3 per-chunk
    statuses = [success for _, success in results]
    assert statuses == [True, True, True]
    assert "This is one." in results[0][0]
    assert "This is two." in results[1][0]
    assert "This is three." in results[2][0]


def test_enrich_batch_truncated_first_entry_per_chunk_also_fails() -> None:
    """Worst case: batch fails AND per-chunk fallback also fails (e.g. true
    LLM outage, rate-limit exhaustion). Every chunk surfaces success=False
    so Fix 2's cache-skip path keeps retries available on the next index."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(3)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        # Batched returns truncated junk; per-chunk fallback returns None
        # (LLM truly down). The fallback path treats None as failure.
        mock_gen.side_effect = [
            '```json\n[{"id":1,"description":"This is one',
            None,
            None,
            None,
        ]
        results = enricher.enrich_batch(chunks, contexts)

    assert mock_gen.call_count == 4
    statuses = [success for _, success in results]
    assert statuses == [False, False, False]


def test_enrich_batch_max_tokens_scales_with_batch_size() -> None:
    """The batched call must pass a generous max_tokens to avoid the
    truncation bug observed against verbose backends like DeepSeek."""
    enricher = ContextualEnricher()
    chunks = [_make_indexed_chunk(i) for i in range(10)]
    contexts = [StageContext(repo_key="t") for _ in chunks]

    with patch("stropha.inference.generate") as mock_gen:
        mock_gen.return_value = (
            "[" + ",".join(
                f'{{"id":{i+1},"description":"This."}}' for i in range(10)
            ) + "]"
        )
        enricher.enrich_batch(chunks, contexts)

    # Read the keyword args from the generate call.
    call_kwargs = mock_gen.call_args.kwargs
    max_tokens_used = call_kwargs["max_tokens"]

    # Per-chunk default is 256 tokens; for batch_size=10 we need significantly
    # more than 10 × 256 to absorb verbose preamble + code fences + envelope.
    # The exact value is implementation detail; assert the lower bound that
    # avoids the original bug (was 256*10+64 = 2624 — too tight in practice).
    assert max_tokens_used >= 256 * 10 + 32 * 10 + 200
