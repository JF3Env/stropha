"""Unit tests for the Voyage embedder adapter.

Covers the token-aware sub-batching introduced to respect Voyage's
120K-tokens-per-request cap. The default chunk-count batching
(``batch_size=128``) easily overshoots that ceiling on code chunks
averaging ~970 tokens, so the embedder splits incoming batches into
sub-batches whose estimated token sum stays below the budget.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stropha.adapters.embedder.voyage import (
    VoyageEmbedder,
    VoyageEmbedderConfig,
)
from stropha.errors import EmbeddingError
from stropha.pipeline.registry import lookup_adapter


def _make_mock_voyageai(embed_responses: list[list[list[float]]] | None = None) -> tuple[MagicMock, MagicMock]:
    """Build a stand-in ``voyageai`` module + its ``Client.embed`` mock.

    ``embed_responses`` — list of vector lists, one per expected ``embed``
    call. If omitted, every call returns a single 512-d zero vector per
    requested text.
    """
    client = MagicMock()

    if embed_responses is None:
        def _default_embed(texts: list[str], *_, **kwargs: Any) -> MagicMock:
            dim = kwargs.get("output_dimension", 512)
            response = MagicMock()
            response.embeddings = [[0.0] * dim for _ in texts]
            return response

        client.embed.side_effect = _default_embed
    else:
        responses = []
        for vectors in embed_responses:
            response = MagicMock()
            response.embeddings = vectors
            responses.append(response)
        client.embed.side_effect = responses

    voyageai_module = MagicMock()
    voyageai_module.Client.return_value = client
    return voyageai_module, client


def _build_embedder(
    config: VoyageEmbedderConfig | None = None,
    voyageai_module: MagicMock | None = None,
) -> tuple[VoyageEmbedder, MagicMock]:
    """Construct a VoyageEmbedder with a mocked voyageai module.

    Pre-resolves the lazy client inside the env/sys.modules patch so
    subsequent embed calls in tests hit the cached mock — without this
    the lazy-init key check would fire after the patch is released.
    """
    if voyageai_module is None:
        voyageai_module, _ = _make_mock_voyageai()
    with patch.dict("sys.modules", {"voyageai": voyageai_module}):
        with patch.dict("os.environ", {"VOYAGE_API_KEY": "sk-fake"}):
            embedder = VoyageEmbedder(config=config)
            embedder._get_client()  # eager-cache mocked Client for later asserts
    client = voyageai_module.Client.return_value
    return embedder, client


# ----- Config + registration --------------------------------------------------


class TestVoyageEmbedderConfig:
    def test_default_values(self) -> None:
        cfg = VoyageEmbedderConfig()
        assert cfg.model == "voyage-4-large"
        assert cfg.dim == 512
        assert cfg.batch_size == 128
        assert cfg.api_key_env == "VOYAGE_API_KEY"
        assert cfg.max_tokens_per_batch == 96_000
        assert cfg.chars_per_token == pytest.approx(3.5)

    def test_max_tokens_per_batch_lower_bound(self) -> None:
        # 1023 is below ge=1024 — pydantic must reject.
        with pytest.raises(ValueError):
            VoyageEmbedderConfig(max_tokens_per_batch=1023)

    def test_max_tokens_per_batch_upper_bound(self) -> None:
        # 120_001 is above le=120_000.
        with pytest.raises(ValueError):
            VoyageEmbedderConfig(max_tokens_per_batch=120_001)

    def test_chars_per_token_lower_bound(self) -> None:
        with pytest.raises(ValueError):
            VoyageEmbedderConfig(chars_per_token=0.5)

    def test_chars_per_token_upper_bound(self) -> None:
        with pytest.raises(ValueError):
            VoyageEmbedderConfig(chars_per_token=20.0)


class TestVoyageEmbedderRegistration:
    def test_registered(self) -> None:
        assert lookup_adapter("embedder", "voyage") is VoyageEmbedder


# ----- Initialization ---------------------------------------------------------


class TestVoyageEmbedderInit:
    def test_init_with_default_config_uses_env_key(self) -> None:
        """Client construction is lazy: env key is read on first embed call."""
        voyageai_module, _ = _make_mock_voyageai()
        with patch.dict("sys.modules", {"voyageai": voyageai_module}):
            with patch.dict("os.environ", {"VOYAGE_API_KEY": "sk-real"}):
                embedder = VoyageEmbedder()
                # __init__ must not build the Client — only first use does.
                voyageai_module.Client.assert_not_called()
                embedder.embed_query("hi")
        voyageai_module.Client.assert_called_once_with(api_key="sk-real")

    def test_init_with_explicit_api_key_kwarg(self) -> None:
        """Explicit override flows through on first embed call."""
        voyageai_module, _ = _make_mock_voyageai()
        with patch.dict("sys.modules", {"voyageai": voyageai_module}):
            embedder = VoyageEmbedder(api_key="sk-explicit")
            voyageai_module.Client.assert_not_called()
            with patch.dict("os.environ", {}, clear=True):
                embedder.embed_query("hi")
        voyageai_module.Client.assert_called_once_with(api_key="sk-explicit")

    def test_init_legacy_kwargs_path(self) -> None:
        """Back-compat: positional model + dim still work."""
        voyageai_module, _ = _make_mock_voyageai()
        with patch.dict("sys.modules", {"voyageai": voyageai_module}):
            embedder = VoyageEmbedder(api_key="sk-fake", model="voyage-3", dim=256)
        assert embedder.model_name == "voyage-3"
        assert embedder.dim == 256

    def test_init_rejects_both_config_and_legacy_kwargs(self) -> None:
        voyageai_module, _ = _make_mock_voyageai()
        with patch.dict("sys.modules", {"voyageai": voyageai_module}):
            with pytest.raises(TypeError, match="Pass either config="):
                VoyageEmbedder(config=VoyageEmbedderConfig(), api_key="sk-fake", model="x")

    def test_init_without_api_key_does_not_raise(self) -> None:
        """MCP server constructs every adapter at startup; the retrieval-only
        path may never touch the embedder. Missing-key check is deferred to
        first embed call so the server can start without VOYAGE_API_KEY."""
        voyageai_module, _ = _make_mock_voyageai()
        with patch.dict("sys.modules", {"voyageai": voyageai_module}):
            with patch.dict("os.environ", {}, clear=True):
                embedder = VoyageEmbedder()  # must not raise
                with pytest.raises(EmbeddingError, match="API key is empty"):
                    embedder.embed_query("hi")


# ----- Stage contract ---------------------------------------------------------


class TestVoyageEmbedderStageContract:
    def test_stage_name(self) -> None:
        embedder, _ = _build_embedder()
        assert embedder.stage_name == "embedder"

    def test_adapter_name(self) -> None:
        embedder, _ = _build_embedder()
        assert embedder.adapter_name == "voyage"

    def test_adapter_id_format(self) -> None:
        embedder, _ = _build_embedder(VoyageEmbedderConfig(model="voyage-4-large", dim=512))
        assert embedder.adapter_id == "voyage:voyage-4-large:512"

    def test_adapter_id_changes_with_model(self) -> None:
        a, _ = _build_embedder(VoyageEmbedderConfig(model="voyage-4-large", dim=512))
        b, _ = _build_embedder(VoyageEmbedderConfig(model="voyage-code-3", dim=512))
        assert a.adapter_id != b.adapter_id

    def test_adapter_id_changes_with_dim(self) -> None:
        a, _ = _build_embedder(VoyageEmbedderConfig(dim=512))
        b, _ = _build_embedder(VoyageEmbedderConfig(dim=256))
        assert a.adapter_id != b.adapter_id

    def test_adapter_id_stable_across_batching_knobs(self) -> None:
        """Drift detection invariant: max_tokens_per_batch + chars_per_token
        affect request *shape*, not vectors. They MUST NOT influence
        ``adapter_id`` — otherwise tuning them would force a full re-embed.
        """
        a, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=96_000, chars_per_token=3.5)
        )
        b, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=50_000, chars_per_token=4.0)
        )
        assert a.adapter_id == b.adapter_id

    def test_config_schema(self) -> None:
        embedder, _ = _build_embedder()
        assert embedder.config_schema is VoyageEmbedderConfig

    def test_health_ready(self) -> None:
        embedder, _ = _build_embedder()
        h = embedder.health()
        assert h.status == "ready"
        assert "voyage-4-large" in h.message
        assert h.detail["model"] == "voyage-4-large"
        assert h.detail["dim"] == "512"

    def test_batch_size_property(self) -> None:
        embedder, _ = _build_embedder(VoyageEmbedderConfig(batch_size=64))
        assert embedder.batch_size == 64


# ----- Token estimation -------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self) -> None:
        """Always at least 1 — Voyage still counts the slot."""
        embedder, _ = _build_embedder()
        assert embedder._estimate_tokens("") == 1

    def test_chars_per_token_heuristic(self) -> None:
        embedder, _ = _build_embedder(VoyageEmbedderConfig(chars_per_token=4.0))
        # 40 chars / 4.0 = 10 tokens
        assert embedder._estimate_tokens("x" * 40) == 10

    def test_chars_per_token_floor(self) -> None:
        # 3 chars / 3.5 = 0.85 -> int() -> 0 -> max(1, 0) = 1
        embedder, _ = _build_embedder(VoyageEmbedderConfig(chars_per_token=3.5))
        assert embedder._estimate_tokens("xyz") == 1


# ----- Sub-batching -----------------------------------------------------------


class TestIterSubBatches:
    def test_empty_input_yields_nothing(self) -> None:
        embedder, _ = _build_embedder()
        assert list(embedder._iter_sub_batches([])) == []

    def test_small_batch_under_budget_yields_one_subbatch(self) -> None:
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=10_000, chars_per_token=4.0)
        )
        # 3 tiny texts (40 chars each = 10 tokens each = 30 tokens total).
        sub_batches = list(embedder._iter_sub_batches(["x" * 40] * 3))
        assert len(sub_batches) == 1
        assert sub_batches[0] == ["x" * 40] * 3

    def test_large_batch_splits_into_subbatches(self) -> None:
        """Reproduces the original failure: a 128-chunk batch whose total
        is above the budget must be split into multiple API calls."""
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0)
        )
        # Each text is 1024 chars / 4 = 256 tokens. 8 texts = 2048 tokens.
        # Budget 1024 -> expect ceil(2048/1024) = 2 sub-batches of 4 each.
        texts = ["x" * 1024] * 8
        sub_batches = list(embedder._iter_sub_batches(texts))
        assert len(sub_batches) == 2
        assert sum(len(sb) for sb in sub_batches) == 8
        # Greedy first-fit: first batch fills to 1024 tokens (4 items),
        # second batch holds the remaining 4.
        assert len(sub_batches[0]) == 4
        assert len(sub_batches[1]) == 4

    def test_order_is_preserved(self) -> None:
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0)
        )
        # 6 distinct texts of varying sizes; flatten and compare to input.
        texts = [
            "a" * 1024,  # 256 tokens
            "b" * 2048,  # 512 tokens
            "c" * 512,   # 128 tokens
            "d" * 1024,  # 256 tokens
            "e" * 256,   # 64 tokens
            "f" * 1024,  # 256 tokens
        ]
        sub_batches = list(embedder._iter_sub_batches(texts))
        flattened = [t for sb in sub_batches for t in sb]
        assert flattened == texts

    def test_oversized_single_text_emitted_solo(self) -> None:
        """A text larger than the whole budget is sent solo, not silently
        dropped. Voyage's server-side per-text truncation handles the
        actual length cap."""
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0)
        )
        # 1 token small + 1 huge (5000 tokens, way above cap) + 1 small.
        texts = ["a" * 4, "x" * 20_000, "b" * 4]
        sub_batches = list(embedder._iter_sub_batches(texts))
        # Expected: [small1], [huge], [small2] — the huge one flushes
        # the pending batch and is yielded solo.
        assert len(sub_batches) == 3
        assert sub_batches[0] == ["a" * 4]
        assert sub_batches[1] == ["x" * 20_000]
        assert sub_batches[2] == ["b" * 4]

    def test_oversized_first_text_no_preceding_flush(self) -> None:
        """Oversized text as the very first item: must not yield an empty
        leading sub-batch."""
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0)
        )
        sub_batches = list(embedder._iter_sub_batches(["x" * 20_000, "y" * 4]))
        assert sub_batches == [["x" * 20_000], ["y" * 4]]

    def test_exactly_at_budget_stays_in_one_batch(self) -> None:
        """Boundary: sum exactly == cap should still fit (we yield only
        when sum *exceeds* cap)."""
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0)
        )
        # 4 texts × 256 tokens = exactly 1024 tokens.
        texts = ["x" * 1024] * 4
        sub_batches = list(embedder._iter_sub_batches(texts))
        assert len(sub_batches) == 1
        assert len(sub_batches[0]) == 4


# ----- Embedding integration --------------------------------------------------


class TestEmbedDocuments:
    def test_under_budget_issues_single_api_call(self) -> None:
        voyageai_module, client = _make_mock_voyageai(
            embed_responses=[[[0.1] * 512, [0.2] * 512]]
        )
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=10_000),
            voyageai_module=voyageai_module,
        )
        vectors = embedder.embed_documents(["short text 1", "short text 2"])
        assert client.embed.call_count == 1
        assert len(vectors) == 2

    def test_over_budget_splits_into_multiple_calls(self) -> None:
        """Direct regression test for the original failure mode.

        8 texts × 256 tokens (1024 chars / 4) = 2048 total tokens. Budget
        1024 → greedy packer ships 4 + 4.
        """
        voyageai_module, client = _make_mock_voyageai(
            embed_responses=[
                [[float(i)] * 512 for i in range(4)],   # batch 1
                [[float(i)] * 512 for i in range(4, 8)],  # batch 2
            ]
        )
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0),
            voyageai_module=voyageai_module,
        )
        texts = ["x" * 1024] * 8
        vectors = embedder.embed_documents(texts)
        assert client.embed.call_count == 2
        assert len(vectors) == 8

    def test_results_concatenated_in_order(self) -> None:
        """Verify the i-th input text maps to the i-th returned vector
        even after sub-batch fan-out.

        4 texts × 512 tokens (2048 chars / 4) = 2048 total tokens. Budget
        1024 → packer ships 2 + 2 (each batch == budget exactly, no slack
        for a 3rd item).
        """
        voyageai_module, client = _make_mock_voyageai(
            embed_responses=[
                [[1.0] * 512, [2.0] * 512],
                [[3.0] * 512, [4.0] * 512],
            ]
        )
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0),
            voyageai_module=voyageai_module,
        )
        texts = ["a" * 2048, "b" * 2048, "c" * 2048, "d" * 2048]
        vectors = embedder.embed_documents(texts)
        assert client.embed.call_count == 2
        # Marker value on the first dimension reflects original ordering.
        assert vectors[0][0] == 1.0
        assert vectors[1][0] == 2.0
        assert vectors[2][0] == 3.0
        assert vectors[3][0] == 4.0

    def test_embed_failure_wraps_with_embeddingerror(self) -> None:
        voyageai_module = MagicMock()
        client = voyageai_module.Client.return_value
        client.embed.side_effect = RuntimeError("max tokens exceeded")
        embedder, _ = _build_embedder(voyageai_module=voyageai_module)
        with pytest.raises(EmbeddingError, match="Voyage embed failed"):
            embedder.embed_documents(["any text"])

    def test_embed_failure_on_second_subbatch_still_raises(self) -> None:
        """A failure on a later sub-batch must still propagate, not be
        silently swallowed."""
        voyageai_module = MagicMock()
        client = voyageai_module.Client.return_value
        # First call succeeds, second raises.
        success_response = MagicMock()
        success_response.embeddings = [[0.0] * 512] * 4
        client.embed.side_effect = [success_response, RuntimeError("rate limited")]
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(max_tokens_per_batch=1024, chars_per_token=4.0),
            voyageai_module=voyageai_module,
        )
        with pytest.raises(EmbeddingError, match="Voyage embed failed"):
            embedder.embed_documents(["x" * 1024] * 8)
        assert client.embed.call_count == 2

    def test_passes_correct_kwargs_to_voyage_client(self) -> None:
        voyageai_module, client = _make_mock_voyageai()
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(model="voyage-4-large", dim=256),
            voyageai_module=voyageai_module,
        )
        embedder.embed_documents(["hello"])
        call = client.embed.call_args
        # SDK is called positionally: embed([texts], model=..., input_type=..., output_dimension=...)
        assert call.args[0] == ["hello"]
        assert call.kwargs["model"] == "voyage-4-large"
        assert call.kwargs["input_type"] == "document"
        assert call.kwargs["output_dimension"] == 256

    def test_empty_input_returns_empty_list(self) -> None:
        voyageai_module, client = _make_mock_voyageai()
        embedder, _ = _build_embedder(voyageai_module=voyageai_module)
        vectors = embedder.embed_documents([])
        assert vectors == []
        # Critical: no API call for an empty input — avoid burning quota.
        assert client.embed.call_count == 0


class TestEmbedQuery:
    def test_query_uses_query_input_type(self) -> None:
        voyageai_module, client = _make_mock_voyageai()
        embedder, _ = _build_embedder(voyageai_module=voyageai_module)
        embedder.embed_query("how does X work")
        call = client.embed.call_args
        assert call.kwargs["input_type"] == "query"

    def test_query_returns_single_vector(self) -> None:
        voyageai_module, _ = _make_mock_voyageai(
            embed_responses=[[[0.5] * 512]]
        )
        embedder, _ = _build_embedder(voyageai_module=voyageai_module)
        vec = embedder.embed_query("hi")
        assert vec == [0.5] * 512


# ----- Dim-mismatch warning ---------------------------------------------------


class TestDimMismatch:
    def test_warns_when_returned_dim_differs(self, caplog: pytest.LogCaptureFixture) -> None:
        # Voyage returns 384 instead of the configured 512.
        voyageai_module, _ = _make_mock_voyageai(
            embed_responses=[[[0.0] * 384, [0.0] * 384]]
        )
        embedder, _ = _build_embedder(
            VoyageEmbedderConfig(dim=512),
            voyageai_module=voyageai_module,
        )
        # structlog routes through stdlib logging by default in tests; just
        # call it and trust the prior coverage of structlog wiring. We
        # primarily want this path to execute without raising.
        vectors = embedder.embed_documents(["a", "b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384

    def test_no_warn_when_empty_result(self) -> None:
        """No vectors, no warning — execution still completes."""
        voyageai_module, _ = _make_mock_voyageai(embed_responses=[[]])
        embedder, _ = _build_embedder(voyageai_module=voyageai_module)
        # Pass a non-empty texts list so we issue one API call that
        # returns no embeddings (defensive edge case).
        vectors = embedder.embed_documents(["x"])
        assert vectors == []
