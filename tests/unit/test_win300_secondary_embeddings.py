"""Focused non-storage integration tests for WIN-300."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
from stropha.errors import ConfigError
from stropha.ingest.secondary_embedding_loader import SecondaryEmbeddingLoader
from stropha.models import SearchHit
from stropha.pipeline.builder import build_stages
from stropha.retrieval.query_rewrite import QueryRewriteResult


class _PrimaryEmbedder:
    adapter_id = "primary:v1"
    adapter_name = "primary"
    model_name = "primary"
    dim = 4
    batch_size = 8

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0] * self.dim


class _SecondaryEmbedder:
    adapter_name = "model2vec"
    model_name = "potion-code"
    dim = 256
    batch_size = 2

    def __init__(self, adapter_id: str = "model2vec:v1:normalize=true") -> None:
        self.adapter_id = adapter_id
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [[float(index)] * self.dim for index, _ in enumerate(texts, start=1)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.5] * self.dim


class _SecondaryStorage:
    def __init__(self) -> None:
        self.chunks = {
            "a": ("hash-a", "content a"),
            "b": ("hash-b", "content b"),
        }
        self.vectors: dict[str, tuple[str, str, list[float]]] = {}
        self.commits = 0

    def list_chunks_needing_secondary_embeddings(
        self, embedder_id: str, embedding_dim: int = 256, limit: int = 1000
    ) -> list[dict[str, str]]:
        assert embedding_dim == 256
        pending = []
        for chunk_id, (content_hash, content) in self.chunks.items():
            current = self.vectors.get(chunk_id)
            if current is None or current[:2] != (content_hash, embedder_id):
                pending.append(
                    {
                        "chunk_id": chunk_id,
                        "content_hash": content_hash,
                        "content": content,
                    }
                )
        return pending[:limit]

    def upsert_secondary_embeddings(
        self, vectors, embedder_id: str, embedding_dim: int = 256
    ) -> int:
        upserted = 0
        for item in vectors:
            current = self.chunks.get(item["chunk_id"])
            if current is None or current[0] != item["content_hash"]:
                continue
            assert len(item["embedding"]) == embedding_dim
            self.vectors[item["chunk_id"]] = (
                item["content_hash"],
                embedder_id,
                item["embedding"],
            )
            upserted += 1
        return upserted

    def count_secondary_embeddings(self, embedder_id: str, embedding_dim: int = 256) -> int:
        assert embedding_dim == 256
        return sum(
            content_hash == self.chunks[chunk_id][0] and stored_id == embedder_id
            for chunk_id, (content_hash, stored_id, _vector) in self.vectors.items()
        )

    def commit(self) -> None:
        self.commits += 1


def _builder_config(tmp_path, model2vec_spec):
    return {
        "walker": {"adapter": "stub", "config": {}},
        "chunker": {"adapter": "stub", "config": {}},
        "enricher": {"adapter": "stub", "config": {}},
        "embedder": {"adapter": "stub", "config": {}},
        "storage": {"adapter": "sqlite-vec", "config": {"path": str(tmp_path / "i.db")}},
        "retrieval": {
            "adapter": "hybrid-rrf",
            "config": {"streams": {"model2vec": model2vec_spec}},
        },
    }


def _patch_simple_stages(monkeypatch) -> _PrimaryEmbedder:
    primary = _PrimaryEmbedder()

    def build_simple(_resolved, stage):
        if stage == "embedder":
            return primary
        return SimpleNamespace(adapter_id=f"{stage}:stub")

    monkeypatch.setattr("stropha.pipeline.builder._build_simple", build_simple)
    return primary


def _mock_model2vec_module(dim: int = 256) -> MagicMock:
    model = MagicMock()
    model.encode.return_value = np.array([[0.1] * dim])
    module = MagicMock()
    module.StaticModel.from_pretrained.return_value = model
    return module


def test_builder_disabled_does_not_construct_model2vec(tmp_path, monkeypatch) -> None:
    _patch_simple_stages(monkeypatch)
    config = _builder_config(tmp_path, None)
    with patch(
        "stropha.adapters.embedder.model2vec.Model2VecEmbedder.__init__",
        side_effect=AssertionError("must stay unloaded"),
    ):
        built = build_stages(config, open_storage=False)
    assert built.secondary_embedder is None


def test_builder_enabled_shares_one_secondary_embedder(tmp_path, monkeypatch) -> None:
    _patch_simple_stages(monkeypatch)
    spec = {
        "adapter": "model2vec-cosine",
        "config": {"dim": 256, "normalize": True, "k": 7},
    }
    with patch.dict("sys.modules", {"model2vec": _mock_model2vec_module()}):
        built = build_stages(_builder_config(tmp_path, spec))
    try:
        stream = built.retrieval._streams["model2vec"]  # type: ignore[attr-defined]
        assert stream._embedder is built.secondary_embedder
        assert stream.k == 7
    finally:
        built.storage.close()


@pytest.mark.parametrize(
    "bad_config",
    [
        {"dim": 128, "normalize": True},
        {"dim": 256, "normalize": False},
    ],
)
def test_builder_rejects_secondary_dim_or_normalize(tmp_path, monkeypatch, bad_config) -> None:
    _patch_simple_stages(monkeypatch)
    spec = {"adapter": "model2vec-cosine", "config": bad_config}
    with pytest.raises(ConfigError):
        build_stages(_builder_config(tmp_path, spec), open_storage=False)


def test_loader_backfill_noop_content_drift_and_adapter_drift() -> None:
    storage = _SecondaryStorage()
    embedder = _SecondaryEmbedder()

    first = SecondaryEmbeddingLoader(storage, embedder).load()
    assert first.embedded == 2
    assert len(embedder.document_calls) == 1

    noop = SecondaryEmbeddingLoader(storage, embedder).load()
    assert noop.embedded == 0
    assert noop.skipped_already_current == 2
    assert len(embedder.document_calls) == 1

    storage.chunks["a"] = ("hash-a2", "changed content a")
    content_drift = SecondaryEmbeddingLoader(storage, embedder).load()
    assert content_drift.embedded == 1
    assert embedder.document_calls[-1] == ["changed content a"]

    replacement = _SecondaryEmbedder("model2vec:v2:normalize=true")
    adapter_drift = SecondaryEmbeddingLoader(storage, replacement).load()
    assert adapter_drift.embedded == 2
    assert storage.count_secondary_embeddings(replacement.adapter_id) == 2


def _hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        rank=1,
        score=1.0,
        rel_path="x.py",
        language="python",
        kind="function",
        start_line=1,
        end_line=1,
        snippet="x",
        chunk_id=chunk_id,
    )


def test_model2vec_stream_embeds_rewritten_literal_not_primary_hyde() -> None:
    storage = MagicMock()
    storage.search_dense.return_value = [_hit("primary")]
    storage.search_secondary_embeddings.return_value = [_hit("secondary")]
    primary = _PrimaryEmbedder()
    secondary = _SecondaryEmbedder()
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(
            query_rewrite_enabled=True,
            hyde_enabled=True,
            streams={
                "sparse": None,
                "symbol": None,
                "model2vec": {
                    "adapter": "model2vec-cosine",
                    "config": {"dim": 256, "normalize": True},
                },
            },
        ),
        storage=storage,
        embedder=primary,
        secondary_embedder=secondary,
    )

    with (
        patch(
            "stropha.retrieval.query_rewrite.rewrite_query",
            return_value=QueryRewriteResult(query="rewritten literal", reason="success"),
        ),
        patch("stropha.retrieval.hyde.maybe_hyde_rewrite", return_value="hyde vector text"),
    ):
        hits = retrieval.search("original")

    assert hits
    assert primary.queries == ["hyde vector text"]
    assert secondary.query_calls == ["rewritten literal"]
    args, kwargs = storage.search_secondary_embeddings.call_args
    assert args[1] == secondary.adapter_id
    assert kwargs == {"embedding_dim": 256, "k": 50}


@pytest.mark.asyncio
async def test_mcp_lifespan_uses_configured_retrieval(tmp_path) -> None:
    from stropha.server_runtime import lifespan

    storage = MagicMock()
    storage.list_sessions.return_value = []
    configured_retrieval = MagicMock()
    built = SimpleNamespace(
        embedder=_PrimaryEmbedder(),
        storage=storage,
        retrieval=configured_retrieval,
    )
    config = SimpleNamespace(
        log_level="INFO",
        target_repo=tmp_path,
        mcp_watch=False,
        mcp_watch_interval=1.0,
        mcp_watch_debounce=1.0,
        resolve_index_path=lambda: tmp_path / "index.db",
    )

    runtime_owner = SimpleNamespace()
    async with lifespan(
        runtime_owner,
        version="test",
        get_config_fn=lambda: config,
        configure_logging_fn=lambda _level: None,
        load_pipeline_config_fn=lambda: {},
        build_stages_fn=lambda _resolved: built,
        watch_controller_cls=MagicMock,
    ) as app:
        assert app.search_engine is configured_retrieval

    from stropha.server_runtime import close_shared_runtime

    close_shared_runtime(runtime_owner)
