"""Tests for Phase 4 retrieval sub-pipeline (streams as adapters)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from stropha import adapters  # noqa: F401
from stropha.adapters.retrieval.hybrid_rrf import (
    HybridRrfConfig,
    HybridRrfRetrieval,
)
from stropha.adapters.retrieval.streams.fts5_bm25 import (
    Fts5Bm25Config,
    Fts5Bm25Stream,
)
from stropha.adapters.retrieval.streams.like_tokens import (
    LikeTokensConfig,
    LikeTokensStream,
)
from stropha.adapters.retrieval.streams.vec_cosine import (
    VecCosineConfig,
    VecCosineStream,
)
from stropha.config.models import Config, RetrievalFeatures
from stropha.errors import ConfigError
from stropha.meta.stream_weights import StreamWeightProfile, write_stream_weight_profile
from stropha.models import RepoRef, SearchHit
from stropha.pipeline.registry import all_adapters
from stropha.server import _features_snapshot

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingStorage:
    """Records calls + returns deterministic synthetic hits."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    @staticmethod
    def _hit(rank: int, chunk_id: str, score: float = 1.0) -> SearchHit:
        return SearchHit(
            rank=rank,
            score=score,
            rel_path=f"file_{chunk_id}.py",
            language="python",
            kind="file",
            symbol=None,
            start_line=1,
            end_line=10,
            snippet=f"snippet {chunk_id}",
            chunk_id=chunk_id,
        )

    def search_dense(
        self, vec: list[float], k: int, *, repo_id: int | None = None
    ) -> list[SearchHit]:
        self.calls.append(("dense", (k, repo_id)))
        return [self._hit(i + 1, f"d{i}") for i in range(k)]

    def search_bm25(self, query: str, k: int, *, repo_id: int | None = None) -> list[SearchHit]:
        self.calls.append(("bm25", (k, repo_id)))
        return [self._hit(i + 1, f"b{i}") for i in range(k)]

    def search_symbol_tokens(
        self, query: str, k: int, *, repo_id: int | None = None
    ) -> list[SearchHit]:
        self.calls.append(("symbol", (k, repo_id)))
        return [self._hit(i + 1, f"s{i}") for i in range(k)]


class _StubEmbedder:
    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "stub"

    @property
    def adapter_id(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dim(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 16

    @property
    def config_schema(self):
        from pydantic import BaseModel

        class _C(BaseModel):
            pass

        return _C

    def health(self):
        from stropha.pipeline.base import StageHealth

        return StageHealth(status="ready", message="stub")

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_streams_registered() -> None:
    reg = all_adapters()
    # The 3 Phase 4 streams plus the L3 graph-vec stream (Trilha A).
    assert {
        "vec-cosine",
        "fts5-bm25",
        "like-tokens",
        "graph-vec",
        "model2vec-cosine",
    } <= set(reg["retrieval-stream"])


# ---------------------------------------------------------------------------
# Individual streams
# ---------------------------------------------------------------------------


def test_vec_cosine_stream_routes_to_storage_search_dense() -> None:
    storage = _RecordingStorage()
    s = VecCosineStream(VecCosineConfig(k=7), storage=storage)
    out = s.search("query", [0.1, 0.2, 0.3, 0.4])
    assert len(out) == 7
    assert storage.calls == [("dense", (7, None))]
    assert s.adapter_id == "vec-cosine:k=7"


def test_vec_cosine_stream_returns_empty_when_no_query_vec() -> None:
    s = VecCosineStream(VecCosineConfig(), storage=_RecordingStorage())
    assert s.search("query", None) == []


def test_fts5_bm25_stream_routes_to_storage_search_bm25() -> None:
    storage = _RecordingStorage()
    s = Fts5Bm25Stream(Fts5Bm25Config(k=12), storage=storage)
    out = s.search("query", None)
    assert len(out) == 12
    assert storage.calls == [("bm25", (12, None))]


def test_like_tokens_stream_routes_to_storage_search_symbol() -> None:
    storage = _RecordingStorage()
    s = LikeTokensStream(LikeTokensConfig(k=5), storage=storage)
    out = s.search("query", None)
    assert len(out) == 5
    assert storage.calls == [("symbol", (5, None))]


def test_streams_require_storage() -> None:
    for cls in (VecCosineStream, Fts5Bm25Stream, LikeTokensStream):
        with pytest.raises(ValueError):
            cls()


# ---------------------------------------------------------------------------
# Hybrid coordinator
# ---------------------------------------------------------------------------


def test_hybrid_rrf_calls_all_default_streams() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder)
    hits = r.search("hello world")
    # All three streams produced data; RRF fused them.
    assert hits
    seen = {c[0] for c in storage.calls}
    assert seen == {"dense", "bm25", "symbol"}


def test_hybrid_rrf_propagates_repo_id_to_every_stream() -> None:
    storage = _RecordingStorage()
    retrieval = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=_StubEmbedder())

    retrieval.search("hello world", repo_id=42)

    assert {details[1] for _, details in storage.calls} == {42}


def test_hybrid_rrf_propagates_repo_id_through_multi_query() -> None:
    storage = _RecordingStorage()
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(multi_query_enabled=True),
        storage=storage,
        embedder=_StubEmbedder(),
    )

    with patch(
        "stropha.retrieval.multi_query.generate_paraphrases",
        return_value=["original", "alternative"],
    ):
        retrieval.search("original", repo_id=42)

    assert len(storage.calls) == 6
    assert {details[1] for _, details in storage.calls} == {42}


def test_adaptive_weighting_ignores_legacy_environment_toggle(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_ADAPTIVE_WEIGHTING", "0")
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(adaptive_weighting=True),
        storage=_RecordingStorage(),
        embedder=_StubEmbedder(),
    )

    assert retrieval._get_stream_weights("UserService") is not None


def test_recursive_merge_ignores_legacy_environment_toggle(monkeypatch) -> None:
    monkeypatch.setenv("STROPHA_RECURSIVE_RETRIEVAL", "1")
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(recursive_enabled=False),
        storage=_RecordingStorage(),
        embedder=_StubEmbedder(),
    )
    hit = object()

    with patch("stropha.retrieval.recursive.merge_hits") as merge_hits:
        result = retrieval._maybe_recursive_merge([hit])  # type: ignore[list-item]

    assert result == [hit]
    merge_hits.assert_not_called()


def test_feature_snapshot_reports_effective_retrieval_adapter_config() -> None:
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(
            adaptive_weighting=False,
            hyde_enabled=True,
            multi_query_count=4,
            query_rewrite_enabled=True,
            recursive_enabled=False,
            recursive_adjacency=9,
        ),
        storage=_RecordingStorage(),
        embedder=_StubEmbedder(),
    )
    config = Config(
        retrieval_features=RetrievalFeatures(
            adaptive_weighting=True,
            recursive_enabled=True,
            recursive_adjacency=2,
        )
    )
    app = SimpleNamespace(
        config=config,
        built=SimpleNamespace(retrieval=retrieval),
    )

    snapshot = _features_snapshot(app)  # type: ignore[arg-type]

    assert snapshot["adaptive_weighting"]["enabled"] is False
    assert snapshot["hyde"]["enabled"] is True
    assert snapshot["multi_query"]["configured_count"] == 4
    assert snapshot["query_rewrite"]["enabled"] is True
    assert snapshot["recursive_retrieval"] == {
        "enabled": False,
        "adjacency": 9,
        "purpose": "Auto-merge adjacent chunks + promote parent when siblings hit.",
    }


def test_hybrid_rrf_disables_stream_set_to_null() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(
        HybridRrfConfig(streams={"symbol": None}),
        storage=storage,
        embedder=embedder,
    )
    hits = r.search("x")
    seen = {c[0] for c in storage.calls}
    # symbol must NOT be in the call set.
    assert "symbol" not in seen
    assert {"dense", "bm25"} <= seen
    assert hits


def test_hybrid_rrf_skips_embedder_when_no_dense_stream() -> None:
    """Performance: no dense stream → don't pay embed_query cost."""
    storage = _RecordingStorage()
    embedder = _StubEmbedder()

    class _CountingEmbedder(_StubEmbedder):
        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, text):
            self.calls += 1
            return super().embed_query(text)

    embedder = _CountingEmbedder()
    r = HybridRrfRetrieval(
        HybridRrfConfig(streams={"dense": None}),
        storage=storage,
        embedder=embedder,
    )
    r.search("xyz")
    assert embedder.calls == 0


def test_single_stream_passes_candidate_k_to_reranker() -> None:
    storage = _RecordingStorage()
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(
            candidate_k=7,
            streams={"dense": None, "sparse": None},
            reranker={"adapter": "noop", "config": {}},
        ),
        storage=storage,
        embedder=_StubEmbedder(),
    )
    assert retrieval._reranker is not None
    original_rerank = retrieval._reranker.rerank
    seen: list[int] = []

    def recording_rerank(query, hits, *, top_k):
        seen.append(len(hits))
        return original_rerank(query, hits, top_k=top_k)

    retrieval._reranker.rerank = recording_rerank
    retrieval.search("query", top_k=3)
    assert seen == [7]


def test_hybrid_rrf_adapter_id_changes_with_stream_swap() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    a = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder).adapter_id
    b = HybridRrfRetrieval(
        HybridRrfConfig(streams={"dense": {"adapter": "vec-cosine", "config": {"k": 100}}}),
        storage=storage,
        embedder=embedder,
    ).adapter_id
    assert a != b


def test_hybrid_rrf_unknown_sub_adapter_raises() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    with pytest.raises(ConfigError):
        HybridRrfRetrieval(
            HybridRrfConfig(streams={"dense": {"adapter": "no-such"}}),
            storage=storage,
            embedder=embedder,
        )


def test_hybrid_rrf_returns_empty_for_blank_query() -> None:
    storage = _RecordingStorage()
    embedder = _StubEmbedder()
    r = HybridRrfRetrieval(HybridRrfConfig(), storage=storage, embedder=embedder)
    assert r.search("") == []


def test_hybrid_rrf_applies_learned_weights_for_hit_repo(tmp_path: Path) -> None:
    class _RepoStorage(_RecordingStorage):
        def search_dense(
            self, vec: list[float], k: int, *, repo_id: int | None = None
        ) -> list[SearchHit]:
            hit = self._hit(1, "dense").model_copy(
                update={
                    "repo": RepoRef(normalized_key="repo-a"),
                }
            )
            return [hit]

        def search_bm25(self, query: str, k: int, *, repo_id: int | None = None) -> list[SearchHit]:
            hit = self._hit(1, "sparse").model_copy(
                update={
                    "repo": RepoRef(normalized_key="repo-a"),
                }
            )
            return [hit]

    write_stream_weight_profile(
        tmp_path,
        StreamWeightProfile(
            repo="repo-a",
            weights={"dense": 0.5, "sparse": 1.5},
            observations={"dense": 10, "sparse": 10},
            mean_scores={"dense": -0.2, "sparse": 0.2},
            generated_at="2026-07-24T00:00:00+00:00",
        ),
    )
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(
            adaptive_weighting=False,
            learned_weights_dir=str(tmp_path),
            streams={"symbol": None},
        ),
        storage=_RepoStorage(),
        embedder=_StubEmbedder(),
    )

    hits = retrieval.search("query")

    assert hits[0].chunk_id == "sparse"
    assert "learned=none" not in retrieval.adapter_id
