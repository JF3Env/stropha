"""Unit tests for ``BuiltStages.indexing_hash`` (WIN-176).

The hash drives the WIN-177 index cache lookup. Two builds with the
same walker/chunker/enricher/embedder adapter_id MUST collide; any
change to those four MUST produce a fresh hash; storage and retrieval
changes MUST NOT affect the hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from stropha.pipeline.builder import BuiltStages


@dataclass
class _StubStage:
    adapter_id: str


@dataclass
class _BatchingStage(_StubStage):
    def enrich_batch(self, chunks, contexts):
        return []


def _built(
    *,
    walker_id: str = "walker:v1",
    chunker_id: str = "chunker:v1",
    enricher_id: str = "enricher:v1",
    embedder_id: str = "embedder:v1",
    storage_id: str = "storage:v1",
    retrieval_id: str = "retrieval:v1",
    secondary_embedder_id: str | None = None,
) -> BuiltStages:
    return BuiltStages(
        walker=_StubStage(walker_id),         # type: ignore[arg-type]
        chunker=_StubStage(chunker_id),       # type: ignore[arg-type]
        enricher=_StubStage(enricher_id),     # type: ignore[arg-type]
        embedder=_StubStage(embedder_id),     # type: ignore[arg-type]
        storage=_StubStage(storage_id),       # type: ignore[arg-type]
        retrieval=_StubStage(retrieval_id),   # type: ignore[arg-type]
        resolved={},
        secondary_embedder=(
            _StubStage(secondary_embedder_id)  # type: ignore[arg-type]
            if secondary_embedder_id is not None
            else None
        ),
    )


class TestIndexingHash:
    def test_stable_across_runs(self) -> None:
        a = _built().indexing_hash()
        b = _built().indexing_hash()
        assert a == b

    def test_12_hex_chars(self) -> None:
        h = _built().indexing_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_changes_when_walker_changes(self) -> None:
        a = _built(walker_id="v1").indexing_hash()
        b = _built(walker_id="v2").indexing_hash()
        assert a != b

    def test_changes_when_chunker_changes(self) -> None:
        assert _built(chunker_id="v1").indexing_hash() != _built(chunker_id="v2").indexing_hash()

    def test_changes_when_enricher_changes(self) -> None:
        assert _built(enricher_id="v1").indexing_hash() != _built(enricher_id="v2").indexing_hash()

    def test_changes_with_effective_enricher_batch_size(self, monkeypatch) -> None:
        built = _built()
        built.enricher = _BatchingStage("enricher:v1")  # type: ignore[assignment]
        monkeypatch.setattr(
            "stropha.config.get_config",
            lambda: SimpleNamespace(
                inference=SimpleNamespace(effective_batch_size=lambda: 4)
            ),
        )
        first = built.indexing_hash()
        monkeypatch.setattr(
            "stropha.config.get_config",
            lambda: SimpleNamespace(
                inference=SimpleNamespace(effective_batch_size=lambda: 8)
            ),
        )
        assert first != built.indexing_hash()

    def test_changes_when_embedder_changes(self) -> None:
        assert _built(embedder_id="v1").indexing_hash() != _built(embedder_id="v2").indexing_hash()

    def test_changes_when_secondary_embedder_changes(self) -> None:
        assert _built().indexing_hash() != _built(
            secondary_embedder_id="model2vec:v1"
        ).indexing_hash()
        assert _built(
            secondary_embedder_id="model2vec:v1"
        ).indexing_hash() != _built(secondary_embedder_id="model2vec:v2").indexing_hash()

    def test_unchanged_when_storage_changes(self) -> None:
        """Storage swap (e.g. sqlite-vec → lancedb) doesn't change WHAT
        chunks are in the index, only how they're queried back. Cache
        should be reusable."""
        a = _built(storage_id="sqlite-vec").indexing_hash()
        b = _built(storage_id="lancedb").indexing_hash()
        assert a == b

    def test_unchanged_when_retrieval_changes(self) -> None:
        """Retrieval swap (e.g. hybrid-rrf weights change) doesn't
        re-index, only re-queries. Cache reuse is correct."""
        a = _built(retrieval_id="hybrid-rrf:k=60").indexing_hash()
        b = _built(retrieval_id="hybrid-rrf-2:k=42").indexing_hash()
        assert a == b

    def test_tolerates_missing_adapter_id_attribute(self) -> None:
        """A stage without adapter_id (e.g. a buggy plugin) falls back
        to '?' rather than raising."""
        plain_stage = type("Plain", (), {})()
        built = BuiltStages(
            walker=plain_stage,     # type: ignore[arg-type]
            chunker=_StubStage("c"),    # type: ignore[arg-type]
            enricher=_StubStage("e"),   # type: ignore[arg-type]
            embedder=_StubStage("b"),   # type: ignore[arg-type]
            storage=_StubStage("s"),    # type: ignore[arg-type]
            retrieval=_StubStage("r"),  # type: ignore[arg-type]
            resolved={},
        )
        h = built.indexing_hash()
        assert len(h) == 12  # didn't raise, produced a valid hash
