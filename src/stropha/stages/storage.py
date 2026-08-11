"""StorageStage protocol — persist chunks + vectors and serve read APIs.

A storage adapter encapsulates everything the index needs:

- writes: ``upsert_chunk``, ``register_repo``, ``commit``, ``clear``
- reads: ``search_dense``, ``search_bm25``, ``search_symbol_tokens``,
        ``lookup_by_symbol``, ``file_outline``, ``list_repos``, ``stats``
- introspection: drift cache hooks (``get_enrichment``, ``put_enrichment``,
        ``chunk_is_fresh``)

Phase 2 ships a single adapter, ``sqlite-vec``, that wraps the existing
``stropha.storage.sqlite.Storage``. Future adapters (Phase 4 spec §10.5):
``qdrant``, ``pgvector``, ``lancedb``.

Concrete adapters MAY add backend-specific helpers; the Pipeline / CLI
must depend only on the protocol surface listed above.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..ingest.git_meta import RepoIdentity
from ..models import Chunk, RepoStats, SearchHit, StoredFileMeta
from ..pipeline.base import StageHealth
from ..secondary_embeddings import (
    SECONDARY_EMBEDDING_DIM,
    SecondaryEmbeddingChunk,
    SecondaryEmbeddingVector,
)


@runtime_checkable
class StorageStage(Protocol):
    """Persistence + read backend for the index."""

    # ----- Stage contract --------------------------------------------------

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    @property
    def dim(self) -> int:
        """Vector dimensionality the backend was provisioned with."""

    # ----- Lifecycle ------------------------------------------------------

    def commit(self) -> None: ...

    def clear(self) -> None: ...

    def clear_repo(self, repo_id: int) -> None: ...

    def vacuum(self) -> tuple[int, int]: ...
    """Reclaim unused disk space. Returns (size_before, size_after) in bytes."""

    def close(self) -> None: ...

    # ----- Writes ---------------------------------------------------------

    def register_repo(self, identity: RepoIdentity) -> int: ...

    def upsert_chunk(
        self,
        chunk: Chunk,
        embedding: list[float],
        embedding_model: str,
        embedding_dim: int,
        repo_id: int | None = None,
        embedding_text: str | None = None,
        enricher_id: str | None = None,
    ) -> int: ...

    def chunk_is_fresh(
        self,
        chunk_id: str,
        content_hash: str,
        embedding_model: str,
        enricher_id: str | None = None,
    ) -> bool: ...

    def get_enrichment(self, content_hash: str, enricher_id: str) -> str | None: ...

    def put_enrichment(
        self, content_hash: str, enricher_id: str, embedding_text: str
    ) -> None: ...

    def count_chunks_without_repo(self) -> int: ...

    def backfill_chunks_to_repo(self, repo_id: int, sample_root: Any) -> int: ...

    def set_meta(self, key: str, value: str) -> None: ...

    def get_meta(self, key: str) -> str | None: ...

    # ----- File-level incremental ops (Phase A/B/C) ----------------------

    def file_is_fresh(
        self,
        repo_id: int,
        rel_path: str,
        *,
        mtime: float,
        size_bytes: int,
        enricher_id: str | None = None,
        embedder_model: str | None = None,
    ) -> bool: ...

    def load_file_meta_map(self, repo_id: int) -> dict[str, StoredFileMeta]: ...

    def upsert_file_meta(
        self,
        *,
        repo_id: int,
        rel_path: str,
        content_hash: str,
        size_bytes: int,
        mtime: float,
        chunk_count: int,
        enricher_id: str | None = None,
        embedder_model: str | None = None,
    ) -> None: ...

    def delete_file_meta(self, repo_id: int, rel_paths: list[str]) -> int: ...

    def list_stale_files(
        self, repo_id: int, current_rel_paths: list[str],
    ) -> list[str]: ...

    def rename_chunks(
        self, repo_id: int, old_rel_path: str, new_rel_path: str,
    ) -> int: ...

    def delete_chunks_by_repo_paths(
        self, repo_id: int, rel_paths: list[str],
    ) -> int: ...

    def lookup_repo_id(self, normalized_key: str) -> int | None: ...

    def lookup_by_chunk_id(self, chunk_id: str) -> SearchHit | None: ...

    def get_latest_session(self) -> Any | None: ...

    def chunk_exists(self, chunk_id: str) -> bool: ...

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None: ...

    def delete_chunk(self, chunk_id: str) -> bool: ...

    # ----- Independent secondary embeddings ------------------------------

    def list_chunks_needing_secondary_embeddings(
        self,
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
        limit: int = 1000,
    ) -> list[SecondaryEmbeddingChunk]: ...

    def upsert_secondary_embeddings(
        self,
        vectors: list[SecondaryEmbeddingVector],
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
    ) -> int: ...

    def search_secondary_embeddings(
        self,
        query_vec: list[float],
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
        k: int = 10,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]: ...

    def count_secondary_embeddings(
        self,
        embedder_id: str,
        embedding_dim: int = SECONDARY_EMBEDDING_DIM,
    ) -> int: ...

    # ----- Context manager (lets ``with storage:`` work) ------------------

    def __enter__(self) -> StorageStage: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    # ----- Escape hatch (pragmatic, not part of the public contract) ------
    # ``_conn`` is the underlying driver handle (a ``sqlite3.Connection`` for
    # the ``sqlite-vec`` adapter). Graph helpers, the graphify loader, the
    # cross-session cache and a few CLI commands reach into it to run
    # bespoke SQL that would be overkill to wrap in a Protocol method.
    # Future adapters (lancedb, pgvector, qdrant) MUST expose a compatible
    # handle here (or refactor the consumers to call typed methods); the
    # ``Any`` annotation is a deliberate stop-gap so mypy stays clean while
    # the migration is incremental.
    _conn: Any

    # ----- Reads ----------------------------------------------------------

    def search_dense(
        self,
        query_vec: list[float],
        k: int = 10,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]: ...

    def search_bm25(
        self,
        query: str,
        k: int = 50,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]: ...

    def search_symbol_tokens(
        self,
        query: str,
        k: int = 20,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]: ...

    def lookup_by_symbol(
        self,
        symbol: str,
        limit: int = 10,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]: ...

    def file_outline(
        self,
        rel_path: str,
        *,
        repo_id: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def list_repos(self) -> list[RepoStats]: ...

    def stats(self) -> dict[str, Any]: ...
