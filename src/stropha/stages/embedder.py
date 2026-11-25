"""EmbedderStage protocol — text → vector.

Same surface the pre-pipeline ``stropha.embeddings.Embedder`` exposed, plus
the ``Stage`` introspection properties (``stage_name``, ``adapter_id``, …)
so the framework can swap implementations and detect drift.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..pipeline.base import StageHealth


@runtime_checkable
class EmbedderStage(Protocol):
    """Synchronous embedder. Async wrappers, if ever needed, live in subclasses.

    Per CLAUDE.md hygiene rule #2 — callers MUST depend on this protocol,
    never on a concrete client (``voyageai.Client``, ``fastembed.TextEmbedding``).
    """

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

    # ----- Embedder-specific surface --------------------------------------

    @property
    def model_name(self) -> str:
        """Identifier persisted alongside vectors (e.g. ``'voyage-code-3'``)."""

    @property
    def dim(self) -> int:
        """Vector dimensionality. Sizes the sqlite-vec virtual table."""

    @property
    def batch_size(self) -> int:
        """Max items per ``embed_documents`` call. Pipeline batches accordingly."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents (chunks during indexing)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query. Providers may use a distinct input_type."""


class SynchronizedEmbedder:
    """Serialize calls to one warm embedder shared by query and index threads."""

    def __init__(self, delegate: EmbedderStage) -> None:
        self._delegate = delegate
        self._lock = threading.RLock()

    @property
    def stage_name(self) -> str:
        return self._delegate.stage_name

    @property
    def adapter_name(self) -> str:
        return self._delegate.adapter_name

    @property
    def adapter_id(self) -> str:
        return self._delegate.adapter_id

    @property
    def config_schema(self) -> type[BaseModel]:
        return self._delegate.config_schema

    def health(self) -> StageHealth:
        with self._lock:
            return self._delegate.health()

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def dim(self) -> int:
        return self._delegate.dim

    @property
    def batch_size(self) -> int:
        return self._delegate.batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._delegate.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._delegate.embed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
