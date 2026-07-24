"""Shared types and constants for the independent secondary vector space."""

from __future__ import annotations

from typing import TypedDict

SECONDARY_EMBEDDING_DIM = 256


class SecondaryEmbeddingChunk(TypedDict):
    """Chunk payload used to build the independent secondary vector space."""

    chunk_id: str
    content_hash: str
    content: str


class SecondaryEmbeddingVector(TypedDict):
    """One secondary vector tied to the chunk content that produced it."""

    chunk_id: str
    content_hash: str
    embedding: list[float]
