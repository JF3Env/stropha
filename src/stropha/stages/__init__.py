"""Stage-specific protocols.

Each module defines the contract a specific stage's adapters must satisfy.
Concrete implementations live under ``stropha.adapters.<stage>``.

Core protocols include walking, chunking, enrichment, embedding, storage,
retrieval, reranking and retrieval streaming.
"""

from .chunker import ChunkerStage, LanguageChunkerStage
from .embedder import EmbedderStage
from .enricher import EnricherStage
from .reranker import RerankerStage
from .retrieval import RetrievalStage
from .retrieval_stream import RetrievalStreamStage
from .storage import StorageStage
from .walker import WalkerStage

__all__ = [
    "ChunkerStage",
    "EmbedderStage",
    "EnricherStage",
    "LanguageChunkerStage",
    "RerankerStage",
    "RetrievalStage",
    "RetrievalStreamStage",
    "StorageStage",
    "WalkerStage",
]
