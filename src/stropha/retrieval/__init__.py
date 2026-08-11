"""Core retrieval primitives for hybrid fusion and adapter-driven reranking."""

from .rrf import rrf_fuse
from .search import SearchEngine

__all__ = ["SearchEngine", "rrf_fuse"]
