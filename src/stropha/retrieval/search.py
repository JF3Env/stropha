"""Hybrid retrieval (spec §6.1): dense (vec) + sparse (FTS5) fused via RRF.

Phase 1: no reranker yet — spec §16 puts Voyage `rerank-2.5` in Phase 2.
"""

from __future__ import annotations

import os

from ..embeddings.base import Embedder
from ..logging import get_logger
from ..models import SearchHit
from ..storage import Storage
from .rrf import DEFAULT_K, rrf_fuse, rrf_fuse_named

log = get_logger(__name__)

# Spec §6.1 pipeline default: top 50 from each side, fuse, return top 10.
CANDIDATE_K = 50


class SearchEngine:
    """Hybrid dense+BM25 + RRF. Falls back gracefully if one side is empty."""

    def __init__(self, storage: Storage, embedder: Embedder) -> None:
        self._storage = storage
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_k: int = CANDIDATE_K,
        rrf_k: int = DEFAULT_K,
        decompose: bool | None = None,
        context_budget: int | None = None,
    ) -> list[SearchHit]:
        """Three-stream hybrid retrieval, fused with RRF.

        Streams:
        - dense  — sqlite-vec ANN over the embedder (semantic similarity).
        - sparse — FTS5 BM25 over content + path + symbol (lexical recall).
        - symbol — direct match on the `symbol` column for identifier tokens
                   in the query (spec §6.3.5 query routing).
        Empty streams are dropped before fusion.

        Query decomposition (optional):
        When enabled (decompose=True or STROPHA_QUERY_DECOMPOSITION_ENABLED=1),
        compound queries like "how does X work and what calls Y" are split
        into atomic sub-queries, searched separately, and fused via RRF.

        Token-budget packing (WIN-208):
        When ``context_budget`` is set, the score-ranked output is greedy-
        packed up to that token budget (using ``estimate_content_tokens``)
        and the truncated tail is dropped. The returned list never exceeds
        the budget — except in the single-first-hit-oversize escape path,
        where the top hit is returned alone even if oversize so the caller
        sees something rather than nothing. ``top_k`` still bounds the
        upper end of what's considered for packing; the packed result may
        contain fewer items.
        """
        if not query.strip():
            return []

        # Check for query decomposition
        if decompose is None:
            decompose = os.environ.get("STROPHA_QUERY_DECOMPOSITION_ENABLED", "0") == "1"

        if decompose:
            from .query_decomposition import decompose_query

            result = decompose_query(query)
            if len(result.sub_queries) > 1:
                log.info(
                    "search.decomposed",
                    original=query[:50],
                    sub_count=len(result.sub_queries),
                    method=result.method,
                )
                # Search each sub-query and fuse results
                all_hits = []
                for sub_query in result.sub_queries:
                    sub_hits = self._search_single(
                        sub_query,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        rrf_k=rrf_k,
                    )
                    all_hits.append(sub_hits)

                # Fuse all sub-query results via RRF
                if all_hits:
                    fused = rrf_fuse(*all_hits, k=rrf_k, top_k=top_k)
                    from .recursive import merge_hits
                    merged = merge_hits(fused, self._storage)
                    return self._maybe_pack(merged, context_budget)
                return []

        single = self._search_single(
            query, top_k=top_k, candidate_k=candidate_k, rrf_k=rrf_k,
        )
        return self._maybe_pack(single, context_budget)

    def _maybe_pack(
        self,
        hits: list[SearchHit],
        context_budget: int | None,
    ) -> list[SearchHit]:
        """Apply token-budget packing if a budget was supplied.

        Packing happens AFTER recursive merge so the budget applies
        to the final (possibly merged + expanded) hit list, not to
        raw RRF output. Side-effect-free pure pass-through when
        ``context_budget`` is None or non-positive.
        """
        if context_budget is None or context_budget <= 0:
            return hits
        from .packing import pack_to_budget

        result = pack_to_budget(hits, context_budget)
        log.info(
            "search.packed",
            budget=result.budget_tokens,
            total_tokens=result.total_tokens,
            packed=len(result.packed),
            spilled=len(result.spilled),
            truncated=result.truncated,
        )
        return result.packed

    def _search_single(
        self,
        query: str,
        top_k: int = 10,
        *,
        candidate_k: int = CANDIDATE_K,
        rrf_k: int = DEFAULT_K,
    ) -> list[SearchHit]:
        """Single-query hybrid search (internal)."""
        # HyDE — only the dense embedding sees the hypothetical doc; BM25
        # and the symbol-token lane keep the literal query. Falls back
        # to the raw query if disabled or the LLM call fails.
        from .hyde import maybe_hyde_rewrite

        dense_query = maybe_hyde_rewrite(query) or query
        if dense_query is not query:
            log.info("search.hyde_rewrite", original_len=len(query),
                     rewritten_len=len(dense_query))
        query_vec = self._embedder.embed_query(dense_query)
        dense_hits = self._storage.search_dense(query_vec, k=candidate_k)
        sparse_hits = self._storage.search_bm25(query, k=candidate_k)
        symbol_hits = self._storage.search_symbol_tokens(query, k=20)

        # WIN-160: keep the source stream label on every hit so the trace
        # recorder can attribute each result to dense / sparse / symbol.
        # The single-stream short-circuit synthesises a streams field too
        # so downstream consumers never see a None when fusion was skipped.
        named_streams = {
            name: hits
            for name, hits in (
                ("dense", dense_hits),
                ("sparse", sparse_hits),
                ("symbol", symbol_hits),
            )
            if hits
        }
        if not named_streams:
            return []
        if len(named_streams) == 1:
            (only_name, only_hits), = named_streams.items()
            fused = [
                h.model_copy(update={"streams": {only_name: h.score}})
                for h in only_hits[:top_k]
            ]
        else:
            # rrf_fuse_named signature is Mapping[str, Iterable[SearchHit]];
            # cast for mypy which sees the value list[...] as invariant.
            from collections.abc import Iterable
            from typing import cast
            fused = rrf_fuse_named(
                cast("dict[str, Iterable[SearchHit]]", named_streams),
                k=rrf_k,
                top_k=top_k,
            )
        # Recursive retrieval / auto-merge (Phase 3 §6.4). No-op when
        # STROPHA_RECURSIVE_RETRIEVAL is unset / 0 — back-compat preserved.
        from .recursive import merge_hits
        return merge_hits(fused, self._storage)

    def search_dense_only(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Escape hatch for debugging or A/B comparison."""
        query_vec = self._embedder.embed_query(query)
        return self._storage.search_dense(query_vec, k=top_k)

    def search_sparse_only(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self._storage.search_bm25(query, k=top_k)
