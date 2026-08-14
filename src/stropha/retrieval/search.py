"""Hybrid retrieval: dense + sparse + symbol streams fused via RRF.

The configurable retrieval adapter can optionally rerank the fused candidates.
"""

from __future__ import annotations

from ..config import get_config
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
        compress: bool = False,
        conversation_context: str | None = None,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        """Three-stream hybrid retrieval, fused with RRF.

        Streams:
        - dense  — vector search from the selected storage backend.
        - sparse — full-text search over content + path + symbol.
        - symbol — direct match on the `symbol` column for identifier tokens
                   in the query (spec §6.3.5 query routing).
        Empty streams are dropped before fusion.

        Query decomposition (optional):
        When enabled explicitly or by ``retrieval_features.query_decomposition_enabled``
        in ``~/.stropha/config.yaml``,
        compound queries like "how does X work and what calls Y" are split
        into atomic sub-queries, searched separately, and fused via RRF.

        Token-budget packing (WIN-208):
        When ``context_budget`` is set, the score-ranked output is greedy-
        packed up to that result-token budget (using ``estimate_content_tokens``)
        and the truncated tail is dropped. The returned list never exceeds
        the budget — except in the single-first-hit-oversize escape path,
        where the top hit is returned alone even if oversize so the caller
        sees something rather than nothing. ``top_k`` still bounds the
        upper end of what's considered for packing; the packed result may
        contain fewer items. This budgets returned snippets, not the caller's
        entire model context window.

        Extractive compression (WIN-210):
        When ``compress=True``, each hit's ``snippet`` is replaced with
        an extractive compression that keeps the signature, matched
        lines, a small context window, and drops boilerplate. Applied
        BEFORE budget packing so the budget reflects the compressed
        cost — caller gets more relevant code per token spent.

        Conversation-aware retrieval (WIN-224):
        When ``conversation_context`` is non-empty, the dense lane's
        query embedding is computed against a concatenation of the
        current query + the context string (typically the last N
        agent queries from the session). BM25 + symbol lanes keep the
        literal query so identifier matches stay intact. Lets queries
        like "tests for it" carry the implicit "it" from earlier
        turns into the dense semantic match.
        """
        if not query.strip():
            return []

        # Check for query decomposition
        if decompose is None:
            decompose = get_config().retrieval_features.query_decomposition_enabled

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
                        repo_id=repo_id,
                    )
                    all_hits.append(sub_hits)

                # Fuse all sub-query results via RRF
                if all_hits:
                    fused = rrf_fuse(*all_hits, k=rrf_k, top_k=top_k)
                    from .recursive import merge_hits
                    merged = merge_hits(fused, self._storage)
                    compressed = self._maybe_compress(merged, query, compress)
                    return self._maybe_pack(compressed, context_budget)
                return []

        single = self._search_single(
            query, top_k=top_k, candidate_k=candidate_k, rrf_k=rrf_k,
            conversation_context=conversation_context,
            repo_id=repo_id,
        )
        compressed = self._maybe_compress(single, query, compress)
        return self._maybe_pack(compressed, context_budget)

    def _maybe_compress(
        self,
        hits: list[SearchHit],
        query: str,
        compress: bool,
    ) -> list[SearchHit]:
        """Apply extractive compression when requested.

        Compression replaces ``hit.snippet`` with a shrunken version
        keeping signature + matched lines + small context window.
        Applied BEFORE budget packing so the budget reflects the
        compressed cost; compressing AFTER packing would defeat the
        purpose (would have already paid the token cost of dropped
        spans).
        """
        if not compress or not hits:
            return hits
        from .compression import compress_hits

        return compress_hits(hits, query)

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
        conversation_context: str | None = None,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        """Single-query hybrid search (internal)."""
        # HyDE — only the dense embedding sees the hypothetical doc; BM25
        # and the symbol-token lane keep the literal query. Falls back
        # to the raw query if disabled or the LLM call fails.
        from .hyde import generate_hypothetical

        dense_query = generate_hypothetical(query).text or query
        if dense_query is not query:
            log.info("search.hyde_rewrite", original_len=len(query),
                     rewritten_len=len(dense_query))
        # WIN-224: blend recent-conversation context into the dense
        # embedding ONLY — BM25 / symbol lanes keep the literal query
        # so identifier matches stay clean. The context is appended
        # after a "Recent:" separator so the embedder treats it as
        # related-but-secondary signal.
        if conversation_context and conversation_context.strip():
            dense_query = (
                f"{dense_query}\n\nRecent conversation: {conversation_context.strip()}"
            )
            log.info(
                "search.conversation_aware",
                context_chars=len(conversation_context),
            )
        query_vec = self._embedder.embed_query(dense_query)
        if repo_id is None:
            dense_hits = self._storage.search_dense(query_vec, k=candidate_k)
            sparse_hits = self._storage.search_bm25(query, k=candidate_k)
            symbol_hits = self._storage.search_symbol_tokens(query, k=20)
        else:
            dense_hits = self._storage.search_dense(
                query_vec, k=candidate_k, repo_id=repo_id
            )
            sparse_hits = self._storage.search_bm25(
                query, k=candidate_k, repo_id=repo_id
            )
            symbol_hits = self._storage.search_symbol_tokens(query, k=20, repo_id=repo_id)

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
        # Recursive retrieval / auto-merge. Configuration is sourced from
        # retrieval_features in ~/.stropha/config.yaml.
        from .recursive import merge_hits
        return merge_hits(fused, self._storage)

    def search_dense_only(
        self,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        """Escape hatch for debugging or A/B comparison."""
        query_vec = self._embedder.embed_query(query)
        if repo_id is None:
            return self._storage.search_dense(query_vec, k=top_k)
        return self._storage.search_dense(query_vec, k=top_k, repo_id=repo_id)

    def search_sparse_only(
        self,
        query: str,
        top_k: int = 10,
        *,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        if repo_id is None:
            return self._storage.search_bm25(query, k=top_k)
        return self._storage.search_bm25(query, k=top_k, repo_id=repo_id)
