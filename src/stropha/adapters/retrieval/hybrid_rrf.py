"""Hybrid retrieval (dense + BM25 + symbol-token) fused via RRF.

Phase 4 refactors this into a sub-pipeline whose streams are themselves
registered adapters under stage ``retrieval-stream``. The legacy default
(3 streams: ``vec-cosine`` + ``fts5-bm25`` + ``like-tokens``) is preserved
when no ``streams`` block is supplied in YAML, so behavior is unchanged
for existing setups.

Optional reranker support: after RRF fusion, a cross-encoder can rescore
the top candidates for improved precision.

YAML example with overrides::

    retrieval:
      adapter: hybrid-rrf
      config:
        top_k: 10
        rrf_k: 60
        candidate_k: 50  # candidates passed to reranker
        streams:
          dense:  { adapter: vec-cosine,  config: { k: 100 } }
          sparse: { adapter: fts5-bm25,   config: { k: 100 } }
          symbol: { adapter: like-tokens, config: { k: 30  } }
        reranker:
          adapter: cross-encoder
          config:
            model: BAAI/bge-reranker-base

A user can also drop a stream by setting it to ``null`` (e.g. to disable
the symbol stream while keeping dense + sparse).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...errors import ConfigError
from ...meta.stream_weights import StreamWeightProfiles
from ...models import SearchHit
from ...pipeline.base import StageHealth
from ...pipeline.registry import lookup_adapter, register_adapter
from ...retrieval.rrf import DEFAULT_K, rrf_fuse, rrf_fuse_named
from ...stages.embedder import EmbedderStage
from ...stages.reranker import RerankerStage
from ...stages.retrieval import RetrievalStage
from ...stages.retrieval_stream import RetrievalStreamStage
from ...stages.storage import StorageStage

# Legacy single-stream defaults — preserves Phase 1/2 behavior.
_DEFAULT_STREAMS: dict[str, dict[str, Any] | None] = {
    "dense": {"adapter": "vec-cosine", "config": {"k": 50}},
    "sparse": {"adapter": "fts5-bm25", "config": {"k": 50}},
    "symbol": {"adapter": "like-tokens", "config": {"k": 20}},
}


class HybridRrfConfig(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    candidate_k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of RRF candidates to pass to reranker.",
    )
    rrf_k: int = Field(
        default=DEFAULT_K,
        ge=1,
        le=1000,
        description="RRF smoothing constant (paper recommends 60).",
    )
    adaptive_weighting: bool = Field(
        default=True,
        description=(
            "Enable adaptive query weighting. Detects symbol-like queries "
            "(CamelCase, snake_case, qualified names) and adjusts stream weights "
            "accordingly. Symbol queries boost lexical/symbol streams; natural "
            "language queries boost dense semantic matching."
        ),
    )
    learned_weights_enabled: bool = Field(
        default=True,
        description=(
            "Apply per-repository stream multipliers learned by outcome backfill. "
            "Missing or invalid profiles preserve the query-adaptive weights."
        ),
    )
    learned_weights_dir: str | None = Field(
        default=None,
        description=(
            "Override the learned profile directory. Defaults to "
            "<index_path_parent>/meta/stream-weights."
        ),
    )
    hyde_enabled: bool = Field(
        default=False,
        description=(
            "Enable HyDE (Hypothetical Document Embeddings). "
            "Generates a hypothetical code snippet via LLM before dense embedding. "
            "Requires Ollama running locally."
        ),
    )
    query_rewrite_enabled: bool = Field(
        default=False,
        description=(
            "Selectively rewrite short natural-language queries into code terms. "
            "Exact and structural queries bypass generation; rewrites affect all streams."
        ),
    )
    query_rewrite_max_terms: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Maximum terms eligible for selective query rewriting. "
            "Null inherits unified config when global features are enabled."
        ),
    )
    multi_query_enabled: bool = Field(
        default=False,
        description=(
            "Enable multi-query expansion. Generates 3-5 paraphrases of the query, "
            "searches with each, and fuses results via RRF. Increases recall for "
            "conceptual queries. Requires Ollama."
        ),
    )
    multi_query_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of paraphrases to generate for multi-query expansion.",
    )
    multi_query_min_terms: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description=(
            "Minimum terms eligible for broad-concept multi-query expansion. "
            "Null inherits unified config when global features are enabled."
        ),
    )
    recursive_enabled: bool = Field(
        default=False,
        description=(
            "Enable recursive retrieval / auto-merging. When multiple sibling "
            "chunks from the same parent match, promotes to the parent chunk. "
            "Also merges adjacent chunks on the same file."
        ),
    )
    recursive_adjacency: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum line gap to consider chunks adjacent for merging.",
    )
    streams: dict[str, dict[str, Any] | None] = Field(
        default_factory=dict,
        description=(
            "Sub-pipeline: each named stream → {adapter, config}. "
            "Set a value to null to disable. Keys omitted inherit defaults."
        ),
    )
    reranker: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional reranker: {adapter, config}. "
            "If null or omitted, RRF output is returned directly."
        ),
    )
    inherit_global_features: bool = Field(
        default=True,
        description=(
            "Inherit unified retrieval feature toggles. Meta-Harness candidates "
            "disable this so the live process config cannot override an arm."
        ),
    )
    hyde_code_tuned: bool | None = Field(
        default=None,
        description="Explicit HyDE prompt style; null inherits unified config.",
    )


@register_adapter(stage="retrieval", name="hybrid-rrf")
class HybridRrfRetrieval(RetrievalStage):
    """RRF over N retrieval streams, with optional cross-encoder reranking."""

    Config = HybridRrfConfig

    def __init__(
        self,
        config: HybridRrfConfig | None = None,
        *,
        storage: StorageStage | None = None,
        embedder: EmbedderStage | None = None,
        secondary_embedder: EmbedderStage | None = None,
    ) -> None:
        if config is None:
            config = HybridRrfConfig()
        if storage is None or embedder is None:
            raise ValueError(
                "HybridRrfRetrieval requires storage= and embedder= (built by "
                "the pipeline builder; do not instantiate manually)."
            )
        self._config = config
        self._storage = storage
        self._embedder = embedder
        self._secondary_embedder = secondary_embedder
        self._streams = self._build_streams(storage, secondary_embedder)
        self._reranker = self._build_reranker()
        self._learned_weights = self._build_learned_weights()
        self._rewrite_latencies_ms: list[float] = []
        self._rewrite_counts: dict[str, int] = {}
        self._rewrite_input_tokens = 0
        self._rewrite_output_tokens = 0
        self._multi_query_latencies_ms: list[float] = []
        self._multi_query_counts: dict[str, int] = {}
        self._multi_query_input_tokens = 0
        self._multi_query_output_tokens = 0
        self._multi_query_requested = 0
        self._multi_query_generated = 0
        self._multi_query_raw = 0
        self._multi_query_duplicate_paraphrases = 0
        self._multi_query_searches = 0
        self._multi_query_candidates_before = 0
        self._multi_query_unique_before = 0
        self._multi_query_candidates_after = 0
        self._multi_query_unique_after = 0
        self.reset_diagnostics()

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "retrieval"

    @property
    def adapter_name(self) -> str:
        return "hybrid-rrf"

    @property
    def adapter_id(self) -> str:
        canonical = sorted((name, s.adapter_id) for name, s in self._streams.items())
        reranker_id = self._reranker.adapter_id if self._reranker else "none"
        learned_id = (
            self._learned_weights.fingerprint() if self._learned_weights is not None else "off"
        )
        digest = hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:8]
        multi_query_enabled, multi_query_count, multi_query_min_terms = (
            self._multi_query_settings()
        )
        if multi_query_enabled:
            from ...inference import get_backend
            from ...retrieval.multi_query import policy_id as multi_query_policy_id

            backend_hash = hashlib.sha256(get_backend().identity.encode("utf-8")).hexdigest()[:8]
            multi_query_id = (
                f"{multi_query_policy_id(multi_query_min_terms)}:"
                f"count={multi_query_count}:llm={backend_hash}"
            )
        else:
            multi_query_id = "off"
        rewrite_enabled, rewrite_max_terms = self._query_rewrite_settings()
        if rewrite_enabled:
            from ...inference import get_backend
            from ...retrieval.query_rewrite import policy_id

            backend_hash = hashlib.sha256(get_backend().identity.encode("utf-8")).hexdigest()[:8]
            rewrite_id = f"{policy_id(rewrite_max_terms)}:llm={backend_hash}"
        else:
            rewrite_id = "off"
        return (
            f"hybrid-rrf:k={self._config.rrf_k}:streams={digest}:"
            f"reranker={reranker_id}:adaptive={int(self._config.adaptive_weighting)}:"
            f"mq={multi_query_id}:"
            f"rewrite={rewrite_id}:"
            f"hyde={int(self._config.hyde_enabled)}:{self._config.hyde_code_tuned}:"
            f"recursive={int(self._config.recursive_enabled)}:"
            f"adjacency={self._config.recursive_adjacency}:"
            f"inherit={int(self._config.inherit_global_features)}:"
            f"learned={learned_id}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return HybridRrfConfig

    def health(self) -> StageHealth:
        if not self._streams:
            return StageHealth(
                status="warning",
                message="hybrid-rrf has no enabled streams",
            )
        detail = {name: s.adapter_id for name, s in self._streams.items()}
        if self._reranker:
            detail["reranker"] = self._reranker.adapter_id
        reranker_msg = " + reranker" if self._reranker else ""
        return StageHealth(
            status="ready",
            message=f"hybrid-rrf with {len(self._streams)} streams{reranker_msg}",
            detail=detail,
        )

    def reset_diagnostics(self) -> None:
        self._rewrite_counts = {
            reason: 0
            for reason in (
                "success",
                "empty_query",
                "symbol_query",
                "structural_query",
                "not_natural",
                "too_long",
                "empty_response",
                "backend_error",
            )
        }
        self._rewrite_latencies_ms = []
        self._rewrite_input_tokens = 0
        self._rewrite_output_tokens = 0
        self._multi_query_counts = {
            reason: 0
            for reason in (
                "success",
                "cache_hit",
                "empty_query",
                "symbol_query",
                "structural_query",
                "not_natural",
                "too_short",
                "not_conceptual",
                "empty_response",
                "no_valid_paraphrases",
                "backend_error",
            )
        }
        self._multi_query_latencies_ms = []
        self._multi_query_input_tokens = 0
        self._multi_query_output_tokens = 0
        self._multi_query_requested = 0
        self._multi_query_generated = 0
        self._multi_query_raw = 0
        self._multi_query_duplicate_paraphrases = 0
        self._multi_query_searches = 0
        self._multi_query_candidates_before = 0
        self._multi_query_unique_before = 0
        self._multi_query_candidates_after = 0
        self._multi_query_unique_after = 0
        reset = getattr(self._reranker, "reset_diagnostics", None)
        if callable(reset):
            reset()

    def diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        rewrite_enabled, rewrite_max_terms = self._query_rewrite_settings()
        from ...retrieval.query_rewrite import policy_id

        generation_calls = sum(
            self._rewrite_counts[reason]
            for reason in ("success", "empty_response", "backend_error")
        )
        fallbacks = self._rewrite_counts["empty_response"] + self._rewrite_counts["backend_error"]
        out["query_rewrite"] = {
            "enabled": rewrite_enabled,
            "policy_id": policy_id(rewrite_max_terms),
            "max_terms": rewrite_max_terms,
            "generation_calls": generation_calls,
            "successes": self._rewrite_counts["success"],
            "fallbacks": fallbacks,
            "fallback_rate": round(fallbacks / generation_calls, 4) if generation_calls else 0.0,
            "estimated_input_tokens": self._rewrite_input_tokens,
            "estimated_output_tokens": self._rewrite_output_tokens,
            "estimated_total_tokens": self._rewrite_input_tokens + self._rewrite_output_tokens,
            "latency_ms": round(sum(self._rewrite_latencies_ms), 3),
            "latency_p95_ms": self._percentile(self._rewrite_latencies_ms, 0.95),
            "skips": {
                reason: self._rewrite_counts[reason]
                for reason in (
                    "empty_query",
                    "symbol_query",
                    "structural_query",
                    "not_natural",
                    "too_long",
                )
            },
        }
        multi_query_enabled, multi_query_count, multi_query_min_terms = (
            self._multi_query_settings()
        )
        from ...retrieval.multi_query import policy_id as multi_query_policy_id

        multi_query_calls = sum(
            self._multi_query_counts[reason]
            for reason in (
                "success",
                "empty_response",
                "no_valid_paraphrases",
                "backend_error",
            )
        )
        multi_query_fallbacks = sum(
            self._multi_query_counts[reason]
            for reason in ("empty_response", "no_valid_paraphrases", "backend_error")
        )
        duplicate_candidates_before = (
            self._multi_query_candidates_before - self._multi_query_unique_before
        )
        duplicate_candidates_after = (
            self._multi_query_candidates_after - self._multi_query_unique_after
        )
        out["multi_query"] = {
            "enabled": multi_query_enabled,
            "policy_id": multi_query_policy_id(multi_query_min_terms),
            "count": multi_query_count,
            "min_terms": multi_query_min_terms,
            "generation_calls": multi_query_calls,
            "successes": self._multi_query_counts["success"],
            "cache_hits": self._multi_query_counts["cache_hit"],
            "fallbacks": multi_query_fallbacks,
            "fallback_rate": (
                round(multi_query_fallbacks / multi_query_calls, 4)
                if multi_query_calls
                else 0.0
            ),
            "requested_paraphrases": self._multi_query_requested,
            "generated_paraphrases": self._multi_query_generated,
            "raw_paraphrases": self._multi_query_raw,
            "duplicate_paraphrases": self._multi_query_duplicate_paraphrases,
            "searches": self._multi_query_searches,
            "estimated_input_tokens": self._multi_query_input_tokens,
            "estimated_output_tokens": self._multi_query_output_tokens,
            "estimated_total_tokens": (
                self._multi_query_input_tokens + self._multi_query_output_tokens
            ),
            "latency_ms": round(sum(self._multi_query_latencies_ms), 3),
            "latency_p95_ms": self._percentile(self._multi_query_latencies_ms, 0.95),
            "candidate_duplicates_before_fusion": duplicate_candidates_before,
            "candidate_duplicate_rate_before_fusion": (
                round(duplicate_candidates_before / self._multi_query_candidates_before, 4)
                if self._multi_query_candidates_before
                else 0.0
            ),
            "candidate_duplicates_after_fusion": duplicate_candidates_after,
            "candidate_duplicate_rate_after_fusion": (
                round(duplicate_candidates_after / self._multi_query_candidates_after, 4)
                if self._multi_query_candidates_after
                else 0.0
            ),
            "skips": {
                reason: self._multi_query_counts[reason]
                for reason in (
                    "empty_query",
                    "symbol_query",
                    "structural_query",
                    "not_natural",
                    "too_short",
                    "not_conceptual",
                )
            },
        }
        get_reranker = getattr(self._reranker, "diagnostics", None)
        if callable(get_reranker):
            out["reranker"] = get_reranker()
        if self._learned_weights is not None:
            out["learned_weights"] = self._learned_weights.report(active=True)
        return out

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[index], 3)

    def _query_rewrite_settings(self) -> tuple[bool, int]:
        from ...retrieval.query_rewrite import DEFAULT_MAX_TERMS

        enabled = self._config.query_rewrite_enabled
        max_terms = self._config.query_rewrite_max_terms
        if self._config.inherit_global_features:
            from ...config import get_config

            features = get_config().retrieval_features
            enabled = enabled or features.query_rewrite_enabled
            if max_terms is None:
                max_terms = features.query_rewrite_max_terms
        effective_max_terms = max_terms or DEFAULT_MAX_TERMS
        return enabled, max(1, min(20, effective_max_terms))

    def _record_query_rewrite(self, result: Any) -> None:
        self._rewrite_counts[result.reason] += 1
        self._rewrite_input_tokens += result.estimated_input_tokens
        self._rewrite_output_tokens += result.estimated_output_tokens
        if result.generation_calls:
            self._rewrite_latencies_ms.append(result.latency_ms)

    def _multi_query_settings(self) -> tuple[bool, int, int]:
        from ...retrieval.multi_query import DEFAULT_MIN_TERMS

        local_enabled = self._config.multi_query_enabled
        enabled = local_enabled
        count = self._config.multi_query_count
        min_terms = self._config.multi_query_min_terms
        if self._config.inherit_global_features:
            from ...config import get_config

            features = get_config().retrieval_features
            enabled = enabled or features.multi_query_enabled
            if not local_enabled and features.multi_query_enabled:
                count = features.multi_query_count
            if min_terms is None:
                min_terms = features.multi_query_min_terms
        return (
            enabled,
            max(1, min(5, count)),
            max(1, min(30, min_terms or DEFAULT_MIN_TERMS)),
        )

    def _record_multi_query(self, result: Any) -> None:
        self._multi_query_counts[result.reason] += 1
        self._multi_query_requested += result.requested_paraphrases
        self._multi_query_generated += max(0, len(result.queries) - 1)
        self._multi_query_raw += result.raw_paraphrases
        self._multi_query_duplicate_paraphrases += result.duplicate_paraphrases
        self._multi_query_input_tokens += result.estimated_input_tokens
        self._multi_query_output_tokens += result.estimated_output_tokens
        if result.generation_calls:
            self._multi_query_latencies_ms.append(result.latency_ms)

    def _record_multi_query_candidates(
        self,
        result_lists: list[list[SearchHit]],
        fused: list[SearchHit],
    ) -> None:
        before_ids = [hit.chunk_id for hits in result_lists for hit in hits]
        after_ids = [hit.chunk_id for hit in fused]
        self._multi_query_candidates_before += len(before_ids)
        self._multi_query_unique_before += len(set(before_ids))
        self._multi_query_candidates_after += len(after_ids)
        self._multi_query_unique_after += len(set(after_ids))

    def set_reranker(self, reranker: RerankerStage | None) -> None:
        """Inject a trusted reranker implementation after pipeline construction."""
        if reranker is not None and not isinstance(reranker, RerankerStage):
            raise TypeError("reranker must implement RerankerStage")
        self._reranker = reranker

    # ----- search ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        conversation_context: str | None = None,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        if not query.strip() or not self._streams:
            return []
        k = top_k or self._config.top_k

        multi_query_enabled, multi_query_count, multi_query_min_terms = (
            self._multi_query_settings()
        )

        if multi_query_enabled:
            return self._search_multi_query(
                query,
                k,
                count=multi_query_count,
                min_terms=multi_query_min_terms,
                conversation_context=conversation_context,
                repo_id=repo_id,
            )
        else:
            return self._search_single_query(
                query,
                k,
                conversation_context=conversation_context,
                repo_id=repo_id,
            )

    def _search_single_query(
        self,
        query: str,
        k: int,
        *,
        search_query: str | None = None,
        conversation_context: str | None = None,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        """Execute search with a single query (the standard path)."""
        # Query rewriting: expand natural language to code terms (affects ALL streams)
        if search_query is None:
            search_query = query
            rewrite_enabled, rewrite_max_terms = self._query_rewrite_settings()
            if rewrite_enabled:
                from ...retrieval.query_rewrite import rewrite_query

                result = rewrite_query(
                    query,
                    force_enabled=True,
                    max_terms=rewrite_max_terms,
                )
                self._record_query_rewrite(result)
                if result.query:
                    search_query = result.query

        # Single embed for the dense streams (avoids paying twice if the
        # user enables multiple dense backends in the future).
        needs_vec = any(s.adapter_name in ("vec-cosine",) for s in self._streams.values())

        # HyDE: generate hypothetical document for dense embedding only
        # Enabled via config.hyde_enabled OR retrieval_features.hyde_enabled in YAML
        dense_query = search_query
        if needs_vec:
            hyde_enabled = self._config.hyde_enabled
            if self._config.inherit_global_features:
                from ...config import get_config

                hyde_enabled = hyde_enabled or get_config().retrieval_features.hyde_enabled
            if hyde_enabled:
                from ...retrieval.hyde import maybe_hyde_rewrite

                rewritten = maybe_hyde_rewrite(
                    query,
                    force_enabled=True,
                    code_tuned=self._config.hyde_code_tuned,
                )  # Use original query for HyDE
                if rewritten:
                    dense_query = rewritten

            if conversation_context and conversation_context.strip():
                dense_query = (
                    f"{dense_query}\n\nRecent conversation: {conversation_context.strip()}"
                )

        query_vec = self._embedder.embed_query(dense_query) if needs_vec else None

        # Collect results from each stream with names for weighted fusion
        ranked_by_name: dict[str, list[SearchHit]] = {}
        for name, stream in self._streams.items():
            # Use search_query for sparse/symbol streams (may be rewritten)
            # query_vec already uses dense_query (may be HyDE-rewritten)
            if repo_id is None:
                hits = stream.search(search_query, query_vec)
            else:
                hits = stream.search(search_query, query_vec, repo_id=repo_id)
            if hits:
                ranked_by_name[name] = hits
        if not ranked_by_name:
            return []

        # RRF fusion with adaptive weighting
        if len(ranked_by_name) == 1:
            only_name, only_hits = next(iter(ranked_by_name.items()))
            candidate_limit = self._config.candidate_k if self._reranker else k
            fused = [
                hit.model_copy(update={"streams": {only_name: hit.score}})
                for hit in only_hits[:candidate_limit]
            ]
        else:
            # Get more candidates if we have a reranker
            fusion_k = self._config.candidate_k if self._reranker else k

            # Adaptive weighting based on query type
            weights = self._get_stream_weights(query)

            fused = rrf_fuse_named(
                ranked_by_name,
                k=self._config.rrf_k,
                top_k=fusion_k,
                weights=weights,
                repo_weights=(
                    self._learned_weights.weights() if self._learned_weights is not None else None
                ),
            )

        # Optional reranking
        if self._reranker and fused:
            fused = self._reranker.rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]

        # Recursive retrieval / auto-merging
        return self._maybe_recursive_merge(fused)

    def _get_stream_weights(self, query: str) -> dict[str, float] | None:
        """Get adaptive stream weights based on query classification.

        Returns None if adaptive weighting is disabled, which means
        equal weights for all streams.
        """
        if not self._config.adaptive_weighting:
            return None

        from ...retrieval.query_classify import get_adaptive_weights

        weights = get_adaptive_weights(query)

        # Map stream names to weight values
        # The stream names in hybrid-rrf are: dense, sparse, symbol, graph_vec
        stream_weights = {
            "dense": weights.dense,
            "sparse": weights.sparse,
            "symbol": weights.symbol,
            "graph_vec": weights.graph_vec,
            "model2vec": weights.model2vec,
        }
        for name, stream in self._streams.items():
            if stream.adapter_name == "model2vec-cosine":
                stream_weights[name] = weights.model2vec
        return stream_weights

    def _search_multi_query(
        self,
        query: str,
        k: int,
        *,
        count: int,
        min_terms: int,
        conversation_context: str | None = None,
        repo_id: int | None = None,
    ) -> list[SearchHit]:
        """Execute search with multiple paraphrases, fusing results via RRF.

        Multi-query expansion generates N paraphrases of the query,
        searches with each, and fuses all results via RRF for better recall.
        """
        from ...retrieval.multi_query import expand_query

        expansion = expand_query(
            query,
            force_enabled=True,
            num_paraphrases=count,
            min_terms=min_terms,
        )
        self._record_multi_query(expansion)
        paraphrases = expansion.queries

        if len(paraphrases) <= 1:
            # Fallback to single query if paraphrase generation failed
            return self._search_single_query(
                query,
                k,
                conversation_context=conversation_context,
                repo_id=repo_id,
            )

        # Search with each paraphrase
        all_results: list[list[SearchHit]] = []
        for paraphrase in paraphrases:
            # Use larger candidate set since we'll fuse multiple result sets
            cand_k = max(k * 2, self._config.candidate_k)
            hits = self._search_single_query(
                query,  # Original query for HyDE/reranking context
                cand_k,
                search_query=paraphrase,  # Paraphrase for actual search
                conversation_context=conversation_context,
                repo_id=repo_id,
            )
            if hits:
                all_results.append(hits)
        self._multi_query_searches += len(paraphrases)

        if not all_results:
            return []

        if len(all_results) == 1:
            fused = all_results[0]
        else:
            # Fuse results from all paraphrases via RRF
            # Use more candidates since we're combining multiple searches
            fusion_k = self._config.candidate_k if self._reranker else k
            fused = rrf_fuse(*all_results, k=self._config.rrf_k, top_k=fusion_k)
        self._record_multi_query_candidates(all_results, fused)

        # Optional reranking on the fused results
        if self._reranker and fused:
            fused = self._reranker.rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]

        # Recursive retrieval / auto-merging
        return self._maybe_recursive_merge(fused)

    def _maybe_recursive_merge(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Apply recursive retrieval / auto-merging if enabled."""
        if not self._config.recursive_enabled or not hits:
            return hits

        from ...retrieval.recursive import merge_hits

        return merge_hits(
            hits,
            self._storage,  # type: ignore[arg-type]
            adjacency_lines=self._config.recursive_adjacency,
            enabled=True,  # Already checked above
        )

    # ----- helpers --------------------------------------------------------

    def _build_learned_weights(self) -> StreamWeightProfiles | None:
        if not self._config.learned_weights_enabled:
            return None
        from ...config import get_config

        configured = self._config.learned_weights_dir
        if configured:
            directory = Path(configured).expanduser().resolve()
        else:
            index_path = Path(get_config().resolve_index_path()).expanduser().resolve()
            directory = index_path.parent / "meta" / "stream-weights"
        return StreamWeightProfiles(directory)

    def _build_streams(
        self,
        storage: StorageStage,
        secondary_embedder: EmbedderStage | None,
    ) -> dict[str, RetrievalStreamStage]:
        merged: dict[str, dict[str, Any] | None] = {**_DEFAULT_STREAMS, **self._config.streams}
        out: dict[str, RetrievalStreamStage] = {}
        for name, spec in merged.items():
            if spec is None:
                continue  # explicit disable
            adapter_name = spec.get("adapter")
            if not adapter_name:
                raise ConfigError(f"retrieval.streams.{name}: missing `adapter` key")
            cls = lookup_adapter("retrieval-stream", adapter_name)
            try:
                cfg_obj = cls.Config(**(spec.get("config") or {}))
            except Exception as exc:
                raise ConfigError(
                    f"Invalid config for retrieval.streams.{name} (adapter={adapter_name!r}): {exc}"
                ) from exc
            if adapter_name == "model2vec-cosine":
                if secondary_embedder is None:
                    raise ConfigError(
                        "model2vec-cosine stream requires the pipeline builder's secondary embedder"
                    )
                out[name] = cls(
                    cfg_obj,
                    storage=storage,
                    secondary_embedder=secondary_embedder,
                )
            else:
                out[name] = cls(cfg_obj, storage=storage)
        return out

    def _build_reranker(self) -> RerankerStage | None:
        """Build optional reranker from config."""
        spec = self._config.reranker
        if spec is None:
            return None

        adapter_name = spec.get("adapter")
        if not adapter_name:
            raise ConfigError("retrieval.reranker: missing `adapter` key")

        cls = lookup_adapter("reranker", adapter_name)
        try:
            cfg_obj = cls.Config(**(spec.get("config") or {}))
        except Exception as exc:
            raise ConfigError(
                f"Invalid config for retrieval.reranker (adapter={adapter_name!r}): {exc}"
            ) from exc
        return cls(cfg_obj)
