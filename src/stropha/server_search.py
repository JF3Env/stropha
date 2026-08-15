"""Search helpers extracted from ``stropha.server``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .server_models import FacetCounts, SearchResponse, SearchResult
from .server_runtime import AppContext

ToResult = Callable[[Any], SearchResult]
PersistSessionState = Callable[[AppContext], None]
ApplyFilters = Callable[..., list[Any]]
TrackSearchSavings = Callable[[str, list[Any]], Any]
ResolveTargetRepoId = Callable[..., int | None]


def _hit_belongs_to_repo(storage: Any, hit: Any, repo_id: int) -> bool:
    chunk = storage.get_chunk_by_id(hit.chunk_id)
    return chunk is not None and chunk.get("repo_id") == repo_id


def apply_filters(
    hits: list[Any],
    *,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    kind: list[str] | None = None,
    exclude_tests: bool = False,
) -> list[Any]:
    """Apply post-retrieval filters to search hits."""
    result = hits

    if language:
        lang_set = {lang.lower() for lang in language}
        result = [hit for hit in result if hit.language.lower() in lang_set]

    if path_prefix:
        result = [hit for hit in result if hit.rel_path.startswith(path_prefix)]

    if kind:
        kind_set = {value.lower() for value in kind}
        result = [hit for hit in result if hit.kind.lower() in kind_set]

    if exclude_tests:
        test_patterns = ("test_", "_test.", ".test.", "/tests/", "/test/", "Test.")
        result = [
            hit for hit in result if not any(pattern in hit.rel_path for pattern in test_patterns)
        ]

    return result


def search_code_impl(
    query: str,
    top_k: int,
    language: list[str] | None,
    path_prefix: str | None,
    kind: list[str] | None,
    exclude_tests: bool,
    include_facets: bool,
    recursive: bool,
    repo_url: str | None,
    include_other_repos: bool,
    refresh: bool,
    context_budget: int | None,
    compress: bool,
    card_level: str,
    relevance_decay: bool,
    dedup: bool,
    conversation_aware: bool,
    app: AppContext,
    *,
    to_result: ToResult,
    persist_session_state: PersistSessionState,
    track_search_savings_fn: TrackSearchSavings,
    is_git_url_fn: Callable[[str], bool],
    search_git_url_fn: Callable[..., Any],
    apply_filters_fn: ApplyFilters,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
    stropha_error_cls: type[Exception],
    log: Any,
) -> SearchResponse:
    top_k = max(1, min(top_k, 30))

    has_filters = any([language, path_prefix, kind, exclude_tests])
    fetch_k = top_k * 3 if has_filters else top_k

    conversation_context: str | None = None
    if conversation_aware and app.recent_queries:
        conversation_context = " | ".join(app.recent_queries)

    repo_id: int | None = None
    try:
        if repo_url and is_git_url_fn(repo_url):
            log.info("mcp.search_code.remote", repo_url=repo_url, refresh=refresh)
            remote = search_git_url_fn(query, repo_url, top_k=fetch_k, refresh=refresh)
            hits = remote.hits
            repo_id = remote.repo_id
        else:
            if not include_other_repos:
                repo_id = (
                    resolve_target_repo_id_fn(app, repo=repo_url)
                    if repo_url
                    else resolve_target_repo_id_fn(app)
                )
                if repo_id is None:
                    log.warning("mcp.search_code.repo_not_resolved")
                    return SearchResponse(
                        results=[], total_candidates=0, query=query, facets=None
                    )
            search_kwargs: dict[str, Any] = {
                "top_k": fetch_k,
                "conversation_context": conversation_context,
            }
            if repo_id is not None:
                search_kwargs["repo_id"] = repo_id
            hits = app.search_engine.search(query, **search_kwargs)
    except stropha_error_cls as exc:
        log.warning("mcp.search_code.error", error=str(exc))
        return SearchResponse(results=[], total_candidates=0, query=query, facets=None)
    if recursive and hits:
        from .config import get_config
        from .retrieval.recursive import merge_hits

        hits = merge_hits(
            hits,
            app.storage,
            adjacency_lines=get_config().retrieval_features.recursive_adjacency,
            enabled=True,
        )

    if has_filters:
        hits = apply_filters_fn(
            hits,
            language=language,
            path_prefix=path_prefix,
            kind=kind,
            exclude_tests=exclude_tests,
        )

    dedup_filtered = 0
    if dedup and hits:
        before = len(hits)
        hits = [
            hit for hit in hits
            if hit.chunk_id in app.pinned_chunk_ids or hit.chunk_id not in app.seen_chunk_ids
        ]
        dedup_filtered = before - len(hits)

    pinned_prefix_count = 0
    if app.pinned_chunk_ids:
        from .retrieval.pinning import prepend_pins, resolve_pins

        pin_ids = app.pinned_chunk_ids
        if repo_id is not None:
            pin_ids = {
                chunk_id
                for chunk_id in pin_ids
                if (
                    (chunk := app.storage.get_chunk_by_id(chunk_id)) is not None
                    and chunk.get("repo_id") == repo_id
                )
            }
        resolved = resolve_pins(app.storage, pin_ids)
        pinned_hits = [resolved_pin.hit for resolved_pin in resolved if resolved_pin.found and resolved_pin.hit is not None]
        if pinned_hits:
            hits = prepend_pins(pinned_hits, hits)
            pinned_prefix_count = len(pinned_hits)

    if repo_id is not None:
        hits = [hit for hit in hits if _hit_belongs_to_repo(app.storage, hit, repo_id)]

    facets: FacetCounts | None = None
    if include_facets and hits:
        chunk_ids = [hit.chunk_id for hit in hits]
        facet_data = app.storage.compute_facets(chunk_ids)
        facets = FacetCounts(
            language=facet_data["language"],
            kind=facet_data["kind"],
            repo=facet_data["repo"],
        )

    hits = hits[:top_k]

    if compress and hits:
        from .retrieval.compression import compress_hits

        hits = compress_hits(hits, query)

    if relevance_decay and hits:
        from .retrieval.cards import apply_relevance_decay

        hits = apply_relevance_decay(hits)
    elif card_level != "full" and hits:
        from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

        if card_level not in ALLOWED_LEVELS:
            log.warning(
                "mcp.search_code.bad_card_level",
                level=card_level,
                allowed=list(ALLOWED_LEVELS),
            )
        else:
            hits = apply_level_to_snippets(hits, card_level)  # type: ignore[arg-type]

    spilled_count = 0
    truncated_oversize = False
    diversity_deferred_chunk_ids: list[str] = []
    if context_budget is not None and context_budget > 0:
        from .config import get_config
        from .retrieval.packing import pack_to_budget

        features = get_config().retrieval_features
        pack = pack_to_budget(
            hits,
            context_budget,
            diversify=features.diversity_enabled,
            max_chunks_per_file=features.diversity_max_chunks_per_file,
            protected_prefix=max(1, pinned_prefix_count),
        )
        hits = pack.packed
        spilled_count = len(pack.spilled)
        truncated_oversize = pack.truncated
        diversity_deferred_chunk_ids = [hit.chunk_id for hit in pack.deferred]
        log.info(
            "mcp.search_code.packed",
            budget=context_budget,
            total_tokens=pack.total_tokens,
            packed=len(hits),
            spilled=spilled_count,
            truncated=truncated_oversize,
            diversity_deferred=len(diversity_deferred_chunk_ids),
        )

    try:
        track_search_savings_fn(query, hits)
    except Exception:
        log.debug("search.savings.track_failed", exc_info=True)

    if query and query.strip():
        app.recent_queries.append(query.strip())

    if dedup and hits:
        new_seen = {hit.chunk_id for hit in hits if hit.chunk_id not in app.pinned_chunk_ids}
        if new_seen:
            app.seen_chunk_ids.update(new_seen)
            persist_session_state(app)
        log.info(
            "mcp.search_code.dedup",
            filtered_before_pack=dedup_filtered,
            served=len(hits),
            session_seen_total=len(app.seen_chunk_ids),
            pinned=len(app.pinned_chunk_ids),
        )

    # WIN-349: stamp each response with per-repo index freshness so the agent
    # can tell whether results reflect current code. Best-effort — a watermark
    # failure must never break the search itself.
    watermark: list[Any] = []
    try:
        from .server_health_tools import build_index_watermark

        repo_keys: set[str] = set()
        for hit in hits:
            repo = getattr(hit, "repo", None)
            if repo is not None:
                repo_keys.add(repo.normalized_key)
        watermark = build_index_watermark(app.storage, repo_keys)
    except Exception:
        log.debug("search.watermark.failed", exc_info=True)

    from .secret_egress import guard_search_results

    results, redacted_count = guard_search_results(
        [to_result(hit) for hit in hits],
        log=log,
    )
    return SearchResponse(
        results=results,
        total_candidates=len(hits),
        query=query,
        facets=facets,
        spilled_count=spilled_count,
        truncated_oversize=truncated_oversize,
        diversity_deferred_chunk_ids=diversity_deferred_chunk_ids,
        redacted_count=redacted_count,
        index_watermark=watermark,
    )
