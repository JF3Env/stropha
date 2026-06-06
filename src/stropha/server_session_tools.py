"""Implementation helpers for session-scoped MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .server_models import (
    ExpandChunkResponse,
    ExpandChunksResponse,
    ListPinsResponse,
    PinListEntry,
    PinResponse,
    ResetDedupResponse,
    SearchResult,
)
from .server_runtime import AppContext

ToResult = Callable[[Any], SearchResult]
PersistSessionState = Callable[[AppContext], None]


def reset_dedup_impl(
    app: AppContext,
    *,
    persist_session_state: PersistSessionState,
    log: Any,
) -> ResetDedupResponse:
    count = len(app.seen_chunk_ids)
    app.seen_chunk_ids.clear()
    persist_session_state(app)
    log.info("mcp.reset_dedup", cleared=count, session_id=app.session_id)
    return ResetDedupResponse(cleared=count)


def pin_chunks_impl(
    chunk_ids: list[str],
    app: AppContext,
    *,
    persist_session_state: PersistSessionState,
    log: Any,
) -> PinResponse:
    before = len(app.pinned_chunk_ids)
    app.pinned_chunk_ids.update(chunk_ids)
    added = len(app.pinned_chunk_ids) - before
    if added:
        persist_session_state(app)
    log.info(
        "mcp.pin_chunks",
        added=added,
        total=len(app.pinned_chunk_ids),
        session_id=app.session_id,
    )
    return PinResponse(pinned=sorted(app.pinned_chunk_ids), added=added)


def unpin_chunks_impl(
    chunk_ids: list[str],
    app: AppContext,
    *,
    persist_session_state: PersistSessionState,
    log: Any,
) -> PinResponse:
    before = len(app.pinned_chunk_ids)
    app.pinned_chunk_ids.difference_update(chunk_ids)
    removed = before - len(app.pinned_chunk_ids)
    if removed:
        persist_session_state(app)
    log.info(
        "mcp.unpin_chunks",
        removed=removed,
        total=len(app.pinned_chunk_ids),
        session_id=app.session_id,
    )
    return PinResponse(pinned=sorted(app.pinned_chunk_ids), removed=removed)


def list_pins_impl(app: AppContext, *, to_result: ToResult) -> ListPinsResponse:
    from .retrieval.pinning import resolve_pins

    resolved = resolve_pins(app.storage, app.pinned_chunk_ids)
    entries: list[PinListEntry] = []
    found = 0
    for resolved_pin in resolved:
        if resolved_pin.found and resolved_pin.hit is not None:
            entries.append(
                PinListEntry(
                    chunk_id=resolved_pin.chunk_id,
                    found=True,
                    result=to_result(resolved_pin.hit),
                )
            )
            found += 1
        else:
            entries.append(
                PinListEntry(
                    chunk_id=resolved_pin.chunk_id,
                    found=False,
                    result=None,
                )
            )
    return ListPinsResponse(
        entries=entries,
        found_count=found,
        missing_count=len(entries) - found,
    )


def expand_chunk_impl(
    chunk_id: str,
    card_level: str,
    app: AppContext,
    *,
    to_result: ToResult,
) -> ExpandChunkResponse:
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    hit = app.storage.lookup_by_chunk_id(chunk_id)
    if hit is None:
        return ExpandChunkResponse(chunk_id=chunk_id, found=False, result=None)
    if card_level != "full" and card_level in ALLOWED_LEVELS:
        hit = apply_level_to_snippets([hit], card_level)[0]  # type: ignore[arg-type]
    return ExpandChunkResponse(chunk_id=chunk_id, found=True, result=to_result(hit))


def expand_chunks_impl(
    chunk_ids: list[str],
    card_level: str,
    app: AppContext,
    *,
    to_result: ToResult,
) -> ExpandChunksResponse:
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    results: list[ExpandChunkResponse] = []
    found = 0
    for chunk_id in chunk_ids:
        hit = app.storage.lookup_by_chunk_id(chunk_id)
        if hit is None:
            results.append(ExpandChunkResponse(chunk_id=chunk_id, found=False, result=None))
            continue
        if card_level != "full" and card_level in ALLOWED_LEVELS:
            hit = apply_level_to_snippets([hit], card_level)[0]  # type: ignore[arg-type]
        results.append(
            ExpandChunkResponse(chunk_id=chunk_id, found=True, result=to_result(hit))
        )
        found += 1
    return ExpandChunksResponse(
        results=results,
        found_count=found,
        missing_count=len(chunk_ids) - found,
    )
