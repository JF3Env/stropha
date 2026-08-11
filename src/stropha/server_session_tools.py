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
from .server_runtime import AppContext, chunk_belongs_to_repo

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
    repo_id: int | None,
    persist_session_state: PersistSessionState,
    log: Any,
) -> PinResponse:
    chunk_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_belongs_to_repo(app.storage, chunk_id, repo_id)
    ]
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
    visible_pins = [
        chunk_id
        for chunk_id in app.pinned_chunk_ids
        if chunk_belongs_to_repo(app.storage, chunk_id, repo_id)
    ]
    return PinResponse(pinned=sorted(visible_pins), added=added)


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


def list_pins_impl(
    app: AppContext,
    *,
    to_result: ToResult,
    repo_id: int | None,
) -> ListPinsResponse:
    from .retrieval.pinning import resolve_pins

    pin_ids = (
        list(app.pinned_chunk_ids)
        if repo_id is None
        else [
            chunk_id
            for chunk_id in app.pinned_chunk_ids
            if chunk_belongs_to_repo(app.storage, chunk_id, repo_id)
        ]
    )
    resolved = resolve_pins(app.storage, pin_ids)
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
    repo_id: int | None,
) -> ExpandChunkResponse:
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    hit = app.storage.lookup_by_chunk_id(chunk_id)
    if hit is None or not chunk_belongs_to_repo(app.storage, chunk_id, repo_id):
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
    repo_id: int | None,
) -> ExpandChunksResponse:
    from .retrieval.cards import ALLOWED_LEVELS, apply_level_to_snippets

    results: list[ExpandChunkResponse] = []
    found = 0
    for chunk_id in chunk_ids:
        hit = app.storage.lookup_by_chunk_id(chunk_id)
        if hit is None or not chunk_belongs_to_repo(app.storage, chunk_id, repo_id):
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
