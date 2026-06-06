"""Repo/session memory related helpers extracted from ``stropha.server``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .server_models import (
    BootstrapResponse,
    CoOccurrenceResponse,
    CrossSessionCachePruneResponse,
    CrossSessionCacheStats,
    DetectedStackEntry,
    EpisodeEntry,
    ForgetResponse,
    MemoryEntry,
    RecallEpisodesResponse,
    RecallResponse,
    RecordEpisodeResponse,
    RememberResponse,
    ScratchpadResponse,
    StalenessResponse,
)
from .server_runtime import AppContext

ResolveTargetRepoId = Callable[[AppContext], int | None]
CurrentHeadCommit = Callable[[AppContext], str | None]
PersistSessionState = Callable[[AppContext], None]


def resolve_target_repo_id(app: AppContext, *, log: Any) -> int | None:
    from .ingest.git_meta import detect as detect_repo

    if app.config.target_repo is None:
        return None
    try:
        identity = detect_repo(app.config.target_repo)
    except Exception as exc:
        log.warning("mcp.memory.detect_repo_failed", error=str(exc))
        return None
    return app.storage.lookup_repo_id(identity.normalized_key)


def current_head_commit(app: AppContext) -> str | None:
    if app.config.target_repo is None:
        return None
    from .meta.provenance import current_head_commit as current_head_commit_impl

    return current_head_commit_impl(app.config.target_repo)


def entry_from_record(rec: Any) -> MemoryEntry:
    return MemoryEntry(
        kind=rec.kind,
        key=rec.key,
        value=rec.value,
        source=rec.source,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        expires_at=rec.expires_at,
        provenance_commit=getattr(rec, "provenance_commit", None),
    )


def episode_to_entry(rec: Any) -> EpisodeEntry:
    return EpisodeEntry(
        id=rec.id,
        title=rec.title,
        summary=rec.summary,
        session_id=rec.session_id,
        touched_files=rec.touched_files,
        touched_symbols=rec.touched_symbols,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        provenance_commit=getattr(rec, "provenance_commit", None),
    )


def _lookup_repo_key(storage: Any, repo_id: int) -> str | None:
    try:
        row = storage._conn.execute(
            "SELECT normalized_key FROM repos WHERE id = ?",
            (repo_id,),
        ).fetchone()
        return row["normalized_key"] if row else None
    except Exception:
        return None


def remember_impl(
    kind: str,
    key: str,
    value: str,
    app: AppContext,
    *,
    expires_at: str | None,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
    current_head_commit_fn: CurrentHeadCommit,
) -> RememberResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return RememberResponse(
            ok=False,
            error=(
                "no indexed repo for target_repo — run `stropha index` "
                "first so the memory entry has a repo to anchor against"
            ),
        )
    rec = app.storage.remember(
        repo_id=repo_id,
        kind=kind,
        key=key,
        value=value,
        source=app.session_id or None,
        expires_at=expires_at,
        provenance_commit=current_head_commit_fn(app),
    )
    return RememberResponse(ok=True, entry=entry_from_record(rec))


def recall_impl(
    app: AppContext,
    *,
    kind: str | None,
    key: str | None,
    limit: int,
    include_expired: bool,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> RecallResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return RecallResponse(entries=[], repo=None)
    rows = app.storage.recall(
        repo_id=repo_id,
        kind=kind,
        key=key,
        limit=max(1, min(limit, 200)),
        include_expired=include_expired,
    )
    repo_key = _lookup_repo_key(app.storage, repo_id) if rows else None
    return RecallResponse(
        entries=[entry_from_record(row) for row in rows],
        repo=repo_key,
    )


def forget_impl(
    kind: str,
    key: str,
    app: AppContext,
    *,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> ForgetResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return ForgetResponse(removed=False, kind=kind, key=key)
    removed = app.storage.forget(repo_id=repo_id, kind=kind, key=key)
    return ForgetResponse(removed=removed, kind=kind, key=key)


def cache_stats_impl(
    app: AppContext,
    *,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> CrossSessionCacheStats:
    from .retrieval.cross_session_cache import stats

    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return CrossSessionCacheStats(total_entries=0, total_hits=0, distinct_models=0, repo=None)
    counters = stats(app.storage, repo_id=repo_id)
    return CrossSessionCacheStats(
        total_entries=counters["total_entries"],
        total_hits=counters["total_hits"],
        distinct_models=counters["distinct_models"],
        repo=None,
    )


def cache_prune_impl(
    app: AppContext,
    *,
    expired_only: bool,
    older_than_iso: str | None,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> CrossSessionCachePruneResponse:
    from .retrieval.cross_session_cache import prune

    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return CrossSessionCachePruneResponse(removed=0)
    removed = prune(
        app.storage,
        repo_id=repo_id,
        expired_only=expired_only,
        older_than_iso=older_than_iso,
    )
    return CrossSessionCachePruneResponse(removed=removed)


def scratchpad_read_impl(app: AppContext) -> ScratchpadResponse:
    return ScratchpadResponse(content=app.scratchpad, char_count=len(app.scratchpad))


def scratchpad_write_impl(
    content: str,
    app: AppContext,
    *,
    persist_session_state_fn: PersistSessionState,
) -> ScratchpadResponse:
    app.scratchpad = content
    persist_session_state_fn(app)
    return ScratchpadResponse(content=app.scratchpad, char_count=len(app.scratchpad))


def scratchpad_append_impl(
    text: str,
    app: AppContext,
    *,
    persist_session_state_fn: PersistSessionState,
) -> ScratchpadResponse:
    if not text:
        return ScratchpadResponse(content=app.scratchpad, char_count=len(app.scratchpad))
    if app.scratchpad and not app.scratchpad.endswith("\n"):
        app.scratchpad = app.scratchpad + "\n" + text
    else:
        app.scratchpad = app.scratchpad + text
    persist_session_state_fn(app)
    return ScratchpadResponse(content=app.scratchpad, char_count=len(app.scratchpad))


def scratchpad_clear_impl(
    app: AppContext,
    *,
    persist_session_state_fn: PersistSessionState,
) -> ScratchpadResponse:
    app.scratchpad = ""
    persist_session_state_fn(app)
    return ScratchpadResponse(content="", char_count=0)


def check_staleness_impl(
    kind: str,
    key: str,
    app: AppContext,
    *,
    anchor_files: list[str] | None,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> StalenessResponse:
    from .meta.provenance import staleness_signal

    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return StalenessResponse(found=False, reason="no indexed repo for target_repo")
    rows = app.storage.recall(repo_id=repo_id, kind=kind, key=key, limit=1)
    if not rows:
        return StalenessResponse(found=False, reason="entry not found")
    entry = rows[0]
    if app.config.target_repo is None:
        return StalenessResponse(
            found=True,
            stale=None,
            since_commit=entry.provenance_commit,
            reason="no target_repo configured",
        )
    signal = staleness_signal(
        app.config.target_repo,
        since_commit=entry.provenance_commit,
        anchor_files=anchor_files,
    )
    return StalenessResponse(
        found=True,
        stale=signal.stale,
        since_commit=signal.since_commit,
        current_commit=signal.current_commit,
        changed_files=signal.changed_files,
        reason=signal.reason,
    )


def bootstrap_environment_impl(app: AppContext) -> BootstrapResponse:
    from .retrieval.bootstrap import detect_bootstrap

    target = app.config.target_repo
    if target is None:
        target = Path.cwd()
    snapshot = detect_bootstrap(Path(target))
    return BootstrapResponse(
        repo_path=snapshot.repo_path,
        stacks=[
            DetectedStackEntry(
                name=stack.name,
                language=stack.language,
                build_command=stack.build_command,
                test_command=stack.test_command,
                run_command=stack.run_command,
                matched_markers=list(stack.matched_markers),
            )
            for stack in snapshot.stacks
        ],
        entry_points=snapshot.entry_points,
        has_git=snapshot.has_git,
        has_tests_dir=snapshot.has_tests_dir,
        has_docs_dir=snapshot.has_docs_dir,
        has_graphify_out=snapshot.has_graphify_out,
    )


def record_episode_impl(
    title: str,
    summary: str,
    touched_files: list[str] | None,
    touched_symbols: list[str] | None,
    app: AppContext,
    *,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
    current_head_commit_fn: CurrentHeadCommit,
) -> RecordEpisodeResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return RecordEpisodeResponse(
            ok=False,
            error=(
                "no indexed repo for target_repo — run `stropha index` "
                "first so the episode has a repo to anchor against"
            ),
        )
    rec = app.storage.record_episode(
        repo_id=repo_id,
        session_id=app.session_id or None,
        title=title,
        summary=summary,
        touched_files=touched_files or [],
        touched_symbols=touched_symbols or [],
        provenance_commit=current_head_commit_fn(app),
    )
    return RecordEpisodeResponse(ok=True, entry=episode_to_entry(rec))


def recall_episodes_impl(
    app: AppContext,
    *,
    query: str | None,
    limit: int,
    since: str | None,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> RecallEpisodesResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return RecallEpisodesResponse(entries=[], repo=None)
    rows = app.storage.recall_episodes(
        repo_id=repo_id,
        query=query,
        limit=max(1, min(limit, 200)),
        since_iso=since,
    )
    repo_key = _lookup_repo_key(app.storage, repo_id) if rows else None
    return RecallEpisodesResponse(
        entries=[episode_to_entry(row) for row in rows],
        repo=repo_key,
    )


def co_occurrence_impl(
    key: str,
    app: AppContext,
    *,
    limit: int,
    in_files: bool,
    in_symbols: bool,
    resolve_target_repo_id_fn: ResolveTargetRepoId,
) -> CoOccurrenceResponse:
    repo_id = resolve_target_repo_id_fn(app)
    if repo_id is None:
        return CoOccurrenceResponse(key=key, files={}, symbols={}, repo=None)
    result = app.storage.episode_co_occurrence(
        repo_id=repo_id,
        key=key,
        in_files=in_files,
        in_symbols=in_symbols,
        limit=max(1, min(limit, 200)),
    )
    return CoOccurrenceResponse(
        key=key,
        files=result.get("files", {}),
        symbols=result.get("symbols", {}),
        repo=_lookup_repo_key(app.storage, repo_id),
    )
