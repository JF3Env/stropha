"""Session-local host observations and bounded runtime artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import ObservationError
from .models import (
    MAX_RUNTIME_ARTIFACT_CONTENT_BYTES,
    MAX_RUNTIME_ARTIFACT_TOTAL_BYTES,
    MAX_RUNTIME_ARTIFACTS,
    AgentSelectionV1,
    AgentStateSnapshotV1,
    RuntimeArtifactV1,
    RuntimeSourceLocationV1,
)
from .server_models import AgentObservationResponse, RuntimeArtifactResponse
from .server_runtime import AppContext


def _indexed_repository_roots(app: AppContext) -> dict[str, Path | None]:
    try:
        repositories = app.storage.list_repos()
    except Exception as exc:
        raise ObservationError("Could not resolve indexed repositories") from exc

    roots: dict[str, Path | None] = {}
    for repository in repositories:
        key = getattr(repository, "normalized_key", None)
        if not isinstance(key, str) or not key:
            continue
        local_path = getattr(repository, "local_path", None)
        if not isinstance(local_path, str) or not local_path:
            roots[key] = None
            continue
        try:
            root = Path(local_path).expanduser()
            roots[key] = root.resolve() if root.is_absolute() else None
        except (OSError, RuntimeError):
            roots[key] = None
    return roots


def _repository_root(roots: dict[str, Path | None], repository: str) -> Path:
    if repository not in roots:
        raise ObservationError(f"Repository is not indexed: {repository}")
    root = roots[repository]
    if root is None or not root.is_dir():
        raise ObservationError(f"Repository has no accessible local root: {repository}")
    return root


def _canonical_relative_path(root: Path, path: str) -> str:
    if "\x00" in path or "\\" in path:
        raise ObservationError(f"Path must be a canonical repository-relative path: {path!r}")

    relative = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        relative.is_absolute()
        or windows_path.drive
        or relative == PurePosixPath(".")
        or ".." in relative.parts
    ):
        raise ObservationError(f"Path must be a canonical repository-relative path: {path!r}")

    canonical = relative.as_posix()
    try:
        resolved = root.joinpath(*relative.parts).resolve()
    except (OSError, RuntimeError) as exc:
        raise ObservationError(f"Path could not be resolved safely: {path!r}") from exc
    if not resolved.is_relative_to(root):
        raise ObservationError(f"Path escapes repository root: {path!r}")
    return canonical


def _canonical_snapshot(
    roots: dict[str, Path | None], snapshot: AgentStateSnapshotV1
) -> AgentStateSnapshotV1:
    root = _repository_root(roots, snapshot.repository)
    selection: AgentSelectionV1 | None = None
    if snapshot.selection is not None:
        selection = snapshot.selection.model_copy(
            update={"path": _canonical_relative_path(root, snapshot.selection.path)}
        )
    return snapshot.model_copy(
        update={
            "active_file": (
                _canonical_relative_path(root, snapshot.active_file)
                if snapshot.active_file is not None
                else None
            ),
            "recent_files": tuple(
                _canonical_relative_path(root, path) for path in snapshot.recent_files
            ),
            "selection": selection,
            "changed_paths": tuple(
                _canonical_relative_path(root, path) for path in snapshot.changed_paths
            ),
        }
    )


def _canonical_artifact(
    roots: dict[str, Path | None], artifact: RuntimeArtifactV1
) -> RuntimeArtifactV1:
    locations: list[RuntimeSourceLocationV1] = []
    for location in artifact.source_locations:
        root = _repository_root(roots, location.repository)
        locations.append(
            location.model_copy(
                update={"path": _canonical_relative_path(root, location.path)}
            )
        )
    return artifact.model_copy(update={"source_locations": tuple(locations)})


def _content_bytes(content: str) -> int:
    try:
        return len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ObservationError("Runtime artifact content must be valid UTF-8") from exc


def _prune_expired(app: AppContext, now: datetime) -> None:
    expired = [
        artifact_id
        for artifact_id, artifact in app.runtime_artifacts.items()
        if artifact.expires_at <= now
    ]
    for artifact_id in expired:
        del app.runtime_artifacts[artifact_id]


def _observation_response(
    app: AppContext,
    *,
    snapshot: AgentStateSnapshotV1 | None = None,
    roots: dict[str, Path | None] | None = None,
) -> AgentObservationResponse:
    current_snapshot = snapshot if snapshot is not None else app.agent_state
    if current_snapshot is None and not app.runtime_artifacts:
        return AgentObservationResponse(snapshot=None, runtime_artifacts=[])

    roots = roots or _indexed_repository_roots(app)
    return AgentObservationResponse(
        snapshot=(
            _canonical_snapshot(roots, current_snapshot)
            if current_snapshot is not None
            else None
        ),
        runtime_artifacts=[
            _canonical_artifact(roots, artifact)
            for artifact in app.runtime_artifacts.values()
        ],
    )


def observe_agent_state_impl(
    snapshot: AgentStateSnapshotV1 | None,
    app: AppContext,
) -> AgentObservationResponse:
    """Replace or retrieve this session's complete ephemeral agent state."""
    with app.observation_lock:
        now = datetime.now(UTC)
        _prune_expired(app, now)
        if snapshot is not None:
            roots = _indexed_repository_roots(app)
            canonical = _canonical_snapshot(roots, snapshot)
            current = app.agent_state
            if current is not None and canonical.revision <= current.revision:
                raise ObservationError(
                    "Agent state revision must be greater than "
                    f"the current revision {current.revision}"
                )
            response = _observation_response(app, snapshot=canonical, roots=roots)
            app.agent_state = canonical
            return response
        return _observation_response(app)


def record_runtime_artifact_impl(
    artifact: RuntimeArtifactV1,
    app: AppContext,
) -> RuntimeArtifactResponse:
    """Record one immutable runtime artifact within deterministic session quotas."""
    with app.observation_lock:
        now = datetime.now(UTC)
        _prune_expired(app, now)
        roots = _indexed_repository_roots(app)
        canonical = _canonical_artifact(roots, artifact)
        if canonical.expires_at <= now:
            raise ObservationError("Runtime artifact expiry must be in the future")
        if canonical.artifact_id in app.runtime_artifact_ids_seen:
            raise ObservationError(
                f"Runtime artifact already exists in this session: {canonical.artifact_id}"
            )
        if len(app.runtime_artifact_ids_seen) >= MAX_RUNTIME_ARTIFACTS:
            raise ObservationError(
                "Runtime artifact quota exceeded: maximum "
                f"{MAX_RUNTIME_ARTIFACTS} unique artifact IDs per session"
            )

        content_bytes = _content_bytes(canonical.content)
        if content_bytes > MAX_RUNTIME_ARTIFACT_CONTENT_BYTES:
            raise ObservationError(
                "Runtime artifact content exceeds "
                f"{MAX_RUNTIME_ARTIFACT_CONTENT_BYTES} UTF-8 bytes"
            )
        stored_content_bytes = sum(
            _content_bytes(stored.content) for stored in app.runtime_artifacts.values()
        )
        if stored_content_bytes + content_bytes > MAX_RUNTIME_ARTIFACT_TOTAL_BYTES:
            raise ObservationError(
                "Runtime artifact session content exceeds "
                f"{MAX_RUNTIME_ARTIFACT_TOTAL_BYTES} UTF-8 bytes"
            )

        response = RuntimeArtifactResponse(
            artifact=canonical,
            stored_artifacts=len(app.runtime_artifacts) + 1,
            stored_content_bytes=stored_content_bytes + content_bytes,
        )
        app.runtime_artifacts[canonical.artifact_id] = canonical
        app.runtime_artifact_ids_seen.add(canonical.artifact_id)
        return response


__all__ = ["observe_agent_state_impl", "record_runtime_artifact_impl"]
