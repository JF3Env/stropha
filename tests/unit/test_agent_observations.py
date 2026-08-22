from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from stropha.errors import ObservationError
from stropha.models import (
    MAX_RUNTIME_ARTIFACT_CONTENT_BYTES,
    MAX_RUNTIME_ARTIFACT_TOTAL_BYTES,
    MAX_RUNTIME_ARTIFACTS,
    AgentStateSnapshotV1,
    RuntimeArtifactV1,
    RuntimeSourceLocationV1,
)
from stropha.server_observation_tools import (
    observe_agent_state_impl,
    record_runtime_artifact_impl,
)
from stropha.server_runtime import AppContext

_REPOSITORY = "local:test-observations"


def _app(root) -> AppContext:
    storage = MagicMock()
    storage.list_repos.return_value = [
        SimpleNamespace(normalized_key=_REPOSITORY, local_path=str(root))
    ]
    return AppContext(
        config=SimpleNamespace(target_repo=root),
        storage=storage,
        embedder=MagicMock(),
        search_engine=MagicMock(),
        built=MagicMock(),
        trace_recorder=MagicMock(),
    )


def _snapshot(revision: int = 1, **changes) -> AgentStateSnapshotV1:
    values = {
        "revision": revision,
        "repository": _REPOSITORY,
        "active_file": "src/current.py",
        "recent_files": ["src/recent.py"],
        "selection": None,
        "changed_paths": ["src/changed.py"],
        "runtime_artifact_ids": ["pytest-failure"],
        "observed_at": datetime.now(UTC),
    }
    values.update(changes)
    return AgentStateSnapshotV1.model_validate(values)


def _artifact(
    artifact_id: str = "pytest-failure",
    content: str = "test failed",
    **changes,
) -> RuntimeArtifactV1:
    values = {
        "artifact_id": artifact_id,
        "kind": "test_failure",
        "media_type": "text/plain",
        "producer": "pytest",
        "content": content,
        "source_locations": [
            RuntimeSourceLocationV1(
                repository=_REPOSITORY,
                path="tests/test_example.py",
                start_line=10,
                end_line=12,
            )
        ],
        "redaction_status": "not_required",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
    }
    values.update(changes)
    return RuntimeArtifactV1.model_validate(values)


def test_agent_state_is_strict_and_replaces_only_with_higher_revision(tmp_path) -> None:
    app = _app(tmp_path)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentStateSnapshotV1.model_validate(
            {**_snapshot().model_dump(), "unexpected": "field"}
        )

    first = observe_agent_state_impl(
        _snapshot(active_file="src//current.py", recent_files=["./src/recent.py"]), app
    )
    assert first.snapshot is not None
    assert first.snapshot.active_file == "src/current.py"
    assert first.snapshot.recent_files == ("src/recent.py",)
    assert app.selected_repository == _REPOSITORY

    second = observe_agent_state_impl(
        _snapshot(
            revision=2,
            active_file="src/next.py",
            recent_files=[],
            changed_paths=[],
            runtime_artifact_ids=[],
        ),
        app,
    )
    assert second.snapshot is not None
    assert second.snapshot.revision == 2
    assert second.snapshot.active_file == "src/next.py"
    assert second.snapshot.recent_files == ()

    with pytest.raises(ObservationError, match="greater than the current revision 2"):
        observe_agent_state_impl(_snapshot(revision=2), app)
    assert app.agent_state == second.snapshot


def test_observations_and_artifacts_are_isolated_between_sessions(tmp_path) -> None:
    first = _app(tmp_path)
    second = _app(tmp_path)
    observe_agent_state_impl(_snapshot(), first)
    observe_agent_state_impl(_snapshot(), second)

    recorded = record_runtime_artifact_impl(_artifact(), first)
    assert recorded.stored_artifacts == 1
    assert recorded.stored_content_bytes == len("test failed")

    first_view = observe_agent_state_impl(None, first)
    second_view = observe_agent_state_impl(None, second)
    assert [item.artifact_id for item in first_view.runtime_artifacts] == ["pytest-failure"]
    assert second_view.runtime_artifacts == []
    first.trace_recorder.record.assert_not_called()
    second.trace_recorder.record.assert_not_called()
    with pytest.raises(ObservationError, match="already exists in this session"):
        record_runtime_artifact_impl(_artifact(), first)

    after_expiry = datetime.now(UTC) + timedelta(hours=2)
    with patch("stropha.server_observation_tools.datetime") as clock:
        clock.now.return_value = after_expiry
        assert observe_agent_state_impl(None, first).runtime_artifacts == []
    with pytest.raises(ObservationError, match="already exists in this session"):
        record_runtime_artifact_impl(_artifact(), first)


def test_paths_cannot_escape_repository_directly_or_through_symlink(tmp_path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    app = _app(root)

    with pytest.raises(ObservationError, match="canonical repository-relative"):
        observe_agent_state_impl(_snapshot(active_file="../outside/secret.py"), app)

    with pytest.raises(ObservationError, match="canonical repository-relative"):
        observe_agent_state_impl(_snapshot(active_file="C:/outside/secret.py"), app)

    with pytest.raises(ObservationError, match="escapes repository root"):
        observe_agent_state_impl(_snapshot(active_file="escape/secret.py"), app)

    with pytest.raises(ObservationError, match="escapes repository root"):
        record_runtime_artifact_impl(
            _artifact(
                source_locations=[
                    RuntimeSourceLocationV1(
                        repository=_REPOSITORY,
                        path="escape/trace.py",
                        start_line=1,
                        end_line=1,
                    )
                ]
            ),
            app,
        )

    pending = root / "pending"
    observe_agent_state_impl(_snapshot(active_file="pending/future.py"), app)
    pending.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ObservationError, match="escapes repository root"):
        observe_agent_state_impl(None, app)

    atomic_app = _app(root)
    observe_agent_state_impl(_snapshot(), atomic_app)
    record_runtime_artifact_impl(
        _artifact(
            source_locations=[
                RuntimeSourceLocationV1(
                    repository=_REPOSITORY,
                    path="artifact-pending/trace.py",
                    start_line=1,
                    end_line=1,
                )
            ]
        ),
        atomic_app,
    )
    (root / "artifact-pending").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ObservationError, match="escapes repository root"):
        observe_agent_state_impl(_snapshot(revision=2), atomic_app)
    assert atomic_app.agent_state is not None
    assert atomic_app.agent_state.revision == 1


def test_runtime_artifact_quotas_and_expiry_are_explicit(tmp_path) -> None:
    app = _app(tmp_path)
    multibyte_content = "é" * (MAX_RUNTIME_ARTIFACT_CONTENT_BYTES // 2 + 1)
    with pytest.raises(ObservationError, match="UTF-8 bytes"):
        record_runtime_artifact_impl(_artifact(content=multibyte_content), app)

    with pytest.raises(ObservationError, match="expiry must be in the future"):
        record_runtime_artifact_impl(
            _artifact(expires_at=datetime.now(UTC) - timedelta(seconds=1)), app
        )

    for index in range(MAX_RUNTIME_ARTIFACTS):
        record_runtime_artifact_impl(_artifact(artifact_id=f"artifact-{index}", content="x"), app)
    with pytest.raises(ObservationError, match=f"maximum {MAX_RUNTIME_ARTIFACTS} unique"):
        record_runtime_artifact_impl(_artifact(artifact_id="over-count", content="x"), app)

    bytes_app = _app(tmp_path)
    full_artifact_count = MAX_RUNTIME_ARTIFACT_TOTAL_BYTES // MAX_RUNTIME_ARTIFACT_CONTENT_BYTES
    for index in range(full_artifact_count):
        record_runtime_artifact_impl(
            _artifact(
                artifact_id=f"full-{index}",
                content="x" * MAX_RUNTIME_ARTIFACT_CONTENT_BYTES,
            ),
            bytes_app,
        )
    with pytest.raises(ObservationError, match="session content exceeds"):
        record_runtime_artifact_impl(
            _artifact(artifact_id="over-bytes", content="x"), bytes_app
        )
