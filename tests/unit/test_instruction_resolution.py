from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stropha.errors import InstructionResolutionError
from stropha.models import MAX_AGENT_PATH_CHARS
from stropha.server_instruction_tools import (
    MAX_INSTRUCTION_CHARS,
    MAX_INSTRUCTION_FILE_BYTES,
    MAX_INSTRUCTION_SOURCES,
    MAX_INSTRUCTION_TOTAL_BYTES,
    _windows_path_is_contained,
    resolve_agent_instructions_impl,
)

_REPOSITORY = "github.com/example/repo"


def _repo_stats(key: str, root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        normalized_key=key,
        url=f"https://{key}.git",
        local_path=str(root),
        default_branch="main",
        head_commit="abc123",
    )


def _app(
    root: Path,
    *,
    repository: str = _REPOSITORY,
    active_file: str | None = "src/pkg/service.py",
    changed_paths: tuple[str, ...] = ("src/pkg/service.py",),
    extra_repositories: list[tuple[str, Path]] | None = None,
) -> MagicMock:
    repositories = [(repository, root), *(extra_repositories or [])]
    ids = {key: index + 1 for index, (key, _root) in enumerate(repositories)}
    app = MagicMock()
    app.selected_repository = repository
    app.agent_state = SimpleNamespace(
        repository=repository,
        active_file=active_file,
        changed_paths=changed_paths,
    )
    app.config.target_repo = root
    app.storage.lookup_repo_id.side_effect = ids.get
    app.storage.list_repos.return_value = [
        _repo_stats(key, repo_root) for key, repo_root in repositories
    ]
    return app


def test_resolves_root_to_active_file_scope_with_stable_peer_order(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "AGENTS.md").write_text("root agents", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("root claude", encoding="utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src agents", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "CLAUDE.md").write_text("package claude", encoding="utf-8")
    (tmp_path / "docs" / "AGENTS.md").write_text("docs only", encoding="utf-8")

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path),
        path=None,
        repo=None,
        max_chars=16_000,
    )

    assert result.repository == _REPOSITORY
    assert result.active_path == "src/pkg/service.py"
    assert result.changed_paths == ["src/pkg/service.py"]
    assert [source.path for source in result.sources] == [
        "AGENTS.md",
        "CLAUDE.md",
        "src/AGENTS.md",
        "src/pkg/CLAUDE.md",
    ]
    assert [source.scope for source in result.sources] == [".", ".", "src", "src/pkg"]
    assert [source.precedence for source in result.sources] == [0, 0, 1, 2]
    assert result.sources[0].sha256 == hashlib.sha256(b"root agents").hexdigest()
    assert result.included_chars == result.total_chars
    assert result.truncated is False


def test_explicit_path_selects_only_its_instruction_ancestors(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "CLAUDE.md").write_text("root", encoding="utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src", encoding="utf-8")
    (tmp_path / "docs" / "AGENTS.md").write_text("docs", encoding="utf-8")

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path),
        path="docs/guide.md",
        repo=None,
        max_chars=16_000,
    )

    assert result.active_path == "docs/guide.md"
    assert [source.path for source in result.sources] == [
        "CLAUDE.md",
        "docs/AGENTS.md",
    ]


def test_explicit_repository_does_not_reuse_foreign_agent_state(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    explicit = tmp_path / "explicit"
    selected.mkdir()
    explicit.mkdir()
    (selected / "CLAUDE.md").write_text("selected", encoding="utf-8")
    (explicit / "AGENTS.md").write_text("explicit", encoding="utf-8")
    explicit_key = "github.com/example/explicit"

    result = resolve_agent_instructions_impl(
        app=_app(selected, extra_repositories=[(explicit_key, explicit)]),
        path=None,
        repo=explicit_key,
        max_chars=16_000,
    )

    assert result.repository == explicit_key
    assert result.active_path is None
    assert result.changed_paths == []
    assert [source.path for source in result.sources] == ["AGENTS.md"]


def test_character_budget_is_hard_and_fair_across_peer_sources(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("a" * 10, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("c" * 10, encoding="utf-8")

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path, active_file=None, changed_paths=()),
        path=None,
        repo=None,
        max_chars=10,
    )

    assert [source.content for source in result.sources] == ["a" * 5, "c" * 5]
    assert [source.included_chars for source in result.sources] == [5, 5]
    assert all(source.truncated for source in result.sources)
    assert result.total_chars == 20
    assert result.included_chars == 10
    assert result.truncated is True


def test_character_budget_counts_unicode_characters_not_bytes(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("λ" * 4, encoding="utf-8")

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path, active_file=None, changed_paths=()),
        path=None,
        repo=None,
        max_chars=3,
    )

    assert result.sources[0].content == "λ" * 3
    assert result.sources[0].full_chars == 4
    assert result.sources[0].included_chars == 3


@pytest.mark.parametrize("max_chars", [0, MAX_INSTRUCTION_CHARS + 1])
def test_rejects_character_budget_outside_bounds(tmp_path: Path, max_chars: int) -> None:
    with pytest.raises(InstructionResolutionError, match="max_chars must be between"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path),
            path=None,
            repo=None,
            max_chars=max_chars,
        )


def test_rejects_oversized_instruction_before_decoding(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"a" * (MAX_INSTRUCTION_FILE_BYTES + 1))

    with pytest.raises(InstructionResolutionError, match="Instruction source exceeds"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file=None),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_rejects_oversized_applicable_source_set(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    source_bytes = b"a" * (MAX_INSTRUCTION_TOTAL_BYTES // 5)
    for directory in (tmp_path, tmp_path / "src", nested):
        (directory / "AGENTS.md").write_bytes(source_bytes)
        (directory / "CLAUDE.md").write_bytes(source_bytes)

    with pytest.raises(InstructionResolutionError, match="aggregate limit"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_rejects_excess_empty_sources_before_metadata_growth(tmp_path: Path) -> None:
    directory = tmp_path
    for depth in range(MAX_INSTRUCTION_SOURCES // 2 + 1):
        (directory / "AGENTS.md").touch()
        (directory / "CLAUDE.md").touch()
        directory /= f"scope-{depth}"
        directory.mkdir()

    active_file = directory.relative_to(tmp_path) / "service.py"
    with pytest.raises(InstructionResolutionError, match="source limit"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file=active_file.as_posix()),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_rejects_non_utf8_instruction(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")

    with pytest.raises(InstructionResolutionError, match="must be UTF-8"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file=None),
            path=None,
            repo=None,
            max_chars=16_000,
        )


@pytest.mark.parametrize("path", ["../secret.py", "/tmp/secret.py", "src\\secret.py"])
def test_rejects_non_canonical_active_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(InstructionResolutionError, match=r"repository-relative|escapes"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path),
            path=path,
            repo=None,
            max_chars=16_000,
        )


def test_rejects_active_path_above_observation_limit(tmp_path: Path) -> None:
    with pytest.raises(InstructionResolutionError, match="character limit"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path),
            path="a" * (MAX_AGENT_PATH_CHARS + 1),
            repo=None,
            max_chars=16_000,
        )


def test_rejects_instruction_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-agents.md"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "AGENTS.md").symlink_to(outside)

    with pytest.raises(InstructionResolutionError, match="escapes repository root"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file="src/service.py"),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_reads_contained_instruction_symlink(tmp_path: Path) -> None:
    target = tmp_path / "shared-instructions.md"
    target.write_text("contained", encoding="utf-8")
    (tmp_path / "AGENTS.md").symlink_to(target)

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path, active_file=None),
        path=None,
        repo=None,
        max_chars=16_000,
    )

    assert result.sources[0].path == "AGENTS.md"
    assert result.sources[0].content == "contained"


def test_rejects_instruction_replaced_by_escape_symlink_during_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = tmp_path / "AGENTS.md"
    instruction.write_text("contained", encoding="utf-8")
    outside = tmp_path.parent / "replacement-agents.md"
    outside.write_text("outside", encoding="utf-8")
    original_resolve = Path.resolve
    replaced = False

    def resolve_then_replace(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal replaced
        resolved = original_resolve(path, *args, **kwargs)
        if path == instruction and not replaced:
            instruction.unlink()
            instruction.symlink_to(outside)
            replaced = True
        return resolved

    monkeypatch.setattr(Path, "resolve", resolve_then_replace)

    with pytest.raises(InstructionResolutionError, match="could not be opened safely"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file=None),
            path=None,
            repo=None,
            max_chars=16_000,
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor traversal")
def test_rejects_repository_ancestor_replaced_by_escape_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = tmp_path / "guarded-root"
    repository = ancestor / "repo"
    repository.mkdir(parents=True)
    (repository / "AGENTS.md").write_text("inside", encoding="utf-8")
    saved_ancestor = tmp_path / "saved-root"
    outside_ancestor = tmp_path / "outside-root"
    outside_repository = outside_ancestor / "repo"
    outside_repository.mkdir(parents=True)
    (outside_repository / "AGENTS.md").write_text("outside", encoding="utf-8")
    import stropha.server_instruction_tools as instruction_tools

    original_open = instruction_tools.os.open
    replaced = False

    def open_after_replacement(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and (path == ancestor.name or Path(path) == repository):
            ancestor.rename(saved_ancestor)
            ancestor.symlink_to(outside_ancestor, target_is_directory=True)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(instruction_tools.os, "open", open_after_replacement)
    try:
        with pytest.raises(InstructionResolutionError, match="root could not be opened safely"):
            resolve_agent_instructions_impl(
                app=_app(repository, active_file=None),
                path=None,
                repo=None,
                max_chars=16_000,
            )
    finally:
        if ancestor.is_symlink():
            ancestor.unlink()
            saved_ancestor.rename(ancestor)


@pytest.mark.skipif(not hasattr(os, "mkfifo") or os.name == "nt", reason="POSIX FIFO")
def test_non_regular_instruction_open_cannot_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instruction = tmp_path / "AGENTS.md"
    os.mkfifo(instruction)
    import stropha.server_instruction_tools as instruction_tools

    original_open = instruction_tools.os.open
    checked_nonblocking = False

    def assert_nonblocking_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal checked_nonblocking
        if path == instruction.name:
            assert flags & os.O_NONBLOCK
            checked_nonblocking = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(instruction_tools.os, "open", assert_nonblocking_open)

    result = resolve_agent_instructions_impl(
        app=_app(tmp_path, active_file=None),
        path=None,
        repo=None,
        max_chars=16_000,
    )

    assert checked_nonblocking is True
    assert result.sources == []


def test_rejects_active_path_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.with_name(f"{tmp_path.name}-outside")
    outside.mkdir()
    (tmp_path / "src").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InstructionResolutionError, match="escapes repository root"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path, active_file="src/service.py"),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_rejects_unknown_repository(tmp_path: Path) -> None:
    with pytest.raises(InstructionResolutionError, match="not indexed"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path),
            path=None,
            repo="github.com/example/missing",
            max_chars=16_000,
        )


def test_windows_handle_containment_preserves_case_sensitive_boundaries() -> None:
    root = r"\Device\HarddiskVolume1\source\repo"

    assert _windows_path_is_contained(root, root)
    assert _windows_path_is_contained(root, root + r"\src\service.py")
    assert not _windows_path_is_contained(root, root + "-other" + r"\service.py")
    assert not _windows_path_is_contained(root, r"\Device\HarddiskVolume1\source\REPO\file.py")


def test_rejects_inaccessible_repository_root(tmp_path: Path) -> None:
    with pytest.raises(InstructionResolutionError, match="no accessible indexed root"):
        resolve_agent_instructions_impl(
            app=_app(tmp_path / "missing"),
            path=None,
            repo=None,
            max_chars=16_000,
        )


def test_mcp_wrapper_uses_session_context(tmp_path: Path) -> None:
    from stropha.server import resolve_agent_instructions

    (tmp_path / "CLAUDE.md").write_text("root", encoding="utf-8")
    app = _app(tmp_path, active_file=None, changed_paths=())
    app.trace_recorder = MagicMock()
    snapshot = app.agent_state
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app

    result = resolve_agent_instructions(ctx=ctx)

    assert result.repository == _REPOSITORY
    assert [source.path for source in result.sources] == ["CLAUDE.md"]
    assert app.agent_state is snapshot
    assert app.trace_recorder.mock_calls == []
    assert {call[0] for call in app.storage.method_calls} <= {
        "ensure_connected",
        "list_repos",
        "lookup_repo_id",
    }
