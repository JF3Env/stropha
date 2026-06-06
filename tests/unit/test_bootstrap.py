"""Unit tests for environment bootstrap (WIN-227).

Covers the ``stropha.retrieval.bootstrap`` detection module + the
MCP ``bootstrap_environment`` tool wrapper.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.retrieval.bootstrap import (
    BootstrapSnapshot,
    DetectedStack,
    detect_bootstrap,
    detect_entry_points,
    detect_stacks,
)

# ---------------------------------------------------------------------------
# detect_stacks
# ---------------------------------------------------------------------------


class TestDetectStacks:
    def test_empty_repo_returns_no_stacks(self, tmp_path: Path) -> None:
        assert detect_stacks(tmp_path) == []

    def test_python_uv_detected(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        (tmp_path / "uv.lock").write_text("# locked")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "python-uv" for s in stacks)

    def test_python_pip_detected_via_requirements(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "requirements.txt").write_text("pytest\n")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "python-pip" for s in stacks)

    def test_node_detected_via_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "node" for s in stacks)

    def test_pnpm_higher_priority_than_node(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "pnpm-lock.yaml").write_text("")
        stacks = detect_stacks(tmp_path)
        # pnpm priority=75 vs node=70 → pnpm should sort first.
        first = stacks[0]
        assert first.name == "pnpm"

    def test_rust_detected(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "rust" for s in stacks)
        rust = next(s for s in stacks if s.name == "rust")
        assert rust.test_command == "cargo test"

    def test_go_detected(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module x\n")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "go" for s in stacks)

    def test_java_gradle_detected(self, tmp_path: Path) -> None:
        (tmp_path / "build.gradle").write_text("")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "java-gradle" for s in stacks)

    def test_java_maven_detected(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text("<project/>")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "java-maven" for s in stacks)

    def test_ruby_detected(self, tmp_path: Path) -> None:
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'\n")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "ruby" for s in stacks)

    def test_csharp_detected_via_glob(self, tmp_path: Path) -> None:
        """Glob-based marker (``*.csproj``) — exercise the
        iterdir() match path."""
        (tmp_path / "App.csproj").write_text("<Project/>")
        stacks = detect_stacks(tmp_path)
        assert any(s.name == "dotnet" for s in stacks)

    def test_polyglot_repo_returns_multiple_stacks(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'")
        names = {s.name for s in detect_stacks(tmp_path)}
        # Python via pyproject (uv if uv.lock present, else just one
        # of pyproject's path matches — at least the Python tag).
        assert any(s.startswith("python") for s in names)
        assert "node" in names
        assert "rust" in names

    def test_sorted_by_priority_desc(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        (tmp_path / "uv.lock").write_text("")
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'")
        stacks = detect_stacks(tmp_path)
        # Both python-uv (80) and rust (80) match at the same priority;
        # the sort is stable so ordering between equals is fine. The
        # important contract is that lower-priority detectors come
        # AFTER higher-priority ones.
        priorities = []
        from stropha.retrieval.bootstrap import _DETECTORS

        priority_map = {d.name: d.priority for d in _DETECTORS}
        for s in stacks:
            priorities.append(priority_map[s.name])
        assert priorities == sorted(priorities, reverse=True)

    def test_marker_glob_no_match_returns_none(
        self, tmp_path: Path,
    ) -> None:
        """Iterdir branch with no matching file → detector silent."""
        (tmp_path / "unrelated.txt").write_text("")
        # No .csproj / .sln present.
        stacks = detect_stacks(tmp_path)
        assert all(s.name != "dotnet" for s in stacks)


# ---------------------------------------------------------------------------
# detect_entry_points
# ---------------------------------------------------------------------------


class TestDetectEntryPoints:
    def test_readme_detected(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Project\n")
        out = detect_entry_points(tmp_path)
        assert "readme" in out
        assert "README.md" in out["readme"]

    def test_multiple_readme_variants(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("")
        (tmp_path / "README.rst").write_text("")
        out = detect_entry_points(tmp_path)
        assert set(out["readme"]) == {"README.md", "README.rst"}

    def test_python_main_detected(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("")
        out = detect_entry_points(tmp_path)
        assert out.get("python") == ["main.py"]

    def test_docs_category(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("")
        out = detect_entry_points(tmp_path)
        assert "docs" in out
        assert "CLAUDE.md" in out["docs"]

    def test_empty_repo_returns_empty_dict(self, tmp_path: Path) -> None:
        assert detect_entry_points(tmp_path) == {}


# ---------------------------------------------------------------------------
# detect_bootstrap end-to-end
# ---------------------------------------------------------------------------


class TestDetectBootstrap:
    def test_nonexistent_path_returns_empty_snapshot(
        self, tmp_path: Path,
    ) -> None:
        ghost = tmp_path / "ghost"
        snap = detect_bootstrap(ghost)
        # Path stringified, no stacks.
        assert snap.stacks == []
        assert snap.entry_points == {}

    def test_full_snapshot_for_python_repo(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        (tmp_path / "README.md").write_text("# X\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "docs").mkdir()
        (tmp_path / "graphify-out").mkdir()
        (tmp_path / ".git").mkdir()
        snap = detect_bootstrap(tmp_path)
        assert snap.has_git is True
        assert snap.has_tests_dir is True
        assert snap.has_docs_dir is True
        assert snap.has_graphify_out is True
        # Python detected.
        assert any(s.language == "python" for s in snap.stacks)
        # README surfaced.
        assert "readme" in snap.entry_points

    def test_test_dir_check_also_matches_singular_test(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "test").mkdir()
        snap = detect_bootstrap(tmp_path)
        assert snap.has_tests_dir is True

    def test_no_git_dir(self, tmp_path: Path) -> None:
        snap = detect_bootstrap(tmp_path)
        assert snap.has_git is False


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


def _ctx(target: Path) -> MagicMock:
    ctx = MagicMock()
    app = MagicMock()
    app.config = MagicMock()
    app.config.target_repo = target
    app.session_id = "sess"
    ctx.request_context.lifespan_context = app
    return ctx


class TestMCPBootstrapEnvironment:
    def test_returns_detected_stacks(self, tmp_path: Path) -> None:
        from stropha.server import bootstrap_environment

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        result = bootstrap_environment(ctx=_ctx(tmp_path))
        assert result.repo_path.endswith(str(tmp_path.name)) or str(tmp_path) in result.repo_path
        assert any(s.language == "python" for s in result.stacks)

    def test_no_target_repo_uses_cwd(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When ``target_repo`` is None, the tool falls back to
        ``Path.cwd()``. We monkeypatch cwd to a controlled tmp dir."""
        from stropha.server import bootstrap_environment

        monkeypatch.chdir(tmp_path)
        ctx = MagicMock()
        app = MagicMock()
        app.config = MagicMock()
        app.config.target_repo = None
        app.session_id = "sess"
        ctx.request_context.lifespan_context = app
        # Drop a marker so detection picks something up.
        (tmp_path / "go.mod").write_text("module x\n")
        result = bootstrap_environment(ctx=ctx)
        assert any(s.language == "go" for s in result.stacks)

    def test_empty_dir_returns_empty_snapshot(
        self, tmp_path: Path,
    ) -> None:
        from stropha.server import bootstrap_environment

        result = bootstrap_environment(ctx=_ctx(tmp_path))
        assert result.stacks == []
        assert result.entry_points == {}

    def test_response_models_serializable(self, tmp_path: Path) -> None:
        """The MCP response is a Pydantic model — must round-trip via
        model_dump for the MCP transport."""
        from stropha.server import bootstrap_environment

        (tmp_path / "package.json").write_text("{}")
        result = bootstrap_environment(ctx=_ctx(tmp_path))
        dumped = result.model_dump()
        assert isinstance(dumped, dict)
        assert "stacks" in dumped


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_bootstrap_environment_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "bootstrap_environment" in names


# ---------------------------------------------------------------------------
# Frozen dataclass sanity
# ---------------------------------------------------------------------------


class TestDataclassImmutability:
    def test_detected_stack_frozen(self) -> None:
        s = DetectedStack(
            name="x", language="py",
            build_command=None, test_command=None, run_command=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            s.name = "other"  # type: ignore[misc]

    def test_snapshot_frozen(self) -> None:
        snap = BootstrapSnapshot(repo_path="/tmp")
        with pytest.raises((AttributeError, TypeError)):
            snap.repo_path = "/other"  # type: ignore[misc]
