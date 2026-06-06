"""Unit tests for memory provenance + staleness (WIN-228, EPIC 1.9 #20).

Covers:
- Schema v11 ``provenance_commit`` columns on project_memory +
  episodes.
- ``Storage.remember`` / ``record_episode`` accept + persist the
  provenance_commit kwarg.
- ``stropha.meta.provenance`` helpers (current_head_commit,
  changed_files_since, staleness_signal).
- MCP ``check_staleness`` tool.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.meta.provenance import (
    StalenessSignal,
    changed_files_since,
    current_head_commit,
    staleness_signal,
)
from stropha.storage.sqlite import SCHEMA_VERSION, Storage

# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_at_least_eleven(self) -> None:
        assert SCHEMA_VERSION >= 11

    def test_provenance_columns_added(self, tmp_path: Path) -> None:
        with Storage(tmp_path / "v11.db", embedding_dim=4) as s:
            for table in ("project_memory", "episodes"):
                cols = s._conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
                names = {c["name"] for c in cols}
                assert "provenance_commit" in names, (
                    f"{table} missing provenance_commit"
                )


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "p.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="k", remote_url=None,
            root_path=tmp_path, default_branch=None,
            head_commit=None,
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


class TestStoragePersistence:
    def test_remember_persists_commit(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            provenance_commit="abc123",
        )
        assert rec.provenance_commit == "abc123"
        # Recall surfaces it too.
        rows = s.recall(repo_id=repo_id, kind="fact", key="x")
        assert rows[0].provenance_commit == "abc123"

    def test_remember_default_none(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        rec = s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
        )
        assert rec.provenance_commit is None

    def test_remember_upsert_updates_commit(
        self, storage_with_repo,
    ) -> None:
        """A subsequent remember with the same (kind, key) updates
        the provenance_commit too — the agent's last write wins."""
        s, repo_id = storage_with_repo
        s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            provenance_commit="old-sha",
        )
        rec = s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y2",
            provenance_commit="new-sha",
        )
        assert rec.provenance_commit == "new-sha"

    def test_record_episode_persists_commit(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(
            repo_id=repo_id, title="t",
            provenance_commit="abc123",
        )
        assert rec.provenance_commit == "abc123"
        rows = s.recall_episodes(repo_id=repo_id)
        assert rows[0].provenance_commit == "abc123"

    def test_record_episode_default_none(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        rec = s.record_episode(repo_id=repo_id, title="t")
        assert rec.provenance_commit is None


# ---------------------------------------------------------------------------
# Helpers — current_head_commit
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> str:
    """Initialise a real git repo and return the first commit SHA."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=path, check=True,
    )
    (path / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=path, check=True, env=env,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, capture_output=True, text=True, check=True,
    ).stdout.strip()


class TestCurrentHeadCommit:
    def test_real_repo_returns_sha(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        out = current_head_commit(repo)
        assert out == sha

    def test_non_git_dir_returns_none(self, tmp_path: Path) -> None:
        assert current_head_commit(tmp_path) is None

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost"
        assert current_head_commit(ghost) is None

    def test_subprocess_failure_returns_none(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        _init_git_repo(repo)

        def boom(*args, **kwargs):
            raise FileNotFoundError("no git here")

        monkeypatch.setattr(subprocess, "run", boom)
        assert current_head_commit(repo) is None

    def test_git_rev_parse_nonzero_returns_none(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Git invocation succeeds but returns non-zero (e.g.
        detached state with no commits). Helper returns None
        instead of leaking the error."""
        repo = tmp_path / "r"
        repo.mkdir()
        _init_git_repo(repo)

        class FakeResult:
            returncode = 1
            stdout = b""
            stderr = b"fatal: ambiguous"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
        assert current_head_commit(repo) is None


# ---------------------------------------------------------------------------
# Helpers — changed_files_since
# ---------------------------------------------------------------------------


class TestChangedFilesSince:
    def test_returns_changed_paths(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        first_sha = _init_git_repo(repo)
        # Add a second commit that changes one file.
        (repo / "second.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "second"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        changed = changed_files_since(repo, first_sha)
        assert changed == ["second.txt"]

    def test_same_sha_returns_empty(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        changed = changed_files_since(repo, sha)
        assert changed == []

    def test_unknown_sha_returns_none(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        _init_git_repo(repo)
        assert changed_files_since(repo, "deadbeef" * 5) is None

    def test_empty_sha_returns_none(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        _init_git_repo(repo)
        assert changed_files_since(repo, "") is None

    def test_non_git_dir_returns_none(self, tmp_path: Path) -> None:
        assert changed_files_since(tmp_path, "abc") is None

    def test_subprocess_failure_returns_none(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)

        def boom(*args, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", boom)
        assert changed_files_since(repo, sha) is None


# ---------------------------------------------------------------------------
# staleness_signal
# ---------------------------------------------------------------------------


class TestStalenessSignal:
    def test_no_since_commit_returns_unknown(self, tmp_path: Path) -> None:
        signal = staleness_signal(tmp_path, since_commit=None)
        assert signal.stale is None
        assert "no provenance" in (signal.reason or "")

    def test_same_sha_is_fresh(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        signal = staleness_signal(repo, since_commit=sha)
        assert signal.stale is False
        assert signal.changed_files == []

    def test_changed_files_make_signal_stale(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        first = _init_git_repo(repo)
        (repo / "extra.txt").write_text("y", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "x"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        signal = staleness_signal(repo, since_commit=first)
        assert signal.stale is True
        assert "extra.txt" in signal.changed_files

    def test_anchor_filter_suppresses_unrelated_changes(
        self, tmp_path: Path,
    ) -> None:
        """The memory was about ``A.py``; a commit touching ``B.py``
        should NOT mark the memory stale when an anchor_files filter
        is passed."""
        repo = tmp_path / "r"
        repo.mkdir()
        first = _init_git_repo(repo)
        # Change B.py — A.py untouched.
        (repo / "B.py").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "x"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        signal = staleness_signal(
            repo, since_commit=first, anchor_files=["A.py"],
        )
        assert signal.stale is False
        assert "B.py" in signal.changed_files  # surfaces still
        assert "anchor" in (signal.reason or "")

    def test_anchor_filter_matches_make_stale(self, tmp_path: Path) -> None:
        repo = tmp_path / "r"
        repo.mkdir()
        first = _init_git_repo(repo)
        (repo / "A.py").write_text("changed", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "x"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        signal = staleness_signal(
            repo, since_commit=first, anchor_files=["A.py"],
        )
        assert signal.stale is True

    def test_git_failure_returns_unknown_with_reason(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When rev-parse fails the verdict is ``None`` (unknown)."""
        repo = tmp_path / "r"
        repo.mkdir()
        # No git init — current_head_commit returns None.
        signal = staleness_signal(repo, since_commit="abc123")
        assert signal.stale is None
        assert signal.reason is not None

    def test_diff_failure_returns_unknown(self, tmp_path: Path) -> None:
        """rev-parse succeeds but the diff fails (unknown SHA)."""
        repo = tmp_path / "r"
        repo.mkdir()
        _init_git_repo(repo)
        signal = staleness_signal(repo, since_commit="ffffffffffff")
        assert signal.stale is None
        assert "git diff" in (signal.reason or "")

    def test_different_sha_no_changed_files_is_fresh(
        self, tmp_path: Path,
    ) -> None:
        """Distinct SHAs with an empty diff (rare — branch merge
        that nets to zero file changes) must still resolve to
        stale=False. Exercises the explicit `if not changed`
        branch."""
        repo = tmp_path / "r"
        repo.mkdir()
        first = _init_git_repo(repo)
        # Create a no-op second commit by changing+reverting a file.
        path = repo / "seed.txt"
        original = path.read_text(encoding="utf-8")
        path.write_text("changed", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "step1"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        path.write_text(original, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "revert"],
            cwd=repo, check=True,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        # Diff from `first` to HEAD touches seed.txt → stale=True;
        # but we want to exercise the "no changes" path explicitly.
        # Mock changed_files_since to return [] so the branch fires.
        with patch(
            "stropha.meta.provenance.changed_files_since",
            return_value=[],
        ):
            signal = staleness_signal(repo, since_commit=first)
        # Different SHAs but the diff is empty → fresh by definition.
        assert signal.stale is False


# ---------------------------------------------------------------------------
# MCP check_staleness tool
# ---------------------------------------------------------------------------


def _make_ctx(storage: Storage, repo_path: Path) -> MagicMock:
    ctx = MagicMock()
    app = MagicMock()
    app.config = MagicMock()
    app.config.target_repo = repo_path
    app.session_id = "sess"
    app.storage = storage
    ctx.request_context.lifespan_context = app
    return ctx


class TestMCPCheckStaleness:
    def test_entry_not_found(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import check_staleness

        s, repo_id = storage_with_repo
        # Need a real-looking target_repo path — use tmp_path.
        ctx = _make_ctx(s, tmp_path)
        # Force _resolve_target_repo_id to return repo_id via patch.
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = check_staleness("decision", "missing", ctx=ctx)
        assert result.found is False
        assert "not found" in (result.reason or "")

    def test_entry_without_provenance_returns_unknown(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import check_staleness

        s, repo_id = storage_with_repo
        # Entry exists but no commit recorded.
        s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
        )
        ctx = _make_ctx(s, tmp_path)
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = check_staleness("fact", "x", ctx=ctx)
        assert result.found is True
        assert result.stale is None
        assert "no provenance" in (result.reason or "")

    def test_fresh_entry_returns_stale_false(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import check_staleness

        s, repo_id = storage_with_repo
        # Init a real git repo for the staleness check.
        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            provenance_commit=sha,
        )
        ctx = _make_ctx(s, repo)
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = check_staleness("fact", "x", ctx=ctx)
        assert result.found is True
        assert result.stale is False
        assert result.since_commit == sha
        assert result.current_commit == sha

    def test_no_target_repo_returns_unknown(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import check_staleness

        s, repo_id = storage_with_repo
        s.remember(
            repo_id=repo_id, kind="fact", key="x", value="y",
            provenance_commit="abc123",
        )
        ctx = MagicMock()
        app = MagicMock()
        app.config = MagicMock()
        app.config.target_repo = None  # ← key
        app.session_id = "sess"
        app.storage = s
        ctx.request_context.lifespan_context = app
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = check_staleness("fact", "x", ctx=ctx)
        assert result.found is True
        assert result.stale is None
        assert "target_repo" in (result.reason or "")

    def test_unindexed_repo_returns_found_false(
        self, tmp_path: Path,
    ) -> None:
        from stropha.server import check_staleness

        s = Storage(tmp_path / "noop.db", embedding_dim=4)
        s.__enter__()
        try:
            ctx = _make_ctx(s, tmp_path)
            result = check_staleness("fact", "x", ctx=ctx)
            assert result.found is False
            assert "no indexed repo" in (result.reason or "")
        finally:
            s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# MCP provenance auto-capture
# ---------------------------------------------------------------------------


class TestProvenanceAutoCapture:
    def test_remember_captures_head_commit(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import remember

        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        s, repo_id = storage_with_repo

        ctx = _make_ctx(s, repo)
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = remember("fact", "k", "v", ctx=ctx)
        assert result.ok is True
        # provenance_commit should match the repo's HEAD.
        assert result.entry is not None
        assert result.entry.provenance_commit == sha

    def test_record_episode_captures_head_commit(
        self, tmp_path: Path, storage_with_repo,
    ) -> None:
        from stropha.server import record_episode

        repo = tmp_path / "r"
        repo.mkdir()
        sha = _init_git_repo(repo)
        s, repo_id = storage_with_repo

        ctx = _make_ctx(s, repo)
        with patch(
            "stropha.server._resolve_target_repo_id", return_value=repo_id
        ):
            result = record_episode("t", ctx=ctx)
        assert result.ok is True
        assert result.entry is not None
        assert result.entry.provenance_commit == sha


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_check_staleness_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "check_staleness" in names


# ---------------------------------------------------------------------------
# StalenessSignal dataclass
# ---------------------------------------------------------------------------


class TestStalenessSignalDataclass:
    def test_frozen(self) -> None:
        signal = StalenessSignal(stale=None, since_commit=None, current_commit=None)
        with pytest.raises((AttributeError, TypeError)):
            signal.stale = True  # type: ignore[misc]

    def test_defaults(self) -> None:
        signal = StalenessSignal(
            stale=False, since_commit="a", current_commit="a",
        )
        assert signal.changed_files == []
        assert signal.reason is None
