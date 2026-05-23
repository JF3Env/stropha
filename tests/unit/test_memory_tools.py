"""Unit tests for the MCP ``remember`` / ``recall`` / ``forget``
tools and the supporting ``Storage.lookup_repo_id`` helper (WIN-223).

The Storage CRUD landed in WIN-222; this file covers only the MCP
exposure layer (parameter routing, target_repo → repo_id
resolution, error envelopes).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.storage.sqlite import Storage


# ---------------------------------------------------------------------------
# Storage.lookup_repo_id
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "lookup.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="github.com/test/repo",
            remote_url="https://github.com/test/repo.git",
            root_path=Path("/tmp/test-repo"),
            default_branch="main",
            head_commit="abc123",
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


class TestLookupRepoId:
    def test_known_key_returns_id(self, storage_with_repo) -> None:
        s, repo_id = storage_with_repo
        assert s.lookup_repo_id("github.com/test/repo") == repo_id

    def test_unknown_key_returns_none(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        assert s.lookup_repo_id("github.com/other/repo") is None

    def test_empty_key_returns_none(self, storage_with_repo) -> None:
        s, _ = storage_with_repo
        assert s.lookup_repo_id("") is None


# ---------------------------------------------------------------------------
# MCP tool helpers — fixtures
# ---------------------------------------------------------------------------


def _make_app_with_real_storage(
    storage: Storage,
    repo_path: Path,
    *,
    session_id: str = "test-session",
) -> MagicMock:
    """Build an AppContext-shaped mock pointing at a real Storage and
    a real path for the target_repo. The git_meta detect() runs
    against ``repo_path`` (which the test fixture initialises as a
    real git repo so normalized_key resolves)."""
    app = MagicMock()
    app.storage = storage
    app.session_id = session_id
    app.config = MagicMock()
    app.config.target_repo = repo_path
    return app


def _make_ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


@pytest.fixture
def fake_repo_path(tmp_path: Path) -> Path:
    """A real-enough on-disk directory whose normalized_key is
    deterministic across test runs. ``detect()`` falls back to
    ``path:<abs>`` when there's no .git/, which is stable enough
    for our test purposes."""
    p = tmp_path / "my-project"
    p.mkdir()
    return p


@pytest.fixture
def storage_with_target(tmp_path: Path, fake_repo_path: Path):
    """Initialise a Storage + register a repo row whose normalized_key
    matches what ``detect(fake_repo_path)`` will return."""
    from stropha.ingest.git_meta import detect

    identity = detect(fake_repo_path)
    s = Storage(tmp_path / "memtools.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(identity)
    yield s, repo_id, identity
    s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# remember tool
# ---------------------------------------------------------------------------


class TestRememberTool:
    def test_writes_when_target_repo_exists(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import remember

        s, repo_id, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        result = remember(
            "decision", "fsrs-authoritative",
            "FSRS is the source of truth for intervals.",
            ctx=_make_ctx(app),
        )
        assert result.ok is True
        assert result.entry is not None
        assert result.entry.kind == "decision"
        assert result.entry.key == "fsrs-authoritative"
        # The Storage row exists.
        rows = s.recall(repo_id=repo_id)
        assert len(rows) == 1

    def test_session_id_recorded_as_source(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(
            s, fake_repo_path, session_id="sess-XYZ",
        )
        result = remember(
            "fact", "x", "y", ctx=_make_ctx(app),
        )
        assert result.entry is not None
        assert result.entry.source == "sess-XYZ"

    def test_expires_at_passed_through(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        result = remember(
            "fact", "x", "y",
            expires_at="2099-01-01T00:00:00+00:00",
            ctx=_make_ctx(app),
        )
        assert result.entry is not None
        assert result.entry.expires_at == "2099-01-01T00:00:00+00:00"

    def test_unindexed_repo_returns_error(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        """When target_repo has no row in the repos table (never
        indexed), remember() should NOT crash — return an actionable
        error envelope."""
        from stropha.server import remember

        # Storage exists but no repo registered.
        s = Storage(tmp_path / "fresh.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app_with_real_storage(s, fake_repo_path)
            result = remember(
                "fact", "x", "y", ctx=_make_ctx(app),
            )
            assert result.ok is False
            assert result.entry is None
            assert "no indexed repo" in (result.error or "")
        finally:
            s.__exit__(None, None, None)

    def test_no_target_repo_returns_error(
        self, storage_with_target,
    ) -> None:
        """``config.target_repo = None`` is a misconfig — surface as
        the same error envelope."""
        from stropha.server import remember

        s, _, _ = storage_with_target
        app = MagicMock()
        app.storage = s
        app.session_id = "sess"
        app.config = MagicMock()
        app.config.target_repo = None
        result = remember("fact", "x", "y", ctx=_make_ctx(app))
        assert result.ok is False

    def test_detect_failure_treated_as_no_repo(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        """If ``ingest.git_meta.detect`` raises, the resolver returns
        None — remember() then surfaces the same error envelope."""
        from stropha.server import remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        with patch(
            "stropha.ingest.git_meta.detect",
            side_effect=RuntimeError("boom"),
        ):
            result = remember("fact", "x", "y", ctx=_make_ctx(app))
        assert result.ok is False


# ---------------------------------------------------------------------------
# recall tool
# ---------------------------------------------------------------------------


class TestRecallTool:
    def test_returns_entries_for_target_repo(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall, remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        remember("decision", "a", "1", ctx=_make_ctx(app))
        remember("fact", "b", "2", ctx=_make_ctx(app))
        result = recall(ctx=_make_ctx(app))
        keys = {(e.kind, e.key) for e in result.entries}
        assert keys == {("decision", "a"), ("fact", "b")}

    def test_filter_by_kind(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall, remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        remember("decision", "a", "1", ctx=_make_ctx(app))
        remember("fact", "b", "2", ctx=_make_ctx(app))
        result = recall(ctx=_make_ctx(app), kind="decision")
        assert len(result.entries) == 1
        assert result.entries[0].kind == "decision"

    def test_filter_by_key_within_kind(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall, remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        remember("fact", "a", "1", ctx=_make_ctx(app))
        remember("fact", "b", "2", ctx=_make_ctx(app))
        result = recall(ctx=_make_ctx(app), kind="fact", key="a")
        assert len(result.entries) == 1
        assert result.entries[0].value == "1"

    def test_no_repo_returns_empty(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall

        s = Storage(tmp_path / "empty.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app_with_real_storage(s, fake_repo_path)
            result = recall(ctx=_make_ctx(app))
            assert result.entries == []
            assert result.repo is None
        finally:
            s.__exit__(None, None, None)

    def test_response_includes_repo_normalized_key(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import recall, remember

        s, _, identity = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        remember("fact", "x", "y", ctx=_make_ctx(app))
        result = recall(ctx=_make_ctx(app))
        assert result.repo == identity.normalized_key

    def test_limit_capped_at_200(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        """The MCP layer caps ``limit`` at 200 so an agent can't
        accidentally ask for the whole table."""
        from stropha.server import recall, remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        for i in range(5):
            remember("fact", f"k{i}", str(i), ctx=_make_ctx(app))
        # Pass an oversized limit — the response should still work.
        result = recall(ctx=_make_ctx(app), limit=999_999)
        assert len(result.entries) == 5


# ---------------------------------------------------------------------------
# forget tool
# ---------------------------------------------------------------------------


class TestForgetTool:
    def test_existing_entry_returns_removed_true(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import forget, remember

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        remember("fact", "x", "y", ctx=_make_ctx(app))
        result = forget("fact", "x", ctx=_make_ctx(app))
        assert result.removed is True
        assert result.kind == "fact"
        assert result.key == "x"

    def test_unknown_entry_returns_removed_false(
        self, storage_with_target, fake_repo_path: Path,
    ) -> None:
        from stropha.server import forget

        s, _, _ = storage_with_target
        app = _make_app_with_real_storage(s, fake_repo_path)
        result = forget("fact", "ghost", ctx=_make_ctx(app))
        assert result.removed is False

    def test_no_target_repo_returns_false_no_crash(
        self, tmp_path: Path, fake_repo_path: Path,
    ) -> None:
        from stropha.server import forget

        s = Storage(tmp_path / "empty.db", embedding_dim=4)
        s.__enter__()
        try:
            app = _make_app_with_real_storage(s, fake_repo_path)
            result = forget("fact", "x", ctx=_make_ctx(app))
            assert result.removed is False
        finally:
            s.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_remember_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "remember" in names

    def test_recall_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "recall" in names

    def test_forget_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "forget" in names
