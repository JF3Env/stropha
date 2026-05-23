"""Unit tests for the ``expand_chunk`` / ``expand_chunks`` MCP tools
and the supporting ``Storage.lookup_by_chunk_id`` method (WIN-216).

Just-in-time dereferencing mirrors Claude Code's file-on-demand
pattern: the agent gets lightweight handles (chunk_ids) from cheap
searches, then expands the handles it actually cares about.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk, SearchHit
from stropha.storage.sqlite import Storage


# ---------------------------------------------------------------------------
# Storage.lookup_by_chunk_id
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    """Spin up a SQLite storage with one chunk + one embedding."""
    db = tmp_path / "test.db"
    s = Storage(db, embedding_dim=4)
    s.__enter__()
    # Insert via the canonical upsert path so the SearchHit projection
    # gets every field (repo joins, etc.) populated.
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="github.com/test/repo",
            remote_url="https://github.com/test/repo.git",
            root_path=Path("/tmp/test-repo"),
            default_branch="main",
            head_commit="abc1234",
        )
    )
    chunk = Chunk(
        chunk_id="known-chunk",
        rel_path="src/foo.py",
        language="python",
        kind="function",
        symbol="my_function",
        parent_chunk_id=None,
        start_line=10,
        end_line=20,
        content="def my_function(x):\n    return x * 2\n",
        content_hash="hash-known",
    )
    s.upsert_chunk(
        chunk=chunk,
        embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_model="test-embed",
        embedding_dim=4,
        repo_id=repo_id,
    )
    yield s
    s.__exit__(None, None, None)


class TestLookupByChunkId:
    def test_known_id_returns_hit(self, storage: Storage) -> None:
        hit = storage.lookup_by_chunk_id("known-chunk")
        assert hit is not None
        assert hit.chunk_id == "known-chunk"
        assert hit.rel_path == "src/foo.py"
        assert hit.symbol == "my_function"
        assert hit.start_line == 10
        assert hit.end_line == 20

    def test_unknown_id_returns_none(self, storage: Storage) -> None:
        assert storage.lookup_by_chunk_id("no-such-chunk") is None

    def test_returns_repo_info(self, storage: Storage) -> None:
        hit = storage.lookup_by_chunk_id("known-chunk")
        assert hit is not None
        assert hit.repo is not None
        assert hit.repo.normalized_key == "github.com/test/repo"
        assert hit.repo.url == "https://github.com/test/repo.git"

    def test_snippet_populated(self, storage: Storage) -> None:
        hit = storage.lookup_by_chunk_id("known-chunk")
        assert hit is not None
        assert "my_function" in hit.snippet


# ---------------------------------------------------------------------------
# MCP expand_chunk / expand_chunks — exercised via direct call
# ---------------------------------------------------------------------------


def _hit(chunk_id: str = "c1") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/x.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="def f():\n    return 1\n",
        language="python",
        kind="function",
        symbol="f",
    )


def _make_app(storage_lookup: dict[str, SearchHit | None]) -> MagicMock:
    """Mock AppContext whose storage.lookup_by_chunk_id consults
    the provided dict."""
    app = MagicMock()
    app.storage.lookup_by_chunk_id.side_effect = (
        lambda cid: storage_lookup.get(cid)
    )
    app.session_id = None
    return app


def _make_ctx(app: MagicMock) -> MagicMock:
    """Mock MCP Context that yields our app via the lifespan field."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestExpandChunk:
    def test_returns_found_with_result_when_id_exists(self) -> None:
        from stropha.server import expand_chunk

        app = _make_app({"c1": _hit("c1")})
        result = expand_chunk("c1", ctx=_make_ctx(app))
        assert result.found is True
        assert result.chunk_id == "c1"
        assert result.result is not None
        assert result.result.chunk_id == "c1"

    def test_returns_not_found_when_id_missing(self) -> None:
        from stropha.server import expand_chunk

        app = _make_app({})
        result = expand_chunk("ghost", ctx=_make_ctx(app))
        assert result.found is False
        assert result.chunk_id == "ghost"
        assert result.result is None

    def test_card_level_line_renders_handle(self) -> None:
        """When card_level='line', the result's snippet is the
        compact path:lines symbol [kind] form."""
        from stropha.server import expand_chunk

        app = _make_app({"c1": _hit("c1")})
        result = expand_chunk("c1", card_level="line", ctx=_make_ctx(app))
        assert result.result is not None
        assert "src/x.py:1-10" in result.result.snippet

    def test_invalid_card_level_falls_back_to_full(self) -> None:
        """Unknown card_level is silently ignored — the agent gets
        the full snippet rather than an error."""
        from stropha.server import expand_chunk

        app = _make_app({"c1": _hit("c1")})
        result = expand_chunk("c1", card_level="quantum", ctx=_make_ctx(app))
        assert result.result is not None
        # Original snippet preserved (no rendering applied).
        assert "def f()" in result.result.snippet


class TestExpandChunks:
    def test_batch_returns_one_result_per_id(self) -> None:
        from stropha.server import expand_chunks

        app = _make_app({
            "c1": _hit("c1"),
            "c2": _hit("c2"),
        })
        result = expand_chunks(["c1", "c2"], ctx=_make_ctx(app))
        assert len(result.results) == 2
        assert result.found_count == 2
        assert result.missing_count == 0

    def test_mixed_found_and_missing(self) -> None:
        from stropha.server import expand_chunks

        app = _make_app({
            "c1": _hit("c1"),
            # c2 is missing
            "c3": _hit("c3"),
        })
        result = expand_chunks(["c1", "c2", "c3"], ctx=_make_ctx(app))
        assert len(result.results) == 3
        assert result.found_count == 2
        assert result.missing_count == 1
        # Order preserved.
        ids = [r.chunk_id for r in result.results]
        assert ids == ["c1", "c2", "c3"]
        # c2 marked as not found.
        c2 = next(r for r in result.results if r.chunk_id == "c2")
        assert c2.found is False
        assert c2.result is None

    def test_empty_input(self) -> None:
        from stropha.server import expand_chunks

        app = _make_app({})
        result = expand_chunks([], ctx=_make_ctx(app))
        assert result.results == []
        assert result.found_count == 0
        assert result.missing_count == 0

    def test_all_missing(self) -> None:
        from stropha.server import expand_chunks

        app = _make_app({})
        result = expand_chunks(["a", "b", "c"], ctx=_make_ctx(app))
        assert result.found_count == 0
        assert result.missing_count == 3
        assert all(not r.found for r in result.results)

    def test_card_level_applied_to_each_hit(self) -> None:
        from stropha.server import expand_chunks

        app = _make_app({"c1": _hit("c1"), "c2": _hit("c2")})
        result = expand_chunks(
            ["c1", "c2"], card_level="line", ctx=_make_ctx(app),
        )
        for r in result.results:
            assert r.result is not None
            assert "src/x.py:1-10" in r.result.snippet


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_expand_chunk_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "expand_chunk" in tool_names

    def test_expand_chunks_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "expand_chunks" in tool_names
