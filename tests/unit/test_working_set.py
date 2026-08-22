"""Unit tests for the working-set assembler and MCP wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.retrieval.working_set import (
    WorkingSet,
    WorkingSetEntry,
    _default_estimator,
    _render_entry,
    _render_markdown,
    _to_entry,
    build_working_set,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Hit:
    def __init__(
        self,
        chunk_id: str = "c1",
        rel_path: str = "src/foo.py",
        symbol: str = "Foo",
        kind: str = "class",
        start_line: int = 10,
        end_line: int = 20,
        snippet: str = "class Foo:\n    pass",
    ) -> None:
        self.chunk_id = chunk_id
        self.rel_path = rel_path
        self.symbol = symbol
        self.kind = kind
        self.start_line = start_line
        self.end_line = end_line
        self.snippet = snippet


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_entry_defaults(self) -> None:
        e = WorkingSetEntry(chunk_id="c1")
        assert e.role == "recent"
        assert e.tokens == 0
        assert e.path == ""

    def test_workingset_defaults(self) -> None:
        ws = WorkingSet()
        assert ws.session_id == ""
        assert ws.entries == []
        assert ws.total_tokens == 0
        assert ws.dropped_count == 0
        assert ws.markdown == ""

    def test_frozen(self) -> None:
        e = WorkingSetEntry(chunk_id="c1")
        with pytest.raises((AttributeError, Exception)):
            e.role = "pinned"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _default_estimator
# ---------------------------------------------------------------------------


class TestDefaultEstimator:
    def test_4_chars_per_token(self) -> None:
        assert _default_estimator("abcd") == 1
        assert _default_estimator("abcdefgh") == 2

    def test_min_one(self) -> None:
        assert _default_estimator("") == 1
        assert _default_estimator("ab") == 1


# ---------------------------------------------------------------------------
# _to_entry
# ---------------------------------------------------------------------------


class TestToEntry:
    def test_basic(self) -> None:
        entry = _to_entry(_Hit(), role="pinned", estimator=_default_estimator)
        assert entry.role == "pinned"
        assert entry.path == "src/foo.py"
        assert entry.symbol == "Foo"
        assert entry.start_line == 10
        assert entry.tokens >= 1

    def test_missing_attributes_defaulted(self) -> None:
        class _Bare:
            pass

        entry = _to_entry(_Bare(), role="recent", estimator=_default_estimator)
        assert entry.chunk_id == ""
        assert entry.snippet == ""


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_empty_returns_message(self) -> None:
        out = _render_markdown([], session_id="")
        assert "no working-set" in out

    def test_session_id_in_header(self) -> None:
        out = _render_markdown([], session_id="sess-123")
        assert "sess-123" in out

    def test_pinned_only(self) -> None:
        entries = [WorkingSetEntry(chunk_id="c1", role="pinned", path="x.py")]
        out = _render_markdown(entries, session_id="")
        assert "## Pinned" in out
        assert "## Recent" not in out

    def test_recent_only(self) -> None:
        entries = [WorkingSetEntry(chunk_id="c1", role="recent", path="x.py")]
        out = _render_markdown(entries, session_id="")
        assert "## Recent" in out
        assert "## Pinned" not in out

    def test_both_sections(self) -> None:
        entries = [
            WorkingSetEntry(chunk_id="p", role="pinned", path="a.py"),
            WorkingSetEntry(chunk_id="r", role="recent", path="b.py"),
        ]
        out = _render_markdown(entries, session_id="")
        assert "## Pinned" in out
        assert "## Recent" in out
        # Pinned section appears before Recent in the output.
        assert out.index("## Pinned") < out.index("## Recent")


class TestRenderEntry:
    def test_full_entry(self) -> None:
        e = WorkingSetEntry(
            chunk_id="c1", role="pinned", path="x.py", symbol="Foo",
            kind="class", start_line=10, end_line=20, snippet="def f(): ...",
        )
        out = _render_entry(e)
        joined = "\n".join(out)
        assert "x.py:10-20" in joined
        assert "Foo" in joined
        assert "class" in joined
        assert "def f(): ..." in joined

    def test_minimal_entry(self) -> None:
        e = WorkingSetEntry(chunk_id="c1")
        out = _render_entry(e)
        joined = "\n".join(out)
        assert "?:0-0" in joined  # no path → '?'

    def test_no_snippet_no_code_block(self) -> None:
        e = WorkingSetEntry(chunk_id="c1", path="x.py")
        out = _render_entry(e)
        # No code block when snippet is empty.
        assert "```" not in "\n".join(out)


# ---------------------------------------------------------------------------
# build_working_set
# ---------------------------------------------------------------------------


class TestBuildWorkingSet:
    def test_empty_inputs(self) -> None:
        ws = build_working_set([], [], lookup_fn=lambda _: None)
        assert ws.entries == []
        assert ws.markdown.startswith("# Working set")
        assert "_no working-set" in ws.markdown

    def test_pinned_always_included(self) -> None:
        hits = {"p1": _Hit(chunk_id="p1"), "p2": _Hit(chunk_id="p2")}
        ws = build_working_set(
            ["p1", "p2"], [],
            lookup_fn=lambda cid: hits.get(cid),
        )
        assert ws.pinned_count == 2
        assert all(e.role == "pinned" for e in ws.entries)

    def test_pinned_deduped(self) -> None:
        ws = build_working_set(
            ["p", "p", "p"], [],
            lookup_fn=lambda _: _Hit(),
        )
        assert ws.pinned_count == 1

    def test_recent_truncated_to_max(self) -> None:
        hits = {f"r{i}": _Hit(chunk_id=f"r{i}") for i in range(50)}
        ws = build_working_set(
            [], [f"r{i}" for i in range(50)],
            lookup_fn=lambda cid: hits.get(cid),
            max_recent=5,
            budget_tokens=100000,  # no budget pressure
        )
        assert ws.recent_count == 5

    def test_recent_most_recent_first(self) -> None:
        """Recent list is processed reversed so the most-recent
        chunk wins when budget runs out."""
        hits = {"old": _Hit(chunk_id="old"), "new": _Hit(chunk_id="new")}
        ws = build_working_set(
            [], ["old", "new"],
            lookup_fn=lambda cid: hits.get(cid),
            max_recent=1,
        )
        # max_recent=1 → only the most recent ("new") makes it.
        assert ws.recent_count == 1
        assert ws.entries[0].chunk_id == "new"

    def test_pinned_excludes_recent(self) -> None:
        """A chunk_id appearing in both pinned + recent lists is
        included once (as pinned)."""
        hit = _Hit(chunk_id="shared")
        ws = build_working_set(
            ["shared"], ["shared"],
            lookup_fn=lambda _: hit,
        )
        assert len(ws.entries) == 1
        assert ws.entries[0].role == "pinned"

    def test_budget_drops_recent(self) -> None:
        """Recent chunks that would push past the budget are
        dropped — pinned chunks always fit."""
        snippet = "x" * 100  # ~25 tokens
        hits = {f"r{i}": _Hit(chunk_id=f"r{i}", snippet=snippet) for i in range(5)}
        ws = build_working_set(
            [], list(hits.keys()),
            lookup_fn=lambda cid: hits.get(cid),
            budget_tokens=50,  # ~2 entries fit
            max_recent=10,
        )
        assert ws.recent_count <= 2
        assert ws.dropped_count >= 3

    def test_pinned_busts_budget_recorded(self) -> None:
        long_snippet = "x" * 1000
        ws = build_working_set(
            ["p"], [],
            lookup_fn=lambda _: _Hit(chunk_id="p", snippet=long_snippet),
            budget_tokens=10,
        )
        # Pinned still included; note explains.
        assert ws.pinned_count == 1
        assert any("exceed budget" in n for n in ws.notes)

    def test_lookup_returns_none_skipped(self) -> None:
        ws = build_working_set(
            ["missing"], ["also-missing"],
            lookup_fn=lambda _: None,
        )
        assert ws.entries == []
        assert ws.pinned_count == 0

    def test_lookup_exception_captured(self) -> None:
        def boom(cid: str):
            raise RuntimeError("db boom")

        ws = build_working_set(["c1"], [], lookup_fn=boom)
        assert ws.entries == []
        assert any("db boom" in n for n in ws.notes)

    def test_empty_chunk_id_skipped(self) -> None:
        ws = build_working_set(
            ["", "  "], ["", ""],
            lookup_fn=lambda _: _Hit(),
        )
        assert ws.entries == []

    def test_session_id_in_markdown(self) -> None:
        ws = build_working_set(
            ["p"], [],
            lookup_fn=lambda _: _Hit(chunk_id="p"),
            session_id="sess-abc",
        )
        assert "sess-abc" in ws.markdown


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import working_set as mod
        assert set(mod.__all__) == {
            "WorkingSet",
            "WorkingSetEntry",
            "build_working_set",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    session_id: str = "in-mem",
    pinned: list[str] | None = None,
    seen: list[str] | None = None,
    hits: dict[str, _Hit] | None = None,
) -> MagicMock:
    app = MagicMock()
    app.session_id = session_id
    app.pinned_chunk_ids = list(pinned or [])
    app.seen_chunk_ids = list(seen or [])
    app.storage.lookup_by_chunk_id.side_effect = (
        lambda cid: (hits or {}).get(cid)
    )
    app.storage.get_chunk_by_id.side_effect = lambda cid: (
        {"chunk_id": cid, "repo_id": 1} if (hits or {}).get(cid) is not None else None
    )
    app.storage.lookup_repo_id.return_value = 1
    app.config.target_repo = Path.cwd()
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestWorkingSetMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "working_set_tool" in tool_names

    def test_context_free_resource_not_registered(self) -> None:
        from stropha.server import mcp

        resource_uris = [r.uri for r in mcp._resource_manager.list_resources()]
        assert not any(str(uri).startswith("stropha://working-set") for uri in resource_uris)

    def test_default_session(self) -> None:
        from stropha.server import working_set_tool

        hits = {"p": _Hit(chunk_id="p"), "r": _Hit(chunk_id="r")}
        ctx = _make_ctx(pinned=["p"], seen=["r"], hits=hits)
        response = working_set_tool("", ctx=ctx)
        assert response.pinned_count == 1
        assert response.recent_count == 1

    def test_explicit_current_session(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(
            session_id="session-1",
            pinned=["s-pin"],
            seen=["s-seen"],
            hits={"s-pin": _Hit(chunk_id="s-pin"), "s-seen": _Hit(chunk_id="s-seen")},
        )
        response = working_set_tool("session-1", ctx=ctx)
        assert response.pinned_count == 1
        assert response.recent_count == 1
        assert response.session_id == "session-1"

    def test_foreign_session_is_rejected(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(
            session_id="session-1",
            pinned=["local"],
            hits={"local": _Hit(chunk_id="local")},
        )
        response = working_set_tool("ghost", ctx=ctx)
        assert response.pinned_count == 0
        assert response.session_id == "session-1"
        assert response.notes == ["Working sets are isolated to the current MCP session."]
        ctx.request_context.lifespan_context.storage.get_session.assert_not_called()

    def test_negative_budget_clamped(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(
            pinned=["p"], hits={"p": _Hit(chunk_id="p")},
        )
        response = working_set_tool("", budget_tokens=-100, ctx=ctx)
        assert response.budget_tokens == 0

    def test_negative_max_recent_clamped(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(
            seen=["r"], hits={"r": _Hit(chunk_id="r")},
        )
        response = working_set_tool("", max_recent=-5, ctx=ctx)
        assert response.recent_count == 0

    def test_foreign_chunks_are_filtered(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(
            pinned=["local", "foreign"],
            hits={"local": _Hit(chunk_id="local"), "foreign": _Hit(chunk_id="foreign")},
        )
        app = ctx.request_context.lifespan_context
        app.storage.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "repo_id": 1 if cid == "local" else 2,
        }

        response = working_set_tool("", ctx=ctx)

        assert [entry.chunk_id for entry in response.entries] == ["local"]

    def test_unresolved_repository_fails_closed(self) -> None:
        from stropha.server import working_set_tool

        ctx = _make_ctx(pinned=["foreign"], hits={"foreign": _Hit(chunk_id="foreign")})
        app = ctx.request_context.lifespan_context
        app.storage.lookup_repo_id.return_value = None

        response = working_set_tool("", ctx=ctx)

        assert response.entries == []
        assert response.notes
