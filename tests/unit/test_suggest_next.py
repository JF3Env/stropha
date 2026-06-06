"""Unit tests for the suggest_next_tool primitive + MCP wrapper
(WIN-238, EPIC 1.10 #29)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stropha.retrieval.suggest_next import (
    CODE_KINDS,
    DOC_KINDS,
    Suggestion,
    _dedupe_suggestions,
    _looks_compound,
    _looks_question,
    suggest_next_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Hit:
    """Minimal SearchHit-like stub for suggestion tests."""

    def __init__(
        self,
        chunk_id: str = "c1",
        symbol: str | None = "FooClass",
        kind: str | None = "class",
        rel_path: str | None = "src/foo.py",
    ) -> None:
        self.chunk_id = chunk_id
        self.symbol = symbol
        self.kind = kind
        self.rel_path = rel_path


# ---------------------------------------------------------------------------
# Suggestion dataclass
# ---------------------------------------------------------------------------


class TestSuggestionDataclass:
    def test_defaults(self) -> None:
        s = Suggestion(tool="x")
        assert s.tool == "x"
        assert s.args == {}
        assert s.reason == ""
        assert s.confidence == 0.5

    def test_frozen(self) -> None:
        s = Suggestion(tool="x")
        with pytest.raises((AttributeError, Exception)):
            s.tool = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Query-shape helpers
# ---------------------------------------------------------------------------


class TestLooksCompound:
    @pytest.mark.parametrize("q", [
        "find x and y",
        "do this then that",
        "x plus y",
        "x & y",
        "FOO AND BAR",  # case-insensitive
    ])
    def test_compound_detected(self, q: str) -> None:
        assert _looks_compound(q) is True

    @pytest.mark.parametrize("q", [
        "how does foo work",
        "where is bar",
        "andoriam",  # 'and' not a standalone word
        "",
    ])
    def test_non_compound(self, q: str) -> None:
        assert _looks_compound(q) is False


class TestLooksQuestion:
    @pytest.mark.parametrize("q", [
        "how does X work",
        "what is Y",
        "WHY did this break",
    ])
    def test_question(self, q: str) -> None:
        assert _looks_question(q) is True

    def test_non_question(self) -> None:
        assert _looks_question("FooClass") is False
        assert _looks_question("") is False


# ---------------------------------------------------------------------------
# Empty-hits branch
# ---------------------------------------------------------------------------


class TestEmptyHits:
    def test_empty_and_no_query_returns_bootstrap(self) -> None:
        out = suggest_next_tool("", [])
        tools = [s.tool for s in out]
        assert "bootstrap_environment" in tools

    def test_empty_and_compound_query_returns_plan(self) -> None:
        out = suggest_next_tool("find X and Y", [])
        tools = [s.tool for s in out]
        assert "plan_investigation" in tools
        # plan ranks above bootstrap (higher confidence).
        plan = next(s for s in out if s.tool == "plan_investigation")
        assert plan.args == {"query": "find X and Y"}

    def test_empty_and_question_returns_hyde(self) -> None:
        out = suggest_next_tool("how does X work", [])
        tools = [s.tool for s in out]
        assert "search_code" in tools
        sc = next(s for s in out if s.tool == "search_code")
        assert sc.args["use_hyde"] is True

    def test_empty_and_compound_question_returns_both(self) -> None:
        out = suggest_next_tool("how does X work and what is Y", [])
        tools = [s.tool for s in out]
        # Compound triggers plan_investigation; question triggers hyde.
        assert "plan_investigation" in tools
        assert "search_code" in tools


# ---------------------------------------------------------------------------
# Code-hits branch
# ---------------------------------------------------------------------------


class TestCodeHits:
    def test_top_class_with_graph_suggests_callers_and_tests(self) -> None:
        out = suggest_next_tool(
            "FooClass",
            [_Hit(symbol="FooClass", kind="class")],
            has_graph=True,
        )
        tools = [s.tool for s in out]
        assert "find_callers" in tools
        assert "find_tests_for" in tools

    @pytest.mark.parametrize("kind", sorted(CODE_KINDS))
    def test_all_code_kinds_trigger_callers(self, kind: str) -> None:
        out = suggest_next_tool(
            "Anchor",
            [_Hit(symbol="Anchor", kind=kind)],
            has_graph=True,
        )
        assert "find_callers" in [s.tool for s in out]

    def test_no_graph_suppresses_callers(self) -> None:
        out = suggest_next_tool(
            "FooClass",
            [_Hit(symbol="FooClass", kind="class")],
            has_graph=False,
        )
        assert "find_callers" not in [s.tool for s in out]
        assert "find_tests_for" not in [s.tool for s in out]

    def test_no_symbol_suppresses_callers(self) -> None:
        out = suggest_next_tool(
            "anonymous",
            [_Hit(symbol=None, kind="class", rel_path="src/x.py")],
            has_graph=True,
        )
        assert "find_callers" not in [s.tool for s in out]


# ---------------------------------------------------------------------------
# Doc-hits branch
# ---------------------------------------------------------------------------


class TestDocHits:
    @pytest.mark.parametrize("kind", sorted(DOC_KINDS))
    def test_doc_kind_with_rationale_suggests_find_rationale(
        self, kind: str,
    ) -> None:
        out = suggest_next_tool(
            "design",
            [_Hit(symbol="Section", kind=kind, rel_path="docs/x.md")],
            has_graph=True,
            has_rationale=True,
        )
        assert "find_rationale" in [s.tool for s in out]

    def test_doc_without_rationale_edges_skips_rationale(self) -> None:
        out = suggest_next_tool(
            "design",
            [_Hit(symbol="Section", kind="doc", rel_path="docs/x.md")],
            has_graph=True,
            has_rationale=False,
        )
        assert "find_rationale" not in [s.tool for s in out]


# ---------------------------------------------------------------------------
# Path distribution
# ---------------------------------------------------------------------------


class TestPathDistribution:
    def test_single_path_suggests_outline(self) -> None:
        hits = [
            _Hit(chunk_id="a", symbol="A", rel_path="src/x.py"),
            _Hit(chunk_id="b", symbol="B", rel_path="src/x.py"),
        ]
        out = suggest_next_tool("query", hits, has_graph=False)
        outline = next(s for s in out if s.tool == "get_file_outline")
        assert outline.args == {"path": "src/x.py"}

    def test_many_paths_suggests_find_related(self) -> None:
        hits = [
            _Hit(chunk_id=str(i), symbol="A", rel_path=f"src/p{i}.py")
            for i in range(5)
        ]
        out = suggest_next_tool("query", hits, has_graph=True)
        assert "find_related" in [s.tool for s in out]

    def test_many_paths_no_graph_no_find_related(self) -> None:
        hits = [
            _Hit(chunk_id=str(i), symbol="A", rel_path=f"src/p{i}.py")
            for i in range(5)
        ]
        out = suggest_next_tool("query", hits, has_graph=False)
        assert "find_related" not in [s.tool for s in out]


# ---------------------------------------------------------------------------
# Compound query with results
# ---------------------------------------------------------------------------


class TestCompoundWithResults:
    def test_compound_query_suggests_plan(self) -> None:
        out = suggest_next_tool(
            "find X and Y",
            [_Hit()],
            has_graph=True,
        )
        assert "plan_investigation" in [s.tool for s in out]


# ---------------------------------------------------------------------------
# Many hits → assemble_context
# ---------------------------------------------------------------------------


class TestManyHits:
    def test_eight_or_more_hits_suggests_assemble(self) -> None:
        hits = [_Hit(chunk_id=str(i)) for i in range(10)]
        out = suggest_next_tool("query", hits, has_graph=True)
        assert "assemble_context" in [s.tool for s in out]

    def test_few_hits_no_assemble(self) -> None:
        out = suggest_next_tool("query", [_Hit(), _Hit()], has_graph=True)
        assert "assemble_context" not in [s.tool for s in out]


# ---------------------------------------------------------------------------
# Ranking + truncation
# ---------------------------------------------------------------------------


class TestRanking:
    def test_sorted_by_confidence_desc(self) -> None:
        hits = [_Hit(symbol="FooClass", kind="class")]
        out = suggest_next_tool("FooClass", hits, has_graph=True)
        confidences = [s.confidence for s in out]
        assert confidences == sorted(confidences, reverse=True)

    def test_max_suggestions_truncates(self) -> None:
        hits = [_Hit(chunk_id=str(i), rel_path=f"p{i}.py") for i in range(10)]
        out = suggest_next_tool(
            "FooClass and BarClass", hits,
            has_graph=True, has_rationale=True,
            max_suggestions=2,
        )
        assert len(out) <= 2

    def test_max_suggestions_zero_returns_empty(self) -> None:
        hits = [_Hit()]
        out = suggest_next_tool("q", hits, max_suggestions=0)
        assert out == []


# ---------------------------------------------------------------------------
# Dedup helper
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_keeps_highest_confidence(self) -> None:
        suggestions = [
            Suggestion(tool="t", args={"a": 1}, confidence=0.3),
            Suggestion(tool="t", args={"a": 1}, confidence=0.9),
            Suggestion(tool="t", args={"a": 1}, confidence=0.5),
        ]
        out = _dedupe_suggestions(suggestions)
        assert len(out) == 1
        assert out[0].confidence == 0.9

    def test_distinct_args_kept(self) -> None:
        suggestions = [
            Suggestion(tool="t", args={"a": 1}),
            Suggestion(tool="t", args={"a": 2}),
        ]
        assert len(_dedupe_suggestions(suggestions)) == 2

    def test_distinct_tools_kept(self) -> None:
        suggestions = [
            Suggestion(tool="t1", args={"a": 1}),
            Suggestion(tool="t2", args={"a": 1}),
        ]
        assert len(_dedupe_suggestions(suggestions)) == 2

    def test_unhashable_args_collapse_per_tool(self) -> None:
        """args with a list value can't be frozen; the fallback key
        groups by tool alone, so two unhashable-args suggestions for
        the same tool collapse to the higher-confidence one."""
        s1 = Suggestion(
            tool="x", args={"items": [1, 2]}, confidence=0.4,
        )
        s2 = Suggestion(
            tool="x", args={"items": [3, 4]}, confidence=0.8,
        )
        out = _dedupe_suggestions([s1, s2])
        assert len(out) == 1
        assert out[0].confidence == 0.8


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import suggest_next as mod
        assert set(mod.__all__) == {
            "CODE_KINDS",
            "DOC_KINDS",
            "Suggestion",
            "suggest_next_tool",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_app(
    hits_by_id: dict[str, _Hit | None],
    *,
    seen: list[str] | None = None,
    graph_loaded_val: bool = True,
    has_rationale_val: bool = True,
    lookup_raises: bool = False,
) -> MagicMock:
    app = MagicMock()
    if lookup_raises:
        app.storage.lookup_by_chunk_id.side_effect = RuntimeError("boom")
    else:
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: hits_by_id.get(cid)
        )
    app.seen_chunk_ids = list(seen) if seen else []
    # graph_loaded / has_rationale_edges are module-level helpers we
    # patch via monkeypatch in the test (see fixtures below).
    app._graph_loaded_val = graph_loaded_val
    app._has_rationale_val = has_rationale_val
    return app


def _make_ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


@pytest.fixture
def patch_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch graph_loaded / has_rationale_edges to read the flags off
    the AppContext mock so tests can toggle them per-case."""
    import stropha.server as srv

    def _gl(storage: object) -> bool:
        # The MagicMock for app.storage references the parent mock;
        # walk up to find the _graph_loaded_val we set in _make_app.
        # In practice tests pass the value via the mock's parent.
        return getattr(storage._mock_parent, "_graph_loaded_val", True)

    def _hr(storage: object) -> bool:
        return getattr(storage._mock_parent, "_has_rationale_val", True)

    monkeypatch.setattr(srv, "graph_loaded", _gl)
    monkeypatch.setattr(srv, "has_rationale_edges", _hr)


class TestSuggestNextToolMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "suggest_next_tool_mcp" in tool_names

    def test_explicit_chunk_ids_used(self, patch_graph: None) -> None:
        from stropha.server import suggest_next_tool_mcp

        app = _make_app({"c1": _Hit(symbol="FooClass", kind="class")})
        out = suggest_next_tool_mcp(
            "FooClass", ["c1"], ctx=_make_ctx(app),
        )
        assert out.based_on_hits == 1
        assert out.graph_loaded is True
        assert any(s.tool == "find_callers" for s in out.suggestions)

    def test_falls_back_to_seen_chunk_ids(self, patch_graph: None) -> None:
        from stropha.server import suggest_next_tool_mcp

        app = _make_app(
            {"seen-1": _Hit(chunk_id="seen-1", symbol="Bar", kind="function")},
            seen=["seen-1"],
        )
        out = suggest_next_tool_mcp("Bar", None, ctx=_make_ctx(app))
        assert out.based_on_hits == 1

    def test_missing_chunk_skipped(self, patch_graph: None) -> None:
        from stropha.server import suggest_next_tool_mcp

        app = _make_app({"c1": _Hit(chunk_id="c1")})
        out = suggest_next_tool_mcp(
            "q", ["c1", "ghost"], ctx=_make_ctx(app),
        )
        assert out.based_on_hits == 1

    def test_graph_not_loaded_disables_graph_suggestions(
        self, patch_graph: None,
    ) -> None:
        from stropha.server import suggest_next_tool_mcp

        app = _make_app(
            {"c1": _Hit(symbol="FooClass", kind="class")},
            graph_loaded_val=False,
        )
        out = suggest_next_tool_mcp("FooClass", ["c1"], ctx=_make_ctx(app))
        assert out.graph_loaded is False
        assert all(s.tool != "find_callers" for s in out.suggestions)

    def test_storage_lookup_exception_swallowed(
        self, patch_graph: None,
    ) -> None:
        """When lookup_by_chunk_id raises, the wrapper must drop the
        offender, not crash the request."""
        from stropha.server import suggest_next_tool_mcp

        app = _make_app({}, lookup_raises=True)
        out = suggest_next_tool_mcp("q", ["c1", "c2"], ctx=_make_ctx(app))
        assert out.based_on_hits == 0

    def test_max_suggestions_clamped(self, patch_graph: None) -> None:
        from stropha.server import suggest_next_tool_mcp

        app = _make_app({"c1": _Hit(symbol="FooClass", kind="class")})
        # max_suggestions=0 must clamp up to 1.
        out = suggest_next_tool_mcp(
            "FooClass", ["c1"], max_suggestions=0, ctx=_make_ctx(app),
        )
        assert len(out.suggestions) >= 1
        # max_suggestions=999 must clamp down to 20.
        out2 = suggest_next_tool_mcp(
            "FooClass", ["c1"], max_suggestions=999, ctx=_make_ctx(app),
        )
        assert len(out2.suggestions) <= 20
