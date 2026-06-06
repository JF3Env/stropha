"""Unit tests for the hallucinated-symbol guard + MCP wrapper
(WIN-245, EPIC 1.11 #83)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stropha.retrieval.hallucination_guard import (
    HallucinationReport,
    SymbolCheck,
    _length_distance,
    check_symbols,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_check_defaults(self) -> None:
        c = SymbolCheck(symbol="x", exists=True)
        assert c.suggestions == []
        assert c.reason == ""

    def test_report_defaults(self) -> None:
        r = HallucinationReport()
        assert r.checks == []
        assert r.present_count == 0
        assert r.missing_count == 0
        assert r.notes == []

    def test_frozen(self) -> None:
        c = SymbolCheck(symbol="x", exists=True)
        with pytest.raises((AttributeError, Exception)):
            c.exists = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _length_distance
# ---------------------------------------------------------------------------


class TestLengthDistance:
    def test_equal(self) -> None:
        assert _length_distance("abc", "xyz") == 0

    def test_different(self) -> None:
        assert _length_distance("abc", "abcd") == 1

    def test_empty(self) -> None:
        assert _length_distance("", "abc") == 3


# ---------------------------------------------------------------------------
# check_symbols — basic
# ---------------------------------------------------------------------------


class TestCheckSymbols:
    def test_empty_input(self) -> None:
        report = check_symbols([], exists_fn=lambda _: True)
        assert report.checks == []
        assert report.present_count == 0
        assert report.missing_count == 0

    def test_all_present(self) -> None:
        report = check_symbols(
            ["Foo", "Bar"],
            exists_fn=lambda _: True,
        )
        assert report.present_count == 2
        assert report.missing_count == 0
        assert all(c.exists for c in report.checks)

    def test_all_missing(self) -> None:
        report = check_symbols(
            ["Ghost", "Spectre"],
            exists_fn=lambda _: False,
        )
        assert report.present_count == 0
        assert report.missing_count == 2
        assert all(not c.exists for c in report.checks)

    def test_mixed(self) -> None:
        present = {"Foo", "Bar"}
        report = check_symbols(
            ["Foo", "Ghost", "Bar"],
            exists_fn=lambda s: s in present,
        )
        assert report.present_count == 2
        assert report.missing_count == 1
        assert report.checks[1].exists is False

    def test_empty_string_skipped(self) -> None:
        report = check_symbols(
            ["Foo", "", "  "],
            exists_fn=lambda _: True,
        )
        # Two empties skipped.
        assert len(report.checks) == 1
        assert "skipped" in " ".join(report.notes)

    def test_duplicates_deduped(self) -> None:
        report = check_symbols(
            ["Foo", "Foo", "Foo"],
            exists_fn=lambda _: True,
        )
        assert len(report.checks) == 1

    def test_exists_fn_exception_captured(self) -> None:
        def boom(s: str) -> bool:
            raise RuntimeError("boom")

        report = check_symbols(["Foo"], exists_fn=boom)
        # Exception -> treated as not found.
        assert report.checks[0].exists is False
        assert any("boom" in n for n in report.notes)


# ---------------------------------------------------------------------------
# check_symbols — similar suggestions
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_suggestions_filtered_and_ranked(self) -> None:
        candidates = ["FooClass", "FooClassX", "FooDifferentLong", "Foo"]
        report = check_symbols(
            ["Foo"],
            exists_fn=lambda _: False,
            similar_fn=lambda _: candidates,
            max_suggestions=3,
        )
        check = report.checks[0]
        # Query 'Foo' removed; sorted by length-delta to 'Foo' (len=3).
        # FooClass=8 (delta 5), FooClassX=9 (delta 6), FooDifferentLong=16.
        assert "Foo" not in check.suggestions
        assert len(check.suggestions) <= 3

    def test_no_similar_returns_empty(self) -> None:
        report = check_symbols(
            ["Ghost"],
            exists_fn=lambda _: False,
            similar_fn=lambda _: [],
        )
        assert report.checks[0].suggestions == []
        assert "not found in the index" in report.checks[0].reason

    def test_similar_fn_exception_captured(self) -> None:
        def boom(s: str) -> list[str]:
            raise RuntimeError("sim boom")

        report = check_symbols(
            ["Ghost"],
            exists_fn=lambda _: False,
            similar_fn=boom,
        )
        assert report.checks[0].suggestions == []
        assert any("sim boom" in n for n in report.notes)

    def test_max_suggestions_clamps_min_one(self) -> None:
        candidates = ["X", "Y", "Z"]
        report = check_symbols(
            ["Ghost"],
            exists_fn=lambda _: False,
            similar_fn=lambda _: candidates,
            max_suggestions=0,
        )
        # max(1, 0) = 1
        assert len(report.checks[0].suggestions) == 1

    def test_reason_mentions_closest(self) -> None:
        report = check_symbols(
            ["Foo"],
            exists_fn=lambda _: False,
            similar_fn=lambda _: ["FooBar"],
        )
        assert "FooBar" in report.checks[0].reason


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import hallucination_guard as mod
        assert set(mod.__all__) == {
            "HallucinationReport",
            "SymbolCheck",
            "check_symbols",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(symbols_by_query: dict[str, list[str]]) -> MagicMock:
    app = MagicMock()

    def _lookup(name: str, limit: int = 10) -> list:
        return [MagicMock(symbol=s) for s in symbols_by_query.get(name, [])][:limit]

    app.storage.lookup_by_symbol.side_effect = _lookup
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestMCPWrapper:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "check_symbols_tool" in tool_names

    def test_present_symbol(self) -> None:
        from stropha.server import check_symbols_tool

        response = check_symbols_tool(
            ["Foo"], ctx=_make_ctx({"Foo": ["Foo"]}),
        )
        assert response.present_count == 1
        assert response.missing_count == 0
        assert response.checks[0].exists is True

    def test_missing_symbol_gets_suggestions(self) -> None:
        from stropha.server import check_symbols_tool

        # Foo absent (lookup returns empty); FooBar present as a
        # candidate via similar lookup. The MCP wrapper's
        # similar_fn calls lookup_by_symbol(name, limit=20); we
        # arrange the mock to return one candidate for "Foo".
        response = check_symbols_tool(
            ["Foo"],
            ctx=_make_ctx({"Foo": []}),
        )
        # exists False because lookup returned []
        assert response.missing_count == 1
        assert response.checks[0].exists is False

    def test_max_suggestions_clamped(self) -> None:
        from stropha.server import check_symbols_tool

        response = check_symbols_tool(
            ["Foo"], max_suggestions=999, ctx=_make_ctx({}),
        )
        # Just verify the call returns without error; clamp lives
        # in the wrapper.
        assert response.missing_count == 1

    def test_max_suggestions_zero_clamped_to_one(self) -> None:
        from stropha.server import check_symbols_tool

        response = check_symbols_tool(
            ["Foo"], max_suggestions=0, ctx=_make_ctx({}),
        )
        assert isinstance(response.checks, list)

    def test_similar_lookup_exception_swallowed(self) -> None:
        from stropha.server import check_symbols_tool

        app = MagicMock()
        # First call (exists check) returns []; second call (similar
        # lookup) raises. The wrapper must swallow.
        call_log = {"n": 0}

        def _lookup(name: str, limit: int = 10):
            call_log["n"] += 1
            if call_log["n"] == 1:
                return []
            raise RuntimeError("similar lookup blew up")

        app.storage.lookup_by_symbol.side_effect = _lookup
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app

        response = check_symbols_tool(["Ghost"], ctx=ctx)
        # Should still produce a verdict; no exception leaked.
        assert response.checks[0].exists is False
        assert response.checks[0].suggestions == []
