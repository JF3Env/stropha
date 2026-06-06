"""Unit tests for the post-edit blast-radius report + MCP wrapper
(WIN-246, EPIC 1.11 #84)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stropha.retrieval.blast_radius import (
    BlastRadiusReport,
    _format_summary,
    _merge_unique,
    _node_key,
    _safe_fetch,
    compute_blast_radius,
)

# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_defaults(self) -> None:
        r = BlastRadiusReport()
        assert r.changed_symbols == []
        assert r.callers == []
        assert r.tests == []
        assert r.related == []
        assert r.rationale == []
        assert r.per_symbol == {}
        assert r.notes == []
        assert r.summary == ""

    def test_frozen(self) -> None:
        r = BlastRadiusReport()
        with pytest.raises((AttributeError, Exception)):
            r.summary = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _node_key
# ---------------------------------------------------------------------------


class TestNodeKey:
    def test_node_id_wins(self) -> None:
        assert _node_key({"node_id": "n1", "chunk_id": "c1"}) == "node:n1"

    def test_chunk_id_fallback(self) -> None:
        assert _node_key({"chunk_id": "c1"}) == "node:c1"

    def test_symbol_path_fallback(self) -> None:
        assert _node_key({"symbol": "Foo", "rel_path": "x.py"}) == "pair:Foo|x.py"

    def test_source_file_fallback(self) -> None:
        assert _node_key({"symbol": "Foo", "source_file": "x.py"}) == "pair:Foo|x.py"

    def test_empty_node(self) -> None:
        assert _node_key({}) == "pair:|"


# ---------------------------------------------------------------------------
# _safe_fetch
# ---------------------------------------------------------------------------


class TestSafeFetch:
    def test_returns_list_when_key_present(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(
            lambda s: {"callers": [{"a": 1}, {"b": 2}]},
            "Foo", "callers", notes,
        )
        assert out == [{"a": 1}, {"b": 2}]
        assert notes == []

    def test_missing_key_returns_empty(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(lambda s: {}, "Foo", "callers", notes)
        assert out == []
        assert notes == []

    def test_non_list_value_returns_empty(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(
            lambda s: {"callers": "not-a-list"},
            "Foo", "callers", notes,
        )
        assert out == []

    def test_non_dict_items_filtered(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(
            lambda s: {"callers": [{"a": 1}, "not-a-dict", {"b": 2}]},
            "Foo", "callers", notes,
        )
        assert out == [{"a": 1}, {"b": 2}]

    def test_exception_captured_as_note(self) -> None:
        notes: list[str] = []

        def boom(_: str) -> dict:
            raise RuntimeError("bad")

        out = _safe_fetch(boom, "Foo", "callers", notes)
        assert out == []
        assert any("bad" in n for n in notes)

    def test_function_without_name_still_handled(self) -> None:
        """Lambda has __name__='<lambda>' — should not crash."""
        notes: list[str] = []

        def make_boom():
            class Caller:
                def __call__(self, s: str) -> dict:
                    raise RuntimeError("explode")
            return Caller()

        callable_obj = make_boom()
        out = _safe_fetch(callable_obj, "Foo", "callers", notes)
        assert out == []
        assert any("explode" in n for n in notes)


# ---------------------------------------------------------------------------
# _merge_unique
# ---------------------------------------------------------------------------


class TestMergeUnique:
    def test_appends_new(self) -> None:
        target: list[dict[str, Any]] = []
        seen: set[str] = set()
        added = _merge_unique(target, [{"node_id": "a"}, {"node_id": "b"}], seen)
        assert added == 2
        assert len(target) == 2

    def test_dedup_by_key(self) -> None:
        target: list[dict[str, Any]] = [{"node_id": "a"}]
        seen = {"node:a"}
        added = _merge_unique(
            target, [{"node_id": "a"}, {"node_id": "b"}], seen,
        )
        assert added == 1
        assert len(target) == 2


# ---------------------------------------------------------------------------
# compute_blast_radius
# ---------------------------------------------------------------------------


class TestComputeBlastRadius:
    def test_empty_symbols(self) -> None:
        report = compute_blast_radius(
            [],
            callers_fn=lambda s: {"callers": []},
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
        )
        assert report.changed_symbols == []
        assert report.callers == []
        assert "nothing to assess" in report.summary

    def test_dedup_input_symbols(self) -> None:
        report = compute_blast_radius(
            ["Foo", "Foo", "Foo"],
            callers_fn=lambda s: {"callers": [{"node_id": f"c-{s}"}]},
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
        )
        assert report.changed_symbols == ["Foo"]
        # Only one caller because dedup prevented multiple calls.
        assert len(report.callers) == 1

    def test_empty_strings_skipped(self) -> None:
        report = compute_blast_radius(
            ["Foo", "", "  "],
            callers_fn=lambda s: {"callers": []},
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
        )
        assert report.changed_symbols == ["Foo"]
        assert any("skipped" in n for n in report.notes)

    def test_merges_across_symbols(self) -> None:
        # FooClass and BarClass share one caller (node_id="shared").
        def callers_fn(s: str) -> dict:
            if s == "FooClass":
                return {"callers": [{"node_id": "shared"}, {"node_id": "foo-only"}]}
            return {"callers": [{"node_id": "shared"}, {"node_id": "bar-only"}]}

        report = compute_blast_radius(
            ["FooClass", "BarClass"],
            callers_fn=callers_fn,
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
        )
        ids = {c["node_id"] for c in report.callers}
        assert ids == {"shared", "foo-only", "bar-only"}
        # per_symbol shows Foo added 2 (shared + foo-only), Bar
        # added 1 (bar-only — shared was already seen).
        assert report.per_symbol["FooClass"]["callers"] == 2
        assert report.per_symbol["BarClass"]["callers"] == 1

    def test_rationale_section_disabled_when_callable_none(self) -> None:
        report = compute_blast_radius(
            ["Foo"],
            callers_fn=lambda s: {"callers": []},
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
            rationale_fn=None,
        )
        assert report.rationale == []
        assert report.per_symbol["Foo"]["rationale"] == 0

    def test_rationale_section_populated(self) -> None:
        report = compute_blast_radius(
            ["Foo"],
            callers_fn=lambda s: {"callers": []},
            tests_fn=lambda s: {"tests": []},
            related_fn=lambda s: {"related": []},
            rationale_fn=lambda s: {"rationale": [{"node_id": "adr1"}]},
        )
        assert len(report.rationale) == 1
        assert report.per_symbol["Foo"]["rationale"] == 1

    def test_summary_includes_counts(self) -> None:
        report = compute_blast_radius(
            ["Foo"],
            callers_fn=lambda s: {"callers": [{"node_id": "c1"}]},
            tests_fn=lambda s: {"tests": [{"node_id": "t1"}]},
            related_fn=lambda s: {"related": [{"node_id": "r1"}]},
        )
        assert "1 caller" in report.summary
        assert "1 test" in report.summary
        assert "Foo" in report.summary


# ---------------------------------------------------------------------------
# _format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_no_symbols(self) -> None:
        assert "nothing" in _format_summary([], [], [], [], [])

    def test_short_list(self) -> None:
        s = _format_summary(["A", "B"], [{"x": 1}], [], [], [])
        assert "A, B" in s

    def test_long_list_truncated(self) -> None:
        s = _format_summary(["A", "B", "C", "D"], [], [], [], [])
        assert "A + 3 more" in s

    def test_rationale_included(self) -> None:
        s = _format_summary(["A"], [], [], [], [{"x": 1}])
        assert "rationale" in s


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import blast_radius as mod
        assert set(mod.__all__) == {
            "BlastRadiusReport",
            "compute_blast_radius",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    app = MagicMock()
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestBlastRadiusMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "blast_radius_tool" in tool_names

    def test_graph_not_loaded(self) -> None:
        from stropha.server import blast_radius_tool

        with patch("stropha.server.graph_loaded", return_value=False):
            response = blast_radius_tool(["Foo"], ctx=_make_ctx())
        assert response.graph_loaded is False
        assert response.changed_symbols == ["Foo"]
        assert "graphify" in response.summary

    def test_graph_loaded_no_rationale_edges(self) -> None:
        from stropha.server import blast_radius_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch("stropha.server.has_rationale_edges", return_value=False), \
             patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": [{"node_id": "c1"}]},
        ), patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": []},
        ), patch(
                "stropha.retrieval.graph.find_related",
                return_value={"related": []},
        ):
            response = blast_radius_tool(["Foo"], ctx=_make_ctx())
        assert response.graph_loaded is True
        assert response.rationale == []
        assert len(response.callers) == 1

    def test_graph_loaded_with_rationale(self) -> None:
        from stropha.server import blast_radius_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch("stropha.server.has_rationale_edges", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": []},
        ), patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": []},
        ), patch(
                "stropha.retrieval.graph.find_related",
                return_value={"related": []},
        ), patch(
                "stropha.retrieval.graph.find_rationale",
                return_value={"rationale": [{"node_id": "adr-1"}]},
        ):
            response = blast_radius_tool(["Foo"], ctx=_make_ctx())
        assert len(response.rationale) == 1

    def test_per_symbol_response_shape(self) -> None:
        from stropha.server import blast_radius_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch("stropha.server.has_rationale_edges", return_value=False), \
             patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": [{"node_id": "c1"}]},
        ), patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": []},
        ), patch(
                "stropha.retrieval.graph.find_related",
                return_value={"related": []},
        ):
            response = blast_radius_tool(["Foo"], ctx=_make_ctx())
        assert "Foo" in response.per_symbol
        assert response.per_symbol["Foo"].callers == 1
        assert response.per_symbol["Foo"].rationale == 0
