"""Unit tests for test-impact analysis + MCP wrapper
(WIN-248, EPIC 1.11 #86)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stropha.retrieval import test_impact as _ti
from stropha.retrieval.test_impact import (
    ImpactedTest,
    _build_run_command,
    _merge,
    _safe_fetch,
    _test_key,
    analyse_test_impact,
)

# Pytest tries to collect any module-level class prefixed with
# ``Test`` — keep ``TestImpactReport`` off the module namespace by
# referencing it via the module alias when needed.


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_impacted_test_defaults(self) -> None:
        t = ImpactedTest(path="x.py")
        assert t.symbol == ""
        assert t.via == []

    def test_report_defaults(self) -> None:
        r = _ti.TestImpactReport()
        assert r.changed_symbols == []
        assert r.tests == []
        assert r.per_symbol == {}
        assert r.notes == []
        assert r.run_command == ""

    def test_frozen(self) -> None:
        r = _ti.TestImpactReport()
        with pytest.raises((AttributeError, Exception)):
            r.run_command = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _test_key
# ---------------------------------------------------------------------------


class TestTestKey:
    def test_path_and_symbol(self) -> None:
        assert _test_key({
            "source_file": "tests/x.py", "symbol": "test_foo",
        }) == "tests/x.py::test_foo"

    def test_rel_path_fallback(self) -> None:
        assert _test_key({
            "rel_path": "tests/y.py", "symbol": "test_bar",
        }) == "tests/y.py::test_bar"

    def test_path_only(self) -> None:
        assert _test_key({"source_file": "tests/x.py"}) == "tests/x.py"

    def test_symbol_only(self) -> None:
        assert _test_key({"symbol": "test_foo"}) == "test_foo"

    def test_empty(self) -> None:
        assert _test_key({}) == "?"


# ---------------------------------------------------------------------------
# _safe_fetch
# ---------------------------------------------------------------------------


class TestSafeFetch:
    def test_returns_list(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(lambda s: {"tests": [{"a": 1}]}, "Foo", "tests", notes)
        assert out == [{"a": 1}]
        assert notes == []

    def test_missing_key(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(lambda s: {}, "Foo", "tests", notes)
        assert out == []

    def test_non_list(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(
            lambda s: {"tests": "string"}, "Foo", "tests", notes,
        )
        assert out == []

    def test_non_dict_items_filtered(self) -> None:
        notes: list[str] = []
        out = _safe_fetch(
            lambda s: {"tests": [{"a": 1}, "junk", {"b": 2}]},
            "Foo", "tests", notes,
        )
        assert out == [{"a": 1}, {"b": 2}]

    def test_exception_captured(self) -> None:
        notes: list[str] = []

        def boom(_: str) -> dict:
            raise RuntimeError("dead")

        out = _safe_fetch(boom, "Foo", "tests", notes)
        assert out == []
        assert any("dead" in n for n in notes)

    def test_callable_without_name(self) -> None:
        """Callable objects without __name__ shouldn't crash."""
        notes: list[str] = []

        class C:
            def __call__(self, s: str) -> dict:
                raise RuntimeError("dead")

        out = _safe_fetch(C(), "Foo", "tests", notes)
        assert out == []
        assert any("dead" in n for n in notes)


# ---------------------------------------------------------------------------
# _merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_new_entry(self) -> None:
        bucket: dict[str, dict[str, Any]] = {}
        _merge(bucket, {"source_file": "x.py", "symbol": "test_a"}, via="direct")
        assert "x.py::test_a" in bucket
        assert bucket["x.py::test_a"]["via"] == {"direct"}

    def test_existing_entry_merges_via(self) -> None:
        bucket: dict[str, dict[str, Any]] = {}
        _merge(bucket, {"source_file": "x.py", "symbol": "test_a"}, via="direct")
        _merge(bucket, {"source_file": "x.py", "symbol": "test_a"}, via="Caller")
        assert bucket["x.py::test_a"]["via"] == {"direct", "Caller"}

    def test_unknown_key_skipped(self) -> None:
        bucket: dict[str, dict[str, Any]] = {}
        _merge(bucket, {}, via="direct")  # _test_key returns "?"
        assert bucket == {}


# ---------------------------------------------------------------------------
# _build_run_command
# ---------------------------------------------------------------------------


class TestBuildRunCommand:
    def test_empty_tests(self) -> None:
        assert _build_run_command([], "pytest") == ""

    def test_pytest(self) -> None:
        tests = [
            ImpactedTest(path="tests/a.py"),
            ImpactedTest(path="tests/b.py"),
            ImpactedTest(path="tests/a.py", symbol="test_x"),
        ]
        # Paths deduplicated to {a.py, b.py}.
        cmd = _build_run_command(tests, "pytest")
        assert cmd == "pytest tests/a.py tests/b.py"

    def test_alternate_runner(self) -> None:
        tests = [ImpactedTest(path="foo_test.go")]
        cmd = _build_run_command(tests, "go test")
        assert cmd == "go test foo_test.go"

    def test_no_paths(self) -> None:
        tests = [ImpactedTest(path="", symbol="test_x")]
        assert _build_run_command(tests, "pytest") == ""


# ---------------------------------------------------------------------------
# analyse_test_impact
# ---------------------------------------------------------------------------


class TestAnalyseTestImpact:
    def test_empty_symbols(self) -> None:
        report = analyse_test_impact(
            [],
            tests_fn=lambda s: {"tests": []},
            callers_fn=lambda s: {"callers": []},
        )
        assert report.changed_symbols == []
        assert report.tests == []
        assert report.run_command == ""

    def test_dedup_input(self) -> None:
        calls: list[str] = []

        def tests_fn(s: str) -> dict:
            calls.append(s)
            return {"tests": [{"source_file": "tests/t.py", "symbol": "test_a"}]}

        report = analyse_test_impact(
            ["Foo", "Foo", "Foo"],
            tests_fn=tests_fn,
            callers_fn=lambda s: {"callers": []},
        )
        # Foo's tests fetched once (dedup), no callers.
        assert calls == ["Foo"]
        assert report.changed_symbols == ["Foo"]
        assert len(report.tests) == 1

    def test_empty_string_skipped(self) -> None:
        report = analyse_test_impact(
            ["Foo", "", " "],
            tests_fn=lambda s: {"tests": []},
            callers_fn=lambda s: {"callers": []},
        )
        assert report.changed_symbols == ["Foo"]
        assert any("skipped" in n for n in report.notes)

    def test_direct_tests_only(self) -> None:
        report = analyse_test_impact(
            ["Foo"],
            tests_fn=lambda s: {"tests": [
                {"source_file": "tests/a.py", "symbol": "test_a"},
            ]},
            callers_fn=lambda s: {"callers": [{"symbol": "Bar"}]},
            transitive_depth=0,  # skip indirect
        )
        # Bar's tests NOT fetched.
        assert len(report.tests) == 1
        assert all("direct" in t.via for t in report.tests)

    def test_transitive_tests(self) -> None:
        def tests_fn(s: str) -> dict:
            if s == "Foo":
                return {"tests": [{"source_file": "tests/foo_test.py",
                                   "symbol": "test_foo"}]}
            if s == "Bar":
                return {"tests": [{"source_file": "tests/bar_test.py",
                                   "symbol": "test_bar"}]}
            return {"tests": []}

        report = analyse_test_impact(
            ["Foo"],
            tests_fn=tests_fn,
            callers_fn=lambda s: {"callers": [{"symbol": "Bar"}]},
            transitive_depth=1,
        )
        # foo_test direct + bar_test indirect.
        paths = {t.path for t in report.tests}
        assert paths == {"tests/foo_test.py", "tests/bar_test.py"}
        # bar_test should have via "Bar".
        bar_t = next(t for t in report.tests if t.path == "tests/bar_test.py")
        assert "Bar" in bar_t.via

    def test_caller_without_symbol_skipped(self) -> None:
        """Caller lacking a symbol can't be queried for tests — skip."""
        report = analyse_test_impact(
            ["Foo"],
            tests_fn=lambda s: {"tests": []},
            callers_fn=lambda s: {"callers": [{"symbol": ""}]},
            transitive_depth=1,
        )
        assert report.tests == []

    def test_merge_across_inputs_dedupes(self) -> None:
        """Same test reached via two input symbols collapses, with
        both routes recorded in ``via``."""

        def tests_fn(s: str) -> dict:
            return {"tests": [{
                "source_file": "tests/shared.py", "symbol": "test_x",
            }]}

        report = analyse_test_impact(
            ["A", "B"],
            tests_fn=tests_fn,
            callers_fn=lambda s: {"callers": []},
            transitive_depth=0,
        )
        assert len(report.tests) == 1

    def test_per_symbol_unique_contribution(self) -> None:
        # A contributes test_a; B contributes test_b; both touch
        # test_shared via direct hit so test_shared is attributed to
        # A only (B's contribution count is 1, not 2).
        def tests_fn(s: str) -> dict:
            common = {"source_file": "tests/shared.py", "symbol": "test_shared"}
            if s == "A":
                return {"tests": [common, {"source_file": "tests/a.py", "symbol": "test_a"}]}
            return {"tests": [common, {"source_file": "tests/b.py", "symbol": "test_b"}]}

        report = analyse_test_impact(
            ["A", "B"],
            tests_fn=tests_fn,
            callers_fn=lambda s: {"callers": []},
            transitive_depth=0,
        )
        # A added 2 (shared + a), B added 1 (b only).
        assert report.per_symbol["A"] == 2
        assert report.per_symbol["B"] == 1

    def test_run_command_built(self) -> None:
        report = analyse_test_impact(
            ["Foo"],
            tests_fn=lambda s: {"tests": [
                {"source_file": "tests/foo_test.py", "symbol": "test_foo"},
            ]},
            callers_fn=lambda s: {"callers": []},
            runner="pytest",
        )
        assert report.run_command == "pytest tests/foo_test.py"


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import test_impact as mod
        assert set(mod.__all__) == {
            "ImpactedTest",
            "TestImpactReport",
            "analyse_test_impact",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    app = MagicMock()
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestTestImpactMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "test_impact_tool" in tool_names

    def test_graph_not_loaded(self) -> None:
        from stropha.server import test_impact_tool

        with patch("stropha.server.graph_loaded", return_value=False):
            response = test_impact_tool(["Foo"], ctx=_make_ctx())
        assert response.graph_loaded is False
        assert response.changed_symbols == ["Foo"]
        assert response.tests == []

    def test_basic_impact(self) -> None:
        from stropha.server import test_impact_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": [
                    {"source_file": "tests/foo_test.py", "symbol": "test_foo"},
                ]},
        ), patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": []},
        ):
            response = test_impact_tool(["Foo"], ctx=_make_ctx())
        assert response.graph_loaded is True
        assert len(response.tests) == 1
        assert "pytest" in response.run_command

    def test_runner_override(self) -> None:
        from stropha.server import test_impact_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": [
                    {"source_file": "foo_test.go", "symbol": "TestFoo"},
                ]},
        ), patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": []},
        ):
            response = test_impact_tool(
                ["Foo"], runner="go test", ctx=_make_ctx(),
            )
        assert response.run_command.startswith("go test ")

    def test_empty_runner_defaults_to_pytest(self) -> None:
        from stropha.server import test_impact_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": [
                    {"source_file": "t.py", "symbol": "test"},
                ]},
        ), patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": []},
        ):
            response = test_impact_tool(
                ["Foo"], runner="", ctx=_make_ctx(),
            )
        assert response.run_command.startswith("pytest ")

    def test_transitive_depth_negative_clamped(self) -> None:
        from stropha.server import test_impact_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": []},
        ), patch(
                "stropha.retrieval.graph.find_callers",
                return_value={"callers": []},
        ):
            # max(0, -1) = 0 -> direct only.
            response = test_impact_tool(
                ["Foo"], transitive_depth=-1, ctx=_make_ctx(),
            )
        assert response.graph_loaded is True
