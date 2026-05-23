"""Unit tests for contract-test surfacing + MCP wrapper
(WIN-251, EPIC 1.11 #89)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stropha.retrieval.contract_tests import (
    ContractTest,
    ContractTestReport,
    classify_test,
    surface_contract_tests,
)


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_contract_test_defaults(self) -> None:
        t = ContractTest(path="x.py")
        assert t.symbol == ""
        assert t.signals == []

    def test_report_defaults(self) -> None:
        r = ContractTestReport()
        assert r.target_symbol == ""
        assert r.contract_tests == []
        assert r.other_tests == []
        assert r.signal_counts == {}
        assert r.notes == []

    def test_frozen(self) -> None:
        t = ContractTest(path="x.py")
        with pytest.raises((AttributeError, Exception)):
            t.path = "y.py"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# classify_test
# ---------------------------------------------------------------------------


class TestClassifyTest:
    def test_no_signals(self) -> None:
        assert classify_test({"source_file": "src/foo.py", "symbol": "test_x"}) == []

    def test_path_signal_contracts_dir(self) -> None:
        sigs = classify_test({
            "source_file": "tests/contracts/foo_test.py",
            "symbol": "test_x",
        })
        assert "path" in sigs

    def test_path_signal_spec_suffix(self) -> None:
        sigs = classify_test({
            "source_file": "tests/foo_spec.py",
            "symbol": "test_x",
        })
        assert "path" in sigs

    def test_path_signal_specs_dir(self) -> None:
        sigs = classify_test({
            "source_file": "tests/specs/x.py",
            "symbol": "test_x",
        })
        assert "path" in sigs

    def test_symbol_signal_contract(self) -> None:
        sigs = classify_test({
            "source_file": "tests/x.py",
            "symbol": "test_contract_foo",
        })
        assert "symbol" in sigs

    def test_symbol_signal_interface(self) -> None:
        sigs = classify_test({"symbol": "test_interface_x"})
        assert "symbol" in sigs

    def test_symbol_signal_behavior(self) -> None:
        sigs = classify_test({"symbol": "test_behavior_x"})
        assert "symbol" in sigs

    def test_tag_signal(self) -> None:
        sigs = classify_test({
            "symbol": "test_x",
            "tags": ["unit", "contract"],
        })
        assert "tag" in sigs

    def test_annotations_signal(self) -> None:
        """``annotations`` is treated the same as ``tags`` — looks for
        a literal lowercase ``"contract"`` entry."""
        sigs = classify_test({
            "symbol": "test_x",
            "annotations": ["unit", "Contract"],
        })
        assert "tag" in sigs

    def test_tags_non_list_ignored(self) -> None:
        """Non-list tags must not crash the classifier."""
        sigs = classify_test({
            "symbol": "test_x",
            "tags": "not-a-list",
        })
        assert "tag" not in sigs

    def test_multiple_signals(self) -> None:
        sigs = classify_test({
            "source_file": "tests/contracts/api_spec.py",
            "symbol": "test_contract_api",
            "tags": ["contract"],
        })
        assert set(sigs) == {"path", "symbol", "tag"}

    def test_rel_path_fallback(self) -> None:
        sigs = classify_test({
            "rel_path": "tests/contracts/x.py",
            "symbol": "t",
        })
        assert "path" in sigs

    def test_label_fallback(self) -> None:
        sigs = classify_test({"label": "test_api_x"})
        assert "symbol" in sigs


# ---------------------------------------------------------------------------
# surface_contract_tests
# ---------------------------------------------------------------------------


class TestSurfaceContractTests:
    def test_empty_symbol(self) -> None:
        report = surface_contract_tests(
            "", tests_fn=lambda s: {"tests": []},
        )
        assert report.target_symbol == ""
        assert any("empty symbol" in n for n in report.notes)

    def test_tests_fn_exception_captured(self) -> None:
        def boom(_: str) -> dict:
            raise RuntimeError("test boom")

        report = surface_contract_tests("Foo", tests_fn=boom)
        assert report.contract_tests == []
        assert any("test boom" in n for n in report.notes)

    def test_partitions_correctly(self) -> None:
        tests = [
            {"source_file": "tests/contracts/foo.py", "symbol": "test_api"},
            {"source_file": "tests/internal.py", "symbol": "test_helper"},
        ]
        report = surface_contract_tests(
            "Foo", tests_fn=lambda s: {"tests": tests},
        )
        assert len(report.contract_tests) == 1
        assert len(report.other_tests) == 1
        assert report.contract_tests[0].path == "tests/contracts/foo.py"

    def test_signal_counts_aggregated(self) -> None:
        tests = [
            {"source_file": "tests/contracts/a.py", "symbol": "test_a"},
            {"source_file": "tests/x.py", "symbol": "test_contract_b"},
            {"source_file": "tests/x.py", "symbol": "test_c", "tags": ["contract"]},
        ]
        report = surface_contract_tests(
            "Foo", tests_fn=lambda s: {"tests": tests},
        )
        assert report.signal_counts["path"] == 1
        assert report.signal_counts["symbol"] == 1
        assert report.signal_counts["tag"] == 1

    def test_non_dict_entries_skipped(self) -> None:
        tests = [
            {"source_file": "tests/contracts/a.py", "symbol": "test_a"},
            "junk-string",  # not a dict
            {"source_file": "tests/x.py", "symbol": "test_x"},
        ]
        report = surface_contract_tests(
            "Foo", tests_fn=lambda s: {"tests": tests},
        )
        assert len(report.contract_tests) + len(report.other_tests) == 2

    def test_empty_tests_payload(self) -> None:
        report = surface_contract_tests(
            "Foo", tests_fn=lambda s: {"tests": []},
        )
        assert report.contract_tests == []
        assert report.other_tests == []

    def test_missing_tests_key(self) -> None:
        report = surface_contract_tests(
            "Foo", tests_fn=lambda s: {},
        )
        assert report.contract_tests == []
        assert report.other_tests == []


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import contract_tests as mod
        assert set(mod.__all__) == {
            "ContractTest",
            "ContractTestReport",
            "classify_test",
            "surface_contract_tests",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    app = MagicMock()
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestContractTestsMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "contract_tests_tool" in tool_names

    def test_graph_not_loaded(self) -> None:
        from stropha.server import contract_tests_tool

        with patch("stropha.server.graph_loaded", return_value=False):
            response = contract_tests_tool("Foo", ctx=_make_ctx())
        assert response.graph_loaded is False
        assert response.contract_tests == []

    def test_graph_loaded_partition(self) -> None:
        from stropha.server import contract_tests_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": [
                    {"source_file": "tests/contracts/x.py",
                     "symbol": "test_api_x"},
                    {"source_file": "tests/internal.py",
                     "symbol": "test_helper"},
                ]},
        ):
            response = contract_tests_tool("Foo", ctx=_make_ctx())
        assert response.graph_loaded is True
        assert len(response.contract_tests) == 1
        assert len(response.other_tests) == 1
        # The contract test has both path + symbol signals.
        assert set(response.contract_tests[0].signals) >= {"path", "symbol"}

    def test_empty_tests(self) -> None:
        from stropha.server import contract_tests_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_tests_for",
                return_value={"tests": []},
        ):
            response = contract_tests_tool("Foo", ctx=_make_ctx())
        assert response.contract_tests == []
        assert response.other_tests == []
