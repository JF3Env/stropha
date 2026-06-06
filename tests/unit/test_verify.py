"""Unit tests for claim verification + MCP wrapper (WIN-244, EPIC 1.11 #82)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stropha.retrieval.verify import (
    Claim,
    Verdict,
    _matches_symbol,
    _slice_original,
    parse_claim,
    verify_claim,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_claim_defaults(self) -> None:
        c = Claim(subject="x", predicate="calls")
        assert c.object is None
        assert c.raw_text == ""

    def test_verdict_defaults(self) -> None:
        v = Verdict(claim=Claim("x", "calls"), verified=True)
        assert v.reason == ""
        assert v.evidence == []
        assert v.alternatives == []

    def test_frozen(self) -> None:
        c = Claim(subject="x", predicate="calls")
        with pytest.raises((AttributeError, Exception)):
            c.predicate = "calls2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# parse_claim
# ---------------------------------------------------------------------------


class TestParseClaim:
    @pytest.mark.parametrize("text,subject,predicate,obj", [
        ("FooClass calls BarService", "FooClass", "calls", "BarService"),
        ("Foo invokes Bar", "Foo", "calls", "Bar"),
        ("Foo invoke Bar", "Foo", "calls", "Bar"),
        ("Foo uses Bar", "Foo", "references", "Bar"),
        ("Foo references Bar", "Foo", "references", "Bar"),
        ("Foo implements IBar", "Foo", "implements", "IBar"),
        ("Foo implement IBar", "Foo", "implements", "IBar"),
        ("Foo is called by Bar", "Foo", "called_by", "Bar"),
        ("Foo is a class", "Foo", "kind", "class"),
        ("Foo is an interface", "Foo", "kind", "interface"),
        ("Foo is a function", "Foo", "kind", "function"),
        ("Foo is in src/foo.py", "Foo", "in_file", "src/foo.py"),
        ("Foo is in file src/foo.py", "Foo", "in_file", "src/foo.py"),
        ("Foo lives in src/foo.py", "Foo", "in_file", "src/foo.py"),
    ])
    def test_recognised_patterns(
        self, text: str, subject: str, predicate: str, obj: str | None,
    ) -> None:
        claim = parse_claim(text)
        assert claim is not None
        assert claim.subject == subject
        assert claim.predicate == predicate
        assert claim.object == obj

    def test_has_tests_unary(self) -> None:
        for text in ("Foo has tests", "Foo has test", "Foo is tested"):
            claim = parse_claim(text)
            assert claim is not None
            assert claim.predicate == "has_tests"
            assert claim.object is None

    def test_empty_returns_none(self) -> None:
        assert parse_claim("") is None
        assert parse_claim("   ") is None

    def test_unknown_pattern_returns_none(self) -> None:
        assert parse_claim("this is gibberish") is None
        assert parse_claim("Foo flarbles Bar") is None

    def test_whitespace_collapsed(self) -> None:
        claim = parse_claim("Foo    calls   Bar")
        assert claim is not None
        assert claim.subject == "Foo"
        assert claim.object == "Bar"

    def test_case_preserved_in_subject_and_object(self) -> None:
        claim = parse_claim("FooClass calls BarService")
        assert claim is not None
        assert claim.subject == "FooClass"
        assert claim.object == "BarService"

    def test_slice_original_fallback(self) -> None:
        # _slice_original returns lower_token when no match (defensive).
        assert _slice_original("hello world", "missing") == "missing"


# ---------------------------------------------------------------------------
# _matches_symbol
# ---------------------------------------------------------------------------


class TestMatchesSymbol:
    def test_matches_on_symbol(self) -> None:
        assert _matches_symbol({"symbol": "FooClass"}, "FooClass") is True

    def test_matches_on_label(self) -> None:
        assert _matches_symbol({"label": "Foo Class"}, "foo class") is True

    def test_matches_on_node_id(self) -> None:
        assert _matches_symbol({"node_id": "pkg.Foo"}, "foo") is True

    def test_matches_on_qualified_name(self) -> None:
        assert _matches_symbol(
            {"qualified_name": "com.example.Foo"}, "Foo",
        ) is True

    def test_no_match_empty_target(self) -> None:
        assert _matches_symbol({"symbol": "Foo"}, "") is False

    def test_no_match_when_keys_missing(self) -> None:
        assert _matches_symbol({"other": "Foo"}, "Foo") is False


# ---------------------------------------------------------------------------
# verify_claim — predicate dispatch
# ---------------------------------------------------------------------------


def _const_fn(value: Any) -> Any:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return value
    return _fn


class TestHasTests:
    def test_with_tests_returns_true(self) -> None:
        claim = Claim(subject="Foo", predicate="has_tests")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({"callers": []}),
            related_fn=_const_fn({"related": [], "edges": []}),
            tests_fn=_const_fn({"tests": [{"symbol": "test_foo"}]}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is True
        assert "test" in verdict.reason

    def test_no_tests_returns_false(self) -> None:
        claim = Claim(subject="Foo", predicate="has_tests")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({"tests": []}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is False


class TestKind:
    def test_kind_matches(self) -> None:
        claim = Claim(subject="Foo", predicate="kind", object="class")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"kind": "class", "symbol": "Foo"}),
        )
        assert verdict.verified is True

    def test_kind_mismatch(self) -> None:
        claim = Claim(subject="Foo", predicate="kind", object="class")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"kind": "function", "symbol": "Foo"}),
        )
        assert verdict.verified is False
        assert any("actual_kind" in alt for alt in verdict.alternatives)

    def test_kind_node_missing_metadata(self) -> None:
        claim = Claim(subject="Foo", predicate="kind", object="class")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"symbol": "Foo"}),  # no kind
        )
        assert verdict.verified is False
        assert "no kind" in verdict.reason

    def test_unresolvable_subject(self) -> None:
        claim = Claim(subject="Ghost", predicate="kind", object="class")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is None


class TestInFile:
    def test_file_matches(self) -> None:
        claim = Claim(subject="Foo", predicate="in_file", object="foo.py")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"source_file": "src/foo.py", "symbol": "Foo"}),
        )
        assert verdict.verified is True

    def test_file_mismatch(self) -> None:
        claim = Claim(subject="Foo", predicate="in_file", object="bar.py")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"source_file": "src/foo.py", "symbol": "Foo"}),
        )
        assert verdict.verified is False
        assert any("actual_path" in alt for alt in verdict.alternatives)

    def test_no_file_metadata(self) -> None:
        claim = Claim(subject="Foo", predicate="in_file", object="bar.py")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"symbol": "Foo"}),
        )
        assert verdict.verified is False
        assert "no file metadata" in verdict.reason

    def test_unresolvable(self) -> None:
        claim = Claim(subject="Ghost", predicate="in_file", object="x.py")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is None

    def test_rel_path_fallback(self) -> None:
        """Node may carry rel_path instead of source_file."""
        claim = Claim(subject="Foo", predicate="in_file", object="foo.py")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn({"rel_path": "src/foo.py"}),
        )
        assert verdict.verified is True


class TestCalls:
    def test_calls_match(self) -> None:
        claim = Claim(subject="Foo", predicate="calls", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({
                "related": [{"symbol": "Bar", "node_id": "n1"}],
                "edges": [],
            }),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is True

    def test_calls_mismatch_returns_alternatives(self) -> None:
        claim = Claim(subject="Foo", predicate="calls", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({
                "related": [{"symbol": "Baz"}, {"symbol": "Qux"}],
                "edges": [],
            }),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is False
        assert len(verdict.alternatives) == 2

    def test_calls_no_edges_returns_false(self) -> None:
        claim = Claim(subject="Foo", predicate="calls", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({"related": [], "edges": []}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is False
        assert verdict.alternatives == []


class TestCalledBy:
    def test_called_by_match(self) -> None:
        claim = Claim(subject="Foo", predicate="called_by", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({"callers": [{"symbol": "Bar"}]}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is True

    def test_called_by_mismatch(self) -> None:
        claim = Claim(subject="Foo", predicate="called_by", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({"callers": [
                {"symbol": "Baz"}, {"symbol": "Qux"},
            ]}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is False
        assert len(verdict.alternatives) == 2


class TestReferencesImplements:
    def test_references_match(self) -> None:
        claim = Claim(subject="Foo", predicate="references", object="Bar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({
                "related": [{"symbol": "Bar"}], "edges": [],
            }),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is True

    def test_implements_match(self) -> None:
        claim = Claim(subject="Foo", predicate="implements", object="IBar")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({
                "related": [{"symbol": "IBar"}], "edges": [],
            }),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is True


class TestUnknownPredicate:
    def test_unknown_returns_none(self) -> None:
        claim = Claim(subject="X", predicate="flarbles", object="Y")
        verdict = verify_claim(
            claim,
            callers_fn=_const_fn({}),
            related_fn=_const_fn({}),
            tests_fn=_const_fn({}),
            resolve_fn=_const_fn(None),
        )
        assert verdict.verified is None
        assert "Unsupported predicate" in verdict.reason


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import verify as mod
        assert set(mod.__all__) == {
            "Claim", "Verdict", "parse_claim", "verify_claim",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(graph_loaded_val: bool = True) -> MagicMock:
    app = MagicMock()
    app._graph_loaded = graph_loaded_val
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestVerifyClaimMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "verify_claim_tool" in tool_names

    def test_unparseable_returns_parsed_false(self) -> None:
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=True):
            response = verify_claim_tool("gibberish input", ctx=_make_ctx())
        assert response.parsed is False
        assert response.verified is None

    def test_graph_unavailable_returns_none(self) -> None:
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=False):
            response = verify_claim_tool(
                "Foo calls Bar", ctx=_make_ctx(),
            )
        assert response.parsed is True
        assert response.verified is None
        assert "graphify" in response.reason

    def test_calls_claim_resolves(self) -> None:
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.find_related",
                return_value={
                    "related": [{"symbol": "Bar"}], "edges": [],
                },
        ):
            response = verify_claim_tool("Foo calls Bar", ctx=_make_ctx())
        assert response.parsed is True
        assert response.verified is True
        assert response.predicate == "calls"

    def test_kind_unresolvable(self) -> None:
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.resolve_symbol_to_node",
                return_value=None,
        ):
            response = verify_claim_tool("Ghost is a class", ctx=_make_ctx())
        assert response.parsed is True
        assert response.verified is None

    def test_kind_resolved_but_mismatch(self) -> None:
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.resolve_symbol_to_node",
                return_value="node-1",
        ), patch(
                "stropha.retrieval.graph._fetch_node_rows",
                return_value={"node-1": "fake-row"},
        ), patch(
                "stropha.retrieval.graph._hydrate_node",
                return_value=MagicMock(as_dict=lambda: {
                    "kind": "function", "symbol": "Foo",
                }),
        ):
            response = verify_claim_tool("Foo is a class", ctx=_make_ctx())
        assert response.parsed is True
        assert response.verified is False
        assert any("actual_kind" in alt for alt in response.alternatives)

    def test_resolve_fetch_rows_missing(self) -> None:
        """When the node id is returned but the row fetch can't find it,
        the resolve callable returns None — verdict is undetermined."""
        from stropha.server import verify_claim_tool

        with patch("stropha.server.graph_loaded", return_value=True), \
             patch(
                "stropha.retrieval.graph.resolve_symbol_to_node",
                return_value="node-1",
        ), patch(
                "stropha.retrieval.graph._fetch_node_rows",
                return_value={},  # node id absent
        ):
            response = verify_claim_tool(
                "Foo is in src/foo.py", ctx=_make_ctx(),
            )
        assert response.verified is None
