"""Unit tests for the investigation planner (WIN-232, EPIC 1.10 #23)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stropha.models import SearchHit
from stropha.retrieval.investigation import (
    InvestigationPlan,
    InvestigationStep,
    plan_investigation,
)


def _hit(
    chunk_id: str = "c1",
    symbol: str | None = "FsrsCalc.update",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/fsrs.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="def update(): pass",
        language="python",
        kind="function",
        symbol=symbol,
    )


def _engine(hits: list[SearchHit]) -> MagicMock:
    eng = MagicMock()
    eng.search.return_value = hits
    return eng


class TestPlanInvestigation:
    def test_broad_scope_supports_legacy_search_signature(self) -> None:
        class LegacySearch:
            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                assert query == "task"
                assert top_k == 5
                return [_hit(symbol="Foo")]

        plan = plan_investigation("task", search_engine=LegacySearch())

        assert plan.primary_symbols == ["Foo"]
        assert all(step.args["include_other_repos"] is True for step in plan.steps)

    def test_concrete_repo_id_is_always_passed(self) -> None:
        engine = _engine([])

        plan_investigation("task", search_engine=engine, repo_id=7)

        assert engine.search.call_args.kwargs["repo_id"] == 7

    def test_first_step_is_search_code(self) -> None:
        plan = plan_investigation(
            "how does mastery decay work",
            search_engine=_engine([_hit()]),
            storage=None,
        )
        assert plan.steps[0].tool == "search_code"
        assert plan.steps[0].order == 1
        assert plan.steps[0].args["query"] == "how does mastery decay work"

    def test_callers_step_emitted_per_symbol(self) -> None:
        hits = [
            _hit("c1", symbol="Foo.bar"),
            _hit("c2", symbol="Baz.qux"),
        ]
        plan = plan_investigation(
            "task", search_engine=_engine(hits), storage=None,
        )
        caller_steps = [s for s in plan.steps if s.tool == "find_callers"]
        assert len(caller_steps) == 2
        symbols = {s.args["symbol"] for s in caller_steps}
        assert symbols == {"Foo.bar", "Baz.qux"}

    def test_tests_step_emitted_per_symbol(self) -> None:
        plan = plan_investigation(
            "task", search_engine=_engine([_hit(symbol="Foo.bar")]),
            storage=None,
        )
        test_steps = [s for s in plan.steps if s.tool == "find_tests_for"]
        assert len(test_steps) == 1
        assert test_steps[0].args["symbol"] == "Foo.bar"

    def test_no_symbols_means_no_followup_steps(self) -> None:
        plan = plan_investigation(
            "task",
            search_engine=_engine([_hit(symbol=None)]),
            storage=None,
        )
        # Only the search_code step survives.
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "search_code"
        # And a note explains the omission.
        assert any("No symbols" in n for n in plan.notes)

    def test_rationale_skipped_when_no_storage(self) -> None:
        plan = plan_investigation(
            "task", search_engine=_engine([_hit(symbol="Foo")]),
            storage=None,
        )
        rat_steps = [s for s in plan.steps if s.tool == "find_rationale"]
        assert rat_steps == []
        assert any("Graph not loaded" in n for n in plan.notes)

    def test_rationale_emitted_when_graph_loaded(self) -> None:
        plan_with_graph_storage = MagicMock()
        with patch(
            "stropha.retrieval.graph.graph_loaded", return_value=True,
        ):
            plan = plan_investigation(
                "task",
                search_engine=_engine([_hit(symbol="Foo")]),
                storage=plan_with_graph_storage,
            )
        rat_steps = [s for s in plan.steps if s.tool == "find_rationale"]
        assert len(rat_steps) == 1
        assert rat_steps[0].args["symbol"] == "Foo"

    def test_rationale_skipped_when_graph_loader_raises(self) -> None:
        with patch(
            "stropha.retrieval.graph.graph_loaded",
            side_effect=RuntimeError("boom"),
        ):
            plan = plan_investigation(
                "task",
                search_engine=_engine([_hit(symbol="Foo")]),
                storage=MagicMock(),
            )
        rat_steps = [s for s in plan.steps if s.tool == "find_rationale"]
        assert rat_steps == []

    def test_step_order_monotonic(self) -> None:
        plan = plan_investigation(
            "task",
            search_engine=_engine([
                _hit("c1", symbol="A"),
                _hit("c2", symbol="B"),
            ]),
            storage=None,
        )
        orders = [s.order for s in plan.steps]
        assert orders == sorted(orders)
        assert orders[0] == 1

    def test_distinct_symbols_only(self) -> None:
        """Duplicate symbols across primary hits don't emit duplicate
        plan steps."""
        plan = plan_investigation(
            "task",
            search_engine=_engine([
                _hit("c1", symbol="Foo"),
                _hit("c2", symbol="Foo"),
                _hit("c3", symbol="Foo"),
            ]),
            storage=None,
        )
        caller_steps = [s for s in plan.steps if s.tool == "find_callers"]
        assert len(caller_steps) == 1

    def test_include_callers_false_skips(self) -> None:
        plan = plan_investigation(
            "task",
            search_engine=_engine([_hit(symbol="Foo")]),
            storage=None,
            include_callers=False,
        )
        caller_steps = [s for s in plan.steps if s.tool == "find_callers"]
        assert caller_steps == []

    def test_include_tests_false_skips(self) -> None:
        plan = plan_investigation(
            "task",
            search_engine=_engine([_hit(symbol="Foo")]),
            storage=None,
            include_tests=False,
        )
        test_steps = [s for s in plan.steps if s.tool == "find_tests_for"]
        assert test_steps == []

    def test_include_rationale_false_skips(self) -> None:
        with patch(
            "stropha.retrieval.graph.graph_loaded", return_value=True,
        ):
            plan = plan_investigation(
                "task",
                search_engine=_engine([_hit(symbol="Foo")]),
                storage=MagicMock(),
                include_rationale=False,
            )
        rat_steps = [s for s in plan.steps if s.tool == "find_rationale"]
        assert rat_steps == []

    def test_empty_search_returns_minimal_plan(self) -> None:
        plan = plan_investigation(
            "task", search_engine=_engine([]), storage=None,
        )
        # Only the initial search_code step.
        assert len(plan.steps) == 1
        assert plan.primary_symbols == []


class TestInvestigationPlanResponse:
    """MCP tool surface tests."""

    def test_plan_investigation_mcp_returns_response(self) -> None:
        from stropha.server import plan_investigation as plan_tool

        app = MagicMock()
        app.search_engine.search.return_value = [_hit(symbol="Foo")]
        app.storage = MagicMock()
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app

        with patch(
            "stropha.retrieval.graph.graph_loaded", return_value=False,
        ):
            result = plan_tool("how does X work", ctx=ctx)
        assert result.task == "how does X work"
        assert result.steps
        assert result.steps[0].tool == "search_code"
        assert "Foo" in result.primary_symbols

    def test_tool_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "plan_investigation" in names


class TestDataclassImmutability:
    def test_plan_frozen(self) -> None:
        plan = InvestigationPlan(task="t")
        with pytest.raises((AttributeError, TypeError)):
            plan.task = "other"  # type: ignore[misc]

    def test_step_frozen(self) -> None:
        step = InvestigationStep(order=1, tool="x", purpose="y")
        with pytest.raises((AttributeError, TypeError)):
            step.order = 2  # type: ignore[misc]
