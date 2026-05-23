"""Unit tests for the retrieval DAG executor (WIN-233, EPIC 1.10 #24)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stropha.models import SearchHit
from stropha.retrieval.dag import execute_plan
from stropha.retrieval.investigation import (
    InvestigationPlan,
    InvestigationStep,
)


def _hit(chunk_id: str, symbol: str | None = "Foo.bar") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/x.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="def f(): pass",
        language="python",
        kind="function",
        symbol=symbol,
    )


def _engine(hits: list[SearchHit]) -> MagicMock:
    eng = MagicMock()
    eng.search.return_value = hits
    return eng


# ---------------------------------------------------------------------------
# execute_plan
# ---------------------------------------------------------------------------


class TestExecutePlan:
    def test_negative_budget_raises(self) -> None:
        plan = InvestigationPlan(task="t")
        with pytest.raises(ValueError, match="context_budget"):
            execute_plan(
                plan, search_engine=_engine([]),
                storage=None, context_budget=0,
            )

    def test_executes_search_code_step(self) -> None:
        plan = InvestigationPlan(
            task="task",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "task", "top_k": 5},
                ),
            ],
        )
        result = execute_plan(
            plan, search_engine=_engine([_hit("c1")]),
            storage=None, context_budget=4000,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 1
        assert primary.hits[0].chunk_id == "c1"

    def test_search_failure_recorded_in_bundle(self) -> None:
        plan = InvestigationPlan(
            task="task",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "task"},
                ),
            ],
        )
        eng = MagicMock()
        eng.search.side_effect = RuntimeError("boom")
        result = execute_plan(
            plan, search_engine=eng, storage=None, context_budget=4000,
        )
        # Empty primary section but the bundle includes the execution
        # error note.
        primary = next(s for s in result.sections if s.key == "primary")
        assert primary.hits == []
        assert "execution error" in result.bundle

    def test_callers_step_aggregated(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="callers",
                    args={"symbol": "Foo.bar"},
                ),
            ],
        )
        caller_hit = _hit("caller-1", symbol="Caller")
        with patch(
            "stropha.retrieval.graph.find_callers",
            return_value=[caller_hit],
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        callers = next(s for s in result.sections if s.key == "callers")
        assert len(callers.hits) == 1
        assert callers.hits[0].chunk_id == "caller-1"

    def test_tests_step_aggregated(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_tests_for", purpose="tests",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        test_hit = _hit("test-1", symbol="test_foo")
        with patch(
            "stropha.retrieval.graph.find_tests_for",
            return_value=[test_hit],
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        tests = next(s for s in result.sections if s.key == "tests")
        assert len(tests.hits) == 1

    def test_rationale_step_normalised_from_graph_dict(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_rationale", purpose="why",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={
                "symbol": "Foo",
                "rationale": [
                    {
                        "node_id": "adr-1", "label": "ADR-1",
                        "kind": "adr", "source_file": "docs/adr/0001.md",
                        "edge": {"confidence_score": 0.9},
                    },
                ],
            },
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        assert len(rationale.hits) == 1
        assert rationale.hits[0].symbol == "ADR-1"

    def test_storage_required_for_graph_steps(self) -> None:
        """Without storage the graph steps silently skip; the
        primary section still gets populated."""
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="x",
                    args={"symbol": "F"},
                ),
            ],
        )
        result = execute_plan(
            plan, search_engine=_engine([_hit("c1")]),
            storage=None, context_budget=4000,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        callers = next(s for s in result.sections if s.key == "callers")
        assert len(primary.hits) == 1
        assert len(callers.hits) == 0

    def test_step_missing_symbol_is_skipped(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="x",
                    args={},  # no symbol
                ),
            ],
        )
        with patch(
            "stropha.retrieval.graph.find_callers",
            return_value=[_hit("ghost")],
        ) as mock_fn:
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        # Library not called because the step skipped before dispatch.
        mock_fn.assert_not_called()
        callers = next(s for s in result.sections if s.key == "callers")
        assert callers.hits == []

    def test_unknown_tool_silently_ignored(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="unknown_future_tool",
                    purpose="x", args={"symbol": "F"},
                ),
            ],
        )
        result = execute_plan(
            plan, search_engine=_engine([_hit("c1")]),
            storage=MagicMock(), context_budget=4000,
        )
        # Other sections empty; primary still populated.
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 1

    def test_dedup_across_sections(self) -> None:
        """Same chunk_id appearing in primary AND callers shouldn't
        be double-counted."""
        shared = _hit("shared", symbol="Foo")
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="x",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        with patch(
            "stropha.retrieval.graph.find_callers", return_value=[shared],
        ):
            result = execute_plan(
                plan, search_engine=_engine([shared]),
                storage=MagicMock(), context_budget=4000,
            )
        primary = next(s for s in result.sections if s.key == "primary")
        callers = next(s for s in result.sections if s.key == "callers")
        # Shared appears in primary only (dedup wins).
        assert len(primary.hits) == 1
        assert len(callers.hits) == 0

    def test_search_code_step_dedups_its_own_duplicates(self) -> None:
        """Same chunk surfaced twice by the search engine collapses
        to one in the primary section."""
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
            ],
        )
        dup_a = _hit("dup")
        dup_b = _hit("dup")  # same chunk_id
        result = execute_plan(
            plan, search_engine=_engine([dup_a, dup_b]),
            storage=None, context_budget=4000,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 1

    def test_tests_step_dedups_against_callers(self) -> None:
        """Same chunk_id returned by both find_callers and
        find_tests_for collapses to one (callers wins by order)."""
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="x",
                    args={"symbol": "Foo"},
                ),
                InvestigationStep(
                    order=3, tool="find_tests_for", purpose="x",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        shared = _hit("shared", symbol="Foo")
        with patch(
            "stropha.retrieval.graph.find_callers",
            return_value=[shared],
        ), patch(
            "stropha.retrieval.graph.find_tests_for",
            return_value=[shared],  # same chunk_id
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        callers = next(s for s in result.sections if s.key == "callers")
        tests = next(s for s in result.sections if s.key == "tests")
        assert len(callers.hits) == 1  # shared went here first
        assert len(tests.hits) == 0    # then dedup'd out

    def test_rationale_step_dedups_against_earlier_sections(self) -> None:
        """Rationale converter produces synthetic SearchHits. If the
        synthetic chunk_id collides with one already seen
        (rationale:n1), the rationale section drops the duplicate."""
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_rationale", purpose="x",
                    args={"symbol": "Foo"},
                ),
                InvestigationStep(
                    order=3, tool="find_rationale", purpose="x",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        node = {
            "node_id": "adr-42", "label": "ADR-42",
            "kind": "adr", "edge": {"confidence_score": 0.9},
        }
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={"symbol": "Foo", "rationale": [node]},
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        # The synthetic chunk_id is `rationale:adr-42`; both steps
        # tried to add it but the second sees it in seen_chunk_ids.
        assert len(rationale.hits) == 1

    def test_graph_helper_exception_recorded(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
                InvestigationStep(
                    order=2, tool="find_callers", purpose="x",
                    args={"symbol": "Foo"},
                ),
            ],
        )
        with patch(
            "stropha.retrieval.graph.find_callers",
            side_effect=RuntimeError("graph not loaded"),
        ):
            result = execute_plan(
                plan, search_engine=_engine([_hit("c1")]),
                storage=MagicMock(), context_budget=4000,
            )
        # Bundle surfaces the error as a note; sections still
        # render whatever succeeded.
        assert "execution error" in result.bundle

    def test_plan_notes_appear_in_bundle(self) -> None:
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
            ],
            notes=["Graph not loaded — find_rationale steps omitted"],
        )
        result = execute_plan(
            plan, search_engine=_engine([_hit("c1")]),
            storage=None, context_budget=4000,
        )
        assert "Planner notes" in result.bundle
        assert "Graph not loaded" in result.bundle

    def test_empty_plan_returns_empty_bundle(self) -> None:
        plan = InvestigationPlan(task="empty")
        result = execute_plan(
            plan, search_engine=_engine([]),
            storage=None, context_budget=4000,
        )
        for sec in result.sections:
            assert sec.hits == []
        assert result.task == "empty"

    def test_budget_caps_total_packed(self) -> None:
        """A tight budget drops the tail. Verifies that spilled_count
        reflects the truncation."""
        plan = InvestigationPlan(
            task="t",
            steps=[
                InvestigationStep(
                    order=1, tool="search_code", purpose="primary",
                    args={"query": "t"},
                ),
            ],
        )
        big_hits = [
            SearchHit(
                chunk_id=f"c{i}",
                rel_path="x.py", start_line=1, end_line=10,
                score=0.5, rank=i,
                snippet="word " * 200,  # big snippet
                language="py", kind="fn", symbol=f"sym{i}",
            )
            for i in range(5)
        ]
        result = execute_plan(
            plan, search_engine=_engine(big_hits),
            storage=None, context_budget=100,
        )
        assert result.spilled_count > 0


# ---------------------------------------------------------------------------
# MCP execute_investigation tool
# ---------------------------------------------------------------------------


class TestMCPExecuteInvestigation:
    def test_returns_bundle_response(self) -> None:
        from stropha.server import execute_investigation

        app = MagicMock()
        app.search_engine.search.return_value = [_hit("c1")]
        app.storage = MagicMock()
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        with patch(
            "stropha.retrieval.graph.graph_loaded", return_value=False,
        ):
            result = execute_investigation("task", ctx=ctx)
        assert result.task == "task"
        assert result.bundle.startswith("# Context for: task")

    def test_zero_budget_returns_empty(self) -> None:
        from stropha.server import execute_investigation

        app = MagicMock()
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        result = execute_investigation(
            "task", context_budget=0, ctx=ctx,
        )
        assert result.bundle == ""
        assert result.total_tokens == 0

    def test_tool_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "execute_investigation" in names
