"""Unit tests for A2A endpoint scaffold (WIN-258, EPIC 1.12 #95)."""

from __future__ import annotations

from typing import Any

import pytest

from stropha.api.a2a import (
    ALL_STATES,
    A2ATask,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INPUT_REQUIRED,
    STATE_SUBMITTED,
    STATE_WORKING,
    dispatch_task,
    needs_input,
    new_task,
    transition,
)
from stropha.api.context_v1 import ContextRequestV1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestStateConstants:
    def test_all_states_set(self) -> None:
        assert ALL_STATES == frozenset({
            STATE_SUBMITTED, STATE_WORKING, STATE_INPUT_REQUIRED,
            STATE_COMPLETED, STATE_FAILED,
        })


# ---------------------------------------------------------------------------
# new_task
# ---------------------------------------------------------------------------


class TestNewTask:
    def test_default_state_submitted(self) -> None:
        task = new_task(ContextRequestV1(query="x"))
        assert task.state == STATE_SUBMITTED
        assert task.task_id  # auto-generated UUID
        assert task.output is None
        assert task.error is None

    def test_custom_task_id(self) -> None:
        task = new_task(ContextRequestV1(query="x"), task_id="my-id")
        assert task.task_id == "my-id"

    def test_capabilities_advertised(self) -> None:
        task = new_task(ContextRequestV1(query="x"))
        assert "context_v1" in task.capabilities


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_submitted_to_working(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        out = transition(t, to_state=STATE_WORKING)
        assert out.state == STATE_WORKING

    def test_working_to_completed(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        out = transition(t, to_state=STATE_COMPLETED)
        assert out.state == STATE_COMPLETED

    def test_submitted_to_completed_rejected(self) -> None:
        """Skipping `working` violates the FSM."""
        t = new_task(ContextRequestV1(query="x"))
        out = transition(t, to_state=STATE_COMPLETED)
        assert out.state == STATE_FAILED
        assert "illegal transition" in (out.error or "")

    def test_completed_is_terminal(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        t = transition(t, to_state=STATE_COMPLETED)
        # No further transitions allowed.
        out = transition(t, to_state=STATE_WORKING)
        assert out.state == STATE_FAILED

    def test_failed_is_terminal(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_FAILED, error="explicit")
        out = transition(t, to_state=STATE_WORKING)
        assert out.state == STATE_FAILED

    def test_unknown_state_rejected(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        out = transition(t, to_state="quantum")
        assert out.state == STATE_FAILED
        assert "unknown target state" in (out.error or "")

    def test_input_required_round_trip(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        t = transition(t, to_state=STATE_INPUT_REQUIRED)
        assert t.state == STATE_INPUT_REQUIRED
        # input_required → working allowed for resubmission.
        out = transition(t, to_state=STATE_WORKING)
        assert out.state == STATE_WORKING

    def test_error_attached(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        out = transition(t, to_state=STATE_FAILED, error="explicit reason")
        assert out.error == "explicit reason"


# ---------------------------------------------------------------------------
# dispatch_task
# ---------------------------------------------------------------------------


class _Hit:
    def __init__(self, chunk_id: str = "c", rel_path: str = "x.py"):
        self.chunk_id = chunk_id
        self.rel_path = rel_path
        self.start_line = 1
        self.end_line = 10
        self.score = 0.5
        self.rank = 0
        self.language = "python"
        self.kind = "function"
        self.symbol = "foo"
        self.snippet = "def foo(): pass"
        self.repo = None


class TestDispatchTask:
    def test_happy_path(self) -> None:
        t = new_task(ContextRequestV1(query="how does X work"))
        out = dispatch_task(t, search_fn=lambda req: [_Hit("a"), _Hit("b")])
        assert out.state == STATE_COMPLETED
        assert out.output is not None
        assert out.output.total_hits == 2

    def test_search_fn_exception_fails_task(self) -> None:
        def boom(req: ContextRequestV1) -> list[Any]:
            raise RuntimeError("db down")

        t = new_task(ContextRequestV1(query="x"))
        out = dispatch_task(t, search_fn=boom)
        assert out.state == STATE_FAILED
        assert "db down" in (out.error or "")

    def test_rejects_non_submitted_task(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        t = transition(t, to_state=STATE_COMPLETED)
        out = dispatch_task(t, search_fn=lambda _: [])
        assert out.state == STATE_FAILED
        assert "submitted/working" in (out.error or "")

    def test_working_task_continues(self) -> None:
        """A task already in `working` state can be dispatched
        (e.g., after `needs_input` was resolved and the operator
        resubmitted)."""
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        out = dispatch_task(t, search_fn=lambda _: [_Hit("a")])
        assert out.state == STATE_COMPLETED
        assert out.output is not None

    def test_session_id_override(self) -> None:
        t = new_task(ContextRequestV1(query="x", session_id="orig"))
        out = dispatch_task(
            t, search_fn=lambda _: [], session_id="override",
        )
        assert out.output is not None
        assert out.output.session_id == "override"

    def test_session_id_from_request(self) -> None:
        t = new_task(ContextRequestV1(query="x", session_id="from-req"))
        out = dispatch_task(t, search_fn=lambda _: [])
        assert out.output is not None
        assert out.output.session_id == "from-req"

    def test_empty_hits(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        out = dispatch_task(t, search_fn=lambda _: [])
        assert out.state == STATE_COMPLETED
        assert out.output is not None
        assert out.output.total_hits == 0


# ---------------------------------------------------------------------------
# needs_input
# ---------------------------------------------------------------------------


class TestNeedsInput:
    def test_from_working(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        out = needs_input(t, reason="clarify symbol")
        assert out.state == STATE_INPUT_REQUIRED
        assert out.error == "clarify symbol"

    def test_from_non_working_fails(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        # Submitted → needs_input not allowed.
        out = needs_input(t)
        assert out.state == STATE_FAILED
        assert "needs_input only valid" in (out.error or "")

    def test_reason_optional(self) -> None:
        t = new_task(ContextRequestV1(query="x"))
        t = transition(t, to_state=STATE_WORKING)
        out = needs_input(t)
        assert out.state == STATE_INPUT_REQUIRED


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.api import a2a as mod
        assert set(mod.__all__) == {
            "ALL_STATES",
            "A2ATask",
            "STATE_COMPLETED",
            "STATE_FAILED",
            "STATE_INPUT_REQUIRED",
            "STATE_SUBMITTED",
            "STATE_WORKING",
            "dispatch_task",
            "needs_input",
            "new_task",
            "transition",
        }
