"""A2A (Agent-to-Agent) endpoint scaffold (WIN-258, EPIC 1.12 #95).

The agent ecosystem is moving past MCP-only — Google's A2A
protocol, Anthropic's agent SDKs, and a wave of inter-agent
buses all want to talk to RAG services via a task-based envelope.
WIN-258 ships the scaffold that lets stropha respond to that
shape without locking the rest of the codebase to any one
spec.

Concept
-------

An A2A interaction is a stateful **task** with five canonical
states:

- ``submitted`` — request created, not yet picked up.
- ``working`` — dispatcher has it.
- ``input_required`` — needs more context from the caller.
- ``completed`` — output ready.
- ``failed`` — terminal error.

The task envelope wraps the WIN-257 ``ContextRequestV1`` /
``ContextResponseV1`` contracts so an A2A-aware client can submit
a context-retrieval task and read the result without depending on
MCP. This module is the adapter layer; transport wiring (HTTP,
gRPC, message bus) lives in future stories.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .context_v1 import (
    CONTEXT_VERSION,
    ContextRequestV1,
    ContextResponseV1,
    to_response,
)


# Canonical task states. Kept as module-level constants (not an
# Enum) so the wire-format is plain strings and a future spec
# bump can add states without breaking the constant import.
STATE_SUBMITTED = "submitted"
STATE_WORKING = "working"
STATE_INPUT_REQUIRED = "input_required"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"

ALL_STATES: frozenset[str] = frozenset({
    STATE_SUBMITTED,
    STATE_WORKING,
    STATE_INPUT_REQUIRED,
    STATE_COMPLETED,
    STATE_FAILED,
})

# Allowed forward transitions. The dispatcher enforces these so a
# task can't skip states (e.g. submitted → completed without going
# through working).
_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_SUBMITTED: frozenset({STATE_WORKING, STATE_FAILED}),
    STATE_WORKING: frozenset({
        STATE_INPUT_REQUIRED, STATE_COMPLETED, STATE_FAILED,
    }),
    STATE_INPUT_REQUIRED: frozenset({STATE_WORKING, STATE_FAILED}),
    STATE_COMPLETED: frozenset(),
    STATE_FAILED: frozenset(),
}


class A2ATask(BaseModel):
    """Task envelope every A2A interaction flows through."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    state: str = STATE_SUBMITTED
    context_version: str = CONTEXT_VERSION
    input: ContextRequestV1
    output: ContextResponseV1 | None = None
    error: str | None = None
    capabilities: list[str] = Field(
        default_factory=lambda: ["context_v1"],
        description=(
            "Capability tags the responder advertises — lets the "
            "caller probe whether stropha supports the requested "
            "contract version before submission."
        ),
    )


SearchFn = Callable[[ContextRequestV1], list[Any]]


def new_task(request: ContextRequestV1, *, task_id: str | None = None) -> A2ATask:
    """Construct a fresh ``submitted`` task wrapping ``request``."""
    return A2ATask(
        task_id=task_id or str(uuid.uuid4()),
        state=STATE_SUBMITTED,
        input=request,
    )


def transition(task: A2ATask, *, to_state: str, error: str | None = None) -> A2ATask:
    """Return a new task with ``state=to_state`` if the transition
    is allowed.

    Disallowed transitions return a ``FAILED`` task with the
    rejection recorded in ``error`` — never raises.
    """
    if to_state not in ALL_STATES:
        return task.model_copy(update={
            "state": STATE_FAILED,
            "error": f"unknown target state: {to_state!r}",
        })
    allowed = _TRANSITIONS.get(task.state, frozenset())
    if to_state not in allowed:
        return task.model_copy(update={
            "state": STATE_FAILED,
            "error": (
                f"illegal transition {task.state!r} → {to_state!r}; "
                f"allowed: {sorted(allowed)}"
            ),
        })
    update: dict[str, Any] = {"state": to_state}
    if error is not None:
        update["error"] = error
    return task.model_copy(update=update)


def dispatch_task(
    task: A2ATask,
    *,
    search_fn: SearchFn,
    session_id: str | None = None,
) -> A2ATask:
    """Drive ``task`` from ``submitted`` → ``working`` → ``completed``
    (or ``failed``) by calling ``search_fn``.

    Args:
        task: a freshly-submitted (or working) task. Other states
            are rejected.
        search_fn: callable that takes the task's
            :class:`ContextRequestV1` and returns a list of
            internal hits. Typically wraps the live SearchEngine.
        session_id: optional override for the response's
            ``session_id`` field; defaults to whatever the request
            carried.

    Returns:
        A new ``A2ATask`` in either ``completed`` (with
        ``output`` populated) or ``failed`` (with ``error``).
        Never raises.
    """
    if task.state not in (STATE_SUBMITTED, STATE_WORKING):
        return task.model_copy(update={
            "state": STATE_FAILED,
            "error": f"dispatch requires submitted/working state, got {task.state!r}",
        })

    working = task if task.state == STATE_WORKING else transition(
        task, to_state=STATE_WORKING,
    )
    if working.state == STATE_FAILED:  # pragma: no cover - defensive: submitted→working is always allowed
        return working

    try:
        hits = search_fn(working.input)
    except Exception as exc:
        return working.model_copy(update={
            "state": STATE_FAILED,
            "error": f"search_fn raised: {exc!r}",
        })

    response = to_response(
        hits or [],
        query=working.input.query,
        session_id=session_id or working.input.session_id,
    )
    completed = transition(working, to_state=STATE_COMPLETED)
    if completed.state == STATE_FAILED:  # pragma: no cover - defensive: working→completed is always allowed
        return completed
    return completed.model_copy(update={"output": response})


def needs_input(task: A2ATask, *, reason: str = "") -> A2ATask:
    """Mark a working task as needing more input from the caller."""
    if task.state != STATE_WORKING:
        return transition(task, to_state=STATE_FAILED, error=(
            f"needs_input only valid from working state, got {task.state!r}"
        ))
    return transition(task, to_state=STATE_INPUT_REQUIRED, error=reason or None)


__all__ = [
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
]
