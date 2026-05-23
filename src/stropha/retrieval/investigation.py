"""Investigation planning (WIN-232, EPIC 1.10 #23).

When the agent says "explain how mastery decay works", today it
has to figure out for itself: "OK, first search_code to find the
entry, then find_callers to see usage, then find_tests_for to
understand contracts, then maybe find_rationale for design intent".
That's three or four tool calls of mechanical orchestration the
agent re-discovers every session.

WIN-232 collapses the orchestration into ``plan_investigation``: a
pure function that takes a task description, runs a quick scoping
search to anchor on symbols, and returns a structured RECIPE the
agent can execute. The recipe is a list of steps —
``(order, tool, purpose, args)`` — that maps directly to the agent's
follow-on MCP calls.

The plan is intentionally PROPOSAL-ONLY: it never executes the
calls itself. The agent stays in control. Following the plan is
just running each step in order.

Strategy
--------

Three-pass recipe:

1. **Entry points** — re-run ``search_code(task)`` (the same call
   the planner already made) and propose that as the agent's
   step 1. Sets the symbol context.
2. **Callers (deps)** — for each primary symbol, propose
   ``find_callers(symbol)``. Shows what depends on this code.
3. **Tests** — for each primary symbol, propose
   ``find_tests_for(symbol)``. Establishes the contract.
4. **Rationale (when graph_loaded)** — for each primary symbol,
   propose ``find_rationale(symbol)``. Captures WHY the code is
   shaped this way.

Steps are emitted only when there's a symbol to bind to. A
file-level chunk with no symbol → no caller / test / rationale
steps for it.

What we don't do
----------------

- No LLM. The plan is template-driven; the agent's LLM is what
  EXECUTES the steps.
- No execution of the steps. Side-effect-free planning by
  design — the agent decides the cost / benefit of each step.
- No multi-hop deductions ("if X calls Y and Y has rationale Z,
  then propose Z first"). Future stories can layer more
  sophisticated DAG construction on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InvestigationStep:
    """One step in an investigation plan.

    ``tool`` is the MCP tool name the agent should call. ``args`` is
    the kwargs dict the agent should pass — already filled in with
    the primary symbol / chunk_id from the scoping search.
    """

    order: int
    tool: str
    purpose: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvestigationPlan:
    """A complete investigation recipe for one task."""

    task: str
    steps: list[InvestigationStep] = field(default_factory=list)
    primary_symbols: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Free-form annotations from the planner — e.g. 'graph not
    loaded, skipping rationale step' so the agent understands
    omissions."""


def plan_investigation(
    task: str,
    *,
    search_engine: Any,
    storage: Any | None = None,
    top_k: int = 5,
    include_callers: bool = True,
    include_tests: bool = True,
    include_rationale: bool = True,
) -> InvestigationPlan:
    """Build an ordered :class:`InvestigationPlan` for ``task``.

    Args:
        task: natural-language task description.
        search_engine: SearchEngine used for the scoping pass.
        storage: optional Storage; needed to gate the rationale
            step on whether the graph mirror is loaded.
        top_k: how many primary symbols to plan around.
        include_callers / include_tests / include_rationale:
            disable each step family individually when the agent
            already has that context.

    Returns:
        :class:`InvestigationPlan`. Empty ``steps`` when the
        scoping search returned nothing — the agent gets a clean
        signal that the task description doesn't match anything
        indexed.
    """
    notes: list[str] = []
    primary_hits = search_engine.search(task, top_k=top_k)

    # Step 1: the scoping search, suggested to the agent so they
    # can re-execute it themselves (the planner already ran it,
    # but the agent benefits from having a tool call to start with).
    steps: list[InvestigationStep] = [
        InvestigationStep(
            order=1,
            tool="search_code",
            purpose=(
                "Identify primary entry points for the task. "
                "Anchors the rest of the plan."
            ),
            args={"query": task, "top_k": top_k},
        )
    ]
    primary_symbols: list[str] = []
    for hit in primary_hits:
        sym = getattr(hit, "symbol", None)
        if sym and sym not in primary_symbols:
            primary_symbols.append(sym)

    next_order = 2

    # Step 2: callers per primary symbol.
    if include_callers and primary_symbols:
        for sym in primary_symbols:
            steps.append(
                InvestigationStep(
                    order=next_order,
                    tool="find_callers",
                    purpose=(
                        f"Discover what depends on '{sym}'. "
                        "Surfaces usage patterns + integration points."
                    ),
                    args={"symbol": sym},
                )
            )
            next_order += 1
    elif include_callers and not primary_symbols:
        notes.append(
            "No symbols on primary hits — skipping find_callers steps"
        )

    # Step 3: tests per primary symbol.
    if include_tests and primary_symbols:
        for sym in primary_symbols:
            steps.append(
                InvestigationStep(
                    order=next_order,
                    tool="find_tests_for",
                    purpose=(
                        f"Establish the contract for '{sym}' — "
                        "tests are the executable specification."
                    ),
                    args={"symbol": sym},
                )
            )
            next_order += 1

    # Step 4: rationale per primary symbol — gated on graph presence.
    if include_rationale and primary_symbols:
        graph_ok = False
        if storage is not None:
            try:
                from .graph import graph_loaded

                graph_ok = bool(graph_loaded(storage))
            except Exception:  # noqa: BLE001 — defensive
                graph_ok = False
        if not graph_ok:
            notes.append(
                "Graph not loaded — find_rationale steps omitted "
                "(run `graphify .` to enable)"
            )
        else:
            for sym in primary_symbols:
                steps.append(
                    InvestigationStep(
                        order=next_order,
                        tool="find_rationale",
                        purpose=(
                            f"Capture WHY '{sym}' is shaped this way "
                            "(ADRs / design docs / rationale edges)."
                        ),
                        args={"symbol": sym},
                    )
                )
                next_order += 1

    return InvestigationPlan(
        task=task,
        steps=steps,
        primary_symbols=primary_symbols,
        notes=notes,
    )


__all__ = [
    "InvestigationPlan",
    "InvestigationStep",
    "plan_investigation",
]
