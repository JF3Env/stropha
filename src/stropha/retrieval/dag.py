"""Server-side retrieval DAG executor (WIN-233, EPIC 1.10 #24).

WIN-232 (``plan_investigation``) returns a recipe — a list of MCP
tool calls the agent must execute in order. That's a good
intermediate primitive but it still costs the agent N round-trips
to walk the DAG.

WIN-233 closes the loop with ``execute_plan``: run every step of
an :class:`InvestigationPlan` SERVER-SIDE and return the
distilled result as a single :class:`AssembleContextResult`. The
agent gets one tool call back instead of N.

Why a separate module
---------------------

The plan is data; the executor is the dispatch table. Keeping them
in different modules:

- Lets a future story plug a different executor (e.g. one that
  runs steps concurrently, or one that LLM-rewrites the plan
  mid-execution) without touching the planner.
- Keeps the planner pure / framework-agnostic — it can run
  without storage / search_engine when the caller just wants to
  inspect the recipe.

Strategy
--------

1. Execute step 1 (always ``search_code``) as the primary hits.
2. For every subsequent ``find_*`` step, dispatch to the matching
   library helper, normalise the result to ``SearchHit`` instances,
   and accumulate into the corresponding section bucket.
3. Dedupe across sections by ``chunk_id`` (same defensive
   contract as ``assemble_context``).
4. Single budget pack across the union — primary first, then
   callers, tests, rationale.
5. Render the WIN-209 markdown bundle.

What we don't do
----------------

- No LLM. The DAG is template-driven by the plan; no model in the
  loop. The agent's LLM is what makes sense of the distilled
  bundle.
- No concurrent execution. Each step is sequential.
  ``find_callers`` for symbol A doesn't need to wait for
  ``find_callers`` for symbol B, but the wall-clock difference
  for the typical 1-5 symbol case is in the milliseconds. A
  future ticket (WIN-237 — Parallel fan-out with early-exit) is
  the right home for concurrency.
- No mid-execution plan rewriting. The recipe is fixed at the
  start. Following a different DAG path requires the agent to
  call ``plan_investigation`` again with a refined task.
"""

from __future__ import annotations

from typing import Any

from ..logging import get_logger
from ..models import SearchHit
from .assemble import (
    AssembleContextResult,
    AssembleSection,
    _format_span,
    _rationale_node_to_hit,
)
from .investigation import InvestigationPlan
from .packing import pack_to_budget

log = get_logger(__name__)


# Section order matches assemble_context (WIN-219). The executor
# emits the same set of buckets so a future caller can swap between
# assemble_context and execute_plan output without rebinding.
_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("primary", "Primary code"),
    ("callers", "Callers"),
    ("tests", "Tests"),
    ("rationale", "Rationale — why this code exists"),
)


def _normalize_to_hits(raw: Any) -> list[SearchHit]:
    """Turn whatever a graph helper returned into a list of SearchHits.

    ``find_callers`` / ``find_tests_for`` already return SearchHits.
    ``find_rationale`` returns a ``{rationale: [node_dict, ...]}``
    payload; we convert via the WIN-219 helper.
    """
    if isinstance(raw, dict) and "rationale" in raw:
        out: list[SearchHit] = []
        for node in raw.get("rationale") or []:
            synth = _rationale_node_to_hit(node)
            if synth is not None:
                out.append(synth)
        return out
    if isinstance(raw, list):
        return [h for h in raw if isinstance(h, SearchHit)]
    return []


def execute_plan(
    plan: InvestigationPlan,
    *,
    search_engine: Any,
    storage: Any | None,
    context_budget: int,
    repo_id: int | None = None,
) -> AssembleContextResult:
    """Execute every step of ``plan`` and return one bundle.

    Args:
        plan: a recipe from :func:`plan_investigation`. The first
            step is always ``search_code``; subsequent steps are
            graph extensions keyed on the primary symbols.
        search_engine: SearchEngine for the scoping search.
        storage: Storage handle for the graph helpers; required
            when the plan includes any ``find_*`` steps.
        context_budget: total token budget for the consolidated
            bundle. Required (positive). The packer enforces it.
        repo_id: optional repository scope for every semantic and graph
            retrieval. ``None`` intentionally searches across repositories.

    Returns:
        An :class:`AssembleContextResult` identical in shape to
        what ``assemble_context`` produces — so downstream
        callers (CLI, MCP layer) reuse the same rendering path.
    """
    if context_budget <= 0:
        raise ValueError("context_budget must be positive")

    # Sections start empty and accumulate as we walk the plan.
    primary_hits: list[SearchHit] = []
    caller_hits: list[SearchHit] = []
    test_hits: list[SearchHit] = []
    rationale_hits: list[SearchHit] = []

    seen_chunk_ids: set[str] = set()
    dispatch_errors: list[str] = []

    for step in plan.steps:
        if step.tool == "search_code":
            query = step.args.get("query", plan.task)
            top_k = int(step.args.get("top_k", 5))
            try:
                search_kwargs: dict[str, Any] = {"top_k": top_k}
                if repo_id is not None:
                    search_kwargs["repo_id"] = repo_id
                hits = search_engine.search(query, **search_kwargs)
            except Exception as exc:
                log.warning(
                    "dag.search_code_failed",
                    error=str(exc), step_order=step.order,
                )
                dispatch_errors.append(f"step {step.order}: {exc}")
                continue
            for h in hits:
                if h.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(h.chunk_id)
                primary_hits.append(h)
            continue

        if storage is None:
            # Graph steps require storage; without it we just skip
            # silently — the plan's notes already explain.
            continue

        symbol = step.args.get("symbol")
        if not symbol:
            continue

        try:
            if step.tool == "find_callers":
                from .graph import find_callers

                graph_kwargs: dict[str, Any] = {"depth": 1, "limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                raw = find_callers(storage, symbol, **graph_kwargs)
                normalised = _normalize_to_hits(raw)
                for h in normalised:
                    if h.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(h.chunk_id)
                    caller_hits.append(h)
            elif step.tool == "find_tests_for":
                from .graph import find_tests_for

                graph_kwargs = {"limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                raw = find_tests_for(storage, symbol, **graph_kwargs)
                normalised = _normalize_to_hits(raw)
                for h in normalised:
                    if h.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(h.chunk_id)
                    test_hits.append(h)
            elif step.tool == "find_rationale":
                from .graph import find_rationale

                graph_kwargs = {"limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                raw = find_rationale(storage, symbol, **graph_kwargs)
                normalised = _normalize_to_hits(raw)
                for h in normalised:
                    if h.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(h.chunk_id)
                    rationale_hits.append(h)
            # Unknown step.tool values are silently ignored — the
            # plan may evolve faster than the executor.
        except Exception as exc:
            log.warning(
                f"dag.{step.tool}_failed",
                symbol=symbol, error=str(exc), step_order=step.order,
            )
            dispatch_errors.append(f"step {step.order}: {exc}")

    # Single budget pack across the union — order matches the
    # planner's intent (primary first, then graph extensions in
    # callers/tests/rationale order).
    ordered_union = (
        primary_hits + caller_hits + test_hits + rationale_hits
    )
    pack = pack_to_budget(ordered_union, context_budget)
    packed_set = {id(h) for h in pack.packed}

    primary_packed = [h for h in primary_hits if id(h) in packed_set]
    callers_packed = [h for h in caller_hits if id(h) in packed_set]
    tests_packed = [h for h in test_hits if id(h) in packed_set]
    rationale_packed = [h for h in rationale_hits if id(h) in packed_set]

    sections = [
        AssembleSection(
            key="primary", title=_SECTION_ORDER[0][1],
            hits=primary_packed,
        ),
        AssembleSection(
            key="callers", title=_SECTION_ORDER[1][1],
            hits=callers_packed,
        ),
        AssembleSection(
            key="tests", title=_SECTION_ORDER[2][1],
            hits=tests_packed,
        ),
        AssembleSection(
            key="rationale", title=_SECTION_ORDER[3][1],
            hits=rationale_packed,
        ),
    ]
    bundle = _render_bundle_with_dispatch_notes(
        plan.task, sections, plan.notes, dispatch_errors,
    )
    log.info(
        "dag.executed",
        task=plan.task[:80],
        steps=len(plan.steps),
        primary=len(primary_packed),
        callers=len(callers_packed),
        tests=len(tests_packed),
        rationale=len(rationale_packed),
        budget=context_budget,
        total_tokens=pack.total_tokens,
        spilled=len(pack.spilled),
        truncated=pack.truncated,
        errors=len(dispatch_errors),
    )
    return AssembleContextResult(
        task=plan.task,
        bundle=bundle,
        sections=sections,
        total_tokens=pack.total_tokens,
        budget_tokens=context_budget,
        spilled_count=len(pack.spilled),
        truncated_oversize=pack.truncated,
    )


def _render_bundle_with_dispatch_notes(
    task: str,
    sections: list[AssembleSection],
    plan_notes: list[str],
    dispatch_errors: list[str],
) -> str:
    """Render the bundle, prepending plan-time + execution-time
    notes so the agent can see what was skipped and why.

    The base shape is the WIN-209 ``_format_span`` per hit; we
    add a leading ``> note: ...`` block for each skip / failure.
    """
    parts: list[str] = [f"# Context for: {task.strip()}", ""]
    if plan_notes or dispatch_errors:
        parts.append("> ## Planner notes")
        for n in plan_notes:
            parts.append(f"> - {n}")
        for e in dispatch_errors:
            parts.append(f"> - execution error: {e}")
        parts.append("")
    for sec in sections:
        if not sec.hits:
            continue
        parts.append(f"## {sec.title} ({len(sec.hits)} spans)")
        parts.append("")
        for hit in sec.hits:
            parts.append(_format_span(hit))
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


__all__ = [
    "execute_plan",
]
