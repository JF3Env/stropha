"""Suggest the next MCP tool to call (WIN-238, EPIC 1.10 #29).

Stropha already routes a single query to the best tool (``smart_search``
picks ``find_callers`` / ``find_tests_for`` etc. via the WIN-234 router).
What it doesn't do today is **steer the agent's next move** after a
search has already run. The agent is left guessing: "I got these
results — do I expand a chunk? Look for callers? Read the whole file?
Decompose the query?"

WIN-238 closes that loop with cheap meta-guidance. Given a query and
the recent results, ``suggest_next_tool`` returns a ranked list of
concrete next actions, each with a ``tool`` name, ready-to-paste
``args``, a human-readable ``reason``, and a confidence score. The
agent picks the highest-confidence one (or follows the ranked list)
without burning a turn on its own planning.

The logic is pure heuristics — no LLM call, no I/O. The signal comes
from three sources:

1. **Query shape** — empty results plus "X and Y" → suggest
   ``plan_investigation``; question-words → suggest HyDE rewrite.
2. **Hit composition** — top-ranked class/function with graph
   available → ``find_callers`` + ``find_tests_for``; doc/heading
   in the top hits → ``find_rationale``.
3. **Hit distribution** — results clustered in one file →
   ``get_file_outline``; results scattered across many files →
   ``find_related`` (chase the connecting symbol).

Suggestions are returned sorted by confidence DESC and truncated to
``max_suggestions``. The module is a standalone primitive matching
the WIN-230 / WIN-236 / WIN-237 pattern — the MCP wrapper in
``server.py`` is a thin Pydantic adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Kinds that count as "callable / referenceable code" for the
# ``find_callers`` / ``find_tests_for`` suggestion.
CODE_KINDS = frozenset({"class", "function", "method", "interface", "trait"})

# Kinds that count as "human-written explanation" for the
# ``find_rationale`` suggestion.
DOC_KINDS = frozenset({"doc", "heading", "paragraph", "comment", "adr"})


@dataclass(frozen=True)
class Suggestion:
    """One suggested next-tool invocation.

    ``args`` is a plain dict so the agent can pass it through to the
    target MCP tool without translation. ``confidence`` is a soft
    score in ``[0, 1]`` — the agent decides whether to act on the
    top suggestion or browse the full list.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.5


def _symbols_in_top(hits: list[Any], limit: int = 3) -> list[str]:
    out: list[str] = []
    for hit in hits[:limit]:
        sym = getattr(hit, "symbol", None)
        if sym:
            out.append(str(sym))
    return out


def _kinds_in_top(hits: list[Any], limit: int = 3) -> list[str]:
    return [str(getattr(h, "kind", "") or "") for h in hits[:limit]]


def _paths_in_top(hits: list[Any], limit: int = 5) -> set[str]:
    return {
        str(getattr(h, "rel_path", "") or "")
        for h in hits[:limit]
        if getattr(h, "rel_path", None)
    }


def _looks_compound(query: str) -> bool:
    """Heuristic: query mentions multiple discrete clauses."""
    q = f" {query.lower()} "
    return any(token in q for token in (" and ", " then ", " plus ", " & "))


def _looks_question(query: str) -> bool:
    q = query.lower().strip()
    return q.startswith(
        ("how ", "what ", "why ", "where ", "when ", "who ", "which ")
    )


def suggest_next_tool(
    query: str,
    hits: list[Any],
    *,
    has_graph: bool = True,
    has_rationale: bool = True,
    max_suggestions: int = 5,
) -> list[Suggestion]:
    """Recommend next tools based on the current query + results.

    Args:
        query: the user's most recent search query (may be empty).
        hits: SearchHit-like objects from the most recent retrieval.
            Each is expected to expose ``symbol``, ``kind``,
            ``rel_path``; missing attributes are treated defensively.
        has_graph: True when the graphify mirror is loaded — gates
            the ``find_callers`` / ``find_tests_for`` / ``find_related``
            suggestions.
        has_rationale: True when the graph has ``rationale_for``
            edges — gates the ``find_rationale`` suggestion. Without
            this, suggesting ``find_rationale`` is a guaranteed
            empty result.
        max_suggestions: ceiling on the returned list. Defaults to 5;
            no API panic when the heuristics produce fewer.

    Returns:
        Suggestions sorted by ``confidence`` DESC, truncated to
        ``max_suggestions``. Always a list — never raises.
    """
    suggestions: list[Suggestion] = []

    # --------------------------------------------------------------
    # Empty-result branch: broaden the search.
    # --------------------------------------------------------------
    if not hits:
        if query and _looks_compound(query):
            suggestions.append(Suggestion(
                tool="plan_investigation",
                args={"query": query},
                reason=(
                    "No results and the query looks compound — "
                    "plan_investigation orders sub-steps over the graph."
                ),
                confidence=0.70,
            ))
        if query and _looks_question(query):
            suggestions.append(Suggestion(
                tool="search_code",
                args={"query": query, "use_hyde": True},
                reason=(
                    "No results — try HyDE-rewritten dense query for "
                    "conceptual match."
                ),
                confidence=0.60,
            ))
        if not query:
            suggestions.append(Suggestion(
                tool="bootstrap_environment",
                args={},
                reason=(
                    "No query and no prior hits — bootstrap_environment "
                    "surfaces the project shape so the agent can pick "
                    "the next move."
                ),
                confidence=0.40,
            ))
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:max_suggestions]

    # --------------------------------------------------------------
    # Result-driven heuristics.
    # --------------------------------------------------------------
    top_symbols = _symbols_in_top(hits)
    top_kinds = _kinds_in_top(hits)
    paths = _paths_in_top(hits)

    has_code = any(k in CODE_KINDS for k in top_kinds)
    has_doc = any(k in DOC_KINDS for k in top_kinds)

    if has_code and has_graph and top_symbols:
        sym = top_symbols[0]
        kind = top_kinds[0] if top_kinds else "symbol"
        suggestions.append(Suggestion(
            tool="find_callers",
            args={"symbol": sym},
            reason=(
                f"Top result is a {kind} ({sym}) — find_callers "
                "surfaces every call site."
            ),
            confidence=0.85,
        ))
        suggestions.append(Suggestion(
            tool="find_tests_for",
            args={"symbol": sym},
            reason=(
                f"Top result is a {kind} ({sym}) — find_tests_for "
                "shows the tests that exercise it."
            ),
            confidence=0.80,
        ))

    if has_doc and has_graph and has_rationale and top_symbols:
        suggestions.append(Suggestion(
            tool="find_rationale",
            args={"symbol": top_symbols[0]},
            reason=(
                "Top results include docs/headings — find_rationale "
                "walks the design notes that explain the why."
            ),
            confidence=0.65,
        ))

    if len(paths) == 1 and hits:
        only = next(iter(paths))
        suggestions.append(Suggestion(
            tool="get_file_outline",
            args={"path": only},
            reason=(
                f"All top hits land in {only} — get_file_outline "
                "shows the full file structure cheaply."
            ),
            confidence=0.70,
        ))

    if len(paths) >= 4 and has_graph and top_symbols:
        suggestions.append(Suggestion(
            tool="find_related",
            args={"symbol": top_symbols[0]},
            reason=(
                f"Results span {len(paths)} files — find_related "
                "surfaces the symbols connecting them."
            ),
            confidence=0.55,
        ))

    if query and _looks_compound(query):
        suggestions.append(Suggestion(
            tool="plan_investigation",
            args={"query": query},
            reason=(
                "Query mentions multiple clauses — plan_investigation "
                "orders the retrieval steps so nothing is missed."
            ),
            confidence=0.50,
        ))

    if hits and len(hits) >= 8:
        suggestions.append(Suggestion(
            tool="assemble_context",
            args={"query": query} if query else {},
            reason=(
                f"Returned {len(hits)} hits — assemble_context packs "
                "the top ones into a token-budgeted bundle."
            ),
            confidence=0.45,
        ))

    deduped = _dedupe_suggestions(suggestions)
    ranked = sorted(deduped, key=lambda s: s.confidence, reverse=True)
    return ranked[:max_suggestions]


def _dedupe_suggestions(suggestions: list[Suggestion]) -> list[Suggestion]:
    """Collapse suggestions with the same ``(tool, args)`` to the
    highest-confidence entry.

    Heuristic branches can overlap (e.g. a compound query that also
    returned 8+ hits would suggest ``plan_investigation`` from two
    paths). The dedup keeps the strongest signal.

    When ``args`` contains an unhashable value (e.g. a list), we
    fall back to a tool-only key — two such suggestions for the same
    tool collapse, which is the desired behaviour.
    """
    bucket: dict[tuple[str, object], Suggestion] = {}
    for s in suggestions:
        try:
            key: tuple[str, object] = (s.tool, frozenset(s.args.items()))
        except TypeError:
            key = (s.tool, "__unhashable__")
        existing = bucket.get(key)
        if existing is None or s.confidence > existing.confidence:
            bucket[key] = s
    return list(bucket.values())


__all__ = [
    "CODE_KINDS",
    "DOC_KINDS",
    "Suggestion",
    "suggest_next_tool",
]
