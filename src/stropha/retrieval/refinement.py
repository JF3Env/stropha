"""Local refinement sub-agent (WIN-239, EPIC 1.10 #30).

When the agent asks ``search_code`` a question, stropha returns the
top-K and goes silent. The agent then has to inspect the results,
notice gaps (no tests covered, only one file represented, no class
definition surfaced) and issue follow-up calls — burning context
window on a loop stropha could have closed internally.

WIN-239 ships a local refinement sub-agent: a pure-Python orchestrator
that critiques a result set, identifies obvious gaps, and runs
targeted re-queries against a caller-supplied ``search_fn``. The
output is an augmented ranked list plus a trail of ``RefinementStep``
records explaining what was added and why.

The sub-agent is NOT LLM-driven — same logic as WIN-238's
``suggest_next``. An LLM-gated refiner defeats the "save the agent a
turn" purpose. Cheap heuristics dominate the design.

Gap detectors
-------------

- **No tests** — top hits contain a class/function but no file
  matching ``test_*`` / ``*_test`` / ``*.spec.*`` / ``*.test.*`` /
  ``/tests/`` / ``/test/``.
- **Single-file dominance** — ≥ ``DOMINANCE_THRESHOLD`` of the top
  hits land in the same file (default 0.75). Likely missing the
  surrounding context that lives in sibling modules.
- **Sparse results** — fewer than ``SPARSE_THRESHOLD`` hits
  (default 3). Worth a HyDE-rewritten retry.
- **Compound query unresolved** — query mentions "and" / "then" but
  the top hits cluster around only one clause's vocabulary.

Each detector that fires produces one ``RefinementStep`` (subject to
``max_steps``) whose ``query`` is the re-query the sub-agent ran.
The refined hits are fused into the original ranked list (preserving
original order; appended afterwards in ranked order, deduplicated by
``chunk_id``).

This module is a **standalone primitive**, matching the WIN-230 /
WIN-236 / WIN-237 / WIN-238 pattern. The MCP wrapper (if added later)
would be a thin Pydantic adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Re-use the same query-shape helpers shipped with WIN-238 so the
# compound / question detection stays consistent across the
# suggester and the refiner.
from .suggest_next import (
    CODE_KINDS,
    _looks_compound,
)

# Path patterns that mark a file as a test artefact. Mirrors
# ``find_tests_for``'s default pattern set so the gap detector and
# the graph query agree.
_TEST_PATH_TOKENS: tuple[str, ...] = (
    "test_",
    "_test",
    ".spec.",
    ".test.",
    "/tests/",
    "/test/",
)

DOMINANCE_THRESHOLD = 0.75
SPARSE_THRESHOLD = 3


@dataclass(frozen=True)
class RefinementStep:
    """One pass of the sub-agent.

    ``gap`` names the heuristic that triggered (e.g., ``no_tests``).
    ``query`` is the re-query stropha ran. ``added_chunk_ids`` lists
    the IDs the step contributed to the merged result set (in rank
    order). ``reason`` is a human-readable note so the agent can
    audit what stropha did on its behalf.
    """

    gap: str
    query: str
    added_chunk_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class RefinementResult:
    """Aggregate of one ``refine_results`` invocation.

    ``hits`` is the merged ranked list (original first, then
    deduplicated additions). ``steps`` records every refinement the
    sub-agent attempted (including no-op ones whose re-query
    returned nothing new). ``notes`` carries cross-cutting messages
    (e.g., "search_fn raised on step 2 — skipped").
    """

    hits: list[Any] = field(default_factory=list)
    steps: list[RefinementStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _hit_path(hit: Any) -> str:
    return str(getattr(hit, "rel_path", "") or "")


def _hit_kind(hit: Any) -> str:
    return str(getattr(hit, "kind", "") or "")


def _hit_symbol(hit: Any) -> str:
    return str(getattr(hit, "symbol", "") or "")


def _hit_id(hit: Any) -> str:
    return str(getattr(hit, "chunk_id", "") or "")


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return any(token in p for token in _TEST_PATH_TOKENS)


def _has_tests(hits: list[Any]) -> bool:
    return any(_is_test_path(_hit_path(h)) for h in hits)


def _has_code_anchor(hits: list[Any]) -> str | None:
    """Return the symbol of the first code-like hit, else None."""
    for hit in hits:
        if _hit_kind(hit) in CODE_KINDS and _hit_symbol(hit):
            return _hit_symbol(hit)
    return None


def _dominant_file(hits: list[Any]) -> str | None:
    """Return the path when one file holds ≥ ``DOMINANCE_THRESHOLD``
    of the hits, else None."""
    if not hits:
        return None
    counts: dict[str, int] = {}
    for hit in hits:
        path = _hit_path(hit)
        if path:
            counts[path] = counts.get(path, 0) + 1
    if not counts:
        return None
    top_path, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count / float(len(hits)) >= DOMINANCE_THRESHOLD:
        return top_path
    return None


def _safe_search(
    search_fn: Callable[[str], list[Any]],
    query: str,
) -> tuple[list[Any], str | None]:
    """Run ``search_fn`` defensively — exceptions become notes, never
    propagate."""
    try:
        out = search_fn(query) or []
    except Exception as exc:
        return [], f"search_fn raised on '{query}': {exc!r}"
    return list(out), None


def refine_results(
    query: str,
    hits: list[Any],
    *,
    search_fn: Callable[[str], list[Any]],
    max_steps: int = 3,
    has_graph: bool = True,
) -> RefinementResult:
    """Critique ``hits`` and run targeted re-queries to fill the gaps.

    Args:
        query: the agent's original search query.
        hits: ranked hits returned by the original search. May be
            empty.
        search_fn: callable that runs a single retrieval pass.
            ``refine_results`` is search-engine agnostic — the caller
            decides whether to plug ``SearchEngine.search``, an
            HTTP client, or a stub.
        max_steps: ceiling on the number of re-queries. Defaults to
            3; the sub-agent always honours the cap even when more
            gaps remain.
        has_graph: True when the graphify mirror is loaded. Disables
            test-discovery refinement when no graph is available
            (the path-pattern fallback works either way, but the
            confidence drops).

    Returns:
        :class:`RefinementResult` whose ``hits`` is the merged
        ranked list (originals first, then additions). Never raises.
    """
    if max_steps <= 0:
        return RefinementResult(hits=list(hits), notes=["max_steps <= 0"])

    notes: list[str] = []
    steps: list[RefinementStep] = []
    seen_ids: set[str] = {_hit_id(h) for h in hits if _hit_id(h)}
    merged: list[Any] = list(hits)
    steps_used = 0

    # Gap 1 — sparse result set: HyDE-rewritten retry.
    if len(hits) < SPARSE_THRESHOLD and query and steps_used < max_steps:
        rewritten = f"{query} (HyDE)"
        extra, err = _safe_search(search_fn, rewritten)
        if err:
            notes.append(err)
        added = _merge_unique(merged, extra, seen_ids)
        steps.append(RefinementStep(
            gap="sparse",
            query=rewritten,
            added_chunk_ids=added,
            reason=(
                f"Original returned {len(hits)} hits (< {SPARSE_THRESHOLD}); "
                "HyDE retry surfaced more candidates."
            ),
        ))
        steps_used += 1

    # Gap 2 — code anchor present but no tests: run a tests-for query.
    anchor = _has_code_anchor(merged)
    if (
        anchor
        and not _has_tests(merged)
        and steps_used < max_steps
    ):
        re_query = f"tests for {anchor}"
        extra, err = _safe_search(search_fn, re_query)
        if err:
            notes.append(err)
        added = _merge_unique(merged, extra, seen_ids)
        gap = "no_tests" if has_graph else "no_tests_no_graph"
        steps.append(RefinementStep(
            gap=gap,
            query=re_query,
            added_chunk_ids=added,
            reason=(
                f"Code anchor {anchor!r} surfaced but no test files in "
                "the top hits — ran a tests-for retry."
            ),
        ))
        steps_used += 1

    # Gap 3 — single-file dominance: broaden to find related sites.
    dom_path = _dominant_file(merged)
    if dom_path and anchor and steps_used < max_steps:
        re_query = f"references to {anchor}"
        extra, err = _safe_search(search_fn, re_query)
        if err:
            notes.append(err)
        added = _merge_unique(merged, extra, seen_ids)
        steps.append(RefinementStep(
            gap="single_file_dominance",
            query=re_query,
            added_chunk_ids=added,
            reason=(
                f"All top hits clustered in {dom_path} — ran a "
                "references retry to broaden the working set."
            ),
        ))
        steps_used += 1

    # Gap 4 — compound query but cluster suggests only one clause met.
    if (
        query
        and _looks_compound(query)
        and steps_used < max_steps
    ):
        # Naive split on the common compound tokens. The decomposer
        # in stropha.retrieval.query_decomposition would do better,
        # but pulling it in here couples this primitive to the
        # decomposer's import surface; the lexical split is good
        # enough for the heuristic.
        clauses = _split_compound(query)
        for clause in clauses:
            if steps_used >= max_steps:
                break
            extra, err = _safe_search(search_fn, clause)
            if err:
                notes.append(err)
            added = _merge_unique(merged, extra, seen_ids)
            steps.append(RefinementStep(
                gap="compound_unresolved",
                query=clause,
                added_chunk_ids=added,
                reason=(
                    f"Original query is compound; re-ran clause "
                    f"{clause!r} on its own."
                ),
            ))
            steps_used += 1

    return RefinementResult(hits=merged, steps=steps, notes=notes)


def _split_compound(query: str) -> list[str]:
    """Naive lexical split on compound tokens.

    Falls back to a single-element list when no token matches, so the
    caller always gets at least the original query back. Tokens are
    matched case-insensitively but case is preserved in the output.
    """
    splitters = (" and ", " then ", " plus ", " & ", " AND ", " THEN ")
    remainder = query
    parts: list[str] = []
    while remainder:
        cut_at: int | None = None
        cut_len = 0
        lower = remainder.lower()
        for token in splitters:
            idx = lower.find(token.lower())
            if idx >= 0 and (cut_at is None or idx < cut_at):
                cut_at = idx
                cut_len = len(token)
        if cut_at is None:
            parts.append(remainder.strip())
            break
        left = remainder[:cut_at].strip()
        if left:
            parts.append(left)
        remainder = remainder[cut_at + cut_len:]
    # Drop empties + duplicates while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return out or [query]


def _merge_unique(
    target: list[Any],
    extra: list[Any],
    seen_ids: set[str],
) -> list[str]:
    """Append ``extra`` into ``target`` skipping ids already seen.

    Mutates ``target`` and ``seen_ids`` in place. Returns the list of
    chunk_ids actually added (in order), so the step record can show
    exactly what was contributed.
    """
    added: list[str] = []
    for hit in extra:
        cid = _hit_id(hit)
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        target.append(hit)
        added.append(cid)
    return added


__all__ = [
    "DOMINANCE_THRESHOLD",
    "RefinementResult",
    "RefinementStep",
    "SPARSE_THRESHOLD",
    "refine_results",
]
