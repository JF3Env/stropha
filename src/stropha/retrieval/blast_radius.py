"""Post-edit blast-radius report (WIN-246, EPIC 1.11 #84).

After the agent edits a file or symbol, the next question is "what
else might be broken?" — callers that rely on the changed contract,
tests that exercise the changed code, related symbols that
compose with it, ADRs/docs that constrain the change. Today the
agent has to call ``find_callers`` + ``find_tests_for`` +
``find_related`` + ``find_rationale`` separately and reason about
the union.

WIN-246 collapses that into a single primitive. Given a list of
changed symbols, ``compute_blast_radius`` queries every relevant
graph reader, fuses the results (deduplicated by ``node_id``),
and returns a structured report with one section per concern plus
a one-line summary the agent can paste into a commit message.

The primitive is standalone — the MCP wrapper binds the existing
graph readers as the four callables. Matches the WIN-238 / WIN-244
/ WIN-245 pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BlastRadiusReport:
    """Aggregate of a :func:`compute_blast_radius` invocation.

    ``callers`` is the union of every caller across the changed
    symbols, deduplicated by ``node_id``. Same shape for ``tests``,
    ``related``, ``rationale``. ``per_symbol`` maps each input
    symbol to its individual counts so the agent can see which
    change has the largest blast radius.
    """

    changed_symbols: list[str] = field(default_factory=list)
    callers: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    related: list[dict[str, Any]] = field(default_factory=list)
    rationale: list[dict[str, Any]] = field(default_factory=list)
    per_symbol: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    summary: str = ""


CallersFn = Callable[[str], dict[str, Any]]
TestsFn = Callable[[str], dict[str, Any]]
RelatedFn = Callable[[str], dict[str, Any]]
RationaleFn = Callable[[str], dict[str, Any]]


def _safe_fetch(
    fn: Callable[[str], dict[str, Any]],
    symbol: str,
    key: str,
    notes: list[str],
) -> list[dict[str, Any]]:
    """Run ``fn(symbol)`` and pull the named list. Defensive shape:
    a missing key returns ``[]``; an exception becomes a note."""
    try:
        payload = fn(symbol) or {}
    except Exception as exc:
        notes.append(f"{fn.__name__ if hasattr(fn, '__name__') else 'fn'}"
                     f"({symbol!r}) raised: {exc!r}")
        return []
    out = payload.get(key)
    if not isinstance(out, list):
        return []
    return [dict(item) for item in out if isinstance(item, dict)]


def _node_key(node: dict[str, Any]) -> str:
    """Stable identity for dedup. Prefer ``node_id``; fall back to
    ``chunk_id``; fall back to ``(symbol, rel_path)``."""
    nid = node.get("node_id") or node.get("chunk_id")
    if nid:
        return f"node:{nid}"
    sym = node.get("symbol") or ""
    path = node.get("rel_path") or node.get("source_file") or ""
    return f"pair:{sym}|{path}"


def _merge_unique(
    target: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    seen: set[str],
) -> int:
    """Append ``extra`` into ``target`` skipping nodes already seen.
    Returns the count actually added."""
    added = 0
    for node in extra:
        key = _node_key(node)
        if key in seen:
            continue
        seen.add(key)
        target.append(node)
        added += 1
    return added


def compute_blast_radius(
    symbols: list[str],
    *,
    callers_fn: CallersFn,
    tests_fn: TestsFn,
    related_fn: RelatedFn,
    rationale_fn: RationaleFn | None = None,
) -> BlastRadiusReport:
    """Build a blast-radius report for ``symbols``.

    Args:
        symbols: list of changed symbol names. Empty entries are
            skipped; duplicates are deduplicated before any graph
            call so we don't pay for the same lookup twice.
        callers_fn: returns the ``find_callers`` payload for a symbol.
        tests_fn: returns the ``find_tests_for`` payload.
        related_fn: returns the ``find_related`` payload.
        rationale_fn: optional ``find_rationale`` callable. ``None``
            disables the rationale section (e.g. when the graph
            doesn't carry ``rationale_for`` edges yet).

    Returns:
        :class:`BlastRadiusReport` with the merged union per concern,
        ``per_symbol`` counters for triage, and a one-line summary.
    """
    notes: list[str] = []
    callers: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    related: list[dict[str, Any]] = []
    rationale: list[dict[str, Any]] = []

    seen_callers: set[str] = set()
    seen_tests: set[str] = set()
    seen_related: set[str] = set()
    seen_rationale: set[str] = set()

    per_symbol: dict[str, dict[str, int]] = {}
    deduped: list[str] = []
    seen_input: set[str] = set()
    for raw in symbols:
        name = (raw or "").strip()
        if not name:
            notes.append("skipped empty / whitespace symbol")
            continue
        if name in seen_input:
            continue
        seen_input.add(name)
        deduped.append(name)

    for symbol in deduped:
        c_extra = _safe_fetch(callers_fn, symbol, "callers", notes)
        t_extra = _safe_fetch(tests_fn, symbol, "tests", notes)
        r_extra = _safe_fetch(related_fn, symbol, "related", notes)
        rat_extra: list[dict[str, Any]] = []
        if rationale_fn is not None:
            rat_extra = _safe_fetch(
                rationale_fn, symbol, "rationale", notes,
            )

        c_added = _merge_unique(callers, c_extra, seen_callers)
        t_added = _merge_unique(tests, t_extra, seen_tests)
        r_added = _merge_unique(related, r_extra, seen_related)
        rat_added = _merge_unique(rationale, rat_extra, seen_rationale)

        per_symbol[symbol] = {
            "callers": c_added,
            "tests": t_added,
            "related": r_added,
            "rationale": rat_added,
        }

    summary = _format_summary(deduped, callers, tests, related, rationale)

    return BlastRadiusReport(
        changed_symbols=deduped,
        callers=callers,
        tests=tests,
        related=related,
        rationale=rationale,
        per_symbol=per_symbol,
        notes=notes,
        summary=summary,
    )


def _format_summary(
    symbols: list[str],
    callers: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    related: list[dict[str, Any]],
    rationale: list[dict[str, Any]],
) -> str:
    """One-line summary suitable for pasting into a commit message."""
    if not symbols:
        return "No symbols supplied; nothing to assess."
    parts = [
        f"{len(callers)} caller(s)",
        f"{len(tests)} test(s)",
        f"{len(related)} related symbol(s)",
    ]
    if rationale:
        parts.append(f"{len(rationale)} rationale node(s)")
    syms = ", ".join(symbols) if len(symbols) <= 3 else (
        f"{symbols[0]} + {len(symbols) - 1} more"
    )
    return f"Editing {syms} touches " + ", ".join(parts) + "."


__all__ = [
    "BlastRadiusReport",
    "compute_blast_radius",
]
