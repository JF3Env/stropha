"""Hallucinated-symbol guard (WIN-245, EPIC 1.11 #83).

LLM agents routinely invent APIs that don't exist — function names
plausible enough to type, classes that "ought to" exist given the
surrounding code, methods extrapolated from a familiar framework.
The cost is a PR review burning on a missing import or a method
that was never written. The earlier we catch the hallucination,
the cheaper the correction.

WIN-245 ships a primitive that vets a batch of symbol names against
the index BEFORE the agent commits them to code. For each name:

- ``exists=True`` — the symbol resolves in the index. No issue.
- ``exists=False`` — the symbol doesn't exist; the report lists
  near-miss matches so the agent can pick the correct name without
  another search.

The module composes with WIN-244's ``verify_claim`` and WIN-238's
``suggest_next_tool`` — same pure-heuristic, no-LLM-call design.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SymbolCheck:
    """One symbol's verification outcome.

    ``exists`` is True when the index returns at least one chunk
    whose ``symbol`` column matches the query (exact / suffix /
    case-insensitive).  ``suggestions`` is a (possibly empty) list
    of near-miss symbol names — usually substring matches sorted by
    edit-distance proxy.
    """

    symbol: str
    exists: bool
    suggestions: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class HallucinationReport:
    """Aggregate of one :func:`check_symbols` invocation."""

    checks: list[SymbolCheck] = field(default_factory=list)
    present_count: int = 0
    missing_count: int = 0
    notes: list[str] = field(default_factory=list)


def _length_distance(a: str, b: str) -> int:
    """Cheap proxy for edit distance — absolute length difference.

    Edit distance would be the textbook choice but it's O(n*m) per
    pair; a length delta is O(1) and good enough for picking the
    five most-similar names from a short candidate list.
    """
    return abs(len(a) - len(b))


def check_symbols(
    symbols: list[str],
    *,
    exists_fn: Callable[[str], bool],
    similar_fn: Callable[[str], list[str]] = lambda _: [],
    max_suggestions: int = 5,
) -> HallucinationReport:
    """Vet a batch of symbol names against the index.

    Args:
        symbols: free-form list of names the agent intends to use.
            Whitespace-only entries are skipped.
        exists_fn: callable returning True when the symbol resolves.
            Typically wraps ``Storage.lookup_by_symbol(symbol,
            limit=1)``.
        similar_fn: callable returning a list of candidate names
            similar to the missing symbol. Called only when
            ``exists_fn`` returns False so the cost is paid per
            miss, not per check. Defaults to "no suggestions" so
            the primitive works without a similarity backend.
        max_suggestions: cap on the suggestions list per missing
            symbol. Defaults to 5.

    Returns:
        :class:`HallucinationReport` listing each symbol's outcome
        in the input order.
    """
    checks: list[SymbolCheck] = []
    present = 0
    missing = 0
    notes: list[str] = []
    seen: set[str] = set()

    for raw in symbols:
        name = (raw or "").strip()
        if not name:
            notes.append("skipped empty / whitespace symbol")
            continue
        if name in seen:
            # Don't burn budget re-checking the same name in the
            # same batch — same result, same suggestions.
            continue
        seen.add(name)

        try:
            exists = bool(exists_fn(name))
        except Exception as exc:
            notes.append(f"exists_fn raised on {name!r}: {exc!r}")
            exists = False

        if exists:
            present += 1
            checks.append(SymbolCheck(
                symbol=name, exists=True,
                reason=f"{name} resolves in the index.",
            ))
            continue

        try:
            raw_suggestions = list(similar_fn(name) or [])
        except Exception as exc:
            notes.append(f"similar_fn raised on {name!r}: {exc!r}")
            raw_suggestions = []

        # Dedupe, drop the query itself, and rank by length-delta.
        ranked = sorted(
            {s for s in raw_suggestions if s and s != name},
            key=lambda s: (_length_distance(s, name), s),
        )[:max(1, int(max_suggestions))]

        reason = (
            f"{name} not found in the index."
            if not ranked else
            f"{name} not found; closest match: {ranked[0]!r}."
        )
        missing += 1
        checks.append(SymbolCheck(
            symbol=name, exists=False,
            suggestions=ranked,
            reason=reason,
        ))

    return HallucinationReport(
        checks=checks,
        present_count=present,
        missing_count=missing,
        notes=notes,
    )


__all__ = [
    "HallucinationReport",
    "SymbolCheck",
    "check_symbols",
]
