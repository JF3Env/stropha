"""Test-impact analysis (WIN-248, EPIC 1.11 #86).

After an edit, the agent doesn't need to run the entire test suite
— only the tests that exercise the changed code (directly) plus
the tests that exercise the callers of the changed code
(transitively, one hop by default). WIN-248 computes that minimal
set.

For each input symbol:

1. **Direct tests** — ``find_tests_for(symbol)``. The tests that
   target the symbol's own contract.
2. **Indirect tests** (when ``transitive_depth >= 1``) —
   ``find_callers(symbol)`` → for each caller, ``find_tests_for``.
   Catches tests that exercise the symbol *through* one of its
   call sites.

The report lists every impacted test deduplicated by path/symbol,
plus a ``per_symbol`` breakdown for triage and a ``run_command``
hint built from the test files (pytest-style by default; caller
overrides for other frameworks).

Like every WIN-23x / WIN-24x primitive shipped to date, this
module is standalone — the MCP wrapper is the thin adapter in
``server.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImpactedTest:
    """One test entry in the impact report.

    ``via`` lists the routes the test was reached through —
    ``"direct"`` for ``find_tests_for(symbol)`` and the caller
    symbol's name for indirect hops.
    """

    path: str
    symbol: str = ""
    via: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TestImpactReport:
    """Aggregate of one :func:`analyse_test_impact` invocation.

    ``tests`` is the deduplicated union sorted by path. ``per_symbol``
    maps each input symbol to the count of *unique* tests it
    contributed (direct + indirect combined).
    """

    changed_symbols: list[str] = field(default_factory=list)
    tests: list[ImpactedTest] = field(default_factory=list)
    per_symbol: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    run_command: str = ""


TestsFn = Callable[[str], dict[str, Any]]
CallersFn = Callable[[str], dict[str, Any]]


def _test_key(test: dict[str, Any]) -> str:
    """Stable identity for dedup. Path-first because two tests with
    the same name in different files are independent; same path with
    different symbols collapses into one entry below."""
    path = str(test.get("source_file") or test.get("rel_path") or "")
    sym = str(test.get("symbol") or test.get("label") or "")
    if path and sym:
        return f"{path}::{sym}"
    if path:
        return path
    return sym or "?"


def _safe_fetch(
    fn: Callable[[str], dict[str, Any]],
    symbol: str,
    key: str,
    notes: list[str],
) -> list[dict[str, Any]]:
    """Run ``fn(symbol)`` and pull the named list; never raise."""
    try:
        payload = fn(symbol) or {}
    except Exception as exc:
        notes.append(
            f"{fn.__name__ if hasattr(fn, '__name__') else 'fn'}"
            f"({symbol!r}) raised: {exc!r}"
        )
        return []
    out = payload.get(key)
    if not isinstance(out, list):
        return []
    return [dict(item) for item in out if isinstance(item, dict)]


def analyse_test_impact(
    symbols: list[str],
    *,
    tests_fn: TestsFn,
    callers_fn: CallersFn,
    transitive_depth: int = 1,
    runner: str = "pytest",
) -> TestImpactReport:
    """Compute the minimal test set impacted by ``symbols``.

    Args:
        symbols: changed symbol names. Empty / whitespace entries
            skipped (recorded in notes); duplicates deduplicated
            before any graph call.
        tests_fn: returns the ``find_tests_for`` payload for a
            symbol.
        callers_fn: returns the ``find_callers`` payload for a
            symbol. Used only when ``transitive_depth >= 1``.
        transitive_depth: 0 = direct tests only;
            1 = direct + immediate callers' tests.
        runner: prefix for the ``run_command`` hint. Defaults to
            ``"pytest"``; pass ``"go test"`` / ``"npm test"`` etc.
            for other ecosystems.

    Returns:
        :class:`TestImpactReport` with deduplicated tests, per-symbol
        attribution, and a one-line ``run_command``.
    """
    notes: list[str] = []
    # Bucket per (path::symbol) -> ImpactedTest accumulator so we
    # can merge ``via`` lists across input symbols.
    bucket: dict[str, dict[str, Any]] = {}
    per_symbol: dict[str, int] = {}

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
        before = set(bucket.keys())
        direct = _safe_fetch(tests_fn, symbol, "tests", notes)
        for test in direct:
            _merge(bucket, test, via="direct")

        if transitive_depth >= 1:
            callers = _safe_fetch(callers_fn, symbol, "callers", notes)
            for caller in callers:
                caller_sym = str(caller.get("symbol") or caller.get("label") or "")
                if not caller_sym:
                    continue
                indirect = _safe_fetch(tests_fn, caller_sym, "tests", notes)
                for test in indirect:
                    _merge(bucket, test, via=caller_sym)

        per_symbol[symbol] = len(set(bucket.keys()) - before)

    tests = sorted(
        (
            ImpactedTest(
                path=str(entry["path"]),
                symbol=str(entry["symbol"]),
                via=sorted(entry["via"]),
            )
            for entry in bucket.values()
        ),
        key=lambda t: (t.path, t.symbol),
    )

    run_command = _build_run_command(tests, runner)

    return TestImpactReport(
        changed_symbols=deduped,
        tests=tests,
        per_symbol=per_symbol,
        notes=notes,
        run_command=run_command,
    )


def _merge(
    bucket: dict[str, dict[str, Any]],
    test: dict[str, Any],
    *,
    via: str,
) -> None:
    """Insert / update the bucket entry for ``test``."""
    key = _test_key(test)
    if key == "?":
        return
    path = str(test.get("source_file") or test.get("rel_path") or "")
    sym = str(test.get("symbol") or test.get("label") or "")
    if key not in bucket:
        bucket[key] = {"path": path, "symbol": sym, "via": set()}
    bucket[key]["via"].add(via)


def _build_run_command(
    tests: list[ImpactedTest],
    runner: str,
) -> str:
    """Build a one-line invocation that runs exactly the impacted
    tests. Empty list → empty string."""
    if not tests:
        return ""
    # Dedup paths only (the same file may contain multiple symbols).
    paths = sorted({t.path for t in tests if t.path})
    if not paths:
        return ""
    return f"{runner} " + " ".join(paths)


__all__ = [
    "ImpactedTest",
    "TestImpactReport",
    "analyse_test_impact",
]
