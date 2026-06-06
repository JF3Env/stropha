"""Contract-test surfacing (WIN-251, EPIC 1.11 #89).

Not every test exercises the same kind of guarantee. **Unit tests**
verify internal implementation; **contract tests** verify the
*public behaviour* an upstream caller depends on. When the agent
edits a symbol, the contract tests are the ones that MUST keep
passing — failing one means the public surface changed, even if
the internal mechanics are still correct.

WIN-251 ships the primitive that pulls the contract-test subset
from a regular ``find_tests_for`` result. Tests are flagged as
contract tests via three signals:

- **Path pattern** — the file lives under ``contracts/`` /
  ``specs/`` / ``api/`` / ``interface/``, or has a
  ``_contract``/``_spec``/``Spec.java``/``Contract.java`` suffix.
- **Symbol pattern** — the test function/class name contains
  ``contract`` / ``interface`` / ``api`` / ``behavior``
  (case-insensitive).
- **Tag annotation** — the test row's ``tags`` /
  ``annotations`` list contains ``"contract"``.

Each surfaced test records ``signals`` (the list of heuristics
that matched) so the caller can score confidence (one signal =
suggestive; multiple = strong). The default rejects nothing —
every signal level is reported, with the caller deciding the
threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Path tokens that mark a file as a contract-test artefact.
_PATH_PATTERNS: tuple[str, ...] = (
    "/contracts/",
    "/contract/",
    "/specs/",
    "/spec/",
    "/api-tests/",
    "/integration/",
    "_contract.",
    "_spec.",
    "spec.java",
    "spec.kt",
    "contract.java",
    "contract.kt",
)

# Substrings that mark a test symbol as contract-style. Matched
# case-insensitively against the test's ``symbol`` field.
_SYMBOL_PATTERNS: tuple[str, ...] = (
    "contract",
    "interface",
    "api",
    "behavior",
    "behaviour",
)


@dataclass(frozen=True)
class ContractTest:
    """One surfaced test with the signals that flagged it."""

    path: str
    symbol: str = ""
    signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContractTestReport:
    """Aggregate of one :func:`surface_contract_tests` invocation."""

    target_symbol: str = ""
    contract_tests: list[ContractTest] = field(default_factory=list)
    other_tests: list[ContractTest] = field(default_factory=list)
    signal_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


TestsFn = Callable[[str], dict[str, Any]]


def classify_test(
    test: dict[str, Any],
    *,
    path_patterns: tuple[str, ...] = _PATH_PATTERNS,
    symbol_patterns: tuple[str, ...] = _SYMBOL_PATTERNS,
) -> list[str]:
    """Return the list of signals matching ``test``.

    Empty list means "no signal" — i.e. the test isn't a contract
    test by our heuristics.
    """
    signals: list[str] = []
    path = str(test.get("source_file") or test.get("rel_path") or "").lower()
    if path and any(token in path for token in path_patterns):
        signals.append("path")
    symbol = str(test.get("symbol") or test.get("label") or "").lower()
    if symbol and any(token in symbol for token in symbol_patterns):
        signals.append("symbol")
    tags = test.get("tags") or test.get("annotations") or []
    if isinstance(tags, list):
        lowered = [str(t).lower() for t in tags]
        if "contract" in lowered:
            signals.append("tag")
    return signals


def surface_contract_tests(
    symbol: str,
    *,
    tests_fn: TestsFn,
    path_patterns: tuple[str, ...] = _PATH_PATTERNS,
    symbol_patterns: tuple[str, ...] = _SYMBOL_PATTERNS,
) -> ContractTestReport:
    """Pull tests for ``symbol`` and partition by contract heuristics.

    Args:
        symbol: the symbol the agent intends to edit.
        tests_fn: callable returning the ``find_tests_for`` payload
            for the symbol.
        path_patterns / symbol_patterns: override the default
            heuristics. Pass project-specific patterns to widen or
            narrow the surfaced set.

    Returns:
        :class:`ContractTestReport` partitioned into ``contract_tests``
        (any signal matched) and ``other_tests`` (no signal).
    """
    notes: list[str] = []
    target = (symbol or "").strip()
    if not target:
        return ContractTestReport(
            target_symbol="",
            notes=["empty symbol — nothing to surface"],
        )

    try:
        payload = tests_fn(target) or {}
    except Exception as exc:
        return ContractTestReport(
            target_symbol=target,
            notes=[f"tests_fn raised: {exc!r}"],
        )

    raw_tests = payload.get("tests") or []
    contract: list[ContractTest] = []
    other: list[ContractTest] = []
    signal_counts: dict[str, int] = {"path": 0, "symbol": 0, "tag": 0}

    for entry in raw_tests:
        if not isinstance(entry, dict):
            continue
        signals = classify_test(
            entry,
            path_patterns=path_patterns,
            symbol_patterns=symbol_patterns,
        )
        for s in signals:
            signal_counts[s] = signal_counts.get(s, 0) + 1
        test = ContractTest(
            path=str(entry.get("source_file") or entry.get("rel_path") or ""),
            symbol=str(entry.get("symbol") or entry.get("label") or ""),
            signals=signals,
        )
        if signals:
            contract.append(test)
        else:
            other.append(test)

    return ContractTestReport(
        target_symbol=target,
        contract_tests=contract,
        other_tests=other,
        signal_counts=signal_counts,
        notes=notes,
    )


__all__ = [
    "ContractTest",
    "ContractTestReport",
    "classify_test",
    "surface_contract_tests",
]
