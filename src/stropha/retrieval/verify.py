"""Claim verification against the graph (WIN-244, EPIC 1.11 #82).

The sharpest form of guidance is telling the agent when it's wrong.
``verify_claim`` takes a natural-language assertion like "FooClass
calls BarService" and confirms or denies it against the graphify
mirror. When the claim is false, the verdict carries the actual
edges from the subject so the agent can correct its mental model
without re-querying.

Supported claim shapes (case-insensitive)
-----------------------------------------

| Shape | Predicate | Verification |
|---|---|---|
| ``X calls Y`` | calls | Y is on ``X -[calls]-> _`` edges. |
| ``X invokes Y`` | calls | alias for the above. |
| ``X uses Y`` / ``X references Y`` | references | Y appears via ``references`` edges. |
| ``X implements Y`` | implements | Y appears via ``implements`` edges. |
| ``X is called by Y`` | called_by | inverse of calls. |
| ``X is a class`` / ``... function`` / ``... method`` | kind | X resolves to a node with that kind. |
| ``X is in F`` / ``X lives in F`` | in_file | X resolves to a node with rel_path containing F. |
| ``X has tests`` / ``X is tested`` | has_tests | tests_fn returns a non-empty list. |

Unrecognised predicates return ``verified=None`` with an explanatory
``reason``; same when the graph mirror is unavailable. ``verified``
is **tri-state**: ``True`` / ``False`` / ``None``.

Like every WIN-23x / WIN-24x primitive shipped to date, this module
is standalone — the MCP wrapper in ``server.py`` is a thin
Pydantic adapter that binds the existing graph-reader functions as
the four callables.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """A parsed claim ready to verify.

    ``predicate`` is normalised to one of:
    ``"calls"`` / ``"called_by"`` / ``"references"`` /
    ``"implements"`` / ``"kind"`` / ``"in_file"`` / ``"has_tests"``.
    ``object`` is ``None`` for unary predicates (``has_tests``).
    """

    subject: str
    predicate: str
    object: str | None = None
    raw_text: str = ""


@dataclass(frozen=True)
class Verdict:
    """Outcome of :func:`verify_claim`.

    ``verified`` is tri-state:
    - ``True`` — graph confirms the claim.
    - ``False`` — graph contradicts the claim.
    - ``None`` — undetermined (unknown predicate, parse failure,
      graph unavailable, subject not resolvable).
    """

    claim: Claim
    verified: bool | None
    reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_KIND_VALUES = ("class", "function", "method", "interface", "trait", "struct")

# Order matters: more-specific patterns first. ``re.fullmatch`` against
# the lowercased, whitespace-collapsed input.
_PATTERNS: tuple[tuple[str, str], ...] = (
    # X has tests / X is tested
    (r"(?P<subject>\S+)\s+(has\s+tests?|is\s+tested)", "has_tests"),
    # X is a <kind>
    (
        r"(?P<subject>\S+)\s+is\s+(a|an)\s+"
        rf"(?P<object>{'|'.join(_KIND_VALUES)})",
        "kind",
    ),
    # X is in (file) F / X lives in F
    (
        r"(?P<subject>\S+)\s+(?:is\s+in(?:\s+file)?|lives\s+in)\s+"
        r"(?P<object>\S+)",
        "in_file",
    ),
    # X is called by Y
    (
        r"(?P<subject>\S+)\s+is\s+called\s+by\s+(?P<object>\S+)",
        "called_by",
    ),
    # X calls / invokes Y
    (r"(?P<subject>\S+)\s+(calls?|invokes?)\s+(?P<object>\S+)", "calls"),
    # X uses / references Y
    (
        r"(?P<subject>\S+)\s+(uses?|references?)\s+(?P<object>\S+)",
        "references",
    ),
    # X implements Y
    (r"(?P<subject>\S+)\s+implements?\s+(?P<object>\S+)", "implements"),
)


def parse_claim(text: str) -> Claim | None:
    """Parse ``text`` into a :class:`Claim`, or ``None`` when no
    pattern matches.

    Whitespace is collapsed before matching; the original text is
    preserved in ``Claim.raw_text`` for downstream display.
    """
    if not text or not text.strip():
        return None
    collapsed = " ".join(text.strip().split())
    lowered = collapsed.lower()
    for pattern, predicate in _PATTERNS:
        match = re.fullmatch(pattern, lowered)
        if not match:
            continue
        # Pull subject + (optional) object from the ORIGINAL case
        # so the verdict carries the agent's casing.
        subject_lower = match.group("subject")
        subject = _slice_original(collapsed, subject_lower)
        obj_lower = match.groupdict().get("object")
        obj = (
            _slice_original(collapsed, obj_lower)
            if obj_lower is not None else None
        )
        return Claim(
            subject=subject,
            predicate=predicate,
            object=obj,
            raw_text=collapsed,
        )
    return None


def _slice_original(collapsed: str, lower_token: str) -> str:
    """Return the substring of ``collapsed`` that lowercases to
    ``lower_token``. Falls back to ``lower_token`` when no exact
    case-preserving slice is found."""
    idx = collapsed.lower().find(lower_token)
    if idx < 0:
        return lower_token
    return collapsed[idx:idx + len(lower_token)]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


CallersFn = Callable[[str], dict[str, Any]]
RelatedFn = Callable[[str, list[str] | None], dict[str, Any]]
TestsFn = Callable[[str], dict[str, Any]]
ResolveFn = Callable[[str], dict[str, Any] | None]


def verify_claim(
    claim: Claim,
    *,
    callers_fn: CallersFn,
    related_fn: RelatedFn,
    tests_fn: TestsFn,
    resolve_fn: ResolveFn,
) -> Verdict:
    """Verify ``claim`` against caller-supplied graph readers.

    Args:
        claim: parsed claim from :func:`parse_claim`.
        callers_fn: returns the find_callers payload for a symbol.
        related_fn: returns the find_related payload for a symbol
            with an optional ``relations`` filter list.
        tests_fn: returns the find_tests_for payload for a symbol.
        resolve_fn: returns a node dict for a symbol or ``None``.

    Returns:
        :class:`Verdict` whose ``verified`` field is tri-state.
    """
    subject = claim.subject

    # --- has_tests (unary) ----------------------------------------------
    if claim.predicate == "has_tests":
        payload = tests_fn(subject)
        tests = payload.get("tests") or []
        if tests:
            return Verdict(
                claim=claim, verified=True,
                reason=f"{subject} has {len(tests)} tracked test(s).",
                evidence=tests[:10],
            )
        return Verdict(
            claim=claim, verified=False,
            reason=f"No tests found for {subject}.",
            evidence=[],
        )

    # --- kind (subject only) --------------------------------------------
    if claim.predicate == "kind":
        node = resolve_fn(subject)
        if node is None:
            return Verdict(
                claim=claim, verified=None,
                reason=f"Symbol {subject!r} not resolvable in the graph.",
            )
        actual_kind = str(node.get("kind") or "").lower()
        target_kind = (claim.object or "").lower()
        if actual_kind == target_kind:
            return Verdict(
                claim=claim, verified=True,
                reason=f"{subject} is a {target_kind}.",
                evidence=[node],
            )
        return Verdict(
            claim=claim, verified=False,
            reason=(
                f"{subject} is a {actual_kind!r}, not {target_kind!r}."
                if actual_kind else
                f"{subject} has no kind metadata; cannot confirm "
                f"{target_kind!r}."
            ),
            evidence=[node],
            alternatives=[{"actual_kind": actual_kind}] if actual_kind else [],
        )

    # --- in_file (subject + file path) ----------------------------------
    if claim.predicate == "in_file":
        node = resolve_fn(subject)
        if node is None:
            return Verdict(
                claim=claim, verified=None,
                reason=f"Symbol {subject!r} not resolvable in the graph.",
            )
        actual_path = str(node.get("source_file") or node.get("rel_path") or "")
        target_path = claim.object or ""
        if actual_path and target_path and target_path in actual_path:
            return Verdict(
                claim=claim, verified=True,
                reason=f"{subject} lives in {actual_path}.",
                evidence=[node],
            )
        return Verdict(
            claim=claim, verified=False,
            reason=(
                f"{subject} lives in {actual_path!r}, not {target_path!r}."
                if actual_path else
                f"{subject} has no file metadata in the graph."
            ),
            evidence=[node],
            alternatives=[{"actual_path": actual_path}] if actual_path else [],
        )

    # --- calls (X calls Y) ----------------------------------------------
    if claim.predicate == "calls":
        return _verify_outgoing(
            claim, related_fn, relation="calls", reverse=False,
        )

    # --- called_by (X is called by Y) -----------------------------------
    if claim.predicate == "called_by":
        # X is called by Y → Y -[calls]-> X.
        # Use callers_fn(X) and check Y appears in the result.
        payload = callers_fn(subject)
        callers = payload.get("callers") or []
        target = claim.object or ""
        for caller in callers:
            if _matches_symbol(caller, target):
                return Verdict(
                    claim=claim, verified=True,
                    reason=f"{target} calls {subject}.",
                    evidence=[caller],
                )
        return Verdict(
            claim=claim, verified=False,
            reason=(
                f"{target} does NOT call {subject}; "
                f"{len(callers)} other caller(s) found."
            ),
            evidence=[],
            alternatives=callers[:10],
        )

    # --- references / implements ----------------------------------------
    if claim.predicate in ("references", "implements"):
        return _verify_outgoing(
            claim, related_fn, relation=claim.predicate, reverse=False,
        )

    # --- unknown ---------------------------------------------------------
    return Verdict(
        claim=claim, verified=None,
        reason=f"Unsupported predicate: {claim.predicate!r}.",
    )


def _verify_outgoing(
    claim: Claim,
    related_fn: RelatedFn,
    *,
    relation: str,
    reverse: bool,
) -> Verdict:
    """Verify a directional claim by inspecting outgoing edges from
    the subject (or reverse if requested)."""
    subject = claim.subject
    target = claim.object or ""
    payload = related_fn(subject, [relation])
    edges = payload.get("edges") or []
    related = payload.get("related") or []

    # Build a quick "neighbour symbol" set to test against the target.
    matched: dict[str, Any] | None = None
    for node in related:
        if _matches_symbol(node, target):
            matched = node
            break

    if matched is not None:
        return Verdict(
            claim=claim, verified=True,
            reason=(
                f"{subject} {relation} {target} "
                f"(matched node {matched.get('symbol') or matched.get('label') or matched.get('node_id')})."
            ),
            evidence=[matched],
        )

    return Verdict(
        claim=claim, verified=False,
        reason=(
            f"{subject} does NOT {relation} {target}; "
            f"{len(related)} other {relation}-edge target(s) found."
        ),
        evidence=[],
        alternatives=related[:10],
    )


def _matches_symbol(node: dict[str, Any], target: str) -> bool:
    """Loose match: target appears in any of the node's identity
    fields (symbol / label / node_id / qualified_name)."""
    if not target:
        return False
    t = target.lower()
    for key in ("symbol", "label", "node_id", "qualified_name"):
        value = node.get(key)
        if value and t in str(value).lower():
            return True
    return False


__all__ = [
    "Claim",
    "Verdict",
    "parse_claim",
    "verify_claim",
]
