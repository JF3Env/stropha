"""Spec-consistency check (WIN-250, EPIC 1.11 #88).

CLAUDE.md / ADR files typically encode invariants the codebase
relies on ("drift detection MUST bake config flags into adapter_id",
"schema migrations are forward-only", "hook foreground <100ms"). An
agent editing related code can silently violate one of those
invariants — the regression only surfaces in review, or worse, in
production.

WIN-250 ships the primitive that flags potential conflicts BEFORE
the agent commits. Given the changed symbols / files + a spec
loader, the check:

1. Walks every spec text the loader returns.
2. Extracts every imperative sentence
   (``MUST``/``MUST NOT``/``NEVER``/``forward-only``/etc.).
3. For each invariant that mentions a changed symbol / file by
   substring, records a ``SpecConflict`` with severity classified
   by the keyword.

The output is a deduplicated list of conflicts the agent can scan
before writing code. The detector is intentionally lexical — no
LLM call, no NLP — so the cost is one regex pass per spec file.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


# Keywords that mark an invariant in the spec text. Order matters
# for severity classification — the first matching keyword wins.
_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("MUST NOT", "HIGH"),
    ("MUST", "HIGH"),
    ("NEVER", "HIGH"),
    ("non-negotiable", "HIGH"),
    ("forward-only", "HIGH"),
    ("required", "HIGH"),
    ("shall not", "HIGH"),
    ("shall", "HIGH"),
    ("should not", "MEDIUM"),
    ("should", "MEDIUM"),
    ("recommended", "MEDIUM"),
    ("avoid", "MEDIUM"),
    ("note:", "LOW"),
)

# Sentence splitter — broader than citation's because spec text has
# bullets, lists, and code blocks.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class SpecInvariant:
    """One imperative sentence from a spec file."""

    text: str
    spec_file: str
    line: int
    keyword: str
    severity: str


@dataclass(frozen=True)
class SpecConflict:
    """A potential conflict between a changed input and a spec
    invariant."""

    invariant: SpecInvariant
    matched: str
    severity: str
    reason: str = ""


@dataclass(frozen=True)
class SpecConsistencyReport:
    """Aggregate of one :func:`check_spec_consistency` invocation."""

    invariants_found: int = 0
    conflicts: list[SpecConflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    spec_files_scanned: list[str] = field(default_factory=list)


SpecLoaderFn = Callable[[], dict[str, str]]


def extract_invariants(
    text: str, spec_file: str = "",
) -> list[SpecInvariant]:
    """Walk ``text`` sentence-by-sentence; classify each one carrying
    an invariant keyword.

    Line numbers are 1-indexed and refer to the first line of the
    sentence within the original text.
    """
    if not text:
        return []
    invariants: list[SpecInvariant] = []
    search_pos = 0
    for sentence in _SENTENCE_RE.split(text):
        clean = sentence.strip()
        if not clean:  # pragma: no cover - defensive: split on non-empty text yields non-empty chunks
            continue
        keyword: str | None = None
        severity: str = "LOW"
        for kw, sev in _KEYWORDS:
            # ``(?<!\w)`` / ``(?!\w)`` boundary works for keywords
            # ending in non-word chars (``note:``) where ``\b``
            # would fail because the boundary requires a transition
            # *to* a word char.
            if re.search(
                rf"(?<!\w){re.escape(kw)}(?!\w)", clean, re.IGNORECASE,
            ):
                keyword = kw
                severity = sev
                break
        if keyword is None:
            # Advance search_pos past this sentence to keep line
            # numbers monotonic.
            located = text.find(clean, search_pos)
            search_pos = (
                located + len(clean) if located >= 0 else search_pos
            )
            continue
        located = text.find(clean, search_pos)
        if located < 0:  # pragma: no cover - defensive: stripped sentence is always a substring of the source
            located = search_pos
        line = text.count("\n", 0, located) + 1
        invariants.append(SpecInvariant(
            text=clean,
            spec_file=spec_file,
            line=line,
            keyword=keyword,
            severity=severity,
        ))
        search_pos = located + len(clean)
    return invariants


def check_spec_consistency(
    changed_symbols: list[str],
    changed_files: list[str],
    *,
    spec_loader: SpecLoaderFn,
) -> SpecConsistencyReport:
    """Cross-reference the changes against spec invariants.

    Args:
        changed_symbols: symbol names the agent is about to touch.
        changed_files: file paths the agent is about to touch.
        spec_loader: callable returning a ``{path: content}`` mapping
            of spec files to inspect. Typically wraps ``Storage.spec_loader``
            or a filesystem walk over ``docs/architecture/`` +
            ``CLAUDE.md``.

    Returns:
        :class:`SpecConsistencyReport` with the list of conflicts
        sorted by severity (HIGH > MEDIUM > LOW), spec file scanned,
        and total invariants found.
    """
    notes: list[str] = []
    scanned: list[str] = []
    invariants_total = 0
    conflicts: list[SpecConflict] = []

    try:
        specs = spec_loader() or {}
    except Exception as exc:
        return SpecConsistencyReport(
            invariants_found=0,
            conflicts=[],
            notes=[f"spec_loader raised: {exc!r}"],
            spec_files_scanned=[],
        )

    # Build the union of search tokens (changed symbols + the base
    # filenames of changed files). Symbol names like ``FooClass``
    # match substring-insensitive; file paths match by basename.
    tokens: list[tuple[str, str]] = []  # (search_token, label)
    for symbol in changed_symbols:
        name = (symbol or "").strip()
        if name:
            tokens.append((name.lower(), name))
    for path in changed_files:
        p = (path or "").strip()
        if not p:
            continue
        # Add both the full path and the basename so the spec text
        # can match either ("stropha/server.py" or "server.py").
        tokens.append((p.lower(), p))
        base = p.rsplit("/", 1)[-1]
        if base != p:
            tokens.append((base.lower(), base))

    if not tokens:
        notes.append("No changed_symbols / changed_files supplied.")

    seen_keys: set[tuple[str, str, str]] = set()
    for path, body in specs.items():
        scanned.append(path)
        invariants = extract_invariants(body, spec_file=path)
        invariants_total += len(invariants)
        if not tokens:
            continue
        body_lower = body.lower()
        for inv in invariants:
            inv_lower = inv.text.lower()
            for token_lower, label in tokens:
                if token_lower in inv_lower:
                    key = (path, inv.text, label)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    conflicts.append(SpecConflict(
                        invariant=inv,
                        matched=label,
                        severity=inv.severity,
                        reason=(
                            f"Spec {path}:{inv.line} ({inv.keyword}) "
                            f"references {label!r} — verify the change "
                            "preserves the invariant."
                        ),
                    ))
        # ``body_lower`` is intentionally available for future
        # whole-document heuristics; unused for now.
        _ = body_lower

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    conflicts.sort(
        key=lambda c: (severity_rank.get(c.severity, 99), c.invariant.spec_file,
                       c.invariant.line),
    )

    return SpecConsistencyReport(
        invariants_found=invariants_total,
        conflicts=conflicts,
        notes=notes,
        spec_files_scanned=scanned,
    )


__all__ = [
    "SpecConflict",
    "SpecConsistencyReport",
    "SpecInvariant",
    "check_spec_consistency",
    "extract_invariants",
]
