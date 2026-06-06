"""Citation enforcement (WIN-249, EPIC 1.11 #87).

LLM agents make claims about a codebase ("``FooClass`` validates X
before persisting", "tests live in ``tests/foo_test.py``"). Without
enforcement, those claims can drift into vibes — assertions that
sound right but aren't backed by anything checkable.

WIN-249 audits a piece of agent output for verifiable citations.
Each claim is expected to be followed by one or more references
in any of these shapes:

- ``[chunk_id]`` — exact ``sha256:...`` chunk id from a stropha
  search result.
- ``[<file>:<start>-<end>]`` — file path with line range.
- ``[<file>:<line>]`` — single-line variant.

The audit:

1. Splits the input into sentence-shaped claims.
2. For each claim, extracts every citation token (regex-based).
3. Resolves the citation via a caller-supplied ``lookup_fn``.
4. Reports counts (cited / uncited / broken citations) and
   per-claim breakdowns.

Like every WIN-23x / WIN-24x primitive shipped to date, this
module is standalone — the MCP wrapper is the thin adapter in
``server.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# Match `[chunk_id]`, `[file:line]`, `[file:start-end]`.
# The id is anything that doesn't contain `]`. We then post-process
# to classify chunk_id vs file:line.
_CITATION_RE = re.compile(r"\[([^\[\]\n]+?)\]")

# A chunk_id looks like `sha256:<hex>` or any `<scheme>:<rest>` where
# the scheme isn't a known file extension.
_CHUNK_ID_RE = re.compile(r"^(sha256:[0-9a-f]{64}|[a-zA-Z][a-zA-Z0-9_-]*:[0-9a-zA-Z._/-]+)$")

# A file:line citation. Accepts `path/to/file.py:42` or
# `path/to/file.py:42-58`.
_FILE_LINE_RE = re.compile(
    r"^(?P<path>[^\s:]+\.[A-Za-z0-9]+):(?P<start>\d+)(?:-(?P<end>\d+))?$"
)

# Sentence-splitter: break on `.!?` followed by whitespace + capital
# letter OR end-of-string. Keeps headings (no terminal punctuation)
# as their own claims via the paragraph fallback.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class Citation:
    """One citation extracted from a claim."""

    fragment: str
    kind: str  # "chunk_id" | "file_line" | "unknown"
    chunk_id: str = ""
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    resolved: bool | None = None
    reason: str = ""


@dataclass(frozen=True)
class ClaimAudit:
    """One claim's citations + outcome."""

    sentence: str
    citations: list[Citation] = field(default_factory=list)
    valid_count: int = 0


@dataclass(frozen=True)
class CitationReport:
    """Aggregate of one :func:`audit_citations` invocation."""

    claims: list[ClaimAudit] = field(default_factory=list)
    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: int = 0
    broken_citations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


LookupFn = Callable[[Citation], bool]


def split_claims(text: str) -> list[str]:
    """Split ``text`` into sentence-shaped claims.

    Paragraph boundaries (blank lines) split first; each paragraph
    is then split on terminal punctuation followed by whitespace +
    capital letter. Headings + bullet items with no terminal
    punctuation pass through as single claims.
    """
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    for para in paragraphs:
        # Collapse internal newlines so the sentence regex sees a flat string.
        flat = " ".join(line.strip() for line in para.splitlines() if line.strip())
        if not flat:  # pragma: no cover - defensive: outer filter already drops all-whitespace paragraphs
            continue
        sentences = _SENT_RE.split(flat)
        for sent in sentences:
            stripped = sent.strip()
            if stripped:
                out.append(stripped)
    return out


def classify_citation(fragment: str) -> Citation:
    """Classify a single ``[...]`` fragment into a :class:`Citation`."""
    raw = fragment.strip()
    chunk_match = _CHUNK_ID_RE.match(raw)
    if chunk_match:
        return Citation(
            fragment=raw, kind="chunk_id", chunk_id=raw,
        )
    file_match = _FILE_LINE_RE.match(raw)
    if file_match:
        path = file_match.group("path")
        start = int(file_match.group("start"))
        end_raw = file_match.group("end")
        end = int(end_raw) if end_raw is not None else start
        return Citation(
            fragment=raw, kind="file_line",
            path=path, start_line=start, end_line=end,
        )
    return Citation(
        fragment=raw, kind="unknown",
        reason="Citation matched neither chunk_id nor file:line shape.",
    )


def extract_citations(sentence: str) -> list[Citation]:
    """Return every citation token in ``sentence``."""
    return [
        classify_citation(match.group(1))
        for match in _CITATION_RE.finditer(sentence)
    ]


def audit_citations(
    text: str,
    *,
    lookup_fn: LookupFn,
) -> CitationReport:
    """Audit ``text`` for verifiable citations.

    Args:
        text: free-form agent output.
        lookup_fn: callable that returns True when a citation
            resolves against the index. Receives the :class:`Citation`
            so the wrapper can dispatch based on ``kind``.

    Returns:
        :class:`CitationReport` with per-claim breakdown, aggregate
        counts, and the broken-citation list.
    """
    claims_text = split_claims(text)
    claims: list[ClaimAudit] = []
    broken: list[str] = []
    notes: list[str] = []
    cited = 0
    uncited = 0

    for sentence in claims_text:
        cits = extract_citations(sentence)
        resolved: list[Citation] = []
        valid_count = 0
        for cit in cits:
            if cit.kind == "unknown":
                resolved.append(cit)
                continue
            try:
                ok = bool(lookup_fn(cit))
            except Exception as exc:
                notes.append(
                    f"lookup raised for {cit.fragment!r}: {exc!r}"
                )
                ok = False
            updated = Citation(
                fragment=cit.fragment,
                kind=cit.kind,
                chunk_id=cit.chunk_id,
                path=cit.path,
                start_line=cit.start_line,
                end_line=cit.end_line,
                resolved=ok,
                reason=(
                    "" if ok else
                    f"Citation {cit.fragment!r} did not resolve."
                ),
            )
            resolved.append(updated)
            if ok:
                valid_count += 1
            else:
                broken.append(cit.fragment)

        claims.append(ClaimAudit(
            sentence=sentence, citations=resolved, valid_count=valid_count,
        ))
        if not cits:
            uncited += 1
        else:
            cited += 1

    return CitationReport(
        claims=claims,
        total_claims=len(claims),
        cited_claims=cited,
        uncited_claims=uncited,
        broken_citations=broken,
        notes=notes,
    )


__all__ = [
    "Citation",
    "CitationReport",
    "ClaimAudit",
    "audit_citations",
    "classify_citation",
    "extract_citations",
    "split_claims",
]
