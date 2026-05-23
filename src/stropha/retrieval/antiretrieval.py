"""Anti-retrieval / near-miss detection (WIN-215).

A search for "how does the user-creation flow work" can legitimately
return both the current implementation AND the deprecated v1 still
sitting in the codebase. The agent often can't tell them apart from
a snippet, and ends up writing code that calls the obsolete function
or worse, "fixes" a bug in the wrong version.

Anti-retrieval flags those near-misses without filtering them out —
the agent SHOULD see they exist (otherwise it'd be surprised when
the code base does include them) but with a strong "do not confuse
with the answer" marker so it knows where to spend its attention.

How we detect a near-miss
-------------------------

Two cheap heuristics that catch most real-world cases:

1. **Deprecated path markers** — paths under ``deprecated/`` /
   ``legacy/`` / ``_legacy/`` / ``old/`` / ``vendor/`` /
   ``vendored/``, plus suffix patterns like ``_deprecated.py`` /
   ``_old.py``. Cross-language; works on rel_path alone, no
   content scan.
2. **Deprecation annotations in the snippet** — ``@deprecated``
   (Python), ``@Deprecated`` (Java), ``[Obsolete]`` (C#),
   ``// DEPRECATED`` (any), ``# deprecated`` (Python comment),
   ``Obsolete`` JSDoc tag. We check only the first ~5 lines of
   the snippet so the cost is constant regardless of chunk size.

Both heuristics are deterministic, no LLM, no token cost.

What we don't do
----------------

We don't try to detect "this is v1 of three implementations" via
score-gap analysis or filename heuristics like ``foo_v1.py``. Those
require taste and tuning; deprecated-marker matching covers ~80% of
the value with zero false positives on the kind of paths that
deserve "do not use" warnings.

Output shape
------------

``partition_hits(hits)`` returns ``(positive, negative)`` lists.
``positive`` keeps everything the agent should treat as a real
answer; ``negative`` is the near-miss list. The caller (e.g.
``assemble_context``) can render the two under separate section
headers.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import SearchHit

# Path patterns that mark a chunk as belonging to a deprecated /
# legacy code path. Matched case-insensitively against the full
# ``rel_path`` so a substring match anywhere in the path triggers.
_DEPRECATED_PATH_MARKERS: tuple[str, ...] = (
    "/deprecated/",
    "/legacy/",
    "/_legacy/",
    "/old/",
    "/_old/",
    "/obsolete/",
    "/vendor/",
    "/vendored/",
    "/third_party/",
    "_deprecated.",
    "_legacy.",
    "_old.",
    ".deprecated.",
    ".legacy.",
    ".old.",
)

# Inline markers indicating the chunk itself is deprecated. We scan
# only the first ``_HEADER_SCAN_LINES`` of the snippet so cost is
# O(chunk count), not O(content size).
_HEADER_SCAN_LINES = 5

_DEPRECATED_INLINE_MARKERS: tuple[str, ...] = (
    "@deprecated",           # Python / TS decorator
    "@Deprecated",           # Java
    "[Obsolete]",            # C#
    "[Deprecated]",          # Various
    "# deprecated",          # Python comment
    "# DEPRECATED",          # Python comment, screaming
    "// deprecated",         # C-family comment
    "// DEPRECATED",         # C-family, screaming
    "/* deprecated",         # block comment
    "/* DEPRECATED",         # block comment, screaming
    "<deprecated>",          # docstring tag
    "Obsolete",              # JSDoc tag (looser)
)


def is_deprecated_path(rel_path: str) -> bool:
    """True when ``rel_path`` matches a known deprecated marker."""
    if not rel_path:
        return False
    lower = rel_path.lower()
    return any(marker in lower for marker in _DEPRECATED_PATH_MARKERS)


def is_deprecated_chunk(hit: SearchHit) -> bool:
    """Heuristic: True when the hit's path OR its snippet header
    contains a deprecation marker."""
    if is_deprecated_path(hit.rel_path or ""):
        return True
    if not hit.snippet:
        return False
    head = "\n".join(hit.snippet.splitlines()[:_HEADER_SCAN_LINES])
    head_lower = head.lower()
    # Case-sensitive first (the screaming variants); fall back to
    # case-insensitive on lowercase forms so we catch both
    # ``@deprecated`` and ``@Deprecated`` cheaply.
    for marker in _DEPRECATED_INLINE_MARKERS:
        if marker in head:
            return True
        # Tolerate case variation on the lowercased markers.
        if marker.lower() in head_lower and marker.startswith(("#", "//", "/*", "<")):
            return True
    return False


def partition_hits(
    hits: Sequence[SearchHit],
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Split ``hits`` into (positive, near-miss) lists.

    Order is preserved within each partition. A hit appears in
    exactly one partition.
    """
    positive: list[SearchHit] = []
    negative: list[SearchHit] = []
    for hit in hits:
        if is_deprecated_chunk(hit):
            negative.append(hit)
        else:
            positive.append(hit)
    return positive, negative


__all__ = [
    "is_deprecated_chunk",
    "is_deprecated_path",
    "partition_hits",
]
