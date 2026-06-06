"""Extractive chunk compression (WIN-210).

A chunk's full content is the right unit for indexing but often
wasteful at query time. A 60-line method may contain a 3-line
signature, a 12-line docstring, 5 lines that actually matter for the
query, and 40 lines of boilerplate (default-handling, fallback
branches, logging). The agent only needs the first three pieces.

**Extractive compression** picks specific lines from the chunk
rather than generating a summary. It's deterministic, no LLM, no
network — runs in microseconds per chunk. The contract:

1. **Always keep the signature** (the chunk's first non-blank line
   when it looks like a `def` / `class` / `function` / similar
   declaration). This anchors what the chunk IS so the agent doesn't
   have to scroll up to figure out.
2. **Keep matched lines** — every line containing a token from the
   query (CamelCase-split for code identifiers).
3. **Keep a small context window** around each matched line so the
   agent sees the immediate surrounding logic.
4. **Drop boilerplate** — blank-only lines outside the kept window,
   isolated comment lines that don't sit on the signature, top-of-
   chunk import blocks.
5. **Mark elisions** — when consecutive lines are dropped, insert a
   single `# ... (N lines elided)` marker so the agent can tell
   compression happened and roughly how aggressive it was.

The output keeps line-relative meaning: kept lines preserve their
indentation and content, only the lines in between change. This is
critical for code — broken indentation would corrupt the agent's
reading of nested scopes.

Why no LLM
----------

Per-chunk LLM compression would be:
- Slow (1-3s per chunk × 10 chunks per query = blocking).
- Expensive (token cost adds up across thousands of queries).
- Non-deterministic (different output across runs, harder to test).

Extractive compression is cheap enough to run unconditionally and
its output is reproducible. The trade-off is some loss of compression
ratio versus an abstractive summary — but the paper's "+7.7 accuracy
points AND 4× fewer tokens" simultaneous gain shows extractive is
already worth a lot.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..logging import get_logger

log = get_logger(__name__)


# Patterns that identify a chunk's signature line. Order matters —
# the first matching pattern wins. We anchor each at the start of a
# stripped line so leading indentation doesn't break the match.
_SIGNATURE_PATTERNS = (
    re.compile(r"^\s*(?:async\s+)?def\s+\w"),      # Python def / async def
    re.compile(r"^\s*class\s+\w"),                  # Python class
    re.compile(r"^\s*(?:public|private|protected|static|final|abstract|"
               r"override)\s+.*\b(?:class|interface|enum|record)\s+\w"),
    re.compile(r"^\s*(?:public|private|protected|static|final|abstract|"
               r"override|synchronized).*\([^)]*\)"),  # Java method
    re.compile(r"^\s*function\s+\w"),               # JS/TS function
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*="),
    re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum|class)\s+\w"),
    re.compile(r"^\s*fn\s+\w"),                     # Rust
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w"),
    re.compile(r"^\s*func\s+\w"),                   # Go
    re.compile(r"^\s*Feature:\s*"),                  # Gherkin feature
    re.compile(r"^\s*Scenario:\s*"),                 # Gherkin scenario
    re.compile(r"^\s*#{1,6}\s+"),                    # Markdown heading
)

# Identifier-token split: CamelCase + snake_case + dotted boundaries.
_CAMEL_BOUNDARY_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
# Splits on anything that isn't a letter / digit — including the
# underscore. This breaks snake_case (notify_warehouse → notify,
# warehouse) and dotted identifiers (User.create → user, create).
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, CamelCase + snake_case + dotted split, return token set.

    Mirrors the FTS5 indexer's tokenisation so a query that hits the
    indexer also hits the compressor — no surprise mismatches. Both
    CamelCase (NotifyWarehouse → notify, warehouse) and snake_case
    (notify_warehouse → notify, warehouse) collapse to the same token
    set, which is what the agent intuitively expects.
    """
    expanded = _CAMEL_BOUNDARY_1.sub(r"\1 \2", text)
    expanded = _CAMEL_BOUNDARY_2.sub(r"\1 \2", expanded)
    return {
        t.lower() for t in _NON_ALNUM.split(expanded) if len(t) > 1
    }


def _is_signature(line: str) -> bool:
    """Return True when ``line`` looks like a code-level declaration."""
    stripped = line.lstrip()
    if not stripped:
        return False
    return any(pattern.match(line) for pattern in _SIGNATURE_PATTERNS)


def _is_pure_comment(line: str) -> bool:
    """A line that contains ONLY a comment (no code).

    Comment-only lines outside the kept window are usually dead
    weight — license headers, TODOs, dividers. The compressor
    keeps them only when they sit immediately under the signature
    (where they're often docstrings).
    """
    stripped = line.lstrip()
    if not stripped:
        return False
    comment_starters = ("#", "//", "/*", "*", "<!--", '"""', "'''")
    return any(stripped.startswith(c) for c in comment_starters)


@dataclass(frozen=True)
class CompressedSnippet:
    """The result of one extractive compression pass.

    ``text`` is the rewritten snippet; ``original_chars`` /
    ``compressed_chars`` lets the caller compute the ratio. The
    explicit ``kept_lines`` / ``dropped_lines`` counters are
    observability — the operator (or a meta-harness scorer) can
    decide if the compressor is being too aggressive.
    """

    text: str
    original_chars: int
    compressed_chars: int
    kept_lines: int
    dropped_lines: int
    elided_blocks: int = 0
    """How many ``# ... (N lines elided)`` markers were inserted.
    Each marker collapses one run of consecutive dropped lines."""

    @property
    def ratio(self) -> float:
        """Compression ratio: 1.0 = no change, 0.5 = half the size."""
        if self.original_chars <= 0:
            return 1.0
        return self.compressed_chars / self.original_chars


def compress_extractive(
    snippet: str,
    query: str,
    *,
    context_window: int = 2,
    keep_signature: bool = True,
    elide_marker_min_lines: int = 2,
    min_lines_to_compress: int = 6,
) -> CompressedSnippet:
    """Compress ``snippet`` to keep signature + matched lines + context.

    Args:
        snippet: the chunk's full text.
        query: the query string. Tokens drive the matched-line check.
        context_window: lines kept on each side of a matched line.
            Default 2 so the agent sees the immediately-adjacent
            logic. Pass 0 for "matched lines only" (more aggressive
            but loses local context).
        keep_signature: when True, the first signature-shaped line
            anchors the chunk and is always kept. When False, the
            output is matched-lines-only.
        elide_marker_min_lines: minimum number of consecutive dropped
            lines that triggers an "# ... (N lines elided)" marker.
            Below this, the lines are silently dropped — single
            dropped lines don't deserve a 30-char marker.
        min_lines_to_compress: if the snippet has fewer lines than
            this, return it unchanged. Compression overhead isn't
            worth it for tiny chunks.

    Returns:
        :class:`CompressedSnippet` with the rewritten text + stats.
    """
    if not snippet:
        return CompressedSnippet(
            text="", original_chars=0, compressed_chars=0,
            kept_lines=0, dropped_lines=0,
        )

    lines = snippet.splitlines()
    original_chars = len(snippet)

    # Short-circuit: snippets below the threshold are returned as-is.
    if len(lines) < min_lines_to_compress:
        return CompressedSnippet(
            text=snippet,
            original_chars=original_chars,
            compressed_chars=original_chars,
            kept_lines=len(lines),
            dropped_lines=0,
        )

    query_tokens = _tokenize(query)
    # Precompute per-line tokens once — used in match check + adjacency
    # window expansion.
    line_tokens: list[set[str]] = [
        _tokenize(line) if line.strip() else set()
        for line in lines
    ]

    # Decide which lines to keep. We use a bitmap to drive the
    # post-processing (window expansion + elision).
    keep = [False] * len(lines)

    # 1. Signature line — first signature-shaped line in the chunk.
    signature_idx: int | None = None
    if keep_signature:
        for idx, line in enumerate(lines):
            if _is_signature(line):
                keep[idx] = True
                signature_idx = idx
                break

    # 2. Lines matching query tokens.
    for idx, tokens in enumerate(line_tokens):
        if tokens & query_tokens:
            keep[idx] = True

    # 3. Expand around matched lines by ``context_window``. Done in a
    #    separate pass so window expansion of one match doesn't
    #    accidentally extend off another.
    if context_window > 0:
        match_indices = [
            i for i, tokens in enumerate(line_tokens)
            if tokens & query_tokens
        ]
        for mi in match_indices:
            for j in range(
                max(0, mi - context_window),
                min(len(lines), mi + context_window + 1),
            ):
                keep[j] = True

    # 4. If we kept the signature, also keep a docstring block
    #    immediately below it. A docstring opens with ``"""`` or
    #    ``'''`` which ``_is_pure_comment`` already classifies as a
    #    comment-starter, so the loop captures both the opener and
    #    every line of the multi-line body until a closing triple-
    #    quote is seen (which `_is_pure_comment` also matches on
    #    most natural codebases since closers are indented bare
    #    ``"""``). When the docstring body has non-comment-shaped
    #    lines (rare in code but possible in docstring tables /
    #    diagrams), we walk past them by extending through balanced
    #    triple-quotes on the same indent level.
    if signature_idx is not None:
        idx = signature_idx + 1
        # First, consume any comment-shaped lines (including docstring
        # opener).
        in_docstring_quote: str | None = None
        while idx < len(lines):
            line = lines[idx]
            stripped = line.lstrip()
            if in_docstring_quote is not None:
                # Inside a multi-line docstring body. Keep every line
                # until we see the closing triple-quote.
                keep[idx] = True
                if in_docstring_quote in stripped:
                    in_docstring_quote = None
                idx += 1
                continue
            if _is_pure_comment(line):
                keep[idx] = True
                # Detect docstring opening to switch into body mode.
                # An unbalanced triple-quote on this line opens; a
                # balanced one (open + close on same line) doesn't.
                for triple in ('"""', "'''"):
                    if stripped.startswith(triple) and stripped.count(triple) < 2:
                        in_docstring_quote = triple
                        break
                idx += 1
                continue
            break

    # 5. Render the kept lines + elision markers.
    output_parts: list[str] = []
    elided_blocks = 0
    dropped = 0
    run_start: int | None = None
    for idx, line in enumerate(lines):
        if keep[idx]:
            if run_start is not None:
                # Close the dropped run with a marker if it's long
                # enough to warrant one.
                n_dropped = idx - run_start
                if n_dropped >= elide_marker_min_lines:
                    indent = " " * (
                        len(lines[run_start]) - len(lines[run_start].lstrip())
                    )
                    output_parts.append(
                        f"{indent}# ... ({n_dropped} lines elided)"
                    )
                    elided_blocks += 1
                run_start = None
            output_parts.append(line)
        else:
            dropped += 1
            if run_start is None:
                run_start = idx
    # Close a trailing dropped run.
    if run_start is not None:
        n_dropped = len(lines) - run_start
        if n_dropped >= elide_marker_min_lines:
            indent = " " * (
                len(lines[run_start]) - len(lines[run_start].lstrip())
            )
            output_parts.append(
                f"{indent}# ... ({n_dropped} lines elided)"
            )
            elided_blocks += 1

    compressed = "\n".join(output_parts)
    return CompressedSnippet(
        text=compressed,
        original_chars=original_chars,
        compressed_chars=len(compressed),
        kept_lines=sum(keep),
        dropped_lines=dropped,
        elided_blocks=elided_blocks,
    )


def compress_hits(
    hits: Sequence,
    query: str,
    **kwargs,
) -> list:
    """Apply :func:`compress_extractive` to every hit's ``snippet``.

    Returns a new list of hits (uses ``hit.model_copy`` so SearchHit
    immutability is preserved). The compression stats are logged at
    info level for cost-dashboard aggregation.
    """
    out = []
    total_original = 0
    total_compressed = 0
    for hit in hits:
        snippet = getattr(hit, "snippet", "") or ""
        result = compress_extractive(snippet, query, **kwargs)
        total_original += result.original_chars
        total_compressed += result.compressed_chars
        # SearchHit is a pydantic BaseModel → use model_copy to keep
        # immutability while replacing the snippet.
        if hasattr(hit, "model_copy"):
            out.append(hit.model_copy(update={"snippet": result.text}))
        else:
            # Defensive: a non-SearchHit-shaped input still works,
            # just less efficiently. Used by tests with bare objects.
            with contextlib.suppress(AttributeError):
                hit.snippet = result.text
            out.append(hit)
    if total_original > 0:
        log.info(
            "compression.applied",
            hits=len(hits),
            original_chars=total_original,
            compressed_chars=total_compressed,
            ratio=round(total_compressed / total_original, 3)
            if total_original else 1.0,
        )
    return out


__all__ = [
    "CompressedSnippet",
    "compress_extractive",
    "compress_hits",
]
