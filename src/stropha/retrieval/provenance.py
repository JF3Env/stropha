"""Per-hit provenance rendering (WIN-214).

Every fused retrieval result carries a ``streams`` map showing which
streams (dense / sparse / symbol / graph-vec) contributed to its
final rank. WIN-160 added the field on ``SearchHit``; WIN-214 makes
it *visible to the agent* so the agent can self-audit its context:

  - "This hit came from the symbol-token lane → it's a literal
    identifier match, low risk of semantic drift."
  - "This hit came only from dense → it's a fuzzy semantic match,
    worth re-checking the relevance."
  - "This hit came from BOTH dense AND sparse → strong signal."

The label is a single-line string suitable for a markdown bundle:

    > provenance: dense(rrf=0.0164) + sparse(rrf=0.0162)

The scores are RRF contributions (small floats < 0.05 typically) so
we render them with 4 decimals and tag the stream name in parentheses.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..models import SearchHit

# Stream names in canonical order for stable rendering. Streams not
# listed appear AFTER these (alphabetically), so a future stream
# (e.g. "rerank") doesn't disrupt existing output.
_CANONICAL_ORDER: tuple[str, ...] = (
    "dense", "sparse", "symbol", "graph-vec", "graph_vec",
)


def _stream_sort_key(name: str) -> tuple[int, str]:
    """Place canonical streams first (in declared order), then unknown
    streams alphabetically. Sort key — used in ``provenance_label``."""
    if name in _CANONICAL_ORDER:
        return (_CANONICAL_ORDER.index(name), name)
    return (len(_CANONICAL_ORDER), name)


def provenance_label(
    streams: Mapping[str, float] | None,
    *,
    show_scores: bool = True,
    score_decimals: int = 4,
) -> str:
    """Render the per-hit streams map as a one-line markdown comment.

    Args:
        streams: ``{stream_name: rrf_score}`` mapping. ``None`` or
            empty returns an empty string so the caller can skip the
            line entirely.
        show_scores: when True (default), include ``(rrf=0.0164)``
            tags. Pass False for a compact label suitable for the
            ``line`` card.
        score_decimals: number of decimal places to render. Default
            4 keeps the typical RRF magnitude visible without overflow.

    Returns:
        Rendered label, e.g. ``"dense(rrf=0.0164) + sparse(rrf=0.0162)"``.
        Empty string when ``streams`` is empty / None.
    """
    if not streams:
        return ""
    items = sorted(streams.items(), key=lambda kv: _stream_sort_key(kv[0]))
    if not show_scores:
        return " + ".join(name for name, _score in items)
    parts: list[str] = []
    for name, score in items:
        parts.append(f"{name}(rrf={score:.{score_decimals}f})")
    return " + ".join(parts)


def hit_provenance_line(hit: SearchHit, *, show_scores: bool = True) -> str:
    """Convenience: render the markdown ``> provenance: ...`` line for
    a single hit. Returns an empty string when the hit has no
    ``streams`` map (e.g. legacy data, single-stream short-circuit
    with no attribution).
    """
    label = provenance_label(hit.streams, show_scores=show_scores)
    if not label:
        return ""
    return f"> provenance: {label}"


__all__ = [
    "hit_provenance_line",
    "provenance_label",
]
