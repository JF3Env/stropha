"""Recursive retrieval / auto-merging (Phase 3 §6.4).

Post-processing pass that takes the top-K hits from the hybrid retriever
and expands them in two ways:

1. **Parent merging** — when a chunk's parent (via ``parent_chunk_id``)
   would have been merged into the same answer anyway, return the
   parent's snippet instead. Reduces fragmentation when several methods
   of the same class all match.

2. **Adjacent merging** — when two retrieved chunks come from the same
   file and are within ``adjacency_lines`` of each other, merge their
   snippets into a single hit covering the union of their line ranges.
   Recovers context that exact-chunk-boundary retrieval misses.

Local-only, no LLM, no embeddings. Pure SQL + dataclass juggling.

Configuration (stropha.yaml):
    retrieval_features:
      recursive_enabled: true
      recursive_adjacency: 5  # line gap considered adjacent
"""

from __future__ import annotations

from collections.abc import Sequence

from ..config import get_config
from ..logging import get_logger
from ..models import SearchHit
from ..storage import Storage

log = get_logger(__name__)

DEFAULT_ADJACENCY_LINES = 5


def _enabled() -> bool:
    """Check if recursive retrieval is enabled via stropha.yaml."""
    return get_config().retrieval_features.recursive_enabled


def _adjacency_lines() -> int:
    """Get the adjacency line gap from stropha.yaml."""
    return get_config().retrieval_features.recursive_adjacency


def merge_hits(
    hits: Sequence[SearchHit],
    storage: Storage,
    *,
    adjacency_lines: int | None = None,
    enabled: bool | None = None,
) -> list[SearchHit]:
    """Return ``hits`` after auto-merge passes.

    Pass-through when disabled or input is empty.

    Order preservation: the first occurrence of any merged group keeps
    its rank. Subsequent merged-in hits are dropped from the result.
    """
    if enabled is None:
        enabled = _enabled()
    if not enabled or not hits:
        return list(hits)
    adjacency_lines = adjacency_lines if adjacency_lines is not None else _adjacency_lines()

    out: list[SearchHit] = []
    used_chunk_ids: set[str] = set()
    # Bucket by rel_path for the adjacency pass.
    by_path: dict[str, list[SearchHit]] = {}
    for h in hits:
        if h.chunk_id in used_chunk_ids:
            continue
        merged = _maybe_merge_parent(h, hits, storage, used_chunk_ids)
        # Dedup on the *emitted* chunk_id, not just the input id. When several
        # sibling hits share a parent, each promotes to the SAME parent chunk;
        # without this guard that parent is appended once per sibling and
        # surfaces at multiple ranks (the "same chunk at ranks 1,2,3,6"
        # duplication). Emit the parent once; mark the consumed sibling so the
        # top-of-loop guard skips it if it recurs.
        if merged.chunk_id in used_chunk_ids:
            used_chunk_ids.add(h.chunk_id)
            continue
        used_chunk_ids.add(h.chunk_id)
        used_chunk_ids.add(merged.chunk_id)
        by_path.setdefault(merged.rel_path, []).append(merged)
        out.append(merged)

    # Adjacency merge — walk each file's hits in start_line order, merge
    # any pair whose gap < adjacency_lines.
    final: list[SearchHit] = []
    for h in out:
        siblings = by_path.get(h.rel_path, [])
        if len(siblings) <= 1:
            final.append(h)
            continue
        # Already-merged check: is there an earlier hit on this path that
        # already covers this one's range?
        absorbed = False
        for i, prev in enumerate(final):
            if (
                prev.rel_path == h.rel_path
                and h.chunk_id != prev.chunk_id
                and _adjacent_or_overlap(prev, h, adjacency_lines)
            ):
                final[i] = _absorb(prev, h, storage)
                absorbed = True
                break
        if not absorbed:
            final.append(h)
    if len(final) != len(hits):
        log.info(
            "recursive.merged",
            input_n=len(hits), output_n=len(final),
            adjacency_lines=adjacency_lines,
        )
    return final


# --------------------------------------------------------------------- helpers


def _maybe_merge_parent(
    hit: SearchHit,
    all_hits: Sequence[SearchHit],
    storage: Storage,
    used_chunk_ids: set[str],
) -> SearchHit:
    """If any sibling in ``all_hits`` shares ``hit``'s parent_chunk_id, promote
    the result to the parent chunk (loaded from storage).

    Backend-agnostic: uses the typed ``get_chunk_by_id`` / ``lookup_by_chunk_id``
    StorageStage methods (implemented by both the sqlite-vec and postgres
    backends) instead of reaching into the SQLite-specific ``_conn`` handle.
    """
    rec = storage.get_chunk_by_id(hit.chunk_id)
    if rec is None or not rec.get("parent_chunk_id"):
        return hit
    parent_id = rec["parent_chunk_id"]
    # Look for any other hit in the result set sharing this parent.
    has_sibling = any(
        h.chunk_id != hit.chunk_id
        and (sib := storage.get_chunk_by_id(h.chunk_id)) is not None
        and sib.get("parent_chunk_id") == parent_id
        for h in all_hits
    )
    if not has_sibling:
        return hit
    parent_hit = storage.lookup_by_chunk_id(parent_id)
    if parent_hit is None:
        return hit
    return parent_hit.model_copy(update={"score": hit.score, "rank": hit.rank})


def _adjacent_or_overlap(a: SearchHit, b: SearchHit, gap: int) -> bool:
    if a.rel_path != b.rel_path:
        return False
    lo_a, hi_a = a.start_line, a.end_line
    lo_b, hi_b = b.start_line, b.end_line
    if hi_a < lo_b:
        return (lo_b - hi_a) <= gap
    if hi_b < lo_a:
        return (lo_a - hi_b) <= gap
    return True  # overlap


def _absorb(host: SearchHit, other: SearchHit, storage: Storage) -> SearchHit:
    """Return a merged hit covering host ∪ other."""
    new_start = min(host.start_line, other.start_line)
    new_end = max(host.end_line, other.end_line)
    # Re-fetch the content spanning the merged range. For simplicity we
    # concatenate the two snippets; a fancier path would read the file
    # from disk, but that introduces I/O on the query hot path.
    new_snippet = host.snippet
    if other.snippet and other.snippet not in host.snippet:
        new_snippet = f"{host.snippet}\n…\n{other.snippet}"
    return host.model_copy(update={
        "start_line": new_start,
        "end_line": new_end,
        "snippet": new_snippet[:1200],
        # Keep the higher of the two scores so RRF semantics aren't lost.
        "score": max(host.score, other.score),
    })
