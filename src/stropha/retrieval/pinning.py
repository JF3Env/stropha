"""Session-scoped context pinning (WIN-218).

Pinning is the inverse of WIN-217 dedup: instead of "drop chunks
the agent has already seen", pinning marks chunks the agent
*explicitly wants to keep* across subsequent retrievals. The agent's
workflow is:

  agent: pin_chunks(["c1", "c2"])    # "these are the answer"
  agent: search_code("refactor X")    # results include c1 + c2 at top
  agent: search_code("test for X")    # results STILL include c1 + c2
  agent: unpin_chunks(["c2"])         # c2 no longer pinned
  agent: search_code("review X")      # results include c1, not c2

This module owns:

- ``resolve_pins(storage, pin_ids)`` — fetch the pinned hits via the
  WIN-216 ``Storage.lookup_by_chunk_id`` path. Unknown ids return
  ``found=False`` entries that the caller can either render as a
  stale-pin warning or silently drop.
- ``prepend_pins(pinned_hits, search_hits)`` — splice pinned hits at
  the top of a search result list, deduped against the search hits
  by ``chunk_id`` and renumber the ``rank`` field so the response
  shape stays canonical (0-indexed, contiguous).

The session-level pin set lives on ``AppContext.pinned_chunk_ids`` —
same pattern as WIN-217's ``seen_chunk_ids``. EPIC 1.9 (WIN-220)
will eventually back both via the Schema v8 sessions table.

Pinning vs dedup composition
----------------------------

A pinned chunk MUST win over dedup. The agent's "pin" signal is
strictly stronger than "skip already-seen". Implementation: the
dedup filter consults the pin set and excludes pinned ids from
its filter target. A pin also does NOT cause the chunk to be
added to ``seen_chunk_ids`` — pins are an explicit agent decision,
not a result-of-search signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger
from ..models import SearchHit

log = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedPin:
    """One pin lookup outcome.

    ``hit`` is the resolved SearchHit when ``found=True``; ``None``
    otherwise. The frozen dataclass makes pin lists safe to pass
    between layers without surprise mutation.
    """

    chunk_id: str
    found: bool
    hit: SearchHit | None = None


def resolve_pins(storage: Any, pin_ids: Iterable[str]) -> list[ResolvedPin]:
    """Dereference each pin id via the storage's ``lookup_by_chunk_id``.

    Args:
        storage: a Storage instance (must expose ``lookup_by_chunk_id``).
            The contract is loose so adapters / mocks compose cleanly.
        pin_ids: chunk ids to resolve. Duplicates are deduplicated
            before the lookup so the storage isn't queried twice for
            the same id.

    Returns:
        list of :class:`ResolvedPin` — one per *unique* input id.
        Unknown ids surface as ``found=False`` with ``hit=None``.
    """
    seen: set[str] = set()
    out: list[ResolvedPin] = []
    for cid in pin_ids:
        if cid in seen:
            continue
        seen.add(cid)
        hit = storage.lookup_by_chunk_id(cid)
        if hit is None:
            out.append(ResolvedPin(chunk_id=cid, found=False, hit=None))
        else:
            out.append(ResolvedPin(chunk_id=cid, found=True, hit=hit))
    return out


def prepend_pins(
    pinned_hits: list[SearchHit],
    search_hits: list[SearchHit],
) -> list[SearchHit]:
    """Splice ``pinned_hits`` at the front, deduped against ``search_hits``.

    Behaviour:
    - If a pinned chunk_id ALSO appears in ``search_hits``, the
      pinned instance wins (keep the pin at the front, drop the
      duplicate from ``search_hits``). The search hit's score may
      have been higher, but the pin signal is stronger.
    - Pinned hits keep their relative order (caller decides).
    - The output's ``rank`` field is renumbered 0..N-1 so the
      response shape stays canonical regardless of how the input
      ranks looked.
    """
    if not pinned_hits and not search_hits:
        return []
    pinned_ids = {h.chunk_id for h in pinned_hits}
    # Filter search hits that collide with pins.
    surviving_search = [
        h for h in search_hits if h.chunk_id not in pinned_ids
    ]
    combined = list(pinned_hits) + surviving_search
    # Renumber ranks. SearchHit is immutable so use model_copy.
    renumbered: list[SearchHit] = []
    for new_rank, hit in enumerate(combined):
        renumbered.append(hit.model_copy(update={"rank": new_rank}))
    return renumbered


__all__ = [
    "ResolvedPin",
    "prepend_pins",
    "resolve_pins",
]
