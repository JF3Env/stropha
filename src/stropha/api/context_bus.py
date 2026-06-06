"""Shared context bus (WIN-260, EPIC 1.12 #97).

Today every MCP session is independent: the WIN-217 seen-set,
WIN-218 pin-set, WIN-230 cross-session cache, and WIN-222
project-memory all live in their own scope. Two agents on the
same team running queries against the same repo can't share
findings — the cache is local to each session, the seen-set is
private, project memory is the only shared surface and it
requires an explicit ``remember`` call.

WIN-260 ships the bus primitive that ties a group of sessions
together by ``team_id`` (or any shared label). Subscribed
sessions pool their warm-cache hits + their pinned chunks so a
search by Alice surfaces Bob's recent results without forcing
explicit handoff.

The bus is in-memory (per-process) so it doesn't add schema; a
future story will mirror the bus state into the v13
cross_session_cache so the savings persist across MCP-server
restarts.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusEntry:
    """One entry on the shared bus."""

    chunk_id: str
    contributed_by: str = ""
    contributed_at: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BusSnapshot:
    """Read-only view of the bus state for a team."""

    team_id: str = ""
    entries: list[BusEntry] = field(default_factory=list)
    subscribers: list[str] = field(default_factory=list)
    total_contributions: int = 0
    capacity: int = 0


class ContextBus:
    """Thread-safe in-memory pub/sub for context entries.

    The bus is partitioned by ``team_id``. Sessions subscribe to a
    team; published entries appear in the snapshot of every
    subscriber (LRU-bounded by ``capacity``).
    """

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = max(1, int(capacity))
        self._teams: dict[str, OrderedDict[str, BusEntry]] = {}
        self._subscribers: dict[str, set[str]] = {}
        self._contribs: dict[str, int] = {}
        self._lock = threading.Lock()

    def capacity(self) -> int:
        return self._capacity

    def subscribe(self, *, team_id: str, session_id: str) -> None:
        """Add ``session_id`` to ``team_id``'s subscriber set."""
        team = (team_id or "").strip()
        sid = (session_id or "").strip()
        if not team or not sid:
            return
        with self._lock:
            self._teams.setdefault(team, OrderedDict())
            self._subscribers.setdefault(team, set()).add(sid)
            self._contribs.setdefault(team, 0)

    def unsubscribe(self, *, team_id: str, session_id: str) -> None:
        """Remove ``session_id`` from ``team_id``'s subscriber set.

        Idempotent: no-op when the team / session aren't known.
        """
        team = (team_id or "").strip()
        sid = (session_id or "").strip()
        if not team or not sid:
            return
        with self._lock:
            subs = self._subscribers.get(team)
            if subs is None:
                return
            subs.discard(sid)
            if not subs:
                # Garbage-collect the team when no subscribers remain.
                self._subscribers.pop(team, None)
                self._teams.pop(team, None)
                self._contribs.pop(team, None)

    def publish(
        self,
        *,
        team_id: str,
        chunk_id: str,
        contributed_by: str = "",
        tags: Iterable[str] | None = None,
    ) -> bool:
        """Append a chunk_id to ``team_id``'s shared cache.

        Returns True when the entry was actually added (new id);
        False when the bus rejected the publish (team unknown,
        empty inputs, or duplicate id that was just LRU-bumped).
        """
        team = (team_id or "").strip()
        cid = (chunk_id or "").strip()
        if not team or not cid:
            return False
        with self._lock:
            store = self._teams.get(team)
            if store is None:
                # No subscribers / unknown team.
                return False
            if cid in store:
                # LRU bump: move to most-recent end without
                # bumping the contribution count.
                store.move_to_end(cid)
                return False
            store[cid] = BusEntry(
                chunk_id=cid,
                contributed_by=contributed_by,
                contributed_at=time.time(),
                tags=list(tags or []),
            )
            self._contribs[team] = self._contribs.get(team, 0) + 1
            # Evict the LRU when over capacity.
            while len(store) > self._capacity:
                store.popitem(last=False)
            return True

    def snapshot(self, team_id: str) -> BusSnapshot:
        """Return a frozen snapshot of ``team_id``'s entries.

        Unknown teams return an empty snapshot — never raises.
        """
        team = (team_id or "").strip()
        with self._lock:
            store = self._teams.get(team)
            if store is None:
                return BusSnapshot(team_id=team, capacity=self._capacity)
            return BusSnapshot(
                team_id=team,
                entries=list(store.values()),
                subscribers=sorted(self._subscribers.get(team, set())),
                total_contributions=self._contribs.get(team, 0),
                capacity=self._capacity,
            )

    def reset(self, *, team_id: str | None = None) -> None:
        """Clear bus state for one team or every team.

        Useful for tests + operator-initiated cache wipes.
        """
        with self._lock:
            if team_id is None:
                self._teams.clear()
                self._subscribers.clear()
                self._contribs.clear()
                return
            team = team_id.strip()
            self._teams.pop(team, None)
            self._subscribers.pop(team, None)
            self._contribs.pop(team, None)


__all__ = [
    "BusEntry",
    "BusSnapshot",
    "ContextBus",
]
