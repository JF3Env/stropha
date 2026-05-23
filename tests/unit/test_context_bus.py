"""Unit tests for the shared context bus (WIN-260, EPIC 1.12 #97)."""

from __future__ import annotations

import threading
import time

import pytest

from stropha.api.context_bus import BusEntry, BusSnapshot, ContextBus


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_entry_defaults(self) -> None:
        e = BusEntry(chunk_id="c1")
        assert e.contributed_by == ""
        assert e.tags == []
        assert e.contributed_at == 0.0

    def test_snapshot_defaults(self) -> None:
        s = BusSnapshot()
        assert s.team_id == ""
        assert s.entries == []
        assert s.subscribers == []
        assert s.total_contributions == 0
        assert s.capacity == 0

    def test_frozen(self) -> None:
        e = BusEntry(chunk_id="c1")
        with pytest.raises((AttributeError, Exception)):
            e.chunk_id = "c2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_capacity_positive(self) -> None:
        bus = ContextBus()
        assert bus.capacity() == 512

    def test_zero_capacity_clamped(self) -> None:
        # max(1, 0) = 1
        bus = ContextBus(capacity=0)
        assert bus.capacity() == 1

    def test_negative_capacity_clamped(self) -> None:
        bus = ContextBus(capacity=-10)
        assert bus.capacity() == 1


# ---------------------------------------------------------------------------
# subscribe / unsubscribe
# ---------------------------------------------------------------------------


class TestSubscribe:
    def test_subscribe_then_snapshot_shows_subscriber(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="team-a", session_id="s1")
        snap = bus.snapshot("team-a")
        assert "s1" in snap.subscribers

    def test_subscribe_empty_inputs_noop(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="", session_id="s1")
        bus.subscribe(team_id="team-a", session_id="")
        snap = bus.snapshot("team-a")
        # Neither call should have created the team.
        assert snap.subscribers == []

    def test_unsubscribe_removes_session(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="team-a", session_id="s1")
        bus.subscribe(team_id="team-a", session_id="s2")
        bus.unsubscribe(team_id="team-a", session_id="s1")
        snap = bus.snapshot("team-a")
        assert snap.subscribers == ["s2"]

    def test_unsubscribe_last_garbage_collects_team(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="team-a", session_id="s1")
        bus.publish(team_id="team-a", chunk_id="c1")
        bus.unsubscribe(team_id="team-a", session_id="s1")
        snap = bus.snapshot("team-a")
        assert snap.entries == []
        assert snap.subscribers == []

    def test_unsubscribe_unknown_noop(self) -> None:
        bus = ContextBus()
        bus.unsubscribe(team_id="ghost", session_id="s1")
        # No raise; snapshot still empty.
        assert bus.snapshot("ghost").subscribers == []

    def test_unsubscribe_empty_inputs_noop(self) -> None:
        bus = ContextBus()
        bus.unsubscribe(team_id="", session_id="s1")
        bus.unsubscribe(team_id="team-a", session_id="")
        # No-op; nothing to assert beyond no raise.


# ---------------------------------------------------------------------------
# publish / snapshot
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_to_subscribed_team(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        added = bus.publish(team_id="t", chunk_id="c1", contributed_by="s")
        assert added is True
        snap = bus.snapshot("t")
        assert len(snap.entries) == 1
        assert snap.entries[0].chunk_id == "c1"
        assert snap.entries[0].contributed_by == "s"

    def test_publish_to_unknown_team_rejected(self) -> None:
        bus = ContextBus()
        added = bus.publish(team_id="ghost", chunk_id="c1")
        assert added is False

    def test_publish_empty_inputs_rejected(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        assert bus.publish(team_id="", chunk_id="c1") is False
        assert bus.publish(team_id="t", chunk_id="") is False

    def test_duplicate_id_returns_false(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1")
        added = bus.publish(team_id="t", chunk_id="c1")
        assert added is False
        # But contributions stays at 1 (duplicate doesn't bump).
        assert bus.snapshot("t").total_contributions == 1

    def test_duplicate_bumps_lru(self) -> None:
        bus = ContextBus(capacity=2)
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1")
        bus.publish(team_id="t", chunk_id="c2")
        # Re-publish c1 -> bumps to most-recent.
        bus.publish(team_id="t", chunk_id="c1")
        bus.publish(team_id="t", chunk_id="c3")
        # c2 should have been evicted; c1 + c3 remain.
        ids = [e.chunk_id for e in bus.snapshot("t").entries]
        assert "c1" in ids
        assert "c3" in ids
        assert "c2" not in ids

    def test_capacity_eviction(self) -> None:
        bus = ContextBus(capacity=2)
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1")
        bus.publish(team_id="t", chunk_id="c2")
        bus.publish(team_id="t", chunk_id="c3")
        snap = bus.snapshot("t")
        # Oldest (c1) evicted.
        ids = [e.chunk_id for e in snap.entries]
        assert "c1" not in ids
        assert "c2" in ids
        assert "c3" in ids

    def test_contributions_counter(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1")
        bus.publish(team_id="t", chunk_id="c2")
        bus.publish(team_id="t", chunk_id="c3")
        assert bus.snapshot("t").total_contributions == 3

    def test_tags_propagated(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1", tags=["hot", "shared"])
        snap = bus.snapshot("t")
        assert snap.entries[0].tags == ["hot", "shared"]

    def test_contributed_at_populated(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        before = time.time()
        bus.publish(team_id="t", chunk_id="c1")
        after = time.time()
        snap = bus.snapshot("t")
        assert before <= snap.entries[0].contributed_at <= after


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_unknown_team_empty(self) -> None:
        bus = ContextBus()
        snap = bus.snapshot("ghost")
        assert snap.team_id == "ghost"
        assert snap.entries == []
        assert snap.capacity == bus.capacity()

    def test_subscribers_sorted(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="zeta")
        bus.subscribe(team_id="t", session_id="alpha")
        bus.subscribe(team_id="t", session_id="mike")
        snap = bus.snapshot("t")
        assert snap.subscribers == ["alpha", "mike", "zeta"]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_team(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="t", session_id="s")
        bus.publish(team_id="t", chunk_id="c1")
        bus.reset(team_id="t")
        snap = bus.snapshot("t")
        assert snap.entries == []
        assert snap.subscribers == []

    def test_reset_all(self) -> None:
        bus = ContextBus()
        bus.subscribe(team_id="a", session_id="s")
        bus.subscribe(team_id="b", session_id="s")
        bus.publish(team_id="a", chunk_id="c1")
        bus.publish(team_id="b", chunk_id="c2")
        bus.reset()
        assert bus.snapshot("a").entries == []
        assert bus.snapshot("b").entries == []


# ---------------------------------------------------------------------------
# Concurrency smoke
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_publishes_consistent(self) -> None:
        """Many threads publishing in parallel — final state should
        match the total publish count (modulo capacity eviction)."""
        bus = ContextBus(capacity=1000)
        bus.subscribe(team_id="t", session_id="s")

        def worker(start: int) -> None:
            for i in range(start, start + 50):
                bus.publish(team_id="t", chunk_id=f"c{i}")

        threads = [
            threading.Thread(target=worker, args=(i * 50,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = bus.snapshot("t")
        assert snap.total_contributions == 200
        assert len(snap.entries) == 200


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.api import context_bus as mod
        assert set(mod.__all__) == {
            "BusEntry",
            "BusSnapshot",
            "ContextBus",
        }
