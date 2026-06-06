"""Unit tests for context pinning (WIN-218).

Covers:
- ``stropha.retrieval.pinning`` library (resolve_pins, prepend_pins,
  ResolvedPin).
- ``AppContext.pinned_chunk_ids`` field default + isolation.
- MCP tools: pin_chunks / unpin_chunks / list_pins.
- search_code integration: pinned chunks always appear, win over
  dedup, never get added to seen_chunk_ids.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from stropha.models import SearchHit
from stropha.retrieval.pinning import (
    prepend_pins,
    resolve_pins,
)


def _hit(
    chunk_id: str,
    *,
    rel_path: str = "src/x.py",
    rank: int = 0,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path=rel_path,
        start_line=1,
        end_line=10,
        score=0.5,
        rank=rank,
        snippet=f"def {chunk_id}():\n    pass\n",
        language="python",
        kind="function",
        symbol=chunk_id,
    )


def _make_app(canned_hits: list[SearchHit]) -> MagicMock:
    app = MagicMock()
    app.session_id = "test-session"
    app.seen_chunk_ids = set()
    app.pinned_chunk_ids = set()
    app.search_engine.search.return_value = canned_hits
    app.storage.compute_facets.return_value = {
        "language": {}, "kind": {}, "repo": {},
    }
    # Default: storage.lookup_by_chunk_id returns None for unknown ids;
    # tests that pin specific ids override this.
    app.storage.lookup_by_chunk_id.return_value = None
    return app


def _make_ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


# ---------------------------------------------------------------------------
# resolve_pins
# ---------------------------------------------------------------------------


class TestResolvePins:
    def test_resolves_known_ids(self) -> None:
        storage = MagicMock()
        storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid) if cid in {"a", "b"} else None
        )
        out = resolve_pins(storage, ["a", "b"])
        assert len(out) == 2
        assert all(r.found for r in out)
        assert [r.chunk_id for r in out] == ["a", "b"]

    def test_unknown_id_surfaces_as_not_found(self) -> None:
        storage = MagicMock()
        storage.lookup_by_chunk_id.return_value = None
        out = resolve_pins(storage, ["ghost"])
        assert len(out) == 1
        assert out[0].found is False
        assert out[0].hit is None

    def test_dedupes_input_before_lookup(self) -> None:
        storage = MagicMock()
        storage.lookup_by_chunk_id.return_value = _hit("a")
        # Pass the same id twice — storage should only be queried once.
        out = resolve_pins(storage, ["a", "a", "a"])
        assert len(out) == 1
        assert storage.lookup_by_chunk_id.call_count == 1

    def test_empty_input(self) -> None:
        storage = MagicMock()
        assert resolve_pins(storage, []) == []
        storage.lookup_by_chunk_id.assert_not_called()

    def test_mixed_found_and_missing(self) -> None:
        storage = MagicMock()
        storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid) if cid == "a" else None
        )
        out = resolve_pins(storage, ["a", "missing"])
        assert out[0].found is True
        assert out[1].found is False


# ---------------------------------------------------------------------------
# prepend_pins
# ---------------------------------------------------------------------------


class TestPrependPins:
    def test_pins_prepended_to_results(self) -> None:
        pinned = [_hit("p1"), _hit("p2")]
        search = [_hit("s1"), _hit("s2")]
        out = prepend_pins(pinned, search)
        assert [h.chunk_id for h in out] == ["p1", "p2", "s1", "s2"]

    def test_pin_collision_dedups_against_search(self) -> None:
        """A search hit that happens to be pinned must not appear
        twice. The pinned instance wins."""
        pinned = [_hit("dup", rel_path="src/pinned.py")]
        search = [
            _hit("dup", rel_path="src/search.py"),  # same id, different path
            _hit("other"),
        ]
        out = prepend_pins(pinned, search)
        assert [h.chunk_id for h in out] == ["dup", "other"]
        # The pinned instance won — its rel_path is what survives.
        assert out[0].rel_path == "src/pinned.py"

    def test_ranks_renumbered_from_zero(self) -> None:
        pinned = [_hit("p", rank=999)]  # bogus rank
        search = [_hit("s", rank=999)]  # bogus rank
        out = prepend_pins(pinned, search)
        assert out[0].rank == 0
        assert out[1].rank == 1

    def test_empty_pins_passes_search_through(self) -> None:
        search = [_hit("s1"), _hit("s2")]
        out = prepend_pins([], search)
        assert [h.chunk_id for h in out] == ["s1", "s2"]
        # Ranks still renumbered.
        assert out[0].rank == 0
        assert out[1].rank == 1

    def test_empty_search_with_pins(self) -> None:
        pinned = [_hit("p1")]
        out = prepend_pins(pinned, [])
        assert len(out) == 1
        assert out[0].rank == 0

    def test_both_empty_returns_empty(self) -> None:
        assert prepend_pins([], []) == []


# ---------------------------------------------------------------------------
# AppContext field
# ---------------------------------------------------------------------------


class TestAppContextPinField:
    def test_pinned_chunk_ids_default_empty(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        assert ctx.pinned_chunk_ids == set()

    def test_two_contexts_have_independent_pin_sets(self) -> None:
        from stropha.server import AppContext

        a = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        b = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        a.pinned_chunk_ids.add("p1")
        assert "p1" not in b.pinned_chunk_ids


# ---------------------------------------------------------------------------
# pin_chunks / unpin_chunks MCP tools
# ---------------------------------------------------------------------------


class TestPinTools:
    def test_pin_chunks_adds_to_set(self) -> None:
        from stropha.server import pin_chunks

        app = _make_app([])
        result = pin_chunks(["a", "b"], ctx=_make_ctx(app))
        assert app.pinned_chunk_ids == {"a", "b"}
        assert result.added == 2
        assert result.pinned == ["a", "b"]

    def test_pin_chunks_idempotent(self) -> None:
        from stropha.server import pin_chunks

        app = _make_app([])
        app.pinned_chunk_ids = {"a"}
        result = pin_chunks(["a", "a"], ctx=_make_ctx(app))
        # Pinning existing ids is a no-op (set semantics).
        assert app.pinned_chunk_ids == {"a"}
        assert result.added == 0

    def test_unpin_chunks_removes_from_set(self) -> None:
        from stropha.server import unpin_chunks

        app = _make_app([])
        app.pinned_chunk_ids = {"a", "b", "c"}
        result = unpin_chunks(["a", "c"], ctx=_make_ctx(app))
        assert app.pinned_chunk_ids == {"b"}
        assert result.removed == 2

    def test_unpin_unknown_id_silently_noop(self) -> None:
        from stropha.server import unpin_chunks

        app = _make_app([])
        app.pinned_chunk_ids = {"a"}
        result = unpin_chunks(["never-pinned"], ctx=_make_ctx(app))
        assert app.pinned_chunk_ids == {"a"}
        assert result.removed == 0


# ---------------------------------------------------------------------------
# list_pins MCP tool
# ---------------------------------------------------------------------------


class TestListPins:
    def test_empty_set_returns_empty(self) -> None:
        from stropha.server import list_pins

        app = _make_app([])
        result = list_pins(ctx=_make_ctx(app))
        assert result.entries == []
        assert result.found_count == 0
        assert result.missing_count == 0

    def test_resolves_each_pin(self) -> None:
        from stropha.server import list_pins

        app = _make_app([])
        app.pinned_chunk_ids = {"a", "b"}
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid)
        )
        result = list_pins(ctx=_make_ctx(app))
        assert result.found_count == 2
        assert result.missing_count == 0
        assert all(e.found for e in result.entries)

    def test_missing_pin_surfaces_as_not_found(self) -> None:
        from stropha.server import list_pins

        app = _make_app([])
        app.pinned_chunk_ids = {"ghost"}
        app.storage.lookup_by_chunk_id.return_value = None
        result = list_pins(ctx=_make_ctx(app))
        assert result.missing_count == 1
        assert result.entries[0].found is False
        assert result.entries[0].result is None


# ---------------------------------------------------------------------------
# search_code integration — pin prepending
# ---------------------------------------------------------------------------


class TestSearchCodePinning:
    def test_pinned_chunks_prepended_to_results(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("s1"), _hit("s2")])
        app.pinned_chunk_ids = {"p1"}
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid) if cid == "p1" else None
        )
        result = search_code("q", ctx=_make_ctx(app))
        # p1 first, then s1 and s2.
        ids = [r.chunk_id for r in result.results]
        assert ids[0] == "p1"
        assert "s1" in ids
        assert "s2" in ids

    def test_pinned_dedup_against_search_collision(self) -> None:
        """If a chunk is BOTH pinned and in the search results, it
        should appear exactly once."""
        from stropha.server import search_code

        app = _make_app([_hit("dup"), _hit("other")])
        app.pinned_chunk_ids = {"dup"}
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid)
        )
        result = search_code("q", ctx=_make_ctx(app))
        ids = [r.chunk_id for r in result.results]
        assert ids.count("dup") == 1

    def test_no_pins_no_storage_lookup(self) -> None:
        """When the pin set is empty, the prepend logic short-circuits
        and storage.lookup_by_chunk_id is NEVER called."""
        from stropha.server import search_code

        app = _make_app([_hit("s1")])
        # Pin set is empty by default.
        search_code("q", ctx=_make_ctx(app))
        app.storage.lookup_by_chunk_id.assert_not_called()


# ---------------------------------------------------------------------------
# Composition: pin wins over dedup
# ---------------------------------------------------------------------------


class TestPinDedupComposition:
    def test_pinned_chunk_not_filtered_by_dedup(self) -> None:
        """A pinned chunk that was previously seen MUST still appear.
        Pin wins over dedup."""
        from stropha.server import search_code

        app = _make_app([_hit("pinned"), _hit("other")])
        app.pinned_chunk_ids = {"pinned"}
        app.seen_chunk_ids = {"pinned", "other"}  # both previously seen
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid)
        )
        result = search_code("q", dedup=True, ctx=_make_ctx(app))
        ids = [r.chunk_id for r in result.results]
        # Pinned chunk survives dedup; "other" gets filtered.
        assert "pinned" in ids
        assert "other" not in ids

    def test_pinned_chunk_not_added_to_seen_set(self) -> None:
        """Pinned chunks shouldn't accumulate into seen_chunk_ids —
        otherwise a future unpin + dedup'd search would drop them."""
        from stropha.server import search_code

        app = _make_app([_hit("regular"), _hit("pinned")])
        app.pinned_chunk_ids = {"pinned"}
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: _hit(cid)
        )
        search_code("q", dedup=True, ctx=_make_ctx(app))
        # Only "regular" should be in the seen set.
        assert app.seen_chunk_ids == {"regular"}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_pin_chunks_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "pin_chunks" in names

    def test_unpin_chunks_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "unpin_chunks" in names

    def test_list_pins_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "list_pins" in names
