"""Unit tests for session-scoped dedup (WIN-217).

The MVP keeps the seen-chunk-id set on ``AppContext`` (in-memory,
one set per MCP process). EPIC 1.9 (WIN-221) will later swap the
backing store for a Schema v8 sessions table without changing this
call site.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from stropha.models import SearchHit
from stropha.server import AppContext


def _hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/x.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="def f():\n    pass\n",
        language="python",
        kind="function",
        symbol="f",
    )


def _make_app(canned_hits: list[SearchHit]) -> MagicMock:
    """Build an AppContext mock whose SearchEngine returns the canned
    hits on every call. ``seen_chunk_ids`` starts empty; the real
    dedup logic exercises it.

    We deliberately do NOT pass ``spec=AppContext`` because the spec
    restricts attribute access to dataclass-declared fields, and the
    helpers we mock (``search_engine.search`` / ``storage.compute_facets``)
    are deep nested attrs the MagicMock would otherwise reject.
    """
    app = MagicMock()
    app.session_id = "test-session"
    app.seen_chunk_ids = set()
    app.search_engine.search.return_value = canned_hits
    app.storage.compute_facets.return_value = {
        "language": {}, "kind": {}, "repo": {},
    }
    return app


def _make_ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


# ---------------------------------------------------------------------------
# AppContext field
# ---------------------------------------------------------------------------


class TestAppContextField:
    def test_seen_chunk_ids_default_empty(self) -> None:
        """A freshly-constructed AppContext has an empty seen-set."""
        from unittest.mock import MagicMock

        ctx = AppContext(
            config=MagicMock(),
            storage=MagicMock(),
            embedder=MagicMock(),
            search_engine=MagicMock(),
            built=MagicMock(),
            session_id="abc",
        )
        assert ctx.seen_chunk_ids == set()

    def test_seen_chunk_ids_is_mutable_set(self) -> None:
        from unittest.mock import MagicMock

        ctx = AppContext(
            config=MagicMock(),
            storage=MagicMock(),
            embedder=MagicMock(),
            search_engine=MagicMock(),
            built=MagicMock(),
        )
        ctx.seen_chunk_ids.add("c1")
        assert "c1" in ctx.seen_chunk_ids

    def test_two_contexts_have_independent_sets(self) -> None:
        """The default_factory contract — each AppContext gets its own
        set, not a shared module-level singleton."""
        from unittest.mock import MagicMock

        a = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        b = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        a.seen_chunk_ids.add("from-a")
        assert "from-a" not in b.seen_chunk_ids


# ---------------------------------------------------------------------------
# search_code(dedup=False) — no-op pass-through
# ---------------------------------------------------------------------------


class TestSearchCodeDedupOff:
    def test_does_not_accumulate_seen_set(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2")])
        result = search_code("q", ctx=_make_ctx(app))
        # No dedup → seen set stays empty.
        assert app.seen_chunk_ids == set()
        assert len(result.results) == 2

    def test_does_not_filter(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2")])
        # Pre-populate the set; dedup=False should ignore it.
        app.seen_chunk_ids = {"c1"}
        result = search_code("q", ctx=_make_ctx(app))
        # Both hits returned despite c1 being "seen".
        assert {r.chunk_id for r in result.results} == {"c1", "c2"}


# ---------------------------------------------------------------------------
# search_code(dedup=True) — filter + accumulate
# ---------------------------------------------------------------------------


class TestSearchCodeDedupOn:
    def test_accumulates_seen_chunk_ids_on_first_call(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2")])
        result = search_code("q", dedup=True, ctx=_make_ctx(app))
        # First call: no prior state, both hits served and recorded.
        assert {r.chunk_id for r in result.results} == {"c1", "c2"}
        assert app.seen_chunk_ids == {"c1", "c2"}

    def test_filters_previously_seen(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2"), _hit("c3")])
        app.seen_chunk_ids = {"c1", "c2"}
        result = search_code("q", dedup=True, ctx=_make_ctx(app))
        # Only c3 survives the dedup filter.
        assert {r.chunk_id for r in result.results} == {"c3"}
        # And c3 gets added to the set.
        assert app.seen_chunk_ids == {"c1", "c2", "c3"}

    def test_all_filtered_returns_empty_results(self) -> None:
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2")])
        app.seen_chunk_ids = {"c1", "c2"}
        result = search_code("q", dedup=True, ctx=_make_ctx(app))
        # Every hit was already seen → empty list, no exception.
        assert result.results == []
        # Set unchanged.
        assert app.seen_chunk_ids == {"c1", "c2"}

    def test_dedup_runs_before_facets_compute(self) -> None:
        """Facets describe what was *served* after dedup — the agent
        sees the truthful distribution, not the pre-filter shape."""
        from stropha.server import search_code

        app = _make_app([_hit("c1"), _hit("c2")])
        app.seen_chunk_ids = {"c1"}
        search_code(
            "q", dedup=True, include_facets=True, ctx=_make_ctx(app),
        )
        # The compute_facets call should have seen only c2.
        # (mock_calls records the args).
        called_with = app.storage.compute_facets.call_args
        assert called_with is not None
        served_ids = called_with[0][0]
        assert served_ids == ["c2"]


# ---------------------------------------------------------------------------
# reset_dedup tool
# ---------------------------------------------------------------------------


class TestResetDedup:
    def test_clears_non_empty_set(self) -> None:
        from stropha.server import reset_dedup

        app = _make_app([])
        app.seen_chunk_ids = {"c1", "c2", "c3"}
        result = reset_dedup(ctx=_make_ctx(app))
        assert result.cleared == 3
        assert app.seen_chunk_ids == set()

    def test_clear_empty_set_returns_zero(self) -> None:
        from stropha.server import reset_dedup

        app = _make_app([])
        app.seen_chunk_ids = set()
        result = reset_dedup(ctx=_make_ctx(app))
        assert result.cleared == 0

    def test_clearing_makes_search_serve_chunks_again(self) -> None:
        """End-to-end: after reset_dedup, a previously-filtered chunk
        becomes available again."""
        from stropha.server import reset_dedup, search_code

        app = _make_app([_hit("c1")])
        # First dedup'd call: serve c1, record it.
        search_code("q", dedup=True, ctx=_make_ctx(app))
        assert app.seen_chunk_ids == {"c1"}
        # Second dedup'd call: filtered out.
        result_filtered = search_code("q", dedup=True, ctx=_make_ctx(app))
        assert result_filtered.results == []
        # Reset.
        reset_dedup(ctx=_make_ctx(app))
        # Third dedup'd call: served again.
        result_again = search_code("q", dedup=True, ctx=_make_ctx(app))
        assert {r.chunk_id for r in result_again.results} == {"c1"}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_reset_dedup_registered(self) -> None:
        from stropha.server import mcp

        names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "reset_dedup" in names
