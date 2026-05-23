"""Unit tests for conversation-aware retrieval (WIN-224).

Covers:
- ``AppContext.recent_queries`` ring buffer (default empty, maxlen=5,
  per-instance isolation).
- ``SearchEngine.search(conversation_context=...)`` appends the
  context to the dense embedding only.
- MCP ``search_code(conversation_aware=True)`` builds the context
  from ``app.recent_queries`` and forwards it.
- Recent-query ring buffer is updated on every call (not gated on
  the flag), so toggling the flag mid-session is immediately useful.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from stropha.models import SearchHit


def _hit(chunk_id: str = "c1") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/x.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="def f(): pass",
        language="python",
        kind="function",
        symbol="f",
    )


# ---------------------------------------------------------------------------
# AppContext.recent_queries field
# ---------------------------------------------------------------------------


class TestRecentQueriesField:
    def test_default_empty_deque(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        assert isinstance(ctx.recent_queries, deque)
        assert len(ctx.recent_queries) == 0

    def test_maxlen_caps_at_five(self) -> None:
        from stropha.server import AppContext

        ctx = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        for i in range(10):
            ctx.recent_queries.append(f"q{i}")
        # Maxlen 5 → only last 5 survive, oldest dropped.
        assert len(ctx.recent_queries) == 5
        assert list(ctx.recent_queries) == ["q5", "q6", "q7", "q8", "q9"]

    def test_per_instance_isolation(self) -> None:
        from stropha.server import AppContext

        a = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        b = AppContext(
            config=MagicMock(), storage=MagicMock(), embedder=MagicMock(),
            search_engine=MagicMock(), built=MagicMock(),
        )
        a.recent_queries.append("from-a")
        assert "from-a" not in b.recent_queries


# ---------------------------------------------------------------------------
# SearchEngine.search(conversation_context=...) wiring
# ---------------------------------------------------------------------------


class TestSearchEngineConversationContext:
    def _engine(self) -> tuple[MagicMock, MagicMock]:
        from stropha.retrieval.search import SearchEngine

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage = MagicMock()
        storage.search_dense.return_value = [_hit("c1")]
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        return SearchEngine(storage, embedder), embedder

    def test_context_appended_to_dense_query_only(self) -> None:
        eng, embedder = self._engine()
        with patch(
            "stropha.retrieval.hyde.maybe_hyde_rewrite", return_value=None
        ):
            eng.search(
                "tests for it",
                conversation_context="how does mastery decay work",
            )
        # The embed_query call should have received the concatenated
        # string containing BOTH the current query AND the context.
        sent = embedder.embed_query.call_args[0][0]
        assert "tests for it" in sent
        assert "how does mastery decay work" in sent
        assert "Recent conversation:" in sent

    def test_no_context_passes_literal_query_to_embedder(self) -> None:
        eng, embedder = self._engine()
        with patch(
            "stropha.retrieval.hyde.maybe_hyde_rewrite", return_value=None
        ):
            eng.search("plain query")
        sent = embedder.embed_query.call_args[0][0]
        assert "Recent conversation:" not in sent

    def test_empty_context_string_treated_as_no_context(self) -> None:
        eng, embedder = self._engine()
        with patch(
            "stropha.retrieval.hyde.maybe_hyde_rewrite", return_value=None
        ):
            eng.search("plain query", conversation_context="   ")
        sent = embedder.embed_query.call_args[0][0]
        assert "Recent conversation:" not in sent

    def test_bm25_and_symbol_search_use_literal_query(self) -> None:
        """The literal-match lanes (BM25, symbol) must NOT see the
        appended conversation context — otherwise spurious tokens
        from the recent history would corrupt the literal-match
        signal."""
        eng, _ = self._engine()
        storage_mock = eng._storage  # type: ignore[attr-defined]
        with patch(
            "stropha.retrieval.hyde.maybe_hyde_rewrite", return_value=None
        ):
            eng.search(
                "current",
                conversation_context="prior turns blah blah",
            )
        # BM25 + symbol must have been called with "current", not
        # the concatenated string.
        assert storage_mock.search_bm25.call_args[0][0] == "current"
        assert storage_mock.search_symbol_tokens.call_args[0][0] == "current"


# ---------------------------------------------------------------------------
# search_code(conversation_aware=True) integration
# ---------------------------------------------------------------------------


def _mcp_app(canned_hits: list[SearchHit]) -> MagicMock:
    """Bare AppContext mock that drives the MCP search_code path."""
    app = MagicMock()
    app.session_id = "test-session"
    app.client_id = None  # no persistence
    app.seen_chunk_ids = set()
    app.pinned_chunk_ids = set()
    app.recent_queries = deque(maxlen=5)
    app.search_engine.search.return_value = canned_hits
    return app


def _ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestSearchCodeConversationAware:
    def test_first_call_no_context_passed(self) -> None:
        """First call in a fresh session — recent_queries is empty,
        so no context is forwarded to the engine."""
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        search_code(
            "first query", conversation_aware=True, ctx=_ctx(app),
        )
        # SearchEngine.search must have been called WITHOUT a context.
        kwargs = app.search_engine.search.call_args.kwargs
        assert kwargs.get("conversation_context") is None

    def test_subsequent_call_blends_recent_queries(self) -> None:
        """After the first call, recent_queries holds the first query
        — the next conversation_aware call sees it as context."""
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        search_code(
            "how does mastery decay work", conversation_aware=True,
            ctx=_ctx(app),
        )
        # Second call.
        search_code(
            "tests for it", conversation_aware=True, ctx=_ctx(app),
        )
        # Second call should have received the first query as context.
        kwargs_second = app.search_engine.search.call_args.kwargs
        assert (
            "how does mastery decay work"
            in kwargs_second.get("conversation_context", "")
        )

    def test_conversation_aware_false_skips_context(self) -> None:
        """When the flag is off, no context is passed even if recent
        queries exist."""
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        # Pre-populate as if prior calls happened.
        app.recent_queries.append("prior 1")
        app.recent_queries.append("prior 2")
        search_code(
            "new query", conversation_aware=False, ctx=_ctx(app),
        )
        kwargs = app.search_engine.search.call_args.kwargs
        # No context forwarded.
        assert kwargs.get("conversation_context") is None

    def test_query_appended_to_recent_buffer(self) -> None:
        """Every search_code call (regardless of the flag) appends to
        the ring buffer so toggling the flag mid-session is
        immediately useful."""
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        search_code("q1", ctx=_ctx(app))  # no flag
        search_code("q2", ctx=_ctx(app))  # no flag
        assert list(app.recent_queries) == ["q1", "q2"]

    def test_empty_query_not_appended(self) -> None:
        """Defensive: empty / whitespace-only queries don't pollute
        the ring buffer."""
        from stropha.server import search_code

        app = _mcp_app([])
        search_code("", ctx=_ctx(app))
        search_code("   ", ctx=_ctx(app))
        assert len(app.recent_queries) == 0

    def test_maxlen_drops_oldest_when_full(self) -> None:
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        for i in range(7):
            search_code(f"q{i}", ctx=_ctx(app))
        # maxlen=5 means q0 and q1 are dropped; the most recent 5 survive.
        assert list(app.recent_queries) == ["q2", "q3", "q4", "q5", "q6"]

    def test_context_joins_with_pipe_separator(self) -> None:
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        # Seed two prior turns.
        app.recent_queries.append("turn a")
        app.recent_queries.append("turn b")
        search_code(
            "current turn", conversation_aware=True, ctx=_ctx(app),
        )
        context = app.search_engine.search.call_args.kwargs.get(
            "conversation_context"
        )
        assert context is not None
        assert "turn a" in context
        assert "turn b" in context
        assert " | " in context

    def test_current_query_not_in_its_own_context(self) -> None:
        """The current query goes to the engine directly; only PRIOR
        queries are in the context string. Otherwise we'd double-count
        the current query's tokens."""
        from stropha.server import search_code

        app = _mcp_app([_hit("c1")])
        app.recent_queries.append("prior turn")
        search_code(
            "current turn", conversation_aware=True, ctx=_ctx(app),
        )
        context = app.search_engine.search.call_args.kwargs.get(
            "conversation_context"
        )
        assert "prior turn" in context
        # Current query is NOT in the context (the engine sees it as
        # the primary query arg).
        assert "current turn" not in context
