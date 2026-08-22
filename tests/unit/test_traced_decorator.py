"""Unit tests for the @traced decorator and the surrounding session-id wiring (WIN-160).

The decorator must:

1. Call the wrapped tool transparently (return value, signature preserved).
2. Hand a properly-shaped trace event to ``app.trace_recorder.record``.
3. Capture latency for the wrapped call only — not for the recording step.
4. Survive when ``ctx`` is missing, when the AppContext has no recorder,
   when the wrapped tool raises, and when the recorder itself raises.
5. Strip ``ctx`` from the params snapshot; promote ``query`` to a top-level field.

Plus a couple of integration-flavoured tests:

- ``AppContext.session_id`` is a real field with a string default — required
  for the ``@dataclass`` to instantiate cleanly even in tests that bypass
  the lifespan.
- ``server._ctx`` binds ``session_id`` to structlog contextvars so every
  log emitted during the request carries it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import deque
from typing import Any
from unittest.mock import MagicMock

import pytest

from stropha.meta.decorator import (
    _attach_trace_id,
    _extract_results,
    _extract_returned_tokens,
    _record_safely,
    _resolve_app,
    _tool_params,
    traced,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx_with(app: Any) -> Any:
    """Build a ctx-shaped MagicMock whose lifespan_context returns ``app``."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


class _FakeRecorder:
    """Minimal TraceRecorder stand-in — captures every record() call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: Exception | None = None

    def record(self, event: dict[str, Any]) -> bool:
        if self.fail_next is not None:
            err = self.fail_next
            self.fail_next = None
            raise err
        self.calls.append(event)
        return True


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


class TestExtractResults:
    """``_extract_results`` must accept every shape MCP tools return."""

    def test_returns_empty_for_none(self) -> None:
        assert _extract_results(None) == []

    def test_pydantic_with_results(self) -> None:
        from stropha.models import SearchHit

        hit = SearchHit(
            rank=1, score=0.9, rel_path="a.py", language="python",
            kind="function", start_line=1, end_line=5, snippet="def f(): ...",
            chunk_id="c1",
        )

        class Resp:
            def __init__(self) -> None:
                self.results = [hit]

        out = _extract_results(Resp())
        assert len(out) == 1
        assert out[0]["chunk_id"] == "c1"

    def test_dict_with_results_key(self) -> None:
        value = {"results": [{"chunk_id": "x"}, {"chunk_id": "y"}]}
        out = _extract_results(value)
        assert [r["chunk_id"] for r in out] == ["x", "y"]

    def test_plain_list_of_dicts(self) -> None:
        out = _extract_results([{"a": 1}, {"a": 2}])
        assert out == [{"a": 1}, {"a": 2}]

    def test_object_without_results_returns_empty(self) -> None:
        class Useless:
            pass

        assert _extract_results(Useless()) == []

    def test_results_with_objects_using_vars(self) -> None:
        class Item:
            def __init__(self) -> None:
                self.foo = 1
                self._private = "hidden"

        out = _extract_results([Item()])
        assert out == [{"foo": 1}]
        assert "_private" not in out[0]

    def test_skips_non_varsable_primitives(self) -> None:
        """list of strings / ints — vars() fails, they should be dropped silently."""
        out = _extract_results(["abc", 42])
        assert out == []

    def test_pydantic_model_dump_failure_skips_item(self) -> None:
        bad = MagicMock()
        bad.model_dump.side_effect = RuntimeError("boom")
        good = MagicMock()
        good.model_dump.return_value = {"ok": True}
        out = _extract_results([bad, good])
        assert out == [{"ok": True}]


class TestExtractReturnedTokens:
    def test_explicit_token_field(self) -> None:
        results = [{"returned_tokens": 100}, {"returned_tokens": 50}]
        assert _extract_returned_tokens(results) == 150

    def test_falls_back_to_snippet_length(self) -> None:
        results = [{"snippet": "a" * 40}, {"snippet": "b" * 80}]
        assert _extract_returned_tokens(results) == 30  # (40+80)//4

    def test_handles_non_int_returned_tokens(self) -> None:
        """A malformed ``returned_tokens`` value must not crash the trace."""
        results = [{"returned_tokens": "not-a-number", "snippet": "abcd"}]
        # Falls through to the snippet-length estimate; assert it didn't raise.
        assert _extract_returned_tokens(results) == 1

    def test_non_string_snippet_is_ignored(self) -> None:
        assert _extract_returned_tokens([{"snippet": None}]) == 0

    def test_empty_list(self) -> None:
        assert _extract_returned_tokens([]) == 0


class TestRecordSafely:
    def test_no_recorder_is_noop(self) -> None:
        # Must not raise even though there's nowhere to send the event.
        _record_safely(None, "x", "q", {}, [], 1.0, 0, "sess")

    def test_records_full_event(self) -> None:
        rec = _FakeRecorder()
        _record_safely(
            rec,
            "search_code",
            "q1",
            {"top_k": 5},
            [{"chunk_id": "sha256:abc", "score": 0.9}],
            2.5,
            12,
            "s",
        )
        assert len(rec.calls) == 1
        e = rec.calls[0]
        assert e["tool"] == "search_code"
        assert e["query"] == _fingerprint("q1")
        assert e["params"] == {"top_k": 5}
        assert e["latency_ms"] == 2.5
        assert e["returned_tokens"] == 12
        assert e["session_id"] == "s"
        assert e["results"] == [{"chunk_id": "sha256:abc", "score": 0.9}]
        assert e["result_count"] == 1
        assert e["status"] == "ok"

    def test_omits_session_id_when_none(self) -> None:
        rec = _FakeRecorder()
        _record_safely(rec, "x", None, {}, [], 0.0, 0, None)
        assert "session_id" not in rec.calls[0]

    def test_swallows_recorder_exception(self) -> None:
        rec = _FakeRecorder()
        rec.fail_next = RuntimeError("simulated")
        # Must not raise.
        _record_safely(rec, "x", "q", {}, [], 0.0, 0, None)

    def test_includes_trace_id_when_provided(self) -> None:
        rec = _FakeRecorder()
        _record_safely(rec, "x", "q", {}, [], 0.0, 0, "s", trace_id="abc123")
        assert rec.calls[0]["trace_id"] == "abc123"

    def test_omits_trace_id_when_none(self) -> None:
        rec = _FakeRecorder()
        _record_safely(rec, "x", "q", {}, [], 0.0, 0, "s")
        assert "trace_id" not in rec.calls[0]


class TestAttachTraceId:
    """``_attach_trace_id`` surfaces the id on responses that declare it."""

    def test_sets_on_model_with_field(self) -> None:
        from stropha.server_models import SearchResponse

        r = SearchResponse(results=[], total_candidates=0, query="q")
        assert _attach_trace_id(r, "tid") is True
        assert r.trace_id == "tid"

    def test_ignores_model_without_field(self) -> None:
        from stropha.server_models import ResetDedupResponse

        r = ResetDedupResponse(cleared=0)
        assert _attach_trace_id(r, "tid") is False
        assert not hasattr(r, "trace_id")

    def test_sets_on_dict_with_key(self) -> None:
        d: dict[str, Any] = {"trace_id": None, "ok": True}
        assert _attach_trace_id(d, "tid") is True
        assert d["trace_id"] == "tid"

    def test_ignores_none(self) -> None:
        assert _attach_trace_id(None, "tid") is False


class TestResolveApp:
    def test_returns_lifespan_context(self) -> None:
        sentinel = object()
        ctx = _ctx_with(sentinel)
        assert _resolve_app(ctx) is sentinel

    def test_returns_none_when_shape_missing(self) -> None:
        class Bare:
            pass

        assert _resolve_app(Bare()) is None


class TestToolParams:
    def test_drops_ctx_and_query(self) -> None:
        params = _tool_params((), {"ctx": object(), "query": "ignored", "top_k": 5})
        assert "ctx" not in params
        assert "query" not in params
        assert params == {"top_k": 5}

    def test_preserves_positional_as_argN(self) -> None:
        params = _tool_params(("first", 42), {"top_k": 5})
        assert params == {"arg0": "first", "arg1": 42, "top_k": 5}


# ---------------------------------------------------------------------------
# @traced decorator — end-to-end semantics
# ---------------------------------------------------------------------------


class TestTracedDecoratorHappyPath:
    def test_preserves_return_value(self) -> None:
        @traced("noop")
        def f(*, ctx: Any) -> dict:
            return {"value": 42}

        rec = _FakeRecorder()
        app = MagicMock(trace_recorder=rec, session_id="s1")
        result = f(ctx=_ctx_with(app))
        assert result == {"value": 42}

    def test_records_event_for_each_call(self) -> None:
        @traced("get_symbol")
        def f(symbol: str, *, ctx: Any) -> list[dict]:
            return [{"symbol": symbol}]

        rec = _FakeRecorder()
        app = MagicMock(trace_recorder=rec, session_id="sess-42")
        f("UserService", ctx=_ctx_with(app))
        f("Order", ctx=_ctx_with(app))
        assert len(rec.calls) == 2
        assert rec.calls[0]["tool"] == "get_symbol"
        assert rec.calls[0]["params"] == {}
        assert rec.calls[0]["session_id"] == "sess-42"
        assert rec.calls[0]["results"] == [{"symbol": "UserService"}]
        assert rec.calls[0]["result_count"] == 1

    def test_promotes_query_to_top_level(self) -> None:
        @traced("search_code")
        def f(query: str, top_k: int = 10, *, ctx: Any) -> dict:
            return {"results": []}

        rec = _FakeRecorder()
        app = MagicMock(trace_recorder=rec, session_id="s")
        f(query="hello world", top_k=20, ctx=_ctx_with(app))
        e = rec.calls[0]
        assert e["query"] == _fingerprint("hello world")
        assert "query" not in e["params"]
        assert e["params"]["top_k"] == 20

    def test_sensitive_arguments_and_results_are_not_recorded(self) -> None:
        canary = "STROPHA_TRACE_SECRET_CANARY_9f72c1"

        @traced("search_code")
        def f(
            query: str,
            *,
            ctx: Any,
            top_k: int,
            card_level: str,
            memory_value: str,
            scratchpad: str,
            episode_summary: str,
            feedback_detail: str,
            runtime_evidence: str,
            api_key: str,
            path_prefix: str,
        ) -> dict:
            return {
                "results": [
                    {
                        "chunk_id": "sha256:abc123",
                        "rank": 1,
                        "score": 0.9,
                        "streams": {"dense": 0.8, canary: 0.2},
                        "snippet": canary,
                        "rel_path": f"/Users/private/{canary}.py",
                        "symbol": "Search.run",
                        "evidence": canary,
                    }
                ]
            }

        rec = _FakeRecorder()
        app = MagicMock(trace_recorder=rec, session_id="s")
        f(
            query=canary,
            top_k=7,
            card_level="line",
            memory_value=canary,
            scratchpad=canary,
            episode_summary=canary,
            feedback_detail=canary,
            runtime_evidence=canary,
            api_key=canary,
            path_prefix=f"/Users/private/{canary}",
            ctx=_ctx_with(app),
        )

        event = rec.calls[0]
        assert canary not in json.dumps(event)
        assert event["query"] == _fingerprint(canary)
        assert event["params"] == {"top_k": 7, "card_level": "line"}
        assert event["result_count"] == 1
        assert event["results"] == [
            {
                "chunk_id": "sha256:abc123",
                "rank": 1,
                "score": 0.9,
                "symbol": "Search.run",
                "streams": {"dense": 0.8},
            }
        ]

    def test_latency_is_measured(self) -> None:
        import time as _time

        @traced("slow")
        def f(*, ctx: Any) -> dict:
            _time.sleep(0.01)
            return {"results": []}

        rec = _FakeRecorder()
        app = MagicMock(trace_recorder=rec, session_id="s")
        f(ctx=_ctx_with(app))
        latency = rec.calls[0]["latency_ms"]
        # ~10ms call; allow generous slop on CI.
        assert latency >= 8.0
        assert latency < 500.0


class TestTracedTraceIdSurfacing:
    """The decorator stamps a trace_id on the response AND records the same
    id, so report_bad_result can resolve the offending call (WIN-160)."""

    def test_stamps_trace_id_on_response_and_matches_record(self) -> None:
        from stropha.server_models import SearchResponse

        @traced("search_code")
        def f(query: str, *, ctx: Any) -> SearchResponse:
            return SearchResponse(results=[], total_candidates=0, query=query)

        rec = _FakeRecorder()
        recent_trace_ids: deque[str] = deque(maxlen=256)
        app = MagicMock(
            trace_recorder=rec,
            session_id="s",
            recent_trace_ids=recent_trace_ids,
        )
        resp = f(query="q", ctx=_ctx_with(app))
        assert resp.trace_id is not None
        # The id on the response equals the id written to the trace stream.
        assert rec.calls[0]["trace_id"] == resp.trace_id
        assert list(recent_trace_ids) == [resp.trace_id]

    def test_response_without_field_untouched_but_still_traced(self) -> None:
        @traced("noop")
        def f(*, ctx: Any) -> dict:
            return {"value": 1}

        rec = _FakeRecorder()
        recent_trace_ids: deque[str] = deque(maxlen=256)
        app = MagicMock(
            trace_recorder=rec,
            session_id="s",
            recent_trace_ids=recent_trace_ids,
        )
        out = f(ctx=_ctx_with(app))
        assert out == {"value": 1}  # plain dict not mutated
        assert "trace_id" in rec.calls[0]  # but server-side trace still has one
        assert list(recent_trace_ids) == []


class TestTracedDecoratorDegradation:
    """The decorator must never break the wrapped tool, no matter what's
    missing or broken in the trace path."""

    def test_no_ctx_argument(self) -> None:
        @traced("noctx")
        def f() -> dict:
            return {"ok": True}

        # ctx defaults to None inside the decorator; record_safely accepts that.
        assert f() == {"ok": True}

    def test_ctx_without_app(self) -> None:
        @traced("noapp")
        def f(*, ctx: Any) -> dict:
            return {"ok": True}

        # A bare ctx with no request_context attribute.
        class Bare:
            pass

        assert f(ctx=Bare()) == {"ok": True}

    def test_app_without_recorder(self) -> None:
        @traced("norec")
        def f(*, ctx: Any) -> dict:
            return {"ok": True}

        app = MagicMock(trace_recorder=None, session_id="s")
        assert f(ctx=_ctx_with(app)) == {"ok": True}

    def test_app_with_failing_recorder(self) -> None:
        @traced("badrec")
        def f(*, ctx: Any) -> dict:
            return {"ok": True}

        rec = _FakeRecorder()
        rec.fail_next = RuntimeError("simulated")
        app = MagicMock(trace_recorder=rec, session_id="s")
        # Result still reaches the caller intact.
        assert f(ctx=_ctx_with(app)) == {"ok": True}

    def test_tool_exception_is_traced_and_reraised(self) -> None:
        @traced("raiser")
        def f(*, ctx: Any) -> dict:
            raise ValueError("simulated tool failure")

        rec = _FakeRecorder()
        recent_trace_ids: deque[str] = deque(maxlen=256)
        app = MagicMock(
            trace_recorder=rec,
            session_id="s",
            recent_trace_ids=recent_trace_ids,
        )
        with pytest.raises(ValueError, match="simulated tool failure"):
            f(ctx=_ctx_with(app))
        # The trace event was still emitted (with empty results).
        assert len(rec.calls) == 1
        assert rec.calls[0]["results"] == []
        assert rec.calls[0]["latency_ms"] >= 0.0
        assert rec.calls[0]["status"] == "error"
        assert rec.calls[0]["error"] == {"type": "ValueError"}
        assert list(recent_trace_ids) == []

    def test_exception_when_no_app_does_not_crash_trace_path(self) -> None:
        @traced("raise_no_ctx")
        def f() -> None:
            raise RuntimeError("boom")

        # No ctx, no app, no recorder — the trace path must still re-raise
        # the original exception unchanged.
        with pytest.raises(RuntimeError, match="boom"):
            f()


# ---------------------------------------------------------------------------
# AppContext + _ctx integration
# ---------------------------------------------------------------------------


class TestAppContextExtensions:
    def test_appcontext_has_session_id_and_trace_recorder(self) -> None:
        from stropha.server import AppContext

        fields = {f.name for f in dataclasses.fields(AppContext)}
        assert "session_id" in fields
        assert "trace_recorder" in fields
        assert "recent_trace_ids" in fields

    def test_appcontext_session_id_has_default(self) -> None:
        """The dataclass must instantiate with only the legacy required args
        so existing tests that build it manually keep working."""
        from stropha.server import AppContext

        app = AppContext(
            config=MagicMock(),
            storage=MagicMock(),
            embedder=MagicMock(),
            search_engine=MagicMock(),
            built=MagicMock(),
        )
        assert app.session_id == ""
        assert app.trace_recorder is None
        assert app.recent_trace_ids.maxlen == 256


class TestCtxBindsSessionId:
    def test_ctx_binds_session_id_to_structlog(self) -> None:
        """``_ctx`` must call ``structlog.contextvars.bind_contextvars`` so
        every subsequent log carries the request's session id."""
        import structlog

        from stropha.server import _ctx

        app = MagicMock(session_id="sess-42")
        ctx = _ctx_with(app)

        structlog.contextvars.clear_contextvars()
        returned = _ctx(ctx)
        # Logs emitted during this request now carry session_id automatically.
        bound = structlog.contextvars.get_contextvars()
        assert bound.get("session_id") == "sess-42"
        assert returned is app

    def test_ctx_skips_bind_when_session_id_empty(self) -> None:
        """Empty session_id means we never bound it — keeps tests / non-MCP
        callers free of leftover context."""
        import structlog

        from stropha.server import _ctx

        app = MagicMock(session_id="")
        structlog.contextvars.clear_contextvars()
        _ctx(_ctx_with(app))
        bound = structlog.contextvars.get_contextvars()
        assert "session_id" not in bound
