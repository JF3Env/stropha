"""Tests for the HTTP transport recovery logic (WIN-427).

Focus: a request carrying a stale ``mcp-session-id`` (a client reusing a
session id from before a daemon restart) must ALWAYS get a complete HTTP
response, never a hang. We drive ``_handle_stateful_request`` with a
monkeypatched ``_serve_with_new_session`` so we can simulate the buggy
"transport tears down without finishing the response" case without standing
up a real MCP server.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from stropha.server_http import RestartTolerantSessionManager


def _bare_manager() -> RestartTolerantSessionManager:
    """Construct the manager without the heavy parent __init__ — we only
    exercise the routing + completion logic, which needs ``_server_instances``."""
    mgr = object.__new__(RestartTolerantSessionManager)
    mgr._server_instances = {}
    mgr.session_idle_timeout = None
    return mgr


async def _drive(mgr, *, session_id: str | None, serve_impl) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    headers = [(b"mcp-session-id", session_id.encode())] if session_id else []
    scope = {"type": "http", "headers": headers}
    mgr._serve_with_new_session = serve_impl  # type: ignore[method-assign]
    await mgr._handle_stateful_request(scope, receive, send)
    return sent


def _starts(sent: list[dict]) -> list[dict]:
    return [m for m in sent if m["type"] == "http.response.start"]


def _bodies(sent: list[dict]) -> list[dict]:
    return [m for m in sent if m["type"] == "http.response.body"]


def test_stale_recovery_passes_through_when_serve_completes() -> None:
    """When the transparent recovery completes the response, we do NOT inject
    a fallback — the client gets the real result."""
    captured: dict[str, Any] = {}

    async def serve_ok(scope, receive, send, *, server_stateless, session_id) -> None:
        captured["stateless"] = server_stateless
        captured["sid"] = session_id
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    sent = asyncio.run(_drive(_bare_manager(), session_id="stale-1", serve_impl=serve_ok))

    assert captured == {"stateless": True, "sid": "stale-1"}
    assert len(_starts(sent)) == 1 and _starts(sent)[0]["status"] == 200
    assert _bodies(sent)[-1]["more_body"] is False


def test_stale_recovery_sends_fallback_when_incomplete() -> None:
    """The bug: recovery returns without finishing the response. The fallback
    must close it with a 404 + JSON-RPC error so the client never hangs."""

    async def serve_silent(scope, receive, send, *, server_stateless, session_id) -> None:
        return  # never touches `send` — simulates the dropped stream

    sent = asyncio.run(_drive(_bare_manager(), session_id="stale-2", serve_impl=serve_silent))

    assert len(_starts(sent)) == 1 and _starts(sent)[0]["status"] == 404
    last = _bodies(sent)[-1]
    assert last["more_body"] is False
    payload = json.loads(last["body"])
    assert payload["error"]["code"] == -32600
    assert "re-initialize" in payload["error"]["message"].lower()
    assert "stale-2" in payload["error"]["message"]


def test_stale_recovery_fallback_on_exception() -> None:
    """If recovery raises mid-flight, we still complete the response."""

    async def serve_boom(scope, receive, send, *, server_stateless, session_id) -> None:
        raise RuntimeError("transport torn down")

    sent = asyncio.run(_drive(_bare_manager(), session_id="stale-3", serve_impl=serve_boom))

    assert len(_starts(sent)) == 1 and _starts(sent)[0]["status"] == 404
    assert sent[-1]["type"] == "http.response.body" and sent[-1]["more_body"] is False


def test_partial_response_is_closed_not_restarted() -> None:
    """If recovery already sent http.response.start, the fallback must NOT
    send a second start (that is an ASGI error) — it just closes the stream."""

    async def serve_partial(scope, receive, send, *, server_stateless, session_id) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"data", "more_body": True})

    sent = asyncio.run(_drive(_bare_manager(), session_id="stale-4", serve_impl=serve_partial))

    assert len(_starts(sent)) == 1 and _starts(sent)[0]["status"] == 200  # no double-start
    assert _bodies(sent)[-1]["more_body"] is False  # stream closed


def test_known_session_uses_existing_transport_no_recovery() -> None:
    """A live (known) session id must reuse its transport, not the recovery path."""
    mgr = _bare_manager()
    calls: list[str] = []

    class FakeTransport:
        idle_scope = None

        async def handle_request(self, scope, receive, send) -> None:
            calls.append("handled")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    mgr._server_instances = {"live-1": FakeTransport()}

    async def serve_must_not_run(*a, **k) -> None:
        raise AssertionError("recovery must not run for a known session")

    sent = asyncio.run(_drive(mgr, session_id="live-1", serve_impl=serve_must_not_run))
    assert calls == ["handled"]
    assert _starts(sent)[0]["status"] == 200
