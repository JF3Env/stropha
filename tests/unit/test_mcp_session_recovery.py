from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient


def _initialize(client: TestClient, headers: dict[str, str] | None = None):
    response = client.post(
        "/mcp",
        headers={
            "host": "127.0.0.1",
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **(headers or {}),
        },
        json={
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        },
    )
    return response


def _list_tools(client: TestClient, session_id: str):
    return client.post(
        "/mcp",
        headers={
            "host": "127.0.0.1",
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            "mcp-protocol-version": "2025-11-25",
            "mcp-session-id": session_id,
        },
        json={
            "jsonrpc": "2.0",
            "id": "tools-1",
            "method": "tools/list",
        },
    )


def _read_sse_json(response) -> dict:
    payload = response.text.split("data: ", 1)[1].split("\r\n\r\n", 1)[0]
    return json.loads(payload)


def test_streamable_http_recovers_from_stale_session_id_after_restart() -> None:
    from stropha.server import mcp

    fake_storage = MagicMock()
    fake_storage.close = MagicMock()
    fake_built = MagicMock(storage=fake_storage, embedder=MagicMock())

    with patch("stropha.server.load_pipeline_config", return_value={}), patch(
        "stropha.server.build_stages", return_value=fake_built
    ):
        original_security = mcp.settings.transport_security
        mcp.settings.transport_security = None
        try:
            app1 = mcp.streamable_http_app()
            with TestClient(app1) as client1:
                init = _initialize(client1)
                assert init.status_code == 200
                stale_session_id = init.headers["mcp-session-id"]

            mcp._session_manager = None

            app2 = mcp.streamable_http_app()
            with TestClient(app2) as client2:
                response1 = _list_tools(client2, stale_session_id)
                response2 = _list_tools(client2, stale_session_id)

            assert response1.status_code == 200
            assert response1.headers["mcp-session-id"] == stale_session_id
            body1 = _read_sse_json(response1)
            assert body1["jsonrpc"] == "2.0"
            assert body1["id"] == "tools-1"
            assert "result" in body1

            assert response2.status_code == 200
            assert response2.headers["mcp-session-id"] == stale_session_id
            body2 = _read_sse_json(response2)
            assert body2["jsonrpc"] == "2.0"
            assert body2["id"] == "tools-1"
            assert "result" in body2
        finally:
            mcp._session_manager = None
            mcp.settings.transport_security = original_security
