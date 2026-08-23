from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from stropha.server_tool_manifest import tool_names_for_profile


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


def _call_tool(
    client: TestClient,
    session_id: str,
    name: str,
    request_id: str,
    arguments: dict | None = None,
):
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
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
    )


def _read_sse_json(response) -> dict:
    payload = response.text.split("data: ", 1)[1].split("\r\n\r\n", 1)[0]
    return json.loads(payload)


def _delete_session(client: TestClient, session_id: str):
    return client.delete(
        "/mcp",
        headers={
            "host": "127.0.0.1",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": "2025-11-25",
            "mcp-session-id": session_id,
        },
    )


def test_streamable_http_recovers_from_stale_session_id_after_restart(tmp_path) -> None:
    from stropha.server import mcp

    fake_storage = MagicMock()
    fake_storage.close = MagicMock()
    fake_built = MagicMock(storage=fake_storage, embedder=MagicMock())
    config = SimpleNamespace(
        log_level="INFO",
        target_repo=tmp_path,
        mcp_watch=False,
        mcp_watch_interval=1.0,
        mcp_watch_debounce=1.0,
        resolve_index_path=lambda: tmp_path / "index.db",
    )

    with (
        patch("stropha.server.get_config", return_value=config),
        patch("stropha.server.load_pipeline_config", return_value={}),
        patch("stropha.server.build_stages", return_value=fake_built) as build_stages,
        patch("stropha.server._resolve_target_repo_id", return_value=None),
    ):
        original_security = mcp.settings.transport_security
        original_profile = mcp.tool_profile
        mcp.set_tool_profile("full")
        mcp.settings.transport_security = None
        try:
            app1 = mcp.streamable_http_app()
            with TestClient(app1) as client1:
                init = _initialize(client1)
                assert init.status_code == 200
                stale_session_id = init.headers["mcp-session-id"]

            mcp._session_manager = None
            mcp.set_tool_profile("core")

            app2 = mcp.streamable_http_app()
            with TestClient(app2) as client2:
                response1 = _list_tools(client2, stale_session_id)
                response2 = _list_tools(client2, stale_session_id)
                hidden = _call_tool(client2, stale_session_id, "remember", "hidden-1")
                capabilities = _call_tool(
                    client2, stale_session_id, "capabilities", "capabilities-1"
                )
                callers = _call_tool(
                    client2,
                    stale_session_id,
                    "find_callers",
                    "callers-1",
                    {"symbol": "Missing.symbol"},
                )

            assert response1.status_code == 200
            assert response1.headers["mcp-session-id"] == stale_session_id
            body1 = _read_sse_json(response1)
            assert body1["jsonrpc"] == "2.0"
            assert body1["id"] == "tools-1"
            assert [tool["name"] for tool in body1["result"]["tools"]] == list(
                tool_names_for_profile("core")
            )
            assert len(response1.content) <= 40 * 1024

            assert response2.status_code == 200
            assert response2.headers["mcp-session-id"] == stale_session_id
            body2 = _read_sse_json(response2)
            assert body2["jsonrpc"] == "2.0"
            assert body2["id"] == "tools-1"
            assert [tool["name"] for tool in body2["result"]["tools"]] == list(
                tool_names_for_profile("core")
            )

            hidden_body = _read_sse_json(hidden)
            assert hidden_body["id"] == "hidden-1"
            assert hidden_body["result"]["isError"] is True
            assert (
                "not available in MCP profile 'core'" in hidden_body["result"]["content"][0]["text"]
            )

            capabilities_body = _read_sse_json(capabilities)
            assert capabilities_body["id"] == "capabilities-1"
            structured = capabilities_body["result"]["structuredContent"]
            assert structured["active_profile"] == "core"
            assert structured["tools"] == list(tool_names_for_profile("core"))
            assert structured["profile_is_static"] is True

            callers_body = _read_sse_json(callers)
            assert callers_body["id"] == "callers-1"
            assert callers_body["result"].get("isError", False) is False
            callers_structured = callers_body["result"]["structuredContent"]
            assert callers_structured["symbol"] == "Missing.symbol"
            assert callers_structured["graph_loaded"] is False
            assert callers_structured["message"] == (
                "Current repository could not be resolved; graph query was not run."
            )

            # One build for each simulated daemon lifetime. The two stale-session
            # requests in app2 reuse its warm runtime instead of rebuilding.
            assert build_stages.call_count == 2
            assert fake_storage.close.call_count == 2
        finally:
            mcp._session_manager = None
            mcp.set_tool_profile(original_profile)
            mcp.settings.transport_security = original_security


def test_agent_observations_are_session_local_and_cleared_on_reconnect(tmp_path) -> None:
    from stropha.server import mcp

    repository = "local:protocol-observations"
    fake_storage = MagicMock()
    fake_storage.list_sessions.return_value = []
    fake_storage.list_repos.return_value = [
        SimpleNamespace(normalized_key=repository, local_path=str(tmp_path))
    ]
    fake_built = MagicMock(
        storage=fake_storage,
        embedder=MagicMock(),
        retrieval=MagicMock(),
    )
    config = SimpleNamespace(
        log_level="INFO",
        target_repo=tmp_path,
        mcp_watch=False,
        mcp_watch_interval=1.0,
        mcp_watch_debounce=1.0,
        resolve_index_path=lambda: tmp_path / "index.db",
    )
    traces_dir = tmp_path / "traces"
    canary = "SESSION-ONLY-FAILURE-CANARY"
    observed_at = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    with (
        patch("stropha.server.get_config", return_value=config),
        patch("stropha.server.load_pipeline_config", return_value={}),
        patch("stropha.server.build_stages", return_value=fake_built),
        patch.dict(os.environ, {"STROPHA_TRACES_DIR": str(traces_dir)}),
    ):
        original_security = mcp.settings.transport_security
        original_profile = mcp.tool_profile
        mcp._session_manager = None
        mcp.set_tool_profile("standard")
        mcp.settings.transport_security = None
        try:
            app = mcp.streamable_http_app()
            with TestClient(app) as client:
                init_first = _initialize(client)
                init_second = _initialize(client)
                first_session = init_first.headers["mcp-session-id"]
                second_session = init_second.headers["mcp-session-id"]

                state = _call_tool(
                    client,
                    first_session,
                    "observe_agent_state",
                    "observe-1",
                    {
                        "snapshot": {
                            "revision": 1,
                            "repository": repository,
                            "active_file": "src/current.py",
                            "recent_files": [],
                            "selection": None,
                            "changed_paths": ["src/current.py"],
                            "runtime_artifact_ids": ["pytest-failure"],
                            "observed_at": observed_at,
                        }
                    },
                )
                artifact = _call_tool(
                    client,
                    first_session,
                    "record_runtime_artifact",
                    "artifact-1",
                    {
                        "artifact": {
                            "artifact_id": "pytest-failure",
                            "kind": "test_failure",
                            "media_type": "text/plain",
                            "producer": "pytest",
                            "content": canary,
                            "source_locations": [
                                {
                                    "repository": repository,
                                    "path": "tests/test_current.py",
                                    "start_line": 5,
                                    "end_line": 7,
                                }
                            ],
                            "redaction_status": "not_required",
                            "expires_at": expires_at,
                        }
                    },
                )
                first_view = _call_tool(client, first_session, "observe_agent_state", "observe-2")
                second_view = _call_tool(client, second_session, "observe_agent_state", "observe-3")

                assert _read_sse_json(state)["result"].get("isError", False) is False
                assert _read_sse_json(artifact)["result"].get("isError", False) is False
                first_structured = _read_sse_json(first_view)["result"]["structuredContent"]
                assert first_structured["snapshot"]["active_file"] == "src/current.py"
                assert first_structured["runtime_artifacts"][0]["content"] == canary
                second_structured = _read_sse_json(second_view)["result"]["structuredContent"]
                assert second_structured == {"snapshot": None, "runtime_artifacts": []}

                assert _delete_session(client, first_session).status_code == 200
                reconnected = _initialize(client)
                reconnected_view = _call_tool(
                    client,
                    reconnected.headers["mcp-session-id"],
                    "observe_agent_state",
                    "observe-4",
                )
                assert _read_sse_json(reconnected_view)["result"]["structuredContent"] == {
                    "snapshot": None,
                    "runtime_artifacts": [],
                }

            trace_text = "".join(
                path.read_text(encoding="utf-8") for path in traces_dir.glob("*.jsonl")
            )
            assert canary not in trace_text
            assert canary not in repr(fake_storage.upsert_session.call_args_list)
        finally:
            mcp._session_manager = None
            mcp.set_tool_profile(original_profile)
            mcp.settings.transport_security = original_security
