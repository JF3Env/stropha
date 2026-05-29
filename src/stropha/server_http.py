"""HTTP transport helpers for ``stropha.server``."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import anyio
from anyio.abc import TaskStatus
from mcp.server.fastmcp.server import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.server.streamable_http_manager import (
    MCP_SESSION_ID_HEADER,
    StreamableHTTPSessionManager,
    logger,
)
from starlette.applications import Starlette


class RestartTolerantTransport(StreamableHTTPServerTransport):
    """Ignore mismatched stale session headers during restart recovery."""

    async def _validate_request_headers(self, request, send) -> bool:  # type: ignore[override]
        request_session_id = request.headers.get(MCP_SESSION_ID_HEADER)
        if request_session_id and request_session_id != self.mcp_session_id:
            return True
        return await super()._validate_request_headers(request, send)


class RestartTolerantSessionManager(StreamableHTTPSessionManager):
    """Recover from stale ``mcp-session-id`` headers after daemon restart."""

    async def _handle_stateful_request(self, scope, receive, send) -> None:  # type: ignore[override]
        request_mcp_session_id = self._header(scope, MCP_SESSION_ID_HEADER)

        if request_mcp_session_id is not None and request_mcp_session_id in self._server_instances:
            transport = self._server_instances[request_mcp_session_id]
            logger.debug("Session already exists, handling request directly")
            if transport.idle_scope is not None and self.session_idle_timeout is not None:
                transport.idle_scope.deadline = anyio.current_time() + self.session_idle_timeout
            await transport.handle_request(scope, receive, send)
            return

        if request_mcp_session_id is not None:
            logger.info("Recovering stale MCP session %s with a fresh transport", request_mcp_session_id)
            await self._serve_with_new_session(
                scope,
                receive,
                send,
                server_stateless=True,
                session_id=request_mcp_session_id,
            )
            return

        await self._serve_with_new_session(scope, receive, send, server_stateless=False, session_id=None)

    @staticmethod
    def _header(scope, name: str) -> str | None:
        needle = name.lower().encode("latin-1")
        for key, value in scope.get("headers", []):
            if key.lower() == needle:
                return value.decode("latin-1")
        return None

    async def _serve_with_new_session(
        self,
        scope,
        receive,
        send,
        *,
        server_stateless: bool,
        session_id: str | None,
    ) -> None:
        if self._task_group is None:
            raise RuntimeError("Task group is not initialized. Make sure to use run().")

        async with self._session_creation_lock:
            new_session_id = session_id or uuid4().hex
            http_transport = RestartTolerantTransport(
                mcp_session_id=new_session_id,
                is_json_response_enabled=self.json_response,
                event_store=self.event_store,
                security_settings=self.security_settings,
                retry_interval=self.retry_interval,
            )

            assert http_transport.mcp_session_id is not None
            self._server_instances[http_transport.mcp_session_id] = http_transport
            logger.info("Created new transport with session ID: %s", new_session_id)

            async def run_server(*, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
                async with http_transport.connect() as streams:
                    read_stream, write_stream = streams
                    task_status.started()
                    try:
                        idle_scope = anyio.CancelScope()
                        if self.session_idle_timeout is not None:
                            idle_scope.deadline = anyio.current_time() + self.session_idle_timeout
                            http_transport.idle_scope = idle_scope

                        with idle_scope:
                            await self.app.run(
                                read_stream,
                                write_stream,
                                self.app.create_initialization_options(),
                                stateless=server_stateless,
                            )

                        if idle_scope.cancelled_caught:
                            assert http_transport.mcp_session_id is not None
                            logger.info("Session %s idle timeout", http_transport.mcp_session_id)
                            self._server_instances.pop(http_transport.mcp_session_id, None)
                            await http_transport.terminate()
                    except Exception:
                        logger.exception("Session %s crashed", http_transport.mcp_session_id)
                    finally:
                        if (
                            http_transport.mcp_session_id
                            and http_transport.mcp_session_id in self._server_instances
                            and not http_transport.is_terminated
                        ):
                            logger.info(
                                "Cleaning up crashed session %s from active instances.",
                                http_transport.mcp_session_id,
                            )
                            del self._server_instances[http_transport.mcp_session_id]

            await self._task_group.start(run_server)
            await http_transport.handle_request(scope, receive, send)


class RestartTolerantFastMCP(FastMCP[Any]):
    """FastMCP variant that tolerates stale HTTP session ids after restart."""

    def streamable_http_app(self) -> Starlette:
        if self._session_manager is None:
            self._session_manager = RestartTolerantSessionManager(
                app=self._mcp_server,
                event_store=self._event_store,
                retry_interval=self._retry_interval,
                json_response=self.settings.json_response,
                stateless=self.settings.stateless_http,
                security_settings=self.settings.transport_security,
            )
        return super().streamable_http_app()
