"""``@traced`` decorator — wires MCP tool calls into the Meta-Harness trace stream.

Stacks under ``@mcp.tool`` so the MCP framework sees the wrapped function::

    @mcp.tool(...)
    @traced("search_code")
    def search_code(query: str, ..., *, ctx: Context) -> SearchResponse: ...

On every call the decorator:

1. Captures the start clock.
2. Calls the underlying tool.
3. Builds a trace record from the input args and the return value.
4. Hands the record to the active :class:`TraceRecorder` on the AppContext.

Failures inside the tracing path NEVER propagate to the caller. The decorator
is observability infrastructure, not business logic. A trace failure logs and
moves on; the tool's return value is what the MCP client receives.

Why a thin custom decorator instead of OpenTelemetry: the recorder lives in
the same process and already implements <2 ms p95 off-thread writes (WIN-159).
OTel would add a dependency, a collector, and a fan-out we don't need. When
the proposer learns to consume traces it reads the same JSONL the decorator
writes.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections.abc import Callable
from typing import Any

from ..logging import get_logger
from .trace import _sanitize_trace_event

log = get_logger(__name__)


def _attach_trace_id(result: Any, trace_id: str) -> bool:
    """Best-effort: stamp ``trace_id`` onto a tool's response so the agent can
    cite it in ``report_bad_result``.

    Only responses that declare a ``trace_id`` field (pydantic) or already
    carry the key (dict) receive it; everything else is left untouched. Returns
    whether this call attached the supplied ID. Never raises because surfacing
    the ID is a convenience, not business logic.
    """
    try:
        fields = getattr(type(result), "model_fields", None)
        if fields is not None and "trace_id" in fields:
            # Only fill an empty slot. Tools like report_bad_result echo a
            # *different* trace_id (the reported call) and must not be clobbered.
            if getattr(result, "trace_id", None) is None:
                result.trace_id = trace_id
                return True
        elif isinstance(result, dict) and result.get("trace_id") is None and "trace_id" in result:
            result["trace_id"] = trace_id
            return True
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("traced.attach_trace_id_failed", error=str(exc))
    return False


def _extract_results(value: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of a ``results`` list from a tool's return value.

    Tools return heterogeneous shapes (pydantic models, dicts, dataclasses,
    plain lists). We only need a list of result dicts for the trace —
    missing or malformed shapes degrade to an empty list, never raise.
    """
    if value is None:
        return []

    # Pydantic v2 models / dataclasses with .results attribute.
    inner = getattr(value, "results", None)
    if inner is None and isinstance(value, dict):
        inner = value.get("results")
    if inner is None and isinstance(value, list):
        inner = value
    if inner is None:
        return []

    out: list[dict[str, Any]] = []
    for item in inner:
        if hasattr(item, "model_dump"):
            try:
                out.append(item.model_dump())
            except Exception:  # pydantic edge cases — drop the item
                continue
        elif isinstance(item, dict):
            out.append(item)
        else:
            # Generic objects: snapshot their public attribute surface.
            try:
                out.append({k: v for k, v in vars(item).items() if not k.startswith("_")})
            except TypeError:
                # Non-vars-able (e.g. primitive in a list). Skip.
                continue
    return out


def _extract_returned_tokens(results: list[dict[str, Any]]) -> int:
    """Sum of token-shaped fields across results, or 0 if absent.

    The trace schema (WIN-159) defines ``returned_tokens`` so the
    Meta-Harness fitness can score token-efficiency without re-tokenising.
    We use ``len(snippet)//4`` as a fast estimate when no explicit token
    count is present, matching ``retrieval/savings.py``.
    """
    total = 0
    for r in results:
        if "returned_tokens" in r:
            try:
                total += int(r["returned_tokens"])
                continue
            except (TypeError, ValueError):
                pass
        snippet = r.get("snippet") or ""
        if isinstance(snippet, str):
            total += len(snippet) // 4
    return total


def _record_safely(
    recorder: Any,
    tool: str,
    query: str | None,
    params: dict[str, Any],
    results: list[dict[str, Any]],
    latency_ms: float,
    returned_tokens: int,
    session_id: str | None,
    trace_id: str | None = None,
    status: str = "ok",
    error_type: str | None = None,
) -> None:
    """Hand the event to the recorder; swallow any failure."""
    if recorder is None:
        return
    try:
        event: dict[str, Any] = {
            "tool": tool,
            "query": query,
            "params": params,
            "results": results,
            "latency_ms": latency_ms,
            "returned_tokens": returned_tokens,
            "status": status,
        }
        if error_type is not None:
            event["error"] = {"type": error_type}
        if session_id is not None:
            event["session_id"] = session_id
        if trace_id is not None:
            event["trace_id"] = trace_id
        recorder.record(_sanitize_trace_event(event))
    except Exception as exc:
        # Defensive: tracing is observability, never break a tool call.
        log.debug(
            "traced.record_failed",
            tool=tool,
            error_type=type(exc).__qualname__,
        )


def _resolve_app(ctx: Any) -> Any:
    """Pluck the AppContext-like object out of an MCP ``Context``.

    The shape is ``ctx.request_context.lifespan_context``. Wrapped in a
    helper so tests can stub a plain object for ``ctx``.
    """
    try:
        return ctx.request_context.lifespan_context
    except AttributeError:
        return None


def traced(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that records every call to the wrapped tool.

    The wrapped function MUST accept a keyword-only ``ctx`` argument that
    points to an MCP ``Context`` (or any object that exposes
    ``request_context.lifespan_context``). The AppContext is expected to
    carry at least these two attributes:

    - ``trace_recorder`` — a :class:`stropha.meta.TraceRecorder` or ``None``.
    - ``session_id`` — the per-process session identifier (``str``).

    Either may be missing; the decorator degrades to a no-op record then.

    The decorator preserves the tool's signature so MCP / FastMCP can still
    introspect it (``functools.wraps``).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            ctx = kwargs.get("ctx")
            # Mint the trace_id up-front so the SAME id is both stamped on the
            # response (so the agent can cite it) and written to the JSONL
            # trace record (so report_bad_result resolves to this call).
            trace_id = uuid.uuid4().hex
            app = _resolve_app(ctx) if ctx is not None else None
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                # Log centrally so every MCP tool failure is visible in
                # the stropha server logs — not just in the JSONL trace
                # stream.  This is the single point through which ALL
                # unhandled tool exceptions flow before FastMCP wraps
                # them as ToolError → CallToolResult(isError=True).
                latency_ms = (time.perf_counter() - start) * 1000
                log.error(
                    "mcp.tool.exception",
                    tool=tool_name,
                    error_type=type(exc).__qualname__,
                    latency_ms=round(latency_ms, 1),
                )
                _record_safely(
                    recorder=getattr(app, "trace_recorder", None) if app else None,
                    tool=tool_name,
                    query=kwargs.get("query"),
                    params=_tool_params(args, kwargs),
                    results=[],
                    latency_ms=latency_ms,
                    returned_tokens=0,
                    session_id=getattr(app, "session_id", None) if app else None,
                    trace_id=trace_id,
                    status="error",
                    error_type=type(exc).__qualname__,
                )
                raise

            latency_ms = (time.perf_counter() - start) * 1000
            results = _extract_results(result)
            _record_safely(
                recorder=getattr(app, "trace_recorder", None) if app else None,
                tool=tool_name,
                query=kwargs.get("query"),
                params=_tool_params(args, kwargs),
                results=results,
                latency_ms=latency_ms,
                returned_tokens=_extract_returned_tokens(results),
                session_id=getattr(app, "session_id", None) if app else None,
                trace_id=trace_id,
            )
            trace_id_exposed = _attach_trace_id(result, trace_id)
            recent_trace_ids = getattr(app, "recent_trace_ids", None) if app else None
            if trace_id_exposed and recent_trace_ids is not None:
                recent_trace_ids.append(trace_id)
            return result

        return wrapper

    return decorator


def _tool_params(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the call arguments for the trace, minus ``ctx`` + positional args.

    Positional args are rare for MCP tools (FastMCP usually serialises by name
    from JSON-RPC) but we include them as a numbered map for completeness.
    """
    params: dict[str, Any] = {}
    for i, value in enumerate(args):
        params[f"arg{i}"] = value
    for key, value in kwargs.items():
        if key in ("ctx", "query"):
            # ctx is infrastructure; query is recorded as a first-class field.
            continue
        params[key] = value
    return params
