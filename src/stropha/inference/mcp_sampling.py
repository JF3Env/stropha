"""MCP sampling primitive (WIN-255, EPIC 1.12 #92).

MCP defines a **sampling** primitive that lets the *server*
request an LLM completion from the *client's* configured model.
For stropha, that means HyDE rewrites, contextual enrichment, and
query decomposition can run on the agent's own LLM — no extra
local model, no extra cloud charges, no separate API key.

The protocol shape:

1. Server builds a sampling request (messages + system prompt +
   model hints + max tokens).
2. Server submits it via the MCP session's ``create_message``
   call.
3. Client (Claude Desktop / Cursor / agent host) routes the
   request to its configured LLM, returns the response.
4. Server consumes the response text.

This module ships the primitive that builds the request payload
and runs the call defensively (the client may decline sampling —
permission-gated, may not be configured, may time out). Failures
degrade gracefully so callers can fall back to their existing
local-inference path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SamplingRequest:
    """Payload for ``ctx.session.create_message``.

    ``messages`` is a list of ``{"role": ..., "content": ...}``
    dicts (MCP-spec shape). ``model_preferences`` is the optional
    hint dict (``{"hints": [{"name": "claude-3-5-sonnet"}],
    "intelligence_priority": 0.8, ...}``). ``system_prompt`` is
    optional — when set, prepended as a system message by the
    client.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 512
    system_prompt: str | None = None
    model_preferences: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None


@dataclass(frozen=True)
class SamplingResponse:
    """Outcome of a sampling call.

    ``ok`` is the tri-state outcome: ``True`` (text returned),
    ``False`` (client returned an empty / refusal response),
    ``None`` (transport / permission error — caller should retry
    with a different strategy).
    """

    text: str = ""
    model: str = ""
    stop_reason: str = ""
    ok: bool | None = None
    error: str = ""


class SamplingSessionLike(Protocol):
    """Subset of the MCP session surface ``run_sampling`` needs."""

    async def create_message(self, **kwargs: Any) -> Any:
        ...


def build_sampling_request(
    prompt: str,
    *,
    max_tokens: int = 512,
    system_prompt: str | None = None,
    model_hints: list[str] | None = None,
    intelligence_priority: float | None = None,
    speed_priority: float | None = None,
    temperature: float | None = None,
) -> SamplingRequest:
    """Construct a :class:`SamplingRequest` for ``prompt``.

    Args:
        prompt: the user-role content to send.
        max_tokens: hard cap on response tokens.
        system_prompt: optional system-role preamble.
        model_hints: list of model name fragments (e.g.
            ``["claude-3-5", "gpt-4o"]``) the client can use to
            pick a compatible local model. The client is free to
            ignore the hints.
        intelligence_priority / speed_priority: floats in [0, 1].
            Tells the client whether to prefer a smarter or faster
            model. Defaults to ``None`` (no preference).
        temperature: optional sampling temperature.
    """
    preferences: dict[str, Any] = {}
    if model_hints:
        preferences["hints"] = [{"name": h} for h in model_hints if h]
    if intelligence_priority is not None:
        preferences["intelligencePriority"] = float(intelligence_priority)
    if speed_priority is not None:
        preferences["speedPriority"] = float(speed_priority)

    return SamplingRequest(
        messages=[{
            "role": "user",
            "content": {"type": "text", "text": str(prompt)},
        }],
        max_tokens=max(1, int(max_tokens)),
        system_prompt=system_prompt,
        model_preferences=preferences,
        temperature=temperature,
    )


async def run_sampling(
    session: SamplingSessionLike,
    request: SamplingRequest,
) -> SamplingResponse:
    """Submit ``request`` via the MCP session; never raise.

    Returns ``ok=None`` when the call itself failed (transport /
    permission), ``ok=False`` when the client returned an empty
    response, ``ok=True`` when text was produced.
    """
    kwargs: dict[str, Any] = {
        "messages": list(request.messages),
        "max_tokens": int(request.max_tokens),
    }
    if request.system_prompt:
        kwargs["system_prompt"] = request.system_prompt
    if request.model_preferences:
        kwargs["model_preferences"] = dict(request.model_preferences)
    if request.temperature is not None:
        kwargs["temperature"] = float(request.temperature)

    try:
        result = await session.create_message(**kwargs)
    except Exception as exc:
        return SamplingResponse(
            ok=None,
            error=f"create_message raised: {exc!r}",
        )

    text = _extract_text(result)
    model = str(getattr(result, "model", "") or "")
    stop_reason = str(getattr(result, "stop_reason", "") or "")
    if not text:
        return SamplingResponse(
            ok=False, model=model, stop_reason=stop_reason,
            error="empty response from client sampling",
        )
    return SamplingResponse(
        text=text, model=model, stop_reason=stop_reason, ok=True,
    )


def _extract_text(result: Any) -> str:
    """Pull the text body out of an MCP sampling result.

    Result shape per spec: ``{"content": {"type": "text", "text": "..."},
    "model": ..., "stopReason": ...}``. Tolerant of dict / object
    variants and of missing fields.
    """
    if result is None:
        return ""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if content is None:
        return ""
    # Content may be a single ``{"type": "text", "text": ...}`` or a
    # list of such items.
    if isinstance(content, list):
        parts = [_extract_text_from_block(b) for b in content]
        return "".join(p for p in parts if p)
    return _extract_text_from_block(content)


def _extract_text_from_block(block: Any) -> str:
    if block is None:
        return ""
    text = getattr(block, "text", None)
    if text is None and isinstance(block, dict):
        text = block.get("text")
    return str(text or "")


__all__ = [
    "SamplingRequest",
    "SamplingResponse",
    "SamplingSessionLike",
    "build_sampling_request",
    "run_sampling",
]
