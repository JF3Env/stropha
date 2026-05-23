"""Unit tests for MCP sampling primitive (WIN-255, EPIC 1.12 #92)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from stropha.inference.mcp_sampling import (
    SamplingRequest,
    SamplingResponse,
    _extract_text,
    _extract_text_from_block,
    build_sampling_request,
    run_sampling,
)


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_request_defaults(self) -> None:
        r = SamplingRequest()
        assert r.messages == []
        assert r.max_tokens == 512
        assert r.system_prompt is None
        assert r.model_preferences == {}
        assert r.temperature is None

    def test_response_defaults(self) -> None:
        r = SamplingResponse()
        assert r.text == ""
        assert r.model == ""
        assert r.stop_reason == ""
        assert r.ok is None
        assert r.error == ""

    def test_frozen(self) -> None:
        r = SamplingRequest()
        with pytest.raises((AttributeError, Exception)):
            r.max_tokens = 1024  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_sampling_request
# ---------------------------------------------------------------------------


class TestBuildRequest:
    def test_basic(self) -> None:
        req = build_sampling_request("hello")
        assert req.max_tokens == 512
        assert len(req.messages) == 1
        assert req.messages[0]["role"] == "user"
        assert req.messages[0]["content"]["text"] == "hello"

    def test_system_prompt_attached(self) -> None:
        req = build_sampling_request("hi", system_prompt="be terse")
        assert req.system_prompt == "be terse"

    def test_model_hints(self) -> None:
        req = build_sampling_request(
            "hi", model_hints=["claude-3-5", "gpt-4o"],
        )
        assert req.model_preferences["hints"] == [
            {"name": "claude-3-5"}, {"name": "gpt-4o"},
        ]

    def test_model_hints_empty_skipped(self) -> None:
        req = build_sampling_request("hi", model_hints=["", "claude"])
        assert req.model_preferences["hints"] == [{"name": "claude"}]

    def test_priorities(self) -> None:
        req = build_sampling_request(
            "hi", intelligence_priority=0.9, speed_priority=0.1,
        )
        assert req.model_preferences["intelligencePriority"] == 0.9
        assert req.model_preferences["speedPriority"] == 0.1

    def test_temperature(self) -> None:
        req = build_sampling_request("hi", temperature=0.3)
        assert req.temperature == 0.3

    def test_max_tokens_clamp(self) -> None:
        req = build_sampling_request("hi", max_tokens=0)
        assert req.max_tokens == 1  # max(1, 0)

    def test_no_hints_no_preferences(self) -> None:
        req = build_sampling_request("hi")
        assert req.model_preferences == {}


# ---------------------------------------------------------------------------
# _extract_text helpers
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_none_returns_empty(self) -> None:
        assert _extract_text(None) == ""

    def test_object_with_text_content(self) -> None:
        result = MagicMock(content=MagicMock(text="hello"))
        assert _extract_text(result) == "hello"

    def test_dict_with_text_content(self) -> None:
        assert _extract_text({"content": {"type": "text", "text": "hi"}}) == "hi"

    def test_list_content(self) -> None:
        result = {"content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}
        assert _extract_text(result) == "ab"

    def test_missing_content_returns_empty(self) -> None:
        assert _extract_text({"other": "x"}) == ""

    def test_block_dict_no_text(self) -> None:
        assert _extract_text_from_block({"type": "text"}) == ""

    def test_block_object_no_text(self) -> None:
        class _Bare:
            pass

        assert _extract_text_from_block(_Bare()) == ""

    def test_block_none(self) -> None:
        assert _extract_text_from_block(None) == ""


# ---------------------------------------------------------------------------
# run_sampling
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


class _Session:
    """Minimal MCP-session-like mock with an async create_message."""

    def __init__(self, *, result=None, raises: bool = False) -> None:
        self.result = result
        self.raises = raises
        self.last_kwargs: dict[str, Any] | None = None

    async def create_message(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self.raises:
            raise RuntimeError("client refused")
        return self.result


class TestRunSampling:
    def test_text_response(self) -> None:
        result = MagicMock(
            content=MagicMock(text="hello world"),
            model="claude-3-5",
            stop_reason="end_turn",
        )
        session = _Session(result=result)
        req = build_sampling_request("hi")
        out = _run(run_sampling(session, req))
        assert out.ok is True
        assert out.text == "hello world"
        assert out.model == "claude-3-5"
        assert out.stop_reason == "end_turn"

    def test_empty_response(self) -> None:
        result = MagicMock(content=MagicMock(text=""), model="x", stop_reason="")
        session = _Session(result=result)
        req = build_sampling_request("hi")
        out = _run(run_sampling(session, req))
        assert out.ok is False
        assert "empty" in out.error

    def test_exception_returns_none_ok(self) -> None:
        session = _Session(raises=True)
        req = build_sampling_request("hi")
        out = _run(run_sampling(session, req))
        assert out.ok is None
        assert "client refused" in out.error

    def test_kwargs_propagated(self) -> None:
        session = _Session(result=MagicMock(content=MagicMock(text="ok")))
        req = build_sampling_request(
            "hi",
            system_prompt="be terse",
            model_hints=["claude"],
            temperature=0.5,
        )
        _run(run_sampling(session, req))
        assert session.last_kwargs is not None
        assert session.last_kwargs["system_prompt"] == "be terse"
        assert session.last_kwargs["temperature"] == 0.5
        assert "hints" in session.last_kwargs["model_preferences"]

    def test_kwargs_minimal_when_no_preferences(self) -> None:
        session = _Session(result=MagicMock(content=MagicMock(text="ok")))
        req = build_sampling_request("hi")
        _run(run_sampling(session, req))
        assert session.last_kwargs is not None
        assert "system_prompt" not in session.last_kwargs
        assert "temperature" not in session.last_kwargs
        assert "model_preferences" not in session.last_kwargs


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.inference import mcp_sampling as mod
        assert set(mod.__all__) == {
            "SamplingRequest",
            "SamplingResponse",
            "SamplingSessionLike",
            "build_sampling_request",
            "run_sampling",
        }
