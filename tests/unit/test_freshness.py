"""Unit tests for snippet-freshness assertion + MCP wrapper
(WIN-247, EPIC 1.11 #85)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stropha.retrieval.freshness import (
    FreshnessVerdict,
    _diff_lines,
    _normalise,
    assert_snippet_freshness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Hit:
    def __init__(self, snippet: str = "") -> None:
        self.snippet = snippet


class _Repo:
    def __init__(self, local_path: str) -> None:
        self.local_path = local_path


class _SrcHit:
    """Hit that carries enough to read the real source slice from disk."""

    def __init__(
        self, *, snippet: str, rel_path: str, start_line: int, end_line: int,
        local_path: str, kind: str = "class",
    ) -> None:
        self.snippet = snippet
        self.rel_path = rel_path
        self.start_line = start_line
        self.end_line = end_line
        self.kind = kind
        self.repo = _Repo(local_path)


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclass:
    def test_defaults(self) -> None:
        v = FreshnessVerdict(chunk_id="c1", fresh=True)
        assert v.reason == ""
        assert v.current_snippet is None
        assert v.expected_snippet == ""
        assert v.drift_details == []

    def test_frozen(self) -> None:
        v = FreshnessVerdict(chunk_id="c1", fresh=True)
        with pytest.raises((AttributeError, Exception)):
            v.fresh = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_strips_trailing_whitespace(self) -> None:
        assert _normalise("abc   \n") == "abc"

    def test_drops_leading_trailing_blank_lines(self) -> None:
        assert _normalise("\n\nabc\n\n") == "abc"

    def test_preserves_internal_blanks(self) -> None:
        assert _normalise("a\n\nb") == "a\n\nb"

    def test_empty_returns_empty(self) -> None:
        assert _normalise("") == ""


# ---------------------------------------------------------------------------
# _diff_lines
# ---------------------------------------------------------------------------


class TestDiffLines:
    def test_identical_returns_empty(self) -> None:
        assert _diff_lines("a\nb", "a\nb") == []

    def test_modified_line(self) -> None:
        out = _diff_lines("a\nb\nc", "a\nXX\nc")
        assert any("~L2" in line for line in out)

    def test_added_line(self) -> None:
        out = _diff_lines("a", "a\nb")
        assert any("+L2" in line for line in out)

    def test_removed_line(self) -> None:
        out = _diff_lines("a\nb", "a")
        assert any("-L2" in line for line in out)

    def test_limit_truncates(self) -> None:
        exp = "\n".join(f"L{i}" for i in range(10))
        cur = "\n".join(f"X{i}" for i in range(10))
        out = _diff_lines(exp, cur, limit=3)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# assert_snippet_freshness
# ---------------------------------------------------------------------------


class TestAssertFreshness:
    def test_fresh_match(self) -> None:
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1",
            lookup_fn=lambda _: _Hit("def foo():\n    return 1"),
        )
        assert verdict.fresh is True
        assert verdict.drift_details == []

    def test_fresh_match_after_whitespace_normalisation(self) -> None:
        verdict = assert_snippet_freshness(
            "c1",
            "def foo():\n    return 1   \n\n",
            lookup_fn=lambda _: _Hit("def foo():\n    return 1\n"),
        )
        assert verdict.fresh is True

    def test_chunk_vanished(self) -> None:
        verdict = assert_snippet_freshness(
            "ghost", "anything",
            lookup_fn=lambda _: None,
        )
        assert verdict.fresh is False
        assert "no longer in the index" in verdict.reason
        assert verdict.current_snippet is None

    def test_content_drifted(self) -> None:
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1",
            lookup_fn=lambda _: _Hit("def foo():\n    return 99"),
        )
        assert verdict.fresh is False
        assert verdict.drift_details
        assert "drifted" in verdict.reason
        assert verdict.current_snippet == "def foo():\n    return 99"

    def test_lookup_exception_returns_none_verdict(self) -> None:
        def boom(_: str):
            raise RuntimeError("db boom")

        verdict = assert_snippet_freshness(
            "c1", "anything", lookup_fn=boom,
        )
        assert verdict.fresh is None
        assert "db boom" in verdict.reason

    def test_strict_mode_catches_whitespace_drift(self) -> None:
        """Strict comparison rejects whitespace-only differences that
        the lenient mode tolerates."""
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1\n",
            lookup_fn=lambda _: _Hit("def foo():\n    return 1   \n"),
            strict=True,
        )
        assert verdict.fresh is False
        assert "strict" in verdict.reason

    def test_strict_mode_passes_byte_exact(self) -> None:
        verdict = assert_snippet_freshness(
            "c1", "def foo(): return 1",
            lookup_fn=lambda _: _Hit("def foo(): return 1"),
            strict=True,
        )
        assert verdict.fresh is True

    def test_missing_snippet_and_no_source_is_undetermined(self) -> None:
        """Hit with neither a snippet nor readable source → fresh=None
        (undetermined), not a misleading False."""
        class _Bare:
            pass

        verdict = assert_snippet_freshness(
            "c1", "expected",
            lookup_fn=lambda _: _Bare(),
        )
        assert verdict.fresh is None
        assert "no source or stored snippet" in verdict.reason

    def test_container_card_fresh_when_source_unchanged(self, tmp_path) -> None:
        """The Problem-3 case: a class chunk's stored snippet is an outline
        card, but the agent cached the real SOURCE. With the file unchanged on
        disk, freshness verifies against source → fresh=True (was false drift)."""
        f = tmp_path / "T.java"
        f.write_text("class T {\n  int a;\n  int b;\n}\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="class T / Members: a, b",  # the misleading card
            rel_path="T.java", start_line=1, end_line=4,
            local_path=str(tmp_path), kind="class",
        )
        v = assert_snippet_freshness(
            "c1", "class T {\n  int a;\n  int b;\n}", lookup_fn=lambda _: hit,
        )
        assert v.fresh is True
        assert (v.current_snippet or "").startswith("class T {")  # source, not the card

    def test_source_drift_detected(self, tmp_path) -> None:
        f = tmp_path / "T.java"
        f.write_text("class T {\n  int a;\n  int c;\n}\n", encoding="utf-8")  # b -> c
        hit = _SrcHit(
            snippet="class T / Members: a, b",
            rel_path="T.java", start_line=1, end_line=4,
            local_path=str(tmp_path), kind="class",
        )
        v = assert_snippet_freshness(
            "c1", "class T {\n  int a;\n  int b;\n}", lookup_fn=lambda _: hit,
        )
        assert v.fresh is False
        assert "drifted" in v.reason

    def test_cached_card_still_matches_card_without_source(self) -> None:
        """When the agent cached the card itself and no source is available,
        the card still matches → fresh=True (no false drift)."""
        hit = _Hit("class T / Members: a, b")
        v = assert_snippet_freshness(
            "c1", "class T / Members: a, b", lookup_fn=lambda _: hit,
        )
        assert v.fresh is True

    def test_subspan_containment_is_fresh(self) -> None:
        """A cached sub-span of a larger chunk body is fresh (containment)."""
        hit = _Hit("def foo():\n    x = 1\n    return x")
        v = assert_snippet_freshness(
            "c1", "x = 1", lookup_fn=lambda _: hit,
        )
        assert v.fresh is True


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import freshness as mod
        assert set(mod.__all__) == {
            "FreshnessVerdict",
            "assert_snippet_freshness",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(hits: dict[str, _Hit | None]) -> MagicMock:
    app = MagicMock()
    app.storage.lookup_by_chunk_id.side_effect = (
        lambda cid: hits.get(cid)
    )
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestFreshnessMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "assert_snippet_freshness_tool" in tool_names

    def test_fresh_match(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        response = assert_snippet_freshness_tool(
            "c1", "snippet text",
            ctx=_make_ctx({"c1": _Hit("snippet text")}),
        )
        assert response.fresh is True
        assert response.current_snippet == "snippet text"

    def test_chunk_vanished(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        response = assert_snippet_freshness_tool(
            "ghost", "anything",
            ctx=_make_ctx({}),
        )
        assert response.fresh is False
        assert response.current_snippet is None

    def test_content_drifted_with_diff(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        response = assert_snippet_freshness_tool(
            "c1", "old content\nline2",
            ctx=_make_ctx({"c1": _Hit("new content\nline2")}),
        )
        assert response.fresh is False
        assert response.drift_details
        assert response.current_snippet == "new content\nline2"

    def test_strict_flag_passed_through(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        # With strict=True a trailing-whitespace difference triggers drift.
        response = assert_snippet_freshness_tool(
            "c1", "abc\n",
            strict=True,
            ctx=_make_ctx({"c1": _Hit("abc   \n")}),
        )
        assert response.fresh is False
