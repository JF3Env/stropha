"""Unit tests for snippet-freshness assertion + MCP wrapper
(WIN-247, EPIC 1.11 #85)."""

from __future__ import annotations

import hashlib
from pathlib import Path
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
        self.normalized_key = "local:test"
        self.head_commit = "abc123"


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
    def test_fresh_match(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_text("def foo():\n    return 1\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo():\n    return 1",
            rel_path="source.py", start_line=1, end_line=2,
            local_path=str(tmp_path),
        )
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1",
            lookup_fn=lambda _: hit,
        )
        assert verdict.fresh is True
        assert verdict.drift_details == []
        assert verdict.source is not None
        assert verdict.source.source_version.layer == "saved_worktree"

    def test_fresh_match_after_whitespace_normalisation(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_text("def foo():\n    return 1\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo():\n    return 1",
            rel_path="source.py", start_line=1, end_line=2,
            local_path=str(tmp_path),
        )
        verdict = assert_snippet_freshness(
            "c1",
            "def foo():\n    return 1   \n\n",
            lookup_fn=lambda _: hit,
        )
        assert verdict.fresh is True

    def test_unicode_line_separator_hashes_the_returned_line(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_text("a\u2028b\nc\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="b",
            rel_path="source.py",
            start_line=2,
            end_line=2,
            local_path=str(tmp_path),
        )

        verdict = assert_snippet_freshness("c1", "b", lookup_fn=lambda _: hit)

        assert verdict.fresh is True
        assert verdict.current_snippet == "b"
        assert verdict.source is not None
        assert verdict.source.source_slice_sha256 == hashlib.sha256(b"b\n").hexdigest()

    def test_invalid_utf8_does_not_claim_a_literal_slice_hash(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_bytes(b"a\n\xff\n")
        hit = _SrcHit(
            snippet="\ufffd",
            rel_path="source.py",
            start_line=2,
            end_line=2,
            local_path=str(tmp_path),
        )

        verdict = assert_snippet_freshness("c1", "\ufffd", lookup_fn=lambda _: hit)

        assert verdict.fresh is True
        assert verdict.source is not None
        assert verdict.source.source_slice_sha256 is None

    def test_symlink_retarget_during_read_is_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = tmp_path / "first.py"
        second = tmp_path / "second.py"
        first.write_text("first\n", encoding="utf-8")
        second.write_text("second\n", encoding="utf-8")
        alias = tmp_path / "source.py"
        alias.symlink_to(first)
        original_read_bytes = Path.read_bytes
        retargeted = False

        def retarget_after_read(path: Path) -> bytes:
            nonlocal retargeted
            raw = original_read_bytes(path)
            if path == first and not retargeted:
                alias.unlink()
                alias.symlink_to(second)
                retargeted = True
            return raw

        monkeypatch.setattr(Path, "read_bytes", retarget_after_read)
        hit = _SrcHit(
            snippet="first",
            rel_path="source.py",
            start_line=1,
            end_line=1,
            local_path=str(tmp_path),
        )

        verdict = assert_snippet_freshness(
            "c1", "first", lookup_fn=lambda _: hit
        )

        assert verdict.fresh is None
        assert verdict.source is not None
        assert verdict.source.freshness == "unknown"

    def test_chunk_vanished(self) -> None:
        verdict = assert_snippet_freshness(
            "ghost", "anything",
            lookup_fn=lambda _: None,
        )
        assert verdict.fresh is False
        assert "no longer in the index" in verdict.reason
        assert verdict.current_snippet is None

    def test_content_drifted(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_text("def foo():\n    return 99\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo():\n    return 1",
            rel_path="source.py", start_line=1, end_line=2,
            local_path=str(tmp_path),
        )
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1",
            lookup_fn=lambda _: hit,
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

    def test_strict_mode_catches_whitespace_drift(self, tmp_path: Path) -> None:
        """Strict comparison rejects whitespace-only differences that
        the lenient mode tolerates."""
        source = tmp_path / "source.py"
        source.write_text("def foo():\n    return 1   \n", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo():\n    return 1   ",
            rel_path="source.py", start_line=1, end_line=2,
            local_path=str(tmp_path),
        )
        verdict = assert_snippet_freshness(
            "c1", "def foo():\n    return 1\n",
            lookup_fn=lambda _: hit,
            strict=True,
        )
        assert verdict.fresh is False
        assert "strict" in verdict.reason

    def test_strict_mode_passes_byte_exact(self, tmp_path: Path) -> None:
        source = tmp_path / "source.py"
        source.write_text("def foo(): return 1", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo(): return 1",
            rel_path="source.py", start_line=1, end_line=1,
            local_path=str(tmp_path),
        )
        verdict = assert_snippet_freshness(
            "c1", "def foo(): return 1",
            lookup_fn=lambda _: hit,
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
        assert "no readable live source" in verdict.reason

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

    def test_cached_card_without_source_is_undetermined(self) -> None:
        hit = _Hit("class T / Members: a, b")
        v = assert_snippet_freshness(
            "c1", "class T / Members: a, b", lookup_fn=lambda _: hit,
        )
        assert v.fresh is None

    def test_subspan_containment_is_fresh(self, tmp_path: Path) -> None:
        """A cached sub-span of a larger chunk body is fresh (containment)."""
        source = tmp_path / "source.py"
        source.write_text("def foo():\n    x = 1\n    return x\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="def foo():\n    x = 1\n    return x",
            rel_path="source.py", start_line=1, end_line=3,
            local_path=str(tmp_path),
        )
        v = assert_snippet_freshness(
            "c1", "x = 1", lookup_fn=lambda _: hit,
        )
        assert v.fresh is True

    def test_current_card_is_trusted_only_when_full_file_hash_matches(
        self, tmp_path: Path,
    ) -> None:
        source = tmp_path / "T.java"
        source.write_text("class T {\n  int a;\n}\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="class T / Members: a",
            rel_path="T.java", start_line=1, end_line=3,
            local_path=str(tmp_path),
        )
        file_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        verdict = assert_snippet_freshness(
            "c1",
            "class T / Members: a",
            lookup_fn=lambda _: hit,
            indexed_file_sha256=file_hash,
        )

        assert verdict.fresh is True
        assert verdict.source is not None
        assert verdict.source.freshness == "fresh"
        assert verdict.source.source_version.file_sha256 == file_hash

    def test_stale_card_is_rejected_when_live_file_hash_changed(
        self, tmp_path: Path,
    ) -> None:
        source = tmp_path / "T.java"
        source.write_text("class T {\n  int changed;\n}\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="class T / Members: old",
            rel_path="T.java", start_line=1, end_line=3,
            local_path=str(tmp_path),
        )

        verdict = assert_snippet_freshness(
            "c1",
            "class T / Members: old",
            lookup_fn=lambda _: hit,
            indexed_file_sha256="0" * 64,
        )

        assert verdict.fresh is False
        assert verdict.source is not None
        assert verdict.source.freshness == "stale"
        assert verdict.source.source_version.dirty is True

    def test_truncated_index_snippet_matches_unchanged_source_range(
        self, tmp_path: Path,
    ) -> None:
        long_line = "x" * 500
        source = tmp_path / "source.py"
        source.write_text(f"{long_line}\nunrelated = 2\n", encoding="utf-8")
        cached = f"{'x' * 480}…"
        hit = _SrcHit(
            snippet=cached,
            rel_path="source.py",
            start_line=1,
            end_line=1,
            local_path=str(tmp_path),
        )

        verdict = assert_snippet_freshness(
            "c1",
            cached,
            lookup_fn=lambda _: hit,
            strict=True,
            indexed_file_sha256="0" * 64,
        )

        assert verdict.fresh is True
        assert verdict.source is not None
        assert verdict.source.freshness == "stale"


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


def _make_ctx(hits: dict[str, _Hit | _SrcHit | None]) -> MagicMock:
    app = MagicMock()
    app.config.target_repo = Path.cwd()
    app.storage.lookup_repo_id.return_value = 1
    app.storage.lookup_by_chunk_id.side_effect = (
        lambda cid: hits.get(cid)
    )
    app.storage.get_chunk_by_id.side_effect = lambda cid: (
        {"chunk_id": cid, "repo_id": 1} if hits.get(cid) is not None else None
    )
    app.storage.load_file_meta_map.return_value = {}
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestFreshnessMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "assert_snippet_freshness_tool" in tool_names

    def test_fresh_match(self, tmp_path: Path) -> None:
        from stropha.server import assert_snippet_freshness_tool

        source = tmp_path / "source.py"
        source.write_text("snippet text\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="snippet text",
            rel_path="source.py", start_line=1, end_line=1,
            local_path=str(tmp_path),
        )
        response = assert_snippet_freshness_tool(
            "c1", "snippet text",
            ctx=_make_ctx({"c1": hit}),
        )
        assert response.fresh is True
        assert response.current_snippet == "snippet text"
        assert response.source is not None
        assert response.source.source_version.layer == "saved_worktree"
        assert response.source.source_slice_sha256 == hashlib.sha256(
            b"snippet text\n"
        ).hexdigest()

    def test_legacy_file_hash_is_not_used_by_freshness_wrapper(
        self, tmp_path: Path,
    ) -> None:
        from stropha.server import assert_snippet_freshness_tool

        source = tmp_path / "source.py"
        source.write_text("snippet text\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="snippet text",
            rel_path="source.py",
            start_line=1,
            end_line=1,
            local_path=str(tmp_path),
        )
        ctx = _make_ctx({"c1": hit})
        app = ctx.request_context.lifespan_context
        app.storage.load_file_meta_map.return_value = {
            "source.py": MagicMock(content_hash="legacy-first-chunk-hash")
        }

        response = assert_snippet_freshness_tool("c1", "snippet text", ctx=ctx)

        assert response.fresh is True
        assert response.source is not None
        assert response.source.indexed_file_sha256 is None
        assert response.source.freshness == "unknown"
        app.storage.load_file_meta_map.assert_not_called()

    def test_storage_failure_is_unknown(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        ctx = _make_ctx({"c1": _Hit("snippet text")})
        app = ctx.request_context.lifespan_context
        app.storage.get_chunk_by_id.side_effect = RuntimeError("boom")

        response = assert_snippet_freshness_tool(
            "c1", "snippet text", ctx=ctx
        )

        assert response.fresh is None
        assert response.current_snippet is None
        assert "lookup raised" in response.reason

    def test_chunk_vanished(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        response = assert_snippet_freshness_tool(
            "ghost", "anything",
            ctx=_make_ctx({}),
        )
        assert response.fresh is False
        assert response.current_snippet is None

    def test_content_drifted_with_diff(self, tmp_path: Path) -> None:
        from stropha.server import assert_snippet_freshness_tool

        source = tmp_path / "source.py"
        source.write_text("new content\nline2\n", encoding="utf-8")
        hit = _SrcHit(
            snippet="old content\nline2",
            rel_path="source.py", start_line=1, end_line=2,
            local_path=str(tmp_path),
        )
        response = assert_snippet_freshness_tool(
            "c1", "old content\nline2",
            ctx=_make_ctx({"c1": hit}),
        )
        assert response.fresh is False
        assert response.drift_details
        assert response.current_snippet == "new content\nline2"

    def test_strict_flag_passed_through(self, tmp_path: Path) -> None:
        from stropha.server import assert_snippet_freshness_tool

        # With strict=True a trailing-whitespace difference triggers drift.
        source = tmp_path / "source.py"
        source.write_text("abc   \n", encoding="utf-8")
        hit = _SrcHit(
            snippet="abc   ",
            rel_path="source.py", start_line=1, end_line=1,
            local_path=str(tmp_path),
        )
        response = assert_snippet_freshness_tool(
            "c1", "abc\n",
            strict=True,
            ctx=_make_ctx({"c1": hit}),
        )
        assert response.fresh is False

    def test_foreign_chunk_does_not_expose_current_snippet(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        ctx = _make_ctx({"foreign": _Hit("secret snippet")})
        app = ctx.request_context.lifespan_context
        app.storage.get_chunk_by_id.side_effect = lambda _: {"repo_id": 2}

        response = assert_snippet_freshness_tool(
            "foreign", "old snippet", ctx=ctx
        )

        assert response.fresh is False
        assert response.current_snippet is None

    def test_unresolved_repository_fails_closed(self) -> None:
        from stropha.server import assert_snippet_freshness_tool

        ctx = _make_ctx({"foreign": _Hit("secret snippet")})
        app = ctx.request_context.lifespan_context
        app.storage.lookup_repo_id.return_value = None

        response = assert_snippet_freshness_tool(
            "foreign", "old snippet", ctx=ctx
        )

        assert response.fresh is None
        assert response.current_snippet is None
