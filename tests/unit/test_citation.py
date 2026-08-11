"""Unit tests for citation enforcement + MCP wrapper
(WIN-249, EPIC 1.11 #87)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.retrieval.citation import (
    Citation,
    CitationReport,
    ClaimAudit,
    audit_citations,
    classify_citation,
    extract_citations,
    split_claims,
)

# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_citation_defaults(self) -> None:
        c = Citation(fragment="x", kind="unknown")
        assert c.chunk_id == ""
        assert c.path == ""
        assert c.start_line == 0
        assert c.end_line == 0
        assert c.resolved is None
        assert c.reason == ""

    def test_claim_defaults(self) -> None:
        ca = ClaimAudit(sentence="x")
        assert ca.citations == []
        assert ca.valid_count == 0

    def test_report_defaults(self) -> None:
        r = CitationReport()
        assert r.claims == []
        assert r.total_claims == 0
        assert r.broken_citations == []

    def test_frozen(self) -> None:
        c = Citation(fragment="x", kind="unknown")
        with pytest.raises((AttributeError, Exception)):
            c.kind = "chunk_id"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# split_claims
# ---------------------------------------------------------------------------


class TestSplitClaims:
    def test_empty_returns_empty(self) -> None:
        assert split_claims("") == []
        assert split_claims("   ") == []

    def test_single_sentence(self) -> None:
        assert split_claims("Foo calls Bar.") == ["Foo calls Bar."]

    def test_multiple_sentences(self) -> None:
        out = split_claims("Foo calls Bar. Bar uses Baz.")
        assert len(out) == 2

    def test_paragraph_split(self) -> None:
        out = split_claims("Foo.\n\nBar.")
        assert out == ["Foo.", "Bar."]

    def test_heading_passes_through(self) -> None:
        out = split_claims("# Heading\n\nSome body.")
        assert "# Heading" in out
        assert "Some body." in out

    def test_internal_newlines_collapsed(self) -> None:
        out = split_claims("Line one\nstill same sentence.")
        assert out == ["Line one still same sentence."]

    def test_empty_paragraph_skipped(self) -> None:
        out = split_claims("Foo.\n\n   \n\nBar.")
        assert out == ["Foo.", "Bar."]


# ---------------------------------------------------------------------------
# classify_citation
# ---------------------------------------------------------------------------


class TestClassifyCitation:
    def test_chunk_id(self) -> None:
        sha = "sha256:" + ("a" * 64)
        cit = classify_citation(sha)
        assert cit.kind == "chunk_id"
        assert cit.chunk_id == sha

    def test_arbitrary_scheme(self) -> None:
        cit = classify_citation("scheme:something/x")
        assert cit.kind == "chunk_id"

    def test_file_line_range(self) -> None:
        cit = classify_citation("src/foo.py:10-20")
        assert cit.kind == "file_line"
        assert cit.path == "src/foo.py"
        assert cit.start_line == 10
        assert cit.end_line == 20

    def test_file_line_single(self) -> None:
        cit = classify_citation("src/foo.py:42")
        assert cit.kind == "file_line"
        assert cit.start_line == 42
        assert cit.end_line == 42

    def test_unknown_shape(self) -> None:
        cit = classify_citation("totally not a citation")
        assert cit.kind == "unknown"
        assert "neither" in cit.reason


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_no_citations(self) -> None:
        assert extract_citations("plain sentence") == []

    def test_one_citation(self) -> None:
        out = extract_citations("Claim [src/foo.py:10]")
        assert len(out) == 1
        assert out[0].path == "src/foo.py"

    def test_multiple_citations(self) -> None:
        out = extract_citations("[a:1] and [b:2]")
        assert len(out) == 2


# ---------------------------------------------------------------------------
# audit_citations
# ---------------------------------------------------------------------------


def _make_lookup(known_chunks: set[str], known_paths: set[str]):
    def _fn(cit: Citation) -> bool:
        if cit.kind == "chunk_id":
            return cit.chunk_id in known_chunks
        if cit.kind == "file_line":
            return cit.path in known_paths
        return False
    return _fn


class TestAuditCitations:
    def test_empty_text(self) -> None:
        report = audit_citations("", lookup_fn=lambda _: True)
        assert report.total_claims == 0
        assert report.cited_claims == 0
        assert report.uncited_claims == 0

    def test_all_cited_and_valid(self) -> None:
        text = (
            "Foo handles X [src/foo.py:10]. Bar handles Y [src/bar.py:20]."
        )
        report = audit_citations(
            text, lookup_fn=_make_lookup(set(), {"src/foo.py", "src/bar.py"}),
        )
        assert report.total_claims == 2
        assert report.cited_claims == 2
        assert report.uncited_claims == 0
        assert report.broken_citations == []

    def test_uncited_claim_counted(self) -> None:
        text = "Foo handles X. Bar handles Y [src/bar.py:20]."
        report = audit_citations(
            text, lookup_fn=_make_lookup(set(), {"src/bar.py"}),
        )
        assert report.uncited_claims == 1
        assert report.cited_claims == 1

    def test_broken_citation_recorded(self) -> None:
        text = "Foo handles X [src/missing.py:10]."
        report = audit_citations(
            text, lookup_fn=_make_lookup(set(), set()),  # nothing exists
        )
        assert "src/missing.py:10" in report.broken_citations
        assert report.claims[0].valid_count == 0

    def test_unknown_citation_not_looked_up(self) -> None:
        """A citation that doesn't classify (kind=unknown) must NOT
        consult lookup_fn."""
        called = {"n": 0}

        def lookup(_cit: Citation) -> bool:
            called["n"] += 1
            return True

        text = "Some claim [not a real cite shape]."
        report = audit_citations(text, lookup_fn=lookup)
        assert called["n"] == 0
        # Unknown citation still appears in claim.citations.
        assert len(report.claims[0].citations) == 1
        assert report.claims[0].citations[0].kind == "unknown"

    def test_lookup_exception_captured(self) -> None:
        def boom(_: Citation) -> bool:
            raise RuntimeError("db down")

        text = "Claim [src/foo.py:10]."
        report = audit_citations(text, lookup_fn=boom)
        assert any("db down" in n for n in report.notes)
        # Broken because lookup failed.
        assert "src/foo.py:10" in report.broken_citations

    def test_chunk_id_citation(self) -> None:
        sha = "sha256:" + ("b" * 64)
        text = f"Foo at [{sha}]."
        report = audit_citations(
            text, lookup_fn=_make_lookup({sha}, set()),
        )
        assert report.claims[0].valid_count == 1
        assert report.broken_citations == []


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import citation as mod
        assert set(mod.__all__) == {
            "Citation",
            "CitationReport",
            "ClaimAudit",
            "audit_citations",
            "classify_citation",
            "extract_citations",
            "split_claims",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(
    chunks: dict[str, object] | None = None,
    symbols: dict[str, list] | None = None,
    raise_chunk: bool = False,
    raise_symbol: bool = False,
) -> MagicMock:
    chunks = chunks or {}
    symbols = symbols or {}
    app = MagicMock()
    app.config.target_repo = Path.cwd()
    app.storage.lookup_repo_id.return_value = 1
    if raise_chunk:
        app.storage.lookup_by_chunk_id.side_effect = RuntimeError("chunk boom")
    else:
        app.storage.lookup_by_chunk_id.side_effect = (
            lambda cid: chunks.get(cid)
        )
    if raise_symbol:
        app.storage.lookup_by_symbol.side_effect = RuntimeError("sym boom")
    else:
        app.storage.lookup_by_symbol.side_effect = (
            lambda path, limit=20, **_: symbols.get(path, [])[:limit]
        )
    symbol_chunk_ids = {
        hit.chunk_id
        for hits in symbols.values()
        for hit in hits
    }
    app.storage.get_chunk_by_id.side_effect = lambda cid: (
        {"chunk_id": cid, "repo_id": 1}
        if cid in chunks or cid in symbol_chunk_ids
        else None
    )
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


def _file_hit(path: str, start: int, end: int):
    hit = MagicMock()
    hit.chunk_id = f"{path}:{start}-{end}"
    hit.rel_path = path
    hit.start_line = start
    hit.end_line = end
    return hit


class TestAuditCitationsMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "audit_citations_tool" in tool_names

    def test_no_text(self) -> None:
        from stropha.server import audit_citations_tool

        response = audit_citations_tool("", ctx=_make_ctx())
        assert response.total_claims == 0

    def test_chunk_id_resolved(self) -> None:
        from stropha.server import audit_citations_tool

        sha = "sha256:" + ("c" * 64)
        ctx = _make_ctx(chunks={sha: MagicMock()})
        response = audit_citations_tool(
            f"Claim [{sha}].", ctx=ctx,
        )
        assert response.broken_citations == []

    def test_chunk_id_missing(self) -> None:
        from stropha.server import audit_citations_tool

        sha = "sha256:" + ("d" * 64)
        response = audit_citations_tool(
            f"Claim [{sha}].", ctx=_make_ctx(),
        )
        assert sha in response.broken_citations

    def test_chunk_id_lookup_exception(self) -> None:
        from stropha.server import audit_citations_tool

        sha = "sha256:" + ("e" * 64)
        ctx = _make_ctx(raise_chunk=True)
        response = audit_citations_tool(f"Claim [{sha}].", ctx=ctx)
        # Exception swallowed -> resolved=False -> broken.
        assert sha in response.broken_citations

    def test_file_line_resolved(self) -> None:
        from stropha.server import audit_citations_tool

        ctx = _make_ctx(symbols={
            "src/foo.py": [_file_hit("src/foo.py", 5, 30)],
        })
        response = audit_citations_tool(
            "Claim [src/foo.py:10-20].", ctx=ctx,
        )
        assert response.broken_citations == []

    def test_file_line_no_overlap(self) -> None:
        from stropha.server import audit_citations_tool

        ctx = _make_ctx(symbols={
            "src/foo.py": [_file_hit("src/foo.py", 100, 200)],
        })
        response = audit_citations_tool(
            "Claim [src/foo.py:10-20].", ctx=ctx,
        )
        assert "src/foo.py:10-20" in response.broken_citations

    def test_file_line_lookup_exception(self) -> None:
        from stropha.server import audit_citations_tool

        ctx = _make_ctx(raise_symbol=True)
        response = audit_citations_tool(
            "Claim [src/foo.py:10].", ctx=ctx,
        )
        assert "src/foo.py:10" in response.broken_citations

    def test_unknown_citation_kind_not_resolved(self) -> None:
        from stropha.server import audit_citations_tool

        response = audit_citations_tool(
            "Claim [not a cite].", ctx=_make_ctx(),
        )
        # unknown citation kind -> wrapper returns False, but the
        # primitive doesn't classify it as "broken" because lookup_fn
        # is never called. Verify the structure stays consistent.
        assert response.total_claims == 1

    def test_no_path_match(self) -> None:
        """File-line lookup returns hits but for a different path."""
        from stropha.server import audit_citations_tool

        ctx = _make_ctx(symbols={
            "src/foo.py": [_file_hit("src/other.py", 1, 100)],
        })
        response = audit_citations_tool(
            "Claim [src/foo.py:10].", ctx=ctx,
        )
        assert "src/foo.py:10" in response.broken_citations

    def test_foreign_chunk_and_file_citations_are_rejected(self) -> None:
        from stropha.server import audit_citations_tool

        chunk_id = "sha256:" + ("f" * 64)
        file_hit = _file_hit("src/foo.py", 1, 20)
        ctx = _make_ctx(
            chunks={chunk_id: MagicMock()},
            symbols={"src/foo.py": [file_hit]},
        )
        app = ctx.request_context.lifespan_context
        app.storage.get_chunk_by_id.side_effect = lambda _: {"repo_id": 2}

        response = audit_citations_tool(
            f"Chunk [{chunk_id}]. File [src/foo.py:10].", ctx=ctx
        )

        assert set(response.broken_citations) == {chunk_id, "src/foo.py:10"}

    def test_unresolved_repository_fails_closed(self) -> None:
        from stropha.server import audit_citations_tool

        ctx = _make_ctx(chunks={"foreign": MagicMock()})
        app = ctx.request_context.lifespan_context
        app.storage.lookup_repo_id.return_value = None

        response = audit_citations_tool("Claim [foreign].", ctx=ctx)

        assert response.claims == []
        assert response.notes
