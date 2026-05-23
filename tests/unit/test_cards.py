"""Unit tests for ``stropha.retrieval.cards`` (WIN-212).

Hierarchical result cards offer three rendering levels per hit:
``line`` (1-line summary), ``paragraph`` (signature + docstring),
``full`` (the chunk's full content). Tests cover each level + the
helper that swaps snippet content on a list of SearchHits.
"""

from __future__ import annotations

import pytest

from stropha.models import SearchHit
from stropha.retrieval.cards import (
    ALLOWED_LEVELS,
    ResultCard,
    apply_level_to_snippets,
    make_cards,
    render_card,
)


def _hit(
    snippet: str,
    *,
    chunk_id: str = "c1",
    rel_path: str = "src/foo.py",
    start_line: int = 10,
    end_line: int = 30,
    symbol: str | None = "do_thing",
    language: str = "python",
    kind: str = "function",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        score=0.9,
        rank=0,
        snippet=snippet,
        language=language,
        kind=kind,
        symbol=symbol,
    )


# ---------------------------------------------------------------------------
# Line-card
# ---------------------------------------------------------------------------


class TestLineCard:
    def test_basic_shape(self) -> None:
        h = _hit("def foo():\n    pass\n", rel_path="src/x.py",
                 start_line=5, end_line=10, symbol="foo", kind="function")
        out = render_card(h, "line")
        assert out == "src/x.py:5-10  foo  [function]"

    def test_symbol_none_renders_dash(self) -> None:
        h = _hit("anything", symbol=None)
        assert "—" in render_card(h, "line")

    def test_line_card_omits_full_content(self) -> None:
        """The whole point of the line card is brevity — no snippet
        body."""
        h = _hit("def foo():\n" + "    x = 1\n" * 50)
        out = render_card(h, "line")
        assert "x = 1" not in out


# ---------------------------------------------------------------------------
# Paragraph-card
# ---------------------------------------------------------------------------


class TestParagraphCard:
    def test_python_signature_only(self) -> None:
        snippet = "\n".join([
            "def my_function(x, y):",
            "    body_a = 1",
            "    body_b = 2",
            "    body_c = 3",
            "    return body_a + body_b + body_c",
        ])
        out = render_card(_hit(snippet), "paragraph")
        assert "def my_function(x, y):" in out
        # Body NOT included.
        assert "body_a = 1" not in out

    def test_python_signature_plus_docstring(self) -> None:
        snippet = "\n".join([
            "def my_function(x, y):",
            '    """Add two numbers.',
            "    Returns: their sum.",
            '    """',
            "    return x + y",
        ])
        out = render_card(_hit(snippet), "paragraph")
        assert "def my_function" in out
        assert '"""Add two numbers.' in out
        assert "Returns: their sum." in out
        assert '"""' in out
        # Body call NOT included.
        assert "return x + y" not in out

    def test_class_signature_plus_short_comment(self) -> None:
        snippet = "\n".join([
            "class Handler:",
            "    # The main request handler",
            "    def serve(self): pass",
            "    def fail(self): pass",
        ])
        out = render_card(_hit(snippet), "paragraph")
        assert "class Handler:" in out
        assert "# The main request handler" in out
        # Method bodies NOT included.
        assert "def serve" not in out

    def test_no_signature_falls_back_to_first_five_lines(self) -> None:
        """A file-level chunk without a declaration: the paragraph
        card falls back to the first 5 lines."""
        snippet = "\n".join(f"line{i} = {i}" for i in range(20))
        out = render_card(_hit(snippet, kind="file"), "paragraph")
        # First 5 lines kept.
        for i in range(5):
            assert f"line{i} = {i}" in out
        # Line 5+ NOT included.
        assert "line10" not in out

    def test_empty_snippet_returns_empty_paragraph(self) -> None:
        assert render_card(_hit(""), "paragraph") == ""

    def test_js_signature(self) -> None:
        snippet = "\n".join([
            "function createUser(name, email) {",
            "    /* validation block */",
            "    const id = uuid();",
            "    return { id, name, email };",
            "}",
        ])
        out = render_card(_hit(snippet, language="javascript"), "paragraph")
        assert "function createUser" in out
        assert "/* validation block */" in out
        assert "uuid()" not in out


# ---------------------------------------------------------------------------
# Full-card
# ---------------------------------------------------------------------------


class TestFullCard:
    def test_full_matches_snippet(self) -> None:
        snippet = "def foo():\n    return 1\n"
        h = _hit(snippet)
        assert render_card(h, "full") == snippet

    def test_full_empty_snippet(self) -> None:
        assert render_card(_hit(""), "full") == ""


# ---------------------------------------------------------------------------
# Unknown level
# ---------------------------------------------------------------------------


class TestUnknownLevel:
    def test_unknown_level_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown card level"):
            render_card(_hit("x"), "summary")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_cards
# ---------------------------------------------------------------------------


class TestMakeCards:
    def test_returns_card_per_hit(self) -> None:
        hits = [_hit("a", chunk_id="c1"), _hit("b", chunk_id="c2")]
        cards = make_cards(hits, "full")
        assert len(cards) == 2
        assert all(isinstance(c, ResultCard) for c in cards)

    def test_card_level_field_set(self) -> None:
        cards = make_cards([_hit("def f(): pass")], "line")
        assert cards[0].level == "line"

    def test_each_card_carries_hit(self) -> None:
        h = _hit("def f(): pass", chunk_id="abc")
        cards = make_cards([h], "line")
        assert cards[0].hit is h


# ---------------------------------------------------------------------------
# apply_level_to_snippets
# ---------------------------------------------------------------------------


class TestApplyLevelToSnippets:
    def test_full_is_no_op(self) -> None:
        hits = [_hit("def f(): pass")]
        out = apply_level_to_snippets(hits, "full")
        # Returns a new list with the same hits (or equivalent).
        assert len(out) == 1
        assert out[0].snippet == hits[0].snippet

    def test_line_replaces_snippet_with_line_card(self) -> None:
        hits = [_hit(
            "def f():\n    return 1",
            rel_path="src/x.py", start_line=1, end_line=2,
            symbol="f", kind="function",
        )]
        out = apply_level_to_snippets(hits, "line")
        assert out[0].snippet == "src/x.py:1-2  f  [function]"

    def test_paragraph_replaces_snippet_with_paragraph_card(self) -> None:
        snippet = "\n".join([
            "def my_function():",
            "    x = 1",
            "    y = 2",
            "    return x + y",
        ])
        hits = [_hit(snippet)]
        out = apply_level_to_snippets(hits, "paragraph")
        assert "def my_function" in out[0].snippet
        assert "x = 1" not in out[0].snippet

    def test_preserves_other_fields(self) -> None:
        h = _hit("def f(): pass", chunk_id="my-chunk")
        out = apply_level_to_snippets([h], "line")
        # chunk_id, rel_path, etc. must be preserved.
        assert out[0].chunk_id == "my-chunk"
        assert out[0].rel_path == h.rel_path
        assert out[0].start_line == h.start_line

    def test_does_not_mutate_input(self) -> None:
        h = _hit("def f(): pass")
        original = h.snippet
        apply_level_to_snippets([h], "line")
        assert h.snippet == original  # immutable, untouched

    def test_handles_stub_with_writable_snippet(self) -> None:
        """Defensive path for tests that pass bare objects."""

        class Stub:
            def __init__(self, snippet: str) -> None:
                self.snippet = snippet
                self.rel_path = "stub.py"
                self.start_line = 1
                self.end_line = 1
                self.symbol = "stub"
                self.kind = "function"

        stub = Stub("def f():\n    pass\n")
        out = apply_level_to_snippets([stub], "line")
        # Stub had its snippet rewritten in place.
        assert "stub.py:1-1" in stub.snippet
        assert out[0] is stub

    def test_handles_frozen_stub_gracefully(self) -> None:
        """Stub with read-only ``snippet`` — apply silently skips the
        write and appends the unchanged hit."""

        class Frozen:
            __slots__ = ("_s",)

            def __init__(self, s: str) -> None:
                self._s = s

            @property
            def snippet(self) -> str:
                return self._s

            rel_path = "f.py"
            start_line = 1
            end_line = 1
            symbol = "f"
            kind = "function"

        f = Frozen("def f(): pass")
        out = apply_level_to_snippets([f], "line")
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Constants / public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_allowed_levels_constant(self) -> None:
        assert ALLOWED_LEVELS == ("line", "paragraph", "full")

    def test_default_level_is_full(self) -> None:
        h = _hit("def f(): pass")
        # No explicit level → full.
        assert render_card(h) == h.snippet

    def test_import_resolves(self) -> None:
        from stropha.retrieval.cards import (  # noqa: F401
            ALLOWED_LEVELS,
            ResultCard,
            apply_level_to_snippets,
            make_cards,
            render_card,
        )
