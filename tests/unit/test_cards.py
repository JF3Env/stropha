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
    DecaySchedule,
    ResultCard,
    apply_level_to_snippets,
    apply_relevance_decay,
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
            DecaySchedule,
            ResultCard,
            apply_level_to_snippets,
            apply_relevance_decay,
            make_cards,
            render_card,
        )


# ---------------------------------------------------------------------------
# DecaySchedule + apply_relevance_decay (WIN-213)
# ---------------------------------------------------------------------------


class TestDecaySchedule:
    def test_default_schedule(self) -> None:
        s = DecaySchedule()
        assert s.full_through == 1
        assert s.paragraph_through == 3
        # Default boundaries.
        assert s.level_for(0) == "full"
        assert s.level_for(1) == "paragraph"
        assert s.level_for(2) == "paragraph"
        assert s.level_for(3) == "line"
        assert s.level_for(100) == "line"

    def test_custom_schedule_boundaries(self) -> None:
        s = DecaySchedule(full_through=2, paragraph_through=5)
        assert s.level_for(0) == "full"
        assert s.level_for(1) == "full"
        assert s.level_for(2) == "paragraph"
        assert s.level_for(4) == "paragraph"
        assert s.level_for(5) == "line"

    def test_schedule_dataclass_immutable(self) -> None:
        s = DecaySchedule()
        with pytest.raises((AttributeError, TypeError)):
            s.full_through = 5  # type: ignore[misc]

    def test_aggressive_schedule_full_zero(self) -> None:
        """``full_through=0`` makes EVERY hit paragraph-or-line —
        nothing rendered at full. Useful for very tight budgets."""
        s = DecaySchedule(full_through=0, paragraph_through=3)
        assert s.level_for(0) == "paragraph"
        assert s.level_for(2) == "paragraph"
        assert s.level_for(3) == "line"


class TestApplyRelevanceDecay:
    def _make_ranked_hits(self, count: int) -> list[SearchHit]:
        out: list[SearchHit] = []
        for rank in range(count):
            out.append(
                _hit(
                    "\n".join([
                        "def my_function():",
                        '    """The docstring."""',
                        "    a = 1",
                        "    b = 2",
                        "    return a + b",
                    ]),
                    chunk_id=f"c{rank}",
                )
            )
        # Patch ranks since _hit always sets rank=0.
        return [
            h.model_copy(update={"rank": i, "symbol": f"sym_{i}"})
            for i, h in enumerate(out)
        ]

    def test_default_schedule_applies(self) -> None:
        hits = self._make_ranked_hits(5)
        out = apply_relevance_decay(hits)
        # Top hit gets full content.
        assert "a = 1" in out[0].snippet
        # Ranks 1-2 get paragraph (signature + docstring).
        assert "def my_function" in out[1].snippet
        assert "a = 1" not in out[1].snippet
        # Ranks 3+ get line card.
        assert out[3].snippet.startswith("src/foo.py:")
        assert "def my_function" not in out[3].snippet

    def test_custom_schedule_respected(self) -> None:
        hits = self._make_ranked_hits(5)
        out = apply_relevance_decay(
            hits, DecaySchedule(full_through=2, paragraph_through=4),
        )
        # Ranks 0-1 full.
        assert "a = 1" in out[0].snippet
        assert "a = 1" in out[1].snippet
        # Ranks 2-3 paragraph.
        assert "def my_function" in out[2].snippet
        assert "a = 1" not in out[2].snippet
        # Rank 4 line.
        assert out[4].snippet.startswith("src/foo.py:")

    def test_empty_hits(self) -> None:
        assert apply_relevance_decay([]) == []

    def test_preserves_other_fields(self) -> None:
        hits = self._make_ranked_hits(3)
        out = apply_relevance_decay(hits)
        for orig, decayed in zip(hits, out, strict=False):
            assert decayed.chunk_id == orig.chunk_id
            assert decayed.rank == orig.rank
            assert decayed.score == orig.score

    def test_does_not_mutate_input(self) -> None:
        hits = self._make_ranked_hits(3)
        originals = [h.snippet for h in hits]
        apply_relevance_decay(hits)
        for h, orig in zip(hits, originals, strict=False):
            assert h.snippet == orig  # immutable, untouched

    def test_handles_stub_with_writable_snippet(self) -> None:
        class Stub:
            def __init__(self, rank: int) -> None:
                self.rank = rank
                self.snippet = "def f():\n    return 1\n"
                self.rel_path = "x.py"
                self.start_line = 1
                self.end_line = 2
                self.symbol = "f"
                self.kind = "function"

        stub = Stub(rank=5)
        out = apply_relevance_decay([stub])
        # Rank 5 → line card.
        assert stub.snippet.startswith("x.py:1-2")
        assert out[0] is stub

    def test_handles_frozen_stub(self) -> None:
        """Defensive: read-only ``snippet`` attribute on the stub —
        append unchanged, no crash."""

        class Frozen:
            __slots__ = ("_r", "_s")

            def __init__(self, snippet: str, rank: int) -> None:
                self._s = snippet
                self._r = rank

            @property
            def snippet(self) -> str:
                return self._s

            @property
            def rank(self) -> int:
                return self._r

            rel_path = "f.py"
            start_line = 1
            end_line = 1
            symbol = "f"
            kind = "function"

        f = Frozen("def f(): pass", rank=0)
        out = apply_relevance_decay([f])
        # Length preserved; original untouched.
        assert len(out) == 1
        assert f.snippet == "def f(): pass"
