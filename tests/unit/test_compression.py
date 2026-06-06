"""Unit tests for ``stropha.retrieval.compression`` (WIN-210).

Extractive compression keeps the signature + matched lines + a small
context window, drops boilerplate, and inserts elision markers for
visibility. These tests exercise each contract individually + the
end-to-end ``compress_hits`` helper.
"""

from __future__ import annotations

import pytest

from stropha.models import SearchHit
from stropha.retrieval.compression import (
    compress_extractive,
    compress_hits,
)


def _hit(snippet: str, language: str = "python") -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        rel_path="src/x.py",
        start_line=1,
        end_line=len(snippet.splitlines()),
        score=0.9,
        rank=0,
        snippet=snippet,
        language=language,
        kind="function",
        symbol="example",
    )


# ---------------------------------------------------------------------------
# Short-circuit / boundary
# ---------------------------------------------------------------------------


class TestShortCircuit:
    def test_empty_snippet_returns_empty_result(self) -> None:
        result = compress_extractive("", "query")
        assert result.text == ""
        assert result.original_chars == 0
        assert result.compressed_chars == 0

    def test_tiny_snippet_passes_through_unchanged(self) -> None:
        """Snippets below the min_lines_to_compress threshold return
        unchanged — compression overhead isn't worth the win."""
        snippet = "def foo():\n    return 1\n"
        result = compress_extractive(snippet, "foo")
        # Under default min_lines_to_compress=6, this 2-line snippet
        # is too small to compress.
        assert result.text == snippet
        assert result.compressed_chars == result.original_chars

    def test_ratio_is_one_when_unchanged(self) -> None:
        snippet = "x = 1\n"
        result = compress_extractive(snippet, "x")
        assert result.ratio == 1.0

    def test_ratio_handles_empty_original(self) -> None:
        result = compress_extractive("", "x")
        # Avoid ZeroDivisionError edge — return 1.0 for empty inputs.
        assert result.ratio == 1.0


# ---------------------------------------------------------------------------
# Signature preservation
# ---------------------------------------------------------------------------


class TestSignaturePreservation:
    def test_python_def_signature_kept(self) -> None:
        snippet = "\n".join([
            "def my_function(x, y):",       # 1 - signature
            "    foo = 1",                  # 2 - boilerplate
            "    bar = 2",                  # 3 - boilerplate
            "    baz = 3",                  # 4 - boilerplate
            "    qux = 4",                  # 5 - boilerplate
            "    quux = 5",                 # 6 - boilerplate
            "    return foo + bar + baz",   # 7 - boilerplate
        ])
        # Query that doesn't match anything inside → only signature kept.
        result = compress_extractive(snippet, "nonexistent")
        assert "def my_function(x, y):" in result.text

    def test_python_class_signature_kept(self) -> None:
        snippet = "\n".join([
            "class MyClass:",
            "    def method1(self): pass",
            "    def method2(self): pass",
            "    def method3(self): pass",
            "    def method4(self): pass",
            "    def method5(self): pass",
            "    def method6(self): pass",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "class MyClass:" in result.text

    def test_javascript_function_signature_kept(self) -> None:
        snippet = "\n".join([
            "function createUser(name, email) {",
            "    const id = uuid();",
            "    const created = Date.now();",
            "    const role = 'user';",
            "    save({id, name, email, created, role});",
            "    return id;",
            "}",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "function createUser(name, email) {" in result.text

    def test_gherkin_feature_signature_kept(self) -> None:
        snippet = "\n".join([
            "Feature: Authentication",
            "  Background:",
            "    Given a user",
            "    And a password",
            "    And a session",
            "    And a token",
            "  Scenario: Logs in",
            "    When they submit",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "Feature: Authentication" in result.text

    def test_keep_signature_false_drops_signature_when_unmatched(
        self,
    ) -> None:
        """With keep_signature=False, the signature is treated like any
        other line — kept only when a query token matches."""
        snippet = "\n".join([
            "def my_function(x, y):",       # signature, no match
            "    foo = 1",
            "    bar = 2",
            "    baz = 3",
            "    qux = 4",
            "    quux = 5",
            "    return foo + bar + baz",
        ])
        result = compress_extractive(
            snippet, "nothing_matches", keep_signature=False,
        )
        # No signature kept → empty pack (just elision marker).
        assert "def my_function" not in result.text

    def test_docstring_below_signature_kept(self) -> None:
        snippet = "\n".join([
            'def my_function(x, y):',
            '    """This is the docstring.',
            '    Multiple lines describing behaviour.',
            '    """',
            '    foo = 1',
            '    bar = 2',
            '    baz = 3',
            '    qux = 4',
            '    return foo',
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert '"""This is the docstring.' in result.text
        # Closing docstring quote should also be kept (balanced).
        assert '"""' in result.text


# ---------------------------------------------------------------------------
# Matched-line extraction
# ---------------------------------------------------------------------------


class TestMatchedLines:
    def test_lines_matching_query_kept(self) -> None:
        snippet = "\n".join([
            "def process_order(order):",
            "    validate(order)",
            "    persist(order)",
            "    notify_warehouse(order)",
            "    log_event(order)",
            "    update_inventory(order)",
            "    return order.id",
        ])
        result = compress_extractive(snippet, "warehouse")
        assert "notify_warehouse(order)" in result.text

    def test_camelcase_query_splits_into_tokens(self) -> None:
        """`notify_warehouse` token should match a CamelCase
        identifier like `NotifyWarehouse` after splitting."""
        snippet = "\n".join([
            "class Handler:",
            "    def NotifyWarehouse(self):",
            "        x = 1",
            "        y = 2",
            "        z = 3",
            "        q = 4",
            "        return self.x",
        ])
        result = compress_extractive(snippet, "warehouse")
        assert "NotifyWarehouse" in result.text

    def test_no_match_returns_only_signature(self) -> None:
        """When no query token matches any line, only the signature
        + elision marker survive (when keep_signature=True)."""
        snippet = "\n".join([
            "def process(x):",
            "    a = 1",
            "    b = 2",
            "    c = 3",
            "    d = 4",
            "    e = 5",
            "    return a + b + c + d + e",
        ])
        result = compress_extractive(snippet, "absolutely_nothing")
        assert "def process(x):" in result.text
        # Body is elided.
        assert "elided" in result.text or "lines elided" in result.text

    def test_short_tokens_filtered(self) -> None:
        """Single-character tokens are too noisy — the tokenizer drops
        anything ≤1 char. Only ``len(t) > 1`` survives."""
        snippet = "\n".join([
            "def process(x):",
            "    z = 1",
            "    z = 2",
            "    z = 3",
            "    z = 4",
            "    z = 5",
            "    return z",
        ])
        # Query is a single character; should NOT match every "z" line.
        result = compress_extractive(snippet, "z")
        # Signature kept; body lines mostly elided.
        kept = result.text.count("z = ")
        assert kept <= 1  # at most a context-window line


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


class TestContextWindow:
    def test_default_window_two_lines(self) -> None:
        """Default window=2 means 2 lines kept above + 2 below."""
        snippet = "\n".join([
            "def f():",
            "    line1 = 1",
            "    line2 = 2",
            "    target_call()",      # match
            "    line4 = 4",
            "    line5 = 5",
            "    line6 = 6",
            "    line7 = 7",
            "    return None",
        ])
        result = compress_extractive(snippet, "target")
        # target line + 2 above + 2 below.
        assert "target_call()" in result.text
        assert "line2 = 2" in result.text
        assert "line4 = 4" in result.text
        # Outside the window — elided.
        assert "line7" not in result.text

    def test_zero_window_keeps_only_matched(self) -> None:
        snippet = "\n".join([
            "def f():",
            "    line1 = 1",
            "    line2 = 2",
            "    target_call()",
            "    line4 = 4",
            "    line5 = 5",
            "    line6 = 6",
        ])
        result = compress_extractive(
            snippet, "target", context_window=0,
        )
        assert "target_call()" in result.text
        # Adjacent lines NOT kept.
        assert "line2 = 2" not in result.text
        assert "line4 = 4" not in result.text

    def test_large_window_keeps_everything(self) -> None:
        snippet = "\n".join([
            "def f():",
            "    line1 = 1",
            "    line2 = 2",
            "    target_call()",
            "    line4 = 4",
            "    line5 = 5",
            "    line6 = 6",
        ])
        result = compress_extractive(
            snippet, "target", context_window=100,
        )
        # Window larger than the chunk → every line kept (no
        # elision).
        assert "line1 = 1" in result.text
        assert "line6 = 6" in result.text
        assert "elided" not in result.text


# ---------------------------------------------------------------------------
# Elision markers
# ---------------------------------------------------------------------------


class TestElisionMarkers:
    def test_marker_inserted_for_long_dropped_runs(self) -> None:
        snippet = "\n".join([
            "def f():",
            "    a = 1",
            "    b = 2",
            "    c = 3",
            "    d = 4",
            "    e = 5",
            "    target_call()",
            "    f = 6",
        ])
        result = compress_extractive(snippet, "target")
        # The 5-line block between signature and match is elided —
        # context_window=2 keeps lines 5-6 (d, e), so 3 lines (a, b,
        # c) are elided.
        assert "elided" in result.text

    def test_marker_includes_drop_count(self) -> None:
        snippet = "\n".join(
            ["def f():"]
            + [f"    line{i} = {i}" for i in range(1, 20)]
            + ["    target = 999", "    end = 'last'"]
        )
        result = compress_extractive(snippet, "target")
        # Some elision marker with a number.
        import re

        match = re.search(r"\((\d+) lines elided\)", result.text)
        assert match is not None
        assert int(match.group(1)) > 0

    def test_short_dropped_runs_silent(self) -> None:
        """elide_marker_min_lines=2 by default → single dropped lines
        don't get a marker (the marker would be longer than the
        dropped content)."""
        snippet = "\n".join([
            "def f():",
            "    a = 1",
            "    target = 1",  # matched
            "    b = 2",
            "    target = 2",  # matched
            "    c = 3",
            "    d = 4",
            "    return c",
        ])
        result = compress_extractive(
            snippet, "target", context_window=0,
        )
        # With context_window=0, between matched lines is single-line
        # drops → no marker (silently dropped).
        assert "elided" not in result.text or result.text.count("elided") <= 1


# ---------------------------------------------------------------------------
# compress_hits — end-to-end
# ---------------------------------------------------------------------------


class TestCompressHits:
    def test_compresses_every_hit(self) -> None:
        hits = [
            _hit("\n".join([
                "def f():",
                "    a = 1",
                "    b = 2",
                "    c = 3",
                "    target()",
                "    d = 4",
                "    e = 5",
                "    return d",
            ])),
            _hit("\n".join([
                "def g():",
                "    x = 1",
                "    y = 2",
                "    z = 3",
                "    target()",
                "    w = 4",
                "    return w",
            ])),
        ]
        compressed = compress_hits(hits, "target")
        # Each hit's snippet is shorter than the original.
        for original, c in zip(hits, compressed, strict=False):
            assert len(c.snippet) <= len(original.snippet)

    def test_returns_new_list_does_not_mutate_input(self) -> None:
        hits = [_hit("\n".join([
            "def f():",
            "    a = 1", "    b = 2", "    c = 3", "    d = 4",
            "    e = 5", "    f = 6",
        ]))]
        original_snippet = hits[0].snippet
        _ = compress_hits(hits, "nonexistent")
        # SearchHit is immutable but the helper shouldn't even try.
        assert hits[0].snippet == original_snippet

    def test_empty_list_returns_empty(self) -> None:
        assert compress_hits([], "query") == []

    def test_handles_non_searchhit_objects(self) -> None:
        """Defensive: bare objects with a ``snippet`` attribute should
        still work — useful for tests with simpler stubs."""

        class Stub:
            def __init__(self, snippet: str) -> None:
                self.snippet = snippet

        stub = Stub("\n".join([
            "def f():",
            "    a = 1", "    b = 2", "    c = 3",
            "    target()", "    d = 4", "    return d",
        ]))
        compress_hits([stub], "target")
        # Snippet was rewritten in place (no model_copy available).
        assert "target" in stub.snippet

    def test_handles_objects_with_readonly_snippet(self) -> None:
        """When the input object lacks both ``model_copy`` and a
        writeable ``snippet`` attribute, ``compress_hits`` should not
        crash — it appends the original object and moves on. Defensive
        path: a future refactor or third-party adapter might pass
        immutable proxies."""

        class Frozen:
            __slots__ = ("_snippet",)

            def __init__(self, snippet: str) -> None:
                self._snippet = snippet

            @property
            def snippet(self) -> str:
                return self._snippet

        # snippet is a read-only property → assigning to .snippet
        # raises AttributeError.
        f = Frozen("\n".join([
            "def f():",
            "    a = 1", "    b = 2", "    c = 3", "    d = 4",
            "    return None",
        ]))
        # Should not raise.
        out = compress_hits([f], "anything")
        assert len(out) == 1


# ---------------------------------------------------------------------------
# CompressedSnippet dataclass
# ---------------------------------------------------------------------------


class TestCompressedSnippetDataclass:
    def test_immutability(self) -> None:
        result = compress_extractive("", "x")
        with pytest.raises((AttributeError, TypeError)):
            result.text = "other"  # type: ignore[misc]

    def test_stats_match_text_length(self) -> None:
        snippet = "\n".join([
            "def f():",
            "    a = 1", "    b = 2", "    c = 3",
            "    target = 1", "    d = 4", "    return d",
        ])
        result = compress_extractive(snippet, "target")
        assert result.original_chars == len(snippet)
        assert result.compressed_chars == len(result.text)


# ---------------------------------------------------------------------------
# Integration with SearchEngine
# ---------------------------------------------------------------------------


class TestLanguageSignatures:
    """Each language pattern in _SIGNATURE_PATTERNS gets at least one
    test so a future change that breaks a language's signature
    detection is caught immediately."""

    def test_java_method_signature(self) -> None:
        snippet = "\n".join([
            "public static void createUser(String name) {",
            "    int a = 1;",
            "    int b = 2;",
            "    int c = 3;",
            "    int d = 4;",
            "    int e = 5;",
            "}",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "public static void createUser" in result.text

    def test_typescript_export_const(self) -> None:
        snippet = "\n".join([
            "export const handler = (req, res) => {",
            "    const a = 1;",
            "    const b = 2;",
            "    const c = 3;",
            "    const d = 4;",
            "    return res.send(a);",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "export const handler" in result.text

    def test_typescript_interface_signature(self) -> None:
        snippet = "\n".join([
            "export interface User {",
            "    id: string;",
            "    name: string;",
            "    email: string;",
            "    role: string;",
            "    created: number;",
            "    updated: number;",
            "}",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "export interface User" in result.text

    def test_rust_pub_fn(self) -> None:
        snippet = "\n".join([
            "pub fn create_user(name: String) -> User {",
            "    let a = 1;",
            "    let b = 2;",
            "    let c = 3;",
            "    let d = 4;",
            "    User { name }",
            "}",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "pub fn create_user" in result.text

    def test_go_func_signature(self) -> None:
        snippet = "\n".join([
            "func CreateUser(name string) *User {",
            "    var a int = 1",
            "    var b int = 2",
            "    var c int = 3",
            "    var d int = 4",
            "    return &User{Name: name}",
            "}",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "func CreateUser" in result.text

    def test_markdown_heading_signature(self) -> None:
        snippet = "\n".join([
            "## Architecture",
            "This module is responsible for ...",
            "Some more content here",
            "Even more lines",
            "And another",
            "And yet another",
            "Final line",
        ])
        result = compress_extractive(snippet, "nonexistent")
        assert "## Architecture" in result.text

    def test_blank_line_not_signature(self) -> None:
        """Empty / whitespace-only lines must never match a signature
        pattern — they would anchor the wrong thing."""
        from stropha.retrieval.compression import _is_signature

        assert _is_signature("") is False
        assert _is_signature("   ") is False
        assert _is_signature("\t\t") is False

    def test_arbitrary_text_not_a_signature(self) -> None:
        """A non-blank line that doesn't match any language pattern
        returns False — covers the final `return False` after the
        for-loop scans every pattern."""
        from stropha.retrieval.compression import _is_signature

        assert _is_signature("just a regular line of text") is False
        assert _is_signature("    indented body") is False
        assert _is_signature("x = 1") is False


class TestPureCommentEdges:
    def test_blank_line_not_pure_comment(self) -> None:
        from stropha.retrieval.compression import _is_pure_comment

        assert _is_pure_comment("") is False
        assert _is_pure_comment("    ") is False


class TestTrailingElision:
    def test_marker_at_end_when_chunk_ends_in_dropped_run(self) -> None:
        """If the matched lines + signature don't cover the END of
        the chunk, the trailing dropped run should also produce an
        elision marker — not silently truncate."""
        snippet = "\n".join([
            "def f():",
            "    target = 1",  # matched
            "    a = 1",
            "    b = 2",
            "    c = 3",
            "    d = 4",
            "    e = 5",
            "    f = 6",
            "    g = 7",
            "    return None",
        ])
        result = compress_extractive(
            snippet, "target", context_window=0,
        )
        # The trailing 9 lines after "target = 1" should be elided.
        assert "elided" in result.text


class TestSearchEngineCompress:
    def _engine(self, snippets: list[str]):
        from unittest.mock import MagicMock

        from stropha.retrieval.search import SearchEngine

        embedder = MagicMock()
        embedder.embed_query.return_value = [0.0] * 8
        storage = MagicMock()
        storage.search_dense.return_value = [
            _hit(s) for s in snippets
        ]
        storage.search_bm25.return_value = []
        storage.search_symbol_tokens.return_value = []
        return SearchEngine(storage, embedder)

    def test_compress_kwarg_shrinks_snippets(self) -> None:
        big_snippet = "\n".join(
            ["def f():"] + [f"    line{i} = {i}" for i in range(20)]
            + ["    target = 99", "    return target"]
        )
        eng = self._engine([big_snippet])
        with_compress = eng.search("target", top_k=1, compress=True)
        without_compress = eng.search("target", top_k=1, compress=False)
        assert len(with_compress[0].snippet) < len(without_compress[0].snippet)

    def test_compress_false_passes_through(self) -> None:
        snippet = "def f():\n    return 1\n"
        eng = self._engine([snippet])
        out = eng.search("foo", top_k=1, compress=False)
        assert out[0].snippet == snippet
