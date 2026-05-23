"""Unit tests for ``stropha.retrieval.antiretrieval`` (WIN-215).

Near-miss / anti-retrieval detection — flags deprecated chunks so
the agent can see them but knows not to confuse them with the
answer. Two heuristics:

1. Deprecated path markers (``deprecated/``, ``legacy/``, …).
2. Inline deprecation annotations (``@deprecated``, ``@Deprecated``,
   ``[Obsolete]``, etc.) in the snippet header.
"""

from __future__ import annotations

from stropha.models import SearchHit
from stropha.retrieval.antiretrieval import (
    is_deprecated_chunk,
    is_deprecated_path,
    partition_hits,
)


def _hit(
    rel_path: str = "src/foo.py",
    snippet: str = "def foo(): pass",
) -> SearchHit:
    return SearchHit(
        chunk_id=f"c-{rel_path}",
        rel_path=rel_path,
        start_line=1,
        end_line=10,
        score=0.9,
        rank=0,
        snippet=snippet,
        language="python",
        kind="function",
        symbol="foo",
    )


# ---------------------------------------------------------------------------
# is_deprecated_path
# ---------------------------------------------------------------------------


class TestIsDeprecatedPath:
    def test_legacy_directory(self) -> None:
        assert is_deprecated_path("src/legacy/auth.py") is True

    def test_deprecated_directory(self) -> None:
        assert is_deprecated_path("src/deprecated/old_handler.py") is True

    def test_old_directory(self) -> None:
        assert is_deprecated_path("backend/old/payment.py") is True

    def test_obsolete_directory(self) -> None:
        assert is_deprecated_path("services/obsolete/v1.py") is True

    def test_vendor_directory(self) -> None:
        assert is_deprecated_path("third_party/vendor/lib.py") is True

    def test_vendored_directory(self) -> None:
        assert is_deprecated_path("src/vendored/pkg/mod.py") is True

    def test_third_party_directory(self) -> None:
        assert is_deprecated_path("src/third_party/dep.py") is True

    def test_suffix_marker_deprecated(self) -> None:
        assert is_deprecated_path("src/auth_deprecated.py") is True

    def test_suffix_marker_legacy(self) -> None:
        assert is_deprecated_path("src/user_legacy.py") is True

    def test_suffix_marker_old(self) -> None:
        assert is_deprecated_path("src/payment_old.py") is True

    def test_dot_deprecated_in_name(self) -> None:
        assert is_deprecated_path("src/foo.deprecated.py") is True

    def test_case_insensitive(self) -> None:
        """Path matching is case-insensitive — ``Legacy`` and
        ``LEGACY`` count the same."""
        assert is_deprecated_path("src/Legacy/auth.py") is True
        assert is_deprecated_path("src/LEGACY/auth.py") is True

    def test_underscore_legacy_directory(self) -> None:
        assert is_deprecated_path("src/_legacy/old.py") is True

    def test_underscore_old_directory(self) -> None:
        assert is_deprecated_path("src/_old/foo.py") is True

    def test_regular_path_not_deprecated(self) -> None:
        assert is_deprecated_path("src/services/auth.py") is False
        assert is_deprecated_path("backend/users.py") is False
        assert is_deprecated_path("tests/test_auth.py") is False

    def test_empty_path_not_deprecated(self) -> None:
        assert is_deprecated_path("") is False


# ---------------------------------------------------------------------------
# is_deprecated_chunk — inline annotation scan
# ---------------------------------------------------------------------------


class TestIsDeprecatedChunkInline:
    def test_python_deprecated_decorator(self) -> None:
        snippet = "@deprecated\ndef old_handler():\n    pass\n"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_python_capitalized_deprecated_decorator(self) -> None:
        snippet = "@Deprecated\ndef old_handler():\n    pass\n"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_java_deprecated_annotation(self) -> None:
        snippet = "@Deprecated\npublic void oldMethod() {\n    // ...\n}\n"
        assert is_deprecated_chunk(
            _hit(rel_path="src/Old.java", snippet=snippet)
        ) is True

    def test_csharp_obsolete(self) -> None:
        snippet = "[Obsolete]\npublic void OldMethod() {\n    // ...\n}\n"
        assert is_deprecated_chunk(
            _hit(rel_path="src/Old.cs", snippet=snippet)
        ) is True

    def test_python_comment_deprecated_lowercase(self) -> None:
        snippet = "# deprecated — use new_handler instead\ndef foo(): pass"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_python_comment_deprecated_uppercase(self) -> None:
        snippet = "# DEPRECATED — use new_handler instead\ndef foo(): pass"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_c_style_comment_deprecated(self) -> None:
        snippet = "// deprecated\nint old() { return 0; }\n"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_block_comment_deprecated(self) -> None:
        snippet = "/* deprecated, use new_thing */\nint old() { return 0; }\n"
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_marker_outside_header_window_ignored(self) -> None:
        """The header scan looks at the first 5 lines only; a
        deprecation marker on line 20 should NOT flag the chunk."""
        snippet = "\n".join(
            ["def foo():"]
            + ["    pass"] * 20
            + ["# deprecated comment buried deep"]
        )
        assert is_deprecated_chunk(_hit(snippet=snippet)) is False

    def test_marker_in_first_five_lines_does_flag(self) -> None:
        snippet = "\n".join([
            "def foo():",
            "    pass",
            "# deprecated",  # line 3 — within window
            "    return None",
        ])
        assert is_deprecated_chunk(_hit(snippet=snippet)) is True

    def test_normal_chunk_not_flagged(self) -> None:
        assert is_deprecated_chunk(
            _hit(snippet="def regular():\n    return 1\n")
        ) is False

    def test_empty_snippet_not_flagged_when_path_clean(self) -> None:
        assert is_deprecated_chunk(_hit(snippet="")) is False


# ---------------------------------------------------------------------------
# is_deprecated_chunk — path takes precedence
# ---------------------------------------------------------------------------


class TestIsDeprecatedChunkPath:
    def test_deprecated_path_flagged_regardless_of_snippet(self) -> None:
        h = _hit(rel_path="src/legacy/auth.py", snippet="def foo(): pass")
        assert is_deprecated_chunk(h) is True

    def test_clean_path_with_clean_snippet_not_flagged(self) -> None:
        h = _hit(rel_path="src/auth.py", snippet="def foo(): pass")
        assert is_deprecated_chunk(h) is False


# ---------------------------------------------------------------------------
# partition_hits
# ---------------------------------------------------------------------------


class TestPartitionHits:
    def test_splits_into_positive_and_negative(self) -> None:
        hits = [
            _hit(rel_path="src/auth.py"),
            _hit(rel_path="src/legacy/auth.py"),  # near-miss
            _hit(rel_path="src/users.py"),
            _hit(rel_path="src/deprecated/payment.py"),  # near-miss
        ]
        pos, neg = partition_hits(hits)
        assert [h.rel_path for h in pos] == ["src/auth.py", "src/users.py"]
        assert [h.rel_path for h in neg] == [
            "src/legacy/auth.py", "src/deprecated/payment.py",
        ]

    def test_order_preserved(self) -> None:
        hits = [
            _hit(rel_path="src/c.py"),
            _hit(rel_path="src/a.py"),
            _hit(rel_path="src/b.py"),
        ]
        pos, neg = partition_hits(hits)
        assert [h.rel_path for h in pos] == ["src/c.py", "src/a.py", "src/b.py"]
        assert neg == []

    def test_empty_input(self) -> None:
        pos, neg = partition_hits([])
        assert pos == []
        assert neg == []

    def test_all_negative(self) -> None:
        hits = [
            _hit(rel_path="src/legacy/a.py"),
            _hit(rel_path="src/legacy/b.py"),
        ]
        pos, neg = partition_hits(hits)
        assert pos == []
        assert len(neg) == 2

    def test_inline_marker_also_partitions(self) -> None:
        hits = [
            _hit(snippet="def foo(): pass"),
            _hit(snippet="@deprecated\ndef bar(): pass"),
        ]
        pos, neg = partition_hits(hits)
        assert len(pos) == 1
        assert len(neg) == 1


# ---------------------------------------------------------------------------
# Integration: assemble_context partitions near-misses (WIN-215)
# ---------------------------------------------------------------------------


class TestAssembleNearMisses:
    def test_near_misses_get_their_own_section(self) -> None:
        from unittest.mock import MagicMock

        from stropha.retrieval.assemble import assemble_context

        hits = [
            _hit(rel_path="src/auth.py"),
            _hit(rel_path="src/legacy/auth.py"),  # near-miss
        ]
        eng = MagicMock()
        eng.search.return_value = hits
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        near_misses = next(s for s in result.sections if s.key == "near_misses")
        assert len(primary.hits) == 1
        assert len(near_misses.hits) == 1
        assert primary.hits[0].rel_path == "src/auth.py"
        assert near_misses.hits[0].rel_path == "src/legacy/auth.py"

    def test_near_misses_section_header_warning(self) -> None:
        from unittest.mock import MagicMock

        from stropha.retrieval.assemble import assemble_context

        hits = [_hit(rel_path="src/legacy/foo.py")]
        eng = MagicMock()
        eng.search.return_value = hits
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
        )
        # Bundle includes the "do NOT confuse" warning header.
        assert "do NOT confuse" in result.bundle

    def test_no_near_misses_means_no_section_header(self) -> None:
        from unittest.mock import MagicMock

        from stropha.retrieval.assemble import assemble_context

        hits = [_hit(rel_path="src/auth.py")]
        eng = MagicMock()
        eng.search.return_value = hits
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
        )
        # Bundle should NOT include the near-misses header when the
        # section is empty (assemble's empty-section suppression).
        assert "do NOT confuse" not in result.bundle


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_imports_resolve(self) -> None:
        from stropha.retrieval.antiretrieval import (  # noqa: F401
            is_deprecated_chunk,
            is_deprecated_path,
            partition_hits,
        )
