"""Unit tests for ``stropha.retrieval.assemble`` (WIN-209).

``assemble_context`` is the 14th MCP tool. It composes
``SearchEngine.search`` + the graph helpers
(``find_callers`` / ``find_tests_for``) into a single
budget-aware one-shot call returning a markdown bundle.

These tests mock the search engine + graph helpers so the assembly
logic (dedup, section ordering, bundle rendering, packing) is
exercised in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stropha.models import SearchHit
from stropha.retrieval.assemble import (
    AssembleContextResult,
    AssembleSection,
    assemble_context,
)


def _hit(
    snippet: str,
    *,
    chunk_id: str | None = None,
    rel_path: str = "src/foo.py",
    start_line: int = 10,
    end_line: int = 20,
    symbol: str | None = "do_thing",
    language: str = "python",
    kind: str = "function",
    score: float = 0.9,
    rank: int = 0,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id or f"chunk-{rel_path}-{start_line}",
        rel_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        score=score,
        rank=rank,
        snippet=snippet,
        language=language,
        kind=kind,
        symbol=symbol,
    )


def _engine(hits: list[SearchHit]) -> MagicMock:
    """Build a mock SearchEngine that returns a canned hit list."""
    eng = MagicMock()
    eng.search.return_value = hits
    return eng


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_zero_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="context_budget"):
            assemble_context(
                "task",
                search_engine=_engine([]),
                context_budget=0,
            )

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="context_budget"):
            assemble_context(
                "task",
                search_engine=_engine([]),
                context_budget=-1,
            )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAssembleHappyPath:
    def test_primary_only_returns_bundle_with_primary_section(self) -> None:
        hits = [
            _hit("def foo(): pass", chunk_id="c1", symbol="foo"),
            _hit("def bar(): pass", chunk_id="c2", symbol="bar", rank=1),
        ]
        result = assemble_context(
            "how does foo work?",
            search_engine=_engine(hits),
            context_budget=4000,
        )
        assert isinstance(result, AssembleContextResult)
        # Primary section populated, callers/tests empty.
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 2
        callers = next(s for s in result.sections if s.key == "callers")
        tests = next(s for s in result.sections if s.key == "tests")
        assert callers.hits == []
        assert tests.hits == []

    def test_bundle_starts_with_task_header(self) -> None:
        result = assemble_context(
            "fix the broken thing",
            search_engine=_engine([_hit("x", chunk_id="c1")]),
            context_budget=4000,
        )
        assert result.bundle.startswith("# Context for: fix the broken thing")

    def test_bundle_includes_file_line_symbol_header_per_hit(self) -> None:
        h = _hit(
            "def do(): pass",
            chunk_id="c1",
            rel_path="src/x.py",
            start_line=42,
            end_line=50,
            symbol="UserService.do",
        )
        result = assemble_context(
            "do",
            search_engine=_engine([h]),
            context_budget=4000,
        )
        assert "### src/x.py:42-50 — UserService.do" in result.bundle
        # Fenced code block with language tag.
        assert "```python" in result.bundle
        assert "def do(): pass" in result.bundle

    def test_bundle_omits_empty_sections(self) -> None:
        result = assemble_context(
            "task",
            search_engine=_engine([_hit("x", chunk_id="c1")]),
            context_budget=4000,
        )
        # callers + tests would render '## Callers' / '## Tests'. Both
        # are empty by default → headers must NOT appear.
        assert "## Callers" not in result.bundle
        assert "## Tests" not in result.bundle
        # Primary section header DOES appear.
        assert "## Primary code" in result.bundle

    def test_section_order_primary_callers_tests(self) -> None:
        result = assemble_context(
            "task",
            search_engine=_engine([]),
            context_budget=4000,
        )
        # Order is fixed regardless of which sections are populated.
        keys = [s.key for s in result.sections]
        assert keys == ["primary", "callers", "tests"]


# ---------------------------------------------------------------------------
# Graph extensions (callers / tests)
# ---------------------------------------------------------------------------


class TestGraphExtensions:
    def test_include_callers_adds_caller_hits(self) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        caller_hit = _hit(
            "def caller(): f()",
            chunk_id="c2",
            symbol="caller",
            rel_path="src/uses_f.py",
        )
        with patch(
            "stropha.retrieval.graph.find_callers", return_value=[caller_hit]
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_callers=True,
                context_budget=4000,
            )
        callers = next(s for s in result.sections if s.key == "callers")
        assert len(callers.hits) == 1
        assert callers.hits[0].chunk_id == "c2"
        # The bundle now has the Callers section header.
        assert "## Callers" in result.bundle

    def test_include_tests_adds_test_hits(self) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        test_hit = _hit(
            "def test_f(): assert f()",
            chunk_id="c2",
            symbol="test_f",
            rel_path="tests/test_f.py",
        )
        with patch(
            "stropha.retrieval.graph.find_tests_for", return_value=[test_hit]
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_tests=True,
                context_budget=4000,
            )
        tests = next(s for s in result.sections if s.key == "tests")
        assert len(tests.hits) == 1
        assert tests.hits[0].chunk_id == "c2"

    def test_extensions_skipped_when_no_storage(self) -> None:
        """Without a storage handle the graph helpers can't run.
        ``include_callers=True`` must NOT crash — silently skip."""
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        result = assemble_context(
            "task",
            search_engine=_engine([primary]),
            storage=None,
            include_callers=True,
            include_tests=True,
            context_budget=4000,
        )
        # Primary still landed; extensions empty.
        primary_sec = next(s for s in result.sections if s.key == "primary")
        callers_sec = next(s for s in result.sections if s.key == "callers")
        tests_sec = next(s for s in result.sections if s.key == "tests")
        assert len(primary_sec.hits) == 1
        assert callers_sec.hits == []
        assert tests_sec.hits == []

    def test_extensions_skipped_when_primary_hit_has_no_symbol(
        self,
    ) -> None:
        """find_callers / find_tests_for require a symbol. A primary
        hit with ``symbol=None`` (file-level chunk) can't be extended
        — skip silently."""
        primary = _hit("contents", chunk_id="c1", symbol=None)
        with patch(
            "stropha.retrieval.graph.find_callers", return_value=[]
        ) as mock_callers:
            assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_callers=True,
                context_budget=4000,
            )
        mock_callers.assert_not_called()

    def test_graph_helper_exception_caught_and_logged(self) -> None:
        """If find_callers raises (e.g. graph not loaded), assemble
        must continue with the primary hits — never crash the agent's
        call."""
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        with patch(
            "stropha.retrieval.graph.find_callers",
            side_effect=RuntimeError("graph not loaded"),
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_callers=True,
                context_budget=4000,
            )
        primary_sec = next(s for s in result.sections if s.key == "primary")
        callers_sec = next(s for s in result.sections if s.key == "callers")
        assert len(primary_sec.hits) == 1
        assert callers_sec.hits == []

    def test_find_tests_exception_caught(self) -> None:
        """Same defensive contract for find_tests_for: graph helper
        failure must not propagate to the agent's call."""
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        with patch(
            "stropha.retrieval.graph.find_tests_for",
            side_effect=RuntimeError("graph not loaded"),
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_tests=True,
                context_budget=4000,
            )
        tests_sec = next(s for s in result.sections if s.key == "tests")
        assert tests_sec.hits == []

    def test_include_tests_skipped_when_primary_hit_has_no_symbol(
        self,
    ) -> None:
        """find_tests_for branch — same symbol-required check as
        find_callers. Covers the `continue` on the test extension
        path."""
        primary = _hit("contents", chunk_id="c1", symbol=None)
        with patch(
            "stropha.retrieval.graph.find_tests_for", return_value=[]
        ) as mock_tests:
            assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_tests=True,
                context_budget=4000,
            )
        mock_tests.assert_not_called()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_primary_dedupe_by_chunk_id(self) -> None:
        """Two SearchHits with the same chunk_id (e.g. same chunk
        surfaced from multiple search streams) must collapse to one
        in the bundle."""
        h_dup_a = _hit("def f(): pass", chunk_id="c1", rank=0)
        h_dup_b = _hit("def f(): pass", chunk_id="c1", rank=1)
        result = assemble_context(
            "task",
            search_engine=_engine([h_dup_a, h_dup_b]),
            context_budget=4000,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 1

    def test_caller_already_in_primary_filtered_out(self) -> None:
        """If a caller happens to be in the primary search results too,
        do not re-render it in the Callers section."""
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        # The "caller" is actually the same chunk as primary.
        with patch(
            "stropha.retrieval.graph.find_callers", return_value=[primary]
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_callers=True,
                context_budget=4000,
            )
        callers = next(s for s in result.sections if s.key == "callers")
        assert callers.hits == []


# ---------------------------------------------------------------------------
# Budget packing
# ---------------------------------------------------------------------------


class TestBudgetPacking:
    def test_budget_caps_total_packed_hits(self) -> None:
        """A tight budget should drop the tail. The packer's
        spilled_count surfaces what was dropped."""
        hits = [_hit("word " * 50, chunk_id=f"c{i}") for i in range(5)]
        result = assemble_context(
            "task",
            search_engine=_engine(hits),
            # Each hit is ~60 tokens (50 words × 1.2). Budget 100 fits
            # ~1-2 hits.
            context_budget=100,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) < 5
        assert result.spilled_count > 0

    def test_truncated_oversize_when_first_hit_too_big(self) -> None:
        """If the single top hit exceeds the budget, return it anyway
        and surface truncated_oversize=True."""
        big = _hit("word " * 500, chunk_id="c1")  # ~600 tokens
        result = assemble_context(
            "task",
            search_engine=_engine([big]),
            context_budget=50,
        )
        primary = next(s for s in result.sections if s.key == "primary")
        assert len(primary.hits) == 1
        assert result.truncated_oversize is True


# ---------------------------------------------------------------------------
# Rendering details
# ---------------------------------------------------------------------------


class TestRendering:
    def test_bundle_includes_section_header_with_count(self) -> None:
        h1 = _hit("a", chunk_id="c1")
        h2 = _hit("b", chunk_id="c2")
        result = assemble_context(
            "task",
            search_engine=_engine([h1, h2]),
            context_budget=4000,
        )
        assert "## Primary code (2 spans)" in result.bundle

    def test_bundle_includes_repo_hint_when_repo_present(self) -> None:
        from stropha.models import RepoRef

        h = SearchHit(
            chunk_id="c1",
            rel_path="src/foo.py",
            start_line=1, end_line=5,
            score=0.9, rank=0,
            snippet="x",
            language="python",
            kind="function",
            symbol="foo",
            repo=RepoRef(
                normalized_key="github.com/me/repo",
                url="https://github.com/me/repo.git",
                local_path="/tmp/repo",
                default_branch="main",
                head_commit="abc123",
            ),
        )
        result = assemble_context(
            "task",
            search_engine=_engine([h]),
            context_budget=4000,
        )
        assert "github.com/me/repo" in result.bundle

    def test_empty_search_yields_empty_bundle_with_task_header(self) -> None:
        result = assemble_context(
            "nothing to find",
            search_engine=_engine([]),
            context_budget=4000,
        )
        # Header always present so the agent can scan deterministically.
        assert "# Context for: nothing to find" in result.bundle
        # No section headers since no hits.
        assert "## Primary code" not in result.bundle
        assert result.total_tokens == 0
        assert result.spilled_count == 0

    def test_section_dataclass_immutability(self) -> None:
        sec = AssembleSection(key="primary", title="P", hits=[])
        with pytest.raises((AttributeError, TypeError)):
            sec.key = "other"  # type: ignore[misc]

    def test_result_dataclass_immutability(self) -> None:
        result = assemble_context(
            "task",
            search_engine=_engine([]),
            context_budget=4000,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.task = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------


class TestPublicSymbols:
    def test_imports_resolve(self) -> None:
        from stropha.retrieval.assemble import (  # noqa: F401
            AssembleContextResult,
            AssembleSection,
            assemble_context,
        )
