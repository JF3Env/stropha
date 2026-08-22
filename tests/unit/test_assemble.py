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


@pytest.fixture(autouse=True)
def _loaded_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stropha.retrieval.graph.graph_loaded",
        lambda storage, **kwargs: True,
    )


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
    def test_broad_scope_supports_legacy_search_signature(self) -> None:
        class LegacySearch:
            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                assert query == "task"
                return [_hit("x", chunk_id="c1")][:top_k]

        result = assemble_context(
            "task", search_engine=LegacySearch(), context_budget=4000
        )

        primary = next(section for section in result.sections if section.key == "primary")
        assert [hit.chunk_id for hit in primary.hits] == ["c1"]

    def test_concrete_repo_id_is_always_passed(self) -> None:
        engine = _engine([])

        assemble_context(
            "task", search_engine=engine, context_budget=4000, repo_id=7
        )

        assert engine.search.call_args.kwargs["repo_id"] == 7

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

    def test_section_order_primary_near_misses_callers_tests_rationale(
        self,
    ) -> None:
        result = assemble_context(
            "task",
            search_engine=_engine([]),
            context_budget=4000,
        )
        # Order is fixed regardless of which sections are populated.
        # WIN-215 inserts near_misses between primary and callers so
        # the agent sees "real answer + the deprecated lookalikes
        # right next to it" before reading caller context. WIN-219
        # appends rationale (docs/ADRs) at the end — agent reads
        # "what" before "why".
        keys = [s.key for s in result.sections]
        assert keys == [
            "primary", "near_misses", "callers", "tests", "rationale",
        ]


# ---------------------------------------------------------------------------
# Graph extensions (callers / tests)
# ---------------------------------------------------------------------------


class TestGraphExtensions:
    def test_include_callers_adds_caller_hits(self) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        payload = {
            "symbol": "f",
            "resolved_node": "f",
            "callers": [
                {
                    "node_id": "caller-node",
                    "label": "caller",
                    "file_type": "function",
                    "chunk": {
                        "rel_path": "src/uses_f.py",
                        "start_line": 10,
                        "end_line": 20,
                        "snippet": "def caller(): f()",
                    },
                },
            ],
        }
        with patch(
            "stropha.retrieval.graph.find_callers", return_value=payload,
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
        assert callers.hits[0].chunk_id == "graph:caller-node"
        assert callers.hits[0].rel_path == "src/uses_f.py"
        assert callers.hits[0].snippet == "def caller(): f()"
        assert result.graph_status == "available"
        assert result.notes == []
        # The bundle now has the Callers section header.
        assert "## Callers" in result.bundle

    def test_include_tests_adds_test_hits(self) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        payload = {
            "symbol": "f",
            "resolved_node": "f",
            "tests": [
                {
                    "node_id": "test-node",
                    "label": "test_f",
                    "source_file": "tests/test_f.py",
                    "source_location": "L10",
                    "edge": {"confidence_score": 0.9},
                },
            ],
        }
        with patch(
            "stropha.retrieval.graph.find_tests_for", return_value=payload,
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
        assert tests.hits[0].chunk_id == "graph:test-node"
        assert tests.hits[0].rel_path == "tests/test_f.py"
        assert tests.hits[0].start_line == 10
        assert tests.hits[0].score == 0.9

    def test_extensions_skipped_when_no_storage(self) -> None:
        """Without storage, graph extensions are explicitly unavailable."""
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
        assert result.graph_status == "unavailable"
        assert result.notes

    def test_graph_unavailable_skips_helpers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        monkeypatch.setattr(
            "stropha.retrieval.graph.graph_loaded",
            lambda storage, **kwargs: False,
        )
        with patch("stropha.retrieval.graph.find_callers") as mock_callers:
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_callers=True,
                context_budget=4000,
            )

        mock_callers.assert_not_called()
        assert result.graph_status == "unavailable"
        assert any("graph is unavailable" in note for note in result.notes)

    def test_malformed_graph_payload_is_reported_as_degraded(self) -> None:
        primary = _hit("def f(): pass", chunk_id="c1", symbol="f")
        with patch(
            "stropha.retrieval.graph.find_callers",
            return_value=[{"node_id": "legacy-list"}],
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
        assert result.graph_status == "degraded"
        assert any("malformed response" in note for note in result.notes)

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
        assert result.graph_status == "degraded"
        assert any("find_callers" in note for note in result.notes)

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
        assert result.graph_status == "degraded"
        assert any("find_tests_for" in note for note in result.notes)

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
# include_rationale (WIN-219)
# ---------------------------------------------------------------------------


class TestRationaleSection:
    """Cross-source fusion (WIN-219): assemble_context can pull
    ``rationale_for`` graph edges and pack the docs/ADRs into a
    dedicated section, deduped against the primary search results."""

    def test_include_rationale_adds_rationale_section(self) -> None:
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={
                "symbol": "auth",
                "resolved_node": "auth",
                "rationale": [
                    {
                        "node_id": "adr-0042",
                        "label": "ADR-0042: JWT auth decision",
                        "kind": "adr",
                        "chunk": {
                            "rel_path": "docs/adr/0042.md",
                            "start_line": 4,
                            "end_line": 12,
                            "snippet": "JWT auth decision",
                        },
                        "edge": {"confidence_score": 0.92},
                    },
                ],
            },
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        assert len(rationale.hits) == 1
        assert "ADR-0042" in rationale.hits[0].symbol
        assert rationale.hits[0].rel_path == "docs/adr/0042.md"
        assert rationale.hits[0].snippet == "JWT auth decision"
        # Rationale section header rendered.
        assert "Rationale" in result.bundle

    def test_rationale_without_storage_reports_unavailable(self) -> None:
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        result = assemble_context(
            "task",
            search_engine=_engine([primary]),
            storage=None,
            include_rationale=True,
            context_budget=4000,
        )
        rationale = next(s for s in result.sections if s.key == "rationale")
        assert rationale.hits == []
        assert result.graph_status == "unavailable"
        assert result.notes

    def test_rationale_skipped_when_primary_has_no_symbol(self) -> None:
        primary = _hit("contents", chunk_id="c1", symbol=None)
        with patch(
            "stropha.retrieval.graph.find_rationale", return_value={}
        ) as mock_rat:
            assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        # No symbol → no call (continue branch).
        mock_rat.assert_not_called()

    def test_find_rationale_exception_caught(self) -> None:
        """find_rationale failure must NOT crash the agent's call —
        same defensive contract as find_callers / find_tests_for."""
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        with patch(
            "stropha.retrieval.graph.find_rationale",
            side_effect=RuntimeError("graph not loaded"),
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        assert rationale.hits == []
        assert result.graph_status == "degraded"
        assert any("find_rationale" in note for note in result.notes)

    def test_rationale_node_without_label_dropped(self) -> None:
        """A node without a ``label`` field can't be acted on by the
        agent (no identity). ``_rationale_node_to_hit`` returns None
        and the entry is silently dropped."""
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={
                "symbol": "auth",
                "rationale": [
                    {"node_id": "no-label", "kind": "doc"},  # missing label
                    {"node_id": "good", "label": "Real ADR", "kind": "adr"},
                ],
            },
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        # Only the labelled entry survives.
        assert len(rationale.hits) == 1
        assert rationale.hits[0].symbol == "Real ADR"

    def test_rationale_node_with_confidence_populates_provenance(
        self,
    ) -> None:
        """A rationale node's edge confidence_score becomes the
        synthetic hit's ``streams`` map under the ``rationale`` key —
        WIN-214 provenance picks this up automatically."""
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={
                "symbol": "auth",
                "rationale": [
                    {
                        "node_id": "adr",
                        "label": "ADR",
                        "kind": "adr",
                        "edge": {"confidence_score": 0.87},
                    },
                ],
            },
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        assert rationale.hits[0].streams == {"rationale": 0.87}

    def test_rationale_uses_chunk_path_when_available(self) -> None:
        """When the graph node was matched to a stropha chunk, use the
        chunk_rel_path; otherwise fall back to source_file; otherwise
        the ``<rationale>`` sentinel."""
        primary = _hit("def auth(): pass", chunk_id="c1", symbol="auth")
        with patch(
            "stropha.retrieval.graph.find_rationale",
            return_value={
                "symbol": "auth",
                "rationale": [
                    {
                        "node_id": "with-chunk",
                        "label": "With chunk path",
                        "chunk_rel_path": "docs/adr/0042.md",
                        "chunk_start_line": 10,
                        "chunk_end_line": 20,
                    },
                    {
                        "node_id": "with-source-only",
                        "label": "Source only",
                        "source_file": "docs/foo.md",
                    },
                    {
                        "node_id": "no-path",
                        "label": "Bare label",
                    },
                ],
            },
        ):
            result = assemble_context(
                "task",
                search_engine=_engine([primary]),
                storage=MagicMock(),
                include_rationale=True,
                context_budget=4000,
            )
        rationale = next(s for s in result.sections if s.key == "rationale")
        paths = [h.rel_path for h in rationale.hits]
        assert "docs/adr/0042.md" in paths
        assert "docs/foo.md" in paths
        assert "<rationale>" in paths


# ---------------------------------------------------------------------------
# Rationale conversion helper (unit-level)
# ---------------------------------------------------------------------------


class TestRationaleNodeToHit:
    def test_unlabeled_node_returns_none(self) -> None:
        from stropha.retrieval.assemble import _rationale_node_to_hit

        assert _rationale_node_to_hit({"node_id": "x"}) is None
        assert _rationale_node_to_hit({}) is None

    def test_labeled_node_returns_searchhit_synth(self) -> None:
        from stropha.retrieval.assemble import _rationale_node_to_hit

        out = _rationale_node_to_hit({
            "node_id": "n1",
            "label": "Foo doc",
            "kind": "doc",
            "source_file": "docs/foo.md",
        })
        assert out is not None
        assert out.chunk_id == "rationale:n1"
        assert out.symbol == "Foo doc"
        assert out.kind == "doc"
        assert out.rel_path == "docs/foo.md"
        assert out.language == "markdown"

    def test_missing_kind_defaults_to_rationale(self) -> None:
        from stropha.retrieval.assemble import _rationale_node_to_hit

        out = _rationale_node_to_hit({
            "node_id": "n1", "label": "X",
        })
        assert out is not None
        assert out.kind == "rationale"


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
            "stropha.retrieval.graph.find_callers",
            return_value={
                "symbol": "f",
                "callers": [
                    {
                        "node_id": "primary-node",
                        "chunk_id": "c1",
                        "label": "f",
                        "source_file": "src/foo.py",
                    },
                ],
            },
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

    def test_no_graph_request_reports_not_requested(self) -> None:
        result = assemble_context(
            "task", search_engine=_engine([]), context_budget=4000,
        )
        assert result.graph_status == "not_requested"
        assert result.notes == []


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

    def test_mcp_response_schema_exposes_graph_source_status(self) -> None:
        from stropha.server import AssembleContextResponse

        properties = AssembleContextResponse.model_json_schema()["properties"]
        assert "graph_status" in properties
        assert "notes" in properties
