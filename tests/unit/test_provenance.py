"""Unit tests for ``stropha.retrieval.provenance`` (WIN-214).

Verifies the per-hit ``streams`` map renders consistently and is
visible to agents via both ``SearchResult.streams`` and the
``assemble_context`` bundle.
"""

from __future__ import annotations

from stropha.models import SearchHit
from stropha.retrieval.provenance import (
    hit_provenance_line,
    provenance_label,
)


def _hit(streams: dict | None = None) -> SearchHit:
    return SearchHit(
        chunk_id="c1",
        rel_path="src/x.py",
        start_line=1,
        end_line=10,
        score=0.5,
        rank=0,
        snippet="x",
        language="python",
        kind="function",
        symbol="f",
        streams=streams,
    )


# ---------------------------------------------------------------------------
# provenance_label
# ---------------------------------------------------------------------------


class TestProvenanceLabel:
    def test_empty_streams_returns_empty(self) -> None:
        assert provenance_label(None) == ""
        assert provenance_label({}) == ""

    def test_single_stream_no_scores(self) -> None:
        out = provenance_label({"dense": 0.0164}, show_scores=False)
        assert out == "dense"

    def test_single_stream_with_scores(self) -> None:
        out = provenance_label({"dense": 0.0164})
        assert out == "dense(rrf=0.0164)"

    def test_two_streams_joined_with_plus(self) -> None:
        out = provenance_label({"dense": 0.0164, "sparse": 0.0162})
        assert " + " in out
        assert "dense" in out
        assert "sparse" in out

    def test_canonical_order_preserved(self) -> None:
        """dense → sparse → symbol → graph-vec is the canonical order."""
        streams = {
            "symbol": 0.01, "sparse": 0.02,
            "dense": 0.03, "graph-vec": 0.04,
        }
        out = provenance_label(streams, show_scores=False)
        assert out == "dense + sparse + symbol + graph-vec"

    def test_unknown_streams_alphabetised_after_canonical(self) -> None:
        streams = {
            "rerank": 0.1, "dense": 0.05, "experimental": 0.2,
        }
        out = provenance_label(streams, show_scores=False)
        # dense first (canonical), then unknown alpha: experimental, rerank.
        assert out == "dense + experimental + rerank"

    def test_score_decimals_configurable(self) -> None:
        out = provenance_label({"dense": 0.123456789}, score_decimals=2)
        assert "dense(rrf=0.12)" == out

    def test_score_precision_default_four(self) -> None:
        out = provenance_label({"dense": 0.123456789})
        assert "dense(rrf=0.1235)" == out


# ---------------------------------------------------------------------------
# hit_provenance_line
# ---------------------------------------------------------------------------


class TestHitProvenanceLine:
    def test_returns_empty_when_no_streams(self) -> None:
        assert hit_provenance_line(_hit(None)) == ""
        assert hit_provenance_line(_hit({})) == ""

    def test_renders_markdown_quote_line(self) -> None:
        line = hit_provenance_line(_hit({"dense": 0.01, "sparse": 0.02}))
        assert line.startswith("> provenance: ")
        assert "dense" in line and "sparse" in line

    def test_compact_mode_no_scores(self) -> None:
        line = hit_provenance_line(
            _hit({"dense": 0.01}), show_scores=False,
        )
        assert line == "> provenance: dense"


# ---------------------------------------------------------------------------
# Integration: SearchResult.streams via _to_result
# ---------------------------------------------------------------------------


class TestSearchResultStreams:
    def test_streams_pass_through_to_response_model(self) -> None:
        """``_to_result`` must copy the ``streams`` dict to the
        Pydantic response model so MCP clients see the provenance."""
        from stropha.server import _to_result

        h = _hit({"dense": 0.05, "sparse": 0.04})
        out = _to_result(h)
        assert out.streams == {"dense": 0.05, "sparse": 0.04}

    def test_streams_none_pass_through(self) -> None:
        from stropha.server import _to_result

        h = _hit(None)
        out = _to_result(h)
        assert out.streams is None


# ---------------------------------------------------------------------------
# Integration: assemble_context bundle includes provenance line
# ---------------------------------------------------------------------------


class TestAssembleBundleProvenance:
    def test_bundle_includes_provenance_line_when_streams_present(
        self,
    ) -> None:
        from unittest.mock import MagicMock

        from stropha.retrieval.assemble import assemble_context

        eng = MagicMock()
        eng.search.return_value = [
            _hit({"dense": 0.0164, "sparse": 0.0162}),
        ]
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
        )
        assert "> provenance:" in result.bundle
        assert "dense" in result.bundle
        assert "sparse" in result.bundle

    def test_bundle_omits_provenance_line_when_streams_none(self) -> None:
        from unittest.mock import MagicMock

        from stropha.retrieval.assemble import assemble_context

        eng = MagicMock()
        eng.search.return_value = [_hit(None)]
        result = assemble_context(
            "task",
            search_engine=eng,
            context_budget=4000,
        )
        # No provenance line because streams is None.
        assert "> provenance:" not in result.bundle


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_imports_resolve(self) -> None:
        from stropha.retrieval.provenance import (  # noqa: F401
            hit_provenance_line,
            provenance_label,
        )
