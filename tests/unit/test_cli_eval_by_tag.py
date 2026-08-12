"""WIN-172 — CLI ``stropha eval`` renders NDCG/Precision/MAP by-tag.

The CLI used to print only Recall@K and MRR per tag. A regression in
ranking quality on one tag (NDCG drops while recall stays) was therefore
invisible. This test guards the new full-metric rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from stropha.cli import app
from stropha.eval import EvalCase, EvalReport, EvalResult


class _FakeHit:
    def __init__(self, rel_path: str, symbol: str | None = None) -> None:
        self.rel_path = rel_path
        self.symbol = symbol


def _golden_with_two_tags(tmp_path: Path) -> Path:
    """Write a minimal golden file with two distinct tags so by_tag has
    more than one bucket — that's the condition that triggers rendering."""
    import json as _json
    f = tmp_path / "golden.jsonl"
    f.write_text(
        _json.dumps({
            "id": "a1", "query": "qa",
            "expected_paths": ["src/a.py"],
            "tags": ["alpha"],
        }) + "\n"
        + _json.dumps({
            "id": "b1", "query": "qb",
            "expected_paths": ["src/b.py"],
            "tags": ["beta"],
        }) + "\n"
    )
    return f


def _stubbed_report(*, holdout: bool = False) -> EvalReport:
    """Build an EvalReport with known per-tag + per-split values."""
    report = EvalReport(top_k=5)
    report.results = [
        EvalResult(
            case=EvalCase(id="a1", query="qa", tags=("alpha",),
                          expected_paths=("src/a.py",), split="search"),
            rank_of_first_hit=1,
            top_k=5,
            relevances=(1.0, 0.0, 0.0, 0.0, 0.0),
            relevant_count=1,
            latency_ms=12.5,
        ),
        EvalResult(
            case=EvalCase(id="b1", query="qb", tags=("beta",),
                          expected_paths=("src/b.py",),
                          split="holdout" if holdout else "search"),
            rank_of_first_hit=2,
            top_k=5,
            relevances=(0.0, 1.0, 0.0, 0.0, 0.0),
            relevant_count=1,
            latency_ms=8.7,
        ),
    ]
    return report


def _install_cli_stubs(monkeypatch: pytest.MonkeyPatch, report: EvalReport) -> None:
    """Stub the heavy CLI bits so the eval command renders against ``report``."""

    class _StorageStub:
        def __enter__(self) -> _StorageStub: return self
        def __exit__(self, *a: object) -> None: pass
        def close(self) -> None: pass

    class _BuiltStub:
        storage = _StorageStub()
        retrieval = object()
        embedder = object()

        @staticmethod
        def effective_enricher_id() -> str:
            return "noop"

    # run_eval is imported inside the function via `from .eval import run_eval`
    # — monkeypatch at the source module so the lazy import picks up the stub.
    monkeypatch.setattr("stropha.eval.run_eval", lambda *_a, **_kw: report)
    monkeypatch.setattr("stropha.cli.build_stages", lambda *_a, **_kw: _BuiltStub())


class TestCliEvalByTagRendering:
    def test_by_tag_table_renders_full_metric_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden = _golden_with_two_tags(tmp_path)
        _install_cli_stubs(monkeypatch, _stubbed_report())

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["eval", "--golden", str(golden), "--top-k", "5"],
        )
        # New columns from WIN-172.
        assert "NDCG@5" in result.output
        assert "Prec@5" in result.output
        assert "MAP@5" in result.output
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_summary_line_includes_new_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden = _golden_with_two_tags(tmp_path)
        _install_cli_stubs(monkeypatch, _stubbed_report())

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--golden", str(golden), "--top-k", "5"])
        assert "latency_p95" in result.output
        for needle in ("ndcg@5=", "precision@5=", "map@5=", "mrr="):
            assert needle in result.output

    def test_by_split_table_renders_when_more_than_one_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        golden = _golden_with_two_tags(tmp_path)
        _install_cli_stubs(monkeypatch, _stubbed_report(holdout=True))

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--golden", str(golden), "--top-k", "5"])
        assert "By split" in result.output
        assert "search" in result.output
        assert "holdout" in result.output

    def test_by_split_table_skipped_when_single_split(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every result is in the same split, the by-split table is
        omitted — no extra noise when the operator is only running on
        the search subset."""
        golden = _golden_with_two_tags(tmp_path)
        _install_cli_stubs(monkeypatch, _stubbed_report())

        runner = CliRunner()
        result = runner.invoke(app, ["eval", "--golden", str(golden), "--top-k", "5"])
        assert "By split" not in result.output
