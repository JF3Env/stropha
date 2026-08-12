"""Tests for reproducible benchmark batches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from stropha.bench.batch import (
    BatchError,
    _promote,
    _run_verification,
    evaluate_gates,
    prepare_batch,
    run_batch,
)
from stropha.bench.matrix import Cell
from stropha.cli import app


def _write_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src/example.py").write_text("def target():\n    pass\n", encoding="utf-8")
    (repo / ".strophaignore").write_text("catalog.yaml\n", encoding="utf-8")
    exposed = repo / "exposed.jsonl"
    exposed.write_text(
        '{"id":"old","query":"old query","expected_paths":["src/example.py"]}\n',
        encoding="utf-8",
    )
    catalog = repo / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "new",
                        "query": "find target implementation",
                        "expected_paths": ["src/example.py"],
                        "expected_symbols": ["target"],
                        "relevance": {"src/example.py": 3, "target": 3},
                        "tags": ["exact-lookup"],
                        "split": "holdout",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec = repo / "batch.yaml"
    (repo / "matrix.yaml").write_text(
        yaml.safe_dump(
            {
                "repo": str(repo),
                "golden": "catalog.yaml",
                "axes": {"rrf_k": [40, 60]},
                "baseline": {"rrf_k": 60},
            }
        ),
        encoding="utf-8",
    )
    spec.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "test-batch",
                "repo": str(repo),
                "catalog": "catalog.yaml",
                "exposed_golden": "exposed.jsonl",
                "matrix": "matrix.yaml",
                "cohort": {
                    "min_cases": 1,
                    "min_per_category": 0,
                    "min_multi_or_graded_ratio": 1.0,
                },
                "verification": ["diff-check"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return spec, catalog


def test_prepare_batch_is_deterministic_and_freezes_manifest(tmp_path: Path) -> None:
    spec, _ = _write_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    first = prepare_batch(spec, artifact_root=artifacts)
    second = prepare_batch(spec, artifact_root=artifacts)
    assert first.cohort_sha256 == second.cohort_sha256
    assert first.cohort_path.read_text(encoding="utf-8") == second.cohort_path.read_text(
        encoding="utf-8"
    )
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["case_count"] == 1


def test_prepare_batch_rejects_changed_frozen_catalog(tmp_path: Path) -> None:
    spec, catalog = _write_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    prepare_batch(spec, artifact_root=artifacts)
    raw = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    raw["cases"][0]["query"] = "changed query"
    catalog.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BatchError, match="already frozen"):
        prepare_batch(spec, artifact_root=artifacts)


def test_prepare_batch_rejects_exposed_query(tmp_path: Path) -> None:
    spec, catalog = _write_repo(tmp_path)
    raw = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    raw["cases"][0]["query"] = "old query"
    catalog.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BatchError, match="overlaps the exposed golden"):
        prepare_batch(spec, artifact_root=tmp_path / "artifacts")


def test_prepare_batch_rejects_indexable_answer_catalog(tmp_path: Path) -> None:
    spec, _ = _write_repo(tmp_path)
    (spec.parent / ".strophaignore").unlink()
    with pytest.raises(BatchError, match=r"catalog.*is indexable"):
        prepare_batch(spec, artifact_root=tmp_path / "artifacts")


def test_prepare_batch_rejects_changed_frozen_protocol(tmp_path: Path) -> None:
    spec, _ = _write_repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    prepare_batch(spec, artifact_root=artifacts)
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["gates"] = {"ndcg_delta_min": 0.01}
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BatchError, match="different cohort or protocol"):
        prepare_batch(spec, artifact_root=artifacts)


def _record(cell: Cell, metrics: dict, *, rank: int | None = 1) -> dict:
    return {
        "cell_id": cell.cell_id,
        "metrics": metrics,
        "per_query": {"q": rank},
        "by_tag": {
            "conceptual": {
                "success@10": metrics["success@10"],
                "recall@10": metrics["recall@10"],
                "mrr": metrics["mrr"],
                "ndcg@10": metrics["ndcg@10"],
            }
        },
    }


def test_evaluate_gates_emits_machine_readable_eligible_decision() -> None:
    baseline = Cell(values={"rrf_k": 60})
    candidate = Cell(values={"rrf_k": 40})
    base_metrics = {
        "success@10": 0.8,
        "recall@10": 0.8,
        "mrr": 0.7,
        "ndcg@10": 0.7,
        "latency_p95_ms": 100.0,
        "returned_tokens": 1000,
        "reranker_fallback_rate": 0.0,
    }
    arm_metrics = {**base_metrics, "success@10": 0.85, "recall@10": 0.85,
                   "mrr": 0.72, "ndcg@10": 0.73, "latency_p95_ms": 110.0,
                   "returned_tokens": 990}
    decision = evaluate_gates(
        [_record(baseline, base_metrics), _record(candidate, arm_metrics)],
        baseline,
        candidate,
        {},
    )
    assert decision["eligible"] is True
    assert all(check["ok"] for check in decision["checks"])


def test_evaluate_gates_rejects_case_rank_regression() -> None:
    baseline = Cell(values={"rrf_k": 60})
    candidate = Cell(values={"rrf_k": 40})
    metrics = {
        "success@10": 0.8,
        "recall@10": 0.8,
        "mrr": 0.7,
        "ndcg@10": 0.73,
        "latency_p95_ms": 100.0,
        "returned_tokens": 1000,
        "reranker_fallback_rate": 0.0,
    }
    decision = evaluate_gates(
        [_record(baseline, metrics, rank=1), _record(candidate, metrics, rank=2)],
        baseline,
        candidate,
        {"ndcg_delta_min": 0.0},
    )
    assert decision["eligible"] is False
    assert next(c for c in decision["checks"] if c["gate"] == "per_case_rank")["ok"] is False


def test_evaluate_gates_rejects_incomplete_candidate_coverage() -> None:
    baseline = Cell(values={"rrf_k": 60})
    candidate = Cell(values={"rrf_k": 40})
    metrics = {
        "success@10": 0.8,
        "recall@10": 0.8,
        "mrr": 0.7,
        "ndcg@10": 0.73,
        "latency_p95_ms": 100.0,
        "returned_tokens": 1000,
        "reranker_fallback_rate": 0.0,
    }
    base_record = _record(baseline, metrics)
    base_record["per_query"]["missing"] = 1
    arm_record = _record(candidate, metrics)
    arm_record["by_tag"] = {}

    decision = evaluate_gates(
        [base_record, arm_record],
        baseline,
        candidate,
        {"ndcg_delta_min": 0.0},
    )

    assert decision["eligible"] is False
    by_gate = {check["gate"]: check for check in decision["checks"]}
    assert by_gate["per_tag"]["missing_tags"] == ["conceptual"]
    assert by_gate["per_case_rank"]["missing_cases"] == ["missing"]


def test_prepare_only_never_runs_matrix_or_updates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _write_repo(tmp_path)
    monkeypatch.setattr(
        "stropha.bench.batch.run_matrix",
        lambda *_args, **_kwargs: pytest.fail("run_matrix must not execute"),
    )
    result = run_batch(
        spec,
        artifact_root=tmp_path / "artifacts",
        prepare_only=True,
        promote=True,
        config_path=tmp_path / "config.yaml",
    )
    assert result["status"] == "frozen"
    assert not (tmp_path / "config.yaml").exists()


def test_verification_rejects_commands_outside_allowlist(tmp_path: Path) -> None:
    with pytest.raises(BatchError, match="unknown verification step"):
        _run_verification(tmp_path, ["arbitrary-command"])


def test_promote_atomically_changes_only_selected_axis(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "secrets": {"anthropic_api_key": "keep-me"},
                "pipeline": {
                    "retrieval": {
                        "adapter": "hybrid-rrf",
                        "config": {"rrf_k": 60, "candidate_k": 20},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _promote(
        {"promotion": {"axis": "rrf_k"}},
        Cell(values={"rrf_k": 60}),
        Cell(values={"rrf_k": 40, "candidate_k": 50}),
        config_path=config,
    )

    promoted = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert promoted["pipeline"]["retrieval"]["config"] == {
        "rrf_k": 40,
        "candidate_k": 20,
    }
    assert promoted["secrets"] == {"anthropic_api_key": "keep-me"}


def test_promote_rejects_production_baseline_drift(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline:\n  retrieval:\n    config:\n      rrf_k: 80\n",
        encoding="utf-8",
    )
    with pytest.raises(BatchError, match="differs from benchmark baseline"):
        _promote(
            {"promotion": {"axis": "rrf_k"}},
            Cell(values={"rrf_k": 60}),
            Cell(values={"rrf_k": 40}),
            config_path=config,
        )


def test_promotion_requires_verification(tmp_path: Path) -> None:
    spec, _ = _write_repo(tmp_path)
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw.pop("verification")
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BatchError, match="promotion requires verification"):
        run_batch(
            spec,
            artifact_root=tmp_path / "artifacts",
            verify=False,
            promote=True,
            config_path=tmp_path / "config.yaml",
        )


def test_candidate_must_differ_only_on_promotion_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _write_repo(tmp_path)
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["baseline"] = {"rrf_k": 60, "candidate_k": 20}
    raw["candidate"] = {"rrf_k": 40, "candidate_k": 50}
    raw["promotion"] = {"axis": "rrf_k"}
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    matrix = spec.parent / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "repo": str(spec.parent),
                "golden": "catalog.yaml",
                "axes": {"rrf_k": [40, 60], "candidate_k": [20, 50]},
                "baseline": {"rrf_k": 60, "candidate_k": 20},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stropha.bench.batch.run_matrix",
        lambda *_args, **_kwargs: pytest.fail("matrix must not run"),
    )

    with pytest.raises(BatchError, match="differ from baseline only"):
        run_batch(spec, artifact_root=tmp_path / "artifacts")


def test_unverified_decision_is_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _write_repo(tmp_path)
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["baseline"] = {"rrf_k": 60}
    raw["candidate"] = {"rrf_k": 40}
    raw["promotion"] = {"axis": "rrf_k"}
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    records = [
        _record(
            Cell(values={"rrf_k": value}),
            {
                "success@10": 0.8,
                "recall@10": 0.8,
                "mrr": 0.7,
                "ndcg@10": ndcg,
                "latency_p95_ms": 100.0,
                "returned_tokens": 1000,
                "reranker_fallback_rate": 0.0,
            },
        )
        for value, ndcg in ((60, 0.7), (40, 0.73))
    ]
    monkeypatch.setattr(
        "stropha.bench.batch.run_matrix",
        lambda *_args, **_kwargs: {"evaluated": 2, "failed": 0},
    )
    monkeypatch.setattr("stropha.bench.batch.load_results", lambda _path: records)

    result = run_batch(
        spec,
        artifact_root=tmp_path / "artifacts",
        verify=False,
    )

    assert result["eligible"] is False
    assert result["verification"] == []


def test_promote_rejects_nested_reranker_policy_drift(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline:\n"
        "  retrieval:\n"
        "    config:\n"
        "      rrf_k: 60\n"
        "      reranker:\n"
        "        adapter: llm\n"
        "        config:\n"
        "          fallback: original\n",
        encoding="utf-8",
    )
    with pytest.raises(BatchError, match="reranker"):
        _promote(
            {"promotion": {"axis": "rrf_k"}},
            Cell(values={"rrf_k": 60, "reranker": "llm"}),
            Cell(values={"rrf_k": 40, "reranker": "llm"}),
            config_path=config,
        )


def test_promote_accepts_explicit_effective_reranker_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline:\n"
        "  retrieval:\n"
        "    config:\n"
        "      rrf_k: 60\n"
        "      reranker:\n"
        "        adapter: llm\n"
        "        config:\n"
        "          fallback: code-aware\n"
        "          repair_enabled: true\n",
        encoding="utf-8",
    )

    _promote(
        {"promotion": {"axis": "rrf_k"}},
        Cell(values={"rrf_k": 60, "reranker": "llm"}),
        Cell(values={"rrf_k": 40, "reranker": "llm"}),
        config_path=config,
    )

    promoted = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert promoted["pipeline"]["retrieval"]["config"]["rrf_k"] == 40


def test_failed_batch_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _write_repo(tmp_path)
    raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
    raw["baseline"] = {"rrf_k": 60}
    raw["candidate"] = {"rrf_k": 40}
    raw["promotion"] = {"axis": "rrf_k"}
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}\n", encoding="utf-8")
    original = config.read_bytes()
    monkeypatch.setattr(
        "stropha.bench.batch.load_matrix",
        lambda _path: type("MatrixStub", (), {"baseline": {"rrf_k": 60}})(),
    )
    monkeypatch.setattr(
        "stropha.bench.batch._select_cell",
        lambda _matrix, selector: Cell(values={"rrf_k": selector.get("rrf_k", 60)}),
    )
    monkeypatch.setattr("stropha.bench.batch._selected_matrix", lambda matrix, *_args: matrix)
    monkeypatch.setattr(
        "stropha.bench.batch.run_matrix",
        lambda *_args, **_kwargs: {"evaluated": 2, "failed": 0},
    )
    monkeypatch.setattr("stropha.bench.batch.load_results", lambda _path: [])

    with pytest.raises(BatchError, match="baseline or candidate result is missing"):
        run_batch(
            spec,
            artifact_root=tmp_path / "artifacts",
            promote=True,
            config_path=config,
        )

    assert config.read_bytes() == original


def test_batch_cli_reports_batch_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stropha.bench.batch.run_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(BatchError("unsafe batch")),
    )

    result = CliRunner().invoke(app, ["bench", "batch", "batch.yaml"])

    assert result.exit_code == 1
    assert "Batch failed: unsafe batch" in result.output
