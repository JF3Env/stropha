"""Unit tests for the config-matrix benchmark (pure logic — no indexing)."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest

from stropha.bench.matrix import Cell, Matrix, MatrixError, load_matrix
from stropha.bench.report import (
    BenchReport,
    ReportError,
    load_results,
    to_csv,
    validate_matrix_id,
)
from stropha.bench.runner import (
    BenchError,
    _experiment_provenance,
    _merge,
    _retrieval_overrides,
    _run_provenance,
    _source_worktree_state,
    _validate_output_identity,
    eval_cell,
    run_matrix,
)
from stropha.config import (
    Config,
    ExperimentBlacklistConfig,
    ExperimentPostgresConfig,
    ExperimentsConfig,
)
from stropha.evaluation.postgres import DatabaseRef

# --------------------------------------------------------------------------- matrix

_SPEC = """
repo: .
golden: g.jsonl
top_k: 5
axes:
  embedder:
    - {adapter: voyage, dim: 512}
    - {adapter: mlx}
  enricher: [noop, hierarchical]
  reranker: [null, code-aware]
  hyde: [false, true]
baseline:
  embedder: {adapter: voyage, dim: 512}
  enricher: noop
  reranker: null
  hyde: false
constraints:
  - skip_when: {embedder: {adapter: mlx}, enricher: hierarchical}
"""


@pytest.fixture
def matrix(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(_SPEC, encoding="utf-8")
    return load_matrix(p)


def test_load_parses_fields(matrix) -> None:
    assert matrix.top_k == 5
    assert matrix.golden == "g.jsonl"
    assert set(matrix.axes) == {"embedder", "enricher", "reranker", "hyde"}


def test_matrix_id_is_stable_and_changes_with_semantics(matrix) -> None:
    same = type(matrix)(
        axes=dict(reversed(list(matrix.axes.items()))),
        golden=matrix.golden,
        top_k=matrix.top_k,
        repo=matrix.repo,
        baseline=matrix.baseline,
        constraints=matrix.constraints,
    )
    changed = type(matrix)(
        axes=matrix.axes,
        golden=matrix.golden,
        top_k=10,
        repo=matrix.repo,
        baseline=matrix.baseline,
        constraints=matrix.constraints,
    )
    assert same.matrix_id == matrix.matrix_id
    assert changed.matrix_id != matrix.matrix_id


def test_matrix_id_and_axis_selection_preserve_pipeline() -> None:
    first = Matrix(
        axes={"candidate_k": [20, 50]},
        pipeline={"retrieval": {"adapter": "hybrid-rrf", "config": {"rrf_k": 60}}},
    )
    second = Matrix(
        axes=first.axes,
        pipeline={"retrieval": {"adapter": "hybrid-rrf", "config": {"rrf_k": 61}}},
    )

    assert first.matrix_id != second.matrix_id
    assert first.select_axes(["candidate_k"]).pipeline == first.pipeline


def test_matrix_without_pipeline_keeps_legacy_identity() -> None:
    matrix = Matrix(axes={"candidate_k": [20]}, golden="golden.jsonl", top_k=10)
    legacy_fields = {
        "axes": matrix.axes,
        "baseline": matrix.baseline,
        "constraints": matrix.constraints,
        "golden": matrix.golden,
        "repo": matrix.repo,
        "top_k": matrix.top_k,
    }
    expected = hashlib.sha256(
        json.dumps(legacy_fields, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:12]

    assert matrix.matrix_id == expected


def test_expand_product_minus_constraints(matrix) -> None:
    cells = matrix.expand()
    # 2*2*2*2 = 16, minus mlx+hierarchical (reranker2 * hyde2 = 4) = 12.
    assert len(cells) == 12
    # No cell violates the skip constraint.
    for c in cells:
        assert not (
            c.values["embedder"] == {"adapter": "mlx"} and c.values["enricher"] == "hierarchical"
        )


def test_cell_ids_unique_and_stable(matrix) -> None:
    cells = matrix.expand()
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    # Stable: same values → same id regardless of dict order.
    a = Cell(values={"hyde": False, "reranker": None})
    b = Cell(values={"reranker": None, "hyde": False})
    assert a.cell_id == b.cell_id


def test_groups_by_indexing_signature(matrix) -> None:
    groups = matrix.groups()
    # indexing axes = embedder x enricher, minus mlx+hierarchical = 3 indexes.
    assert len(groups) == 3
    # Each index is swept by reranker(2) x hyde(2) = 4 query variants.
    for sig, cells in groups.items():
        assert len(cells) == 4
        assert len({c.indexing_sig for c in cells}) == 1
        assert all(c.indexing_sig == sig for c in cells)
        # Within a group, only query axes vary.
        assert len({(str(c.values["embedder"]), c.values["enricher"]) for c in cells}) == 1


def test_query_only_change_shares_index(matrix) -> None:
    a = Cell(
        values={"embedder": {"adapter": "mlx"}, "enricher": "noop", "reranker": None, "hyde": False}
    )
    b = Cell(
        values={
            "embedder": {"adapter": "mlx"},
            "enricher": "noop",
            "reranker": "code-aware",
            "hyde": True,
        }
    )
    assert a.indexing_sig == b.indexing_sig  # only query axes differ
    assert a.cell_id != b.cell_id


def test_baseline_cell_resolves(matrix) -> None:
    base = matrix.baseline_cell()
    assert base is not None
    assert base.values["enricher"] == "noop"
    assert base.values["reranker"] is None


def test_select_axes_pins_other_values_to_baseline(matrix) -> None:
    staged = matrix.select_axes(["reranker"])
    cells = staged.expand()
    assert len(cells) == 2
    assert {cell.values["reranker"] for cell in cells} == {None, "code-aware"}
    assert {cell.values["hyde"] for cell in cells} == {False}
    assert {str(cell.values["embedder"]) for cell in cells} == {
        str({"adapter": "voyage", "dim": 512})
    }
    assert {cell.values["enricher"] for cell in cells} == {"noop"}


def test_select_axes_rejects_unknown_or_unpinned_axes(matrix) -> None:
    with pytest.raises(MatrixError, match="selected axes not in matrix"):
        matrix.select_axes(["missing"])

    matrix.baseline = {"reranker": None}
    with pytest.raises(MatrixError, match="must be selected or declared in baseline"):
        matrix.select_axes(["reranker"])


def test_retrieval_axes_map_to_runtime_config_fields() -> None:
    overrides = _retrieval_overrides(
        Cell(
            values={
                "rrf_k": 80,
                "candidate_k": 40,
                "recursive": True,
                "recursive_adjacency": 9,
                "adaptive_weighting": False,
                "learned_weights": True,
                "query_rewrite": True,
                "query_rewrite_max_terms": 6,
                "multi_query": True,
                "multi_query_count": 2,
                "multi_query_min_terms": 7,
            }
        )
    )
    config = overrides["retrieval"]["config"]
    assert config == {
        "rrf_k": 80,
        "candidate_k": 40,
        "recursive_enabled": True,
        "recursive_adjacency": 9,
        "adaptive_weighting": False,
        "learned_weights_enabled": True,
        "query_rewrite_enabled": True,
        "query_rewrite_max_terms": 6,
        "multi_query_enabled": True,
        "multi_query_count": 2,
        "multi_query_min_terms": 7,
    }


def test_merge_drops_stale_config_when_adapter_changes() -> None:
    resolved = {
        "embedder": {
            "adapter": "mlx",
            "config": {"model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"},
        }
    }

    _merge(resolved, {"embedder": {"adapter": "model2vec"}})

    assert resolved == {"embedder": {"adapter": "model2vec"}}


def test_query_rewrite_matrix_is_two_isolated_search_arms() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/query_rewrite_matrix.yaml"
    matrix = load_matrix(path)

    assert [cell.values["query_rewrite"] for cell in matrix.expand()] == [False, True]
    assert matrix.baseline_cell() is not None
    assert matrix.baseline_cell().values["query_rewrite"] is False


def test_multi_query_matrix_is_two_selective_candidate_arms() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/multi_query_matrix.yaml"
    matrix = load_matrix(path)
    cells = matrix.expand()

    assert [(cell.values["multi_query"], cell.values["multi_query_count"]) for cell in cells] == [
        (False, 2),
        (True, 2),
        (True, 3),
    ]
    assert {cell.values["multi_query_min_terms"] for cell in cells} == {7}
    assert matrix.baseline_cell() == cells[0]


def test_hyde_matrix_is_one_selective_candidate_arm() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/hyde_matrix.yaml"
    matrix = load_matrix(path)
    cells = matrix.expand()

    assert [cell.values["hyde"] for cell in cells] == [False, True]
    assert {cell.values["hyde_code_tuned"] for cell in cells} == {True}
    assert matrix.baseline_cell() == cells[0]


def test_diversity_matrix_compares_fixed_budget_arms() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/diversity_matrix.yaml"
    matrix = load_matrix(path)
    cells = matrix.expand()

    assert len(cells) == 3
    assert {cell.values["context_budget"] for cell in cells} == {200}
    assert {cell.values["diversity"] for cell in cells} == {False, True}
    assert {cell.values["diversity_max_chunks_per_file"] for cell in cells} == {1, 2}
    assert all(
        cell.values["diversity"] or cell.values["diversity_max_chunks_per_file"] == 2
        for cell in cells
    )
    assert {cell.indexing_sig for cell in cells} == {cells[0].indexing_sig}
    assert matrix.baseline_cell() is not None
    assert matrix.baseline_cell().values["context_budget"] == 200


def test_active_matrix_parses_and_has_declared_baseline() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/active_matrix.yaml"
    active = load_matrix(path)
    assert active.baseline_cell() is not None
    assert active.baseline_cell().values["reranker"] == "llm"


def test_active_matrix_pins_deployed_contextual_identity() -> None:
    from stropha.adapters.enricher.contextual import ContextualEnricherConfig

    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/active_matrix.yaml"
    spec = load_matrix(path).axes["enricher"]
    assert len(spec) == 1
    config_values = dict(spec[0])
    assert config_values.pop("adapter") == "contextual"
    config = ContextualEnricherConfig(**config_values)
    prompt_hash = hashlib.sha256(config.prompt_template.encode()).hexdigest()[:8]
    config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:8]
    assert (prompt_hash, config_hash) == ("93f8f282", "c5ab6279")


def test_phase4_quality_candidates_freeze_exact_three_cells() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = load_matrix(root / "src/stropha/bench/phase4_quality_candidates_matrix.yaml")
    cells = matrix.expand()

    assert len(cells) == 3
    assert matrix.baseline_cell() is not None
    values = [cell.values for cell in cells]
    assert any(
        value["enricher"] == "hierarchical" and value["rrf_k"] == 60
        for value in values
    )
    assert any(
        isinstance(value["enricher"], dict) and value["rrf_k"] == 40 for value in values
    )
    assert all(value["learned_weights"] is False for value in values)


def test_phase2_shortlist_is_frozen_to_four_cells() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/phase2_shortlist_matrix.yaml"
    shortlist = load_matrix(path)
    cells = shortlist.expand()

    assert len(cells) == 4
    assert {(cell.values["adaptive_weighting"], cell.values["rrf_k"]) for cell in cells} == {
        (True, 60),
        (True, 40),
        (False, 60),
        (False, 80),
    }
    assert shortlist.baseline_cell() is not None
    active_path = Path(__file__).resolve().parents[2] / "src/stropha/bench/active_matrix.yaml"
    assert set(shortlist.groups()) == set(load_matrix(active_path).groups())


def test_local_quality_matrix_covers_seven_indexes_and_112_cells() -> None:
    path = Path(__file__).resolve().parents[2] / "src/stropha/bench/local_quality_matrix.yaml"
    matrix = load_matrix(path)

    assert len(matrix.groups()) == 7
    assert len(matrix.expand()) == 112
    assert matrix.baseline_cell() is not None
    assert matrix.baseline_cell().values["reranker"] == "llm"


def test_empty_axes_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("axes: {}\n", encoding="utf-8")
    with pytest.raises(MatrixError):
        load_matrix(p)


def test_all_excluded_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(
        "axes:\n  enricher: [noop]\nconstraints:\n  - skip_when: {enricher: noop}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError):
        load_matrix(p).expand()


# --------------------------------------------------------------------------- report


def _results() -> list[dict]:
    provenance = {"run_id": "run-one", "golden_sha256": "golden-one"}
    return [
        {
            "ok": True,
            "cell_id": "a",
            "indexing_sig": "i1",
            "label": "reranker=None",
            "values": {"reranker": None, "hyde": False},
            "metrics": {"recall@10": 0.8, "mrr": 0.5},
            "provenance": provenance,
            "per_query": {"q1": 1, "q2": None, "q3": 2},
            "per_query_metrics": {
                "q1": {"ndcg@10": 1.0},
                "q2": {"ndcg@10": 0.0},
                "q3": {"ndcg@10": 0.5},
            },
        },
        {
            "ok": True,
            "cell_id": "b",
            "indexing_sig": "i1",
            "label": "reranker=code-aware",
            "values": {"reranker": "code-aware", "hyde": False},
            "metrics": {"recall@10": 0.7, "mrr": 0.55},
            "provenance": provenance,
            "per_query": {"q1": 1, "q2": 3, "q3": None},
            "per_query_metrics": {
                "q1": {"ndcg@10": 1.0},
                "q2": {"ndcg@10": 0.25},
                "q3": {"ndcg@10": 0.0},
            },
        },
    ]


def test_load_results_filters_failures(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    lines = [json.dumps(r) for r in _results()]
    lines.append(
        json.dumps(
            {
                "ok": False,
                "cell_id": "c",
                "provenance": {"run_id": "run-one"},
                "error": "boom",
            }
        )
    )
    lines.append("")  # blank line tolerated
    p.write_text("\n".join(lines), encoding="utf-8")
    out = load_results(p)
    assert [r["cell_id"] for r in out] == ["a", "b"]


def test_load_results_rejects_mixed_or_missing_run_identity(tmp_path: Path) -> None:
    p = tmp_path / "mixed.jsonl"
    records = _results()
    records[1]["provenance"] = {"run_id": "run-two"}
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ReportError, match="incomparable results"):
        load_results(p)

    records[1].pop("provenance")
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ReportError, match=r"lacks provenance\.run_id"):
        load_results(p)


def test_resume_rejects_incomparable_output(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    p.write_text(
        json.dumps(
            {
                "ok": True,
                "cell_id": "a",
                "provenance": {"run_id": "old"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchError, match="use a new --out path"):
        _validate_output_identity(p, "new")


def test_resume_removes_only_a_torn_final_record(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    valid = json.dumps(
        {"ok": True, "cell_id": "a", "provenance": {"run_id": "run-one"}}
    )
    p.write_text(valid + "\n{\"ok\":true", encoding="utf-8")

    assert _validate_output_identity(p, "run-one") == {"a"}
    assert p.read_text(encoding="utf-8") == valid + "\n"


def _experiment_config(
    *,
    storage: Literal["sqlite-vec", "postgres"] = "sqlite-vec",
    blacklist: dict[str, tuple[str, ...]] | None = None,
) -> Config:
    return Config(
        experiments=ExperimentsConfig(
            storage=storage,
            postgres=ExperimentPostgresConfig(
                admin_dsn="postgresql://eval_user:top-secret@db.internal:5544/eval_admin"
            ),
            blacklist=ExperimentBlacklistConfig(adapters=blacklist or {}),
        )
    )


def test_run_matrix_filters_policy_before_dry_run_totals_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = Matrix(
        axes={"enricher": ["noop", "hierarchical"], "hyde": [False, True]},
        repo=str(tmp_path),
        golden=str(tmp_path / "golden.jsonl"),
    )
    original = copy.deepcopy(source)
    monkeypatch.setattr(
        "stropha.bench.runner.get_config",
        lambda: _experiment_config(blacklist={"enricher": ("hierarchical",)}),
    )
    monkeypatch.setattr(
        "stropha.bench.runner._run_provenance",
        lambda _matrix, _split, **_kwargs: {
            "run_id": "run-one",
            "source_diff_sha256": None,
        },
    )
    monkeypatch.setattr("stropha.bench.runner._resolved_indexing_hash", lambda _cell: "index-hash")
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")
    monkeypatch.setattr("stropha.bench.runner._assert_run_environment", lambda *_args: None)

    plan = run_matrix(source, tmp_path / "results.jsonl", dry_run=True)

    assert plan["total_cells"] == 2
    assert all(
        "hierarchical" not in label for labels in plan["groups"].values() for label in labels
    )
    assert plan["storage"] == "sqlite-vec"
    assert plan["policy_fingerprint"]
    assert source == original


def test_no_resume_rejects_existing_output(
    matrix, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "results.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr("stropha.bench.runner.get_config", lambda: _experiment_config())
    with pytest.raises(BenchError, match="requires a new --out path"):
        run_matrix(matrix, output, resume=False)


def test_report_rejects_matrix_identity_mismatch() -> None:
    records = [{"provenance": {"matrix_id": "recorded"}}]
    validate_matrix_id(records, "recorded")
    with pytest.raises(ReportError, match="matrix does not match"):
        validate_matrix_id(records, "different")


def test_benchmark_split_is_part_of_run_identity(
    matrix, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden = tmp_path / "g.jsonl"
    golden.write_text("{}\n", encoding="utf-8")
    matrix.golden = str(golden)
    matrix.repo = str(tmp_path)
    monkeypatch.setattr(
        "stropha.ingest.git_meta.detect",
        lambda _repo: SimpleNamespace(head_commit="abc", normalized_key="repo"),
    )
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")

    search = _run_provenance(matrix, "search")
    holdout = _run_provenance(matrix, "holdout")

    assert search["split"] == "search"
    assert holdout["split"] == "holdout"
    assert search["source_state_id"] == holdout["source_state_id"]
    assert search["run_id"] != holdout["run_id"]


def test_source_state_includes_untracked_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    untracked = tmp_path / "new.py"
    untracked.write_text("first\n", encoding="utf-8")
    runs = [
        SimpleNamespace(returncode=0, stdout=b"tracked diff"),
        SimpleNamespace(returncode=0, stdout=b"new.py\0"),
    ]
    monkeypatch.setattr("stropha.bench.runner.subprocess.run", lambda *_a, **_kw: runs.pop(0))

    first = _source_worktree_state(tmp_path)
    untracked.write_text("second\n", encoding="utf-8")
    runs.extend(
        [
            SimpleNamespace(returncode=0, stdout=b"tracked diff"),
            SimpleNamespace(returncode=0, stdout=b"new.py\0"),
        ]
    )

    assert first != _source_worktree_state(tmp_path)


def test_source_state_fails_closed_on_git_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "stropha.bench.runner.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout=b""),
    )
    with pytest.raises(BenchError, match="cannot fingerprint benchmark source"):
        _source_worktree_state(tmp_path)


def test_runtime_identity_changes_with_backend(
    matrix, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    golden = tmp_path / "g.jsonl"
    golden.write_text("{}\n", encoding="utf-8")
    matrix.golden = str(golden)
    matrix.repo = str(tmp_path)
    monkeypatch.setattr(
        "stropha.ingest.git_meta.detect",
        lambda _repo: SimpleNamespace(head_commit="abc", normalized_key="repo"),
    )
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")

    first = _run_provenance(matrix, "search", runtime_id="runtime-one")
    second = _run_provenance(matrix, "search", runtime_id="runtime-two")

    assert first["run_id"] != second["run_id"]


def test_run_matrix_filters_cases_to_selected_split(
    matrix, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = [
        SimpleNamespace(id="search-case", split="search"),
        SimpleNamespace(id="holdout-case", split="holdout"),
    ]
    seen: list[str] = []
    monkeypatch.setattr("stropha.bench.runner.get_config", lambda: _experiment_config())
    monkeypatch.setattr("stropha.eval.load_golden", lambda _path: cases)
    monkeypatch.setattr(
        "stropha.bench.runner._run_provenance",
        lambda _matrix, split, **_kwargs: {
            "run_id": split,
            "split": split,
            "source_state_id": "source-one",
            "source_diff_sha256": None,
            "source_commit": "abc",
            "runtime_id": "runtime-one",
            "index_runtime_id": "index-runtime-one",
        },
    )
    monkeypatch.setattr("stropha.bench.runner._resolved_indexing_hash", lambda _cell: "index-hash")
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")
    monkeypatch.setattr("stropha.bench.runner._assert_run_environment", lambda *_args: None)
    index_paths: list[Path] = []
    monkeypatch.setattr(
        "stropha.bench.runner.build_index",
        lambda _matrix, _cell, path: index_paths.append(path),
    )

    def fake_eval_cell(_matrix, _cell, _db_path, selected, **_kwargs):
        seen.extend(case.id for case in selected)
        return {"ok": True, "cell_id": _cell.cell_id}

    monkeypatch.setattr("stropha.bench.runner.eval_cell", fake_eval_cell)

    summary = run_matrix(
        matrix,
        tmp_path / "results.jsonl",
        index_dir=tmp_path / "indexes",
        limit=1,
        split="search",
    )

    assert summary["evaluated"] == 1
    assert seen == ["search-case"]
    first_sig = next(iter(matrix.groups()))
    assert index_paths == [
        tmp_path / f"indexes/index_{first_sig}_source-one_index-runtime-one_index-hash.db"
    ]


def test_postgres_dry_run_reports_storage_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = Matrix(
        axes={"enricher": ["noop"]},
        repo=str(tmp_path),
        golden=str(tmp_path / "golden.jsonl"),
    )
    manager_inits: list[dict[str, object]] = []

    class Manager:
        def __init__(self, **kwargs: object) -> None:
            manager_inits.append(kwargs)

        @staticmethod
        def preflight() -> None:
            pytest.fail("dry-run must not connect to PostgreSQL")

    monkeypatch.setattr(
        "stropha.bench.runner.get_config", lambda: _experiment_config(storage="postgres")
    )
    monkeypatch.setattr(
        "stropha.evaluation.postgres.PostgresEvaluationDatabaseManager",
        Manager,
    )
    monkeypatch.setattr("stropha.bench.runner._production_database_for_admin", lambda _dsn: None)
    monkeypatch.setattr(
        "stropha.bench.runner._run_provenance",
        lambda _matrix, _split, **_kwargs: {
            "run_id": "run-one",
            "source_diff_sha256": None,
            "experiment": {"postgres_cluster": {"host": "db.internal", "port": 5544}},
        },
    )
    monkeypatch.setattr("stropha.bench.runner._resolved_indexing_hash", lambda _cell: "index-hash")
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")
    monkeypatch.setattr("stropha.bench.runner._assert_run_environment", lambda *_args: None)
    ignored_index_dir = tmp_path / "must-not-exist"

    plan = run_matrix(
        source,
        tmp_path / "results.jsonl",
        index_dir=ignored_index_dir,
        dry_run=True,
    )

    assert len(manager_inits) == 1
    assert plan["storage"] == "postgres"
    assert plan["postgres_cluster"] == {"host": "db.internal", "port": 5544}
    assert plan["index_dir"].startswith("ignored")
    assert not ignored_index_dir.exists()


def test_postgres_run_reuses_cache_and_clones_each_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = Matrix(
        axes={"enricher": ["noop"], "hyde": [False, True]},
        repo=str(tmp_path),
        golden=str(tmp_path / "golden.jsonl"),
    )
    events: list[str] = []
    cache_keys: list[str] = []
    manager_kwargs: list[dict[str, object]] = []
    build_dsns: list[str] = []
    eval_dsns: list[str] = []

    class Manager:
        def __init__(self, **kwargs: object) -> None:
            manager_kwargs.append(kwargs)

        @staticmethod
        def preflight() -> SimpleNamespace:
            events.append("preflight")
            return SimpleNamespace(
                host="db.internal",
                port=5544,
                database="eval_admin",
                user="eval_user",
                server_version="17.2",
                template_database="eval_template",
                template_extensions=("plpgsql=1.0", "vector=0.8.1"),
            )

        @staticmethod
        def get_or_create_cache(cache_key: str, builder) -> DatabaseRef:
            cache_keys.append(cache_key)
            builder(
                DatabaseRef(
                    name="eval_staging_0123456789abcdef",
                    dsn="postgresql://eval_user:secret@db.internal:5544/eval_staging",
                )
            )
            return DatabaseRef(
                name="eval_cache_0123456789abcdef01234567",
                dsn="postgresql://eval_user:secret@db.internal:5544/eval_cache",
            )

        @staticmethod
        @contextmanager
        def writable_clone(_cache: DatabaseRef):
            clone_number = len(eval_dsns) + 1
            events.append(f"clone-{clone_number}")
            yield DatabaseRef(
                name=f"eval_work_{clone_number:016x}",
                dsn=f"postgresql://eval_user:secret@db.internal:5544/eval_work_{clone_number}",
            )
            events.append(f"cleanup-{clone_number}")

    config = _experiment_config(storage="postgres")
    monkeypatch.setattr("stropha.bench.runner.get_config", lambda: config)
    monkeypatch.setattr(
        "stropha.evaluation.postgres.PostgresEvaluationDatabaseManager",
        Manager,
    )
    monkeypatch.setattr(
        "stropha.pipeline.config.load_pipeline_config",
        lambda: {
            "storage": {
                "adapter": "postgres",
                "config": {"dsn": "postgresql://prod:password@db.internal:5544/stropha_production"},
            }
        },
    )
    monkeypatch.setattr(
        "stropha.bench.runner._run_provenance",
        lambda _matrix, _split, **_kwargs: {
            "run_id": "run-one",
            "source_diff_sha256": None,
            "source_commit": "abc",
            "source_state_id": "source-one",
            "runtime_id": "runtime-one",
            "index_runtime_id": "index-runtime-one",
            "repo_key": "repo-one",
            "experiment": {"policy_fingerprint": "policy-one"},
            "indexing_identities": {source.expand()[0].indexing_sig: "index-hash"},
        },
    )
    monkeypatch.setattr("stropha.bench.runner._resolved_indexing_hash", lambda _cell: "index-hash")
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")
    monkeypatch.setattr("stropha.bench.runner._assert_run_environment", lambda *_args: None)

    def load_golden(_path: Path) -> list[SimpleNamespace]:
        events.append("golden")
        return [SimpleNamespace(id="search-case", split="search")]

    monkeypatch.setattr("stropha.eval.load_golden", load_golden)

    def fake_build_index(
        _matrix: Matrix,
        _cell: Cell,
        db_path: Path | None = None,
        *,
        storage_dsn: str | None = None,
    ) -> dict[str, object]:
        assert db_path is None
        assert storage_dsn is not None
        build_dsns.append(storage_dsn)
        return {"index_seconds": 12.3, "index_reused": False}

    def fake_eval_cell(
        _matrix: Matrix,
        cell: Cell,
        db_path: Path | None,
        _cases: object,
        *,
        storage_dsn: str | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert db_path is None
        assert storage_dsn is not None
        eval_dsns.append(storage_dsn)
        events.append(f"storage-closed-{len(eval_dsns)}")
        return {"ok": True, "cell_id": cell.cell_id}

    monkeypatch.setattr("stropha.bench.runner.build_index", fake_build_index)
    monkeypatch.setattr("stropha.bench.runner.eval_cell", fake_eval_cell)
    monkeypatch.setattr(
        "stropha.bench.runner._verify_index",
        lambda _path: pytest.fail("PostgreSQL run must not use SQLite verification"),
    )

    summary = run_matrix(source, tmp_path / "results.jsonl")

    assert summary["evaluated"] == 2
    assert len(cache_keys) == 1
    assert json.loads(cache_keys[0]) == {
        "experiment": {"policy_fingerprint": "policy-one"},
        "index_runtime_id": "index-runtime-one",
        "indexing_hash": "index-hash",
        "indexing_sig": source.expand()[0].indexing_sig,
        "repo_key": "repo-one",
        "source_state_id": "source-one",
        "version": 2,
    }
    assert len(build_dsns) == 1
    assert len(eval_dsns) == 2
    assert events.index("preflight") < events.index("golden")
    assert events.index("storage-closed-1") < events.index("cleanup-1")
    assert events.index("storage-closed-2") < events.index("cleanup-2")
    assert manager_kwargs[0]["production_database"] == "stropha_production"


def test_build_index_postgres_uses_storage_protocol_not_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from stropha.bench.runner import build_index

    class Storage:
        adapter_id = "postgres:test"

        def __init__(self) -> None:
            self.meta = {
                "active_enricher_id": "noop",
                "active_embedder_id": "embedder:test",
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def set_meta(self, key: str, value: str) -> None:
            self.meta[key] = value

        def get_meta(self, key: str) -> str | None:
            return self.meta.get(key)

        @staticmethod
        def commit() -> None:
            return None

        @staticmethod
        def stats() -> dict[str, object]:
            return {
                "chunks": 2,
                "models": [{"embedding_model": "test", "n": 2}],
                "enrichers": [{"enricher_id": "noop", "n": 2}],
            }

    storage = Storage()
    built = SimpleNamespace(
        storage=storage,
        embedder=SimpleNamespace(adapter_id="embedder:test"),
        secondary_embedder=None,
        enricher=object(),
        walker=object(),
        chunker=object(),
    )
    resolved: list[dict[str, object]] = []

    def pipeline_config(overrides: dict[str, object], *, resolve_storage_path: bool = True):
        assert resolve_storage_path is False
        resolved.append(overrides)
        return overrides

    class Pipeline:
        def __init__(self, **_kwargs: object) -> None:
            return None

        @staticmethod
        def run(*, rebuild: bool, mode: str) -> SimpleNamespace:
            assert rebuild is True
            assert mode == "full"
            return SimpleNamespace(chunks_enriched_failed=0)

    monkeypatch.setattr("stropha.bench.runner._patch_graphify_off", lambda: None)
    monkeypatch.setattr(
        "stropha.pipeline.config.load_pipeline_config",
        lambda: {
            "walker": {"adapter": "git-ls-files"},
            "chunker": {"adapter": "tree-sitter-dispatch"},
            "enricher": {"adapter": "hierarchical"},
            "embedder": {"adapter": "local"},
            "storage": {"adapter": "sqlite-vec"},
            "retrieval": {"adapter": "hybrid-rrf"},
        },
    )
    monkeypatch.setattr("stropha.pipeline.config.pipeline_config_from_dict", pipeline_config)
    monkeypatch.setattr(
        "stropha.pipeline.builder.build_stages",
        lambda _cfg, *, force_rebuild: built,
    )
    monkeypatch.setattr("stropha.pipeline.pipeline.Pipeline", Pipeline)
    monkeypatch.setattr(
        "stropha.bench.runner._verify_index",
        lambda _path: pytest.fail("PostgreSQL build must not use SQLite verification"),
    )
    monkeypatch.setattr("stropha.bench.runner.get_config", lambda: Config())
    dsn = "postgresql://eval_user:secret@db.internal:5544/eval_staging"

    result = build_index(
        Matrix(axes={"enricher": ["noop"]}, repo=str(tmp_path)),
        Cell(values={"enricher": "noop"}),
        storage_dsn=dsn,
    )

    assert result["complete"] is True
    assert result["chunks"] == result["vecs"] == 2
    assert result["index_reused"] is False
    assert result["index_seconds"] >= 0
    assert resolved[0]["storage"] == {"adapter": "postgres", "config": {"dsn": dsn}}


def test_query_only_benchmark_inherits_active_indexing_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stropha.bench.runner import _pipeline_config_for_bench

    active = {
        "walker": {"adapter": "git-ls-files"},
        "chunker": {"adapter": "tree-sitter-dispatch"},
        "enricher": {"adapter": "contextual", "config": {"max_tokens": 64}},
        "embedder": {"adapter": "mlx", "config": {"model": "qwen"}},
        "storage": {"adapter": "postgres", "config": {"dsn": "production"}},
        "retrieval": {"adapter": "hybrid-rrf", "config": {"candidate_k": 20}},
    }
    monkeypatch.setattr(
        "stropha.pipeline.config.load_pipeline_config", lambda: copy.deepcopy(active)
    )

    resolved = _pipeline_config_for_bench(
        {
            "storage": {"adapter": "postgres", "config": {"dsn": "evaluation"}},
            "retrieval": {"config": {"query_rewrite_enabled": True}},
        },
        resolve_storage_path=False,
    )

    assert resolved["embedder"] == active["embedder"]
    assert resolved["enricher"] == active["enricher"]
    assert resolved["storage"]["config"]["dsn"] == "evaluation"
    assert resolved["retrieval"]["config"]["candidate_k"] == 20
    assert resolved["retrieval"]["config"]["query_rewrite_enabled"] is True


def test_checked_in_pipeline_isolated_from_active_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stropha.bench.runner import _pipeline_config_for_bench

    monkeypatch.setattr(
        "stropha.pipeline.config.load_pipeline_config",
        lambda: pytest.fail("hermetic matrix must not load the active pipeline"),
    )
    resolved = _pipeline_config_for_bench(
        {"retrieval": {"config": {"candidate_k": 20}}},
        resolve_storage_path=False,
        base_pipeline={
            "embedder": {"adapter": "mlx", "config": {"model": "qwen"}},
            "enricher": {"adapter": "hierarchical"},
            "retrieval": {
                "adapter": "hybrid-rrf",
                "config": {"candidate_k": 99, "inherit_global_features": False},
            },
        },
    )

    assert resolved["embedder"]["config"]["model"] == "qwen"
    assert resolved["enricher"]["adapter"] == "hierarchical"
    assert resolved["retrieval"]["config"]["candidate_k"] == 20
    assert resolved["retrieval"]["config"]["inherit_global_features"] is False


def test_postgres_provenance_is_secret_free_and_changes_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text("{}\n", encoding="utf-8")
    source = Matrix(
        axes={"enricher": ["noop"]},
        repo=str(tmp_path),
        golden=str(golden),
    )
    monkeypatch.setattr(
        "stropha.bench.runner.get_config", lambda: _experiment_config(storage="postgres")
    )
    monkeypatch.setattr(
        "stropha.ingest.git_meta.detect",
        lambda _repo: SimpleNamespace(head_commit="abc", normalized_key="repo"),
    )
    monkeypatch.setattr("stropha.bench.runner._source_worktree_state", lambda _repo: b"")

    postgres = _run_provenance(source, "search", runtime_id="runtime-one")
    monkeypatch.setattr("stropha.bench.runner.get_config", lambda: _experiment_config())
    sqlite = _run_provenance(source, "search", runtime_id="runtime-one")
    monkeypatch.setattr(
        "stropha.bench.runner.get_config",
        lambda: _experiment_config(blacklist={"embedder": ("voyage",)}),
    )
    blacklisted = _run_provenance(source, "search", runtime_id="runtime-one")
    encoded = json.dumps(postgres, sort_keys=True)

    assert postgres["experiment"]["storage"] == "postgres"
    assert postgres["experiment"]["policy_fingerprint"]
    assert postgres["experiment"]["postgres_cluster"] == {
        "host": "db.internal",
        "port": 5544,
        "database": "eval_admin",
        "user": "eval_user",
    }
    assert "top-secret" not in encoded
    assert "postgresql://" not in encoded
    assert postgres["run_id"] != sqlite["run_id"]
    assert sqlite["run_id"] != blacklisted["run_id"]


def test_postgres_runtime_provenance_uses_preflighted_cluster_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stropha.bench.runner.get_config",
        lambda: _experiment_config(storage="postgres"),
    )

    provenance = _experiment_provenance(
        cluster_identity=SimpleNamespace(
            host="resolved.internal",
            port=6432,
            database="eval_admin",
            user="eval_user",
            server_version="17.2",
            template_database="eval_template",
            template_extensions=("plpgsql=1.0", "vector=0.8.1"),
        )
    )

    assert provenance["postgres_cluster"] == {
        "host": "resolved.internal",
        "port": 6432,
        "database": "eval_admin",
        "user": "eval_user",
        "server_version": "17.2",
        "template_database": "eval_template",
        "template_extensions": ["plpgsql=1.0", "vector=0.8.1"],
    }


def test_rank_table_sorts_desc() -> None:
    rep = BenchReport(_results())
    ranked = rep.rank_table("recall@10")
    assert [r["cell_id"] for r in ranked] == ["a", "b"]
    assert rep.rank_table("mrr")[0]["cell_id"] == "b"  # b has higher mrr


def test_metric_names_prioritize_quality_and_guardrails() -> None:
    results = _results()
    results[0]["metrics"].update(
        {
            "success@10": 0.8,
            "ndcg@10": 0.7,
            "returned_tokens": 100,
            "latency_p99_ms": 12.0,
            "reranker_fallback_rate": 0.1,
        }
    )
    names = BenchReport(results).metric_names()
    assert names.index("ndcg@10") < names.index("returned_tokens")
    assert names.index("returned_tokens") < names.index("latency_p99_ms")
    assert "reranker_fallback_rate" in names


def test_quality_columns_are_concise_and_keep_selected_metric() -> None:
    results = _results()
    results[0]["metrics"].update(
        {
            "success@10": 0.8,
            "ndcg@10": 0.7,
            "returned_tokens": 100,
            "latency_p95_ms": 12.0,
            "latency_p99_ms": 15.0,
            "reranker_fallback_rate": 0.1,
        }
    )
    columns = BenchReport(results).quality_columns("ndcg@10")
    assert columns == [
        "ndcg@10",
        "success@10",
        "recall@10",
        "mrr",
        "returned_tokens",
        "latency_p95_ms",
        "reranker_fallback_rate",
    ]
    assert "latency_p99_ms" not in columns


def test_display_label_omits_fixed_indexing_values() -> None:
    record = {
        "label": "full label",
        "values": {
            "embedder": {"adapter": "mlx"},
            "enricher": "contextual",
            "reranker": "llm",
            "rrf_k": 40,
        },
    }
    assert BenchReport.display_label(record) == "reranker=llm rrf_k=40"


def test_marginals_mean_per_axis_value() -> None:
    rep = BenchReport(_results())
    marg = rep.marginals("recall@10")
    assert marg["reranker"]["None"]["mean"] == pytest.approx(0.8)
    assert marg["reranker"]["code-aware"]["mean"] == pytest.approx(0.7)
    # Both cells share hyde=False → one bucket averaging the two.
    assert marg["hyde"]["False"]["mean"] == pytest.approx(0.75)
    assert marg["hyde"]["False"]["n"] == 2.0


def test_baseline_deltas_shape_and_sign() -> None:
    rep = BenchReport(_results())
    deltas = rep.baseline_deltas("a")
    assert len(deltas) == 1
    d = deltas[0]
    assert d["cell_id"] == "b"
    # each delta = (point, lo, hi, p_gt_0)
    assert len(d["d_success"]) == 4 and len(d["d_mrr"]) == 4
    assert len(d["d_ndcg"]) == 4
    # recall identical (2/3 each) → point 0; mrr lower for b → negative point.
    assert d["d_success"][0] == pytest.approx(0.0)
    assert d["d_mrr"][0] < 0
    assert d["d_ndcg"][0] < 0
    assert d["label"] == "hyde=False reranker=code-aware"


def test_to_csv_has_header_and_rows() -> None:
    csv = to_csv(_results())
    lines = csv.strip().splitlines()
    assert lines[0].startswith("cell_id,indexing_sig,")
    assert "recall@10" in lines[0] and "reranker" in lines[0]
    assert len(lines) == 3  # header + 2 rows


def test_to_csv_quotes_mapping_values_with_commas_and_newlines() -> None:
    results = _results()
    results[0]["values"]["enricher"] = {
        "adapter": "contextual",
        "prompt": "first, second\nthird",
    }

    rows = list(csv.reader(io.StringIO(to_csv(results))))

    assert len(rows) == 3
    assert all(len(row) == len(rows[0]) for row in rows)
    enricher = rows[0].index("enricher")
    assert rows[1][enricher] == "contextual/first, second\nthird"


def test_eval_cell_records_full_metrics_provenance_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Storage:
        adapter_id = "storage:test"

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        @staticmethod
        def get_meta(key: str) -> str:
            return "enricher:test" if "enricher" in key else "embedder:test"

    built = SimpleNamespace(
        storage=Storage(),
        retrieval=SimpleNamespace(adapter_id="retrieval:test"),
        embedder=SimpleNamespace(adapter_id="embedder:test"),
        walker=SimpleNamespace(adapter_id="walker:test"),
        chunker=SimpleNamespace(adapter_id="chunker:test"),
        effective_enricher_id=lambda: "enricher:test",
    )
    summary = {
        "success@10": 0.9,
        "recall@10": 0.8,
        "mrr": 0.7,
        "ndcg@10": 0.75,
        "map@10": 0.65,
        "precision@10": 0.1,
        "returned_tokens": 123,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 2.0,
        "latency_p99_ms": 3.0,
        "reranker_attempts": 2,
        "reranker_fallbacks": 1,
        "reranker_fallback_rate": 0.5,
    }
    report = SimpleNamespace(
        summary=lambda: summary,
        results=[
            SimpleNamespace(
                case=SimpleNamespace(id="q1"),
                rank_of_first_hit=1,
                hit=True,
                reciprocal_rank=1.0,
                ndcg=1.0,
                latency_ms=1.23456,
            )
        ],
        diagnostics={"reranker": {"primary_latency_p95_ms": 4.0}},
        by_tag=lambda: {},
        n=1,
    )
    run_eval = MagicMock(return_value=report)
    monkeypatch.setattr("stropha.pipeline.builder.build_stages", lambda _cfg: built)
    monkeypatch.setattr("stropha.eval.run_eval", run_eval)
    monkeypatch.setattr(
        "stropha.pipeline.config.pipeline_config_from_dict",
        lambda _overrides, **_kwargs: {},
    )

    record = eval_cell(
        SimpleNamespace(top_k=10),
        Cell(
            values={
                "reranker": None,
                "context_budget": 200,
                "diversity": True,
                "diversity_max_chunks_per_file": 1,
            }
        ),
        tmp_path / "index.db",
        [object()],
        run_provenance={"run_id": "run-one"},
        index_diagnostics={
            "index_seconds": 12.3,
            "index_reused": False,
            "enricher": {
                "estimated_input_tokens": 100,
                "estimated_output_tokens": 20,
                "estimated_total_tokens": 120,
            },
        },
    )

    assert record["metrics"] == {
        **summary,
        "reranker_primary_latency_p95_ms": 4.0,
        "index_seconds": 12.3,
        "enricher_estimated_input_tokens": 100,
        "enricher_estimated_output_tokens": 20,
        "enricher_estimated_total_tokens": 120,
    }
    assert record["diagnostics"] == {
        **report.diagnostics,
        "index": {
            "index_seconds": 12.3,
            "index_reused": False,
            "enricher": {
                "estimated_input_tokens": 100,
                "estimated_output_tokens": 20,
                "estimated_total_tokens": 120,
            },
        },
    }
    assert record["provenance"]["run_id"] == "run-one"
    assert record["provenance"]["index"]["comparable"] is True
    assert record["provenance"]["pipeline"]["retrieval"] == "retrieval:test"
    assert record["per_query_metrics"] == {
        "q1": {
            "success@10": 1.0,
            "mrr": 1.0,
            "ndcg@10": 1.0,
            "latency_ms": 1.235,
        }
    }
    assert run_eval.call_args.kwargs["provenance"] == record["provenance"]
    assert run_eval.call_args.kwargs["context_budget"] == 200
    assert run_eval.call_args.kwargs["diversity_enabled"] is True
    assert run_eval.call_args.kwargs["diversity_max_chunks_per_file"] == 1


def test_eval_cell_rejects_index_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage = MagicMock()
    storage.__enter__.return_value = storage
    storage.get_meta.side_effect = lambda key: f"indexed-{key}"
    built = SimpleNamespace(
        storage=storage,
        retrieval=SimpleNamespace(adapter_id="retrieval:test"),
        embedder=SimpleNamespace(adapter_id="embedder:test"),
        walker=SimpleNamespace(adapter_id="walker:test"),
        chunker=SimpleNamespace(adapter_id="chunker:test"),
        effective_enricher_id=lambda: "enricher:test",
    )
    monkeypatch.setattr("stropha.pipeline.builder.build_stages", lambda _cfg: built)
    monkeypatch.setattr(
        "stropha.pipeline.config.pipeline_config_from_dict",
        lambda _overrides, **_kwargs: {},
    )

    with pytest.raises(BenchError, match="index identity drift"):
        eval_cell(
            SimpleNamespace(top_k=10),
            Cell(values={"reranker": None}),
            tmp_path / "index.db",
            [object()],
        )
