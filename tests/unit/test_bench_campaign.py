from __future__ import annotations

import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stropha.bench.campaign import (
    CampaignError,
    _assert_plan_unchanged,
    _assert_runtime_budgets,
    _build_campaign_plan,
    _cell_llm_reranker_invocations,
    _cell_query_intelligence_calls,
    _completed_group_stability,
    _configuration_label,
    _exclusive_start,
    _human_report,
    _public_status,
    _run_stage,
    _run_stages_parallel,
    _run_stages_sequentially,
    _run_worker,
    _wait_for_stage_process,
    _winner_stability,
    _write_stage_report,
    load_campaign_spec,
)
from stropha.bench.matrix import load_matrix
from stropha.config import Config


def _stub_plan_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stropha.bench.runner._index_runtime_id", lambda: "index-runtime")
    monkeypatch.setattr(
        "stropha.bench.runner._resolved_indexing_hash",
        lambda _cell, _pipeline=None: "indexing-hash",
    )
    monkeypatch.setattr(
        "stropha.ingest.git_meta.detect",
        lambda _path: SimpleNamespace(head_commit="a" * 40, normalized_key="repo"),
    )


def test_comprehensive_campaign_has_repeats_and_frozen_holdout() -> None:
    repo = Path(__file__).resolve().parents[2]
    spec = load_campaign_spec(repo / "src/stropha/bench/comprehensive_campaign.yaml")

    assert spec["split"] == "search"
    assert any(stage.get("split") == "holdout" for stage in spec["stages"])
    assert any(stage.get("repetitions", 1) > 1 for stage in spec["stages"])
    total = 0
    for stage in spec["stages"]:
        matrix = load_matrix(repo / stage["matrix"])
        if stage.get("axes"):
            matrix = matrix.select_axes(stage["axes"])
        total += len(matrix.expand()) * stage.get("repetitions", 1)
    assert total == 418

    indexing = load_matrix(repo / "src/stropha/bench/indexing_quality_shortlist_matrix.yaml")
    retrieval = load_matrix(repo / "src/stropha/bench/active_quality_shortlist_matrix.yaml")
    assert len(indexing.expand()) == 2
    assert len(retrieval.expand()) == 3
    assert indexing.baseline_cell() is not None
    assert retrieval.baseline_cell() is not None
    assert retrieval.baseline_cell().values["learned_weights"] is False


def test_retrieval_quality_campaign_is_exhaustive_and_bounded() -> None:
    repo = Path(__file__).resolve().parents[2]
    spec = load_campaign_spec(
        repo / "src/stropha/bench/retrieval_quality_exhaustive_campaign.yaml"
    )

    assert spec["mode"] == "exhaustive"
    assert spec["budgets"] == {
        "max_cells": 10_000,
        "max_queries": 1_000_000,
        "max_index_builds": 1,
    }
    assert len(spec["stages"]) == 1


def test_retrieval_quality_campaign_plan_matches_declared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    rel = Path("src/stropha/bench/retrieval_quality_exhaustive_campaign.yaml")
    spec = load_campaign_spec(repo / rel)
    monkeypatch.setattr("stropha.bench.campaign.get_config", Config)
    _stub_plan_runtime(monkeypatch)

    plan = _build_campaign_plan(
        spec,
        source_root=repo,
        spec_rel=rel,
        source_commit="a" * 40,
        config_sha="config",
    )

    assert plan["allowed"] is True
    assert plan["totals"]["cells"] == 6_216
    assert plan["totals"]["queries"] == 540_792
    assert plan["totals"]["index_builds"] == 1
    assert plan["totals"]["max_llm_reranker_invocations"] == 439_872
    assert plan["totals"]["max_query_intelligence_inference_calls"] == 987_624


def test_llm_reranker_plan_counts_possible_repair_generations() -> None:
    assert (
        _cell_llm_reranker_invocations(
            {"reranker": {"adapter": "llm", "repair_enabled": True}}, None
        )
        == 2
    )
    assert (
        _cell_llm_reranker_invocations(
            {
                "reranker": {"adapter": "llm", "repair_enabled": True},
                "multi_query": True,
                "multi_query_count": 3,
            },
            None,
        )
        == 10
    )
    assert (
        _cell_llm_reranker_invocations(
            {"reranker": {"adapter": "llm", "config": {"repair_enabled": False}}},
            None,
        )
        == 1
    )


def test_query_intelligence_plan_counts_multi_query_rewrite_fallback() -> None:
    values = {
        "multi_query": True,
        "multi_query_count": 3,
        "multi_query_min_terms": 4,
        "query_rewrite": True,
        "query_rewrite_max_terms": 6,
    }

    assert _cell_query_intelligence_calls(values, None) == 2
    assert (
        _cell_query_intelligence_calls({**values, "multi_query_min_terms": 7}, None)
        == 1
    )


def test_campaign_spec_rejects_invalid_split(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        "version: 1\nname: bad-campaign\nsplit: training\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="split"):
        load_campaign_spec(path)


def test_campaign_spec_rejects_minimum_above_repetition_cap(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        "version: 1\nname: bad-campaign\nsplit: search\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n"
        "    repetitions: 3\n    min_repetitions: 4\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="min_repetitions"):
        load_campaign_spec(path)


def test_exhaustive_campaign_requires_hard_budgets_and_all_repetitions(tmp_path: Path) -> None:
    missing_budgets = tmp_path / "missing-budgets.yaml"
    missing_budgets.write_text(
        "version: 1\nname: exhaustive\nmode: exhaustive\nsplit: search\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(CampaignError, match="required budgets"):
        load_campaign_spec(missing_budgets)

    adaptive_repetitions = tmp_path / "adaptive-repetitions.yaml"
    adaptive_repetitions.write_text(
        "version: 1\nname: exhaustive\nmode: exhaustive\nsplit: search\n"
        "budgets: {max_cells: 10, max_queries: 10, max_index_builds: 1}\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n"
        "    repetitions: 2\n    min_repetitions: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(CampaignError, match="every repetition"):
        load_campaign_spec(adaptive_repetitions)


def test_exhaustive_plan_resolves_cells_and_rejects_budget_overrun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "case-one",
                "query": "query",
                "expected_paths": ["src/a.py"],
                "split": "search",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "repo: .\n"
        "golden: golden.jsonl\n"
        "pipeline:\n"
        "  walker: {adapter: git-ls-files}\n"
        "  chunker: {adapter: tree-sitter-dispatch}\n"
        "  enricher: {adapter: hierarchical}\n"
        "  embedder: {adapter: local}\n"
        "  retrieval:\n"
        "    adapter: hybrid-rrf\n"
        "    config: {inherit_global_features: false}\n"
        "axes:\n  candidate_k: [20, 50]\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "campaign.yaml"
    spec_path.write_text(
        "version: 1\nname: exhaustive\nmode: exhaustive\nsplit: search\n"
        "budgets: {max_cells: 3, max_queries: 4, max_index_builds: 1}\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n"
        "    repetitions: 2\n    min_repetitions: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("stropha.bench.campaign.get_config", Config)
    _stub_plan_runtime(monkeypatch)
    spec = load_campaign_spec(spec_path)

    plan = _build_campaign_plan(
        spec,
        source_root=tmp_path,
        spec_rel=Path("campaign.yaml"),
        source_commit="a" * 40,
        config_sha="config",
    )

    assert plan["totals"]["cells"] == 4
    assert plan["totals"]["queries"] == 4
    assert plan["totals"]["index_builds"] == 1
    assert plan["allowed"] is False
    assert plan["budget_checks"][0]["ok"] is False

    matrix.write_text(
        matrix.read_text(encoding="utf-8").replace("20, 50", "20, 51"),
        encoding="utf-8",
    )
    changed = _build_campaign_plan(
        spec,
        source_root=tmp_path,
        spec_rel=Path("campaign.yaml"),
        source_commit="a" * 40,
        config_sha="config",
    )
    assert changed["plan_sha256"] != plan["plan_sha256"]

    tampered = {**plan, "totals": {**plan["totals"], "cells": 1}}
    with pytest.raises(CampaignError, match="plan drifted"):
        _assert_plan_unchanged(tampered, plan, plan["plan_sha256"])


def test_exhaustive_plan_rejects_golden_outside_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-golden.jsonl"
    outside.write_text("", encoding="utf-8")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        f"repo: .\ngolden: {outside}\n"
        "pipeline:\n"
        "  walker: {adapter: git-ls-files}\n"
        "  chunker: {adapter: tree-sitter-dispatch}\n"
        "  enricher: {adapter: hierarchical}\n"
        "  embedder: {adapter: local}\n"
        "  retrieval:\n"
        "    adapter: hybrid-rrf\n"
        "    config: {inherit_global_features: false}\n"
        "axes: {candidate_k: [20]}\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "campaign.yaml"
    spec_path.write_text(
        "version: 1\nname: exhaustive\nmode: exhaustive\nsplit: search\n"
        "budgets: {max_cells: 1, max_queries: 1, max_index_builds: 1}\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("stropha.bench.campaign.get_config", Config)
    _stub_plan_runtime(monkeypatch)

    with pytest.raises(CampaignError, match="golden escapes"):
        _build_campaign_plan(
            load_campaign_spec(spec_path),
            source_root=tmp_path,
            spec_rel=Path("campaign.yaml"),
            source_commit="a" * 40,
            config_sha="config",
        )


def test_stage_wait_enforces_wall_budget_during_single_stage(tmp_path: Path) -> None:
    process = MagicMock()
    process.wait.side_effect = [
        subprocess.TimeoutExpired("bench", 1),
        0,
    ]
    state = {"started_epoch": time.time() - 10, "budgets": {"max_wall_seconds": 1}}

    with pytest.raises(CampaignError, match="max_wall_seconds"):
        _wait_for_stage_process(process, tmp_path, state, cancel_event=None)


def test_exhaustive_sequential_run_does_not_apply_stability_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: list[str] = []
    stages = [{"id": "one"}, {"id": "two"}]
    monkeypatch.setattr(
        "stropha.bench.campaign._completed_group_stability",
        lambda *_args: {"confident": True, "runs": 4},
    )
    monkeypatch.setattr("stropha.bench.campaign._assert_runtime_budgets", lambda *_args: None)
    monkeypatch.setattr("stropha.bench.campaign._assert_config_unchanged", lambda *_args: None)
    monkeypatch.setattr(
        "stropha.bench.campaign._run_stage",
        lambda stage, *_args, **_kwargs: called.append(stage["id"]),
    )

    _run_stages_sequentially(
        stages,
        stages,
        {"mode": "exhaustive"},
        {},
        tmp_path / "state.json",
        tmp_path,
        tmp_path,
    )

    assert called == ["one", "two"]


def test_parallel_stage_runner_respects_worker_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guard = threading.Lock()
    active = 0
    maximum = 0

    def fake_run_stage(*_args: object, **_kwargs: object) -> None:
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with guard:
            active -= 1

    monkeypatch.setattr("stropha.bench.campaign._run_stage", fake_run_stage)
    monkeypatch.setattr("stropha.bench.campaign._assert_runtime_budgets", lambda *_args: None)
    monkeypatch.setattr("stropha.bench.campaign._assert_config_unchanged", lambda *_args: None)

    _run_stages_parallel(
        [{"id": str(index)} for index in range(6)],
        2,
        {},
        tmp_path / "state.json",
        tmp_path,
        tmp_path,
    )

    assert maximum == 2


def test_parallel_stage_submission_failure_cancels_partial_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first: concurrent.futures.Future[None] = concurrent.futures.Future()
    shutdown: list[tuple[bool, bool]] = []

    class Executor:
        calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def submit(self, *_args: object, **_kwargs: object):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return first

        @staticmethod
        def shutdown(*, wait: bool, cancel_futures: bool) -> None:
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr("stropha.bench.campaign.concurrent.futures.ThreadPoolExecutor", Executor)

    with pytest.raises(KeyboardInterrupt):
        _run_stages_parallel(
            [{"id": "one"}, {"id": "two"}],
            2,
            {},
            tmp_path / "state.json",
            tmp_path,
            tmp_path,
        )

    assert first.cancelled()
    assert shutdown == [(True, True)]


def test_artifact_budget_ignores_files_removed_during_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vanished = tmp_path / "vanished.tmp"
    monkeypatch.setattr(Path, "rglob", lambda _self, _pattern: iter([vanished]))
    original_stat = Path.stat

    def racing_stat(path: Path, *args: object, **kwargs: object):
        if path == vanished:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)

    _assert_runtime_budgets(tmp_path, {"budgets": {"max_artifact_bytes": 1}})


def test_human_report_shows_repeatability_and_weaknesses(tmp_path: Path) -> None:
    stage_root = tmp_path / "stages" / "quality-r01"
    stage_root.mkdir(parents=True)
    report_path = stage_root / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "stage": "quality-r01",
                "group": "quality",
                "split": "search",
                "top": [
                    {
                        "label": "reranker=llm",
                        "metrics": {
                            "ndcg@10": 0.8,
                            "mrr": 0.9,
                            "success@10": 1.0,
                            "recall@10": 0.8,
                            "returned_tokens": 100,
                        },
                        "by_tag": {
                            "strong": {"ndcg@10": 1.0, "cases": 2},
                            "weak": {"ndcg@10": 0.1, "cases": 3},
                        },
                    }
                ],
                "baseline_deltas": [],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "name": "quality",
        "status": "completed",
        "source_commit": "abc",
        "stages": [
            {"id": "quality-r01", "status": "completed", "report": str(report_path)},
            {"id": "quality-r02", "status": "completed", "report": str(report_path)},
        ],
    }

    rendered = _human_report(summary, tmp_path)

    assert "reranker=llm` won 2/2 runs" in rendered
    assert "Strongest tags: strong" in rendered
    assert "weak (0.100" in rendered


def test_configuration_label_identifies_indexing_values() -> None:
    label = _configuration_label(
        {
            "cell_id": "abc",
            "values": {
                "embedder": {"adapter": "mlx", "model": "qwen"},
                "enricher": {"adapter": "contextual", "max_tokens": 64},
                "reranker": "llm",
            },
        }
    )

    assert label == "[abc] embedder=mlx/qwen enricher=contextual reranker=llm"


def test_winner_stability_requires_wilson_lower_bound_above_half() -> None:
    winner = {"top": [{"cell_id": "a", "label": "A"}]}
    loser = {"top": [{"cell_id": "b", "label": "B"}]}

    assert _winner_stability([winner] * 4)["confident"] is True
    assert _winner_stability([winner] * 3 + [loser])["confident"] is False


def test_completed_group_stability_honours_minimum_repetitions(tmp_path: Path) -> None:
    stages = []
    for repetition in range(1, 5):
        report = tmp_path / f"report-{repetition}.json"
        report.write_text(
            json.dumps({"top": [{"cell_id": "a", "label": "A"}]}),
            encoding="utf-8",
        )
        stages.append(
            {
                "group": "quality",
                "repetition": repetition,
                "status": "completed",
                "report": str(report),
            }
        )
    current = {"group": "quality", "repetition": 5, "min_repetitions": 4}

    stability = _completed_group_stability(stages, current)

    assert stability is not None
    assert stability["confident"] is True


def test_run_stage_forwards_split_and_selected_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = MagicMock(pid=123)
    process.wait.return_value = 0
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("stropha.bench.campaign.subprocess.Popen", popen)
    monkeypatch.setattr("stropha.bench.campaign._assert_config_unchanged", lambda _state: None)
    monkeypatch.setattr(
        "stropha.bench.campaign._write_stage_report",
        lambda _stage, _matrix: {"successful_cells": 1, "failed_cells": 0},
    )
    state_path = tmp_path / "state.json"
    stage = {
        "id": "holdout",
        "matrix": "matrix.yaml",
        "results": str(tmp_path / "results.jsonl"),
        "split": "holdout",
        "axes": ["reranker", "rrf_k"],
        "total_cells": 1,
        "status": "pending",
    }
    state = {"config_sha256": "config", "stages": [stage]}

    _run_stage(stage, state, state_path, tmp_path, tmp_path)

    command = popen.call_args.args[0]
    assert command[command.index("--split") + 1] == "holdout"
    assert command.count("--axis") == 2
    assert stage["status"] == "completed"


def test_stage_report_preserves_cost_diagnostics_without_double_counting_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "repo: .\ngolden: golden.jsonl\naxes:\n  hyde: [false, true]\n"
        "baseline:\n  hyde: false\n",
        encoding="utf-8",
    )
    matrix = load_matrix(matrix_path)
    records = []
    for index, cell in enumerate(matrix.expand()):
        records.append(
            {
                "ok": True,
                "cell_id": cell.cell_id,
                "indexing_sig": "shared-index",
                "label": cell.label(),
                "values": cell.values,
                "metrics": {"ndcg@10": 0.8 - index * 0.1},
                "provenance": {"run_id": "run-one"},
                "diagnostics": {
                    "index": {
                        "index_seconds": 10.0,
                        "index_reused": False,
                        "enricher": {
                            "estimated_input_tokens": 100,
                            "estimated_output_tokens": 20,
                        },
                    }
                },
                "per_query": {"q1": 1},
                "per_query_metrics": {"q1": {"ndcg@10": 1.0, "latency_ms": 2.0}},
                "by_tag": {},
            }
        )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr("stropha.bench.campaign.get_config", lambda: object())
    monkeypatch.setattr(
        "stropha.bench.campaign.ExperimentPolicy.from_config",
        lambda _config: SimpleNamespace(filter_matrix=lambda selected: selected),
    )
    stage = {
        "id": "costs",
        "results": str(results),
        "report": str(tmp_path / "report.json"),
        "csv": str(tmp_path / "results.csv"),
    }

    payload = _write_stage_report(stage, matrix_path)

    assert payload["indexing"] == {
        "unique_indexes": 1,
        "built_indexes": 1,
        "reused_indexes": 0,
        "total_seconds": 10.0,
        "enricher_estimated_input_tokens": 100,
        "enricher_estimated_output_tokens": 20,
    }
    assert payload["top"][0]["diagnostics"] == records[0]["diagnostics"]
    assert payload["top"][0]["per_query_metrics"] == records[0]["per_query_metrics"]


def test_public_status_counts_latest_record_per_cell(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({"cell_id": "a", "ok": False}),
                json.dumps({"cell_id": "a", "ok": True}),
                json.dumps({"cell_id": "b", "ok": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    state = {
        "name": "campaign",
        "status": "running",
        "pid": None,
        "stages": [{"id": "stage", "results": str(results)}],
    }

    status = _public_status(state, tmp_path)

    assert status["stages"][0]["completed_cells"] == 1
    assert status["stages"][0]["recorded_failures"] == 1


def test_campaign_start_lock_rejects_concurrent_launcher(tmp_path: Path) -> None:
    lock = tmp_path / "start.lock"

    def acquire_again() -> None:
        with _exclusive_start(lock):
            pytest.fail("second launcher acquired the lock")

    with _exclusive_start(lock), pytest.raises(CampaignError, match="already in progress"):
        acquire_again()


def test_worker_setup_failure_records_state_and_releases_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    released: list[Path] = []
    config_sha = "config-one"
    state = {
        "source_commit": "a" * 40,
        "config_sha256": config_sha,
        "stages": [],
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr("stropha.bench.campaign._config_sha256", lambda: config_sha)
    monkeypatch.setattr(
        "stropha.bench.campaign.Snapshot.release",
        lambda snapshot: released.append(snapshot.path),
    )

    result = _run_worker(
        spec_path=tmp_path / "missing.yaml",
        source_repo=tmp_path,
        snapshot_path=tmp_path / "snapshot",
        artifact_root=tmp_path,
    )

    recorded = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert result == 1
    assert recorded["status"] == "failed"
    assert "could not load campaign spec" in recorded["error"]
    assert released == [tmp_path / "snapshot"]
    assert not (tmp_path / "campaign.pid").exists()
