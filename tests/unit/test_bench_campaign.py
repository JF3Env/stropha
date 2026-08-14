from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.bench.campaign import (
    CampaignError,
    _completed_group_stability,
    _configuration_label,
    _exclusive_start,
    _human_report,
    _public_status,
    _run_stage,
    _run_worker,
    _winner_stability,
    load_campaign_spec,
)
from stropha.bench.matrix import load_matrix


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
