from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.bench.campaign import (
    CampaignError,
    _exclusive_start,
    _public_status,
    _run_worker,
    load_campaign_spec,
)
from stropha.bench.matrix import load_matrix


def test_comprehensive_campaign_is_search_only_and_has_318_cells() -> None:
    repo = Path(__file__).resolve().parents[2]
    spec = load_campaign_spec(repo / "src/stropha/bench/comprehensive_campaign.yaml")

    assert spec["split"] == "search"
    assert not any("holdout" in stage["matrix"] for stage in spec["stages"])
    assert sum(len(load_matrix(repo / stage["matrix"]).expand()) for stage in spec["stages"]) == 318


def test_campaign_spec_rejects_holdout(tmp_path: Path) -> None:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        "version: 1\nname: bad-campaign\nsplit: holdout\n"
        "stages:\n  - id: one\n    matrix: matrix.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="search"):
        load_campaign_spec(path)


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
