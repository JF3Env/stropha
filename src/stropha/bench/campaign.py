"""Detached, resumable orchestration for multi-matrix benchmark campaigns."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..config import get_config
from ..errors import StrophaError
from ..experiment_policy import ExperimentPolicy
from ..meta.snapshot import Snapshot, create_snapshot, resolve_sha
from .matrix import load_matrix
from .report import BenchReport, load_results, to_csv

_STAGE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class CampaignError(StrophaError):
    """Raised when a benchmark campaign cannot be started or inspected safely."""


def load_campaign_spec(path: Path) -> dict[str, Any]:
    """Load and validate a checked-in campaign specification."""
    path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CampaignError(f"could not load campaign spec {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise CampaignError("campaign spec version must be 1")
    if not isinstance(raw.get("name"), str) or not _STAGE_ID.fullmatch(raw["name"]):
        raise CampaignError("campaign name must contain only lowercase letters, digits, and dashes")
    if raw.get("split") not in {"search", "holdout"}:
        raise CampaignError("campaign split must be 'search' or 'holdout'")
    if "holdout_note" in raw and not isinstance(raw["holdout_note"], str):
        raise CampaignError("campaign holdout_note must be a string")
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        raise CampaignError("campaign spec must declare at least one stage")
    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            raise CampaignError("each campaign stage must be a mapping")
        stage_id = stage.get("id")
        if not isinstance(stage_id, str) or not _STAGE_ID.fullmatch(stage_id):
            raise CampaignError(f"invalid campaign stage id: {stage_id!r}")
        if stage_id in seen:
            raise CampaignError(f"duplicate campaign stage id: {stage_id!r}")
        if not isinstance(stage.get("matrix"), str) or not stage["matrix"]:
            raise CampaignError(f"campaign stage {stage_id!r} has no matrix")
        if stage.get("split", raw["split"]) not in {"search", "holdout"}:
            raise CampaignError(f"invalid split for campaign stage {stage_id!r}")
        repetitions = stage.get("repetitions", 1)
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
            raise CampaignError(f"campaign stage {stage_id!r} repetitions must be positive")
        min_repetitions = stage.get("min_repetitions", repetitions)
        if (
            not isinstance(min_repetitions, int)
            or isinstance(min_repetitions, bool)
            or not 1 <= min_repetitions <= repetitions
        ):
            raise CampaignError(
                f"campaign stage {stage_id!r} min_repetitions must be between 1 and repetitions"
            )
        axes = stage.get("axes", [])
        if not isinstance(axes, list) or not all(isinstance(axis, str) and axis for axis in axes):
            raise CampaignError(f"campaign stage {stage_id!r} axes must be a list of names")
        seen.add(stage_id)
    return raw


def start_campaign(
    spec_path: Path,
    *,
    source_repo: Path,
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    """Pin HEAD and start one detached campaign worker."""
    source_repo = source_repo.expanduser().resolve()
    spec_path = spec_path.expanduser().resolve()
    spec = load_campaign_spec(spec_path)
    try:
        spec_rel = spec_path.relative_to(source_repo)
    except ValueError as exc:
        raise CampaignError("campaign spec must be inside the source repository") from exc

    sha = resolve_sha("HEAD", source_repo)
    base = (
        artifact_base.expanduser().resolve()
        if artifact_base is not None
        else source_repo / ".stropha" / "bench" / "campaigns"
    )
    campaign_home = base / spec["name"]
    config_sha = _config_sha256()
    root = campaign_home / f"{sha[:12]}-{config_sha[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    campaign_home.mkdir(parents=True, exist_ok=True)
    with _exclusive_start(campaign_home / "start.lock"):
        return _start_campaign_locked(
            spec=spec,
            spec_rel=spec_rel,
            source_repo=source_repo,
            sha=sha,
            config_sha=config_sha,
            campaign_home=campaign_home,
            root=root,
        )


def _start_campaign_locked(
    *,
    spec: dict[str, Any],
    spec_rel: Path,
    source_repo: Path,
    sha: str,
    config_sha: str,
    campaign_home: Path,
    root: Path,
) -> dict[str, Any]:
    """Start a worker while holding the campaign's cross-process launch lock."""
    state_path = root / "state.json"
    existing = _read_json(state_path)
    if existing and _campaign_alive(existing, root):
        raise CampaignError(f"campaign is already running (pid {existing.get('pid')})")
    if existing and existing.get("status") == "completed":
        return _public_status(existing, root)

    snapshot: Snapshot | None = None
    process: subprocess.Popen[Any] | None = None
    state: dict[str, Any] = {}
    try:
        snapshot = create_snapshot(
            source_repo=source_repo,
            ref=sha,
            snapshot_root=root / "worktrees",
        )
        snapshot_spec = snapshot.path / spec_rel
        if not snapshot_spec.is_file():
            raise CampaignError("campaign spec is not committed at the selected HEAD")

        log_path = root / "campaign.log"
        state = {
            "version": 1,
            "name": spec["name"],
            "status": "starting",
            "source_commit": sha,
            "config_sha256": config_sha,
            "holdout_note": spec.get("holdout_note"),
            "source_repo": str(source_repo),
            "snapshot": str(snapshot.path),
            "spec": str(spec_rel),
            "artifact_root": str(root),
            "log": str(log_path),
            "started_at": _now(),
            "pid": None,
            "active_child_pid": None,
            "stages": [],
        }
        _atomic_json(state_path, state)
        env = os.environ.copy()
        python_path = str(snapshot.path / "src")
        if env.get("PYTHONPATH"):
            python_path += os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = python_path
        command = [
            sys.executable,
            "-m",
            "stropha.bench.campaign",
            "worker",
            "--spec",
            str(snapshot_spec),
            "--source-repo",
            str(source_repo),
            "--snapshot",
            str(snapshot.path),
            "--artifact-root",
            str(root),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            launched = subprocess.Popen(
                command,
                cwd=snapshot.path,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        process = launched
        (root / "campaign.pid").write_text(f"{launched.pid}\n", encoding="ascii")
        for _ in range(50):
            ready = _read_json(state_path) or {}
            if ready.get("pid") == launched.pid or launched.poll() is not None:
                break
            time.sleep(0.1)
        _atomic_json(
            campaign_home / "latest.json",
            {"artifact_root": str(root), "source_commit": sha, "config_sha256": config_sha},
        )
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if snapshot is not None:
            snapshot.release()
        if state:
            state.update({"status": "failed", "error": str(exc), "finished_at": _now()})
            with contextlib.suppress(OSError):
                _atomic_json(state_path, state)
        if isinstance(exc, CampaignError):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise CampaignError(f"could not start detached campaign: {exc}") from exc
    assert process is not None
    return {
        **_public_status(state, root),
        "pid": process.pid,
        "status": "starting",
    }


def campaign_status(
    spec_path: Path,
    *,
    source_repo: Path,
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    """Return live status for the latest run of a campaign spec."""
    spec = load_campaign_spec(spec_path)
    source_repo = source_repo.expanduser().resolve()
    base = (
        artifact_base.expanduser().resolve()
        if artifact_base is not None
        else source_repo / ".stropha" / "bench" / "campaigns"
    )
    latest = _read_json(base / spec["name"] / "latest.json")
    if not latest or not latest.get("artifact_root"):
        raise CampaignError(f"campaign {spec['name']!r} has not been started")
    root = Path(latest["artifact_root"])
    state = _read_json(root / "state.json")
    if not state:
        raise CampaignError(f"campaign state is missing: {root / 'state.json'}")
    return _public_status(state, root)


def stop_campaign(
    spec_path: Path,
    *,
    source_repo: Path,
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    """Request graceful termination of the latest campaign worker."""
    status = campaign_status(
        spec_path,
        source_repo=source_repo,
        artifact_base=artifact_base,
    )
    root = Path(status["artifact_root"])
    state = _read_json(root / "state.json") or {}
    pid = state.get("pid")
    if not isinstance(pid, int):
        pid = _read_pid(root / "campaign.pid")
    child_pid = state.get("active_child_pid")
    worker_matches = isinstance(pid, int) and _process_matches(pid, root)
    child_matches = isinstance(child_pid, int) and _process_matches(child_pid, root)
    if not worker_matches and not child_matches:
        return status
    try:
        if isinstance(pid, int) and worker_matches:
            os.kill(pid, signal.SIGTERM)
        elif isinstance(child_pid, int) and child_matches:
            os.kill(child_pid, signal.SIGINT)
    except OSError as exc:
        raise CampaignError(f"could not stop campaign process: {exc}") from exc
    return {**status, "stop_requested": True}


def _run_worker(
    *,
    spec_path: Path,
    source_repo: Path,
    snapshot_path: Path,
    artifact_root: Path,
) -> int:
    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    state_path = artifact_root / "state.json"
    pid_path = artifact_root / "campaign.pid"
    snapshot = Snapshot(
        sha="",
        path=snapshot_path,
        source_repo=source_repo,
    )
    state: dict[str, Any] = {}
    interrupted = False
    try:
        prior = _read_json(state_path) or {}
        state = {
            **prior,
            "status": "running",
            "pid": os.getpid(),
            "active_child_pid": None,
            "stages": prior.get("stages") or [],
            "updated_at": _now(),
        }
        snapshot.sha = str(state["source_commit"])
        _atomic_json(state_path, state)
        pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
        _assert_config_unchanged(state)
        spec = load_campaign_spec(spec_path)
        stages = _prepare_stages(
            spec,
            snapshot_path,
            artifact_root,
            state["stages"],
        )
        state["stages"] = stages
        _atomic_json(state_path, state)
        for stage in stages:
            if stage.get("status") in {"completed", "not_needed"}:
                continue
            stability = _completed_group_stability(stages, stage)
            if stability and stability["confident"]:
                stage["status"] = "not_needed"
                stage["reason"] = (
                    f"winner stability reached after {stability['runs']} repetitions"
                )
                state["updated_at"] = _now()
                _atomic_json(state_path, state)
                continue
            _assert_config_unchanged(state)
            _run_stage(stage, state, state_path, snapshot_path, artifact_root)
            _assert_config_unchanged(state)
        failed = any(stage.get("status") not in {"completed", "not_needed"} for stage in stages)
        state["status"] = "completed_with_failures" if failed else "completed"
        _write_campaign_summary(state, artifact_root)
    except KeyboardInterrupt:
        interrupted = True
        state["status"] = "interrupted"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
    finally:
        if state:
            state["active_child_pid"] = None
            state["finished_at"] = _now()
            state["updated_at"] = _now()
            with contextlib.suppress(OSError):
                _atomic_json(state_path, state)
        with contextlib.suppress(OSError):
            pid_path.unlink(missing_ok=True)
        try:
            snapshot.release()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    status = state.get("status")
    return 130 if interrupted else (0 if status == "completed" else 1)


def _prepare_stages(
    spec: dict[str, Any],
    snapshot_path: Path,
    artifact_root: Path,
    prior_stages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prior = {stage.get("id"): stage for stage in prior_stages}
    policy = ExperimentPolicy.from_config(get_config())
    stages: list[dict[str, Any]] = []
    for declared in spec["stages"]:
        declared_id = declared["id"]
        matrix_path = (snapshot_path / declared["matrix"]).resolve()
        try:
            matrix_path.relative_to(snapshot_path)
        except ValueError as exc:
            raise CampaignError(f"matrix for stage {declared_id!r} escapes the snapshot") from exc
        matrix = policy.filter_matrix(load_matrix(matrix_path))
        axes = declared.get("axes", [])
        if axes:
            matrix = matrix.select_axes(axes)
        cells = matrix.expand()
        repetitions = declared.get("repetitions", 1)
        for repetition in range(1, repetitions + 1):
            stage_id = (
                declared_id if repetitions == 1 else f"{declared_id}-r{repetition:02d}"
            )
            stage_root = artifact_root / "stages" / stage_id
            stage_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(matrix_path, stage_root / "matrix.yaml")
            existing = prior.get(stage_id, {})
            stages.append(
                {
                    **existing,
                    "id": stage_id,
                    "group": declared_id,
                    "repetition": repetition,
                    "min_repetitions": declared.get("min_repetitions", repetitions),
                    "split": declared.get("split", spec["split"]),
                    "axes": axes,
                    "matrix": str(declared["matrix"]),
                    "matrix_id": matrix.matrix_id,
                    "total_cells": len(cells),
                    "results": str(stage_root / "results.jsonl"),
                    "report": str(stage_root / "report.json"),
                    "csv": str(stage_root / "results.csv"),
                    "status": existing.get("status", "pending"),
                }
            )
    return stages


def _run_stage(
    stage: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    snapshot_path: Path,
    artifact_root: Path,
) -> None:
    stage["status"] = "running"
    stage["started_at"] = _now()
    stage.pop("error", None)
    state["updated_at"] = _now()
    _atomic_json(state_path, state)
    matrix_path = snapshot_path / stage["matrix"]
    results_path = Path(stage["results"])
    command = [
        sys.executable,
        "-c",
        (
            "from stropha.bench.campaign import _require_expected_config; "
            "_require_expected_config(); from stropha.cli import app; app()"
        ),
        "bench",
        "run",
        "--matrix",
        str(matrix_path),
        "--out",
        str(results_path),
        "--index-dir",
        str(artifact_root / "indexes"),
        "--split",
        stage["split"],
        "--resume",
    ]
    for axis in stage.get("axes", []):
        command.extend(["--axis", axis])
    env = os.environ.copy()
    env["STROPHA_BENCH_EXPECTED_CONFIG_SHA256"] = str(state["config_sha256"])
    process = subprocess.Popen(command, cwd=snapshot_path, env=env)
    try:
        state["active_child_pid"] = process.pid
        _atomic_json(state_path, state)
        returncode = process.wait()
    except BaseException:
        _interrupt_child(process)
        raise
    finally:
        state["active_child_pid"] = None
    _assert_config_unchanged(state)
    try:
        report = _write_stage_report(stage, matrix_path)
        stage["successful_cells"] = report["successful_cells"]
        stage["failed_cells"] = report["failed_cells"]
        if returncode != 0:
            raise CampaignError(f"bench process exited with status {returncode}")
        if report["failed_cells"]:
            raise CampaignError(f"{report['failed_cells']} benchmark cells failed")
        if report["successful_cells"] != stage["total_cells"]:
            raise CampaignError(
                f"expected {stage['total_cells']} successful cells, "
                f"found {report['successful_cells']}"
            )
        stage["status"] = "completed"
    except Exception as exc:
        stage["status"] = "failed"
        stage["error"] = str(exc)
    stage["finished_at"] = _now()
    state["updated_at"] = _now()
    _atomic_json(state_path, state)


def _write_stage_report(stage: dict[str, Any], matrix_path: Path) -> dict[str, Any]:
    results_path = Path(stage["results"])
    records = load_results(results_path)
    matrix = ExperimentPolicy.from_config(get_config()).filter_matrix(load_matrix(matrix_path))
    if stage.get("axes"):
        matrix = matrix.select_axes(stage["axes"])
    report = BenchReport(records)
    metric = "ndcg@10"
    baseline = matrix.baseline_cell()
    latest = _latest_records(results_path)
    failed = sum(not record.get("ok", True) for record in latest.values())
    payload: dict[str, Any] = {
        "stage": stage["id"],
        "group": stage.get("group", stage["id"]),
        "repetition": stage.get("repetition", 1),
        "split": stage.get("split", "search"),
        "matrix_id": matrix.matrix_id,
        "metric": metric,
        "successful_cells": len(records),
        "failed_cells": failed,
        "top": [
            {
                "cell_id": record.get("cell_id"),
                "label": _configuration_label(record),
                "values": record.get("values"),
                "metrics": record.get("metrics"),
                "by_tag": record.get("by_tag"),
            }
            for record in report.rank_table(metric)[:20]
        ],
        "marginals": report.marginals(metric),
        "baseline_cell_id": baseline.cell_id if baseline else None,
        "baseline_deltas": report.baseline_deltas(baseline.cell_id) if baseline else [],
        "generated_at": _now(),
    }
    _atomic_json(Path(stage["report"]), payload)
    Path(stage["csv"]).write_text(to_csv(records), encoding="utf-8")
    return payload


def _write_campaign_summary(state: dict[str, Any], artifact_root: Path) -> None:
    payload = {
        "name": state["name"],
        "status": state["status"],
        "source_commit": state["source_commit"],
        "config_sha256": state["config_sha256"],
        "holdout_note": state.get("holdout_note"),
        "started_at": state.get("started_at"),
        "finished_at": _now(),
        "stages": state["stages"],
    }
    _atomic_json(artifact_root / "summary.json", payload)
    (artifact_root / "report.md").write_text(
        _human_report(payload, artifact_root), encoding="utf-8"
    )


def _human_report(summary: dict[str, Any], artifact_root: Path) -> str:
    """Render a concise campaign report; CSV files remain the exhaustive appendix."""
    reports = []
    for stage in summary["stages"]:
        report = _read_json(Path(stage["report"]))
        if report:
            reports.append(report)

    lines = [
        f"# Benchmark Campaign: {summary['name']}",
        "",
        f"- Status: **{summary['status']}**",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Started: {summary.get('started_at') or 'unknown'}",
        f"- Finished: {summary.get('finished_at') or 'unknown'}",
        f"- Holdout qualification: {summary.get('holdout_note') or 'not specified'}",
        f"- Successfully executed stages: "
        f"{sum(stage.get('status') == 'completed' for stage in summary['stages'])}",
        f"- Confidence-skipped repetitions: "
        f"{sum(stage.get('status') == 'not_needed' for stage in summary['stages'])}",
        "",
        "## Stage Results",
        "",
        "| Stage | Split | Winner | NDCG@10 | MRR | Success@10 | Recall@10 | Tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        top = (report.get("top") or [{}])[0]
        metrics = top.get("metrics") or {}
        label = str(top.get("label") or "no successful cells").replace("|", "\\|")
        lines.append(
            f"| {report['stage']} | {report.get('split', 'search')} | {label} | "
            f"{_metric(metrics, 'ndcg@10')} | {_metric(metrics, 'mrr')} | "
            f"{_metric(metrics, 'success@10')} | {_metric(metrics, 'recall@10')} | "
            f"{_metric(metrics, 'returned_tokens', decimals=0)} |"
        )

    lines.extend(["", "## Repeatability", ""])
    groups: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        groups.setdefault(str(report.get("group", report["stage"])), []).append(report)
    repeated = False
    for group, group_reports in groups.items():
        if len(group_reports) < 2:
            continue
        repeated = True
        stability = _winner_stability(group_reports)
        verdict = "confident" if stability["confident"] else "inconclusive"
        lines.append(
            f"- **{group}:** {verdict}; `{stability['winner_label']}` won "
            f"{stability['wins']}/{stability['runs']} runs "
            f"(win rate `{stability['rate']:.3f}`, 95% Wilson CI "
            f"`{stability['lo']:.3f}` to `{stability['hi']:.3f}`)."
        )
    if not repeated:
        lines.append("- No stages were repeated; winner stability was not measured.")

    lines.extend(["", "## Strengths And Weaknesses", ""])
    for report in reports:
        top = (report.get("top") or [{}])[0]
        tags = top.get("by_tag") or {}
        ranked = sorted(
            (
                (tag, values.get("ndcg@10", 0.0), values.get("cases", 0))
                for tag, values in tags.items()
                if isinstance(values, dict)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            continue
        strong = ", ".join(f"{tag} ({score:.3f}, n={cases})" for tag, score, cases in ranked[:5])
        weak = ", ".join(f"{tag} ({score:.3f}, n={cases})" for tag, score, cases in ranked[-5:])
        lines.extend(
            [
                f"### {report['stage']} ({report.get('split', 'search')})",
                "",
                f"- Strongest tags: {strong}",
                f"- Weakest tags: {weak}",
            ]
        )
        top_id = top.get("cell_id")
        if top_id == report.get("baseline_cell_id"):
            lines.append("- Baseline retained the highest NDCG@10 in this stage.")
        delta = next(
            (item for item in report.get("baseline_deltas") or [] if item["cell_id"] == top_id),
            None,
        )
        if delta:
            interval = delta.get("d_ndcg") or []
            if len(interval) >= 4:
                comparisons = max(1, len(report.get("baseline_deltas") or []))
                required = 1.0 - (0.05 / comparisons)
                verdict = "passes" if interval[1] > 0 and interval[3] >= required else "inconclusive"
                lines.append(
                    f"- Winner NDCG delta vs baseline: `{interval[0]:+.4f}` "
                    f"(nominal 95% CI `{interval[1]:+.4f}` to `{interval[2]:+.4f}`, "
                    f"P(delta > 0) `{interval[3]:.3f}`); {verdict} the "
                    f"Bonferroni-adjusted `{required:.3f}` probability threshold for "
                    f"{comparisons} comparisons."
                )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefer configurations that win repeatedly and retain gains on the frozen holdout.",
            "- A confidence interval crossing zero is inconclusive even when the point estimate is positive.",
            "- Token counts measure returned context, not all inference cost from enrichment or reranking.",
            "- Per-stage `results.csv`, `results.jsonl`, and `report.json` contain every tested configuration.",
            f"- Artifact root: `{artifact_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def _metric(metrics: dict[str, Any], name: str, *, decimals: int = 4) -> str:
    value = metrics.get(name)
    return f"{value:.{decimals}f}" if isinstance(value, int | float) else "n/a"


def _configuration_label(record: dict[str, Any]) -> str:
    """Return a concise label that still uniquely identifies indexing cells."""
    parts = [f"[{record.get('cell_id', '?')}]"]
    for name, value in sorted((record.get("values") or {}).items()):
        if isinstance(value, dict):
            adapter = value.get("adapter")
            model = value.get("model")
            shown = "/".join(str(item) for item in (adapter, model) if item)
            value = shown or "configured"
        parts.append(f"{name}={value}")
    return " ".join(parts)


def _winner_stability(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Return modal winner frequency with a 95% Wilson binomial interval."""
    winners: dict[str, tuple[str, int]] = {}
    for report in reports:
        top = (report.get("top") or [{}])[0]
        cell_id = str(top.get("cell_id") or "no-winner")
        label = str(top.get("label") or "no winner")
        _, count = winners.get(cell_id, (label, 0))
        winners[cell_id] = (label, count + 1)
    winner_id, (winner_label, wins) = max(winners.items(), key=lambda item: item[1][1])
    runs = len(reports)
    rate = wins / runs
    z = 1.959963984540054
    denominator = 1 + (z * z / runs)
    centre = (rate + z * z / (2 * runs)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) / runs) + z * z / (4 * runs * runs))
    margin /= denominator
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return {
        "winner_cell_id": winner_id,
        "winner_label": winner_label,
        "wins": wins,
        "runs": runs,
        "rate": rate,
        "lo": lo,
        "hi": hi,
        "confident": lo > 0.5,
    }


def _completed_group_stability(
    stages: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any] | None:
    completed = [
        stage
        for stage in stages
        if stage.get("group") == current.get("group")
        and stage.get("status") == "completed"
        and stage.get("repetition", 1) < current.get("repetition", 1)
    ]
    if len(completed) < current.get("min_repetitions", 1):
        return None
    reports = [_read_json(Path(stage["report"])) for stage in completed]
    available = [report for report in reports if report]
    return _winner_stability(available) if available else None


def _public_status(state: dict[str, Any], root: Path) -> dict[str, Any]:
    stages = []
    for stage in state.get("stages") or []:
        results = Path(stage["results"])
        latest = _latest_records(results) if results.is_file() else {}
        stages.append(
            {
                **stage,
                "completed_cells": sum(record.get("ok", True) for record in latest.values()),
                "recorded_failures": sum(
                    not record.get("ok", True) for record in latest.values()
                ),
            }
        )
    pid = state.get("pid")
    if not isinstance(pid, int):
        pid = _read_pid(root / "campaign.pid")
    return {
        "name": state.get("name"),
        "status": state.get("status"),
        "running": _campaign_alive(state, root),
        "pid": pid,
        "source_commit": state.get("source_commit"),
        "artifact_root": str(root),
        "log": state.get("log", str(root / "campaign.log")),
        "summary": str(root / "summary.json"),
        "stages": stages,
        "error": state.get("error"),
    }


def _campaign_alive(state: dict[str, Any], root: Path) -> bool:
    pid = state.get("pid")
    if not isinstance(pid, int):
        pid = _read_pid(root / "campaign.pid")
    child_pid = state.get("active_child_pid")
    return bool(
        (isinstance(pid, int) and _process_matches(pid, root))
        or (isinstance(child_pid, int) and _process_matches(child_pid, root))
    )


def _process_matches(pid: int, root: Path) -> bool:
    if not _pid_alive(pid):
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and str(root) in result.stdout


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                continue
            raise
        cell_id = record.get("cell_id")
        if isinstance(cell_id, str):
            latest[cell_id] = record
    return latest


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _config_sha256() -> str:
    path = Path.home() / ".stropha" / "config.yaml"
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        content = b"<missing>"
    except OSError as exc:
        raise CampaignError(f"could not fingerprint config {path}: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


def _interrupt_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _assert_config_unchanged(state: dict[str, Any]) -> None:
    expected = state.get("config_sha256")
    actual = _config_sha256()
    if expected != actual:
        raise CampaignError(
            "machine config changed during the campaign; start again to create a new run"
        )


def _require_expected_config() -> None:
    expected = os.environ.get("STROPHA_BENCH_EXPECTED_CONFIG_SHA256")
    if not expected or _config_sha256() != expected:
        raise CampaignError("machine config differs from the pinned campaign config")


@contextlib.contextmanager
def _exclusive_start(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a", encoding="ascii")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another campaign start is already in progress") from exc
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"could not read campaign state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"campaign state is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--spec", type=Path, required=True)
    worker.add_argument("--source-repo", type=Path, required=True)
    worker.add_argument("--snapshot", type=Path, required=True)
    worker.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    return _run_worker(
        spec_path=args.spec,
        source_repo=args.source_repo,
        snapshot_path=args.snapshot,
        artifact_root=args.artifact_root,
    )


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CampaignError",
    "campaign_status",
    "load_campaign_spec",
    "start_campaign",
    "stop_campaign",
]
