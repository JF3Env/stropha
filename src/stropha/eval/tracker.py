"""Eval score tracker — append-only JSONL log with regression detection.

WIN-382: Runs the golden set after every commit (from the hook's detached
background), appends a timestamped score entry to a JSONL log, and detects
regression by comparing the current Recall@K against the rolling mean of
the last N entries.

The log file defaults to ``~/.stropha/eval-scores.jsonl``. Each line is:
    {"ts": "...", "commit": "...", "recall": 0.63, "mrr": 0.27, ...}

Usage from CLI:
    stropha eval --track          # run + append + regression check
    stropha eval --track-log <p>  # custom log path
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)

DEFAULT_EVAL_LOG = Path("~/.stropha/eval-scores.jsonl").expanduser()
REGRESSION_WINDOW = 10  # Rolling window for regression detection
REGRESSION_THRESHOLD = 0.05  # Alert if Recall@K drops by more than 5pp


def _get_head_sha() -> str:
    """Get the current HEAD commit SHA (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def append_score(
    report: Any,
    *,
    log_path: Path | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Append eval scores to the JSONL log.

    Args:
        report: An EvalReport instance with metric properties.
        log_path: Path to the JSONL log file (default ~/.stropha/eval-scores.jsonl).
        commit: Git commit SHA (auto-detected if None).

    Returns:
        The score entry that was appended.
    """
    path = log_path or DEFAULT_EVAL_LOG
    path.parent.mkdir(parents=True, exist_ok=True)

    sha = commit or _get_head_sha()

    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": sha,
        "recall_at_k": round(report.recall_at_k, 4),
        "mrr": round(report.mrr, 4),
        "ndcg_at_k": round(report.ndcg_at_k, 4),
        "precision_at_k": round(report.precision_at_k, 4),
        "map_at_k": round(report.map_at_k, 4),
        "latency_p50_ms": round(report.latency_p50_ms, 1),
        "latency_p95_ms": round(report.latency_p95_ms, 1),
        "n_cases": report.n,
        "n_hits": report.n_hits,
        "top_k": report.top_k,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    log.info(
        "eval.tracker.appended",
        commit=sha,
        recall=entry["recall_at_k"],
        log_path=str(path),
    )
    return entry


def load_scores(log_path: Path | None = None) -> list[dict[str, Any]]:
    """Load all score entries from the JSONL log."""
    path = log_path or DEFAULT_EVAL_LOG
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def check_regression(
    current_recall: float,
    *,
    log_path: Path | None = None,
    window: int = REGRESSION_WINDOW,
    threshold: float = REGRESSION_THRESHOLD,
) -> tuple[bool, str]:
    """Check if current Recall@K is a regression vs the rolling mean.

    Args:
        current_recall: The Recall@K from the current eval run.
        log_path: Path to the JSONL log.
        window: Number of recent entries to average.
        threshold: How much of a drop (in absolute points) triggers a warning.

    Returns:
        (is_regression, message) — True if Recall@K dropped significantly.
    """
    scores = load_scores(log_path)

    if len(scores) < 2:
        return False, "not enough history for regression detection"

    # Take the last `window` entries (excluding the current one we just appended)
    recent = scores[-(window + 1):-1] if len(scores) > window else scores[:-1]
    if not recent:
        return False, "not enough history for regression detection"

    rolling_mean = sum(e.get("recall_at_k", 0.0) for e in recent) / len(recent)
    delta = current_recall - rolling_mean

    if delta < -threshold:
        msg = (
            f"REGRESSION: Recall@K dropped from rolling mean "
            f"{rolling_mean:.4f} to {current_recall:.4f} "
            f"(delta={delta:+.4f}, threshold={threshold})"
        )
        log.warning("eval.tracker.regression", msg=msg, delta=delta)
        return True, msg

    msg = (
        f"OK: Recall@K={current_recall:.4f} "
        f"(rolling mean={rolling_mean:.4f}, delta={delta:+.4f})"
    )
    log.info("eval.tracker.ok", recall=current_recall, mean=rolling_mean)
    return False, msg
