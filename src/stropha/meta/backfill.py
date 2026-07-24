"""Outcome backfill — post-commit labeling of the trace stream (WIN-164).

The post-commit hook v=5 runs this in its detached ``nohup`` background
process so ``git commit`` is never blocked by the labeler. Per CLAUDE.md
§3.5 the hook must exit the foreground in <100 ms; the heavy lifting
(walking traces, computing labels, writing outcomes) lives here.

The CLI surface ``stropha meta outcome backfill --since <SHA>`` is the
operator-facing entry point. Internally it:

1. Computes the diff between ``<SHA>`` and ``HEAD``: changed paths +
   heuristic symbol extraction.
2. Iterates every JSONL trace under the meta traces directory.
3. Labels every hit in every trace with :func:`stropha.meta.outcome.label_hits`.
4. Aggregates per-trace + per-session + per-tier scores and token efficiency.
5. Fits regularized per-repository RRF stream weights from attributed hits.
6. Writes a single JSON summary to
   ``<meta>/outcomes/<HEAD_sha>.json``.

The summary is what the Phase 1 evaluator (WIN-167) consumes to compute
the outcome term of the fitness function.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import StrophaError
from ..ingest.git_meta import detect as detect_repo
from ..logging import get_logger
from .feedback import BadResultStore, resolve_bad_chunks
from .outcome import (
    LabelingContext,
    Tier,
    TierLabel,
    aggregate_score,
    label_hits,
    tier_summary,
)
from .stream_weights import fit_stream_weights, write_stream_weight_profile

log = get_logger(__name__)


class BackfillError(StrophaError):
    """Raised when ``backfill`` cannot satisfy a request."""


# Heuristic symbol extractor — catches the common identifier shapes the
# diff is likely to contain. Conservative on purpose: false negatives at
# T2 fall through to T3 (graph hops) or T7 (neutral); false positives
# would mislabel unrelated chunks as positive.
_DIFF_SYMBOL_RE = re.compile(
    r"""
    \b(?:
        def\s+(\w+)            # python function
      | class\s+(\w+)          # python / generic class
      | function\s+(\w+)       # js / ts function
      | (?:public|private|protected)\s+\w+\s+(\w+)\s*\(   # java method
    )
    """,
    re.VERBOSE,
)


@dataclass
class BackfillSummary:
    """One backfill run's result."""

    since_sha: str
    head_sha: str
    diff_paths: list[str] = field(default_factory=list)
    diff_symbols: list[str] = field(default_factory=list)
    traces_scanned: int = 0
    hits_labeled: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    per_trace_scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    tasks_observed: int = 0
    solved_tasks: int = 0
    total_returned_tokens: int = 0
    returned_tokens_per_solved_task: float | None = None
    stream_weights_repo: str | None = None
    stream_weights: dict[str, float] = field(default_factory=dict)
    stream_weight_observations: dict[str, int] = field(default_factory=dict)
    stream_weights_path: str | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "since_sha": self.since_sha,
            "head_sha": self.head_sha,
            "diff_paths": self.diff_paths,
            "diff_symbols": self.diff_symbols,
            "traces_scanned": self.traces_scanned,
            "hits_labeled": self.hits_labeled,
            "tier_counts": self.tier_counts,
            "per_trace_scores": self.per_trace_scores,
            "overall_score": self.overall_score,
            "tasks_observed": self.tasks_observed,
            "solved_tasks": self.solved_tasks,
            "total_returned_tokens": self.total_returned_tokens,
            "returned_tokens_per_solved_task": self.returned_tokens_per_solved_task,
            "stream_weights_repo": self.stream_weights_repo,
            "stream_weights": self.stream_weights,
            "stream_weight_observations": self.stream_weight_observations,
            "stream_weights_path": self.stream_weights_path,
            "output_path": self.output_path,
            "generated_at": datetime.now(tz=UTC).isoformat(),
        }


# --------------------------------------------------------------------------- #
# Git helpers                                                                 #
# --------------------------------------------------------------------------- #


def _run_git(args: list[str], cwd: Path) -> str:
    """Run ``git ARGS`` and return stdout. Raises ``BackfillError`` on non-zero.

    Subprocess is wrapped here so tests can monkeypatch one entry point.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BackfillError("git binary not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise BackfillError(
            f"git {' '.join(args)} failed (rc={exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    return result.stdout


def diff_paths(since_sha: str, head_sha: str, cwd: Path) -> list[str]:
    """Repo-relative paths that changed between ``since_sha`` and ``head_sha``."""
    raw = _run_git(
        ["diff", "--name-only", f"{since_sha}..{head_sha}"],
        cwd=cwd,
    )
    return [line.strip() for line in raw.splitlines() if line.strip()]


def diff_symbols(since_sha: str, head_sha: str, cwd: Path) -> list[str]:
    """Heuristic — names of functions / classes appearing in the diff.

    Only catches the unambiguous shapes (``def NAME``, ``class NAME``,
    ``function NAME``, ``public TYPE NAME(``). The labeler tolerates an
    incomplete symbol set (graph hops at T3 backfill the rest).
    """
    raw = _run_git(["diff", f"{since_sha}..{head_sha}"], cwd=cwd)
    out: set[str] = set()
    for match in _DIFF_SYMBOL_RE.finditer(raw):
        for group in match.groups():
            if group:
                out.add(group)
    return sorted(out)


def head_sha(cwd: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=cwd).strip()


# --------------------------------------------------------------------------- #
# Trace IO                                                                    #
# --------------------------------------------------------------------------- #


def load_traces(traces_dir: Path) -> list[dict[str, Any]]:
    """Read every JSONL trace under ``traces_dir`` (recursive, sorted)."""
    if not traces_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        with contextlib.suppress(OSError), open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    log.warning(
                        "backfill.bad_trace_line",
                        path=str(path),
                        line=raw[:120],
                    )
    return records


def _trace_task_key(trace: dict[str, Any], index: int) -> str:
    """Use session identity as the task key, with stable legacy fallbacks."""
    session_id = str(trace.get("session_id") or "").strip()
    if session_id:
        return f"session:{session_id}"
    trace_id = str(trace.get("trace_id") or "").strip()
    if trace_id:
        return f"trace:{trace_id}"
    return f"legacy:{index}"


def _trace_returned_tokens(trace: dict[str, Any]) -> int:
    """Read the trace-level count, falling back to legacy result snippets."""
    try:
        if "returned_tokens" in trace:
            return max(0, int(trace["returned_tokens"]))
    except (TypeError, ValueError):
        pass

    total = 0
    for result in trace.get("results") or []:
        if not isinstance(result, dict):
            continue
        try:
            if "returned_tokens" in result:
                total += max(0, int(result["returned_tokens"]))
                continue
        except (TypeError, ValueError):
            pass
        snippet = result.get("snippet")
        if isinstance(snippet, str):
            total += len(snippet) // 4
    return total


# --------------------------------------------------------------------------- #
# Backfill entry point                                                        #
# --------------------------------------------------------------------------- #


def run_backfill(
    *,
    since_sha: str,
    meta_dir: Path,
    repo_dir: Path,
    head: str | None = None,
) -> BackfillSummary:
    """Label every trace under ``meta_dir/traces`` against the diff and persist.

    Args:
        since_sha: previous commit. The diff is computed `since_sha..HEAD`.
        meta_dir: where ``traces/`` lives + where the outcome JSON is
            written (under ``outcomes/<head_sha>.json``).
        repo_dir: working tree to run ``git`` in.
        head: HEAD SHA. Computed when not provided.

    Returns the :class:`BackfillSummary`.
    """
    head_resolved = head or head_sha(repo_dir)

    paths = diff_paths(since_sha, head_resolved, repo_dir)
    symbols = diff_symbols(since_sha, head_resolved, repo_dir)

    traces_dir = meta_dir / "traces"
    traces = load_traces(traces_dir)

    # Resolve bad results once.
    bad_chunks: set[str] = set()
    try:
        store = BadResultStore(meta_dir)
        bad_trace_ids = store.trace_ids()
        if bad_trace_ids:
            bad_chunks = resolve_bad_chunks(
                bad_trace_ids,
                sorted(traces_dir.glob("*.jsonl")),
            )
    except OSError as exc:
        # Feedback channel missing is not fatal — T6 just won't fire.
        log.debug("backfill.bad_results_unavailable", error=str(exc))

    ctx = LabelingContext(
        diff_paths=set(paths),
        diff_symbols=set(symbols),
        session_traces=traces,
        bad_results=bad_chunks,
    )

    all_labels: list[TierLabel] = []
    labeled_hits: list[dict[str, Any]] = []
    per_trace: dict[str, float] = {}
    task_keys: set[str] = set()
    solved_task_keys: set[str] = set()
    total_returned_tokens = 0
    for index, trace in enumerate(traces):
        task_key = _trace_task_key(trace, index)
        task_keys.add(task_key)
        total_returned_tokens += _trace_returned_tokens(trace)
        hits = [
            hit for hit in (trace.get("results") or [])
            if isinstance(hit, dict)
        ]
        if not hits:
            continue
        labels = label_hits(hits, ctx)
        labeled_hits.extend(hits)
        all_labels.extend(labels)
        if any(label.score > 0 for label in labels):
            solved_task_keys.add(task_key)
        trace_id = str(trace.get("trace_id") or "")
        if trace_id:
            per_trace[trace_id] = aggregate_score(labels)

    repo_key = detect_repo(repo_dir).normalized_key
    profile = fit_stream_weights(
        labeled_hits,
        all_labels,
        repo_key=repo_key,
    )
    stream_weights_path: str | None = None
    if profile is not None:
        try:
            path = write_stream_weight_profile(meta_dir / "stream-weights", profile)
            stream_weights_path = str(path)
        except OSError as exc:
            log.warning(
                "backfill.stream_weights_write_failed",
                repo=repo_key,
                error=str(exc),
            )

    counts = tier_summary(all_labels)
    summary = BackfillSummary(
        since_sha=since_sha,
        head_sha=head_resolved,
        diff_paths=paths,
        diff_symbols=symbols,
        traces_scanned=len(traces),
        hits_labeled=len(all_labels),
        tier_counts={t.value: c for t, c in counts.items()},
        per_trace_scores=per_trace,
        overall_score=aggregate_score(all_labels),
        tasks_observed=len(task_keys),
        solved_tasks=len(solved_task_keys),
        total_returned_tokens=total_returned_tokens,
        returned_tokens_per_solved_task=(
            total_returned_tokens / len(solved_task_keys)
            if solved_task_keys
            else None
        ),
        stream_weights_repo=repo_key,
        stream_weights=dict(profile.weights) if profile is not None else {},
        stream_weight_observations=(
            dict(profile.observations) if profile is not None else {}
        ),
        stream_weights_path=stream_weights_path,
    )

    out_dir = meta_dir / "outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{head_resolved}.json"
    summary.output_path = str(out_path)
    payload = summary.to_dict()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info(
        "backfill.done",
        since=since_sha,
        head=head_resolved,
        traces=summary.traces_scanned,
        hits=summary.hits_labeled,
        overall=summary.overall_score,
        tasks_observed=summary.tasks_observed,
        solved_tasks=summary.solved_tasks,
        total_returned_tokens=summary.total_returned_tokens,
        returned_tokens_per_solved_task=summary.returned_tokens_per_solved_task,
        stream_weights_repo=summary.stream_weights_repo,
        stream_weights=summary.stream_weights,
        stream_weight_observations=summary.stream_weight_observations,
        stream_weights_path=summary.stream_weights_path,
        output=summary.output_path,
    )
    return summary


# --------------------------------------------------------------------------- #
# Helpers exposed for tests                                                   #
# --------------------------------------------------------------------------- #


def _extract_symbols_from_text(text: str) -> set[str]:
    """Public-ish hook so tests can probe the regex without touching git."""
    out: set[str] = set()
    for match in _DIFF_SYMBOL_RE.finditer(text):
        for group in match.groups():
            if group:
                out.add(group)
    return out


__all__ = [
    "BackfillError",
    "BackfillSummary",
    "_extract_symbols_from_text",
    "diff_paths",
    "diff_symbols",
    "head_sha",
    "load_traces",
    "run_backfill",
]


# Re-exported so callers can iterate the same Tier constants the labeler does.
_TIERS: tuple[Tier, ...] = tuple(Tier)
