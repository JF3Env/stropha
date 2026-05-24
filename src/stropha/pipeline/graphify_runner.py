"""Run ``graphify`` as a subprocess to produce / refresh ``graph.json``.

The pipeline calls :func:`run_graphify` **before** loading the graph
mirror so the ``find_callers`` / ``find_related`` / ``trace_feature``
MCP tools always have fresh structural data — even on the very first
``stropha index`` run when no ``graphify-out/`` directory exists yet.

Design constraints:

* **Graceful degradation** — when the ``graphify`` binary is not
  installed the function returns ``GraphifyResult(skipped=True)`` and
  the pipeline proceeds without a graph.  No hard dependency.
* **Idempotent** — ``graphify update <repo>`` handles both first-run
  (creates ``graphify-out/``) and incremental (diffs the AST) cases.
* **Timeout-guarded** — a ``timeout`` kwarg caps wall-clock time so a
  stuck process never blocks ``stropha index`` forever.
* **No clustering on incremental** — passes ``--no-cluster`` when
  ``graph.json`` already exists to keep the fast path sub-second.
  Full initial runs include clustering (produces community labels for
  the ``graph-aware`` enricher + ``get_community`` tool).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

log = get_logger(__name__)

# Default wall-clock timeout for a single graphify invocation (seconds).
# ``graphify update`` is typically 1-5s incremental; a full first-run on
# a large repo (~50k LOC) can take 30-60s.  120s leaves generous margin.
DEFAULT_TIMEOUT_S = 120


@dataclass
class GraphifyResult:
    """Outcome of a single :func:`run_graphify` call."""

    repo: Path
    skipped: bool = False
    """True when graphify was not found or the caller disabled the step."""
    created: bool = False
    """True when ``graph.json`` did not exist before and was created."""
    updated: bool = False
    """True when an existing ``graph.json`` was refreshed."""
    error: str | None = None
    """Non-None when graphify exited non-zero or timed out."""
    elapsed_s: float = 0.0
    """Wall-clock seconds spent in the subprocess."""


def find_graphify() -> str | None:
    """Locate the ``graphify`` binary.

    Resolution order:
    1. ``STROPHA_HOOK_GRAPHIFY`` env var (explicit user override).
    2. ``shutil.which("graphify")`` (standard PATH lookup).

    Returns the absolute path string, or ``None`` when not found.
    """
    override = os.environ.get("STROPHA_HOOK_GRAPHIFY", "").strip()
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    return shutil.which("graphify")


def run_graphify(
    repo: Path,
    *,
    graphify_bin: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    no_cluster: bool | None = None,
) -> GraphifyResult:
    """Run ``graphify update <repo>`` as a subprocess.

    Args:
        repo: Path to the git repository root.
        graphify_bin: Explicit path to the ``graphify`` binary.
            Auto-detected via :func:`find_graphify` when ``None``.
        timeout_s: Maximum wall-clock seconds before the subprocess is
            killed.
        no_cluster: Force ``--no-cluster`` (``True``) or full clustering
            (``False``).  When ``None`` (default) the function decides
            automatically: ``--no-cluster`` when ``graph.json`` already
            exists (fast incremental path), full clustering on first run
            (so community labels are available for the enricher).
    """
    import time

    result = GraphifyResult(repo=repo)

    # 1. Resolve binary.
    binary = graphify_bin or find_graphify()
    if binary is None:
        result.skipped = True
        log.info(
            "graphify_runner.skipped",
            reason="graphify binary not found on PATH",
            repo=str(repo),
        )
        return result

    # 2. Decide clustering strategy.
    graph_json = repo / "graphify-out" / "graph.json"
    had_graph = graph_json.is_file()

    if no_cluster is None:
        # Incremental: skip clustering (fast).  First run: cluster.
        use_no_cluster = had_graph
    else:
        use_no_cluster = no_cluster

    # 3. Build command.
    cmd: list[str] = [binary, "update", str(repo)]
    if use_no_cluster:
        cmd.append("--no-cluster")

    log.info(
        "graphify_runner.start",
        cmd=" ".join(cmd),
        repo=str(repo),
        had_graph=had_graph,
    )

    # 4. Run.
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(repo),
        )
        result.elapsed_s = time.monotonic() - t0

        if proc.returncode != 0:
            result.error = (proc.stderr or proc.stdout or "").strip()
            log.warning(
                "graphify_runner.failed",
                returncode=proc.returncode,
                error=result.error,
                elapsed_s=round(result.elapsed_s, 2),
                repo=str(repo),
            )
            return result

        # Success.
        if had_graph:
            result.updated = True
        else:
            result.created = graph_json.is_file()
            if not result.created:
                # graphify succeeded but didn't produce graph.json — unusual
                # but possible for repos with no parseable code.
                result.skipped = True
                log.info(
                    "graphify_runner.no_output",
                    reason="graphify exited 0 but graph.json was not created",
                    repo=str(repo),
                )
                return result

        log.info(
            "graphify_runner.done",
            action="created" if result.created else "updated",
            elapsed_s=round(result.elapsed_s, 2),
            repo=str(repo),
        )
        return result

    except subprocess.TimeoutExpired:
        result.elapsed_s = time.monotonic() - t0
        result.error = f"timed out after {timeout_s}s"
        log.warning(
            "graphify_runner.timeout",
            timeout_s=timeout_s,
            repo=str(repo),
        )
        return result
    except OSError as exc:
        result.elapsed_s = time.monotonic() - t0
        result.error = str(exc)
        log.warning(
            "graphify_runner.os_error",
            error=str(exc),
            repo=str(repo),
        )
        return result


def run_graphify_multi(
    repos: list[Path],
    *,
    graphify_bin: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> list[GraphifyResult]:
    """Run graphify for each repo sequentially.

    Binary resolution is done once and shared across all repos.
    """
    binary = graphify_bin or find_graphify()
    if binary is None:
        log.info(
            "graphify_runner.skipped_all",
            reason="graphify binary not found on PATH",
            repo_count=len(repos),
        )
        return [GraphifyResult(repo=r, skipped=True) for r in repos]

    return [
        run_graphify(repo, graphify_bin=binary, timeout_s=timeout_s)
        for repo in repos
    ]
