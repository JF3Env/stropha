"""Cost dashboard (Phase 3 §11.4).

Aggregates the structured-log trail produced by the indexer/server into
a per-adapter / per-repo / per-day summary. No external dependencies —
just walks the log file and builds buckets.

What we count:
- ``pipeline.repo_done`` events → files, chunks_seen, chunks_embedded
  per repo per run.
- ``local_embedder.ready`` / ``pipeline.adapter.built`` → adapter
  identities seen.
- Hook log refresh markers → number of background refresh cycles.
- ``graph_vec_loader.done`` → graph nodes embedded per run.

Because stropha is local-first there is no per-token money cost to
report. The "cost" is computed time: total embeddings, total chunks,
wall-clock estimates derived from event timestamps. That's enough to
spot regressions and budget large indexing runs.

Output formats:
- Rich table (default for terminal) via :func:`render_table`
- JSON via :func:`render_json` for programmatic consumption
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HOOK_LOG = Path.home() / ".cache" / "stropha-hook.log"

# Captures the "N iters" hint from the v=6 hook marker
# "[stropha-hook] meta loop (commit SHA, N iters)" so we can sum the
# attempted-iteration budget per loop launch. WIN-205.
_META_LOOP_LAUNCH_RE = re.compile(
    r"\[stropha-hook\] meta loop \(commit [^,]+, (\d+) iters\)"
)


# --------------------------------------------------------------------- model


@dataclass
class RepoCost:
    repo: str
    runs: int = 0
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    enricher_cache_hits: int = 0
    last_event: str | None = None  # ISO timestamp


@dataclass
class AdapterCost:
    adapter_id: str
    stage: str
    build_count: int = 0


@dataclass
class GraphCost:
    nodes_loaded: int = 0
    edges_loaded: int = 0
    edges_filtered: int = 0
    embed_runs: int = 0
    embedded_nodes: int = 0
    last_load: str | None = None


@dataclass
class HookCost:
    refresh_launched: int = 0
    refresh_done: int = 0
    refresh_failed: int = 0
    graphify_updates: int = 0


@dataclass
class SavingsCost:
    """Token savings from RAG search vs. naive grep+read."""
    
    total_searches: int = 0
    total_returned_tokens: int = 0
    total_baseline_tokens: int = 0
    total_tokens_saved: int = 0
    total_results_returned: int = 0
    
    @property
    def avg_savings_pct(self) -> float:
        """Average savings percentage across all searches."""
        if self.total_baseline_tokens <= 0:
            return 0.0
        return (self.total_tokens_saved / self.total_baseline_tokens) * 100
    
    @property
    def avg_results_per_search(self) -> float:
        """Average number of results per search."""
        if self.total_searches <= 0:
            return 0.0
        return self.total_results_returned / self.total_searches


@dataclass
class MetaDiskCost:
    """Per-run disk footprint of ``.stropha/meta/`` (WIN-166).

    Populated lazily — only when the caller actually points ``aggregate``
    at a meta directory via ``meta_dir=...``. Zero defaults so callers
    that don't care about Meta-Harness state still get a coherent dict.
    """

    total_bytes: int = 0
    traces_bytes: int = 0
    outcomes_bytes: int = 0
    candidates_bytes: int = 0
    meta_dir: str | None = None


@dataclass
class MetaLoopCost:
    """Sleep-time Meta-Harness loop activity (WIN-205).

    Aggregates two log streams:

    1. Plaintext hook markers (``[stropha-hook] meta loop``,
       ``meta done``, ``meta failed``) — counts how many post-commit
       bandit cycles fired and how many succeeded. Source: the hook
       log written by v=6 (WIN-203).
    2. structlog events from ``stropha.meta.loop`` — ``loop.iter_done``
       (per iteration), ``loop.done`` (per loop), ``loop.propose_failed``
       / ``loop.evaluate_failed``. Source: the same hook log because
       the meta step runs inside the nohup body.

    Each loop launch also emits an "N iters" hint that the parser
    extracts into ``iterations_total`` so the operator can correlate
    "how many candidates were attempted in the background" vs "how
    many made it into the archive" (``candidates_evaluated``).
    """

    loops_launched: int = 0
    loops_done: int = 0
    loops_failed: int = 0
    iterations_total: int = 0
    candidates_evaluated: int = 0
    propose_failures: int = 0
    evaluate_failures: int = 0
    last_event: str | None = None


@dataclass
class CostReport:
    repos: dict[str, RepoCost] = field(default_factory=dict)
    adapters: dict[str, AdapterCost] = field(default_factory=dict)
    graph: GraphCost = field(default_factory=GraphCost)
    hook: HookCost = field(default_factory=HookCost)
    savings: SavingsCost = field(default_factory=SavingsCost)
    meta_disk: MetaDiskCost = field(default_factory=MetaDiskCost)
    meta_loop: MetaLoopCost = field(default_factory=MetaLoopCost)
    sources_scanned: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        savings_dict = vars(self.savings).copy()
        savings_dict["avg_savings_pct"] = round(self.savings.avg_savings_pct, 1)
        savings_dict["avg_results_per_search"] = round(self.savings.avg_results_per_search, 1)
        return {
            "repos": {k: vars(v) for k, v in sorted(self.repos.items())},
            "adapters": {k: vars(v) for k, v in sorted(self.adapters.items())},
            "graph": vars(self.graph),
            "hook": vars(self.hook),
            "savings": savings_dict,
            "meta_disk": vars(self.meta_disk),
            "meta_loop": vars(self.meta_loop),
            "sources_scanned": self.sources_scanned,
        }


# --------------------------------------------------------------------- parser


def _iter_log_lines(paths: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    for p in paths:
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                yield p, line.strip()
        except OSError:
            continue


def _try_json(line: str) -> dict[str, Any] | None:
    if not line.startswith("{"):
        return None
    try:
        d = json.loads(line)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def aggregate(
    paths: Iterable[Path],
    *,
    meta_dir: Path | None = None,
) -> CostReport:
    """Walk the given log paths and return aggregated counters.

    Args:
        paths: log files to scan for counters.
        meta_dir: optional ``.stropha/meta/`` directory. When provided,
            the resulting :class:`CostReport.meta_disk` is populated with
            the per-run disk footprint (WIN-166).
    """
    report = CostReport()
    if meta_dir is not None and Path(meta_dir).is_dir():
        # Lazy import keeps the cost module independent of meta when
        # callers don't ask for disk reporting.
        from .meta.prune import disk_usage
        usage = disk_usage(meta_dir)
        report.meta_disk = MetaDiskCost(
            total_bytes=usage["total"],
            traces_bytes=usage["traces"],
            outcomes_bytes=usage["outcomes"],
            candidates_bytes=usage["candidates"],
            meta_dir=str(meta_dir),
        )
    for src, line in _iter_log_lines(paths):
        if str(src) not in report.sources_scanned:
            report.sources_scanned.append(str(src))

        # Hook log entries are plain text, not JSON. Match by substring
        # because the "launching refresh" line is the date-prefixed banner
        # (no `stropha-hook` marker), while the other lines use the
        # `[stropha-hook]` prefix from the hook script.
        if not line.startswith("{"):
            matched = False
            if "launching refresh" in line:
                report.hook.refresh_launched += 1
                matched = True
            elif "[stropha-hook] refresh done" in line:
                report.hook.refresh_done += 1
                matched = True
            elif "[stropha-hook] failed" in line or "[stropha-hook] graphify update failed" in line:
                report.hook.refresh_failed += 1
                matched = True
            if "graphify update (commit" in line:
                report.hook.graphify_updates += 1
                matched = True
            # v=6 sleep-time meta loop markers (WIN-203 → WIN-205).
            launch_match = _META_LOOP_LAUNCH_RE.search(line)
            if launch_match is not None:
                report.meta_loop.loops_launched += 1
                report.meta_loop.iterations_total += int(launch_match.group(1))
                matched = True
            elif "[stropha-hook] meta done" in line:
                report.meta_loop.loops_done += 1
                matched = True
            elif "[stropha-hook] meta failed" in line:
                report.meta_loop.loops_failed += 1
                matched = True
            if matched:
                continue

        d = _try_json(line)
        if d is None:
            continue
        event = d.get("event")
        if event == "pipeline.repo_done":
            repo = str(d.get("repo") or "(unknown)")
            rc = report.repos.setdefault(repo, RepoCost(repo=repo))
            rc.runs += 1
            rc.files_visited += int(d.get("files") or 0)
            rc.chunks_seen += int(d.get("chunks") or 0)
            rc.chunks_embedded += int(d.get("embedded") or 0)
            rc.chunks_reused += int(d.get("reused") or 0)
            rc.enricher_cache_hits += int(d.get("enricher_cache_hits") or 0)
            ts = d.get("timestamp")
            if ts and (rc.last_event is None or ts > rc.last_event):
                rc.last_event = ts
        elif event == "pipeline.adapter.built":
            aid = str(d.get("adapter_id") or "?")
            stage = str(d.get("stage") or "?")
            ac = report.adapters.setdefault(aid, AdapterCost(adapter_id=aid, stage=stage))
            ac.build_count += 1
        elif event == "graphify_loader.done":
            g = report.graph
            g.nodes_loaded += int(d.get("nodes") or 0)
            g.edges_loaded += int(d.get("edges_loaded") or 0)
            g.edges_filtered += int(d.get("edges_filtered") or 0)
            ts = d.get("timestamp")
            if ts and (g.last_load is None or ts > g.last_load):
                g.last_load = ts
        elif event == "graph_vec_loader.done":
            report.graph.embed_runs += 1
            report.graph.embedded_nodes += int(d.get("embedded") or 0)
        elif event == "search.served":
            s = report.savings
            s.total_searches += 1
            s.total_returned_tokens += int(d.get("returned_tokens") or 0)
            s.total_baseline_tokens += int(d.get("baseline_tokens") or 0)
            s.total_tokens_saved += int(d.get("tokens_saved") or 0)
            s.total_results_returned += int(d.get("results_count") or 0)
        elif event == "loop.iter_done":
            # Per-iteration success — counts each candidate that made it
            # into the archive. Bound below by the structlog stream
            # available in the hook log (which captures stdout+stderr
            # of the nohup body).
            report.meta_loop.candidates_evaluated += 1
            ts = d.get("timestamp")
            if ts and (
                report.meta_loop.last_event is None
                or ts > report.meta_loop.last_event
            ):
                report.meta_loop.last_event = ts
        elif event == "loop.seed_done":
            # Seed iterations (the warm-start before the proposer takes
            # over) also produce real evaluated candidates — counting
            # them keeps the candidates_evaluated tally honest.
            report.meta_loop.candidates_evaluated += 1
            ts = d.get("timestamp")
            if ts and (
                report.meta_loop.last_event is None
                or ts > report.meta_loop.last_event
            ):
                report.meta_loop.last_event = ts
        elif event == "loop.propose_failed":
            report.meta_loop.propose_failures += 1
        elif event == "loop.evaluate_failed":
            report.meta_loop.evaluate_failures += 1
        elif event == "loop.done":
            ts = d.get("timestamp")
            if ts and (
                report.meta_loop.last_event is None
                or ts > report.meta_loop.last_event
            ):
                report.meta_loop.last_event = ts
    return report


def default_log_paths() -> list[Path]:
    """Standard locations stropha writes structured logs to.

    Honours ``STROPHA_HOOK_LOG`` env var for the post-commit hook log. We
    also scan ``~/.cache/stropha-hook-*.log`` to pick up per-repo trails
    created by the v=3 cross-repo install.
    """
    main = Path(os.environ.get("STROPHA_HOOK_LOG", str(DEFAULT_HOOK_LOG)))
    candidates: list[Path] = []
    if main.is_file():
        candidates.append(main)
    cache_dir = main.parent
    if cache_dir.is_dir():
        for p in sorted(cache_dir.glob("stropha-hook-*.log")):
            if p not in candidates:
                candidates.append(p)
    return candidates


# --------------------------------------------------------------------- render


def render_table(report: CostReport):
    """Build a :class:`rich.console.RenderableType` table summary."""
    from rich.console import Group
    from rich.table import Table

    renders = []

    if report.repos:
        repos_t = Table(title="Per-repo activity", show_lines=False)
        repos_t.add_column("Repo", style="cyan", overflow="fold")
        repos_t.add_column("Runs", justify="right")
        repos_t.add_column("Files", justify="right")
        repos_t.add_column("Chunks", justify="right")
        repos_t.add_column("Embedded", justify="right")
        repos_t.add_column("Reused", justify="right")
        repos_t.add_column("Enricher cache", justify="right")
        repos_t.add_column("Last", overflow="fold")
        for rc in sorted(report.repos.values(), key=lambda r: -r.chunks_embedded):
            repos_t.add_row(
                rc.repo, str(rc.runs), str(rc.files_visited), str(rc.chunks_seen),
                str(rc.chunks_embedded), str(rc.chunks_reused),
                str(rc.enricher_cache_hits), rc.last_event or "—",
            )
        renders.append(repos_t)

    if report.adapters:
        adp = Table(title="Adapters seen", show_lines=False)
        adp.add_column("Stage", style="cyan")
        adp.add_column("Adapter ID", overflow="fold")
        adp.add_column("Builds", justify="right")
        for ac in sorted(
            report.adapters.values(), key=lambda a: (a.stage, -a.build_count),
        ):
            adp.add_row(ac.stage, ac.adapter_id, str(ac.build_count))
        renders.append(adp)

    if report.graph.nodes_loaded or report.graph.embed_runs:
        gt = Table(title="Graphify activity", show_lines=False)
        gt.add_column("Field", style="cyan")
        gt.add_column("Value")
        gt.add_row("Total nodes loaded (cumulative)", str(report.graph.nodes_loaded))
        gt.add_row("EXTRACTED edges loaded", str(report.graph.edges_loaded))
        gt.add_row("INFERRED edges filtered", str(report.graph.edges_filtered))
        gt.add_row("Vec-embed runs", str(report.graph.embed_runs))
        gt.add_row("Total nodes embedded", str(report.graph.embedded_nodes))
        gt.add_row("Last load", report.graph.last_load or "—")
        renders.append(gt)

    if report.hook.refresh_launched or report.hook.refresh_done or report.hook.refresh_failed:
        ht = Table(title="Post-commit hook", show_lines=False)
        ht.add_column("Event", style="cyan")
        ht.add_column("Count", justify="right")
        ht.add_row("Refresh launched", str(report.hook.refresh_launched))
        ht.add_row("Refresh done", str(report.hook.refresh_done))
        ht.add_row("Refresh failed", str(report.hook.refresh_failed))
        ht.add_row("graphify update invocations", str(report.hook.graphify_updates))
        renders.append(ht)

    if report.savings.total_searches > 0:
        st = Table(title="Token savings (RAG vs. naive grep+read)", show_lines=False)
        st.add_column("Metric", style="cyan")
        st.add_column("Value", justify="right")
        st.add_row("Total searches", str(report.savings.total_searches))
        st.add_row("Total results returned", str(report.savings.total_results_returned))
        st.add_row("Avg results/search", f"{report.savings.avg_results_per_search:.1f}")
        st.add_row("Tokens returned (total)", f"{report.savings.total_returned_tokens:,}")
        st.add_row("Baseline tokens (estimated)", f"{report.savings.total_baseline_tokens:,}")
        st.add_row("Tokens saved", f"{report.savings.total_tokens_saved:,}")
        st.add_row("Avg savings %", f"{report.savings.avg_savings_pct:.1f}%")
        renders.append(st)

    m = report.meta_loop
    if (
        m.loops_launched
        or m.candidates_evaluated
        or m.loops_done
        or m.loops_failed
    ):
        mt = Table(
            title="Meta-Harness sleep-time loop (WIN-203/204/205)",
            show_lines=False,
        )
        mt.add_column("Metric", style="cyan")
        mt.add_column("Value", justify="right")
        mt.add_row("Loops launched", str(m.loops_launched))
        mt.add_row("Loops done", str(m.loops_done))
        mt.add_row(
            "Loops failed",
            f"[red]{m.loops_failed}[/red]" if m.loops_failed else "0",
        )
        mt.add_row("Iterations requested (sum)", str(m.iterations_total))
        mt.add_row("Candidates evaluated", str(m.candidates_evaluated))
        mt.add_row(
            "Propose failures",
            f"[yellow]{m.propose_failures}[/yellow]"
            if m.propose_failures
            else "0",
        )
        mt.add_row(
            "Evaluate failures",
            f"[yellow]{m.evaluate_failures}[/yellow]"
            if m.evaluate_failures
            else "0",
        )
        mt.add_row("Last meta event", m.last_event or "—")
        renders.append(mt)

    if not renders:
        return None
    return Group(*renders)


def render_json(report: CostReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
