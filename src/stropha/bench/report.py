"""Analysis over a benchmark results JSONL.

Reads the append-only results store the runner writes (one JSON object per
cell) and produces:

- a **ranked table** of every cell by a chosen metric;
- **baseline deltas** with a paired bootstrap CI (when per-query ranks are
  stored), so a difference can be told from noise on a small golden set;
- **per-axis marginals** — the mean metric for each value of each axis,
  averaged over all other axes ("on average, what does flipping reranker to
  code-aware do?").
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import StrophaError

_BOOTSTRAP = 5000
_SEED = 1234


class ReportError(StrophaError):
    """Raised when results cannot be loaded or analysed."""


def load_results(path: Path | str) -> list[dict[str, Any]]:
    """Load the results JSONL. Skips blank lines; keeps only ``ok`` cells."""
    p = Path(path)
    if not p.is_file():
        raise ReportError(f"results file not found: {p}")
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok", True):
            out.append(rec)
    return out


# ----------------------------------------------------------------- metric math
def _hit(rank: Any) -> bool:
    return rank is not None


def _recall(ranks: dict[str, Any], ids: list[str]) -> float:
    return sum(_hit(ranks.get(i)) for i in ids) / len(ids) if ids else 0.0


def _mrr(ranks: dict[str, Any], ids: list[str]) -> float:
    if not ids:
        return 0.0
    return sum((1.0 / r if r else 0.0) for r in (ranks.get(i) for i in ids)) / len(ids)


def _bootstrap_delta(
    base: dict[str, Any], arm: dict[str, Any], metric: Any
) -> tuple[float, float, float, float]:
    """Paired bootstrap of ``metric(arm) - metric(base)`` over shared queries.

    Returns ``(point, lo95, hi95, p_gt_0)``.
    """
    ids = sorted(set(base) & set(arm))
    if not ids:
        return 0.0, 0.0, 0.0, 0.5
    rng = random.Random(_SEED)
    n = len(ids)
    deltas: list[float] = []
    for _ in range(_BOOTSTRAP):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        deltas.append(metric(arm, sample) - metric(base, sample))
    deltas.sort()
    point = metric(arm, ids) - metric(base, ids)
    lo = deltas[int(0.025 * _BOOTSTRAP)]
    hi = deltas[int(0.975 * _BOOTSTRAP)]
    p_gt = sum(1 for d in deltas if d > 0) / _BOOTSTRAP
    return point, lo, hi, p_gt


@dataclass
class BenchReport:
    """In-memory view over a list of result records."""

    results: list[dict[str, Any]]

    def metric_names(self) -> list[str]:
        keys: set[str] = set()
        for r in self.results:
            keys.update((r.get("metrics") or {}).keys())
        # Stable, useful order.
        order = ["recall@10", "mrr", "ndcg@10", "map@10", "precision@10",
                 "latency_p50_ms", "latency_p95_ms"]
        ranked = [k for k in order if k in keys]
        ranked += sorted(k for k in keys if k not in order)
        return ranked

    def rank_table(self, metric: str, *, descending: bool = True) -> list[dict[str, Any]]:
        rows = [r for r in self.results if metric in (r.get("metrics") or {})]
        rows.sort(key=lambda r: r["metrics"][metric], reverse=descending)
        return rows

    def marginals(self, metric: str) -> dict[str, dict[str, dict[str, float]]]:
        """``{axis: {value_label: {mean, n}}}`` — mean metric per axis value."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        axes: set[str] = set()
        for r in self.results:
            axes.update((r.get("values") or {}).keys())
        for axis in sorted(axes):
            buckets: dict[str, list[float]] = {}
            for r in self.results:
                m = (r.get("metrics") or {}).get(metric)
                if m is None or axis not in (r.get("values") or {}):
                    continue
                label = _vlabel(r["values"][axis])
                buckets.setdefault(label, []).append(float(m))
            out[axis] = {
                label: {"mean": sum(v) / len(v), "n": float(len(v))}
                for label, v in sorted(buckets.items())
            }
        return out

    def baseline_deltas(
        self, baseline_id: str
    ) -> list[dict[str, Any]]:
        """Per-cell paired bootstrap of recall@10 and mrr vs the baseline cell.

        Requires ``per_query`` ranks on both the baseline and the arm.
        """
        base = next((r for r in self.results if r.get("cell_id") == baseline_id), None)
        if base is None:
            raise ReportError(f"baseline cell_id {baseline_id!r} not in results")
        base_ranks = base.get("per_query") or {}
        out: list[dict[str, Any]] = []
        for r in self.results:
            if r.get("cell_id") == baseline_id:
                continue
            arm_ranks = r.get("per_query") or {}
            if not arm_ranks or not base_ranks:
                continue
            dR = _bootstrap_delta(base_ranks, arm_ranks, _recall)
            dM = _bootstrap_delta(base_ranks, arm_ranks, _mrr)
            out.append({
                "cell_id": r.get("cell_id"),
                "label": r.get("label", ""),
                "d_recall": dR,
                "d_mrr": dM,
            })
        return out


def _vlabel(v: Any) -> str:
    if isinstance(v, dict):
        return "/".join(str(x) for x in v.values())
    return str(v)


def to_csv(results: list[dict[str, Any]]) -> str:
    """Flatten results to CSV (axes + metrics columns)."""
    axes: list[str] = sorted({k for r in results for k in (r.get("values") or {})})
    metrics: list[str] = sorted({k for r in results for k in (r.get("metrics") or {})})
    header = ["cell_id", "indexing_sig", *axes, *metrics]
    lines = [",".join(header)]
    for r in results:
        vals = r.get("values") or {}
        met = r.get("metrics") or {}
        row = [str(r.get("cell_id", "")), str(r.get("indexing_sig", ""))]
        row += [_vlabel(vals.get(a, "")) for a in axes]
        row += [f"{met.get(m, '')}" for m in metrics]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


__all__ = ["BenchReport", "ReportError", "load_results", "to_csv"]
