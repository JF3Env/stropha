"""Benchmark runner: build each index once, sweep every query-time cell.

Flow per matrix:

1. Group cells by ``indexing_sig`` (shared embedder/enricher/chunker).
2. For each group, build ONE isolated sqlite-vec index of ``matrix.repo``
   (guarded: vecs == chunks, and non-noop enrichers actually enriched).
3. For each cell in the group, build retrieval with the cell's query-time
   overrides (reranker / candidate_k / k) and global feature overrides
   (HyDE / recursive / …), run the golden eval, append a result line.

Resumable: cells whose ``cell_id`` already appears in the output JSONL are
skipped. Isolated sqlite-vec indexes keep cells independent and trivially
disposable; retrieval logic is storage-agnostic so the rankings transfer to
the Postgres backend.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import get_config
from ..errors import StrophaError
from ..logging import get_logger
from .matrix import FEATURE_AXES, INDEXING_AXES, Cell, Matrix, MatrixError

log = get_logger(__name__)

# axis name -> RetrievalFeatures field (global, query-time)
_FEATURE_FIELD: dict[str, str] = {
    "hyde": "hyde_enabled",
    "hyde_code_tuned": "hyde_code_tuned",
    "recursive": "recursive_enabled",
    "recursive_adjacency": "recursive_adjacency",
    "query_rewrite": "query_rewrite_enabled",
    "query_decomposition": "query_decomposition_enabled",
    "multi_query": "multi_query_enabled",
    "multi_query_count": "multi_query_count",
    "cache_enabled": "cache_enabled",
    "adaptive_weighting": "adaptive_weighting",
}
# axis name -> retrieval.config key (pipeline, query-time)
_RETRIEVAL_FIELD: dict[str, str] = {
    "candidate_k": "candidate_k",
    "rrf_k": "k",
    "k": "k",
}


class BenchError(StrophaError):
    """Raised when a benchmark cell cannot be built or evaluated."""


# --------------------------------------------------------------- cell → config
def _adapter_override(value: Any, axis: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"adapter": value}
    if isinstance(value, dict):
        d = dict(value)
        adapter = d.pop("adapter", None)
        if not adapter:
            raise MatrixError(f"axis {axis!r} value missing 'adapter': {value}")
        return {"adapter": adapter, "config": d}
    raise MatrixError(f"axis {axis!r} value must be str or mapping, got {value!r}")


def _indexing_overrides(cell: Cell) -> dict[str, Any]:
    """Pipeline overrides for the index-building axes (embedder/enricher/...)."""
    ov: dict[str, Any] = {}
    for axis, value in cell.values.items():
        if axis not in INDEXING_AXES:
            continue
        if axis in ("embedder", "enricher", "chunker", "walker"):
            ov[axis] = _adapter_override(value, axis)
        else:
            raise MatrixError(
                f"indexing axis {axis!r} is not supported by the runner yet"
            )
    return ov


def _retrieval_overrides(cell: Cell) -> dict[str, Any]:
    """Pipeline ``retrieval.config`` overrides for query-time axes."""
    cfg: dict[str, Any] = {}
    for axis, value in cell.values.items():
        if axis == "reranker":
            cfg["reranker"] = None if value in (None, "null", "none") else _adapter_override(value, axis)
        elif axis in _RETRIEVAL_FIELD:
            cfg[_RETRIEVAL_FIELD[axis]] = value
    return {"retrieval": {"config": cfg}} if cfg else {}


def _feature_overrides(cell: Cell) -> dict[str, Any]:
    """Global ``retrieval_features`` field overrides for query-time axes."""
    out: dict[str, Any] = {}
    for axis, value in cell.values.items():
        if axis in FEATURE_AXES:
            field = _FEATURE_FIELD.get(axis)
            if field is None:
                raise MatrixError(f"feature axis {axis!r} has no known mapping")
            out[field] = value
    return out


def _validate_axes(matrix: Matrix) -> None:
    known = set(INDEXING_AXES) | set(FEATURE_AXES) | {"reranker"} | set(_RETRIEVAL_FIELD)
    unknown = [a for a in matrix.axes if a not in known]
    if unknown:
        raise MatrixError(
            f"unsupported axes {unknown}; known axes: {sorted(known)}"
        )


@contextlib.contextmanager
def _override_features(changes: dict[str, Any]):
    """Temporarily swap the cached global Config's retrieval_features."""
    import stropha.config as cfgmod

    base = cfgmod.get_config()
    if changes:
        try:
            new_rf = dataclasses.replace(base.retrieval_features, **changes)
        except TypeError as exc:
            raise MatrixError(f"bad feature override {changes}: {exc}") from exc
        new_cfg = dataclasses.replace(base, retrieval_features=new_rf)
    else:
        new_cfg = base
    prev = cfgmod._config_instance
    cfgmod._config_instance = new_cfg
    try:
        yield
    finally:
        cfgmod._config_instance = prev


_GRAPHIFY_PATCHED = False


def _patch_graphify_off() -> None:
    """Make indexing hermetic: never run/refresh the (large) graphify mirror."""
    global _GRAPHIFY_PATCHED
    if _GRAPHIFY_PATCHED:
        return
    from ..pipeline.pipeline import Pipeline

    Pipeline._refresh_graphify_mirror = lambda self: None  # type: ignore[assignment]
    Pipeline._run_graphify = lambda self: None  # type: ignore[assignment]
    _GRAPHIFY_PATCHED = True


# ------------------------------------------------------------------- indexing
def _verify_index(db_path: Path) -> dict[str, int]:
    import sqlite_vec  # type: ignore[import-untyped]  # local — optional dep boundary

    c = sqlite3.connect(db_path)
    try:
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        q = lambda s: c.execute(s).fetchone()[0]  # noqa: E731
        return {
            "chunks": q("select count(*) from chunks"),
            "vecs": q("select count(*) from vec_chunks"),
            "enriched": q(
                "select count(*) from chunks "
                "where embedding_text is not null and embedding_text != content"
            ),
        }
    finally:
        c.close()


def build_index(matrix: Matrix, cell: Cell, db_path: Path) -> dict[str, Any]:
    """Build the isolated sqlite-vec index for ``cell``'s indexing axes.

    Idempotent: if ``db_path`` already exists and passes the completeness
    guard, the build is skipped. Returns the verify summary.
    """
    from ..pipeline.builder import build_stages
    from ..pipeline.config import pipeline_config_from_dict
    from ..pipeline.pipeline import Pipeline

    _patch_graphify_off()

    if db_path.is_file():
        with contextlib.suppress(Exception):
            v = _verify_index(db_path)
            if v["chunks"] > 0 and v["vecs"] == v["chunks"]:
                log.info("bench.index_reuse", db=str(db_path), **v)
                return v
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    overrides = _indexing_overrides(cell)
    overrides["storage"] = {"adapter": "sqlite-vec", "config": {"path": str(db_path)}}
    cfg = pipeline_config_from_dict(overrides)
    built = build_stages(cfg, force_rebuild=True)
    inference = get_config().inference
    repo = Path(matrix.repo).resolve()
    t0 = time.perf_counter()
    with built.storage as storage:
        pipe = Pipeline(
            storage=storage,
            embedder=built.embedder,
            secondary_embedder=built.secondary_embedder,
            enricher=built.enricher,
            walker=built.walker,
            chunker=built.chunker,
            repos=[repo],
            graphify=False,
            enricher_concurrency=inference.enricher_concurrency,
            enricher_batch_size=inference.effective_batch_size(),
        )
        pipe.run(rebuild=True, mode="full")
    dt = time.perf_counter() - t0

    v = _verify_index(db_path)
    if v["chunks"] == 0 or v["vecs"] != v["chunks"]:
        raise BenchError(
            f"index incomplete for {cell.indexing_sig}: {v} (db={db_path})"
        )
    enr_axis = cell.values.get("enricher")
    if enr_axis not in (None, "noop") and v["enriched"] == 0:
        raise BenchError(
            f"enricher={enr_axis!r} applied to 0 chunks (no-op) for {cell.indexing_sig}"
        )
    log.info("bench.index_built", db=str(db_path), seconds=round(dt, 1), **v)
    return {**v, "index_seconds": round(dt, 1)}


# ---------------------------------------------------------------- evaluation
def eval_cell(
    matrix: Matrix, cell: Cell, db_path: Path, cases: Any
) -> dict[str, Any]:
    """Score one cell against ``cases`` using the index at ``db_path``."""
    from ..eval import run_eval
    from ..pipeline.builder import build_stages
    from ..pipeline.config import pipeline_config_from_dict

    overrides = _indexing_overrides(cell)  # embedder must match the index dim
    overrides["storage"] = {"adapter": "sqlite-vec", "config": {"path": str(db_path)}}
    _merge(overrides, _retrieval_overrides(cell))

    cfg = pipeline_config_from_dict(overrides)
    with _override_features(_feature_overrides(cell)):
        built = build_stages(cfg)
        with built.storage:
            report = run_eval(cases, built.retrieval, top_k=matrix.top_k)

    s = report.summary()
    per_query = {r.case.id: r.rank_of_first_hit for r in report.results}
    return {
        "ok": True,
        "cell_id": cell.cell_id,
        "indexing_sig": cell.indexing_sig,
        "label": cell.label(),
        "values": cell.values,
        "metrics": {
            k: s[k]
            for k in (
                f"recall@{matrix.top_k}", "mrr", f"ndcg@{matrix.top_k}",
                f"precision@{matrix.top_k}", f"map@{matrix.top_k}",
                "latency_p50_ms", "latency_p95_ms",
            )
            if k in s
        },
        "per_query": per_query,
        "n_queries": report.n,
        "ts": datetime.now(tz=UTC).isoformat(),
    }


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


# ------------------------------------------------------------------- driver
def run_matrix(
    matrix: Matrix,
    out_path: Path,
    *,
    index_dir: Path | None = None,
    resume: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
    on_cell: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the full matrix, appending one JSONL line per evaluated cell.

    Returns a summary dict ``{total, done, evaluated, failed, skipped}``.
    """
    _validate_axes(matrix)
    cells = matrix.expand()
    groups = matrix.groups()
    index_dir = index_dir or (out_path.parent / "indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if resume and out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec.get("cell_id", ""))

    total = len(cells)
    if dry_run:
        plan = {
            "total_cells": total,
            "unique_indexes": len(groups),
            "already_done": len(done & {c.cell_id for c in cells}),
            "groups": {sig: [c.label() for c in cs] for sig, cs in groups.items()},
        }
        return plan

    from ..eval import load_golden

    cases = load_golden(Path(matrix.golden))

    evaluated = failed = skipped = 0
    seen = 0
    for sig, group in groups.items():
        pending = [c for c in group if c.cell_id not in done]
        if not pending:
            skipped += len(group)
            continue
        db_path = index_dir / f"index_{sig}.db"
        try:
            build_index(matrix, pending[0], db_path)
        except Exception as exc:
            log.warning("bench.index_failed", indexing_sig=sig, error=str(exc))
            for cell in pending:
                rec = {"ok": False, "cell_id": cell.cell_id, "indexing_sig": sig,
                       "label": cell.label(), "values": cell.values,
                       "error": f"index: {exc}", "ts": datetime.now(tz=UTC).isoformat()}
                _append(out_path, rec)
                failed += 1
            continue
        for cell in group:
            if cell.cell_id in done:
                skipped += 1
                continue
            if limit is not None and evaluated >= limit:
                break
            seen += 1
            try:
                rec = eval_cell(matrix, cell, db_path, cases)
                evaluated += 1
            except Exception as exc:  # record + continue
                rec = {"ok": False, "cell_id": cell.cell_id, "indexing_sig": sig,
                       "label": cell.label(), "values": cell.values,
                       "error": str(exc), "ts": datetime.now(tz=UTC).isoformat()}
                failed += 1
            _append(out_path, rec)
            done.add(cell.cell_id)
            if on_cell is not None:
                on_cell(seen, total, rec)
        if limit is not None and evaluated >= limit:
            break

    return {
        "total": total,
        "evaluated": evaluated,
        "failed": failed,
        "skipped": skipped,
        "out": str(out_path),
    }


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


__all__ = ["BenchError", "build_index", "eval_cell", "run_matrix"]
