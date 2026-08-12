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
import hashlib
import json
import sqlite3
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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
    "rrf_k": "rrf_k",
    "k": "rrf_k",
    "recursive": "recursive_enabled",
    "recursive_adjacency": "recursive_adjacency",
    "adaptive_weighting": "adaptive_weighting",
    "learned_weights": "learned_weights_enabled",
}


class BenchError(StrophaError):
    """Raised when a benchmark cell cannot be built or evaluated."""


BenchSplit = Literal["search", "holdout"]


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

    def _skip_refresh(_pipeline: Pipeline) -> None:
        return None

    def _skip_update(_pipeline: Pipeline, _repo_stats: Any) -> None:
        return None

    Pipeline._refresh_graphify_mirror = _skip_refresh  # type: ignore[assignment]
    Pipeline._run_graphify = _skip_update  # type: ignore[assignment]
    _GRAPHIFY_PATCHED = True


# ------------------------------------------------------------------- indexing
def _verify_index(db_path: Path) -> dict[str, int | bool]:
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
            "complete": bool(
                q("select count(*) from meta where key='bench_index_complete' and value='1'")
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
            if v["complete"] and v["chunks"] > 0 and v["vecs"] == v["chunks"]:
                log.info("bench.index_reuse", db=str(db_path), **v)
                return v
    resume_partial = db_path.is_file()

    overrides = _indexing_overrides(cell)
    overrides["storage"] = {"adapter": "sqlite-vec", "config": {"path": str(db_path)}}
    cfg = pipeline_config_from_dict(overrides)
    built = build_stages(cfg, force_rebuild=not resume_partial)
    inference = get_config().inference
    repo = Path(matrix.repo).resolve()
    t0 = time.perf_counter()
    enrichment_failures = 0
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
        stats = pipe.run(
            rebuild=not resume_partial,
            mode="reconcile" if resume_partial else "full",
        )
        enrichment_failures = stats.chunks_enriched_failed
        if not enrichment_failures:
            storage.set_meta("bench_index_complete", "1")
            storage.commit()
    if enrichment_failures:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        raise BenchError(f"index enrichment failed for {enrichment_failures} chunks")
    dt = time.perf_counter() - t0

    v = _verify_index(db_path)
    if not v["complete"] or v["chunks"] == 0 or v["vecs"] != v["chunks"]:
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
    matrix: Matrix,
    cell: Cell,
    db_path: Path,
    cases: Any,
    *,
    run_provenance: dict[str, Any] | None = None,
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
            get_meta = getattr(built.storage, "get_meta", None)
            indexed_enricher = (
                get_meta("active_enricher_id") if callable(get_meta) else None
            )
            indexed_embedder = (
                get_meta("active_embedder_id") if callable(get_meta) else None
            )
            active_enricher = built.effective_enricher_id()
            active_embedder = built.embedder.adapter_id
            drifted = bool(
                indexed_enricher != active_enricher
                or indexed_embedder != active_embedder
            )
            if drifted:
                raise BenchError(
                    "benchmark index identity drift: "
                    f"enricher indexed={indexed_enricher!r} active={active_enricher!r}; "
                    f"embedder indexed={indexed_embedder!r} active={active_embedder!r}"
                )
            provenance = {
                **(run_provenance or {}),
                "pipeline": {
                    "walker": built.walker.adapter_id,
                    "chunker": built.chunker.adapter_id,
                    "enricher": active_enricher,
                    "embedder": active_embedder,
                    "storage": built.storage.adapter_id,
                    "retrieval": built.retrieval.adapter_id,
                },
                "index": {
                    "indexing_sig": cell.indexing_sig,
                    "indexed_enricher": indexed_enricher,
                    "indexed_embedder": indexed_embedder,
                    "config_drift": False,
                    "comparable": True,
                },
            }
            report = run_eval(
                cases,
                built.retrieval,
                top_k=matrix.top_k,
                provenance=provenance,
            )

    s = report.summary()
    reranker = report.diagnostics.get("reranker")
    if isinstance(reranker, dict):
        for key in (
            "primary_latency_p95_ms",
            "repair_latency_p95_ms",
            "latency_ms",
        ):
            value = reranker.get(key)
            if isinstance(value, (int, float)):
                s[f"reranker_{key}"] = value
    per_query = {r.case.id: r.rank_of_first_hit for r in report.results}
    by_tag = {tag: tagged.summary() for tag, tagged in report.by_tag().items()}
    return {
        "ok": True,
        "cell_id": cell.cell_id,
        "indexing_sig": cell.indexing_sig,
        "label": cell.label(),
        "values": cell.values,
        "metrics": dict(s),
        "provenance": provenance,
        "diagnostics": report.diagnostics,
        "per_query": per_query,
        "by_tag": by_tag,
        "n_queries": report.n,
        "ts": datetime.now(tz=UTC).isoformat(),
    }


def _run_provenance(
    matrix: Matrix,
    split: BenchSplit,
    runtime_id: str | None = None,
) -> dict[str, Any]:
    """Build the immutable identity shared by every result in one matrix run."""
    from ..ingest.git_meta import detect

    repo = Path(matrix.repo).expanduser().resolve()
    golden = Path(matrix.golden).expanduser().resolve()
    if not golden.is_file():
        raise BenchError(f"golden file not found: {golden}")
    identity = detect(repo)
    golden_sha = hashlib.sha256(golden.read_bytes()).hexdigest()
    worktree_state = _source_worktree_state(repo)
    source_diff_sha = hashlib.sha256(worktree_state).hexdigest() if worktree_state else None
    source_state_id = hashlib.sha256(
        f"{identity.head_commit}:{source_diff_sha or ''}".encode()
    ).hexdigest()[:12]
    resolved_runtime_id = runtime_id or _runtime_id(matrix)
    fields = {
        "matrix_id": matrix.matrix_id,
        "golden_path": str(golden),
        "golden_sha256": golden_sha,
        "source_commit": identity.head_commit,
        "source_dirty": bool(worktree_state),
        "source_diff_sha256": source_diff_sha,
        "source_state_id": source_state_id,
        "repo_key": identity.normalized_key,
        "repo_path": str(repo),
        "top_k": matrix.top_k,
        "split": split,
        "runtime_id": resolved_runtime_id,
        "index_runtime_id": runtime_id or _index_runtime_id(),
    }
    fields["run_id"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return fields


def _source_worktree_state(repo: Path) -> bytes:
    """Return tracked changes plus non-ignored untracked content indexed by the walker."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=repo,
            capture_output=True,
            check=False,
            timeout=10,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard"],
            cwd=repo,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchError(f"cannot fingerprint benchmark source: {exc}") from exc
    if diff.returncode != 0 or untracked.returncode != 0:
        raise BenchError(
            "cannot fingerprint benchmark source: "
            f"git diff exited {diff.returncode}, git ls-files exited {untracked.returncode}"
        )
    state = bytearray(diff.stdout)
    for raw_path in sorted(path for path in untracked.stdout.split(b"\0") if path):
        path = repo / raw_path.decode("utf-8", errors="surrogateescape")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise BenchError(f"cannot fingerprint untracked file {path}: {exc}") from exc
        state.extend(b"\0UNTRACKED\0" + raw_path + b"\0" + content)
    return bytes(state)


def _runtime_id(matrix: Matrix) -> str:
    """Fingerprint query-time dependencies not fully declared by the matrix."""
    from ..inference import get_backend
    from ..meta.stream_weights import StreamWeightProfiles
    from ..pipeline.config import load_pipeline_config

    config = get_config()
    pipeline_defaults = load_pipeline_config()
    learned_dirs: set[Path] = set()
    for cell in matrix.expand():
        if not cell.values.get("learned_weights"):
            continue
        configured = cell.values.get("learned_weights_dir")
        if configured:
            learned_dirs.add(Path(str(configured)).expanduser().resolve())
        else:
            learned_dirs.add(
                config.resolve_index_path().expanduser().resolve().parent
                / "meta"
                / "stream-weights"
            )
    fields = {
        "inference": get_backend().identity,
        "inference_config": {
            "backend": config.inference.backend,
            "lms_timeout": config.inference.lms_timeout,
            "lms_max_retries": config.inference.lms_max_retries,
            "lms_think": config.inference.lms_think,
        },
        "retrieval_features": dataclasses.asdict(config.retrieval_features),
        "pipeline_defaults": {
            key: pipeline_defaults.get(key)
            for key in ("retrieval", "router")
        },
        "learned_weights": {
            str(path): StreamWeightProfiles(path).fingerprint()
            for path in sorted(learned_dirs)
        },
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def _index_runtime_id() -> str:
    """Fingerprint inference settings that can change contextual index output."""
    from ..inference import get_backend

    config = get_config()
    fields = {
        "inference": get_backend().identity,
        "enricher_batch_size": config.inference.effective_batch_size(),
        "enricher_concurrency": config.inference.enricher_concurrency,
        "backend_controls": {
            name: value
            for name, value in vars(config.inference).items()
            if name.endswith(("_timeout", "_max_retries"))
        },
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def _assert_run_environment(
    matrix: Matrix,
    provenance: dict[str, Any],
    source_state: bytes,
) -> None:
    """Fail if source or mutable query-time dependencies changed mid-run."""
    from ..ingest.git_meta import detect

    repo = Path(matrix.repo).expanduser().resolve()
    identity = detect(repo)
    if identity.head_commit != provenance["source_commit"]:
        raise BenchError("benchmark source commit changed during the run")
    if _source_worktree_state(repo) != source_state:
        raise BenchError("benchmark source worktree changed during the run")
    if _runtime_id(matrix) != provenance["runtime_id"]:
        raise BenchError("benchmark runtime dependencies changed during the run")


def _validate_output_identity(path: Path, run_id: str) -> set[str]:
    """Return completed cells after proving every existing record is comparable."""
    done: set[str] = set()
    if not path.is_file():
        return done
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchError(f"invalid result JSON at {path}:{line_number}") from exc
        existing_run_id = (rec.get("provenance") or {}).get("run_id")
        if existing_run_id != run_id:
            raise BenchError(
                f"incomparable result at {path}:{line_number}: "
                f"run_id={existing_run_id!r}, expected {run_id!r}; use a new --out path"
            )
        if rec.get("ok"):
            done.add(rec.get("cell_id", ""))
    return done


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
    split: BenchSplit = "search",
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
    if not resume and out_path.is_file():
        raise BenchError("--no-resume requires a new --out path")
    run_provenance = _run_provenance(matrix, split)
    repo = Path(matrix.repo).expanduser().resolve()
    initial_source_state = _source_worktree_state(repo)
    initial_source_sha = (
        hashlib.sha256(initial_source_state).hexdigest() if initial_source_state else None
    )
    if initial_source_sha != run_provenance["source_diff_sha256"]:
        raise BenchError("benchmark source changed while preparing the run")
    _assert_run_environment(matrix, run_provenance, initial_source_state)

    completed = _validate_output_identity(out_path, run_provenance["run_id"])
    done = completed if resume else set()

    total = len(cells)
    if dry_run:
        plan = {
            "total_cells": total,
            "unique_indexes": len(groups),
            "already_done": len(done & {c.cell_id for c in cells}),
            "run_id": run_provenance["run_id"],
            "matrix_id": matrix.matrix_id,
            "groups": {sig: [c.label() for c in cs] for sig, cs in groups.items()},
        }
        return plan

    from ..eval import filter_by_split, load_golden

    cases = filter_by_split(load_golden(Path(matrix.golden)), split)
    if not cases:
        raise BenchError(f"golden file contains no {split!r} cases: {matrix.golden}")

    evaluated = failed = skipped = 0
    seen = 0
    for sig, group in groups.items():
        pending = [c for c in group if c.cell_id not in done]
        if not pending:
            skipped += len(group)
            continue
        db_path = index_dir / (
            f"index_{sig}_{run_provenance['source_state_id']}_"
            f"{run_provenance['index_runtime_id']}.db"
        )
        try:
            build_index(matrix, pending[0], db_path)
            _assert_run_environment(matrix, run_provenance, initial_source_state)
        except Exception as exc:
            log.warning("bench.index_failed", indexing_sig=sig, error=str(exc))
            for cell in pending:
                rec = {"ok": False, "cell_id": cell.cell_id, "indexing_sig": sig,
                       "label": cell.label(), "values": cell.values,
                       "provenance": run_provenance,
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
                _assert_run_environment(matrix, run_provenance, initial_source_state)
                rec = eval_cell(
                    matrix,
                    cell,
                    db_path,
                    cases,
                    run_provenance=run_provenance,
                )
                _assert_run_environment(matrix, run_provenance, initial_source_state)
                evaluated += 1
            except Exception as exc:  # record + continue
                rec = {"ok": False, "cell_id": cell.cell_id, "indexing_sig": sig,
                       "label": cell.label(), "values": cell.values,
                       "provenance": run_provenance,
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
