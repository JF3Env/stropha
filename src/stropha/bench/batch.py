"""Reproducible golden-cohort generation and benchmark execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..errors import StrophaError
from ..eval import EvalCase, load_golden
from ..ingest.walker import Walker
from ..pipeline.registry import lookup_adapter
from ..server_frontend import (
    merge_config_document,
    read_config_document,
    write_config_document,
)
from .matrix import Cell, Matrix, load_matrix
from .report import load_results
from .runner import _FEATURE_FIELD, _RETRIEVAL_FIELD, _run_provenance, run_matrix

_RETRIEVAL_PROMOTION_FIELDS = _RETRIEVAL_FIELD
_FEATURE_PROMOTION_FIELDS = _FEATURE_FIELD


class BatchError(StrophaError):
    """Raised when a benchmark batch is invalid or cannot complete safely."""


@dataclass(frozen=True)
class PreparedBatch:
    name: str
    root: Path
    cohort_path: Path
    manifest_path: Path
    cohort_sha256: str
    case_count: int


_CATEGORY_TAGS = {
    "exact-lookup",
    "conceptual",
    "callers-usage",
    "tests-contracts",
    "feature-trace",
    "config-ops",
    "docs-rationale",
    "ambiguous-symbols",
    "noisy-negative",
    "multi-repo",
}

_VERIFY_COMMANDS = {
    "ruff": ["uv", "run", "ruff", "check", "src", "tests"],
    "mypy": ["uv", "run", "mypy", "src/stropha"],
    "pytest": ["uv", "run", "pytest", "-q"],
    "diff-check": ["git", "diff", "--check"],
}


def load_batch_spec(path: Path) -> dict[str, Any]:
    """Load and minimally validate one checked-in batch specification."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BatchError(f"could not load batch spec {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BatchError("batch spec root must be a mapping")
    if raw.get("version") != 1:
        raise BatchError("batch spec version must be 1")
    if raw.get("invalidated"):
        raise BatchError(f"batch is invalidated: {raw['invalidated']}")
    for key in ("name", "matrix"):
        if not raw.get(key):
            raise BatchError(f"batch spec missing {key!r}")
    catalogs = raw.get("catalogs")
    if raw.get("catalog") and catalogs:
        raise BatchError("batch spec must declare either catalog or catalogs, not both")
    if catalogs is not None and (
        not isinstance(catalogs, list)
        or not catalogs
        or not all(isinstance(item, str) and item for item in catalogs)
    ):
        raise BatchError("batch spec catalogs must be a non-empty list of paths")
    if not raw.get("catalog") and catalogs is None:
        raise BatchError("batch spec missing 'catalog' or 'catalogs'")
    return raw


def prepare_batch(spec_path: Path, *, artifact_root: Path | None = None) -> PreparedBatch:
    """Validate and freeze the catalog, then materialize canonical JSONL."""
    spec_path = spec_path.resolve()
    spec = load_batch_spec(spec_path)
    repo = Path(spec.get("repo", ".")).expanduser().resolve()
    catalog_values = spec.get("catalogs") or [spec["catalog"]]
    catalog_paths = [_repo_path(repo, catalog) for catalog in catalog_values]
    matrix_path = _repo_path(repo, spec["matrix"])
    for catalog_path in catalog_paths:
        _assert_catalog_excluded(repo, catalog_path)
    exposed_path = _repo_path(repo, spec.get("exposed_golden", "tests/eval/golden.jsonl"))
    cases = [case for catalog_path in catalog_paths for case in _load_catalog(catalog_path)]
    exposed = load_golden(exposed_path) if exposed_path.is_file() else []
    exposed_catalogs = spec.get("exposed_catalogs") or []
    if not isinstance(exposed_catalogs, list):
        raise BatchError("exposed_catalogs must be a list")
    exposed_catalog_paths = [
        _repo_path(repo, str(exposed_catalog)) for exposed_catalog in exposed_catalogs
    ]
    for exposed_catalog_path in exposed_catalog_paths:
        exposed.extend(_load_catalog(exposed_catalog_path))
    validation = _validate_cases(cases, exposed, repo, spec.get("cohort") or {})

    root = (artifact_root or _repo_path(repo, ".stropha/bench/batches")) / str(spec["name"])
    root.mkdir(parents=True, exist_ok=True)
    cohort_path = root / "cohort.jsonl"
    manifest_path = root / "manifest.json"
    payload = "".join(
        json.dumps(_case_dict(case), sort_keys=True, separators=(",", ":")) + "\n"
        for case in sorted(cases, key=lambda item: item.id)
    )
    cohort_sha = hashlib.sha256(payload.encode()).hexdigest()
    catalog_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in catalog_paths
    }
    protocol_sha = _protocol_sha256(
        spec, matrix_path, exposed_path, exposed_catalog_paths
    )

    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("cohort_sha256") != cohort_sha
            or existing.get("protocol_sha256") != protocol_sha
        ):
            raise BatchError(
                f"batch {spec['name']!r} is already frozen with a different cohort or protocol; "
                "choose a new batch name"
            )
    _atomic_write(cohort_path, payload)
    manifest = {
        "version": 1,
        "name": spec["name"],
        "status": "frozen",
        "catalogs": [str(path) for path in catalog_paths],
        "catalog_sha256": (
            next(iter(catalog_hashes.values()))
            if len(catalog_hashes) == 1
            else hashlib.sha256(
                json.dumps(catalog_hashes, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        ),
        "cohort": str(cohort_path),
        "cohort_sha256": cohort_sha,
        "protocol_sha256": protocol_sha,
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "validation": validation,
    }
    _atomic_json(manifest_path, manifest)
    return PreparedBatch(
        name=str(spec["name"]),
        root=root,
        cohort_path=cohort_path,
        manifest_path=manifest_path,
        cohort_sha256=cohort_sha,
        case_count=len(cases),
    )


def run_batch(
    spec_path: Path,
    *,
    artifact_root: Path | None = None,
    prepare_only: bool = False,
    dry_run: bool = False,
    verify: bool = True,
    promote: bool = False,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare, execute, gate, verify, and optionally promote one batch."""
    spec_path = spec_path.resolve()
    spec = load_batch_spec(spec_path)
    if promote and not verify:
        raise BatchError("promotion requires verification; remove --no-verify")
    if promote and not spec.get("verification"):
        raise BatchError("promotion requires a non-empty verification profile")
    prepared = prepare_batch(spec_path, artifact_root=artifact_root)
    if prepare_only:
        return _prepared_result(prepared)
    results_path = prepared.root / "results.jsonl"
    decision_path = prepared.root / "decision.json"
    has_results = results_path.is_file() and results_path.stat().st_size > 0
    if not dry_run and (decision_path.is_file() or has_results):
        raise BatchError(
            f"batch {prepared.name!r} has already been executed; create a new unseen batch"
        )

    repo = Path(spec.get("repo", ".")).expanduser().resolve()
    matrix = load_matrix(_repo_path(repo, spec["matrix"]))
    matrix.golden = str(prepared.cohort_path)
    matrix.repo = str(repo)
    baseline = _select_cell(matrix, spec.get("baseline") or matrix.baseline or {})
    candidate = _select_cell(matrix, spec.get("candidate") or {})
    _validate_promotion_axis(spec, baseline, candidate)
    selected = _selected_matrix(matrix, baseline, candidate)
    index_dir = prepared.root / "indexes"
    if dry_run:
        plan = run_matrix(
            selected,
            results_path,
            index_dir=index_dir,
            dry_run=True,
            split="holdout",
        )
        return {**_prepared_result(prepared), "plan": plan}

    summary = run_matrix(
        selected,
        results_path,
        index_dir=index_dir,
        split="holdout",
    )
    records = load_results(results_path)
    decision = evaluate_gates(records, baseline, candidate, spec.get("gates") or {})
    verification = _run_verification(repo, spec.get("verification") or []) if verify else []
    decision["verification"] = verification
    decision["eligible"] = (
        bool(decision["eligible"])
        and bool(verification)
        and all(item["ok"] for item in verification)
    )
    decision.update(
        {
            "batch": prepared.name,
            "cohort_sha256": prepared.cohort_sha256,
            "results": str(results_path),
            "summary": summary,
        }
    )
    _atomic_json(decision_path, decision)
    if promote:
        if not decision["eligible"]:
            raise BatchError("batch is not promotion-eligible; configuration was not changed")
        current_run_id = _run_provenance(selected, "holdout")["run_id"]
        recorded_run_ids = {
            (record.get("provenance") or {}).get("run_id") for record in records
        }
        if recorded_run_ids != {current_run_id}:
            raise BatchError(
                "benchmark environment changed before promotion; configuration was not changed"
            )
        _promote(spec, baseline, candidate, config_path=config_path)
        decision["promoted"] = True
        _atomic_json(decision_path, decision)
    return decision


def evaluate_gates(
    records: list[dict[str, Any]],
    baseline: Cell,
    candidate: Cell,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate aggregate, per-tag, and per-case promotion gates."""
    base = next((record for record in records if record.get("cell_id") == baseline.cell_id), None)
    arm = next((record for record in records if record.get("cell_id") == candidate.cell_id), None)
    if base is None or arm is None:
        raise BatchError("baseline or candidate result is missing")
    bm = base.get("metrics") or {}
    cm = arm.get("metrics") or {}
    checks: list[dict[str, Any]] = []

    def delta(name: str, minimum: float) -> None:
        actual = float(cm[name]) - float(bm[name])
        checks.append({"gate": name, "actual": actual, "required_min": minimum,
                       "ok": actual >= minimum})

    delta("success@10", float(gates.get("success_delta_min", -0.01)))
    delta("recall@10", float(gates.get("recall_delta_min", -0.01)))
    delta("ndcg@10", float(gates.get("ndcg_delta_min", 0.02)))
    delta("mrr", float(gates.get("mrr_delta_min", -0.01)))
    fallback_max = float(gates.get("fallback_rate_max", 0.05))
    fallback = float(cm.get("reranker_fallback_rate", 0.0))
    checks.append({"gate": "fallback_rate", "actual": fallback, "required_max": fallback_max,
                   "ok": fallback < fallback_max})
    latency_ratio = float(cm["latency_p95_ms"]) / max(float(bm["latency_p95_ms"]), 0.001)
    latency_max = float(gates.get("latency_ratio_max", 1.25))
    checks.append({"gate": "latency_ratio", "actual": latency_ratio,
                   "required_max": latency_max, "ok": latency_ratio <= latency_max})
    token_ratio = float(cm["returned_tokens"]) / max(float(bm["returned_tokens"]), 1.0)
    recall_gain = float(cm["recall@10"]) - float(bm["recall@10"])
    token_max = float(gates.get("token_ratio_max", 1.0)) + max(recall_gain, 0.0)
    checks.append({"gate": "token_ratio", "actual": token_ratio,
                   "required_max": token_max, "ok": token_ratio <= token_max})

    per_tag_limit = float(gates.get("per_tag_regression_max", 0.01))
    base_tags = base.get("by_tag") or {}
    arm_tags = arm.get("by_tag") or {}
    base_tag_names = set(base_tags)
    arm_tag_names = set(arm_tags)
    tag_regressions: list[dict[str, Any]] = []
    for tag in sorted(base_tag_names & arm_tag_names):
        for metric in ("success@10", "recall@10", "mrr", "ndcg@10"):
            regression = float(base_tags[tag][metric]) - float(arm_tags[tag][metric])
            if regression > per_tag_limit:
                tag_regressions.append({"tag": tag, "metric": metric, "regression": regression})
    missing_tags = sorted(base_tag_names - arm_tag_names)
    extra_tags = sorted(arm_tag_names - base_tag_names)
    checks.append({"gate": "per_tag", "regressions": tag_regressions,
                   "missing_tags": missing_tags, "extra_tags": extra_tags,
                   "ok": not tag_regressions and not missing_tags and not extra_tags})

    case_regressions = []
    base_queries = base.get("per_query") or {}
    arm_queries = arm.get("per_query") or {}
    base_case_ids = set(base_queries)
    arm_case_ids = set(arm_queries)
    for case_id in sorted(base_case_ids & arm_case_ids):
        base_rank = base_queries[case_id]
        arm_rank = arm_queries[case_id]
        if base_rank is not None and (arm_rank is None or arm_rank > base_rank):
            case_regressions.append(
                {"case_id": case_id, "baseline_rank": base_rank, "candidate_rank": arm_rank}
            )
    if gates.get("no_case_rank_regression", True):
        missing_cases = sorted(base_case_ids - arm_case_ids)
        extra_cases = sorted(arm_case_ids - base_case_ids)
        checks.append({"gate": "per_case_rank", "regressions": case_regressions,
                       "missing_cases": missing_cases, "extra_cases": extra_cases,
                       "ok": not case_regressions and not missing_cases and not extra_cases})
    return {
        "eligible": all(check["ok"] for check in checks),
        "baseline_cell_id": baseline.cell_id,
        "candidate_cell_id": candidate.cell_id,
        "baseline_metrics": bm,
        "candidate_metrics": cm,
        "checks": checks,
    }


def _load_catalog(path: Path) -> list[EvalCase]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BatchError(f"could not load cohort catalog {path}: {exc}") from exc
    rows = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise BatchError("cohort catalog must contain a cases list")
    try:
        return [EvalCase.from_dict(row) for row in rows]
    except (TypeError, ValueError, KeyError) as exc:
        raise BatchError(f"invalid cohort catalog {path}: {exc}") from exc


def _validate_cases(
    cases: list[EvalCase],
    exposed: list[EvalCase],
    repo: Path,
    cohort: dict[str, Any],
) -> dict[str, Any]:
    ids: set[str] = set()
    queries: set[str] = set()
    exposed_ids = {case.id for case in exposed}
    exposed_queries = {_normalize_query(case.query) for case in exposed}
    category_counts = {tag: 0 for tag in _CATEGORY_TAGS}
    multi_or_graded = 0
    for case in cases:
        if case.id in ids:
            raise BatchError(f"duplicate cohort case id {case.id!r}")
        ids.add(case.id)
        normalized_query = _normalize_query(case.query)
        if normalized_query in queries:
            raise BatchError(f"duplicate cohort query {case.query!r}")
        queries.add(normalized_query)
        if case.id in exposed_ids or _normalize_query(case.query) in exposed_queries:
            raise BatchError(f"case {case.id!r} overlaps the exposed golden set")
        if case.split != "holdout":
            raise BatchError(f"case {case.id!r} must use split='holdout'")
        categories = _CATEGORY_TAGS.intersection(case.tags)
        if len(categories) != 1:
            raise BatchError(f"case {case.id!r} must have exactly one category tag")
        category_counts[next(iter(categories))] += 1
        expected = set(case.expected_paths) | set(case.expected_symbols)
        if not expected:
            raise BatchError(f"case {case.id!r} has no expected entities")
        if case.relevance_map and set(case.relevance_map) != expected:
            raise BatchError(f"case {case.id!r} relevance keys must equal expected entities")
        if any(score < 1 or score > 3 for score in case.relevance_map.values()):
            raise BatchError(f"case {case.id!r} relevance must be in the range 1..3")
        if len(case.expected_symbols) > len(case.expected_paths):
            raise BatchError(
                f"case {case.id!r} must provide a paired path for every expected symbol"
            )
        if case.relevant_entity_count > 1 or case.relevance_map:
            multi_or_graded += 1
        texts: list[str] = []
        for rel_path in case.expected_paths:
            target = repo / rel_path
            if not target.is_file():
                raise BatchError(f"case {case.id!r} expected path does not exist: {rel_path}")
            texts.append(target.read_text(encoding="utf-8", errors="replace"))
        for index, symbol in enumerate(case.expected_symbols):
            leaf = symbol.rsplit(".", 1)[-1]
            if leaf not in texts[index]:
                raise BatchError(
                    f"case {case.id!r} symbol {symbol!r} not found in paired path "
                    f"{case.expected_paths[index]!r}"
                )
    minimum = int(cohort.get("min_cases", 1))
    if len(cases) < minimum:
        raise BatchError(f"cohort has {len(cases)} cases; requires at least {minimum}")
    ratio = multi_or_graded / len(cases) if cases else 0.0
    required_ratio = float(cohort.get("min_multi_or_graded_ratio", 0.5))
    if ratio < required_ratio:
        raise BatchError(f"multi-entity/graded ratio {ratio:.3f} is below {required_ratio:.3f}")
    min_per_category = int(cohort.get("min_per_category", 0))
    missing = {tag: count for tag, count in category_counts.items() if count < min_per_category}
    if missing:
        raise BatchError(f"category quotas not met: {missing}")
    return {
        "category_counts": category_counts,
        "multi_or_graded_ratio": ratio,
        "exposed_case_count": len(exposed),
    }


def _selected_matrix(matrix: Matrix, baseline: Cell, candidate: Cell) -> Matrix:
    allowed = {baseline.cell_id, candidate.cell_id}
    constraints = list(matrix.constraints)
    for cell in matrix.expand():
        if cell.cell_id not in allowed:
            constraints.append({"skip_when": cell.values})
    return Matrix(
        axes=matrix.axes,
        golden=matrix.golden,
        top_k=matrix.top_k,
        repo=matrix.repo,
        baseline=baseline.values,
        constraints=constraints,
    )


def _select_cell(matrix: Matrix, selector: dict[str, Any]) -> Cell:
    matches = [
        cell for cell in matrix.expand()
        if all(cell.values.get(key) == value for key, value in selector.items())
    ]
    if len(matches) != 1:
        raise BatchError(f"selector {selector!r} matched {len(matches)} matrix cells")
    return matches[0]


def _run_verification(repo: Path, names: list[str]) -> list[dict[str, Any]]:
    results = []
    for name in names:
        command = _VERIFY_COMMANDS.get(name)
        if command is None:
            raise BatchError(f"unknown verification step {name!r}")
        completed = subprocess.run(command, cwd=repo, check=False)
        results.append({"name": name, "command": command, "ok": completed.returncode == 0,
                        "returncode": completed.returncode})
    return results


def _promote(
    spec: dict[str, Any],
    baseline: Cell,
    candidate: Cell,
    *,
    config_path: Path | None,
) -> None:
    promotion = spec.get("promotion") or {}
    axis = str(promotion.get("axis", "rrf_k"))
    if axis not in candidate.values:
        raise BatchError(f"candidate has no promotable axis {axis!r}")
    target = config_path or Path.home() / ".stropha" / "config.yaml"
    current = read_config_document(target)
    _assert_baseline_config(current, baseline)
    if axis in _RETRIEVAL_PROMOTION_FIELDS:
        patch = {
            "pipeline": {
                "retrieval": {
                    "config": {_RETRIEVAL_PROMOTION_FIELDS[axis]: candidate.values[axis]}
                }
            }
        }
    elif axis in _FEATURE_PROMOTION_FIELDS:
        patch = {
            "retrieval_features": {
                _FEATURE_PROMOTION_FIELDS[axis]: candidate.values[axis]
            }
        }
    else:
        raise BatchError(f"promotion axis {axis!r} is not a query-time setting")
    write_config_document(merge_config_document(current, patch), target)


def _validate_promotion_axis(
    spec: dict[str, Any], baseline: Cell, candidate: Cell
) -> None:
    axis = str((spec.get("promotion") or {}).get("axis", "rrf_k"))
    differing = {
        key
        for key in set(baseline.values) | set(candidate.values)
        if baseline.values.get(key) != candidate.values.get(key)
    }
    if differing != {axis}:
        raise BatchError(
            f"candidate must differ from baseline only on promotion axis {axis!r}; "
            f"differing axes: {sorted(differing)}"
        )


def _assert_catalog_excluded(repo: Path, catalog: Path) -> None:
    try:
        rel_path = catalog.relative_to(repo).as_posix()
    except ValueError:
        return
    if Walker(repo).evaluate_path(rel_path) is not None:
        raise BatchError(
            f"holdout catalog {rel_path!r} is indexable; exclude it in .strophaignore"
        )


def _protocol_sha256(
    spec: dict[str, Any],
    matrix_path: Path,
    exposed_path: Path,
    exposed_catalog_paths: list[Path],
) -> str:
    payload = {
        "spec": spec,
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "exposed_sha256": (
            hashlib.sha256(exposed_path.read_bytes()).hexdigest()
            if exposed_path.is_file()
            else None
        ),
        "exposed_catalog_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in exposed_catalog_paths
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _assert_baseline_config(current: dict[str, Any], baseline: Cell) -> None:
    pipeline = current.get("pipeline") or {}
    retrieval = (pipeline.get("retrieval") or {}).get("config") or {}
    features = current.get("retrieval_features") or {}
    mismatches: list[str] = []
    for axis, expected in baseline.values.items():
        if axis in {"embedder", "enricher", "chunker", "walker"}:
            section = pipeline.get(axis) or {}
            actual = _normalize_adapter_spec(axis, section)
            expected = _normalize_adapter_spec(axis, expected)
        elif axis == "reranker":
            configured = retrieval.get("reranker")
            actual = _normalize_adapter_spec("reranker", configured)
            expected = _normalize_adapter_spec("reranker", expected)
        elif axis in _RETRIEVAL_PROMOTION_FIELDS:
            actual = retrieval.get(_RETRIEVAL_PROMOTION_FIELDS[axis])
        elif axis in _FEATURE_PROMOTION_FIELDS:
            actual = features.get(_FEATURE_PROMOTION_FIELDS[axis])
        else:
            raise BatchError(f"cannot verify production baseline axis {axis!r}")
        if actual != expected:
            mismatches.append(f"{axis}: configured={actual!r}, benchmark={expected!r}")
    if mismatches:
        raise BatchError("production config differs from benchmark baseline: " + "; ".join(mismatches))


def _normalize_adapter_spec(stage: str, value: Any) -> Any:
    if value is None:
        return None
    adapter: Any
    config: dict[str, Any]
    if isinstance(value, str):
        adapter, config = value, {}
    elif isinstance(value, dict):
        adapter = value.get("adapter")
        nested = value.get("config")
        config = nested if isinstance(nested, dict) else {
            key: item for key, item in value.items() if key != "adapter"
        }
    else:
        return value
    if not adapter:
        return value
    adapter_cls = lookup_adapter(stage, str(adapter))
    config_cls = getattr(adapter_cls, "Config", None)
    normalized = config_cls(**config).model_dump(mode="json") if config_cls else config
    return {"adapter": adapter, "config": normalized}


def _case_dict(case: EvalCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "query": case.query,
        "expected_paths": list(case.expected_paths),
        "expected_symbols": list(case.expected_symbols),
        "relevance": case.relevance_map,
        "tags": list(case.tags),
        "split": case.split,
    }


def _repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _prepared_result(prepared: PreparedBatch) -> dict[str, Any]:
    return {
        "batch": prepared.name,
        "status": "frozen",
        "cohort": str(prepared.cohort_path),
        "cohort_sha256": prepared.cohort_sha256,
        "case_count": prepared.case_count,
        "manifest": str(prepared.manifest_path),
    }


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "BatchError",
    "PreparedBatch",
    "evaluate_gates",
    "load_batch_spec",
    "prepare_batch",
    "run_batch",
]
