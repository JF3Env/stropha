"""Promotion planning — pure functions for diff + atomic apply (WIN-202).

The Meta-Harness produces candidate configs that *propose* changes to the
production unified config. Promotion is human-gated by design (§3 spirit of
the master spec): the harness never auto-applies. This module produces a
:class:`PromotionPlan` that captures the current config, the safely merged
result, and a unified diff. The CLI prints the diff by default; ``--apply``
opts the operator into writing.

Why diff-only by default
------------------------
Auto-applying candidate configs to the unified config would:

1. Surprise the operator with a live config swap on every ``meta promote``
   run, regardless of intent.
2. Destroy keys outside the search space (the candidate ``config`` is the
   sampled subset — the manifest may not cover every YAML key, so a naive
   overwrite drops e.g. inference credentials or hand-tuned retrieval
   weights).
3. Skip the only human-readable summary of "what is about to change",
   which is exactly what a code review depends on.

The diff renderer here is git-style unified diff against the rendered
YAML for both sides — easy to eyeball, easy to ``git apply`` if the
operator wants to stage the change manually instead of letting the CLI
overwrite the file.

Apply semantics
---------------
Candidate changes are normalized to the unified runtime shape, then deep-
merged into the existing document. Structured pipeline candidates land under
``pipeline``; flat search-space stages and features are translated to their
runtime sections. ``apply_plan`` atomically writes that pre-computed merged
document, preserving secrets and unrelated settings.
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from .store import CandidateRecord

_PIPELINE_STAGES = frozenset({"walker", "chunker", "enricher", "embedder", "storage", "retrieval"})
_FEATURE_FIELDS = {
    "hyde": "hyde_enabled",
    "hyde_code_tuned": "hyde_code_tuned",
    "query_rewrite": "query_rewrite_enabled",
    "multi_query": "multi_query_enabled",
}
_RETRIEVAL_CONFIG_FIELDS = {
    "multi_query_count": "multi_query_count",
    "query_rewrite_max_terms": "query_rewrite_max_terms",
    "recursive": "recursive_enabled",
    "recursive_adjacency": "recursive_adjacency",
    "adaptive_weighting": "adaptive_weighting",
}


class PromotionError(RuntimeError):
    """Raised when a promotion plan cannot be built or applied."""


@dataclass(frozen=True)
class PromotionPlan:
    """A pre-computed promotion: what changes, where, and whether to apply.

    ``current_config`` is ``None`` when ``target_path`` does not exist yet
    (first promotion on a fresh checkout). ``is_noop`` is ``True`` when
    the current and candidate configs are byte-identical after YAML round-
    tripping — the apply step is a no-op in that case and the CLI surfaces
    it to the operator instead of pretending work happened.
    """

    candidate_id: str
    target_path: Path
    current_config: dict[str, Any] | None
    candidate_config: dict[str, Any]
    diff_text: str
    is_noop: bool


def _render_yaml(data: dict[str, Any]) -> str:
    """Render a dict as deterministic YAML for diff purposes.

    Sorted keys + default style ensure two calls with the same payload
    produce byte-identical output — required so the diff doesn't churn
    on dict-ordering noise across Python versions.
    """
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


def _read_target(target_path: Path) -> dict[str, Any] | None:
    """Load the current unified config if present.

    Returns ``None`` when the file is missing (first-time promotion).
    Raises :class:`PromotionError` when the file exists but is malformed —
    overwriting a broken config without asking would silently swallow the
    operator's mistake.
    """
    if not target_path.is_file():
        return None
    try:
        loaded = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromotionError(f"target {target_path} exists but is not valid YAML: {exc}") from exc
    if loaded is None:
        # Empty YAML file → treat as empty config (legitimate edge case).
        return {}
    if not isinstance(loaded, dict):
        raise PromotionError(
            f"target {target_path} must be a YAML mapping at the top level, "
            f"got {type(loaded).__name__}"
        )
    return loaded


def _deep_merge(target: dict[str, Any], changes: dict[str, Any]) -> None:
    """Deep-merge ``changes`` into ``target`` in place."""
    for key, value in changes.items():
        current = target.get(key)
        adapter_changed = (
            isinstance(value, dict)
            and isinstance(current, dict)
            and "adapter" in value
            and value["adapter"] != current.get("adapter")
        )
        if isinstance(value, dict) and isinstance(current, dict) and not adapter_changed:
            _deep_merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


def _candidate_changes(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize pipeline or flat search-space config to unified sections."""
    remaining = copy.deepcopy(config)
    changes: dict[str, Any] = {}

    for section in ("pipeline", "retrieval_features", "graph"):
        if section not in remaining:
            continue
        value = remaining.pop(section)
        if not isinstance(value, dict):
            raise PromotionError(f"candidate `{section}` must be a mapping")
        changes[section] = value

    pipeline = changes.setdefault("pipeline", {})
    for stage in _PIPELINE_STAGES:
        if stage not in remaining:
            continue
        value = remaining.pop(stage)
        stage_changes = value if isinstance(value, dict) else {"adapter": value}
        existing = pipeline.setdefault(stage, {})
        if not isinstance(existing, dict):
            raise PromotionError(f"candidate pipeline `{stage}` must be a mapping")
        _deep_merge(existing, stage_changes)

    if "reranker" in remaining:
        value = remaining.pop("reranker")
        reranker = value if isinstance(value, dict) else {"adapter": value}
        retrieval = pipeline.setdefault("retrieval", {})
        retrieval_config = retrieval.setdefault("config", {})
        retrieval_config["reranker"] = reranker

    features = changes.setdefault("retrieval_features", {})
    for candidate_key, runtime_key in _FEATURE_FIELDS.items():
        if candidate_key in remaining:
            features[runtime_key] = remaining.pop(candidate_key)

    for candidate_key, runtime_key in _RETRIEVAL_CONFIG_FIELDS.items():
        if candidate_key not in remaining:
            continue
        retrieval = pipeline.setdefault("retrieval", {})
        retrieval_config = retrieval.setdefault("config", {})
        retrieval_config[runtime_key] = remaining.pop(candidate_key)

    if "graph_fts_augment" in remaining:
        graph = changes.setdefault("graph", {})
        graph["fts_augment"] = remaining.pop("graph_fts_augment")

    if remaining:
        unsupported = ", ".join(sorted(remaining))
        raise PromotionError(f"unsupported candidate config keys: {unsupported}")

    return {key: value for key, value in changes.items() if value}


def build_plan(candidate: CandidateRecord, target_path: Path) -> PromotionPlan:
    """Build a :class:`PromotionPlan` for ``candidate`` vs ``target_path``.

    Pure function — does not touch the filesystem beyond reading
    ``target_path`` (if it exists). Safe to call from anywhere, including
    from tests that mock the CLI layer.
    """
    current = _read_target(target_path)
    candidate_config = copy.deepcopy(current) if current is not None else {}
    _deep_merge(candidate_config, _candidate_changes(candidate.config or {}))

    current_yaml = _render_yaml(current) if current is not None else ""
    candidate_yaml = _render_yaml(candidate_config)

    # When the target doesn't exist, the "from" side is empty so the diff
    # shows the entire candidate config as an addition. The header still
    # uses the target path so the operator can copy-paste the diff and
    # know where it lands.
    from_label = (
        f"a/{target_path.name}" if current is not None else f"a/{target_path.name} (missing)"
    )
    to_label = f"b/{target_path.name} (candidate {candidate.id})"
    diff_lines = list(
        difflib.unified_diff(
            current_yaml.splitlines(keepends=True),
            candidate_yaml.splitlines(keepends=True),
            fromfile=from_label,
            tofile=to_label,
        )
    )
    diff_text = "".join(diff_lines)
    is_noop = current_yaml == candidate_yaml

    return PromotionPlan(
        candidate_id=candidate.id,
        target_path=target_path,
        current_config=current,
        candidate_config=candidate_config,
        diff_text=diff_text,
        is_noop=is_noop,
    )


def apply_plan(plan: PromotionPlan) -> None:
    """Atomically write the merged config to ``plan.target_path`` as YAML.

    A no-op plan still re-writes the file: idempotent, but cheap. We do
    NOT short-circuit even when ``is_noop`` so that recovery from a
    partially-written target (truncated YAML) still completes — the
    operator can re-run ``--apply`` to settle the file.

    Atomic write: render to a temp file in the same directory and
    ``Path.replace`` over the target. On POSIX, ``replace`` is atomic so
    a SIGKILL mid-write leaves either the old or the new content, never
    a half-written hybrid.
    """
    plan.target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _render_yaml(plan.candidate_config)
    tmp = plan.target_path.with_name(plan.target_path.name + ".promote.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(plan.target_path)


def render_summary(plan: PromotionPlan) -> str:
    """Return a one-line human-readable summary for log lines / TUIs."""
    if plan.is_noop:
        return (
            f"candidate {plan.candidate_id} matches current "
            f"{plan.target_path} — apply would be a no-op"
        )
    if plan.current_config is None:
        return f"candidate {plan.candidate_id} would create new {plan.target_path}"
    return (
        f"candidate {plan.candidate_id} would modify {plan.target_path} "
        f"(diff has {plan.diff_text.count(chr(10))} lines)"
    )
