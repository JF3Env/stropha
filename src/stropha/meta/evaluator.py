"""``evaluate_candidate(config, split) → ScoreCard`` (WIN-171).

The Meta-Harness loop (Phase 4) needs a single Python entry point that
scores one candidate end-to-end:

    1. Build the SearchEngine the candidate's ``config`` describes.
    2. Run :func:`stropha.eval.run_eval` over the requested ``split`` of
       the golden set.
    3. Optionally pull outcome labels from a backfill summary so the
       T1-T7 component lands on the ScoreCard.
    4. Compose everything into the :class:`stropha.meta.ScoreCard`
       (WIN-169).

This module is the *glue*. It does NOT own the search engine, the
golden set, the labeler, or the fitness math — each of those belongs to
its own module. Keeping evaluator.py thin means the proposer can swap
any layer without touching this one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

from ..errors import StrophaError
from ..eval import EvalCase, EvalReport, filter_by_split, load_golden, run_eval
from ..logging import get_logger
from .fitness import ScoreCard, build_scorecard
from .outcome import TierLabel

log = get_logger(__name__)


class EvaluatorError(StrophaError):
    """Raised when ``evaluate_candidate`` cannot satisfy a request."""


class _SearchEngineLike(Protocol):
    """Minimal slice of the SearchEngine surface the evaluator depends on."""

    def search(self, query: str, top_k: int = 10) -> list[Any]:
        ...


def _load_labels_from_outcomes(
    outcomes_dir: Path,
    head_sha: str | None = None,
) -> list[TierLabel]:
    """Best-effort hydration of TierLabels from a backfill outcome JSON.

    The backfill (WIN-164) persists outcomes/<head_sha>.json with
    per-trace scores and tier counts. v1 of this loader only consumes
    the tier_counts to synthesise placeholder labels so the ScoreCard
    has a non-zero outcome_score component when historical traces
    exist. A future refinement (when the proposer wants per-chunk
    introspection) will hydrate full TierLabels by re-running the
    labeler — kept out of v1 to avoid coupling the evaluator to the
    backfill internals.
    """
    if not outcomes_dir.is_dir():
        return []
    candidates: list[Path]
    if head_sha is not None:
        candidates = [outcomes_dir / f"{head_sha}.json"]
    else:
        candidates = sorted(outcomes_dir.glob("*.json"))
    labels: list[TierLabel] = []
    from .outcome import TIER_SCORE, Tier
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tier_counts = payload.get("tier_counts") or {}
        for tier_name, count in tier_counts.items():
            try:
                tier = Tier(tier_name)
            except ValueError:
                continue
            score = TIER_SCORE.get(tier, 0.0)
            for i in range(int(count)):
                labels.append(
                    TierLabel(
                        chunk_id=f"{path.stem}:{tier.value}:{i}",
                        tier=tier,
                        score=score,
                        rationale=f"hydrated from {path.name}",
                    )
                )
    return labels


def _baseline_tokens_from_repo(repo_size_bytes: int | None) -> int:
    """Same convention as ``retrieval/savings.py`` — 10% of repo / 4 bytes per token."""
    if not repo_size_bytes or repo_size_bytes <= 0:
        return 0
    return (repo_size_bytes // 10) // 4


def _returned_tokens_from_report(report: EvalReport) -> int:
    """Sum the snippet-length-estimate across every hit in the report."""
    total = 0
    for r in report.results:
        for rel in r.relevances:
            # Each non-zero relevance corresponds to a hit; we do not have
            # the snippet text here so approximate at 50 tokens/chunk.
            if rel > 0:
                total += 50
    return total


def evaluate_candidate(
    *,
    engine: _SearchEngineLike,
    cases: Iterable[EvalCase],
    top_k: int = 10,
    split: Literal["search", "holdout", "all"] = "search",
    outcomes_dir: Path | None = None,
    head_sha: str | None = None,
    repo_size_bytes: int | None = None,
    labels: Iterable[TierLabel] | None = None,
) -> ScoreCard:
    """Run the golden against ``engine`` and build a fitness ScoreCard.

    Args:
        engine: object with a ``search(query, top_k)`` method.
        cases: golden eval cases to score against. Loadable via
            :func:`stropha.eval.load_golden`.
        top_k: retrieval top-K. Default 10.
        split: which subset of ``cases`` to evaluate. ``"search"`` is
            what the Meta-Harness proposer should call by default;
            ``"holdout"`` is reserved for promotion-time generalisation
            measurement; ``"all"`` evaluates everything (regression
            sanity).
        outcomes_dir / head_sha: optional path to a WIN-164 backfill
            output. When provided the loader hydrates TierLabels so the
            outcome_score component is non-zero.
        repo_size_bytes: hint for the context_efficiency baseline. When
            None, the savings dimension defaults to 0.
        labels: explicit TierLabel iterable. If passed, takes precedence
            over ``outcomes_dir`` — the proposer can hand-craft a label
            set to score against.

    Returns the :class:`ScoreCard`.
    """
    cases_list = list(cases)
    if split != "all":
        cases_list = filter_by_split(cases_list, split)
    if not cases_list:
        raise EvaluatorError(
            f"no cases to evaluate after filtering split={split!r}"
        )

    report = run_eval(cases_list, engine, top_k=top_k)  # type: ignore[arg-type]

    tier_labels: list[TierLabel] = []
    if labels is not None:
        tier_labels = list(labels)
    elif outcomes_dir is not None:
        tier_labels = _load_labels_from_outcomes(outcomes_dir, head_sha=head_sha)

    n_hits = sum(1 for r in report.results if r.hit)
    card = build_scorecard(
        recall_at_k=report.recall_at_k,
        ndcg_at_k=report.ndcg_at_k,
        mrr=report.mrr,
        returned_tokens=_returned_tokens_from_report(report),
        baseline_tokens=_baseline_tokens_from_repo(repo_size_bytes),
        latency_ms=report.latency_p95_ms,
        labels=tier_labels,
        n_queries=report.n,
        n_hits=n_hits,
    )
    log.info(
        "evaluator.done",
        split=split,
        n_queries=report.n,
        quality=round(card.quality, 4),
        outcome_score=round(card.outcome_score, 4),
        latency_p95_ms=round(card.latency_ms, 3),
    )
    return card


def evaluate_from_golden(
    *,
    engine: _SearchEngineLike,
    golden_path: Path,
    **kwargs: Any,
) -> ScoreCard:
    """Convenience wrapper: load the golden then call :func:`evaluate_candidate`."""
    cases = load_golden(golden_path)
    return evaluate_candidate(engine=engine, cases=cases, **kwargs)


__all__ = [
    "EvaluatorError",
    "evaluate_candidate",
    "evaluate_from_golden",
]
