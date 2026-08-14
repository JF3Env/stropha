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

import copy
import json
from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, overload

from ..errors import StrophaError
from ..eval import EvalCase, EvalReport, filter_by_split, load_golden, run_eval
from ..logging import get_logger
from .fitness import ScoreCard, build_scorecard
from .outcome import TierLabel
from .trace import TraceRecorder

log = get_logger(__name__)


class EvaluatorError(StrophaError):
    """Raised when ``evaluate_candidate`` cannot satisfy a request."""


class _SearchEngineLike(Protocol):
    """Minimal slice of the SearchEngine surface the evaluator depends on."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        repo_id: int | None = None,
    ) -> list[Any]:
        ...


class EvaluationRecorder(Protocol):
    """Recorder lifecycle required by candidate evaluation tracing."""

    def record(self, event: dict[str, Any]) -> object: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Route one candidate evaluation run to its trace directory."""

    candidate_id: str
    run_id: str
    traces_dir: Path
    recorder: EvaluationRecorder | None = None

    @classmethod
    def for_candidate(
        cls,
        *,
        candidate_id: str,
        run_id: str,
        candidate_dir: Path,
        recorder: EvaluationRecorder | None = None,
    ) -> EvaluationContext:
        """Build a context for the ArtifactStore ``candidates/<id>/`` layout."""
        return cls(
            candidate_id=candidate_id,
            run_id=run_id,
            traces_dir=candidate_dir / "traces",
            recorder=recorder,
        )

    def open_recorder(self) -> EvaluationRecorder:
        """Return the injected recorder or create the standard JSONL recorder."""
        if self.recorder is not None:
            return self.recorder
        return TraceRecorder(base_dir=self.traces_dir, session_id=self.run_id)


@dataclass(frozen=True, slots=True)
class CandidateEvaluationResult:
    """One retrieval pass represented as both fitness and raw eval evidence."""

    scorecard: ScoreCard
    report: EvalReport


def _serializable_hit(hit: Any) -> dict[str, Any]:
    """Snapshot a hit's complete public surface as JSON-compatible data."""
    payload: dict[str, Any]
    if hasattr(hit, "model_dump"):
        payload = dict(hit.model_dump(mode="json"))
    elif isinstance(hit, dict):
        payload = dict(hit)
    elif is_dataclass(hit) and not isinstance(hit, type):
        payload = asdict(hit)
    else:
        try:
            payload = {key: value for key, value in vars(hit).items() if not key.startswith("_")}
        except TypeError:
            payload = {}

    for field in ("chunk_id", "rel_path", "path", "symbol", "streams"):
        if field not in payload:
            payload[field] = getattr(hit, field, None)
    path = payload.get("rel_path") or payload.get("path")
    payload["rel_path"] = path
    payload["path"] = path
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _serializable_hits(hits: Iterable[Any]) -> list[dict[str, Any]]:
    """Serialize every usable hit without letting tracing break evaluation."""
    results = []
    for hit in hits:
        try:
            results.append(_serializable_hit(hit))
        except Exception as exc:  # pragma: no cover - defensive observability boundary
            log.debug("evaluator.trace_result_failed", error=str(exc))
    return results


class _TracingSearchEngine:
    """Transparent search wrapper that records one event per retrieval call."""

    def __init__(
        self,
        engine: _SearchEngineLike,
        context: EvaluationContext,
        recorder: EvaluationRecorder,
    ) -> None:
        self._engine = engine
        self._context = context
        self._recorder = recorder

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        repo_id: int | None = None,
    ) -> list[Any]:
        import time

        start = time.perf_counter()
        try:
            if repo_id is None:
                hits = self._engine.search(query, top_k=top_k)
            else:
                hits = self._engine.search(query, top_k=top_k, repo_id=repo_id)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._record(
                query=query,
                top_k=top_k,
                repo_id=repo_id,
                results=[],
                latency_ms=latency_ms,
                returned_tokens=0,
                error={"type": type(exc).__qualname__, "message": str(exc)},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        results = _serializable_hits(hits)
        self._record(
            query=query,
            top_k=top_k,
            repo_id=repo_id,
            results=results,
            latency_ms=latency_ms,
            returned_tokens=sum(
                max(1, len(str(getattr(hit, "snippet", ""))) // 4) for hit in hits
            ),
        )
        return hits

    def reset_diagnostics(self) -> None:
        reset = getattr(self._engine, "reset_diagnostics", None)
        if callable(reset):
            reset()

    def diagnostics(self) -> dict[str, Any] | None:
        diagnostics = getattr(self._engine, "diagnostics", None)
        if not callable(diagnostics):
            return None
        result = diagnostics()
        return result if isinstance(result, dict) else None

    def _record(
        self,
        *,
        query: str,
        top_k: int,
        repo_id: int | None,
        results: list[dict[str, Any]],
        latency_ms: float,
        returned_tokens: int,
        error: dict[str, str] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "tool": "search_code",
            "candidate_id": self._context.candidate_id,
            "run_id": self._context.run_id,
            "query": query,
            "params": {"top_k": top_k, "repo_id": repo_id},
            "results": results,
            "latency_ms": latency_ms,
            "returned_tokens": returned_tokens,
        }
        if error is not None:
            event["error"] = error
        try:
            self._recorder.record(event)
        except Exception as exc:  # pragma: no cover - defensive observability boundary
            log.debug("evaluator.trace_record_failed", error=str(exc))


def _close_recorder(recorder: EvaluationRecorder) -> None:
    """Best-effort drain and close without changing evaluation outcomes."""
    try:
        recorder.flush()
    except Exception as exc:  # pragma: no cover - defensive observability boundary
        log.warning("evaluator.trace_flush_failed", error=str(exc))
    try:
        recorder.close()
    except Exception as exc:  # pragma: no cover - defensive observability boundary
        log.warning("evaluator.trace_close_failed", error=str(exc))


def _load_labels_from_outcomes(
    outcomes_dir: Path,
    head_sha: str | None = None,
    chunk_ids: Collection[str] | None = None,
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
        hit_outcomes = payload.get("hit_outcomes")
        if chunk_ids is not None and isinstance(hit_outcomes, list):
            wanted = set(chunk_ids)
            for row in hit_outcomes:
                label_data = row.get("label") if isinstance(row, dict) else None
                if not isinstance(label_data, dict) or label_data.get("chunk_id") not in wanted:
                    continue
                try:
                    labels.append(TierLabel.model_validate(label_data))
                except ValueError:
                    continue
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
    """Sum actual snippet token estimates captured by the eval runner."""
    captured = sum(r.returned_tokens for r in report.results)
    if captured:
        return captured
    # Compatibility for synthetic EvalResults built without actual hits.
    return sum(50 for r in report.results for rel in r.relevances if rel > 0)


_QUERY_CONFIG_FIELDS = {
    "hyde": "hyde_enabled",
    "query_rewrite": "query_rewrite_enabled",
    "query_rewrite_max_terms": "query_rewrite_max_terms",
    "multi_query": "multi_query_enabled",
    "multi_query_count": "multi_query_count",
    "multi_query_min_terms": "multi_query_min_terms",
    "recursive": "recursive_enabled",
    "recursive_adjacency": "recursive_adjacency",
    "adaptive_weighting": "adaptive_weighting",
    "candidate_k": "candidate_k",
    "rrf_k": "rrf_k",
}


def query_candidate_config(
    base_config: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply a flat Meta-Harness query candidate to a resolved pipeline config.

    Indexing stages deliberately stay inherited from ``base_config``. This
    prevents scoring an embedder/enricher proposal against vectors produced by
    a different indexing identity.
    """
    resolved = copy.deepcopy(base_config)
    retrieval = resolved.setdefault("retrieval", {}).setdefault("config", {})
    remaining = dict(candidate)
    if "reranker" in remaining:
        reranker = remaining.pop("reranker")
        if reranker in (None, "none", "null"):
            retrieval["reranker"] = None
        elif isinstance(reranker, dict):
            retrieval["reranker"] = copy.deepcopy(reranker)
        else:
            retrieval["reranker"] = {"adapter": str(reranker)}
    for candidate_key, config_key in _QUERY_CONFIG_FIELDS.items():
        if candidate_key in remaining:
            retrieval[config_key] = remaining.pop(candidate_key)
    if remaining:
        raise EvaluatorError(
            "candidate contains indexing or unsupported dimensions: "
            + ", ".join(sorted(remaining))
        )
    return resolved


@overload
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
    repo_id: int | None = None,
    evaluation_context: EvaluationContext | None = None,
    return_result: Literal[False] = False,
) -> ScoreCard: ...


@overload
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
    repo_id: int | None = None,
    evaluation_context: EvaluationContext | None = None,
    return_result: Literal[True],
) -> CandidateEvaluationResult: ...


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
    repo_id: int | None = None,
    evaluation_context: EvaluationContext | None = None,
    return_result: bool = False,
) -> ScoreCard | CandidateEvaluationResult:
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
        evaluation_context: optional candidate/run trace routing. Every
            attempted golden query is written to ``traces_dir`` and the
            recorder is flushed and closed before this function returns
            or raises.

    Returns the :class:`ScoreCard`.
    """
    recorder = evaluation_context.open_recorder() if evaluation_context else None
    traced_engine: _SearchEngineLike = engine
    if evaluation_context is not None and recorder is not None:
        traced_engine = _TracingSearchEngine(engine, evaluation_context, recorder)
    try:
        cases_list = list(cases)
        if split != "all":
            cases_list = filter_by_split(cases_list, split)
        if not cases_list:
            raise EvaluatorError(
                f"no cases to evaluate after filtering split={split!r}"
            )

        report = run_eval(
            cases_list,
            traced_engine,  # type: ignore[arg-type]
            top_k=top_k,
            repo_id=repo_id,
        )

        tier_labels: list[TierLabel] = []
        if labels is not None:
            tier_labels = list(labels)
        elif outcomes_dir is not None:
            returned_chunk_ids = {
                chunk_id
                for result in report.results
                for chunk_id in result.returned_chunk_ids
            }
            tier_labels = _load_labels_from_outcomes(
                outcomes_dir,
                head_sha=head_sha,
                chunk_ids=returned_chunk_ids,
            )

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
        if return_result:
            return CandidateEvaluationResult(scorecard=card, report=report)
        return card
    finally:
        if recorder is not None:
            _close_recorder(recorder)


def evaluate_from_golden(
    *,
    engine: _SearchEngineLike,
    golden_path: Path,
    evaluation_context: EvaluationContext | None = None,
    **kwargs: Any,
) -> ScoreCard:
    """Convenience wrapper: load the golden then call :func:`evaluate_candidate`."""
    try:
        cases = load_golden(golden_path)
    except Exception:
        if evaluation_context is not None:
            _close_recorder(evaluation_context.open_recorder())
        raise
    return evaluate_candidate(
        engine=engine,
        cases=cases,
        evaluation_context=evaluation_context,
        **kwargs,
    )


__all__ = [
    "CandidateEvaluationResult",
    "EvaluationContext",
    "EvaluationRecorder",
    "EvaluatorError",
    "evaluate_candidate",
    "evaluate_from_golden",
    "query_candidate_config",
]
