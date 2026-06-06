"""Speculative retrieval prefetch (WIN-242, EPIC 1.10 #33).

Most agent sessions exhibit predictable follow-up patterns. After
"what is ``FooClass``" comes "tests for FooClass" 60% of the time;
after "find the bug in ``parser``" comes "callers of parser" or
"tests for parser". WIN-242 ships the primitive that exploits the
predictability: forecast the next 2-3 queries and pre-compute them
during idle time so the cache is already warm when the agent
actually asks.

The module has two halves:

1. ``predict_next_queries`` — pure heuristics over the recent query
   history + the recent result set. Produces ``SpeculativeQuery``
   records (query string + reason + confidence). No I/O.

2. ``prefetch`` — orchestrator that runs the predicted queries via
   a caller-supplied ``search_fn`` (concurrently via WIN-237's
   ``parallel_fanout``) and optionally stores results in a cache
   (caller passes ``cache_put`` — typically the WIN-230
   ``cross_session_cache.store``).

Both halves are standalone primitives — no auto-integration into
SearchEngine yet. Matches the WIN-230 / WIN-236 / WIN-237 / WIN-238
/ WIN-239 / WIN-240 / WIN-241 pattern.

Prediction heuristics
---------------------

The predictor cascades over signals (most-recent query first):

- Code anchor in top hits → ``"tests for {symbol}"``,
  ``"callers of {symbol}"``, ``"docs for {symbol}"`` (graph-gated).
- Question query ("what is X") → mirror toward "tests for X" /
  "where is X" / "examples of X".
- Path-narrowed query (top hits share a file) → broaden to "imports
  of {file}".
- Compound query → split into clauses and predict each clause's
  follow-up.

Predictions are deduplicated against the recent_queries deque so
we don't re-fetch what the agent already asked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .parallel_fanout import parallel_fanout
from .suggest_next import CODE_KINDS, _looks_compound, _looks_question


@dataclass(frozen=True)
class SpeculativeQuery:
    """A single predicted next query.

    ``confidence`` is a soft score in ``[0, 1]``; the caller may
    threshold low-confidence predictions before paying for them.
    """

    query: str
    reason: str = ""
    confidence: float = 0.5


@dataclass(frozen=True)
class PrefetchResult:
    """Aggregate outcome of :func:`prefetch`.

    ``results`` maps query → list of hits returned by ``search_fn``.
    ``cached`` is the parallel list of queries successfully written
    to the cache (subset of ``results.keys()``). ``failed`` records
    which queries raised inside ``search_fn`` / ``cache_put``.
    ``elapsed_ms`` is total wall-clock.
    """

    results: dict[str, list[Any]] = field(default_factory=dict)
    cached: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _hit_kind(hit: Any) -> str:
    return str(getattr(hit, "kind", "") or "")


def _hit_symbol(hit: Any) -> str:
    return str(getattr(hit, "symbol", "") or "")


def _hit_path(hit: Any) -> str:
    return str(getattr(hit, "rel_path", "") or "")


def _first_code_anchor(hits: list[Any]) -> str | None:
    for hit in hits:
        if _hit_kind(hit) in CODE_KINDS and _hit_symbol(hit):
            return _hit_symbol(hit)
    return None


def _dominant_path(hits: list[Any], threshold: float = 0.75) -> str | None:
    if not hits:
        return None
    counts: dict[str, int] = {}
    for hit in hits:
        path = _hit_path(hit)
        if path:
            counts[path] = counts.get(path, 0) + 1
    if not counts:
        return None
    top_path, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count / float(len(hits)) >= threshold:
        return top_path
    return None


def predict_next_queries(
    recent_queries: list[str],
    recent_hits: list[Any],
    *,
    max_predictions: int = 3,
    has_graph: bool = True,
) -> list[SpeculativeQuery]:
    """Forecast the next ``max_predictions`` queries the agent will issue.

    Args:
        recent_queries: the agent's history (most recent first).
            Empty list disables history-based predictions but does
            not disable hit-based ones.
        recent_hits: SearchHit-like objects from the most recent
            retrieval. Drives the code-anchor and path-dominance
            heuristics.
        max_predictions: ceiling on the returned list. Always
            sorted by confidence DESC.
        has_graph: gates graph-flavoured predictions ("callers of
            X" requires the mirror).

    Returns:
        Up to ``max_predictions`` :class:`SpeculativeQuery` records,
        deduplicated against ``recent_queries``.
    """
    if max_predictions <= 0:
        return []

    history = [q.strip() for q in recent_queries if q and q.strip()]
    seen_lower = {q.lower() for q in history}
    predictions: list[SpeculativeQuery] = []

    anchor = _first_code_anchor(recent_hits)
    if anchor:
        predictions.append(SpeculativeQuery(
            query=f"tests for {anchor}",
            reason=(
                f"Top hit is code-like ({anchor}); agents typically "
                "follow with a tests-for lookup."
            ),
            confidence=0.80,
        ))
        if has_graph:
            predictions.append(SpeculativeQuery(
                query=f"callers of {anchor}",
                reason=(
                    f"Anchor symbol {anchor} — callers reveal the "
                    "blast radius of any change."
                ),
                confidence=0.75,
            ))
        predictions.append(SpeculativeQuery(
            query=f"docs for {anchor}",
            reason=(
                f"Anchor symbol {anchor} — design notes / ADRs "
                "often follow."
            ),
            confidence=0.55,
        ))

    last_query = history[0] if history else ""
    if last_query and _looks_question(last_query):
        # Mirror toward concrete forms: questions usually precede
        # "where is X" / "examples of X" follow-ups.
        tail = _strip_question_prefix(last_query)
        if tail:
            predictions.append(SpeculativeQuery(
                query=f"examples of {tail}",
                reason=(
                    "Last query was a question — agents typically "
                    "follow with concrete examples."
                ),
                confidence=0.50,
            ))

    dom_path = _dominant_path(recent_hits)
    if dom_path:
        predictions.append(SpeculativeQuery(
            query=f"imports of {dom_path}",
            reason=(
                f"Top hits cluster in {dom_path} — imports-of often "
                "reveals dependents."
            ),
            confidence=0.45,
        ))

    if last_query and _looks_compound(last_query):
        # Decompose the compound and predict each clause's follow-up
        # (best-effort: a real decomposer is in WIN-160's module).
        for clause in _naive_split(last_query):
            predictions.append(SpeculativeQuery(
                query=clause,
                reason=(
                    "Compound query — running clauses individually "
                    "is a typical follow-up."
                ),
                confidence=0.40,
            ))

    # Dedup against history.
    deduped: list[SpeculativeQuery] = []
    seen_pred: set[str] = set()
    for p in predictions:
        norm = p.query.strip().lower()
        if not norm or norm in seen_lower or norm in seen_pred:
            continue
        seen_pred.add(norm)
        deduped.append(p)

    deduped.sort(key=lambda p: p.confidence, reverse=True)
    return deduped[:max_predictions]


def _strip_question_prefix(q: str) -> str:
    lowered = q.lower().strip()
    for prefix in (
        "how does ", "how do ", "what is ", "what are ",
        "where is ", "where are ", "why does ", "when does ",
    ):
        if lowered.startswith(prefix):
            return q.strip()[len(prefix):]
    return q.strip()


def _naive_split(q: str) -> list[str]:
    """Split on " and " / " then " / " plus " case-insensitively."""
    splitters = (" and ", " then ", " plus ", " & ")
    out: list[str] = []
    remainder = q
    while remainder:
        lower = remainder.lower()
        cut_at: int | None = None
        cut_len = 0
        for token in splitters:
            idx = lower.find(token)
            if idx >= 0 and (cut_at is None or idx < cut_at):
                cut_at = idx
                cut_len = len(token)
        if cut_at is None:
            piece = remainder.strip()
            if piece:
                out.append(piece)
            break
        piece = remainder[:cut_at].strip()
        if piece:
            out.append(piece)
        remainder = remainder[cut_at + cut_len:]
    seen: set[str] = set()
    deduped: list[str] = []
    for piece in out:
        if piece and piece not in seen:
            seen.add(piece)
            deduped.append(piece)
    return deduped or [q]


def prefetch(
    queries: list[SpeculativeQuery],
    *,
    search_fn: Callable[[str], list[Any]],
    cache_put: Callable[[str, list[Any]], None] | None = None,
    max_concurrency: int = 2,
    timeout: float | None = None,
) -> PrefetchResult:
    """Run ``queries`` in parallel and optionally cache the results.

    Uses WIN-237's ``parallel_fanout`` so failures in individual
    queries are isolated (one query raising doesn't kill the others).

    Args:
        queries: predicted queries to run.
        search_fn: callable invoking a single retrieval pass.
        cache_put: optional callable taking ``(query, results)``.
            Exceptions from ``cache_put`` are captured in ``failed``
            so a flaky cache doesn't poison the prefetch.
        max_concurrency: pool size; defaults to 2 to balance latency
            vs. resource pressure on the embedder / FTS layer.
        timeout: total wall-clock cap forwarded to ``parallel_fanout``.

    Returns:
        :class:`PrefetchResult` with the merged results, the
        cached-queries list, and per-query failure notes.
    """
    if not queries:
        return PrefetchResult()

    tasks: dict[str, Callable[[], object]] = {}
    for q in queries:
        # Capture ``q.query`` by default-argument so the closure binds the
        # value at iteration time, not at call time. The ``str`` annotation
        # is explicit so mypy can resolve the lambda return type.
        def _make_task(query: str = q.query) -> object:
            return search_fn(query)

        tasks[q.query] = _make_task

    fanout = parallel_fanout(
        tasks,
        fuse_fn=lambda partial: partial,
        max_workers=max(1, int(max_concurrency)),
        timeout=timeout,
    )

    results: dict[str, list[Any]] = {}
    failed: dict[str, str] = {}
    for name, outcome in fanout.outcomes.items():
        if outcome.error is not None:
            failed[name] = f"search_fn: {outcome.error!r}"
            continue
        if outcome.cancelled:
            failed[name] = "cancelled (timeout or pool teardown)"
            continue
        # ``outcome.result`` is ``object`` because the parallel-fanout helper
        # is generic; coerce defensively at this seam — the contract here is
        # that ``search_fn`` returns an iterable of result rows.
        raw = outcome.result if outcome.result is not None else []
        results[name] = list(raw)  # type: ignore[call-overload]

    cached: list[str] = []
    if cache_put is not None:
        for query, hits in results.items():
            try:
                cache_put(query, hits)
            except Exception as exc:
                failed[query] = f"cache_put: {exc!r}"
                continue
            cached.append(query)

    return PrefetchResult(
        results=results,
        cached=cached,
        failed=failed,
        elapsed_ms=fanout.elapsed_ms,
    )


__all__ = [
    "PrefetchResult",
    "SpeculativeQuery",
    "predict_next_queries",
    "prefetch",
]
