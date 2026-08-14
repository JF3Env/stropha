"""Token-budget result packing (WIN-208).

The paper (Lee et al. §4.1) shows the Meta-Harness wins on text
classification by +7.7 accuracy points AND simultaneously consumes
~4× fewer tokens. The "fewer tokens" half of that gain comes from
*context shaping*: instead of returning a fixed ``top_k`` slice of
chunks regardless of size, the retriever packs as many high-rank
chunks as the caller's result-token budget allows and stops there.

Why a separate packer
---------------------

Today the SearchEngine returns ``top_k`` results no matter what.
Three failure modes follow:

1. **Over-packing**: top_k=10 of 1000-line chunks blows the agent's
   context window. The agent dumps half the response into the
   "previous tool output" buffer and runs out of room for reasoning.
2. **Under-packing**: top_k=3 of 30-line stubs leaves most of the
   result budget unused. The agent had room for the surrounding
   methods but doesn't get them.
3. **No back-pressure**: the caller has no way to say "give me as
   much as fits in 4K tokens" — only "give me 10 things".

A budget-aware packer solves all three: the caller passes
``context_budget=N`` (tokens), the packer walks the selected policy
order, summing the per-hit token estimate, and stops at the first hit
that would overflow. Score order is the default; opt-in diversity
stably moves redundant hits after distinct evidence. The remaining tail is returned as
``spilled`` for observability (so the operator can see "we had 15
more relevant hits but ran out of room"). This is a budget for the
returned snippets, not the caller's entire model context window.

Spill policy
------------

The packer is **greedy first-fit with monotonic ordering**:

- Walk hits in the selected policy order.
- For each hit, if ``current_total + estimate(hit) <= budget``, pack.
- Otherwise stop — do NOT skip and try later hits. Keeping the selected
  order intact makes budget behavior deterministic.

One exception: when the very first hit alone exceeds the budget,
include it anyway and mark ``truncated=True``. Returning empty in
that case conflates "budget too tight" with "nothing found", which
breaks the agent's recovery logic. Better to surface a single
oversize hit with a flag than to be silent.

Token estimator
---------------

Default estimator is ``estimate_content_tokens`` from
``retrieval/savings.py`` — a cheap whitespace+punctuation heuristic
that takes microseconds per hit. Callers that need exact counts (for
billing against a precise budget) can pass a tokenizer-backed
estimator at call time. The packer is contract-agnostic: any
callable ``str -> int`` will work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..logging import get_logger
from ..models import SearchHit
from .savings import estimate_content_tokens

log = get_logger(__name__)


TokenEstimatorFn = Callable[[str], int]


@dataclass(frozen=True)
class PackingResult:
    """A greedy-packed list of hits plus the spilled tail.

    ``packed`` keeps the selected policy order (score order unless
    diversification is enabled). ``spilled`` is the remaining ordered tail;
    its presence tells the caller "you had more relevant hits, bump
    the budget". ``total_tokens`` is the sum the packer measured —
    it can be slightly above ``budget_tokens`` when ``truncated`` is
    True (the single-first-hit oversize edge case).
    """

    packed: list[SearchHit]
    spilled: list[SearchHit] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    truncated: bool = False
    """``True`` only in the single-first-hit-oversize escape path.
    Indicates that ``packed`` contains exactly one hit whose token
    estimate is larger than ``budget_tokens``. Callers should treat
    this as a soft warning, not an error."""
    deferred: list[SearchHit] = field(default_factory=list)
    """Hits moved to the stable second pass by the diversity policy."""


@dataclass(frozen=True)
class DiversityStats:
    """Redundancy measurements for one ordered result list."""

    unique_files: int = 0
    unique_symbols: int = 0
    duplicate_entities: int = 0
    max_chunks_per_file: int = 0


def _repo_key(hit: SearchHit) -> str:
    repo = getattr(hit, "repo", None)
    return str(getattr(repo, "normalized_key", ""))


def _file_key(hit: SearchHit) -> tuple[str, str]:
    path = getattr(hit, "rel_path", None) or getattr(hit, "path", "")
    return _repo_key(hit), str(path)


def _entity_key(hit: SearchHit) -> tuple[str, str, str]:
    symbol = str(getattr(hit, "symbol", None) or "").strip()
    kind = str(getattr(hit, "kind", "")).lower()
    if symbol and kind in {"constructor", "method"} and "." in symbol:
        symbol = symbol.rsplit(".", 1)[0]
    chunk_id = str(getattr(hit, "chunk_id", ""))
    return (*_file_key(hit), symbol or chunk_id)


def measure_diversity(
    hits: Sequence[SearchHit],
    *,
    max_chunks_per_file: int = 2,
) -> DiversityStats:
    """Measure file, symbol, and policy-level redundancy in ``hits``."""
    files = Counter(_file_key(hit) for hit in hits)
    symbols = {
        (*_file_key(hit), getattr(hit, "symbol", None))
        for hit in hits
        if getattr(hit, "symbol", None)
    }
    _, deferred = diversify_hits(hits, max_chunks_per_file=max_chunks_per_file)
    return DiversityStats(
        unique_files=len(files),
        unique_symbols=len(symbols),
        duplicate_entities=len(deferred),
        max_chunks_per_file=max(files.values(), default=0),
    )


def diversify_hits(
    hits: Sequence[SearchHit],
    *,
    max_chunks_per_file: int = 2,
    protected_prefix: int = 1,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Prefer new file/parent-symbol entities after a protected prefix.

    Both passes preserve input order, protected hits are never displaced, and
    no hit is discarded. Repository identity is part of every grouping key.
    """
    if not hits:
        return [], []

    file_limit = max(1, max_chunks_per_file)
    preferred: list[SearchHit] = []
    deferred: list[SearchHit] = []
    seen_entities: set[tuple[str, str, str]] = set()
    file_counts: Counter[tuple[str, str]] = Counter()
    prefix_length = min(len(hits), max(0, protected_prefix))

    for index, hit in enumerate(hits):
        file_key = _file_key(hit)
        entity_key = _entity_key(hit)
        if index < prefix_length or (
            entity_key not in seen_entities and file_counts[file_key] < file_limit
        ):
            preferred.append(hit)
            seen_entities.add(entity_key)
            file_counts[file_key] += 1
        else:
            deferred.append(hit)

    return preferred + deferred, deferred


def pack_to_budget(
    hits: list[SearchHit],
    budget_tokens: int,
    *,
    estimator: TokenEstimatorFn = estimate_content_tokens,
    diversify: bool = False,
    max_chunks_per_file: int = 2,
    protected_prefix: int = 1,
) -> PackingResult:
    """Greedy-pack ``hits`` until ``budget_tokens`` is reached.

    Args:
        hits: search results in score order (highest first). The
            packer does not re-sort; the caller is responsible for
            having ranked the input.
        budget_tokens: maximum cumulative tokens the packer is
            allowed to return. Zero or negative values produce an
            empty pack (no escape hatch) — the caller asked for
            nothing.
        estimator: callable ``str -> int`` for per-hit token cost.
            Defaults to ``estimate_content_tokens``.
        diversify: whether to prefer unseen repository/file/parent-symbol
            entities before filling from deferred hits.
        max_chunks_per_file: first-pass cap per repository/file. The stable
            second pass still retains every deferred hit.
        protected_prefix: number of leading hits that diversification cannot
            defer. Search uses this to preserve every pinned hit.

    Returns:
        :class:`PackingResult` with packed + spilled split.
    """
    ordered_hits, deferred = (
        diversify_hits(
            hits,
            max_chunks_per_file=max_chunks_per_file,
            protected_prefix=protected_prefix,
        )
        if diversify
        else (list(hits), [])
    )

    # Defensive: a zero/negative budget is a legitimate "do nothing"
    # request (e.g. caller is probing the spill count). We honour
    # the contract and don't trigger the first-hit-oversize escape.
    if budget_tokens <= 0:
        return PackingResult(
            packed=[],
            spilled=ordered_hits,
            total_tokens=0,
            budget_tokens=max(0, budget_tokens),
            deferred=deferred,
        )

    packed: list[SearchHit] = []
    total = 0
    spill_start = len(ordered_hits)  # index of the first un-packed hit

    for idx, hit in enumerate(ordered_hits):
        cost = max(0, estimator(hit.snippet or ""))
        # First-hit-oversize escape: if no hit has been packed yet
        # and even this one exceeds the budget, include it anyway
        # and signal truncation. Returning empty in this case
        # conflates "budget too tight" with "nothing found".
        if not packed and cost > budget_tokens:
            packed.append(hit)
            total = cost
            spill_start = idx + 1
            log.info(
                "packing.first_hit_oversize",
                budget=budget_tokens,
                cost=cost,
                snippet_preview=(hit.snippet or "")[:60],
            )
            return PackingResult(
                packed=packed,
                spilled=list(ordered_hits[spill_start:]),
                total_tokens=total,
                budget_tokens=budget_tokens,
                truncated=True,
                deferred=deferred,
            )
        if total + cost > budget_tokens:
            # Stop on first overflow — preserve the selected policy order.
            spill_start = idx
            break
        packed.append(hit)
        total += cost
    else:
        # Loop completed without breaking: nothing spilled.
        spill_start = len(ordered_hits)

    spilled = list(ordered_hits[spill_start:])
    return PackingResult(
        packed=packed,
        spilled=spilled,
        total_tokens=total,
        budget_tokens=budget_tokens,
        deferred=deferred,
    )


__all__ = [
    "DiversityStats",
    "PackingResult",
    "TokenEstimatorFn",
    "diversify_hits",
    "measure_diversity",
    "pack_to_budget",
]
