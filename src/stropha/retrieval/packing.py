"""Token-budget result packing (WIN-208).

The paper (Lee et al. §4.1) shows the Meta-Harness wins on text
classification by +7.7 accuracy points AND simultaneously consumes
~4× fewer tokens. The "fewer tokens" half of that gain comes from
*context shaping*: instead of returning a fixed ``top_k`` slice of
chunks regardless of size, the retriever packs as many high-rank
chunks as the caller's context budget allows and stops there.

Why a separate packer
---------------------

Today the SearchEngine returns ``top_k`` results no matter what.
Three failure modes follow:

1. **Over-packing**: top_k=10 of 1000-line chunks blows the agent's
   context window. The agent dumps half the response into the
   "previous tool output" buffer and runs out of room for reasoning.
2. **Under-packing**: top_k=3 of 30-line stubs leaves 90% of the
   agent's budget unused. The agent had room for the surrounding
   methods but doesn't get them.
3. **No back-pressure**: the caller has no way to say "give me as
   much as fits in 4K tokens" — only "give me 10 things".

A budget-aware packer solves all three: the caller passes
``context_budget=N`` (tokens), the packer walks the score-ranked
hits in order, summing the per-hit token estimate, and stops at the
first hit that would overflow. The truncated tail is returned as
``spilled`` for observability (so the operator can see "we had 15
more relevant hits but ran out of room").

Spill policy
------------

The packer is **greedy first-fit with monotonic ordering**:

- Walk hits in score-rank order.
- For each hit, if ``current_total + estimate(hit) <= budget``, pack.
- Otherwise stop — do NOT skip and try later hits. Score order is
  the caller's "goodness" signal; keeping it intact lets downstream
  reranking and the agent's own attention work the way they should.

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

from collections.abc import Callable
from dataclasses import dataclass, field

from ..logging import get_logger
from ..models import SearchHit
from .savings import estimate_content_tokens

log = get_logger(__name__)


TokenEstimatorFn = Callable[[str], int]


@dataclass(frozen=True)
class PackingResult:
    """A greedy-packed list of hits plus the spilled tail.

    ``packed`` keeps the score-ranked order from the search engine
    (no re-sorting). ``spilled`` is the tail that would not fit;
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


def pack_to_budget(
    hits: list[SearchHit],
    budget_tokens: int,
    *,
    estimator: TokenEstimatorFn = estimate_content_tokens,
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

    Returns:
        :class:`PackingResult` with packed + spilled split.
    """
    # Defensive: a zero/negative budget is a legitimate "do nothing"
    # request (e.g. caller is probing the spill count). We honour
    # the contract and don't trigger the first-hit-oversize escape.
    if budget_tokens <= 0:
        return PackingResult(
            packed=[],
            spilled=list(hits),
            total_tokens=0,
            budget_tokens=max(0, budget_tokens),
        )

    packed: list[SearchHit] = []
    total = 0
    spill_start = len(hits)  # index of the first un-packed hit

    for idx, hit in enumerate(hits):
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
                spilled=list(hits[spill_start:]),
                total_tokens=total,
                budget_tokens=budget_tokens,
                truncated=True,
            )
        if total + cost > budget_tokens:
            # Stop on first overflow — preserve score-rank monotonicity.
            spill_start = idx
            break
        packed.append(hit)
        total += cost
    else:
        # Loop completed without breaking: nothing spilled.
        spill_start = len(hits)

    spilled = list(hits[spill_start:])
    return PackingResult(
        packed=packed,
        spilled=spilled,
        total_tokens=total,
        budget_tokens=budget_tokens,
    )


__all__ = [
    "PackingResult",
    "TokenEstimatorFn",
    "pack_to_budget",
]
