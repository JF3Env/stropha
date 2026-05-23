"""One-shot context assembly (WIN-209).

An agent that needs to understand "how does X work" today has to
chain several MCP calls:

  1. ``search_code`` to find the primary spans.
  2. ``find_callers`` on each top hit to see what depends on them.
  3. ``find_tests_for`` to see how the behaviour is exercised.
  4. Sometimes ``get_file_outline`` to plan a deeper read.

Each call is a turn of the agent loop. Each turn costs latency,
tokens, and (for the agent's planner) cognitive overhead. WIN-209
collapses the workflow into a single ``assemble_context`` tool that
runs the steps internally, packs the results under a token budget,
and returns a ready-to-paste markdown bundle.

The bundle is the *output medium*. It's organised by provenance
(primary hits / callers / tests) with file:line:symbol headers that
the agent can paste straight into a follow-up reasoning step or a
human review comment. No JSON wrapper, no parsing — just markdown.

Why a separate module
---------------------

``assemble_context`` is composition logic, not a new retrieval
primitive. It calls into ``SearchEngine.search`` (WIN-208 packing
included) and the graph helpers (WIN-1.5b ``find_callers`` /
``find_tests_for``) — orchestration only. Keeping it out of
``search.py`` keeps that file focused on the three-stream RRF flow.

The token-budget contract is the same as WIN-208: greedy first-fit,
score-order monotonic, with a truncation-aware spill counter. The
budget covers BOTH the primary hits and the optional caller/test
extensions — when ``include_callers=True`` the extension shares the
budget with the primary set, so the operator's budget is honoured
end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from ..models import SearchHit
from .packing import pack_to_budget
from .savings import estimate_content_tokens

log = get_logger(__name__)


# Sections appear in this order inside the rendered bundle. Each entry
# is (section key, human-readable header). The agent reads top-down so
# primary hits come first; graph extensions follow.
_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("primary", "Primary code"),
    ("callers", "Callers"),
    ("tests", "Tests"),
)


@dataclass(frozen=True)
class AssembleSection:
    """One section of a rendered bundle (primary / callers / tests)."""

    key: str
    title: str
    hits: list[SearchHit] = field(default_factory=list)


@dataclass(frozen=True)
class AssembleContextResult:
    """The full ``assemble_context`` outcome.

    ``bundle`` is the ready-to-paste markdown. ``sections`` is the
    structured view per provenance so callers that want JSON can
    bypass the rendered text. ``total_tokens`` is the packer's
    measurement of what landed in ``bundle`` (sum across sections).
    ``spilled_count`` is the tail dropped because the budget ran out
    across all sections.
    """

    task: str
    bundle: str
    sections: list[AssembleSection]
    total_tokens: int
    budget_tokens: int
    spilled_count: int
    truncated_oversize: bool = False


def _format_span(hit: SearchHit) -> str:
    """Render one hit as a markdown code-block with provenance header.

    Heading format::

        ### {rel_path}:{start_line}-{end_line} — {symbol or kind}

    Followed by a fenced code block tagged with the chunk's language
    when known. The language tag lets the agent's syntax highlighter
    pick the right grammar.
    """
    symbol = hit.symbol or hit.kind or "code"
    repo_hint = ""
    if hit.repo and hit.repo.normalized_key:
        # Show repo only when the bundle could span multiple repos
        # (the rendering layer doesn't know upstream, so we always
        # emit it as a leading subtle marker).
        repo_hint = f"  *(repo: {hit.repo.normalized_key})*"
    header = (
        f"### {hit.rel_path}:{hit.start_line}-{hit.end_line} — {symbol}"
    )
    fence_lang = (hit.language or "").lower()
    body = hit.snippet or ""
    return f"{header}{repo_hint}\n```{fence_lang}\n{body}\n```"


def _render_bundle(task: str, sections: list[AssembleSection]) -> str:
    """Stitch sections into the final markdown bundle.

    Empty sections are skipped — the agent shouldn't see a "## Tests"
    heading followed by nothing. The top-level ``# Context for: ...``
    header always appears so the agent can scan past it deterministically.
    """
    parts: list[str] = [f"# Context for: {task.strip()}", ""]
    for sec in sections:
        if not sec.hits:
            continue
        parts.append(f"## {sec.title} ({len(sec.hits)} spans)")
        parts.append("")
        for hit in sec.hits:
            parts.append(_format_span(hit))
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _dedupe_in_order(seen: set[str], hits: Iterable[SearchHit]) -> list[SearchHit]:
    """Filter out hits whose chunk_id is already in ``seen``.

    Mutates ``seen`` in place — callers chain calls to walk each
    section. The dedupe is by ``chunk_id`` because the same code span
    might surface from search and from a graph extension; showing it
    twice burns tokens for no information gain.
    """
    fresh: list[SearchHit] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        fresh.append(hit)
    return fresh


def assemble_context(
    task: str,
    *,
    search_engine: Any,
    context_budget: int,
    top_k: int = 10,
    include_callers: bool = False,
    include_tests: bool = False,
    storage: Any = None,
    diff_scope: bool = False,
    diff_repo: Any = None,
    diff_ref_a: str | None = None,
    diff_ref_b: str | None = None,
) -> AssembleContextResult:
    """One-shot assembly: search → optional graph extensions → pack → render.

    Args:
        task: natural-language task description that the agent is about
            to work on.
        search_engine: a :class:`SearchEngine` instance used for the
            primary retrieval pass.
        context_budget: total token budget across all sections. Required
            (positive) — without it the bundle has no upper bound.
        top_k: candidate count fetched from the primary search before
            packing. Defaults to 10.
        include_callers: when True, also fetch up-the-graph callers for
            each primary hit (via :func:`stropha.retrieval.graph.find_callers`)
            and pack them into a "Callers" section.
        include_tests: when True, also fetch tests-that-cover for each
            primary hit (via :func:`stropha.retrieval.graph.find_tests_for`)
            and pack them into a "Tests" section.
        storage: storage handle for the graph helpers. Required when
            ``include_callers`` or ``include_tests`` is True; ignored
            otherwise. Same handle the SearchEngine was built with.
        diff_scope: WIN-211 — when True, filter the primary search
            results to only chunks whose ``rel_path`` is in the
            current git diff at ``diff_repo``. Use this for PR
            review workflows where the global top-K is too noisy.
        diff_repo: path to the git repository to compute the diff
            against. Required when ``diff_scope=True``.
        diff_ref_a: starting git ref for the diff. ``None`` = working
            tree (uncommitted + staged) diff against HEAD.
        diff_ref_b: ending git ref. Defaults to ``HEAD`` when
            ``diff_ref_a`` is set.

    Returns:
        :class:`AssembleContextResult` with the rendered bundle and the
        structured per-section view.
    """
    if context_budget <= 0:
        raise ValueError(
            "context_budget must be positive — assemble_context has no "
            "implicit cap, the caller must declare one"
        )

    # 1. Primary search (re-using SearchEngine, including its WIN-208
    #    packing path) with a generous candidate pool — we'll pack
    #    again at the bundle level so the budget covers extensions too.
    #    Diff-scoping (WIN-211) over-samples top_k so post-filtering
    #    against the diff doesn't leave an empty list.
    primary_search_k = top_k
    if diff_scope:
        primary_search_k = max(top_k * 5, 30)
    primary_hits: list[SearchHit] = list(
        search_engine.search(task, top_k=primary_search_k)
    )

    if diff_scope:
        # Filter to only files in the diff. On failure (git missing,
        # not a repo, bad ref) we log + fall back to the unfiltered
        # list rather than crashing — diff-scope is a refinement,
        # not a hard requirement of the agent's call.
        if diff_repo is None:
            log.warning(
                "assemble.diff_scope_no_repo",
                message="diff_scope=True requires diff_repo; ignoring scope",
            )
        else:
            from .diff_scope import DiffScopeError, diff_paths, filter_hits_to_diff

            try:
                paths = diff_paths(
                    diff_repo, ref_a=diff_ref_a, ref_b=diff_ref_b,
                )
            except DiffScopeError as exc:
                log.warning(
                    "assemble.diff_scope_failed",
                    repo=str(diff_repo), error=str(exc),
                )
            else:
                filtered = filter_hits_to_diff(primary_hits, paths)
                log.info(
                    "assemble.diff_scope_applied",
                    diff_files=len(paths),
                    before=len(primary_hits),
                    after=len(filtered),
                )
                primary_hits = filtered[:top_k]

    seen_chunk_ids: set[str] = set()
    primary_unique = _dedupe_in_order(seen_chunk_ids, primary_hits)

    # 2. Optional graph extensions — pulled in score-rank order from the
    #    primary hits. The graph helpers return their own ranked lists
    #    so we extend per-primary-hit with simple concatenation; the
    #    final packer will trim anyway.
    caller_hits: list[SearchHit] = []
    test_hits: list[SearchHit] = []
    if include_callers and storage is not None:
        from .graph import find_callers as _find_callers

        for hit in primary_unique:
            if not hit.symbol:
                continue
            try:
                extras = _find_callers(
                    storage, hit.symbol, depth=1, limit=5,
                )
            except Exception as exc:
                log.warning(
                    "assemble.find_callers_failed",
                    symbol=hit.symbol, error=str(exc),
                )
                continue
            # Normalise to SearchHit if the helper returned dicts.
            normalised = [h for h in extras if isinstance(h, SearchHit)]
            caller_hits.extend(_dedupe_in_order(seen_chunk_ids, normalised))

    if include_tests and storage is not None:
        from .graph import find_tests_for as _find_tests_for

        for hit in primary_unique:
            if not hit.symbol:
                continue
            try:
                extras = _find_tests_for(
                    storage, hit.symbol, limit=5,
                )
            except Exception as exc:
                log.warning(
                    "assemble.find_tests_failed",
                    symbol=hit.symbol, error=str(exc),
                )
                continue
            normalised = [h for h in extras if isinstance(h, SearchHit)]
            test_hits.extend(_dedupe_in_order(seen_chunk_ids, normalised))

    # 3. Single budget pack across the union — primary first (highest
    #    score signal), then extensions. The packer's monotonic stop
    #    means once we overflow we don't sneak smaller hits in from
    #    later sections — score order wins.
    ordered_union: list[SearchHit] = primary_unique + caller_hits + test_hits
    pack = pack_to_budget(ordered_union, context_budget)
    truncated = pack.truncated
    spilled = len(pack.spilled)

    # 4. Re-partition packed hits back into sections (preserving order).
    packed_set = {id(h) for h in pack.packed}
    primary_packed = [h for h in primary_unique if id(h) in packed_set]
    callers_packed = [h for h in caller_hits if id(h) in packed_set]
    tests_packed = [h for h in test_hits if id(h) in packed_set]

    sections = [
        AssembleSection(key="primary", title=_SECTION_ORDER[0][1],
                        hits=primary_packed),
        AssembleSection(key="callers", title=_SECTION_ORDER[1][1],
                        hits=callers_packed),
        AssembleSection(key="tests", title=_SECTION_ORDER[2][1],
                        hits=tests_packed),
    ]
    bundle = _render_bundle(task, sections)
    # The packer's total_tokens is the sum of snippet costs; the rendered
    # bundle adds headers + code fences. We use the packer measurement
    # as the canonical figure because that's what the budget capped.
    log.info(
        "assemble.done",
        task=task[:80],
        primary=len(primary_packed),
        callers=len(callers_packed),
        tests=len(tests_packed),
        budget=context_budget,
        total_tokens=pack.total_tokens,
        spilled=spilled,
        truncated=truncated,
    )
    return AssembleContextResult(
        task=task,
        bundle=bundle,
        sections=sections,
        total_tokens=pack.total_tokens,
        budget_tokens=context_budget,
        spilled_count=spilled,
        truncated_oversize=truncated,
    )


__all__ = [
    "AssembleContextResult",
    "AssembleSection",
    "assemble_context",
]
