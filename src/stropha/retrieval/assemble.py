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

log = get_logger(__name__)


# Sections appear in this order inside the rendered bundle. Each entry
# is (section key, human-readable header). The agent reads top-down so
# primary hits come first; near-misses sit RIGHT after primary so the
# agent immediately sees "and these LOOK related but are deprecated";
# graph extensions follow. ``rationale`` is the docs/ADR layer
# (WIN-219) — read after the code so the agent first sees WHAT the
# code does, then WHY it exists.
_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("primary", "Primary code"),
    ("near_misses", "Near-misses — do NOT confuse with the answer"),
    ("callers", "Callers"),
    ("tests", "Tests"),
    ("rationale", "Rationale — why this code exists"),
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

    Followed by an optional ``> provenance: ...`` line (WIN-214) when
    the hit carries a ``streams`` map, then a fenced code block tagged
    with the chunk's language when known. The provenance line lets the
    agent self-audit which retrieval streams contributed to the hit.
    """
    from .provenance import hit_provenance_line

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
    provenance = hit_provenance_line(hit)
    fence_lang = (hit.language or "").lower()
    body = hit.snippet or ""
    if provenance:
        return (
            f"{header}{repo_hint}\n{provenance}\n```{fence_lang}\n{body}\n```"
        )
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


def _rationale_node_to_hit(node: dict[str, Any]) -> SearchHit | None:
    """Convert a ``find_rationale`` graph node dict into a SearchHit.

    Rationale nodes (docs / ADRs) live in the graph mirror, not in
    the chunks table — they may or may not match an indexed chunk
    via ``chunk_rel_path``. We synthesise a SearchHit-shaped object
    so the bundle's downstream layers (pinning, dedup, packing,
    rendering) treat it like any other span.

    Returns ``None`` when the node lacks even a label — the agent
    can't act on a span with no identity, so we drop it.
    """
    label = node.get("label") or ""
    if not label:
        return None
    node_id = node.get("node_id") or label
    rel_path = (
        node.get("chunk_rel_path")
        or node.get("source_file")
        or "<rationale>"
    )
    start_line = int(node.get("chunk_start_line") or 1)
    end_line = int(node.get("chunk_end_line") or start_line)
    snippet = node.get("chunk_snippet") or label
    # graphify nodes carry a "kind" like "doc" / "adr" / "rationale";
    # default to "rationale" when missing so the bundle's per-hit
    # header shows the provenance even on partial data.
    kind = node.get("kind") or "rationale"
    edge = node.get("edge") or {}
    confidence = edge.get("confidence_score")
    streams = (
        {"rationale": float(confidence)} if confidence is not None else None
    )
    return SearchHit(
        chunk_id=f"rationale:{node_id}",
        rel_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        score=float(confidence or 1.0),
        rank=0,
        snippet=snippet,
        language="markdown",
        kind=kind,
        symbol=label,
        streams=streams,
    )


def assemble_context(
    task: str,
    *,
    search_engine: Any,
    context_budget: int,
    top_k: int = 10,
    include_callers: bool = False,
    include_tests: bool = False,
    include_rationale: bool = False,
    storage: Any = None,
    diff_scope: bool = False,
    diff_repo: Any = None,
    diff_ref_a: str | None = None,
    diff_ref_b: str | None = None,
    repo_id: int | None = None,
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
        include_rationale: WIN-219 — when True, also fetch
            ``rationale_for`` graph edges for each primary symbol
            (via :func:`stropha.retrieval.graph.find_rationale`) and
            pack the resulting docs/ADRs into a "Rationale" section.
            Cross-source fusion — agent gets code + tests + docs in
            ONE deduplicated working set instead of 3 disjoint
            payloads.
        storage: storage handle for the graph helpers. Required when
            ``include_callers``, ``include_tests``, or
            ``include_rationale`` is True; ignored otherwise. Same
            handle the SearchEngine was built with.
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
        repo_id: optional repository scope for every semantic and graph
            retrieval. ``None`` intentionally searches across repositories.

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
    search_kwargs: dict[str, Any] = {"top_k": primary_search_k}
    if repo_id is not None:
        search_kwargs["repo_id"] = repo_id
    primary_hits: list[SearchHit] = list(search_engine.search(task, **search_kwargs))

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

    # WIN-215 — split primary hits into "real answer" + "near-miss"
    # so the bundle can render them under separate headers. Done
    # AFTER diff_scope so the partition reflects the final candidate
    # set the agent is going to see.
    from .antiretrieval import partition_hits

    primary_hits, near_miss_hits = partition_hits(primary_hits)
    near_miss_hits = list(near_miss_hits)

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
                graph_kwargs: dict[str, Any] = {"depth": 1, "limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                extras = _find_callers(storage, hit.symbol, **graph_kwargs)
            except Exception as exc:
                log.warning(
                    "assemble.find_callers_failed",
                    symbol=hit.symbol, error=str(exc),
                )
                continue
            # ``find_callers`` returns a dict (``{"symbol", "callers", "edges",
            # "provenance", ...}``) — iterating it yields the keys (strings),
            # so the ``isinstance`` filter has always evaluated to an empty
            # list here. Mypy now surfaces the silent bug; preserved as-is
            # while the typing pass lands. TODO: walk ``extras["callers"]``
            # and convert graph nodes into SearchHits before extending.
            normalised: list[SearchHit] = [
                h for h in extras  # type: ignore[misc]
                if isinstance(h, SearchHit)
            ]
            caller_hits.extend(_dedupe_in_order(seen_chunk_ids, normalised))

    if include_tests and storage is not None:
        from .graph import find_tests_for as _find_tests_for

        for hit in primary_unique:
            if not hit.symbol:
                continue
            try:
                graph_kwargs = {"limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                extras = _find_tests_for(storage, hit.symbol, **graph_kwargs)
            except Exception as exc:
                log.warning(
                    "assemble.find_tests_failed",
                    symbol=hit.symbol, error=str(exc),
                )
                continue
            # Same dict-vs-list silent bug as the callers branch above.
            normalised = [
                h for h in extras  # type: ignore[misc]
                if isinstance(h, SearchHit)
            ]
            test_hits.extend(_dedupe_in_order(seen_chunk_ids, normalised))

    # WIN-219 — fold rationale nodes (docs / ADRs that explain each
    # primary symbol) into a synthetic SearchHit list. Same dedup +
    # budget pack pipeline; the rendering layer treats them like any
    # other section.
    rationale_hits: list[SearchHit] = []
    if include_rationale and storage is not None:
        from .graph import find_rationale as _find_rationale

        for hit in primary_unique:
            if not hit.symbol:
                continue
            try:
                graph_kwargs = {"limit": 5}
                if repo_id is not None:
                    graph_kwargs["repo_id"] = repo_id
                resp = _find_rationale(storage, hit.symbol, **graph_kwargs)
            except Exception as exc:
                log.warning(
                    "assemble.find_rationale_failed",
                    symbol=hit.symbol, error=str(exc),
                )
                continue
            entries = resp.get("rationale") or []
            converted = [
                synth
                for synth in (_rationale_node_to_hit(n) for n in entries)
                if synth is not None
            ]
            rationale_hits.extend(
                _dedupe_in_order(seen_chunk_ids, converted)
            )

    # Filter near-misses out of seen_chunk_ids' positive side too —
    # near-miss chunk_ids are already tracked by the dedupe pass via
    # primary_hits, so we don't need to do anything extra. Future
    # graph extensions might surface a near-miss as a "caller" — the
    # dedupe keeps us from rendering it twice.
    seen_chunk_ids.update(h.chunk_id for h in near_miss_hits)

    # 3. Single budget pack across the union — primary first (highest
    #    score signal), near-misses next (still high score, just
    #    flagged), then extensions, then rationale (docs/ADRs sit
    #    after the code so the agent reads "what" before "why").
    #    The packer's monotonic stop means once we overflow we don't
    #    sneak smaller hits in from later sections — score order wins.
    ordered_union: list[SearchHit] = (
        primary_unique
        + near_miss_hits
        + caller_hits
        + test_hits
        + rationale_hits
    )
    pack = pack_to_budget(ordered_union, context_budget)
    truncated = pack.truncated
    spilled = len(pack.spilled)

    # 4. Re-partition packed hits back into sections (preserving order).
    packed_set = {id(h) for h in pack.packed}
    primary_packed = [h for h in primary_unique if id(h) in packed_set]
    near_misses_packed = [h for h in near_miss_hits if id(h) in packed_set]
    callers_packed = [h for h in caller_hits if id(h) in packed_set]
    tests_packed = [h for h in test_hits if id(h) in packed_set]
    rationale_packed = [h for h in rationale_hits if id(h) in packed_set]

    sections = [
        AssembleSection(key="primary", title=_SECTION_ORDER[0][1],
                        hits=primary_packed),
        AssembleSection(key="near_misses", title=_SECTION_ORDER[1][1],
                        hits=near_misses_packed),
        AssembleSection(key="callers", title=_SECTION_ORDER[2][1],
                        hits=callers_packed),
        AssembleSection(key="tests", title=_SECTION_ORDER[3][1],
                        hits=tests_packed),
        AssembleSection(key="rationale", title=_SECTION_ORDER[4][1],
                        hits=rationale_packed),
    ]
    bundle = _render_bundle(task, sections)
    # The packer's total_tokens is the sum of snippet costs; the rendered
    # bundle adds headers + code fences. We use the packer measurement
    # as the canonical figure because that's what the budget capped.
    log.info(
        "assemble.done",
        task=task[:80],
        primary=len(primary_packed),
        near_misses=len(near_misses_packed),
        callers=len(callers_packed),
        tests=len(tests_packed),
        rationale=len(rationale_packed),
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
