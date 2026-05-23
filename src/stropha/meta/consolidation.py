"""Memory consolidation — episodic → semantic compaction (WIN-226).

EPIC 1.9 ships two memory layers:

- **Episodic** (Schema v10, WIN-225): every task the agent worked
  on, append-only. Grows linearly with use.
- **Semantic** (Schema v9, WIN-222): stable facts, kind/key keyed,
  UPSERTed in place. Bounded.

Left alone, episodic memory grows unbounded. Consolidation is the
periodic background pass that mines the recent episodic history,
finds high-frequency co-occurrence patterns, writes them as
project_memory entries, and prunes the underlying episodes. The
net result: a small set of high-quality facts replaces a long tail
of low-signal episodes.

The pass is intentionally simple and deterministic:

1. For each repo with at least ``min_episodes_to_consolidate`` episodes,
   enumerate every distinct file/symbol touched.
2. For each candidate key, run :func:`Storage.episode_co_occurrence`
   to count co-touched neighbours.
3. Filter to keys with at least ``min_count`` co-touches in the
   top neighbour. (Otherwise we'd write thousands of useless
   single-touch facts.)
4. For each surviving key, write/upsert a ``project_memory`` entry
   under ``kind="co-touched"``.
5. If ``prune_after_days`` is set, delete episodes older than that
   threshold for the same repo so the store doesn't keep growing.

The CLI exposes this as ``stropha memory consolidate``. The
background hook can call into the same code path via a future
ticket; this module is intentionally framework-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


# Conservative defaults — fire only on real patterns, not noise.
DEFAULT_MIN_EPISODES = 3
DEFAULT_MIN_COUNT = 2
DEFAULT_TOP_N_PER_KEY = 5
DEFAULT_PRUNE_AFTER_DAYS = 90
# kind bucket the consolidated facts go into. Matches the WIN-222
# recall(kind=...) filter so the agent can `recall("co-touched")` to
# pull just the consolidated entries.
CONSOLIDATED_KIND = "co-touched"


@dataclass(frozen=True)
class ConsolidationCandidate:
    """One key whose co-touch pattern will be written to project_memory."""

    key: str  # the file path or symbol
    top_files: dict[str, int] = field(default_factory=dict)
    top_symbols: dict[str, int] = field(default_factory=dict)

    def render_value(self) -> str:
        """Render the consolidated fact as a human-readable string.

        Format::

            Co-touched with files: a.py (3x), b.py (2x). Symbols: Foo.bar (4x).

        Skips empty sections. The agent reads the project_memory
        value verbatim, so the rendering is deliberately compact.
        """
        parts: list[str] = []
        if self.top_files:
            file_pairs = ", ".join(
                f"{f} ({n}x)" for f, n in self.top_files.items()
            )
            parts.append(f"files: {file_pairs}")
        if self.top_symbols:
            sym_pairs = ", ".join(
                f"{s} ({n}x)" for s, n in self.top_symbols.items()
            )
            parts.append(f"symbols: {sym_pairs}")
        if not parts:
            return f"No co-touches recorded for {self.key}."
        return "Co-touched with " + "; ".join(parts) + "."


@dataclass(frozen=True)
class ConsolidationPlan:
    """A pre-computed consolidation pass. ``apply`` writes it."""

    repo_id: int
    candidates: list[ConsolidationCandidate] = field(default_factory=list)
    episodes_to_prune: int = 0
    prune_cutoff_iso: str | None = None


def _episode_keys(storage: Any, repo_id: int, limit: int) -> set[str]:
    """Collect every distinct file/symbol touched across the repo's
    episodes.

    Reads up to ``limit`` recent episodes. Uses the JSON storage
    directly (no extra DB queries beyond ``recall_episodes``).
    """
    episodes = storage.recall_episodes(repo_id=repo_id, limit=limit)
    keys: set[str] = set()
    for ep in episodes:
        for f in ep.touched_files:
            if f:
                keys.add(f)
        for s in ep.touched_symbols:
            if s:
                keys.add(s)
    return keys


def plan_consolidation(
    storage: Any,
    *,
    repo_id: int,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    min_count: int = DEFAULT_MIN_COUNT,
    top_n: int = DEFAULT_TOP_N_PER_KEY,
    episode_scan_limit: int = 200,
    prune_after_days: int | None = DEFAULT_PRUNE_AFTER_DAYS,
) -> ConsolidationPlan:
    """Compute a :class:`ConsolidationPlan` without mutating storage.

    Args:
        storage: the SQLite Storage instance (must expose
            ``recall_episodes`` + ``episode_co_occurrence``).
        repo_id: target repo's id.
        min_episodes: lower bound on the episode count for the
            pass to do anything. Avoids consolidating noise on a
            cold repo.
        min_count: minimum co-occurrence count for a neighbour to
            be included in a candidate's top list. Single co-touches
            are usually noise.
        top_n: cap on the neighbours kept per candidate. Five is
            plenty for a human-readable fact.
        episode_scan_limit: cap on the episodes scanned to enumerate
            candidate keys. Bounded so the pass is O(scan_limit)
            even on a long-running repo.
        prune_after_days: episodes older than this many days are
            queued for deletion. Pass ``None`` to skip pruning.

    Returns:
        A :class:`ConsolidationPlan` describing exactly what
        ``apply_consolidation`` will do. Empty candidate list +
        zero prune count means the pass is a no-op.
    """
    # Step 1 — count episodes; bail when below the threshold.
    episodes = storage.recall_episodes(
        repo_id=repo_id, limit=episode_scan_limit,
    )
    if len(episodes) < min_episodes:
        log.info(
            "consolidation.skip.few_episodes",
            repo_id=repo_id, episodes=len(episodes),
            min_required=min_episodes,
        )
        return ConsolidationPlan(repo_id=repo_id)

    # Step 2 — collect candidate keys.
    candidate_keys = _episode_keys(
        storage, repo_id=repo_id, limit=episode_scan_limit,
    )

    # Step 3 — per-key co-occurrence + min_count filter.
    candidates: list[ConsolidationCandidate] = []
    for key in sorted(candidate_keys):
        result = storage.episode_co_occurrence(
            repo_id=repo_id, key=key, limit=max(1, top_n),
        )
        files = {
            f: n for f, n in result.get("files", {}).items()
            if n >= min_count
        }
        symbols = {
            s: n for s, n in result.get("symbols", {}).items()
            if n >= min_count
        }
        if not files and not symbols:
            continue
        candidates.append(
            ConsolidationCandidate(
                key=key,
                top_files=files,
                top_symbols=symbols,
            )
        )

    # Step 4 — prune-cutoff computation.
    prune_count = 0
    cutoff_iso: str | None = None
    if prune_after_days is not None and prune_after_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=prune_after_days)
        cutoff_iso = cutoff.isoformat()
        # Count without deleting yet (the plan is read-only).
        try:
            row = storage._conn.execute(
                """SELECT COUNT(*) AS n FROM episodes
                   WHERE repo_id = ? AND created_at < ?""",
                (repo_id, cutoff_iso),
            ).fetchone()
            prune_count = int(row["n"] or 0) if row else 0
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "consolidation.prune_count_failed",
                repo_id=repo_id, error=str(exc),
            )

    log.info(
        "consolidation.plan",
        repo_id=repo_id,
        candidates=len(candidates),
        episodes_scanned=len(episodes),
        prune_count=prune_count,
        prune_cutoff=cutoff_iso,
    )
    return ConsolidationPlan(
        repo_id=repo_id,
        candidates=candidates,
        episodes_to_prune=prune_count,
        prune_cutoff_iso=cutoff_iso,
    )


def apply_consolidation(
    storage: Any,
    plan: ConsolidationPlan,
    *,
    source: str | None = None,
) -> dict[str, int]:
    """Execute the plan against ``storage``. Returns counters.

    Args:
        storage: same Storage instance used to build the plan.
        plan: an immutable :class:`ConsolidationPlan` produced by
            :func:`plan_consolidation`.
        source: optional session_id / agent identifier to record as
            the ``source`` on the written project_memory entries.

    Returns ``{"written": int, "pruned": int}``.
    """
    written = 0
    for candidate in plan.candidates:
        storage.remember(
            repo_id=plan.repo_id,
            kind=CONSOLIDATED_KIND,
            key=candidate.key,
            value=candidate.render_value(),
            source=source,
        )
        written += 1
    pruned = 0
    if plan.prune_cutoff_iso is not None:
        try:
            cur = storage._conn.execute(
                """DELETE FROM episodes
                   WHERE repo_id = ? AND created_at < ?""",
                (plan.repo_id, plan.prune_cutoff_iso),
            )
            storage._conn.commit()
            pruned = int(cur.rowcount or 0)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "consolidation.prune_failed",
                repo_id=plan.repo_id, error=str(exc),
            )
    log.info(
        "consolidation.applied",
        repo_id=plan.repo_id,
        written=written,
        pruned=pruned,
    )
    return {"written": written, "pruned": pruned}


__all__ = [
    "CONSOLIDATED_KIND",
    "ConsolidationCandidate",
    "ConsolidationPlan",
    "apply_consolidation",
    "plan_consolidation",
    "DEFAULT_MIN_COUNT",
    "DEFAULT_MIN_EPISODES",
    "DEFAULT_PRUNE_AFTER_DAYS",
    "DEFAULT_TOP_N_PER_KEY",
]
