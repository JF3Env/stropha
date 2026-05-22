"""ArtifactStore — filesystem layout for Meta-Harness candidates (WIN-161).

The Meta-Harness proposer needs durable, grep-able access to every prior
candidate — its config, its scores, and the traces it produced when it
ran. We chose a filesystem layout (not SQLite) so an agentic proposer
can read past candidates with the same primitives it uses to read source
code (``ls``, ``cat``, ``grep``).

::

    .stropha/meta/
      candidates/<id>/
        config.yaml      # the candidate's full stropha.yaml
        meta.json        # provenance: parent_id, created_at, label, …
        scores.json      # evaluator output; empty {} until evaluation
        traces/*.jsonl   # WIN-159 trace files for this candidate's runs
      index.jsonl        # append-only ledger; one line per registered candidate
      runs/<run_id>/     # evaluator run scratch space (reserved for Phase 1)

Design notes:

- IDs are short (8-char) uuid hex by default. Callers may pass an explicit
  ``id`` for reproducibility (tests, replay scenarios). Collision raises.
- ``register_candidate`` is the only path that creates directories and
  writes ``index.jsonl``. It holds an exclusive ``fcntl.flock`` on the
  ledger during the append so concurrent proposers can't tear the file.
- ``read_candidate`` is read-only; safe to call from any thread.
- ``pareto_frontier`` is sign-aware: caller passes a dict of
  ``{score_name: +1 | -1}`` so the helper knows whether to maximise or
  minimise each dimension. v1 ignores any dimension the caller doesn't
  list (defensive default).
- ``compact`` deletes pruned candidates' directories but keeps a stub
  line in ``index.jsonl`` (``pruned: true``) so the proposer can still
  see "candidate X existed and was pruned" — informative absence beats
  silent disappearance.

WIN-159's ``TraceRecorder`` writes JSONL into a path the caller chooses.
WIN-160 currently defaults that path to ``<index_parent>/meta/traces/``;
when the Meta-Harness loop ships (Phase 4) it will instead route to
``.stropha/meta/candidates/<id>/traces/`` per the layout above.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


class ArtifactStoreError(StrophaError):
    """Raised when the artifact store cannot satisfy a request."""


def _now_utc_iso() -> str:
    """Match the timestamp shape used by ``meta.trace``."""
    return datetime.now(tz=UTC).isoformat()


def _short_id() -> str:
    """8-char hex id — short enough to type, wide enough for ~4 billion candidates."""
    return uuid.uuid4().hex[:8]


class CandidateRecord(BaseModel):
    """One candidate's complete on-disk shape.

    ``base_dir`` is excluded from JSON round-trips because it depends on
    where the store lives; the rest of the fields are durable.
    """

    id: str
    parent_id: str | None = None
    created_at: str
    label: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    """Score map. Values are usually float, but WIN-169's ScoreCard
    also surfaces ``tier_counts`` (a dict) for provenance. Pydantic
    accepts any JSON-serialisable value here; fitness math filters
    to numeric values."""
    trace_count: int = 0
    pruned: bool = False
    base_dir: Path | None = Field(default=None, exclude=True)

    def fitness(self, weights: dict[str, float] | None = None) -> float:
        """Composite fitness — weighted sum of scores (defaults to equal weights).

        Returns ``-inf`` when no scores are populated so unscored candidates
        always sort behind scored ones.
        """
        if not self.scores:
            return float("-inf")
        numeric = {
            k: float(v) for k, v in self.scores.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            return float("-inf")
        if weights is None:
            return sum(numeric.values()) / len(numeric)
        total = 0.0
        for name, value in numeric.items():
            total += value * weights.get(name, 1.0)
        return total


class ArtifactStore:
    """The ``.stropha/meta/`` directory, as a typed object.

    Args:
        base_dir: directory that holds ``candidates/`` + ``index.jsonl``.
            Created (and its sub-tree) on first use.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.candidates_dir = self.base_dir / "candidates"
        self.runs_dir = self.base_dir / "runs"
        self.index_path = self.base_dir / "index.jsonl"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # index.jsonl is created lazily on first append so a brand-new
        # store doesn't litter the filesystem.

    # ------------------------------------------------------------------ #
    # Writes                                                             #
    # ------------------------------------------------------------------ #

    def register_candidate(
        self,
        config: dict[str, Any],
        *,
        parent: str | None = None,
        label: str | None = None,
        meta: dict[str, Any] | None = None,
        candidate_id: str | None = None,
    ) -> CandidateRecord:
        """Persist a new candidate and return its record.

        Side effects (in order, all under ``index.jsonl``'s flock):

        1. Reserve an id (generated or caller-supplied).
        2. Create ``candidates/<id>/`` + its ``traces/`` subdir.
        3. Write ``config.yaml``, ``meta.json``, ``scores.json`` (empty).
        4. Append one JSON line to ``index.jsonl``.

        Raises ``ArtifactStoreError`` if the id is already used, the
        config is not JSON-serialisable, or any step fails on the way.
        """
        cid = candidate_id or _short_id()
        cand_dir = self.candidates_dir / cid
        try:
            cand_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactStoreError(
                f"candidate id {cid!r} already exists at {cand_dir}"
            ) from exc

        traces_dir = cand_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        created_at = _now_utc_iso()
        record = CandidateRecord(
            id=cid,
            parent_id=parent,
            created_at=created_at,
            label=label,
            config=dict(config),
            meta=dict(meta or {}),
            scores={},
            trace_count=0,
            pruned=False,
            base_dir=cand_dir,
        )

        # YAML for the config — humans (and the agentic proposer) will read
        # this file. JSON for the structured side-cars.
        try:
            (cand_dir / "config.yaml").write_text(
                yaml.safe_dump(record.config, sort_keys=True), encoding="utf-8"
            )
            (cand_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "id": cid,
                        "parent_id": parent,
                        "created_at": created_at,
                        "label": label,
                        "meta": record.meta,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (cand_dir / "scores.json").write_text("{}\n", encoding="utf-8")
        except (OSError, TypeError) as exc:
            # Roll back the directory so a failed register doesn't
            # leave the store in an inconsistent state.
            shutil.rmtree(cand_dir, ignore_errors=True)
            raise ArtifactStoreError(
                f"failed to write candidate {cid!r}: {exc}"
            ) from exc

        self._append_index({"id": cid, "parent_id": parent, "created_at": created_at})
        log.info("artifactstore.register", id=cid, parent=parent, label=label)
        return record

    def write_scores(self, candidate_id: str, scores: dict[str, float]) -> None:
        """Replace ``scores.json`` for ``candidate_id``.

        Caller is expected to call this exactly once per evaluator run;
        re-runs overwrite. We do NOT merge to avoid stale-score leakage.
        """
        cand_dir = self._require_dir(candidate_id)
        try:
            (cand_dir / "scores.json").write_text(
                json.dumps(scores, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, TypeError) as exc:
            raise ArtifactStoreError(
                f"failed to write scores for {candidate_id!r}: {exc}"
            ) from exc
        log.info("artifactstore.scores", id=candidate_id, fields=sorted(scores))

    # ------------------------------------------------------------------ #
    # Reads                                                              #
    # ------------------------------------------------------------------ #

    def read_candidate(self, candidate_id: str) -> CandidateRecord:
        """Load every side-car for ``candidate_id`` into a single record.

        Raises ``ArtifactStoreError`` when the candidate directory does
        not exist or any required side-car is missing / malformed.
        """
        cand_dir = self._require_dir(candidate_id)
        try:
            meta_raw = json.loads((cand_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError(
                f"meta.json missing or malformed for {candidate_id!r}: {exc}"
            ) from exc

        config_raw: dict[str, Any] = {}
        with contextlib.suppress(OSError):
            text = (cand_dir / "config.yaml").read_text(encoding="utf-8")
            loaded = yaml.safe_load(text) or {}
            if isinstance(loaded, dict):
                config_raw = loaded

        scores_raw: dict[str, float] = {}
        with contextlib.suppress(OSError, json.JSONDecodeError):
            scores_raw = json.loads(
                (cand_dir / "scores.json").read_text(encoding="utf-8")
            ) or {}

        trace_count = 0
        traces_dir = cand_dir / "traces"
        if traces_dir.is_dir():
            trace_count = sum(1 for _ in traces_dir.glob("*.jsonl"))

        # ``pruned`` is sourced from the ledger, not the candidate dir,
        # because compact() may have removed the dir entirely.
        pruned = self._is_pruned(candidate_id)

        return CandidateRecord(
            id=candidate_id,
            parent_id=meta_raw.get("parent_id"),
            created_at=meta_raw.get("created_at", _now_utc_iso()),
            label=meta_raw.get("label"),
            config=config_raw,
            meta=meta_raw.get("meta", {}),
            scores=scores_raw,
            trace_count=trace_count,
            pruned=pruned,
            base_dir=cand_dir,
        )

    def list_candidates(
        self,
        *,
        limit: int | None = None,
        sort_by: str = "created_at",
        include_pruned: bool = False,
    ) -> list[CandidateRecord]:
        """Enumerate candidates from the ledger.

        ``sort_by`` accepts:
        - ``"created_at"`` — newest first.
        - ``"id"`` — lexicographic.
        - ``"fitness"`` — descending mean-of-scores; un-scored last.
        - Any score key (e.g. ``"recall@10"``) — descending; un-scored last.
        """
        records = []
        for entry in self._read_index():
            is_pruned = bool(entry.get("pruned"))
            if is_pruned and not include_pruned:
                continue
            cid = entry["id"]
            if is_pruned:
                # The candidate directory was deleted by compact(); synthesise
                # a tombstone record from the ledger entry so the proposer can
                # see "candidate X existed and was pruned" without crashing.
                records.append(
                    CandidateRecord(
                        id=cid,
                        parent_id=entry.get("parent_id"),
                        created_at=entry.get("created_at", _now_utc_iso()),
                        pruned=True,
                    )
                )
                continue
            try:
                records.append(self.read_candidate(cid))
            except ArtifactStoreError as exc:
                # Index points at a missing/malformed candidate that is NOT
                # marked pruned. Either deleted out-of-band, or compact()
                # didn't update the ledger. Skip and log; never crash list().
                log.warning("artifactstore.list.skip", id=cid, error=str(exc))

        if sort_by == "created_at":
            records.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "id":
            records.sort(key=lambda r: r.id)
        elif sort_by == "fitness":
            records.sort(key=lambda r: r.fitness(), reverse=True)
        else:
            # Treat as a score key. Un-scored candidates sort last via -inf.
            records.sort(
                key=lambda r: r.scores.get(sort_by, float("-inf")),
                reverse=True,
            )

        if limit is not None:
            records = records[:limit]
        return records

    def pareto_frontier(
        self,
        objective_signs: dict[str, int],
        *,
        include_pruned: bool = False,
    ) -> list[CandidateRecord]:
        """Non-dominated candidates across the given multi-objective surface.

        ``objective_signs`` maps a score name to ``+1`` (maximise) or
        ``-1`` (minimise). Dimensions not listed are ignored. Candidates
        missing any listed score are excluded (they can't be ranked).

        A candidate `a` is dominated by `b` when, after sign-flipping,
        `b` is ≥ `a` on every listed dimension and `>` on at least one.
        """
        if not objective_signs:
            return []
        dims = list(objective_signs.keys())
        contenders = [
            r
            for r in self.list_candidates(include_pruned=include_pruned)
            if all(d in r.scores for d in dims)
        ]

        def _vec(r: CandidateRecord) -> list[float]:
            return [r.scores[d] * objective_signs[d] for d in dims]

        frontier: list[CandidateRecord] = []
        for cand in contenders:
            cand_vec = _vec(cand)
            dominated = False
            for other in contenders:
                if other.id == cand.id:
                    continue
                other_vec = _vec(other)
                if all(o >= c for o, c in zip(other_vec, cand_vec, strict=True)) and any(
                    o > c for o, c in zip(other_vec, cand_vec, strict=True)
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(cand)
        return frontier

    # ------------------------------------------------------------------ #
    # Compaction                                                         #
    # ------------------------------------------------------------------ #

    def compact(
        self,
        *,
        keep_top: int = 10,
        keep_bottom: int = 5,
        keep_recent: int = 10,
        fitness_key: str = "fitness",
    ) -> list[str]:
        """Delete candidate directories outside the ``keep_*`` buckets.

        Three independent buckets are merged:

        - ``keep_top``: highest fitness (descending mean-of-scores).
        - ``keep_bottom``: lowest fitness — preserved so the proposer can
          learn what failed (anti-examples are signal).
        - ``keep_recent``: newest by ``created_at``.

        Candidates outside the union have their on-disk directory removed
        but the ledger entry is rewritten with ``pruned: true`` so the
        proposer can still see "candidate X existed and was pruned".

        Returns the list of pruned candidate ids.
        """
        survivors: set[str] = set()
        all_live = self.list_candidates(include_pruned=False)
        if not all_live:
            return []

        # keep_top + keep_bottom by fitness.
        by_fit = sorted(all_live, key=lambda r: r.fitness(), reverse=True)
        for r in by_fit[: max(0, keep_top)]:
            survivors.add(r.id)
        for r in by_fit[-max(0, keep_bottom) :] if keep_bottom > 0 else []:
            survivors.add(r.id)

        # keep_recent by created_at (independent of fitness).
        by_recent = sorted(all_live, key=lambda r: r.created_at, reverse=True)
        for r in by_recent[: max(0, keep_recent)]:
            survivors.add(r.id)

        pruned_ids: list[str] = []
        for r in all_live:
            if r.id in survivors:
                continue
            if r.base_dir is not None and r.base_dir.is_dir():
                shutil.rmtree(r.base_dir, ignore_errors=True)
            pruned_ids.append(r.id)

        if pruned_ids:
            self._mark_pruned(pruned_ids)
            log.info(
                "artifactstore.compact",
                pruned=len(pruned_ids),
                survivors=len(survivors),
                fitness_key=fitness_key,
            )
        return pruned_ids

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _require_dir(self, candidate_id: str) -> Path:
        cand_dir = self.candidates_dir / candidate_id
        if not cand_dir.is_dir():
            raise ArtifactStoreError(
                f"candidate directory missing: {cand_dir} (id={candidate_id!r})"
            )
        return cand_dir

    def _append_index(self, entry: dict[str, Any]) -> None:
        """Atomic append to ``index.jsonl`` under a POSIX file lock."""
        line = json.dumps(entry) + "\n"
        # ``a+`` keeps the fd seekable so flock can take effect.
        with open(self.index_path, "a+", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        with open(self.index_path, encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entries.append(json.loads(raw))
                    except json.JSONDecodeError:
                        # Tolerate the occasional torn line — log + skip.
                        log.warning("artifactstore.index.bad_line", line=raw[:120])
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return entries

    def _is_pruned(self, candidate_id: str) -> bool:
        for entry in self._read_index():
            if entry.get("id") == candidate_id and entry.get("pruned"):
                return True
        return False

    def _mark_pruned(self, ids: list[str]) -> None:
        """Rewrite ``index.jsonl`` flipping ``pruned: true`` on the listed ids.

        Implemented as read-all → mutate → atomic-rename so a crash mid-write
        cannot leave the ledger half-rewritten.
        """
        pruned_set = set(ids)
        entries = self._read_index()
        for entry in entries:
            if entry.get("id") in pruned_set:
                entry["pruned"] = True

        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                for entry in entries:
                    fh.write(json.dumps(entry) + "\n")
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        tmp.replace(self.index_path)
