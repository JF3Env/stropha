"""Bad-result feedback channel for the Meta-Harness labeler (WIN-163).

The ``report_bad_result`` MCP tool lets an agent give the loop an explicit
negative signal — far stronger than the heuristic ``T5 re-query`` tier.
We persist these reports as JSONL (``bad_results.jsonl`` in the meta dir)
so:

- The labeler (WIN-162) can read them back and assemble the
  ``LabelingContext.bad_results`` set.
- An operator can ``grep`` recent bad results without spinning up Python.

Shape of one record::

    {
      "trace_id":    "abc123…",        # the trace the agent is complaining about
      "reason":      "irrelevant",     # one of the enum values
      "detail":      "...",            # optional free-text
      "session_id":  "...",            # which session reported it
      "ts":          "2026-05-22T…Z",  # ISO 8601 UTC
    }

Resolving from ``trace_id`` to the actual chunk_ids the labeler wants
happens via :func:`resolve_bad_chunks`, which scans the trace JSONLs
once. This split (durable record in one file, resolved set computed
on demand) keeps the writer cheap and the reader explicit.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


# Closed enum — the labeler relies on these tags being stable. Adding a
# new value is a forward-compatible change; renaming or removing one
# breaks downstream consumers.
BadResultReason = Literal[
    "irrelevant",
    "wrong_symbol",
    "stale",
    "duplicate",
    "too_verbose",
    "other",
]
LearningChannel = Literal["relevance", "freshness", "diversity", "packing", "review"]

_REASON_CHANNELS: dict[str, LearningChannel] = {
    "irrelevant": "relevance",
    "wrong_symbol": "relevance",
    "stale": "freshness",
    "duplicate": "diversity",
    "too_verbose": "packing",
    "other": "review",
}


def learning_channel(reason: BadResultReason) -> LearningChannel:
    """Return the only optimization channel a feedback reason may affect."""
    try:
        return _REASON_CHANNELS[reason]
    except KeyError as exc:
        raise FeedbackError(
            f"unknown reason {reason!r}; expected one of {sorted(_VALID_REASONS)}"
        ) from exc


class FeedbackError(StrophaError):
    """Raised when bad-result IO cannot satisfy a request."""


@dataclass
class BadResult:
    """One report from the agent."""

    trace_id: str
    reason: BadResultReason
    detail: str | None
    session_id: str | None
    ts: str

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "reason": self.reason,
            "detail": self.detail,
            "session_id": self.session_id,
            "ts": self.ts,
        }


def _now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class BadResultStore:
    """Append-only persistence for bad-result reports.

    The file lives at ``<base_dir>/bad_results.jsonl``. Appends are guarded
    by ``fcntl.flock`` exclusive locks so concurrent MCP requests cannot
    tear the file. Reads use a shared lock and tolerate malformed lines
    (skip + warn).

    Args:
        base_dir: directory that holds ``bad_results.jsonl``. Created if
            missing. Typically ``.stropha/meta/``.
    """

    FILENAME = "bad_results.jsonl"

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / self.FILENAME

    # ------------------------------------------------------------------ #
    # Writes                                                             #
    # ------------------------------------------------------------------ #

    def report(
        self,
        *,
        trace_id: str,
        reason: BadResultReason,
        detail: str | None = None,
        session_id: str | None = None,
    ) -> BadResult:
        """Append one bad-result record and return what was written.

        Validates ``reason`` against the closed enum so a typo in the
        MCP call surfaces as a clean error rather than a silent
        record-with-bad-tag.
        """
        if reason not in _VALID_REASONS:
            raise FeedbackError(
                f"unknown reason {reason!r}; expected one of {sorted(_VALID_REASONS)}"
            )
        record = BadResult(
            trace_id=str(trace_id),
            reason=reason,
            detail=detail,
            session_id=session_id,
            ts=_now_utc_iso(),
        )
        line = json.dumps(record.to_dict()) + "\n"
        with open(self.path, "a+", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        log.info(
            "feedback.report",
            trace_id=trace_id,
            reason=reason,
            session_id=session_id,
        )
        return record

    # ------------------------------------------------------------------ #
    # Reads                                                              #
    # ------------------------------------------------------------------ #

    def list_all(self) -> list[BadResult]:
        """Return every report ever written, oldest first."""
        if not self.path.is_file():
            return []
        records: list[BadResult] = []
        with open(self.path, encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning(
                            "feedback.bad_line",
                            line=raw[:120],
                        )
                        continue
                    reason = obj.get("reason", "other")
                    if reason not in _VALID_REASONS:
                        log.warning("feedback.unknown_reason", reason=reason)
                        continue
                    records.append(
                        BadResult(
                            trace_id=str(obj.get("trace_id", "")),
                            reason=reason,
                            detail=obj.get("detail"),
                            session_id=obj.get("session_id"),
                            ts=str(obj.get("ts", "")),
                        )
                    )
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return records

    def trace_ids(self) -> set[str]:
        """Convenience: every trace_id that has been reported as bad."""
        return {r.trace_id for r in self.list_all()}


def resolve_bad_chunks(
    bad_trace_ids: Iterable[str],
    trace_files: Iterable[Path],
) -> set[str]:
    """Walk ``trace_files`` and collect chunk_ids belonging to bad traces.

    The trace recorder (WIN-159) writes JSONL one record per line; each
    record's ``results`` list carries the chunk ids the tool returned. We
    only need to look up the traces whose ids appear in
    ``bad_trace_ids`` — everything else is skipped.

    Malformed lines are tolerated (skip + warn). Missing files are
    tolerated (skip silently). This function is a labeler-side helper —
    never the hot path; correctness over speed.
    """
    targets = set(bad_trace_ids)
    if not targets:
        return set()
    chunks: set[str] = set()
    for path in trace_files:
        if not path.is_file():
            continue
        with contextlib.suppress(OSError), open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning(
                        "feedback.resolve.bad_line",
                        path=str(path),
                        line=raw[:120],
                    )
                    continue
                if obj.get("trace_id") not in targets:
                    continue
                for result in obj.get("results") or []:
                    cid = result.get("chunk_id")
                    if cid:
                        chunks.add(str(cid))
    return chunks


_VALID_REASONS: frozenset[str] = frozenset(
    {
        "irrelevant",
        "wrong_symbol",
        "stale",
        "duplicate",
        "too_verbose",
        "other",
    }
)
