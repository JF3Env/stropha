"""Token-budgeted proposer history bundles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..fitness import DIMENSION_SIGNS, ScoreCard, scalar_fitness
from ..store import CandidateRecord


def estimate_tokens(value: Any) -> int:
    """Conservative, tokenizer-independent estimate for compact JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return max(1, math.ceil(len(encoded.encode("utf-8")) / 4))


@dataclass(frozen=True)
class HistorySpill:
    candidate_id: str
    estimated_tokens: int
    reason: str = "token_budget"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "estimated_tokens": self.estimated_tokens,
            "reason": self.reason,
        }


@dataclass
class HistoryBundle:
    """A complete ledger plus whole, prioritized candidate records."""

    budget_tokens: int
    all_ids: list[str]
    ledger: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(default_factory=list)
    included_ids: list[str] = field(default_factory=list)
    spilled: list[HistorySpill] = field(default_factory=list)
    estimated_tokens: int = 0
    ledger_tokens: int = 0

    @property
    def spilled_ids(self) -> list[str]:
        return [entry.candidate_id for entry in self.spilled]

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "estimated_tokens": self.estimated_tokens,
            "ledger_tokens": self.ledger_tokens,
            "all_ids": self.all_ids,
            "ledger": self.ledger,
            "records": self.records,
            "included_ids": self.included_ids,
            "spilled_ids": self.spilled_ids,
            "spilled": [entry.to_dict() for entry in self.spilled],
        }


def pack_history(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    *,
    budget_tokens: int,
    recent_count: int = 5,
    fixed_tokens: int = 0,
) -> HistoryBundle:
    """Pack full history without truncating any candidate record.

    The compact ledger is mandatory and names every candidate. Full records are
    selected in baseline/frontier/recent/failure/objective-diversity order. A
    record that does not fit is spilled whole and reported explicitly.
    """
    if budget_tokens <= 0 or fixed_tokens < 0:
        raise ValueError("history token budget must be positive")

    all_ids = [record.id for record, _ in archive]
    ledger = [_ledger_entry(record, card) for record, card in archive]
    mandatory = {"all_ids": all_ids, "ledger": ledger}
    ledger_tokens = estimate_tokens(mandatory)
    if ledger_tokens + fixed_tokens > budget_tokens:
        raise ValueError(
            f"history ledger and prompt require {ledger_tokens + fixed_tokens} tokens, "
            f"budget is {budget_tokens}"
        )

    bundle = HistoryBundle(
        budget_tokens=budget_tokens,
        all_ids=all_ids,
        ledger=ledger,
        ledger_tokens=ledger_tokens,
    )
    prioritized: list[tuple[dict[str, Any], HistorySpill]] = []
    for index, reasons in _priority_order(archive, recent_count=recent_count):
        record, card = archive[index]
        full = _full_record(record, card, reasons=reasons)
        cost = estimate_tokens(full)
        prioritized.append((full, HistorySpill(record.id, cost)))
    bundle.spilled = [spill for _, spill in prioritized]
    bundle.estimated_tokens = _bundle_tokens(bundle, fixed_tokens=fixed_tokens)
    if bundle.estimated_tokens > budget_tokens:
        raise ValueError(
            "history ledger and spill metadata require "
            f"{bundle.estimated_tokens} tokens, budget is {budget_tokens}"
        )

    for full, spill in prioritized:
        bundle.records.append(full)
        bundle.included_ids.append(spill.candidate_id)
        bundle.spilled.remove(spill)
        trial_tokens = _bundle_tokens(bundle, fixed_tokens=fixed_tokens)
        if trial_tokens <= budget_tokens:
            bundle.estimated_tokens = trial_tokens
        else:
            bundle.records.pop()
            bundle.included_ids.pop()
            bundle.spilled.append(spill)
    bundle.estimated_tokens = _bundle_tokens(bundle, fixed_tokens=fixed_tokens)
    return bundle


def _bundle_tokens(bundle: HistoryBundle, *, fixed_tokens: int) -> int:
    previous = -1
    bundle.estimated_tokens = 0
    for _ in range(4):
        total = fixed_tokens + estimate_tokens(bundle.to_dict())
        if total == previous:
            break
        previous = total
        bundle.estimated_tokens = total
    return bundle.estimated_tokens


def _ledger_entry(record: CandidateRecord, card: ScoreCard) -> dict[str, Any]:
    return {
        "id": record.id,
        "parent_id": record.parent_id,
        "label": record.label,
        "scores": _score_view(card),
    }


def _score_view(card: ScoreCard) -> dict[str, Any]:
    return {
        "quality": round(card.quality, 4),
        "context_efficiency": round(card.context_efficiency, 4),
        "outcome_score": round(card.outcome_score, 4),
        "latency_ms": round(card.latency_ms, 1),
        "tier_counts": dict(card.tier_counts),
    }


def _priority_order(
    archive: list[tuple[CandidateRecord, ScoreCard]],
    *,
    recent_count: int,
) -> list[tuple[int, list[str]]]:
    reasons: dict[int, list[str]] = {}

    def add(index: int, reason: str) -> None:
        reasons.setdefault(index, []).append(reason)

    for index, (record, _) in enumerate(archive):
        role = str(record.meta.get("role", "")).lower()
        if role == "baseline" or (record.label or "").lower().startswith("baseline"):
            add(index, "baseline")

    for index, (_, card) in enumerate(archive):
        if not any(other.dominates(card) for j, (_, other) in enumerate(archive) if j != index):
            add(index, "pareto_frontier")

    start = max(0, len(archive) - recent_count)
    for index in range(len(archive) - 1, start - 1, -1):
        add(index, "recent")

    if archive:
        failure = min(
            range(len(archive)),
            key=lambda i: (scalar_fitness(archive[i][1].to_dict()), i),
        )
        add(failure, "representative_failure")
        for objective, sign in DIMENSION_SIGNS.items():
            diverse = max(
                range(len(archive)),
                key=lambda i: (getattr(archive[i][1], objective) * sign, -i),
            )
            add(diverse, f"objective_extreme:{objective}")

    for index in range(len(archive) - 1, -1, -1):
        add(index, "remaining_history")

    return list(reasons.items())


def _full_record(
    record: CandidateRecord,
    card: ScoreCard,
    *,
    reasons: list[str],
) -> dict[str, Any]:
    hypothesis = record.meta.get("hypothesis")
    if hypothesis is None and isinstance(record.meta.get("proposal"), dict):
        hypothesis = record.meta["proposal"].get("hypothesis")
    return {
        "id": record.id,
        "parent_id": record.parent_id,
        "created_at": record.created_at,
        "label": record.label,
        "priority_reasons": reasons,
        "config": record.config,
        "scores": card.to_dict(),
        "meta": record.meta,
        "hypothesis": hypothesis,
        "artifacts": _read_artifacts(record.base_dir),
    }


def _read_artifacts(base_dir: Path | None) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "code": {},
        "diagnostics": {},
        "traces": [],
    }
    if base_dir is None or not base_dir.is_dir():
        return artifacts

    adapters_dir = base_dir / "adapters"
    if adapters_dir.is_dir():
        for path in sorted(adapters_dir.rglob("*.py")):
            artifacts["code"][str(path.relative_to(base_dir))] = _read_text(path)

    for name in ("gate.json", "sandbox.json", "hypothesis-verdict.json"):
        path = base_dir / name
        if path.is_file():
            artifacts["diagnostics"][name.removesuffix(".json")] = _read_json(path)

    traces_dir = base_dir / "traces"
    if traces_dir.is_dir():
        for path in sorted(item for item in traces_dir.rglob("*") if item.is_file()):
            artifacts["traces"].append({
                "path": str(path.relative_to(base_dir)),
                "content": _read_json_lines(path),
            })
    return artifacts


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"malformed_json": text}


def _read_json_lines(path: Path) -> list[Any]:
    text = _read_text(path)
    out: list[Any] = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"malformed_json": line})
    return out


__all__ = [
    "HistoryBundle",
    "HistorySpill",
    "estimate_tokens",
    "pack_history",
]
