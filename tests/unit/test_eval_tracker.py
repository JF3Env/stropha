"""Tests for the eval score tracker (WIN-382)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stropha.eval.tracker import (
    append_score,
    check_regression,
    load_scores,
)


@dataclass
class _FakeReport:
    """Minimal fake that quacks like an EvalReport for tracker tests."""

    recall_at_k: float = 0.6
    mrr: float = 0.25
    ndcg_at_k: float = 0.35
    precision_at_k: float = 0.15
    map_at_k: float = 0.28
    latency_p50_ms: float = 100.0
    latency_p95_ms: float = 200.0
    n: int = 30
    n_hits: int = 18
    top_k: int = 10


class TestAppendScore:
    """Tests for append_score."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        report = _FakeReport()

        entry = append_score(report, log_path=log_path, commit="abc123")

        assert log_path.is_file()
        assert entry["commit"] == "abc123"
        assert entry["recall_at_k"] == 0.6

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        report = _FakeReport()

        append_score(report, log_path=log_path, commit="aaa")
        append_score(report, log_path=log_path, commit="bbb")
        append_score(report, log_path=log_path, commit="ccc")

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["commit"] == "aaa"
        assert json.loads(lines[2])["commit"] == "ccc"

    def test_entry_has_all_fields(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        report = _FakeReport(recall_at_k=0.75, mrr=0.4)

        entry = append_score(report, log_path=log_path, commit="def456")

        assert "ts" in entry
        assert entry["recall_at_k"] == 0.75
        assert entry["mrr"] == 0.4
        assert entry["n_cases"] == 30
        assert entry["top_k"] == 10

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        log_path = tmp_path / "deep" / "nested" / "scores.jsonl"
        append_score(_FakeReport(), log_path=log_path, commit="x")
        assert log_path.is_file()


class TestLoadScores:
    """Tests for load_scores."""

    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        scores = load_scores(tmp_path / "nonexistent.jsonl")
        assert scores == []

    def test_loads_valid_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        log_path.write_text(
            '{"commit": "a", "recall_at_k": 0.5}\n'
            '{"commit": "b", "recall_at_k": 0.6}\n'
        )
        scores = load_scores(log_path)
        assert len(scores) == 2
        assert scores[1]["recall_at_k"] == 0.6

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        log_path.write_text(
            '{"commit": "a", "recall_at_k": 0.5}\n'
            'not json\n'
            '{"commit": "c", "recall_at_k": 0.7}\n'
        )
        scores = load_scores(log_path)
        assert len(scores) == 2


class TestCheckRegression:
    """Tests for check_regression."""

    def test_not_enough_history(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        log_path.write_text('{"recall_at_k": 0.5}\n')
        is_reg, msg = check_regression(0.5, log_path=log_path)
        assert is_reg is False
        assert "not enough history" in msg

    def test_no_regression_when_stable(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        # 5 entries with recall=0.6, then append current also 0.6
        entries = [json.dumps({"recall_at_k": 0.6}) for _ in range(5)]
        entries.append(json.dumps({"recall_at_k": 0.6}))  # "current"
        log_path.write_text("\n".join(entries) + "\n")

        is_reg, msg = check_regression(0.6, log_path=log_path)
        assert is_reg is False
        assert "OK" in msg

    def test_detects_regression(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        # 5 entries with recall=0.7, then current drops to 0.5
        entries = [json.dumps({"recall_at_k": 0.7}) for _ in range(5)]
        entries.append(json.dumps({"recall_at_k": 0.5}))  # "current"
        log_path.write_text("\n".join(entries) + "\n")

        is_reg, msg = check_regression(0.5, log_path=log_path)
        assert is_reg is True
        assert "REGRESSION" in msg

    def test_no_regression_within_threshold(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        # 5 entries with recall=0.6, current at 0.57 (within 0.05 threshold)
        entries = [json.dumps({"recall_at_k": 0.6}) for _ in range(5)]
        entries.append(json.dumps({"recall_at_k": 0.57}))
        log_path.write_text("\n".join(entries) + "\n")

        is_reg, _msg = check_regression(0.57, log_path=log_path)
        assert is_reg is False

    def test_improvement_is_not_regression(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        # 5 entries with recall=0.5, current improves to 0.7
        entries = [json.dumps({"recall_at_k": 0.5}) for _ in range(5)]
        entries.append(json.dumps({"recall_at_k": 0.7}))
        log_path.write_text("\n".join(entries) + "\n")

        is_reg, msg = check_regression(0.7, log_path=log_path)
        assert is_reg is False
        assert "OK" in msg

    def test_respects_window_size(self, tmp_path: Path) -> None:
        log_path = tmp_path / "scores.jsonl"
        # Old entries with recall=0.3, recent entries with recall=0.7
        old = [json.dumps({"recall_at_k": 0.3}) for _ in range(20)]
        recent = [json.dumps({"recall_at_k": 0.7}) for _ in range(10)]
        current = [json.dumps({"recall_at_k": 0.65})]
        log_path.write_text("\n".join(old + recent + current) + "\n")

        # With window=10, rolling mean should be ~0.7, so 0.65 is within threshold
        is_reg, _msg = check_regression(0.65, log_path=log_path, window=10)
        assert is_reg is False
