"""Unit tests for the WIN-163 bad-result feedback channel.

Covers:
- ``BadResultStore`` writes / reads / concurrency / malformed lines.
- ``resolve_bad_chunks`` traces → chunk_ids resolution.
- The ``report_bad_result`` MCP tool wired in server.py.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.meta import (
    BadResultStore,
    FeedbackError,
    learning_channel,
    resolve_bad_chunks,
)

# ---------------------------------------------------------------------------
# BadResultStore basics
# ---------------------------------------------------------------------------


class TestBadResultStore:
    def test_report_creates_file_and_returns_record(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        rec = store.report(
            trace_id="t-abc",
            reason="irrelevant",
            detail="wrong file",
            session_id="s-1",
        )
        assert rec.trace_id == "t-abc"
        assert rec.reason == "irrelevant"
        assert rec.detail == "wrong file"
        assert rec.session_id == "s-1"
        assert rec.ts  # ISO timestamp populated
        # The file exists with exactly one JSON line.
        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == 1
        on_disk = json.loads(lines[0])
        assert on_disk["trace_id"] == "t-abc"

    def test_multiple_reports_append(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        for i in range(5):
            store.report(trace_id=f"t-{i}", reason="other")
        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == 5
        ids = [json.loads(line)["trace_id"] for line in lines]
        assert ids == [f"t-{i}" for i in range(5)]

    def test_unknown_reason_raises(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        with pytest.raises(FeedbackError, match="unknown reason"):
            store.report(trace_id="t-x", reason="typo")  # type: ignore[arg-type]

    def test_each_enum_value_accepted(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        for r in (
            "irrelevant", "wrong_symbol", "stale",
            "duplicate", "too_verbose", "other",
        ):
            rec = store.report(trace_id=f"t-{r}", reason=r)  # type: ignore[arg-type]
            assert rec.reason == r

    def test_directory_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "meta"
        store = BadResultStore(nested)
        assert nested.is_dir()
        store.report(trace_id="t-1", reason="other")
        assert store.path.is_file()

    def test_accepts_string_base_dir(self, tmp_path: Path) -> None:
        store = BadResultStore(str(tmp_path / "m"))
        assert store.base_dir == tmp_path / "m"

    @pytest.mark.parametrize(
        ("reason", "channel"),
        [
            ("irrelevant", "relevance"),
            ("wrong_symbol", "relevance"),
            ("stale", "freshness"),
            ("duplicate", "diversity"),
            ("too_verbose", "packing"),
            ("other", "review"),
        ],
    )
    def test_reason_maps_to_one_learning_channel(self, reason: str, channel: str) -> None:
        assert learning_channel(reason) == channel  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# list_all + trace_ids
# ---------------------------------------------------------------------------


class TestListAndTraceIds:
    def test_empty_when_no_reports(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        assert store.list_all() == []
        assert store.trace_ids() == set()

    def test_skips_unknown_historical_reason(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        store.path.write_text(
            json.dumps({"trace_id": "bad", "reason": "invented"}) + "\n",
            encoding="utf-8",
        )

        assert store.list_all() == []

    def test_lists_each_record(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        store.report(trace_id="t-1", reason="irrelevant", detail="d1")
        store.report(trace_id="t-2", reason="stale")
        records = store.list_all()
        assert len(records) == 2
        assert {r.trace_id for r in records} == {"t-1", "t-2"}
        ids = store.trace_ids()
        assert ids == {"t-1", "t-2"}

    def test_record_to_dict_round_trip(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        store.report(trace_id="t-x", reason="duplicate", detail="dup hit")
        rec = store.list_all()[0]
        d = rec.to_dict()
        assert d == {
            "trace_id": "t-x",
            "reason": "duplicate",
            "detail": "dup hit",
            "session_id": None,
            "ts": rec.ts,
        }

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        store.report(trace_id="t-1", reason="other")
        with open(store.path, "a") as fh:
            fh.write("\n\n")
        assert len(store.list_all()) == 1

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        store.report(trace_id="t-1", reason="other")
        with open(store.path, "a") as fh:
            fh.write("garbage{{not-json\n")
        records = store.list_all()
        assert len(records) == 1
        assert records[0].trace_id == "t-1"

    def test_list_handles_missing_fields_in_record(self, tmp_path: Path) -> None:
        """A hand-written record missing some fields shouldn't crash; the
        store fills sensible defaults."""
        store = BadResultStore(tmp_path)
        store.path.write_text(json.dumps({"trace_id": "t-old"}) + "\n")
        records = store.list_all()
        assert len(records) == 1
        assert records[0].trace_id == "t-old"
        assert records[0].reason == "other"
        assert records[0].detail is None


# ---------------------------------------------------------------------------
# resolve_bad_chunks
# ---------------------------------------------------------------------------


class TestResolveBadChunks:
    def test_empty_set_short_circuits(self, tmp_path: Path) -> None:
        # No files passed, no targets — must return empty.
        assert resolve_bad_chunks(set(), [tmp_path / "missing.jsonl"]) == set()

    def test_resolves_chunks_from_single_trace_file(self, tmp_path: Path) -> None:
        f = tmp_path / "trace.jsonl"
        f.write_text(
            json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]}) + "\n"
            + json.dumps({"trace_id": "good-2", "results": [{"chunk_id": "c3"}]}) + "\n"
        )
        chunks = resolve_bad_chunks({"bad-1"}, [f])
        assert chunks == {"c1", "c2"}

    def test_handles_multiple_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text(json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c1"}]}) + "\n")
        f2.write_text(json.dumps({"trace_id": "bad-2", "results": [{"chunk_id": "c2"}]}) + "\n")
        assert resolve_bad_chunks({"bad-1", "bad-2"}, [f1, f2]) == {"c1", "c2"}

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        present = tmp_path / "present.jsonl"
        present.write_text(json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c1"}]}) + "\n")
        chunks = resolve_bad_chunks({"bad-1"}, [tmp_path / "missing.jsonl", present])
        assert chunks == {"c1"}

    def test_skips_unrelated_traces(self, tmp_path: Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text(
            json.dumps({"trace_id": "other", "results": [{"chunk_id": "c-other"}]}) + "\n"
            + json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c-bad"}]}) + "\n"
        )
        chunks = resolve_bad_chunks({"bad-1"}, [f])
        assert chunks == {"c-bad"}

    def test_tolerates_torn_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text(
            "garbage\n"
            + json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c1"}]}) + "\n"
            + "\n"
        )
        assert resolve_bad_chunks({"bad-1"}, [f]) == {"c1"}

    def test_skips_results_without_chunk_id(self, tmp_path: Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text(
            json.dumps({"trace_id": "bad-1", "results": [{"chunk_id": "c1"}, {"rel_path": "no-id"}]}) + "\n"
        )
        assert resolve_bad_chunks({"bad-1"}, [f]) == {"c1"}

    def test_tolerates_record_without_results_key(self, tmp_path: Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text(json.dumps({"trace_id": "bad-1"}) + "\n")
        assert resolve_bad_chunks({"bad-1"}, [f]) == set()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_parallel_reports_no_corruption(self, tmp_path: Path) -> None:
        store = BadResultStore(tmp_path)
        N_THREADS, N_EACH = 4, 25

        def worker() -> None:
            for _ in range(N_EACH):
                store.report(trace_id="t", reason="other")

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = store.path.read_text().strip().splitlines()
        assert len(lines) == N_THREADS * N_EACH
        # Every line must parse cleanly (no interleaving).
        for line in lines:
            assert json.loads(line)["trace_id"] == "t"


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        assert hasattr(meta, "BadResult")
        assert hasattr(meta, "BadResultStore")
        assert hasattr(meta, "FeedbackError")
        assert hasattr(meta, "resolve_bad_chunks")


# ---------------------------------------------------------------------------
# report_bad_result MCP tool wiring
# ---------------------------------------------------------------------------


class TestReportBadResultTool:
    @staticmethod
    def _ctx(app: object) -> object:
        ctx = MagicMock()
        ctx.request_context.lifespan_context = app
        return ctx

    def test_persists_when_store_available(self, tmp_path: Path) -> None:
        from stropha.server import report_bad_result

        store = BadResultStore(tmp_path)
        app = MagicMock(
            bad_result_store=store,
            trace_recorder=None,
            session_id="s-test",
        )
        out = report_bad_result(
            trace_id="t-1",
            reason="irrelevant",
            detail="wrong file",
            ctx=self._ctx(app),
        )
        assert out["ok"] is True
        assert out["trace_id"] == "t-1"
        assert out["reason"] == "irrelevant"
        # The record landed on disk.
        records = store.list_all()
        assert len(records) == 1
        assert records[0].session_id == "s-test"
        assert records[0].detail == "wrong file"

    def test_returns_error_when_store_unavailable(self, tmp_path: Path) -> None:
        from stropha.server import report_bad_result

        app = MagicMock(bad_result_store=None, trace_recorder=None, session_id="s")
        out = report_bad_result(
            trace_id="t-1", reason="other", ctx=self._ctx(app),
        )
        assert out["ok"] is False
        assert "unavailable" in out["error"]

    def test_returns_error_on_unknown_reason(self, tmp_path: Path) -> None:
        from stropha.server import report_bad_result

        store = BadResultStore(tmp_path)
        app = MagicMock(bad_result_store=store, trace_recorder=None, session_id="s")
        out = report_bad_result(
            trace_id="t-1", reason="typo", ctx=self._ctx(app),
        )
        assert out["ok"] is False
        assert "unknown reason" in out["error"]
        # The bad record was NOT persisted.
        assert store.list_all() == []

    def test_empty_session_id_passed_as_none(self, tmp_path: Path) -> None:
        """A blank session_id (default for non-MCP callers) must become None
        on disk, not the literal empty string."""
        from stropha.server import report_bad_result

        store = BadResultStore(tmp_path)
        app = MagicMock(bad_result_store=store, trace_recorder=None, session_id="")
        report_bad_result(
            trace_id="t-1", reason="other", ctx=self._ctx(app),
        )
        on_disk = store.list_all()[0]
        assert on_disk.session_id is None
