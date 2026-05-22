"""Unit tests for ArtifactStore (WIN-161).

Coverage targets every branch of every method, plus the file-locking +
crash-safety contracts that compact() and register_candidate() promise.
The tests are pure stdlib + tmp_path; no external services needed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from stropha.meta import (
    ArtifactStore,
    ArtifactStoreError,
    CandidateRecord,
)
from stropha.meta.store import (
    _now_utc_iso,
    _short_id,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_short_id_is_eight_hex(self) -> None:
        for _ in range(50):
            sid = _short_id()
            assert len(sid) == 8
            assert all(c in "0123456789abcdef" for c in sid)

    def test_short_id_collisions_are_rare(self) -> None:
        """4 billion-space, sample 1000 → expect zero collisions."""
        ids = {_short_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_now_utc_iso_has_timezone(self) -> None:
        from datetime import datetime
        ts = _now_utc_iso()
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# CandidateRecord
# ---------------------------------------------------------------------------


class TestCandidateRecord:
    def test_fitness_returns_neg_inf_when_unscored(self) -> None:
        rec = CandidateRecord(id="x", created_at="2026-01-01T00:00:00+00:00")
        assert rec.fitness() == float("-inf")

    def test_fitness_default_weights_average(self) -> None:
        rec = CandidateRecord(
            id="x",
            created_at="t",
            scores={"recall": 0.8, "mrr": 0.6},
        )
        assert rec.fitness() == pytest.approx(0.7)

    def test_fitness_with_weights(self) -> None:
        rec = CandidateRecord(
            id="x",
            created_at="t",
            scores={"recall": 0.8, "mrr": 0.6},
        )
        # recall*2 + mrr*1 = 1.6 + 0.6 = 2.2
        assert rec.fitness({"recall": 2.0, "mrr": 1.0}) == pytest.approx(2.2)

    def test_fitness_missing_weight_defaults_to_one(self) -> None:
        rec = CandidateRecord(id="x", created_at="t", scores={"a": 1.0, "b": 2.0})
        # weight for 'b' missing → 1.0; total 1.0 + 2.0 = 3.0
        assert rec.fitness({"a": 1.0}) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ArtifactStore construction
# ---------------------------------------------------------------------------


class TestStoreConstruction:
    def test_creates_directories(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "meta")
        assert (tmp_path / "meta" / "candidates").is_dir()
        assert (tmp_path / "meta" / "runs").is_dir()
        # index.jsonl is created lazily, not on construction.
        assert not (tmp_path / "meta" / "index.jsonl").exists()
        assert store.index_path == tmp_path / "meta" / "index.jsonl"

    def test_accepts_string_base_dir(self, tmp_path: Path) -> None:
        store = ArtifactStore(str(tmp_path / "meta"))
        assert store.base_dir == tmp_path / "meta"

    def test_idempotent_on_existing_dir(self, tmp_path: Path) -> None:
        ArtifactStore(tmp_path / "meta")
        ArtifactStore(tmp_path / "meta")  # second construction must not raise.


# ---------------------------------------------------------------------------
# register_candidate
# ---------------------------------------------------------------------------


class TestRegisterCandidate:
    def test_basic_registration(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({"inference": {"backend": "mlx"}})
        # Filesystem layout matches the spec exactly.
        cand_dir = store.candidates_dir / rec.id
        assert cand_dir.is_dir()
        assert (cand_dir / "config.yaml").is_file()
        assert (cand_dir / "meta.json").is_file()
        assert (cand_dir / "scores.json").is_file()
        assert (cand_dir / "traces").is_dir()
        # index.jsonl now has one line.
        lines = store.index_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["id"] == rec.id
        assert entry["parent_id"] is None
        assert "created_at" in entry

    def test_config_serialised_as_yaml(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        cfg = {"a": 1, "nested": {"b": [1, 2, 3]}}
        rec = store.register_candidate(cfg)
        roundtrip = yaml.safe_load(
            (store.candidates_dir / rec.id / "config.yaml").read_text()
        )
        assert roundtrip == cfg

    def test_meta_includes_label_and_parent(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        parent = store.register_candidate({"v": 1}, label="root")
        child = store.register_candidate({"v": 2}, parent=parent.id, label="v2")
        meta = json.loads((store.candidates_dir / child.id / "meta.json").read_text())
        assert meta["parent_id"] == parent.id
        assert meta["label"] == "v2"

    def test_caller_supplied_id(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({}, candidate_id="fixed-id-1")
        assert rec.id == "fixed-id-1"
        assert (store.candidates_dir / "fixed-id-1").is_dir()

    def test_collision_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.register_candidate({}, candidate_id="dup")
        with pytest.raises(ArtifactStoreError, match="already exists"):
            store.register_candidate({}, candidate_id="dup")

    def test_register_with_custom_meta(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({}, meta={"proposer": "bandit", "seed": 42})
        assert rec.meta == {"proposer": "bandit", "seed": 42}
        on_disk = json.loads((store.candidates_dir / rec.id / "meta.json").read_text())
        assert on_disk["meta"]["seed"] == 42

    def test_register_rolls_back_on_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If side-car writing fails, the candidate dir must be removed.

        Forcing yaml.safe_dump to fail simulates an unserialisable config.
        """
        import yaml as _yaml

        store = ArtifactStore(tmp_path)

        def boom(*args: Any, **kwargs: Any) -> str:
            raise TypeError("non-serialisable")

        monkeypatch.setattr(_yaml, "safe_dump", boom)
        # Also patch the symbol inside the store module since the import
        # was already resolved.
        monkeypatch.setattr("stropha.meta.store.yaml.safe_dump", boom)

        with pytest.raises(ArtifactStoreError, match="failed to write"):
            store.register_candidate({"x": 1}, candidate_id="rollme")
        # The candidate directory must be GONE — no half-state on disk.
        assert not (store.candidates_dir / "rollme").exists()


class TestWriteScores:
    def test_writes_and_overwrites(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        store.write_scores(rec.id, {"recall": 0.7})
        on_disk = json.loads((store.candidates_dir / rec.id / "scores.json").read_text())
        assert on_disk == {"recall": 0.7}
        # Second call REPLACES (does not merge).
        store.write_scores(rec.id, {"mrr": 0.5})
        on_disk = json.loads((store.candidates_dir / rec.id / "scores.json").read_text())
        assert on_disk == {"mrr": 0.5}

    def test_unknown_candidate_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(ArtifactStoreError, match="missing"):
            store.write_scores("nonexistent", {"recall": 0.9})

    def test_write_scores_handles_serialisation_error(
        self, tmp_path: Path
    ) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})

        class Unserialisable:
            pass

        # A custom object that json.dumps cannot handle.
        with pytest.raises(ArtifactStoreError, match="failed to write scores"):
            store.write_scores(rec.id, {"x": Unserialisable()})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# read_candidate
# ---------------------------------------------------------------------------


class TestReadCandidate:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        cfg = {"inference": {"backend": "deepseek"}}
        rec = store.register_candidate(cfg, label="rt", meta={"k": 1})
        store.write_scores(rec.id, {"recall": 0.9})
        loaded = store.read_candidate(rec.id)
        assert loaded.id == rec.id
        assert loaded.label == "rt"
        assert loaded.config == cfg
        assert loaded.meta == {"k": 1}
        assert loaded.scores == {"recall": 0.9}
        assert loaded.trace_count == 0
        assert loaded.pruned is False

    def test_missing_candidate_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(ArtifactStoreError, match="missing"):
            store.read_candidate("nope")

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        (store.candidates_dir / rec.id / "meta.json").unlink()
        with pytest.raises(ArtifactStoreError, match=r"meta\.json"):
            store.read_candidate(rec.id)

    def test_malformed_meta_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        (store.candidates_dir / rec.id / "meta.json").write_text("not json{{")
        with pytest.raises(ArtifactStoreError, match=r"meta\.json"):
            store.read_candidate(rec.id)

    def test_missing_config_tolerated(self, tmp_path: Path) -> None:
        """A candidate with no config.yaml still reads — config defaults to {}."""
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({"a": 1})
        (store.candidates_dir / rec.id / "config.yaml").unlink()
        loaded = store.read_candidate(rec.id)
        assert loaded.config == {}

    def test_yaml_non_dict_tolerated(self, tmp_path: Path) -> None:
        """A config.yaml that holds a non-dict (e.g. plain string) is ignored."""
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({"a": 1})
        (store.candidates_dir / rec.id / "config.yaml").write_text("just a string\n")
        loaded = store.read_candidate(rec.id)
        assert loaded.config == {}

    def test_missing_scores_tolerated(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        (store.candidates_dir / rec.id / "scores.json").unlink()
        loaded = store.read_candidate(rec.id)
        assert loaded.scores == {}

    def test_malformed_scores_tolerated(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        (store.candidates_dir / rec.id / "scores.json").write_text("not json{{")
        loaded = store.read_candidate(rec.id)
        assert loaded.scores == {}

    def test_trace_count_reflects_files(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        traces = store.candidates_dir / rec.id / "traces"
        (traces / "trace-2026-05-22.jsonl").write_text('{"ok": 1}\n')
        (traces / "trace-2026-05-23.jsonl").write_text('{"ok": 2}\n')
        loaded = store.read_candidate(rec.id)
        assert loaded.trace_count == 2


# ---------------------------------------------------------------------------
# list_candidates
# ---------------------------------------------------------------------------


class TestListCandidates:
    def test_empty_store(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.list_candidates() == []

    def test_lists_in_newest_first_order(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        ids = []
        for i in range(3):
            rec = store.register_candidate({"v": i})
            ids.append(rec.id)
            import time
            time.sleep(0.005)  # ensure created_at differs
        listed = store.list_candidates()
        # newest first = reverse of insertion order
        assert [r.id for r in listed] == list(reversed(ids))

    def test_sort_by_id(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.register_candidate({}, candidate_id="bbb")
        store.register_candidate({}, candidate_id="aaa")
        store.register_candidate({}, candidate_id="ccc")
        listed = store.list_candidates(sort_by="id")
        assert [r.id for r in listed] == ["aaa", "bbb", "ccc"]

    def test_sort_by_fitness(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        a = store.register_candidate({})
        b = store.register_candidate({})
        c = store.register_candidate({})  # unscored — sorts last via -inf
        store.write_scores(a.id, {"recall": 0.6})
        store.write_scores(b.id, {"recall": 0.9})
        listed = store.list_candidates(sort_by="fitness")
        assert [r.id for r in listed] == [b.id, a.id, c.id]

    def test_sort_by_score_key(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        a = store.register_candidate({})
        b = store.register_candidate({})
        store.write_scores(a.id, {"recall": 0.3, "latency": 100})
        store.write_scores(b.id, {"recall": 0.9, "latency": 50})
        # Highest latency first.
        listed = store.list_candidates(sort_by="latency")
        assert [r.id for r in listed] == [a.id, b.id]

    def test_limit_applies(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        for _ in range(5):
            store.register_candidate({})
        listed = store.list_candidates(limit=2)
        assert len(listed) == 2

    def test_skips_missing_candidate_in_index(self, tmp_path: Path) -> None:
        """If the ledger references a candidate whose dir was deleted out
        of band, list() must skip + log, not crash."""
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        # Delete the candidate directory directly (simulating data corruption).
        import shutil
        shutil.rmtree(store.candidates_dir / rec.id)
        listed = store.list_candidates()
        assert listed == []

    def test_pruned_excluded_by_default(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        for _ in range(3):
            store.register_candidate({})
        # Compact aggressively so something gets pruned.
        store.compact(keep_top=1, keep_bottom=0, keep_recent=0)
        listed = store.list_candidates(include_pruned=False)
        assert all(not r.pruned for r in listed)
        listed_all = store.list_candidates(include_pruned=True)
        assert any(r.pruned for r in listed_all)


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    def test_empty_objectives_returns_empty(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.register_candidate({})
        assert store.pareto_frontier({}) == []

    def test_single_candidate_is_on_frontier(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        store.write_scores(rec.id, {"recall": 0.8})
        front = store.pareto_frontier({"recall": +1})
        assert len(front) == 1 and front[0].id == rec.id

    def test_dominated_candidate_excluded(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        a = store.register_candidate({})  # better on both dims
        b = store.register_candidate({})  # strictly worse
        store.write_scores(a.id, {"recall": 0.9, "mrr": 0.8})
        store.write_scores(b.id, {"recall": 0.5, "mrr": 0.4})
        front = store.pareto_frontier({"recall": +1, "mrr": +1})
        assert [r.id for r in front] == [a.id]

    def test_tradeoff_pair_both_on_frontier(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        a = store.register_candidate({})
        b = store.register_candidate({})
        # Each wins on one dimension → both non-dominated.
        store.write_scores(a.id, {"recall": 0.9, "latency": 100})
        store.write_scores(b.id, {"recall": 0.7, "latency": 30})
        front = store.pareto_frontier({"recall": +1, "latency": -1})
        assert {r.id for r in front} == {a.id, b.id}

    def test_missing_score_excludes_candidate(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        a = store.register_candidate({})
        b = store.register_candidate({})
        # `a` has both metrics; `b` has only one — `b` is excluded.
        store.write_scores(a.id, {"recall": 0.5, "mrr": 0.5})
        store.write_scores(b.id, {"recall": 0.9})
        front = store.pareto_frontier({"recall": +1, "mrr": +1})
        assert [r.id for r in front] == [a.id]


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


class TestCompact:
    def test_empty_store_returns_empty(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.compact() == []

    def test_keep_top_survives(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        records = []
        for i in range(5):
            r = store.register_candidate({"v": i})
            store.write_scores(r.id, {"score": float(i)})  # 0.0 .. 4.0
            records.append(r)
        pruned = store.compact(keep_top=2, keep_bottom=0, keep_recent=0)
        survivors = {r.id for r in store.list_candidates(include_pruned=False)}
        # The top-2 by score are records 4 and 3.
        assert survivors == {records[4].id, records[3].id}
        # The other 3 should be in pruned_ids.
        assert set(pruned) == {records[0].id, records[1].id, records[2].id}

    def test_keep_bottom_preserves_anti_examples(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        records = []
        for i in range(5):
            r = store.register_candidate({"v": i})
            store.write_scores(r.id, {"score": float(i)})
            records.append(r)
        store.compact(keep_top=0, keep_bottom=2, keep_recent=0)
        survivors = {r.id for r in store.list_candidates(include_pruned=False)}
        # Bottom-2 by score = records 0 and 1.
        assert survivors == {records[0].id, records[1].id}

    def test_keep_recent_by_creation_time(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        records = []
        for i in range(4):
            r = store.register_candidate({"v": i})
            records.append(r)
            import time
            time.sleep(0.005)
        store.compact(keep_top=0, keep_bottom=0, keep_recent=2)
        survivors = {r.id for r in store.list_candidates(include_pruned=False)}
        # Most recent 2 = records 2 and 3.
        assert survivors == {records[2].id, records[3].id}

    def test_buckets_union(self, tmp_path: Path) -> None:
        """Top + bottom + recent buckets should union, not intersect."""
        store = ArtifactStore(tmp_path)
        records = []
        for i in range(6):
            r = store.register_candidate({})
            store.write_scores(r.id, {"score": float(i)})
            records.append(r)
            import time
            time.sleep(0.005)
        store.compact(keep_top=1, keep_bottom=1, keep_recent=1)
        survivors = {r.id for r in store.list_candidates(include_pruned=False)}
        # Top by score = record 5 (also the most recent).
        # Bottom = record 0.
        # Recent = record 5.
        # Union: {0, 5}.
        assert survivors == {records[0].id, records[5].id}

    def test_pruned_marker_in_ledger(self, tmp_path: Path) -> None:
        """Pruned candidates leave a tombstone in index.jsonl."""
        store = ArtifactStore(tmp_path)
        ids = []
        for _ in range(3):
            ids.append(store.register_candidate({}).id)
        store.compact(keep_top=1, keep_bottom=0, keep_recent=0)
        ledger = [
            json.loads(line)
            for line in store.index_path.read_text().splitlines()
            if line.strip()
        ]
        pruned = [e for e in ledger if e.get("pruned")]
        assert len(pruned) == 2

    def test_pruned_directories_removed(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        loser_ids = []
        for i in range(3):
            r = store.register_candidate({})
            store.write_scores(r.id, {"score": float(i)})
            loser_ids.append(r.id)
        store.compact(keep_top=1, keep_bottom=0, keep_recent=0)
        # The two losers' directories no longer exist on disk.
        survivors = {r.id for r in store.list_candidates(include_pruned=False)}
        for cid in loser_ids:
            if cid not in survivors:
                assert not (store.candidates_dir / cid).exists()

    def test_negative_keep_values_treated_as_zero(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        for _ in range(3):
            store.register_candidate({})
        # Negative bucket sizes should not crash; treated as zero.
        pruned = store.compact(keep_top=-5, keep_bottom=-5, keep_recent=-5)
        assert len(pruned) == 3


# ---------------------------------------------------------------------------
# Concurrency & atomicity
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_parallel_register_no_corruption(self, tmp_path: Path) -> None:
        """N threads each register M candidates; the ledger has N*M valid lines."""
        store = ArtifactStore(tmp_path)
        N_THREADS, M_EACH = 4, 25

        def worker() -> None:
            for _ in range(M_EACH):
                store.register_candidate({})

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = store.index_path.read_text().strip().splitlines()
        assert len(lines) == N_THREADS * M_EACH
        ids = {json.loads(line)["id"] for line in lines}
        assert len(ids) == N_THREADS * M_EACH

    def test_index_torn_line_tolerated(self, tmp_path: Path) -> None:
        """A malformed line in index.jsonl is skipped, not raised."""
        store = ArtifactStore(tmp_path)
        store.register_candidate({})
        # Append a garbled line directly.
        with open(store.index_path, "a") as fh:
            fh.write("not-valid-json{{\n")
        # list_candidates() must still succeed (logs and skips bad line).
        listed = store.list_candidates()
        assert len(listed) == 1


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_package_exports(self) -> None:
        from stropha import meta
        assert hasattr(meta, "ArtifactStore")
        assert hasattr(meta, "ArtifactStoreError")
        assert hasattr(meta, "CandidateRecord")


# ---------------------------------------------------------------------------
# Branch coverage — tolerated edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_blank_line_in_index_skipped(self, tmp_path: Path) -> None:
        """Empty lines inside ``index.jsonl`` must be skipped silently."""
        store = ArtifactStore(tmp_path)
        store.register_candidate({})
        # Append a blank line directly to the ledger.
        with open(store.index_path, "a") as fh:
            fh.write("\n\n")
        # _read_index must skip blank lines without errors.
        listed = store.list_candidates()
        assert len(listed) == 1

    def test_read_candidate_reflects_pruned_flag(self, tmp_path: Path) -> None:
        """When a candidate survives compact (kept top) AND someone manually
        flips its ledger entry to pruned=true, ``read_candidate`` reports it.
        """
        store = ArtifactStore(tmp_path)
        rec = store.register_candidate({})
        # Manually mark this id as pruned in the ledger (without deleting dir).
        store._mark_pruned([rec.id])
        loaded = store.read_candidate(rec.id)
        assert loaded.pruned is True
