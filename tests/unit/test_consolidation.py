"""Unit tests for memory consolidation (WIN-226).

Covers the ``stropha.meta.consolidation`` module + the
``stropha memory consolidate`` CLI command. The pass turns
episodic memory's append-only timeline into bounded semantic
memory entries (kind="co-touched").
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stropha.cli import app
from stropha.ingest.git_meta import RepoIdentity
from stropha.meta.consolidation import (
    CONSOLIDATED_KIND,
    DEFAULT_MIN_COUNT,
    DEFAULT_MIN_EPISODES,
    ConsolidationCandidate,
    ConsolidationPlan,
    apply_consolidation,
    plan_consolidation,
)
from stropha.storage.sqlite import Storage


@pytest.fixture
def storage_with_repo(tmp_path: Path):
    s = Storage(tmp_path / "cons.db", embedding_dim=4)
    s.__enter__()
    repo_id = s.register_repo(
        RepoIdentity(
            normalized_key="github.com/test/repo",
            remote_url=None,
            root_path=tmp_path / "repo",
            default_branch="main",
            head_commit="abc",
        )
    )
    yield s, repo_id
    s.__exit__(None, None, None)


def _seed_episodes(
    storage: Storage,
    repo_id: int,
    episodes: list[dict],
) -> None:
    """Insert multiple episodes via Storage.record_episode."""
    for ep in episodes:
        storage.record_episode(
            repo_id=repo_id,
            title=ep.get("title", "t"),
            summary=ep.get("summary", ""),
            touched_files=ep.get("files", []),
            touched_symbols=ep.get("symbols", []),
        )


# ---------------------------------------------------------------------------
# ConsolidationCandidate.render_value
# ---------------------------------------------------------------------------


class TestCandidateRender:
    def test_files_and_symbols(self) -> None:
        c = ConsolidationCandidate(
            key="src/fsrs.py",
            top_files={"tests/test_fsrs.py": 4, "docs/fsrs.md": 2},
            top_symbols={"FsrsCalc.update": 3},
        )
        out = c.render_value()
        assert "tests/test_fsrs.py (4x)" in out
        assert "docs/fsrs.md (2x)" in out
        assert "FsrsCalc.update (3x)" in out

    def test_files_only(self) -> None:
        c = ConsolidationCandidate(
            key="x", top_files={"a.py": 2},
        )
        assert "files: a.py (2x)" in c.render_value()
        assert "symbols" not in c.render_value()

    def test_symbols_only(self) -> None:
        c = ConsolidationCandidate(
            key="x", top_symbols={"Foo.bar": 2},
        )
        assert "symbols: Foo.bar (2x)" in c.render_value()

    def test_empty_renders_fallback(self) -> None:
        c = ConsolidationCandidate(key="lonely")
        assert "No co-touches" in c.render_value()
        assert "lonely" in c.render_value()


# ---------------------------------------------------------------------------
# plan_consolidation
# ---------------------------------------------------------------------------


class TestPlanConsolidation:
    def test_below_min_episodes_returns_empty(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        # 2 episodes < default min_episodes=3.
        _seed_episodes(s, repo_id, [
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id)
        assert plan.candidates == []
        assert plan.episodes_to_prune == 0

    def test_basic_co_occurrence_pattern_surfaced(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["src/fsrs.py", "tests/test_fsrs.py"]},
            {"files": ["src/fsrs.py", "tests/test_fsrs.py"]},
            {"files": ["src/fsrs.py", "tests/test_fsrs.py", "docs/fsrs.md"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, min_count=2)
        # Two candidates: fsrs.py and test_fsrs.py both reach the
        # threshold against each other.
        keys = {c.key for c in plan.candidates}
        assert "src/fsrs.py" in keys
        assert "tests/test_fsrs.py" in keys
        fsrs_candidate = next(
            c for c in plan.candidates if c.key == "src/fsrs.py"
        )
        assert fsrs_candidate.top_files.get("tests/test_fsrs.py") == 3

    def test_min_count_filter_drops_noise(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        # a.py + b.py together 3 times; a.py + c.py once.
        _seed_episodes(s, repo_id, [
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py", "c.py"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, min_count=2)
        a_candidate = next(c for c in plan.candidates if c.key == "a.py")
        # b.py survives (3x >= 2), c.py is dropped (1x < 2).
        assert a_candidate.top_files.get("b.py") == 3
        assert "c.py" not in a_candidate.top_files

    def test_no_co_occurrence_no_candidate(
        self, storage_with_repo,
    ) -> None:
        """Files touched alone (never paired) don't qualify."""
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["solo1.py"]},
            {"files": ["solo2.py"]},
            {"files": ["solo3.py"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, min_count=2)
        # Every "neighbour" count is 0 → no candidates pass min_count.
        assert plan.candidates == []

    def test_prune_count_computed(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        # Plant an old episode directly so we control timestamps.
        old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "old", "", "[]", "[]", old, old),
        )
        s._conn.commit()
        # Plus 2 recent episodes so min_episodes is met.
        _seed_episodes(s, repo_id, [
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
        ])
        plan = plan_consolidation(
            s, repo_id=repo_id, prune_after_days=90,
        )
        assert plan.episodes_to_prune == 1
        assert plan.prune_cutoff_iso is not None

    def test_prune_disabled_when_days_is_none(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
        ])
        plan = plan_consolidation(
            s, repo_id=repo_id, prune_after_days=None,
        )
        assert plan.prune_cutoff_iso is None
        assert plan.episodes_to_prune == 0


# ---------------------------------------------------------------------------
# apply_consolidation
# ---------------------------------------------------------------------------


class TestApplyConsolidation:
    def test_writes_project_memory_entries(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, min_count=2)
        result = apply_consolidation(s, plan)
        assert result["written"] >= 2  # a.py + b.py
        # project_memory now has entries under the CONSOLIDATED_KIND.
        rows = s.recall(repo_id=repo_id, kind=CONSOLIDATED_KIND)
        assert len(rows) >= 2

    def test_apply_idempotent_via_upsert(
        self, storage_with_repo,
    ) -> None:
        """Running apply twice doesn't duplicate — Storage.remember
        UPSERTs by (repo_id, kind, key)."""
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
            {"files": ["a.py", "b.py"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, min_count=2)
        apply_consolidation(s, plan)
        first_count = len(s.recall(repo_id=repo_id, kind=CONSOLIDATED_KIND))
        # Same plan applied again — count stays the same.
        apply_consolidation(s, plan)
        second_count = len(s.recall(repo_id=repo_id, kind=CONSOLIDATED_KIND))
        assert second_count == first_count

    def test_prune_removes_old_episodes(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (repo_id, "ancient", "", "[]", "[]", old, old),
        )
        s._conn.commit()
        # Add recent episodes so min_episodes is met.
        _seed_episodes(s, repo_id, [
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
        ])
        plan = plan_consolidation(
            s, repo_id=repo_id, prune_after_days=90,
        )
        result = apply_consolidation(s, plan)
        assert result["pruned"] == 1
        # The old episode is gone.
        titles = {ep.title for ep in s.recall_episodes(repo_id=repo_id)}
        assert "ancient" not in titles

    def test_apply_records_source(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id)
        apply_consolidation(s, plan, source="consolidator-job-1")
        rows = s.recall(repo_id=repo_id, kind=CONSOLIDATED_KIND)
        assert rows and all(r.source == "consolidator-job-1" for r in rows)

    def test_empty_plan_no_op(
        self, storage_with_repo,
    ) -> None:
        s, repo_id = storage_with_repo
        plan = ConsolidationPlan(repo_id=repo_id)
        result = apply_consolidation(s, plan)
        assert result == {"written": 0, "pruned": 0}


# ---------------------------------------------------------------------------
# CLI: stropha memory consolidate
# ---------------------------------------------------------------------------


class TestCliMemoryConsolidate:
    def test_command_registered(self) -> None:
        memory_group = None
        for group in app.registered_groups:
            if group.name == "memory":
                memory_group = group.typer_instance
                break
        assert memory_group is not None
        names = {cmd.name for cmd in memory_group.registered_commands}
        assert "consolidate" in names

    def test_unindexed_repo_exits_nonzero(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """CLI fails cleanly when target_repo has no row in the
        repos table. We monkeypatch build_stages so the test doesn't
        depend on the embedder backend being installed."""
        import stropha.cli as cli_mod

        storage = Storage(tmp_path / "no-repo.db", embedding_dim=4)
        storage.__enter__()

        class FakeBuilt:
            def __init__(self):
                self.storage = storage

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            cli_mod, "load_pipeline_config", lambda: {},
        )
        monkeypatch.setattr(
            cli_mod, "build_stages", lambda _cfg: FakeBuilt(),
        )

        try:
            result = CliRunner().invoke(
                app,
                ["memory", "consolidate", "--repo", str(tmp_path)],
            )
            assert result.exit_code == 1
            assert "No indexed repo" in result.output
        finally:
            storage.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Default constants exported (public API stability)
# ---------------------------------------------------------------------------


class TestDefensivePaths:
    """Exercise the defensive exception handlers + the empty-string
    skip in _episode_keys so the 6 uncovered lines hit 100%."""

    def test_empty_symbol_skipped(
        self, storage_with_repo, monkeypatch,
    ) -> None:
        """An episode whose touched_symbols contains an empty string
        — the ``if s:`` guard skips it without adding to the keys
        set. We patch record_episode's dedup so the empty makes it
        into the row, then verify _episode_keys ignores it."""
        from stropha.meta.consolidation import _episode_keys

        s, repo_id = storage_with_repo
        # Plant a row with an empty-string symbol directly.
        import json as _json
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        s._conn.execute(
            """INSERT INTO episodes (
                   repo_id, title, summary, touched_files,
                   touched_symbols, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                repo_id, "t", "",
                _json.dumps(["a.py", ""]),  # empty file too
                _json.dumps(["", "Foo.bar"]),
                now, now,
            ),
        )
        s._conn.commit()
        keys = _episode_keys(s, repo_id=repo_id, limit=10)
        # Empty strings dropped; real keys survive.
        assert "" not in keys
        assert "a.py" in keys
        assert "Foo.bar" in keys

    def test_plan_swallows_prune_count_db_error(
        self, storage_with_repo,
    ) -> None:
        """Defensive handler for SELECT COUNT during prune-count.
        Wraps the storage's ``_conn`` in a proxy that fails the
        prune-count query specifically; confirms plan_consolidation
        still returns (with episodes_to_prune=0)."""
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
        ])
        real_conn = s._conn

        class FailingConn:
            def execute(self, query, *args, **kwargs):
                if "COUNT(*)" in query and "episodes" in query:
                    raise RuntimeError("simulated db error")
                return real_conn.execute(query, *args, **kwargs)

            def commit(self):
                return real_conn.commit()

        s._conn = FailingConn()  # type: ignore[assignment]
        try:
            plan = plan_consolidation(
                s, repo_id=repo_id, prune_after_days=30,
            )
        finally:
            s._conn = real_conn  # restore
        assert plan.episodes_to_prune == 0  # error swallowed
        # The plan still has candidates from the successful co-occur
        # step — the prune-count failure didn't abort the whole pass.
        assert plan.candidates

    def test_apply_swallows_prune_delete_db_error(
        self, storage_with_repo,
    ) -> None:
        """Defensive handler for the DELETE during apply."""
        s, repo_id = storage_with_repo
        _seed_episodes(s, repo_id, [
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
            {"files": ["a", "b"]},
        ])
        plan = plan_consolidation(s, repo_id=repo_id, prune_after_days=30)
        real_conn = s._conn

        class FailingDelete:
            def execute(self, query, *args, **kwargs):
                if query.strip().upper().startswith("DELETE FROM EPISODES"):
                    raise RuntimeError("simulated delete error")
                return real_conn.execute(query, *args, **kwargs)

            def commit(self):
                return real_conn.commit()

            def cursor(self):
                return real_conn.cursor()

            def __getattr__(self, name):
                # Proxy any other attribute access to the real conn.
                return getattr(real_conn, name)

        s._conn = FailingDelete()  # type: ignore[assignment]
        try:
            result = apply_consolidation(s, plan)
        finally:
            s._conn = real_conn
        assert result["pruned"] == 0
        assert result["written"] >= 1


class TestPublicExports:
    def test_defaults_exposed(self) -> None:
        from stropha.meta.consolidation import (
            CONSOLIDATED_KIND,
            DEFAULT_MIN_COUNT,
            DEFAULT_MIN_EPISODES,
            DEFAULT_PRUNE_AFTER_DAYS,
            DEFAULT_TOP_N_PER_KEY,
        )
        assert CONSOLIDATED_KIND == "co-touched"
        assert DEFAULT_MIN_COUNT == 2
        assert DEFAULT_MIN_EPISODES == 3
        assert DEFAULT_PRUNE_AFTER_DAYS == 90
        assert DEFAULT_TOP_N_PER_KEY == 5
