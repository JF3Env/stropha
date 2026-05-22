"""Unit tests for the ``stropha meta`` CLI commands (WIN-193) +
end-to-end loop integration (WIN-194)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from stropha.cli import app
from stropha.meta import (
    ArtifactStore,
    ScoreCard,
    run_loop,
)
from stropha.meta.proposer import make_bandit_proposer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(tmp_path: Path) -> Path:
    """Build a meta dir with 4 candidates of varying fitness."""
    meta = tmp_path / "meta"
    store = ArtifactStore(meta)
    for i, q in enumerate([0.3, 0.6, 0.9, 0.5]):
        rec = store.register_candidate({"v": i}, label=f"v{i}")
        store.write_scores(rec.id, {
            "quality": q,
            "context_efficiency": 0.5,
            "outcome_score": 0.2,
            "latency_ms": 20.0 * (i + 1),
        })
    return meta


def _runner_invoke(tmp_path: Path, args: list[str]) -> Any:
    """Run the CLI with a fake config so _meta_dir_default works."""
    return CliRunner().invoke(app, args)


# ---------------------------------------------------------------------------
# stropha meta run (WIN-193)
# ---------------------------------------------------------------------------


class TestMetaRun:
    def test_runs_bandit_loop(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            ["meta", "run",
             "--iterations", "3",
             "--proposer", "bandit",
             "--seed", "42",
             "--meta-dir", str(tmp_path / "meta")],
        )
        assert result.exit_code == 0, result.output
        # The output mentions the loop result.
        assert "iterations" in result.output or "frontier_size" in result.output

    def test_unknown_proposer_exits_nonzero(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            ["meta", "run",
             "--iterations", "1",
             "--proposer", "unknown",
             "--meta-dir", str(tmp_path / "meta")],
        )
        assert result.exit_code == 1
        assert "Unknown proposer" in result.output

    def test_local_agentic_proposer_can_be_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no backend is available, the local-agentic proposer
        falls back to the bandit — the CLI command should still exit 0."""
        # Force get_backend to raise so we exercise the fallback path.
        monkeypatch.setattr(
            "stropha.meta.proposer.inference_helper.get_backend",
            lambda **_: (_ for _ in ()).throw(RuntimeError("no backend")),
        )
        result = CliRunner().invoke(
            app,
            ["meta", "run",
             "--iterations", "2",
             "--proposer", "local-agentic",
             "--seed", "42",
             "--meta-dir", str(tmp_path / "meta")],
        )
        assert result.exit_code == 0, result.output

    def test_cloud_agentic_proposer_can_be_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "stropha.meta.proposer.agentic.get_backend",
            lambda **_: (_ for _ in ()).throw(RuntimeError("no key")),
        )
        result = CliRunner().invoke(
            app,
            ["meta", "run",
             "--iterations", "2",
             "--proposer", "cloud-agentic",
             "--seed", "42",
             "--meta-dir", str(tmp_path / "meta")],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# stropha meta status
# ---------------------------------------------------------------------------


class TestMetaStatus:
    def test_no_candidates_returns_yellow(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app, ["meta", "status", "--meta-dir", str(tmp_path / "empty")]
        )
        assert result.exit_code == 0
        assert "No candidates" in result.output

    def test_reports_archive_and_frontier(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "status", "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0
        assert "archive" in result.output
        assert "frontier" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "status", "--meta-dir", str(meta), "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["archive_size"] == 4


# ---------------------------------------------------------------------------
# stropha meta report
# ---------------------------------------------------------------------------


class TestMetaReport:
    def test_no_candidates(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(tmp_path / "x")]
        )
        assert result.exit_code == 0
        assert "No candidates" in result.output

    def test_table_output(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta), "--limit", "10"]
        )
        assert result.exit_code == 0
        # Highest-fitness candidate (v2, quality 0.9) listed first.
        assert "v2" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta), "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        assert len(payload) >= 1


# ---------------------------------------------------------------------------
# stropha meta promote
# ---------------------------------------------------------------------------


class TestMetaPromote:
    def test_unknown_candidate_exits_nonzero(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", "no-such-id",
             "--meta-dir", str(meta)],
        )
        assert result.exit_code == 1

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        # Pick a real candidate id.
        rec = ArtifactStore(meta).list_candidates()[0]
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id,
             "--meta-dir", str(meta), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_real_promote_writes_stropha_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        # Run inside a temp cwd so we don't clobber the real stropha.yaml.
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 0
        assert (target_cwd / "stropha.yaml").is_file()
        assert "Promoted" in result.output


# ---------------------------------------------------------------------------
# stropha meta trace
# ---------------------------------------------------------------------------


class TestMetaTrace:
    def test_unknown_candidate_exits_nonzero(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "trace", "nope", "--meta-dir", str(meta)]
        )
        assert result.exit_code == 1

    def test_lists_traces(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        # Manually drop a trace file.
        assert rec.base_dir is not None
        (rec.base_dir / "traces" / "t.jsonl").write_text('{"x": 1}\n')
        result = CliRunner().invoke(
            app, ["meta", "trace", rec.id, "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0
        assert "t.jsonl" in result.output

    def test_no_traces_dir(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        assert rec.base_dir is not None
        import shutil
        shutil.rmtree(rec.base_dir / "traces")
        result = CliRunner().invoke(
            app, ["meta", "trace", rec.id, "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0
        assert "No traces" in result.output


# ---------------------------------------------------------------------------
# stropha meta snapshot
# ---------------------------------------------------------------------------


class TestMetaSnapshot:
    def _init_repo(self, repo: Path) -> None:
        import subprocess
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t",
             "commit", "-q", "-m", "init"],
            cwd=repo, check=True,
        )

    def test_create_snapshot(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        result = CliRunner().invoke(
            app, ["meta", "snapshot", "--repo", str(repo)]
        )
        assert result.exit_code == 0, result.output
        assert "Snapshot:" in result.output

    def test_list_snapshots(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        # First create one.
        CliRunner().invoke(app, ["meta", "snapshot", "--repo", str(repo)])
        # Then list.
        result = CliRunner().invoke(
            app, ["meta", "snapshot", "--repo", str(repo), "--list"]
        )
        assert result.exit_code == 0
        # Output has at least one SHA.
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert any(len(line) == 40 for line in lines)


# ---------------------------------------------------------------------------
# WIN-194 — End-to-end integration tests
# ---------------------------------------------------------------------------


class TestLoopWithBanditEnd2End:
    def _constraint_free_space(self) -> Any:
        """Use a stub space without env-gated constraints so the loop's
        budget is deterministic regardless of which env vars are set
        in the test environment."""
        from stropha.meta import Dimension, SearchSpace
        return SearchSpace(
            dimensions=[
                Dimension(name="walker", kind="stage", indexing=True,
                          type="choice", choices=("git-ls-files", "filesystem")),
                Dimension(name="enricher", kind="stage", indexing=True,
                          type="choice", choices=("noop", "hierarchical")),
                Dimension(name="embedder", kind="stage", indexing=True,
                          type="choice", choices=("local",)),
            ],
        )

    def test_loop_persists_full_pipeline(self, tmp_path: Path) -> None:
        """proposer → loop → store → frontier full round trip."""
        space = self._constraint_free_space()
        prop = make_bandit_proposer(space, seed=42)
        store = ArtifactStore(tmp_path / "meta")

        def eval_(cfg: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.5, latency_ms=50)

        result = run_loop(
            proposer=prop, evaluator=eval_, store=store,
            max_iterations=5,
        )
        # Loop completed.
        assert result.iterations == 5
        # Store has the same number of candidates.
        assert len(store.list_candidates()) == 5
        # Each candidate has a scores.json populated.
        for c in store.list_candidates():
            assert "quality" in c.scores

    def test_loop_with_seeds_then_proposer(self, tmp_path: Path) -> None:
        space = self._constraint_free_space()
        prop = make_bandit_proposer(space, seed=42)
        store = ArtifactStore(tmp_path / "meta")
        seeds = [{"walker": "git-ls-files"}, {"walker": "filesystem"}]

        def eval_(_cfg: dict[str, Any]) -> ScoreCard:
            return ScoreCard(quality=0.7)

        result = run_loop(
            proposer=prop, evaluator=eval_, store=store,
            seeds=seeds, max_iterations=4,
        )
        assert result.iterations == 4
        labels = {c.label for c in store.list_candidates()}
        # Both seeds got labels seed-0, seed-1.
        assert "seed-0" in labels
        assert "seed-1" in labels

    def test_frontier_size_matches_pareto(self, tmp_path: Path) -> None:
        from stropha.meta import scorecard_pareto_frontier

        space = self._constraint_free_space()
        prop = make_bandit_proposer(space, seed=42)
        store = ArtifactStore(tmp_path / "meta")

        # Mix of trade-off candidates so the frontier has >1 entry.
        i = {"n": 0}

        def eval_(_cfg: dict[str, Any]) -> ScoreCard:
            i["n"] += 1
            return ScoreCard(
                quality=0.5 + 0.1 * (i["n"] % 5),
                latency_ms=10 + 20 * (5 - i["n"] % 5),
            )

        result = run_loop(
            proposer=prop, evaluator=eval_, store=store,
            max_iterations=10,
        )
        # The frontier computed by the loop matches the standalone Pareto.
        archive_cards = [card for _rec, card in result.archive]
        independent_frontier = scorecard_pareto_frontier(archive_cards)
        assert len(result.frontier) == len(independent_frontier)


# ---------------------------------------------------------------------------
# CLI wiring smoke
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_meta_commands_registered(self) -> None:
        meta_app = None
        for group in app.registered_groups:
            if group.name == "meta":
                meta_app = group.typer_instance
                break
        assert meta_app is not None
        commands = {cmd.name for cmd in meta_app.registered_commands}
        # WIN-166 prune + WIN-193 run/status/report/promote/trace/snapshot.
        for name in (
            "run", "status", "report", "promote", "trace", "snapshot",
            "prune",
        ):
            assert name in commands, f"missing command: {name}"
        # WIN-164 backfill lives in a nested outcome subapp.
        outcome_app = None
        for group in meta_app.registered_groups:
            if group.name == "outcome":
                outcome_app = group.typer_instance
                break
        assert outcome_app is not None
        outcome_cmds = {cmd.name for cmd in outcome_app.registered_commands}
        assert "backfill" in outcome_cmds