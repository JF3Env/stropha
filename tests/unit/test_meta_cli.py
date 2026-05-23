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
        """JSON output is now an object with candidates + baseline +
        alerts (WIN-204). The legacy list-only schema was replaced when
        regression-alert tracking landed."""
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta), "--json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, dict)
        assert "candidates" in payload
        assert "baseline" in payload
        assert "alerts" in payload
        assert isinstance(payload["candidates"], list)
        assert len(payload["candidates"]) >= 1
        # No baseline yet → alerts is empty + baseline is null.
        assert payload["baseline"] is None
        assert payload["alerts"] == []

    def test_no_baseline_hint_in_output(self, tmp_path: Path) -> None:
        """When the operator has never promoted, the report nudges
        them toward `meta promote` to start tracking regressions."""
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0
        assert "No baseline recorded" in result.output


# ---------------------------------------------------------------------------
# stropha meta report — regression alerts (WIN-204)
# ---------------------------------------------------------------------------


class TestMetaReportBaseline:
    """Regression-alert behaviour of ``stropha meta report``.

    After a candidate is promoted via ``--apply``, a baseline record
    lives alongside the meta dir. Subsequent `meta report` runs surface
    every other candidate whose fitness beats the live baseline so the
    operator can decide whether to promote again.
    """

    def _promote(
        self, meta: Path, cwd: Path, candidate_id: str
    ) -> None:
        """Convenience: run `meta promote --apply` against a temp cwd."""
        import os

        prev = Path.cwd()
        os.chdir(cwd)
        try:
            result = CliRunner().invoke(
                app,
                ["meta", "promote", candidate_id,
                 "--meta-dir", str(meta), "--apply"],
            )
            assert result.exit_code == 0, result.output
        finally:
            os.chdir(prev)

    def test_apply_writes_baseline_record(self, tmp_path: Path) -> None:
        """`meta promote --apply` persists the baseline next to the
        meta dir so the next report can compare against it."""
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        # Pick the lowest-fitness candidate (v0) so subsequent ones
        # in the archive will read as regressions.
        recs = ArtifactStore(meta).list_candidates()
        target = next(r for r in recs if r.label == "v0")
        self._promote(meta, target_cwd, target.id)
        assert (meta / "baseline.json").is_file()

    def test_report_flags_candidates_above_baseline(
        self, tmp_path: Path
    ) -> None:
        """Promote the worst candidate → every better candidate in the
        archive should appear in the regression alerts section."""
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        recs = ArtifactStore(meta).list_candidates()
        worst = next(r for r in recs if r.label == "v0")
        self._promote(meta, target_cwd, worst.id)

        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0, result.output
        assert "Regression alerts" in result.output
        # The top candidates v1/v2/v3 all have higher fitness than v0.
        assert "v2" in result.output

    def test_report_no_regressions_when_baseline_is_best(
        self, tmp_path: Path
    ) -> None:
        """Promote the BEST candidate → no candidates beat it → the
        report shows the all-clear message. We pick the highest-fitness
        record via list_candidates(sort_by='fitness') so the assertion
        is robust against the _make_meta fixture's scoring scheme
        (latency_ms dominates the mean so the highest-latency candidate
        is paradoxically 'best')."""
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        best = ArtifactStore(meta).list_candidates(sort_by="fitness")[0]
        self._promote(meta, target_cwd, best.id)

        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta)]
        )
        assert result.exit_code == 0
        assert "No regressions" in result.output

    def test_report_json_includes_baseline_and_alerts(
        self, tmp_path: Path
    ) -> None:
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        recs = ArtifactStore(meta).list_candidates()
        worst = next(r for r in recs if r.label == "v0")
        self._promote(meta, target_cwd, worst.id)

        result = CliRunner().invoke(
            app,
            ["meta", "report", "--meta-dir", str(meta), "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["baseline"] is not None
        assert payload["baseline"]["candidate_id"] == worst.id
        assert len(payload["alerts"]) >= 1
        # Each alert carries id + delta + fitness.
        sample = payload["alerts"][0]
        assert "id" in sample and "delta" in sample
        assert sample["delta"] > 0

    def test_regression_threshold_suppresses_small_deltas(
        self, tmp_path: Path
    ) -> None:
        """A high ``--regression-threshold`` should filter out
        slight-improvement candidates so the alert section stays
        focused on meaningful gains."""
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        recs = ArtifactStore(meta).list_candidates()
        # Promote v1 (mid-fitness) so v2 is the only candidate that
        # beats it. Threshold is set above the delta v2→v1 to suppress
        # everything.
        mid = next(r for r in recs if r.label == "v1")
        self._promote(meta, target_cwd, mid.id)

        result = CliRunner().invoke(
            app,
            ["meta", "report", "--meta-dir", str(meta),
             "--regression-threshold", "1000.0"],
        )
        assert result.exit_code == 0
        assert "No regressions" in result.output

    def test_malformed_baseline_surfaces_error_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """A hand-edited baseline.json with invalid JSON must surface
        the error in red but still print the candidate table — never
        crash the operator's report run."""
        meta = _make_meta(tmp_path)
        (meta / "baseline.json").write_text(
            "{{{ not valid json", encoding="utf-8"
        )
        result = CliRunner().invoke(
            app, ["meta", "report", "--meta-dir", str(meta)]
        )
        # Exits 0 — the report is still useful.
        assert result.exit_code == 0
        assert "Baseline read failed" in result.output


# ---------------------------------------------------------------------------
# stropha meta promote
# ---------------------------------------------------------------------------


class TestMetaPromote:
    """``stropha meta promote`` is diff-only by default (WIN-202).

    Promotion preserves human gating: the harness prints a unified diff
    and exits unless ``--apply`` is passed. Auto-apply would surprise
    the operator and could destroy keys outside the search space.
    """

    def test_unknown_candidate_exits_nonzero(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", "no-such-id",
             "--meta-dir", str(meta)],
        )
        assert result.exit_code == 1

    def test_default_emits_diff_does_not_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No flag → unified diff is printed, no file is created."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 0, result.output
        # Diff header markers — unified diff always emits these for an
        # add-from-empty (the cwd has no stropha.yaml).
        assert "+++" in result.output or "---" in result.output
        # No file was created.
        assert not (target_cwd / "stropha.yaml").exists()
        # Footer reminds the operator how to commit the change.
        assert "--apply" in result.output

    def test_dry_run_still_works_as_back_compat_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` is deprecated but kept for back-compat with the
        WIN-193 CLI. It behaves like the new default (diff-only) and
        emits a deprecation banner."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id,
             "--meta-dir", str(meta), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "--dry-run is deprecated" in result.output
        # Still does not write.
        assert not (target_cwd / "stropha.yaml").exists()

    def test_apply_writes_stropha_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--apply`` is the explicit opt-in that writes stropha.yaml."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta),
             "--apply"],
        )
        assert result.exit_code == 0
        assert (target_cwd / "stropha.yaml").is_file()
        assert "Promoted" in result.output

    def test_noop_when_candidate_matches_current(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the current stropha.yaml is already identical to the
        candidate, the CLI surfaces the no-op and exits 0 — no diff,
        no rewrite."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        # Seed the cwd with the candidate's own config first.
        import yaml as _yaml

        (target_cwd / "stropha.yaml").write_text(
            _yaml.safe_dump(rec.config, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_apply_to_explicit_target_path(
        self, tmp_path: Path
    ) -> None:
        """The ``--target`` option overrides ./stropha.yaml so promotion
        can land in a sibling project or a snapshot directory."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        elsewhere = tmp_path / "elsewhere" / "stropha.yaml"
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta),
             "--target", str(elsewhere), "--apply"],
        )
        assert result.exit_code == 0
        assert elsewhere.is_file()

    def test_malformed_target_yaml_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken ``stropha.yaml`` must not be silently overwritten —
        the CLI errors out so the operator can investigate."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        # Plant invalid YAML (mismatched braces).
        (target_cwd / "stropha.yaml").write_text(
            "this is: { not valid: ]]]", encoding="utf-8"
        )
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 1
        # Normalize whitespace — Rich console wraps long paths across
        # multiple lines, but the phrase itself stays intact.
        normalized = " ".join(result.output.split())
        assert "not valid YAML" in normalized


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