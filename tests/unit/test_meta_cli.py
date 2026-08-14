"""Unit tests for the ``stropha meta`` CLI commands (WIN-193) +
end-to-end loop integration (WIN-194)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from stropha.cli import app
from stropha.config import (
    Config,
    ExperimentBlacklistConfig,
    ExperimentPostgresConfig,
    ExperimentsConfig,
    ExperimentStorage,
    get_config,
)
from stropha.eval import EvalCase, EvalReport, EvalResult
from stropha.meta import (
    ArtifactStore,
    BaselineValidationArtifact,
    CandidateEvaluationResult,
    EvaluationEvidence,
    EvaluationMetrics,
    EvaluationValidationArtifact,
    RunValidationArtifact,
    ScoreCard,
    ValidationGate,
    run_loop,
)
from stropha.meta.proposer import make_bandit_proposer
from stropha.meta.stream_weights import (
    MIN_ACTIVATION_EXAMPLES,
    StreamWeightBenchmark,
    StreamWeightProfile,
    StreamWeightProfiles,
    write_stream_weight_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(tmp_path: Path) -> Path:
    """Build a meta dir with 4 candidates of varying fitness."""
    meta = tmp_path / "meta"
    store = ArtifactStore(meta)
    for i, q in enumerate([0.3, 0.6, 0.9, 0.5]):
        rec = store.register_candidate(
            {"embedder": {"adapter": "local", "config": {"model": f"model-{i}"}}},
            label=f"v{i}",
        )
        store.write_scores(
            rec.id,
            {
                "quality": q,
                "context_efficiency": 0.5,
                "outcome_score": 0.2,
                "latency_ms": 20.0 * (i + 1),
            },
        )
    return meta


def _runner_invoke(tmp_path: Path, args: list[str]) -> Any:
    """Run the CLI with a fake config so _meta_dir_default works."""
    return CliRunner().invoke(app, args)


def _meta_runtime_config(
    *,
    storage: ExperimentStorage = "sqlite-vec",
    blacklist: dict[str, tuple[str, ...]] | None = None,
    admin_dsn: str | None = None,
) -> Config:
    base = get_config()
    return dataclasses.replace(
        base,
        paths=dataclasses.replace(
            base.paths,
            target_repo=Path(__file__).resolve().parents[2],
        ),
        experiments=ExperimentsConfig(
            storage=storage,
            postgres=ExperimentPostgresConfig(admin_dsn=admin_dsn),
            blacklist=ExperimentBlacklistConfig(adapters=blacklist or {}),
        ),
    )


def _stream_weight_profile(*, examples: int) -> StreamWeightProfile:
    return StreamWeightProfile(
        repo="github.com/acme/app",
        weights={"dense": 1.1, "sparse": 0.9},
        observations={"dense": examples, "sparse": examples},
        mean_scores={"dense": 0.1, "sparse": -0.1},
        generated_at="2026-08-13T00:00:00+00:00",
        examples=examples,
        source_counts={"outcome": examples},
        since_sha="a",
        head_sha="b",
    )


def _install_eligible_run(meta: Path, candidate: Any) -> str:
    import stropha.cli as cli

    run_id = "eligible-run"
    repo = Path(__file__).resolve().parents[2]
    _space, _registry, manifest_hash, registry_hash = cli._meta_preflight()
    golden_hash = "golden"
    snapshot_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _graphify, graph_identity = cli._meta_graph_identity()
    identity = cli._meta_runtime_identity(
        target_repo=repo,
        snapshot_sha=snapshot_sha,
        base_config=cli.load_pipeline_config(),
        golden_sha256=golden_hash,
        manifest_sha256=manifest_hash,
        registry_sha256=registry_hash,
        graph_identity=graph_identity,
    )
    metrics = EvaluationMetrics(
        success_at_k=1.0,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        latency_p95_ms=1.0,
        returned_tokens=1,
    )
    evidence = EvaluationEvidence(
        search=metrics,
        holdout=metrics,
        search_identity={"split": "search"},
        holdout_identity={"split": "holdout"},
        runtime_identity=identity,
    )
    validation = EvaluationValidationArtifact(
        baseline_id="live",
        candidate_id=candidate.id,
        baseline=evidence,
        candidate=evidence,
        verification_checks={"pipeline_health": True, "candidate_gate": True},
        validated_identity=identity,
        gate=ValidationGate(eligible=True),
    )
    store = ArtifactStore(meta)
    store.write_run_artifact(
        run_id,
        "provenance.json",
        {
            "run_id": run_id,
            "golden_sha256": golden_hash,
            "manifest_sha256": manifest_hash,
            "registry_sha256": registry_hash,
            "repository": {"path": str(repo)},
            "snapshot": {"sha": snapshot_sha, "path": str(repo)},
        },
    )
    store.write_run_artifact(
        run_id,
        "shortlist.json",
        {
            "run_id": run_id,
            "entries": [
                {
                    "candidate_id": candidate.id,
                    "config_sha256": cli._meta_candidate_hash(candidate),
                }
            ],
        },
    )
    store.write_run_artifact(
        run_id,
        "validation.json",
        RunValidationArtifact(
            run_id=run_id,
            baseline=BaselineValidationArtifact(
                baseline_id="live",
                evidence=evidence,
                validated_identity=identity,
            ),
            candidates={candidate.id: validation},
        ),
    )
    return run_id


# ---------------------------------------------------------------------------
# stropha meta run (WIN-193)
# ---------------------------------------------------------------------------


class TestMetaRun:
    def test_runs_bandit_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_live_eval_stubs(monkeypatch)
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "3",
                "--proposer",
                "bandit",
                "--seed",
                "42",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert result.exit_code == 0, result.output
        # The output mentions the loop result.
        assert "iterations" in result.output or "frontier_size" in result.output
        run_dirs = list((tmp_path / "meta" / "runs").iterdir())
        assert len(run_dirs) == 1
        provenance = json.loads((run_dirs[0] / "provenance.json").read_text())
        report = json.loads((run_dirs[0] / "report.json").read_text())
        shortlist = json.loads((run_dirs[0] / "shortlist.json").read_text())
        cost = json.loads((run_dirs[0] / "cost.json").read_text())
        assert provenance["proposer"] == "bandit"
        assert provenance["seed"] == 42
        assert provenance["golden_sha256"]
        assert provenance["manifest_sha256"]
        assert provenance["registry_sha256"]
        assert report["evaluation_scope"] == "search_only"
        assert "holdout" not in json.dumps(report)
        assert shortlist["baseline_id"]
        assert shortlist["entries"]
        assert all(entry["search_metrics"] for entry in shortlist["entries"])
        assert cost["by_scope"]["search"]["retrieval_queries"] > 0
        assert cost["proposer_tokens"]["exact"] == 0
        candidate_id = shortlist["entries"][0]["candidate_id"]
        assert list((tmp_path / "meta" / "candidates" / candidate_id / "traces").glob("*.jsonl"))

    def test_options_and_resume_restore_checkpoint_without_rescoring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluations = self._install_live_eval_stubs(monkeypatch)

        first = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "2",
                "--seed",
                "42",
                "--candidates",
                "2",
                "--parallel",
                "2",
                "--evaluation-timeout",
                "30",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert first.exit_code == 0, first.output
        run_dir = next((tmp_path / "meta" / "runs").iterdir())
        run_id = run_dir.name
        first_ids = set(evaluations)

        resumed = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "2",
                "--seed",
                "42",
                "--candidates",
                "2",
                "--parallel",
                "2",
                "--evaluation-timeout",
                "30",
                "--resume",
                run_id,
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )

        assert resumed.exit_code == 0, resumed.output
        assert first_ids <= set(evaluations)
        assert all(evaluations[candidate_id] == 1 for candidate_id in first_ids)
        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert provenance["resume"]["resumed"] is True
        assert provenance["resume"]["rng_state_restored"] is True

    def test_resume_rejects_incompatible_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_live_eval_stubs(monkeypatch)
        meta = tmp_path / "meta"
        first = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "2", "--seed", "42", "--meta-dir", str(meta)],
        )
        assert first.exit_code == 0, first.output
        run_id = next((meta / "runs").iterdir()).name

        resumed = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "3",
                "--seed",
                "42",
                "--resume",
                run_id,
                "--meta-dir",
                str(meta),
            ],
        )

        assert resumed.exit_code == 1
        assert "Resume options differ" in resumed.output

    def test_policy_restricts_proposer_provider_and_baseline_before_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_live_eval_stubs(monkeypatch)
        monkeypatch.setattr(
            "stropha.cli._load_config",
            lambda: _meta_runtime_config(blacklist={"embedder": ("voyage",)}),
        )
        observed: dict[str, Any] = {}
        from stropha.meta.proposer import make_bandit_proposer as real_make_bandit

        def make_bandit(space: Any, *, seed: int | None = None):
            observed["proposer_space"] = space
            return real_make_bandit(space, seed=seed)

        monkeypatch.setattr("stropha.meta.proposer.make_bandit_proposer", make_bandit)
        import stropha.meta.candidate_index as candidate_index

        base_provider = candidate_index.CandidateIndexProvider

        class Provider(base_provider):
            def __init__(self, **kwargs: Any) -> None:
                observed["provider_space"] = kwargs["search_space"]
                super().__init__(**kwargs)

        monkeypatch.setattr(candidate_index, "CandidateIndexProvider", Provider)
        meta = tmp_path / "meta"

        result = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "1", "--seed", "42", "--meta-dir", str(meta)],
        )

        assert result.exit_code == 0, result.output
        proposer_space = observed["proposer_space"]
        assert proposer_space is observed["provider_space"]
        assert "voyage" not in proposer_space.by_name["embedder"].choices
        assert "git-diff" not in proposer_space.by_name["walker"].choices
        assert proposer_space.by_name["storage"].choices == ("sqlite-vec",)
        for record in ArtifactStore(meta).list_candidates():
            assert record.config["embedder"] != "voyage"
            assert record.config["walker"] != "git-diff"

    def test_blacklisted_active_baseline_fails_before_proposer_or_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_live_eval_stubs(monkeypatch)
        monkeypatch.setattr(
            "stropha.cli._load_config",
            lambda: _meta_runtime_config(blacklist={"embedder": ("local",)}),
        )
        monkeypatch.setattr(
            "stropha.cli.load_pipeline_config",
            lambda: {"embedder": {"adapter": "local", "config": {}}},
        )
        proposer_called = False

        def unexpected_proposer(*_args: object, **_kwargs: object) -> None:
            nonlocal proposer_called
            proposer_called = True

        monkeypatch.setattr("stropha.meta.proposer.make_bandit_proposer", unexpected_proposer)
        meta = tmp_path / "meta"

        result = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "1", "--meta-dir", str(meta)],
        )

        assert result.exit_code == 1
        assert "active baseline stage embedder='local' is excluded" in result.output
        assert proposer_called is False
        assert not (meta / "candidates").exists()

    def test_postgres_backend_is_preflighted_frozen_and_reused_for_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluations = self._install_live_eval_stubs(monkeypatch)
        admin_dsn = "postgresql://eval_user:super-secret@db.internal:5544/stropha_meta_admin"
        monkeypatch.setattr(
            "stropha.cli._load_config",
            lambda: _meta_runtime_config(storage="postgres", admin_dsn=admin_dsn),
        )
        monkeypatch.setattr(
            "stropha.cli.load_pipeline_config",
            lambda: {
                "storage": {
                    "adapter": "postgres",
                    "config": {
                        "dsn": (
                            "postgresql://prod:production-secret@db.internal:5544/"
                            "stropha_production"
                        )
                    },
                }
            },
        )
        events: list[str] = []
        manager_kwargs: list[dict[str, Any]] = []

        class Manager:
            cluster = SimpleNamespace(
                host="db.internal",
                port=5544,
                database="stropha_meta_admin",
                user="eval_user",
                server_version="17.2",
                template_database="stropha_meta_template",
                template_extensions=("plpgsql=1.0", "vector=0.8.1"),
            )

            def __init__(self, **kwargs: Any) -> None:
                manager_kwargs.append(kwargs)

            def preflight(self) -> Any:
                events.append("preflight")
                return self.cluster

        monkeypatch.setattr(
            "stropha.evaluation.postgres.PostgresEvaluationDatabaseManager",
            Manager,
        )
        import stropha.meta.candidate_index as candidate_index

        provider_kwargs: list[dict[str, Any]] = []
        base_provider = candidate_index.CandidateIndexProvider

        class Provider(base_provider):
            def __init__(self, **kwargs: Any) -> None:
                events.append("provider")
                provider_kwargs.append(kwargs)
                super().__init__(**kwargs)

        monkeypatch.setattr(candidate_index, "CandidateIndexProvider", Provider)
        from stropha.meta.proposer import make_bandit_proposer as real_make_bandit

        def make_bandit(space: Any, *, seed: int | None = None):
            events.append("proposer")
            return real_make_bandit(space, seed=seed)

        monkeypatch.setattr("stropha.meta.proposer.make_bandit_proposer", make_bandit)
        meta = tmp_path / "meta"

        run = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "3", "--seed", "42", "--meta-dir", str(meta)],
        )

        assert run.exit_code == 0, run.output
        assert events.index("preflight") < events.index("proposer")
        assert manager_kwargs[0]["production_database"] == "stropha_production"
        assert provider_kwargs[0]["postgres_manager"].__class__ is Manager
        assert "postgres" in provider_kwargs[0]["cache_format"]
        run_dir = next((meta / "runs").iterdir())
        provenance = json.loads((run_dir / "provenance.json").read_text())
        encoded = json.dumps(provenance, sort_keys=True)
        assert provenance["experiment"] == provenance["identity"]["experiment"]
        assert provenance["experiment"]["cluster_identity"]["server_version"] == "17.2"
        assert provenance["experiment"]["storage"] == "postgres"
        assert "super-secret" not in encoded
        assert "production-secret" not in encoded
        assert "postgresql://" not in encoded

        validation = CliRunner().invoke(
            app,
            ["meta", "validate", run_dir.name, "--meta-dir", str(meta)],
        )

        assert validation.exit_code == 0, validation.output
        assert provider_kwargs[-1]["postgres_manager"].__class__ is Manager
        assert provider_kwargs[-1]["cache_format"] == provenance["experiment"]["cache_format"]
        assert len(evaluations) > 1

        Manager.cluster = SimpleNamespace(
            **{**vars(Manager.cluster), "host": "replacement.internal"}
        )
        evaluations_before_resume = dict(evaluations)
        resumed = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "3",
                "--seed",
                "42",
                "--resume",
                run_dir.name,
                "--meta-dir",
                str(meta),
            ],
        )

        assert resumed.exit_code == 1
        assert "experiment cluster differs" in resumed.output
        assert evaluations == evaluations_before_resume

    def test_resume_rejects_policy_drift_before_candidate_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        evaluations = self._install_live_eval_stubs(monkeypatch)
        monkeypatch.setattr("stropha.cli._load_config", lambda: _meta_runtime_config())
        meta = tmp_path / "meta"
        run = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "1", "--seed", "42", "--meta-dir", str(meta)],
        )
        assert run.exit_code == 0, run.output
        run_id = next((meta / "runs").iterdir()).name
        monkeypatch.setattr(
            "stropha.cli._load_config",
            lambda: _meta_runtime_config(blacklist={"embedder": ("voyage",)}),
        )
        evaluations_before_resume = dict(evaluations)

        resumed = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "1",
                "--seed",
                "42",
                "--resume",
                run_id,
                "--meta-dir",
                str(meta),
            ],
        )

        assert resumed.exit_code == 1
        assert "experiment policy differs" in resumed.output
        assert evaluations == evaluations_before_resume

    def test_validate_uses_frozen_shortlist_and_adds_holdout_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_live_eval_stubs(monkeypatch)
        meta = tmp_path / "meta"
        run = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "3",
                "--seed",
                "42",
                "--meta-dir",
                str(meta),
            ],
        )
        assert run.exit_code == 0, run.output
        run_id = next((meta / "runs").iterdir()).name

        validation = CliRunner().invoke(
            app,
            ["meta", "validate", run_id, "--meta-dir", str(meta)],
        )

        assert validation.exit_code == 0, validation.output
        run_dir = meta / "runs" / run_id
        artifact = json.loads((run_dir / "validation.json").read_text())
        cost = json.loads((run_dir / "cost.json").read_text())
        assert artifact["run_id"] == run_id
        assert artifact["baseline"]["evidence"]["holdout"] is not None
        assert artifact["candidates"]
        assert all(
            item["candidate"]["holdout"] is not None for item in artifact["candidates"].values()
        )
        assert "holdout_validation" in cost["by_scope"]
        assert cost["candidate_states"]

    def test_validate_rejects_top_k_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_live_eval_stubs(monkeypatch)
        meta = tmp_path / "meta"
        run = CliRunner().invoke(
            app,
            ["meta", "run", "--iterations", "1", "--seed", "42", "--meta-dir", str(meta)],
        )
        assert run.exit_code == 0, run.output
        run_id = next((meta / "runs").iterdir()).name

        validation = CliRunner().invoke(
            app,
            ["meta", "validate", run_id, "--top-k", "20", "--meta-dir", str(meta)],
        )

        assert validation.exit_code == 1
        assert "must match frozen" in validation.output

    def test_unknown_proposer_exits_nonzero(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "1",
                "--proposer",
                "unknown",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert result.exit_code == 1
        assert "Unknown proposer" in result.output

    @pytest.mark.parametrize(
        "args, message",
        [
            (["--candidates", "0"], "must be at least 1"),
            (["--parallel", "0"], "must be at least 1"),
            (["--evaluation-timeout", "0"], "must be greater than 0"),
        ],
    )
    def test_invalid_run_options_fail_before_indexing(
        self,
        args: list[str],
        message: str,
        tmp_path: Path,
    ) -> None:
        result = CliRunner().invoke(
            app,
            ["meta", "run", *args, "--meta-dir", str(tmp_path / "meta")],
        )
        assert result.exit_code == 1
        assert message in result.output

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
        self._install_live_eval_stubs(monkeypatch)
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "2",
                "--proposer",
                "local-agentic",
                "--seed",
                "42",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cloud_agentic_proposer_can_be_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "stropha.meta.proposer.agentic.get_backend",
            lambda **_: (_ for _ in ()).throw(RuntimeError("no key")),
        )
        self._install_live_eval_stubs(monkeypatch)
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "2",
                "--proposer",
                "cloud-agentic",
                "--seed",
                "42",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("mode", ["local-code", "cloud-code"])
    def test_code_proposer_modes_can_be_selected(
        self,
        mode: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend_path = (
            "stropha.meta.proposer.inference_helper.get_backend"
            if mode == "local-code"
            else "stropha.meta.proposer.agentic.get_backend"
        )
        monkeypatch.setattr(
            backend_path,
            lambda **_: (_ for _ in ()).throw(RuntimeError("no backend")),
        )
        self._install_live_eval_stubs(monkeypatch)
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "run",
                "--iterations",
                "2",
                "--proposer",
                mode,
                "--seed",
                "42",
                "--meta-dir",
                str(tmp_path / "meta"),
            ],
        )
        assert result.exit_code == 0, result.output

    @staticmethod
    def _install_live_eval_stubs(
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, int]:
        import dataclasses

        from stropha.config import get_config

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        class _Retrieval:
            def health(self):
                return type("Health", (), {"status": "ready", "message": "ok"})()

            def set_reranker(self, _reranker: object) -> None:
                pass

        base = get_config()
        config = dataclasses.replace(
            base,
            paths=dataclasses.replace(base.paths, target_repo=Path(__file__).resolve().parents[2]),
        )
        monkeypatch.setattr("stropha.cli._load_config", lambda: config)
        monkeypatch.setattr("stropha.cli.load_pipeline_config", lambda: {})
        evaluations: dict[str, int] = {}

        class _Provider:
            def __init__(self, **kwargs: object) -> None:
                self.cache = kwargs["cache"]

            def pin_snapshot(self, *, source_repo: Path, **_kwargs: object):
                from stropha.meta import Snapshot

                return Snapshot(head, source_repo, source_repo)

            @contextlib.contextmanager
            def open_candidate(self, *, candidate_id: str, **_kwargs: object):
                evaluations[candidate_id] = evaluations.get(candidate_id, 0) + 1
                yield SimpleNamespace(
                    built=SimpleNamespace(retrieval=_Retrieval()),
                    repo_id=1,
                    cache_hit=len(evaluations) > 1,
                )

        monkeypatch.setattr(
            "stropha.meta.candidate_index.CandidateIndexProvider",
            _Provider,
        )

        def evaluate(**kwargs: object) -> CandidateEvaluationResult:
            split = str(kwargs.get("split", "search"))
            case = EvalCase(
                id=f"{split}-case",
                query=split,
                expected_paths=("src/a.py",),
                split=split,
            )
            report = EvalReport(
                top_k=10,
                results=[
                    EvalResult(
                        case=case,
                        rank_of_first_hit=1,
                        top_k=10,
                        relevances=(1.0,),
                        relevant_count=1,
                        total_relevant=1,
                        latency_ms=10.0,
                        returned_tokens=20,
                    )
                ],
            )
            context = kwargs.get("evaluation_context")
            if context is not None:
                recorder = context.open_recorder()
                recorder.record(
                    {
                        "tool": "search_code",
                        "candidate_id": context.candidate_id,
                        "run_id": context.run_id,
                        "query": split,
                        "results": [],
                        "latency_ms": 10.0,
                        "returned_tokens": 20,
                    }
                )
                recorder.close()
            return CandidateEvaluationResult(
                scorecard=ScoreCard(
                    quality=0.7,
                    context_efficiency=0.8,
                    latency_ms=10.0,
                    n_queries=1,
                    n_hits=1,
                ),
                report=report,
            )

        monkeypatch.setattr(
            "stropha.meta.evaluate_candidate",
            evaluate,
        )
        return evaluations


# ---------------------------------------------------------------------------
# stropha meta status
# ---------------------------------------------------------------------------


class TestMetaStatus:
    def test_no_candidates_returns_yellow(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["meta", "status", "--meta-dir", str(tmp_path / "empty")])
        assert result.exit_code == 0
        assert "No candidates" in result.output

    def test_reports_archive_and_frontier(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(app, ["meta", "status", "--meta-dir", str(meta)])
        assert result.exit_code == 0
        assert "archive" in result.output
        assert "frontier" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(app, ["meta", "status", "--meta-dir", str(meta), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["archive_size"] == 4


# ---------------------------------------------------------------------------
# stropha meta report
# ---------------------------------------------------------------------------


class TestMetaReport:
    def test_no_candidates(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(tmp_path / "x")])
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
        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(meta), "--json"])
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
        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(meta)])
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

    def _promote(self, meta: Path, cwd: Path, candidate_id: str) -> None:
        """Convenience: promote to an explicit temporary config."""
        import os

        prev = Path.cwd()
        os.chdir(cwd)
        try:
            candidate = ArtifactStore(meta).read_candidate(candidate_id)
            run_id = _install_eligible_run(meta, candidate)
            result = CliRunner().invoke(
                app,
                [
                    "meta",
                    "promote",
                    candidate_id,
                    "--meta-dir",
                    str(meta),
                    "--target",
                    str(cwd / "config.yaml"),
                    "--apply",
                    "--run-id",
                    run_id,
                ],
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

    def test_report_flags_candidates_above_baseline(self, tmp_path: Path) -> None:
        """Promote the worst candidate → every better candidate in the
        archive should appear in the regression alerts section."""
        meta = _make_meta(tmp_path)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        recs = ArtifactStore(meta).list_candidates()
        worst = next(r for r in recs if r.label == "v0")
        self._promote(meta, target_cwd, worst.id)

        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(meta)])
        assert result.exit_code == 0, result.output
        assert "Regression alerts" in result.output
        # The top candidates v1/v2/v3 all have higher fitness than v0.
        assert "v2" in result.output

    def test_report_no_regressions_when_baseline_is_best(self, tmp_path: Path) -> None:
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

        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(meta)])
        assert result.exit_code == 0
        assert "No regressions" in result.output

    def test_report_json_includes_baseline_and_alerts(self, tmp_path: Path) -> None:
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

    def test_regression_threshold_suppresses_small_deltas(self, tmp_path: Path) -> None:
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
            ["meta", "report", "--meta-dir", str(meta), "--regression-threshold", "1000.0"],
        )
        assert result.exit_code == 0
        assert "No regressions" in result.output

    def test_malformed_baseline_surfaces_error_does_not_crash(self, tmp_path: Path) -> None:
        """A hand-edited baseline.json with invalid JSON must surface
        the error in red but still print the candidate table — never
        crash the operator's report run."""
        meta = _make_meta(tmp_path)
        (meta / "baseline.json").write_text("{{{ not valid json", encoding="utf-8")
        result = CliRunner().invoke(app, ["meta", "report", "--meta-dir", str(meta)])
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
            ["meta", "promote", "no-such-id", "--meta-dir", str(meta)],
        )
        assert result.exit_code == 1

    def test_default_emits_diff_does_not_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No flag diffs the canonical config path without writing it."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / ".stropha" / "config.yaml"
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 0, result.output
        # Diff header markers — unified diff always emits these for an
        # add-from-empty (the canonical config does not exist).
        assert "+++" in result.output or "---" in result.output
        # Neither the retired cwd target nor the canonical target was created.
        assert not (target_cwd / "stropha.yaml").exists()
        assert not target.exists()
        assert str(target) in result.output.replace("\n", "")
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
        monkeypatch.setenv("HOME", str(tmp_path))
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "--dry-run is deprecated" in result.output
        # Still does not write.
        assert not (tmp_path / ".stropha" / "config.yaml").exists()

    def test_apply_writes_canonical_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--apply`` writes the canonical unified config by default."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        run_id = _install_eligible_run(meta, rec)
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta), "--apply", "--run-id", run_id],
        )
        assert result.exit_code == 0
        assert (tmp_path / ".stropha" / "config.yaml").is_file()
        assert not (target_cwd / "stropha.yaml").exists()
        assert "Promoted" in result.output

    def test_noop_when_candidate_matches_current(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the merged config is already current, the CLI exits 0 — no diff,
        no rewrite."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / ".stropha" / "config.yaml"
        target.parent.mkdir()
        # Seed the canonical target with the normalized candidate config.
        import yaml as _yaml

        target.write_text(
            _yaml.safe_dump({"pipeline": rec.config}, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_validated_noop_apply_records_verified_baseline(self, tmp_path: Path) -> None:
        import yaml as _yaml

        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        run_id = _install_eligible_run(meta, rec)
        target = tmp_path / "config.yaml"
        target.write_text(
            _yaml.safe_dump({"pipeline": rec.config}, sort_keys=True),
            encoding="utf-8",
        )

        result = CliRunner().invoke(
            app,
            [
                "meta",
                "promote",
                rec.id,
                "--meta-dir",
                str(meta),
                "--target",
                str(target),
                "--apply",
                "--run-id",
                run_id,
            ],
        )

        assert result.exit_code == 0, result.output
        baseline = json.loads((meta / "baseline.json").read_text())
        assert baseline["verification_status"] == "verified"

    def test_apply_to_explicit_target_path(self, tmp_path: Path) -> None:
        """``--target`` uses the same merge semantics as the default path."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        run_id = _install_eligible_run(meta, rec)
        elsewhere = tmp_path / "elsewhere" / "config.yaml"
        elsewhere.parent.mkdir()
        elsewhere.write_text(
            "secrets:\n  voyage_api_key: keep-me\nlogging:\n  level: DEBUG\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            [
                "meta",
                "promote",
                rec.id,
                "--meta-dir",
                str(meta),
                "--target",
                str(elsewhere),
                "--apply",
                "--run-id",
                run_id,
            ],
        )
        assert result.exit_code == 0
        assert elsewhere.is_file()
        import yaml as _yaml

        written = _yaml.safe_load(elsewhere.read_text(encoding="utf-8"))
        assert written["secrets"] == {"voyage_api_key": "keep-me"}
        assert written["logging"] == {"level": "DEBUG"}
        assert written["pipeline"] == rec.config

    def test_apply_without_validated_run_is_blocked(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target = tmp_path / "config.yaml"

        result = CliRunner().invoke(
            app,
            [
                "meta",
                "promote",
                rec.id,
                "--meta-dir",
                str(meta),
                "--target",
                str(target),
                "--apply",
            ],
        )

        assert result.exit_code == 1
        assert "requires --run-id" in result.output
        assert not target.exists()

    def test_malformed_target_yaml_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken canonical config must not be silently overwritten —
        the CLI errors out so the operator can investigate."""
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        target_cwd = tmp_path / "cwd"
        target_cwd.mkdir()
        monkeypatch.chdir(target_cwd)
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / ".stropha" / "config.yaml"
        target.parent.mkdir()
        # Plant invalid YAML (mismatched braces).
        target.write_text("this is: { not valid: ]]]", encoding="utf-8")
        result = CliRunner().invoke(
            app,
            ["meta", "promote", rec.id, "--meta-dir", str(meta)],
        )
        assert result.exit_code == 1
        # Normalize whitespace — Rich console wraps long paths across
        # multiple lines, but the phrase itself stays intact.
        normalized = " ".join(result.output.split())
        assert "not valid YAML" in normalized

    def test_help_describes_canonical_merged_target(self) -> None:
        result = CliRunner().invoke(app, ["meta", "promote", "--help"])
        assert result.exit_code == 0
        normalized = " ".join(result.output.split())
        assert "~/.stropha/config.yaml" in normalized
        assert "merged into explicit targets" in normalized


# ---------------------------------------------------------------------------
# stropha meta trace
# ---------------------------------------------------------------------------


class TestMetaTrace:
    def test_unknown_candidate_exits_nonzero(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        result = CliRunner().invoke(app, ["meta", "trace", "nope", "--meta-dir", str(meta)])
        assert result.exit_code == 1

    def test_lists_traces(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        # Manually drop a trace file.
        assert rec.base_dir is not None
        (rec.base_dir / "traces" / "t.jsonl").write_text('{"x": 1}\n')
        result = CliRunner().invoke(app, ["meta", "trace", rec.id, "--meta-dir", str(meta)])
        assert result.exit_code == 0
        assert "t.jsonl" in result.output

    def test_no_traces_dir(self, tmp_path: Path) -> None:
        meta = _make_meta(tmp_path)
        rec = ArtifactStore(meta).list_candidates()[0]
        assert rec.base_dir is not None
        import shutil

        shutil.rmtree(rec.base_dir / "traces")
        result = CliRunner().invoke(app, ["meta", "trace", rec.id, "--meta-dir", str(meta)])
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
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
            cwd=repo,
            check=True,
        )

    def test_create_snapshot(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        result = CliRunner().invoke(app, ["meta", "snapshot", "--repo", str(repo)])
        assert result.exit_code == 0, result.output
        assert "Snapshot:" in result.output

    def test_list_snapshots(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        self._init_repo(repo)
        # First create one.
        CliRunner().invoke(app, ["meta", "snapshot", "--repo", str(repo)])
        # Then list.
        result = CliRunner().invoke(app, ["meta", "snapshot", "--repo", str(repo), "--list"])
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
                Dimension(
                    name="walker",
                    kind="stage",
                    indexing=True,
                    type="choice",
                    choices=("git-ls-files", "filesystem"),
                ),
                Dimension(
                    name="enricher",
                    kind="stage",
                    indexing=True,
                    type="choice",
                    choices=("noop", "hierarchical"),
                ),
                Dimension(
                    name="embedder", kind="stage", indexing=True, type="choice", choices=("local",)
                ),
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
            proposer=prop,
            evaluator=eval_,
            store=store,
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
            proposer=prop,
            evaluator=eval_,
            store=store,
            seeds=seeds,
            max_iterations=4,
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
            proposer=prop,
            evaluator=eval_,
            store=store,
            max_iterations=10,
        )
        # The frontier computed by the loop matches the standalone Pareto.
        archive_cards = [card for _rec, card in result.archive]
        independent_frontier = scorecard_pareto_frontier(archive_cards)
        assert len(result.frontier) == len(independent_frontier)


# ---------------------------------------------------------------------------
# stropha meta stream-weights
# ---------------------------------------------------------------------------


class TestStreamWeightsCli:
    def test_report_json_explains_blocked_candidate(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta"
        write_stream_weight_profile(
            meta / "stream-weight-candidates",
            _stream_weight_profile(examples=1),
        )

        result = CliRunner().invoke(
            app,
            ["meta", "stream-weights", "report", "--meta-dir", str(meta), "--json"],
        )

        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert rows[0]["status"] == "blocked"
        assert rows[0]["examples"] == 1
        assert "requires 50 examples" in rows[0]["blocked_reasons"][0]

    def test_benchmark_rejects_low_sample_candidate(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta"
        profile = _stream_weight_profile(examples=1)
        write_stream_weight_profile(meta / "stream-weight-candidates", profile)

        result = CliRunner().invoke(
            app,
            [
                "meta",
                "stream-weights",
                "benchmark",
                profile.repo,
                "--meta-dir",
                str(meta),
            ],
        )

        assert result.exit_code == 1
        assert "Insufficient evidence" in result.output
        assert not (meta / "stream-weights").exists()

    def test_benchmark_activates_candidate_from_computed_evidence(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        meta = tmp_path / "meta"
        repo = tmp_path / "repo"
        repo.mkdir()
        profile = _stream_weight_profile(examples=MIN_ACTIVATION_EXAMPLES)
        write_stream_weight_profile(meta / "stream-weight-candidates", profile)
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps(
                {
                    "id": "s",
                    "query": "search",
                    "expected_paths": ["src/a.py"],
                    "split": "search",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "h",
                    "query": "holdout",
                    "expected_paths": ["src/b.py"],
                    "split": "holdout",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class _Storage:
            def __enter__(self) -> _Storage:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def lookup_repo_id(self, _repo_key: str) -> int:
                return 1

            def get_meta(self, key: str) -> str:
                return {
                    "active_enricher_id": "enricher",
                    "active_embedder_id": "embedder",
                }[key]

            def list_repos(self) -> list[Any]:
                return [
                    type(
                        "Repo",
                        (),
                        {
                            "normalized_key": profile.repo,
                            "head_commit": "head",
                        },
                    )()
                ]

        class _Built:
            def __init__(self, adapter_id: str) -> None:
                self.storage = _Storage()
                self.retrieval = type("Retrieval", (), {"adapter_id": adapter_id})()
                self.embedder = type("Embedder", (), {"adapter_id": "embedder"})()

            def effective_enricher_id(self) -> str:
                return "enricher"

        configs: list[dict[str, Any]] = []

        def build(config: dict[str, Any]) -> _Built:
            configs.append(config)
            enabled = config["retrieval"]["config"]["learned_weights_enabled"]
            return _Built("candidate" if enabled else "baseline")

        monkeypatch.setattr(
            "stropha.cli.load_pipeline_config",
            lambda: {
                "retrieval": {"config": {"learned_weights_enabled": True}},
            },
        )
        monkeypatch.setattr("stropha.cli.build_stages", build)
        monkeypatch.setattr(
            "stropha.ingest.git_meta.detect",
            lambda _path: type("Repo", (), {"normalized_key": profile.repo})(),
        )
        monkeypatch.setattr(
            "stropha.meta.stream_weights.benchmark_stream_weight_profile",
            lambda **_kwargs: StreamWeightBenchmark(
                baseline_search_ndcg=0.7,
                candidate_search_ndcg=0.72,
                baseline_holdout_ndcg=0.8,
                candidate_holdout_ndcg=0.8,
                search_cases=1,
                holdout_cases=1,
            ),
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *_args, **_kwargs: type("Completed", (), {"stdout": "head\n"})(),
        )

        result = CliRunner().invoke(
            app,
            [
                "meta",
                "stream-weights",
                "benchmark",
                profile.repo,
                "--golden",
                str(golden),
                "--repo",
                str(repo),
                "--meta-dir",
                str(meta),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        assert configs[0]["retrieval"]["config"]["learned_weights_enabled"] is False
        assert configs[0]["retrieval"]["config"]["reranker"] is None
        assert configs[1]["retrieval"]["config"]["learned_weights_enabled"] is True
        assert configs[1]["retrieval"]["config"]["reranker"] is None
        active = StreamWeightProfiles(meta / "stream-weights").report(active=True)[0]
        assert active["status"] == "active"
        assert active["validation"]["golden_sha256"]
        assert active["validation"]["baseline_retrieval_id"] == "baseline"
        assert active["validation"]["candidate_retrieval_id"] == "candidate"
        assert active["validation"]["indexed_commit"] == "head"


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
            "run",
            "validate",
            "status",
            "report",
            "promote",
            "trace",
            "snapshot",
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
        stream_weights_app = None
        for group in meta_app.registered_groups:
            if group.name == "stream-weights":
                stream_weights_app = group.typer_instance
                break
        assert stream_weights_app is not None
        stream_weight_cmds = {cmd.name for cmd in stream_weights_app.registered_commands}
        assert stream_weight_cmds == {"benchmark", "report"}
