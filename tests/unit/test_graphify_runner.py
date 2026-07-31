"""Tests for the graphify subprocess runner (pipeline.graphify_runner).

Covers:
  - Binary discovery via ``find_graphify``
  - First-run (creates graph.json) vs incremental (updates existing)
  - Automatic ``--no-cluster`` on incremental, full clustering on first run
  - Explicit ``no_cluster`` override
  - Graceful skip when binary is not found
  - Timeout handling
  - Non-zero exit code handling
  - OSError handling (e.g. permission denied)
  - ``run_graphify_multi`` batch behaviour
  - Pipeline integration (``_run_graphify`` called from ``Pipeline.run``)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from stropha.pipeline.graphify_runner import (
    DEFAULT_TIMEOUT_S,
    GraphifyResult,
    find_graphify,
    run_graphify,
    run_graphify_multi,
)

# ---------------------------------------------------------------------------
# find_graphify
# ---------------------------------------------------------------------------


class TestFindGraphify:
    def test_found(self) -> None:
        with patch("shutil.which", return_value="/usr/local/bin/graphify"):
            assert find_graphify() == "/usr/local/bin/graphify"

    def test_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            assert find_graphify() is None


# ---------------------------------------------------------------------------
# run_graphify — binary not found
# ---------------------------------------------------------------------------


class TestRunGraphifySkip:
    def test_skipped_when_binary_not_found(self, tmp_path: Path) -> None:
        with patch(
            "stropha.pipeline.graphify_runner.find_graphify",
            return_value=None,
        ):
            result = run_graphify(tmp_path)
        assert result.skipped is True
        assert result.created is False
        assert result.updated is False
        assert result.error is None

    def test_explicit_binary_path_skips_auto_detection(
        self, tmp_path: Path,
    ) -> None:
        """When graphify_bin is given, find_graphify is NOT called."""
        with patch(
            "stropha.pipeline.graphify_runner.find_graphify",
        ) as mock_find, patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ):
            run_graphify(tmp_path, graphify_bin="/my/graphify")
        mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# run_graphify — first run (no graph.json)
# ---------------------------------------------------------------------------


class TestRunGraphifyFirstRun:
    def test_creates_graph_json(self, tmp_path: Path) -> None:
        """First run: no graph.json exists, graphify creates it."""
        graph_json = tmp_path / "graphify-out" / "graph.json"

        def _side_effect(*args, **kwargs):
            # Simulate graphify creating the file.
            graph_json.parent.mkdir(parents=True, exist_ok=True)
            graph_json.write_text("{}")
            return subprocess.CompletedProcess(
                args=args[0], returncode=0, stdout="ok\n", stderr="",
            )

        with patch("subprocess.run", side_effect=_side_effect) as mock_run:
            result = run_graphify(tmp_path, graphify_bin="/bin/graphify")

        assert result.created is True
        assert result.updated is False
        assert result.skipped is False
        assert result.error is None
        assert result.elapsed_s >= 0

        # Should NOT pass --no-cluster on first run (we want communities).
        cmd = mock_run.call_args[0][0]
        assert "--no-cluster" not in cmd

    def test_no_output_produced(self, tmp_path: Path) -> None:
        """graphify exits 0 but doesn't create graph.json (no parseable code)."""
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ):
            result = run_graphify(tmp_path, graphify_bin="/bin/graphify")

        assert result.skipped is True
        assert result.created is False


# ---------------------------------------------------------------------------
# run_graphify — incremental (graph.json exists)
# ---------------------------------------------------------------------------


class TestRunGraphifyIncremental:
    def test_updates_existing_graph(self, tmp_path: Path) -> None:
        """Incremental: graph.json exists, graphify refreshes it."""
        graph_dir = tmp_path / "graphify-out"
        graph_dir.mkdir()
        (graph_dir / "graph.json").write_text("{}")

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok\n", stderr="",
            ),
        ) as mock_run:
            result = run_graphify(tmp_path, graphify_bin="/bin/graphify")

        assert result.updated is True
        assert result.created is False
        assert result.skipped is False

        # Should pass --no-cluster on incremental (fast path).
        cmd = mock_run.call_args[0][0]
        assert "--no-cluster" in cmd

    def test_no_cluster_false_override(self, tmp_path: Path) -> None:
        """Explicit no_cluster=False forces full clustering even when graph exists."""
        graph_dir = tmp_path / "graphify-out"
        graph_dir.mkdir()
        (graph_dir / "graph.json").write_text("{}")

        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ) as mock_run:
            run_graphify(
                tmp_path, graphify_bin="/bin/graphify", no_cluster=False,
            )

        cmd = mock_run.call_args[0][0]
        assert "--no-cluster" not in cmd

    def test_no_cluster_true_override(self, tmp_path: Path) -> None:
        """Explicit no_cluster=True skips clustering even on first run."""
        # No graph.json — would normally cluster.
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ) as mock_run:
            run_graphify(
                tmp_path, graphify_bin="/bin/graphify", no_cluster=True,
            )

        cmd = mock_run.call_args[0][0]
        assert "--no-cluster" in cmd


# ---------------------------------------------------------------------------
# run_graphify — error paths
# ---------------------------------------------------------------------------


class TestRunGraphifyErrors:
    def test_nonzero_exit(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="parse error\n",
            ),
        ):
            result = run_graphify(tmp_path, graphify_bin="/bin/graphify")

        assert result.error == "parse error"
        assert result.created is False
        assert result.updated is False

    def test_timeout(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="graphify", timeout=5),
        ):
            result = run_graphify(
                tmp_path, graphify_bin="/bin/graphify", timeout_s=5,
            )

        assert result.error is not None
        assert "timed out" in result.error

    def test_os_error(self, tmp_path: Path) -> None:
        with patch(
            "subprocess.run",
            side_effect=OSError("Permission denied"),
        ):
            result = run_graphify(tmp_path, graphify_bin="/bin/graphify")

        assert result.error == "Permission denied"


# ---------------------------------------------------------------------------
# run_graphify — command construction
# ---------------------------------------------------------------------------


class TestRunGraphifyCommand:
    def test_command_shape(self, tmp_path: Path) -> None:
        """Verify the subprocess receives the expected argv."""
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ) as mock_run:
            run_graphify(tmp_path, graphify_bin="/usr/bin/graphify")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/graphify"
        assert cmd[1] == "update"
        assert cmd[2] == str(tmp_path)

    def test_cwd_is_repo(self, tmp_path: Path) -> None:
        """subprocess.run should use repo as cwd."""
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ) as mock_run:
            run_graphify(tmp_path, graphify_bin="/bin/graphify")

        kwargs = mock_run.call_args[1]
        assert kwargs["cwd"] == str(tmp_path)

    def test_timeout_passed(self, tmp_path: Path) -> None:
        """Custom timeout is forwarded to subprocess.run."""
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="",
            ),
        ) as mock_run:
            run_graphify(
                tmp_path, graphify_bin="/bin/graphify", timeout_s=42,
            )

        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 42


# ---------------------------------------------------------------------------
# run_graphify_multi
# ---------------------------------------------------------------------------


class TestRunGraphifyMulti:
    def test_all_skipped_when_no_binary(self, tmp_path: Path) -> None:
        repos = [tmp_path / "a", tmp_path / "b"]
        with patch(
            "stropha.pipeline.graphify_runner.find_graphify",
            return_value=None,
        ):
            results = run_graphify_multi(repos)

        assert len(results) == 2
        assert all(r.skipped for r in results)

    def test_binary_resolved_once(self, tmp_path: Path) -> None:
        """find_graphify is called once, not per-repo."""
        repos = [tmp_path / "a", tmp_path / "b"]
        for r in repos:
            r.mkdir()

        with patch(
            "stropha.pipeline.graphify_runner.find_graphify",
            return_value="/bin/graphify",
        ) as mock_find, patch(
            "stropha.pipeline.graphify_runner.run_graphify",
        ) as mock_run:
            mock_run.return_value = GraphifyResult(repo=tmp_path)
            run_graphify_multi(repos)

        # find_graphify called once in run_graphify_multi.
        mock_find.assert_called_once()
        # run_graphify called for each repo with the resolved binary.
        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            assert call[1]["graphify_bin"] == "/bin/graphify"

    def test_explicit_binary_skips_find(self, tmp_path: Path) -> None:
        repos = [tmp_path]
        with patch(
            "stropha.pipeline.graphify_runner.find_graphify",
        ) as mock_find, patch(
            "stropha.pipeline.graphify_runner.run_graphify",
            return_value=GraphifyResult(repo=tmp_path),
        ):
            run_graphify_multi(repos, graphify_bin="/my/graphify")

        mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineGraphifyIntegration:
    """Verify Pipeline.run() calls _run_graphify when graphify=True."""

    def _make_pipeline(self, tmp_path: Path, *, graphify: bool):
        """Build a Pipeline with all stubs, return it with run() patchable."""
        from stropha.pipeline.pipeline import Pipeline

        storage = MagicMock()
        storage.adapter_id = "stub-storage"
        storage.count_chunks_without_repo.return_value = 0
        storage.commit = MagicMock()
        storage.set_meta = MagicMock()

        embedder = MagicMock()
        embedder.adapter_id = "stub-embed:v1"

        enricher = MagicMock()
        enricher.adapter_id = "stub-enrich:v1"

        walker = MagicMock()
        walker.adapter_name = "stub-walker"
        walker.walk.return_value = []

        chunker = MagicMock()

        pipeline = Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=enricher,
            repos=[tmp_path],
            walker=walker,
            chunker=chunker,
            graphify=graphify,
        )
        return pipeline

    def test_graphify_enabled_calls_run_graphify(
        self, tmp_path: Path,
    ) -> None:
        pipeline = self._make_pipeline(tmp_path, graphify=True)
        with patch.object(pipeline, "_run_graphify") as mock_gf, \
             patch.object(pipeline, "_refresh_graphify_mirror") as mock_refresh:
            pipeline.run()
        mock_gf.assert_called_once()
        mock_refresh.assert_called_once()

    def test_graphify_disabled_skips_run_graphify(
        self, tmp_path: Path,
    ) -> None:
        pipeline = self._make_pipeline(tmp_path, graphify=False)
        with patch.object(pipeline, "_run_graphify") as mock_gf, \
             patch.object(pipeline, "_refresh_graphify_mirror") as mock_refresh:
            pipeline.run()
        mock_gf.assert_not_called()
        mock_refresh.assert_not_called()

    def test_graphify_default_is_true(self, tmp_path: Path) -> None:
        """Default Pipeline() should have graphify enabled."""
        pipeline = self._make_pipeline(tmp_path, graphify=True)
        assert pipeline._graphify is True


class TestDefaultTimeout:
    def test_default_timeout_value(self) -> None:
        assert DEFAULT_TIMEOUT_S == 120
