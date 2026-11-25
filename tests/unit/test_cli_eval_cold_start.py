"""Cold CLI processes keep evaluation JSON separate from bootstrap logs."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path


def test_eval_cold_start_emits_only_json_on_stdout(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text('{"id":"a","query":"query","expected_paths":["a.py"]}\n')
    script = textwrap.dedent(
        """
        import sys
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from stropha.cli import app
        from stropha.eval import EvalCase, EvalReport, EvalResult
        from stropha.logging import get_logger

        def load_config():
            get_logger().info("bootstrap.config.loaded")
            return SimpleNamespace(log_level="INFO", target_repo=None)

        report = EvalReport(top_k=5)
        report.results = [
            EvalResult(
                case=EvalCase(id="a", query="query", expected_paths=("a.py",), split=split),
                rank_of_first_hit=1, top_k=5, relevances=(1.0,), relevant_count=1,
            )
            for split in ("search", "holdout")
        ]

        def evaluate(*args, **kwargs):
            get_logger().info("eval.completed", cases=2)
            return report

        storage = MagicMock()
        storage.lookup_repo_id = None
        storage.get_meta.return_value = None
        built = SimpleNamespace(
            storage=storage, retrieval=object(), effective_enricher_id=lambda: "noop",
        )
        with (
            patch("stropha.cli._load_config", side_effect=load_config),
            patch("stropha.cli.load_pipeline_config", return_value=object()),
            patch("stropha.cli.build_stages", return_value=built),
            patch("stropha.eval.run_eval", side_effect=evaluate),
        ):
            app(args=["eval", "--json", "--golden", sys.argv[1]])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(golden)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["cases"] == 2
    assert set(payload["by_split"]) == {"search", "holdout"}
    assert all(summary["cases"] == 1 for summary in payload["by_split"].values())
    assert "bootstrap.config.loaded" in result.stderr
    assert "eval.completed" in result.stderr
