"""Unit tests for the outcome backfill (WIN-164).

Covers:
- Symbol extraction regex against real diff text shapes.
- ``diff_paths`` / ``diff_symbols`` / ``head_sha`` git helpers via real
  ``git init`` repos in ``tmp_path``.
- ``load_traces`` — recursive JSONL ingestion + malformed-line tolerance.
- ``run_backfill`` end-to-end: writes ``outcomes/<sha>.json`` with the
  right aggregate score and tier counts.
- The hook template bumps to ``v=5`` and includes the backfill step.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from stropha.meta.backfill import (
    BackfillError,
    BackfillSummary,
    _extract_symbols_from_text,
    diff_paths,
    diff_symbols,
    head_sha,
    load_traces,
    run_backfill,
)
from stropha.tools.hook_install import HOOK_VERSION, _render_hook_block

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "root"], cwd=repo, check=True)


def _git_commit(repo: Path, msg: str = "change") -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", msg], cwd=repo, check=True)
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


class TestSymbolExtraction:
    def test_python_def(self) -> None:
        text = "+def my_func(a, b):\n+    return a + b"
        assert _extract_symbols_from_text(text) == {"my_func"}

    def test_python_class(self) -> None:
        text = "+class MyClass:\n+    pass"
        assert _extract_symbols_from_text(text) == {"MyClass"}

    def test_javascript_function(self) -> None:
        text = "+function doStuff(x) {\n+    return x;\n+}"
        assert _extract_symbols_from_text(text) == {"doStuff"}

    def test_java_method(self) -> None:
        text = "+public int calculate(int x) {\n+    return x * 2;\n+}"
        assert _extract_symbols_from_text(text) == {"calculate"}

    def test_multiple_symbols(self) -> None:
        text = (
            "+def foo():\n"
            "+    pass\n"
            "+class Bar:\n"
            "+    def method(self):\n"
            "+        pass\n"
        )
        assert _extract_symbols_from_text(text) == {"foo", "Bar", "method"}

    def test_no_symbols_returns_empty(self) -> None:
        text = "+# just a comment\n+x = 42\n"
        assert _extract_symbols_from_text(text) == set()

    def test_handles_empty_text(self) -> None:
        assert _extract_symbols_from_text("") == set()


# ---------------------------------------------------------------------------
# Git helpers — exercised against a real tmp_path git repo
# ---------------------------------------------------------------------------


class TestGitHelpers:
    def test_head_sha_returns_sha(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        sha = head_sha(tmp_path)
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_diff_paths_finds_changed_files(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        prev = head_sha(tmp_path)
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("def bar(): pass\n")
        _git_commit(tmp_path)
        paths = diff_paths(prev, "HEAD", tmp_path)
        assert set(paths) == {"a.py", "b.py"}

    def test_diff_symbols_extracts_from_real_diff(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        prev = head_sha(tmp_path)
        (tmp_path / "x.py").write_text("def added_func():\n    return 1\n")
        _git_commit(tmp_path)
        symbols = diff_symbols(prev, "HEAD", tmp_path)
        assert "added_func" in symbols

    def test_git_failure_raises_backfill_error(self, tmp_path: Path) -> None:
        # tmp_path is not a git repo → git commands fail.
        with pytest.raises(BackfillError, match=r"git "):
            diff_paths("badsha", "HEAD", tmp_path)

    def test_git_not_found_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If git binary is missing, BackfillError is raised cleanly."""
        import subprocess as _sub

        def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("git not on PATH")

        monkeypatch.setattr(_sub, "run", boom)
        with pytest.raises(BackfillError, match="git binary not found"):
            head_sha(tmp_path)


# ---------------------------------------------------------------------------
# load_traces
# ---------------------------------------------------------------------------


class TestLoadTraces:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_traces(tmp_path) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_traces(tmp_path / "nope") == []

    def test_reads_multiple_jsonl_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.jsonl").write_text(
            json.dumps({"tool": "search_code", "trace_id": "t1"}) + "\n"
            + json.dumps({"tool": "get_symbol", "trace_id": "t2"}) + "\n"
        )
        (tmp_path / "b.jsonl").write_text(
            json.dumps({"tool": "search_code", "trace_id": "t3"}) + "\n"
        )
        records = load_traces(tmp_path)
        assert len(records) == 3
        ids = {r["trace_id"] for r in records}
        assert ids == {"t1", "t2", "t3"}

    def test_tolerates_blank_and_malformed_lines(self, tmp_path: Path) -> None:
        (tmp_path / "t.jsonl").write_text(
            "garbage{{\n"
            + "\n"
            + json.dumps({"tool": "search_code"}) + "\n"
        )
        records = load_traces(tmp_path)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# run_backfill — end-to-end
# ---------------------------------------------------------------------------


class TestRunBackfill:
    def _build_scenario(self, tmp_path: Path) -> tuple[Path, Path, str, str]:
        """Build a repo + meta dir with traces; return (repo, meta, prev, head)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        prev = head_sha(repo)
        # Make a commit that adds a function — diff_symbols should catch it.
        (repo / "src.py").write_text("def added_one(): return 1\n")
        head = _git_commit(repo, "add fn")

        meta = tmp_path / "meta"
        traces = meta / "traces"
        traces.mkdir(parents=True)
        # One trace whose result chunk's symbol matches the diff → T2.
        # Another whose chunk's file is in the diff → T1.
        # A third with no signal → T7.
        (traces / "t.jsonl").write_text(
            json.dumps({
                "trace_id": "tr-1",
                "tool": "search_code",
                "query": "added",
                "results": [
                    {"chunk_id": "c1", "rel_path": "src.py", "symbol": "added_one"},
                    {"chunk_id": "c2", "rel_path": "other.py", "symbol": "helper"},
                ],
            }) + "\n"
        )
        return repo, meta, prev, head

    def test_writes_outcome_json_with_correct_keys(self, tmp_path: Path) -> None:
        repo, meta, prev, head = self._build_scenario(tmp_path)
        run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        out_path = meta / "outcomes" / f"{head}.json"
        assert out_path.is_file()
        payload = json.loads(out_path.read_text())
        # Required keys per the spec.
        for key in (
            "since_sha", "head_sha", "diff_paths", "diff_symbols",
            "traces_scanned", "hits_labeled", "tier_counts",
            "per_trace_scores", "overall_score", "output_path",
            "generated_at",
        ):
            assert key in payload, f"missing key {key!r}"

    def test_finds_t1_and_t2_in_realistic_scenario(self, tmp_path: Path) -> None:
        repo, meta, prev, _head = self._build_scenario(tmp_path)
        summary = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        # T1 fires for chunk c1 (rel_path src.py is in diff)
        # T2 would fire if T1 wasn't first — chunk c1's symbol IS added_one.
        # Either way the chunk c1 should be scored positively. T1 > T2.
        assert summary.tier_counts["T1"] >= 1
        # c2 (unrelated to diff) should land on T7.
        assert summary.tier_counts["T7"] >= 1
        # Overall positive because the +1 from T1 outweighs the 0 from T7.
        assert summary.overall_score > 0

    def test_per_trace_scores_populated(self, tmp_path: Path) -> None:
        repo, meta, prev, _head = self._build_scenario(tmp_path)
        summary = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        assert "tr-1" in summary.per_trace_scores

    def test_no_traces_yields_zero(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        prev = head_sha(repo)
        (repo / "x.py").write_text("def a(): pass\n")
        _git_commit(repo)
        meta = tmp_path / "meta"
        summary = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        assert summary.traces_scanned == 0
        assert summary.hits_labeled == 0
        assert summary.overall_score == 0.0

    def test_backfill_skips_traces_with_no_hits(self, tmp_path: Path) -> None:
        """A trace whose results list is empty contributes zero labels."""
        repo, meta, prev, _head = self._build_scenario(tmp_path)
        # Append an extra trace with no hits.
        with open(meta / "traces" / "t.jsonl", "a") as fh:
            fh.write(json.dumps({"trace_id": "empty", "tool": "list_repos", "results": []}) + "\n")
        summary = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        # The empty trace must NOT have a per-trace score entry.
        assert "empty" not in summary.per_trace_scores

    def test_backfill_tolerates_feedback_store_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When BadResultStore cannot open (e.g. read-only fs), backfill
        keeps going — T6 just never fires, no exception bubbles up."""
        repo, meta, prev, _head = self._build_scenario(tmp_path)

        from stropha.meta import feedback as fb_mod

        def broken_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("simulated read-only fs")

        monkeypatch.setattr(fb_mod.BadResultStore, "__init__", broken_init)
        # Must not raise — the backfill swallows the OSError and skips T6.
        result = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        assert result.tier_counts["T6"] == 0

    def test_backfill_consumes_bad_results(self, tmp_path: Path) -> None:
        """When BadResultStore has reports, the labeler emits T6 for them."""
        from stropha.meta import BadResultStore

        repo, meta, prev, _head = self._build_scenario(tmp_path)
        # Mark trace tr-1 as bad — its results c1 and c2 must label T6.
        store = BadResultStore(meta)
        store.report(trace_id="tr-1", reason="irrelevant")

        summary = run_backfill(since_sha=prev, meta_dir=meta, repo_dir=repo)
        # T6 fires for both chunks of the reported trace.
        assert summary.tier_counts["T6"] >= 1
        # Overall score should be dragged down by the strong-negative signal.
        assert summary.overall_score < 1.0

    def test_summary_to_dict_round_trip(self) -> None:
        s = BackfillSummary(
            since_sha="a",
            head_sha="b",
            diff_paths=["x.py"],
            diff_symbols=["foo"],
            traces_scanned=3,
            hits_labeled=7,
            tier_counts={"T1": 1, "T7": 6},
            per_trace_scores={"t1": 0.5},
            overall_score=0.143,
            output_path="/tmp/x.json",
        )
        d = s.to_dict()
        assert d["since_sha"] == "a"
        assert d["tier_counts"]["T1"] == 1
        assert "generated_at" in d


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------


class TestHookV5:
    def test_hook_version_is_five(self) -> None:
        assert HOOK_VERSION == 5

    def test_template_has_v5_marker(self, tmp_path: Path) -> None:
        text = _render_hook_block(tmp_path)
        assert "v=5" in text

    def test_template_includes_backfill_step(self, tmp_path: Path) -> None:
        text = _render_hook_block(tmp_path)
        assert "outcome backfill" in text
        assert "BACKFILL_CMD" in text
        assert "stropha meta outcome backfill" in text
        assert "--since" in text

    def test_template_skips_backfill_when_no_parent(self, tmp_path: Path) -> None:
        """Backfill is only invoked when ``PREV_SHA`` resolved (parent exists).
        The template guards on ``[ -n '$BACKFILL_CMD' ]`` so the initial
        commit on a fresh repo doesn't try to diff against a nonexistent SHA."""
        text = _render_hook_block(tmp_path)
        assert "git rev-parse HEAD~1" in text
        assert "if [ -n '$BACKFILL_CMD' ]" in text

    def test_template_keeps_existing_steps(self, tmp_path: Path) -> None:
        """v=5 must NOT regress the v=2/3/4 features."""
        text = _render_hook_block(tmp_path)
        # graphify update step still there
        assert "graphify update" in text
        # incremental index still there
        assert "--incremental" in text
        # nohup + flock background still there
        assert "nohup sh" in text
        assert "flock -n 9" in text


# ---------------------------------------------------------------------------
# CLI smoke (typer-level)
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_meta_outcome_backfill_command_exists(self) -> None:
        """The CLI app must expose ``meta outcome backfill``."""
        from stropha.cli import app

        # Walk Typer's registered groups.
        meta_app = None
        for group in app.registered_groups:
            if group.name == "meta":
                meta_app = group.typer_instance
                break
        assert meta_app is not None, "meta sub-app not registered"

        outcome_app = None
        for group in meta_app.registered_groups:
            if group.name == "outcome":
                outcome_app = group.typer_instance
                break
        assert outcome_app is not None, "outcome sub-app not registered"

        commands = {cmd.name for cmd in outcome_app.registered_commands}
        assert "backfill" in commands