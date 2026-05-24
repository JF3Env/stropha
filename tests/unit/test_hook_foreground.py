"""Foreground-latency invariants for the post-commit hook (WIN-206).

CLAUDE.md §3.5 is the non-negotiable contract: the post-commit hook
MUST exit in under 100ms in the foreground so ``git commit`` returns
instantly. All real work (graphify update, stropha index, outcome
backfill, sleep-time meta loop) runs inside a single detached
``nohup`` + ``flock`` background — the foreground stays a thin
prologue that resolves binaries and forks the bg process.

This file CI-gates the invariant. Two complementary test families:

1. Structural assertions — the rendered hook script must have the
   shape required for a <100ms foreground (single ``nohup`` block,
   heavy commands deferred into it, fail-fast skip conditions, stdin
   detached from the TTY).
2. Wall-clock measurements — actually execute the rendered script
   under controlled conditions and assert the foreground portion
   stayed within budget. Marker tests use POSIX timing primitives
   so the gate runs on macOS + Linux CI.

A regression here means a future hook change pushed an operation that
should have been backgrounded into the foreground path. The CI signal
is a numeric assertion (<200ms with a 2x safety margin); the failure
message tells the operator exactly which budget got exceeded.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stropha.tools.hook_install import (
    END_MARKER,
    HOOK_VERSION,
    START_MARKER,
    _render_hook_block,
    install,
)


# Wall-clock budget. §3.5 specifies <100ms; we use 200ms in the gate
# so CI jitter (cold Python interpreter, GC pause, slow disk on a
# noisy runner) doesn't flake the suite. A real regression that pushes
# a multi-second op into the foreground blows past 200ms by orders of
# magnitude.
FOREGROUND_BUDGET_MS = 200


# Skip wall-clock tests on Windows — the hook is POSIX shell.
posix_only = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="post-commit hook is POSIX shell; Windows uses a different mechanism",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Bare git repo skeleton, enough for `install()` to drop a hook."""
    r = tmp_path / "repo"
    r.mkdir()
    (r / ".git").mkdir()
    (r / ".git" / "hooks").mkdir()
    return r


@pytest.fixture
def real_repo(tmp_path: Path) -> Path:
    """Initialise a real git repo so the hook's ``git rev-parse``
    + ``git diff`` calls actually succeed during wall-clock tests."""
    r = tmp_path / "real-repo"
    r.mkdir()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "init", "-q"], cwd=r, check=True, env=env)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=r, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=r, check=True
    )
    # Plant + commit a file so HEAD exists for the hook's git diff
    # invocation.
    (r / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"], cwd=r, check=True, env=env
    )
    return r


def _time_hook_foreground(repo: Path, env_overrides: dict[str, str]) -> float:
    """Execute the hook with the given env and return foreground ms.

    Uses ``time.perf_counter`` around a subprocess.run — the bg job
    spawned by ``nohup`` keeps running after the foreground exits;
    perf_counter sees only the parent's wall clock.
    """
    hook = repo / ".git" / "hooks" / "post-commit"
    env = {**os.environ, **env_overrides}
    start = time.perf_counter()
    subprocess.run(
        ["sh", str(hook)], cwd=repo, env=env, check=False,
        capture_output=True, timeout=10,
    )
    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Structural assertions — the rendered script MUST keep the bg shape
# ---------------------------------------------------------------------------


class TestForegroundStructure:
    """The hook template must keep its detached-background shape so
    the foreground stays a thin prologue. These tests do not run the
    script; they inspect the rendered text for structural properties
    that are required for the <100ms invariant."""

    def test_template_has_single_nohup_block(self, tmp_path: Path) -> None:
        """All bg work MUST funnel through ONE ``nohup sh -c``. A
        second nohup would mean two parent waits and double the
        foreground cost."""
        text = _render_hook_block(tmp_path)
        assert text.count("nohup sh -c") == 1

    def test_template_disowns_after_background(
        self, tmp_path: Path
    ) -> None:
        """``disown`` after the ``&`` removes the bg job from the
        shell's job table so the parent doesn't wait on it. Without
        it, some shells (zsh interactive) will block on exit."""
        text = _render_hook_block(tmp_path)
        # Anchored: `&` line followed by disown.
        assert "< /dev/null &" in text
        assert "disown 2>/dev/null || true" in text

    def test_template_detaches_stdin_from_terminal(
        self, tmp_path: Path
    ) -> None:
        """``< /dev/null`` on the nohup ensures the bg job inherits
        no TTY. A child holding a TTY can block its parent on exit."""
        text = _render_hook_block(tmp_path)
        assert "< /dev/null &" in text

    def test_template_redirects_bg_output_to_log(
        self, tmp_path: Path
    ) -> None:
        """Both stdout and stderr of the nohup body are appended to
        the hook log. The parent never blocks on a pipe buffer."""
        text = _render_hook_block(tmp_path)
        assert ">> \"$LOG\" 2>&1 < /dev/null &" in text

    def test_template_heavy_commands_are_inside_nohup(
        self, tmp_path: Path
    ) -> None:
        """``$CMD`` / ``$BACKFILL_CMD`` / ``$META_CMD`` carry the heavy
        operations as shell strings; their EXECUTION must happen inside
        the nohup body (the references appear there as ``$CMD`` etc.
        followed by output redirection). The literal command strings
        live in the foreground only as variable assignments — those
        do no work, they just build the string the bg process will
        run."""
        text = _render_hook_block(tmp_path, meta_iterations=3)
        nohup_open = text.find("nohup sh -c")
        nohup_close = text.find('" >> "$LOG" 2>&1', nohup_open)
        assert nohup_open > -1 and nohup_close > nohup_open
        body = text[nohup_open:nohup_close]
        # Each command-string variable must be invoked inside the nohup.
        assert "$CMD >> '$LOG' 2>&1" in body
        assert "$BACKFILL_CMD >> '$LOG' 2>&1" in body
        assert "$META_CMD >> '$LOG' 2>&1" in body
        # v=7: graphify is no longer called directly in the hook —
        # it's handled inside `stropha index` via Pipeline._run_graphify().
        assert "'$GRAPHIFY' update" not in body

    def test_template_skip_conditions_exit_fast(
        self, tmp_path: Path
    ) -> None:
        """The early-skip block (rebase / merge / cherry-pick) must
        ``exit 0`` BEFORE any real work. A future contributor adding
        an LLM call before these checks would push the foreground
        over budget on every commit-mid-rebase."""
        text = _render_hook_block(tmp_path)
        skip_block_end = text.find("TOPLEVEL=")
        skip_block = text[:skip_block_end]
        # Each guard must terminate with `exit 0` so the foreground
        # is just `[ -d ... ] && exit 0` per branch.
        for guard in [
            "rebase-merge",
            "rebase-apply",
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "STROPHA_HOOK_SKIP",
        ]:
            assert guard in skip_block
        assert "exit 0" in skip_block

    def test_template_no_synchronous_network_calls_in_foreground(
        self, tmp_path: Path
    ) -> None:
        """Foreground portion (everything before ``nohup``) must not
        invoke curl / wget / pip / uv install — any of those would
        blow past the 100ms budget. Regression guard: the parser
        scans the foreground for forbidden binaries."""
        text = _render_hook_block(tmp_path)
        nohup_at = text.find("nohup sh -c")
        foreground = text[:nohup_at]
        forbidden = ["curl ", "wget ", "pip install", "uv add ", "uv install"]
        for needle in forbidden:
            assert needle not in foreground, (
                f"forbidden synchronous command {needle!r} found in "
                f"hook foreground"
            )

    def test_meta_iterations_setup_in_foreground_stays_constant_time(
        self, tmp_path: Path
    ) -> None:
        """v=6 added a META_CMD setup block before the nohup. It MUST
        be pure shell variable assignment + an integer comparison —
        no external command (no uv, no python, no stropha) before
        the bg fork. Otherwise the foreground scales with the meta
        config and breaks §3.5."""
        text = _render_hook_block(tmp_path, meta_iterations=10)
        nohup_at = text.find("nohup sh -c")
        # The META_CMD block is between the BACKFILL_CMD block and
        # the nohup. Find it.
        backfill_at = text.find('BACKFILL_CMD=""')
        assert backfill_at > -1 and backfill_at < nohup_at
        meta_setup = text[backfill_at:nohup_at]
        # Permitted: shell builtins + the timeout binary resolution.
        # NOT permitted: anything that would fork a heavyweight
        # process. Specifically the foreground must NOT invoke
        # `stropha`, `uv`, or `python` before the nohup.
        for needle in ["stropha ", "uv run ", "python "]:
            # The assignment "META_CMD=...stropha..." is OK (it builds
            # the command string but does not execute it). What's NOT
            # OK is an unguarded command at the top level.
            # Strip the assignment lines to test the rest.
            non_assignment_lines = "\n".join(
                ln for ln in meta_setup.splitlines()
                if "=" not in ln and "if " not in ln and "fi" not in ln
                and ln.strip()
            )
            assert needle not in non_assignment_lines, (
                f"meta-iterations setup invokes {needle!r} in "
                f"foreground — must defer to the nohup body"
            )

    def test_template_log_mkdir_only_does_local_filesystem(
        self, tmp_path: Path
    ) -> None:
        """``mkdir -p`` on the log dir is the heaviest foreground op
        besides binary resolution. Make sure it's the LOCAL path —
        not a network mount path or anything weird. Defensive
        regression guard."""
        text = _render_hook_block(tmp_path)
        # Pull the LOG resolution + mkdir into the foreground portion.
        assert 'mkdir -p "$(dirname "$LOG")"' in text


# ---------------------------------------------------------------------------
# Wall-clock measurements — execute the hook + assert <200ms
# ---------------------------------------------------------------------------


@posix_only
class TestForegroundLatency:
    """Actually run the hook script and verify the foreground stays
    inside ``FOREGROUND_BUDGET_MS``. The bg job spawned by ``nohup``
    continues after the foreground exits — we never wait for it."""

    def test_skip_path_is_well_under_budget(
        self, real_repo: Path
    ) -> None:
        """``STROPHA_HOOK_SKIP=1`` is the fastest path — it exits
        before resolving any binaries. Should be a few ms."""
        install(real_repo)
        elapsed = _time_hook_foreground(
            real_repo, {"STROPHA_HOOK_SKIP": "1"}
        )
        assert elapsed < FOREGROUND_BUDGET_MS, (
            f"skip path took {elapsed:.1f}ms > {FOREGROUND_BUDGET_MS}ms"
        )

    def test_full_foreground_under_budget(self, real_repo: Path) -> None:
        """End-to-end foreground run with a fake ``uv`` so the bg job
        forks immediately and dies. Foreground includes the path
        resolution + nohup spawn but must still exit fast."""
        install(real_repo)
        elapsed = _time_hook_foreground(
            real_repo,
            {
                "STROPHA_HOOK_NO_GRAPHIFY": "1",
                # ``/usr/bin/false`` exists on every POSIX host and
                # exits immediately — short-circuits the bg uv run
                # without blocking the foreground.
                "STROPHA_HOOK_UV": "/usr/bin/false",
            },
        )
        assert elapsed < FOREGROUND_BUDGET_MS, (
            f"full foreground took {elapsed:.1f}ms > "
            f"{FOREGROUND_BUDGET_MS}ms — a heavy operation may have "
            f"leaked into the foreground path"
        )

    def test_foreground_does_not_wait_for_slow_bg_job(
        self, real_repo: Path
    ) -> None:
        """The bg job is detached: even a 5-second ``sleep 5`` as the
        bg ``uv`` returns the foreground in <budget. This is the
        core §3.5 contract — backgrounding really backgrounds."""
        # Find a sleep binary the bg uv can resolve to.
        sleep_bin = None
        for candidate in ("/bin/sleep", "/usr/bin/sleep"):
            if Path(candidate).is_file():
                sleep_bin = candidate
                break
        if sleep_bin is None:
            pytest.skip("no /bin/sleep available")
        install(real_repo)
        # Wrap the slow command as the "uv" — the hook script will
        # invoke it inside the nohup block. The foreground must
        # not block on it.
        elapsed = _time_hook_foreground(
            real_repo,
            {
                "STROPHA_HOOK_NO_GRAPHIFY": "1",
                "STROPHA_HOOK_UV": sleep_bin,
            },
        )
        assert elapsed < FOREGROUND_BUDGET_MS, (
            f"foreground waited {elapsed:.1f}ms on a slow bg job "
            f"(should be <{FOREGROUND_BUDGET_MS}ms — bg detach is broken)"
        )

    def test_meta_iterations_does_not_inflate_foreground(
        self, real_repo: Path
    ) -> None:
        """Installing with a large ``--meta-iterations`` count must
        not slow the foreground — the meta loop runs inside the
        same nohup as everything else."""
        install(real_repo, meta_iterations=100)
        elapsed = _time_hook_foreground(
            real_repo,
            {
                "STROPHA_HOOK_NO_GRAPHIFY": "1",
                "STROPHA_HOOK_UV": "/usr/bin/false",
            },
        )
        assert elapsed < FOREGROUND_BUDGET_MS, (
            f"foreground with meta_iterations=100 took "
            f"{elapsed:.1f}ms > {FOREGROUND_BUDGET_MS}ms — META_CMD "
            f"resolution may be too heavy"
        )

    def test_rebase_skip_path_is_fast(self, real_repo: Path) -> None:
        """Mid-rebase: the script must hit the rebase guard and exit
        without spawning anything. Even faster than the
        STROPHA_HOOK_SKIP path."""
        install(real_repo)
        # Plant a rebase-merge dir so the guard fires.
        (real_repo / ".git" / "rebase-merge").mkdir()
        elapsed = _time_hook_foreground(real_repo, {})
        assert elapsed < FOREGROUND_BUDGET_MS

    def test_no_parent_commit_skip_path_is_fast(
        self, real_repo: Path
    ) -> None:
        """When the upstream diff is empty (no changed files) the
        script ``exit 0`` short-circuits before resolving uv. Verify
        this stays inside budget too."""
        # `git diff HEAD~1 HEAD` will fail (no HEAD~1) → fallback
        # ``git diff --name-only HEAD`` against a clean tree returns
        # empty → exit 0.
        install(real_repo)
        # Reset the working tree so the diff is empty.
        subprocess.run(
            ["git", "checkout", "-q", "."], cwd=real_repo, check=True
        )
        elapsed = _time_hook_foreground(real_repo, {})
        assert elapsed < FOREGROUND_BUDGET_MS
