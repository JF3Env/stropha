"""Unit tests for ``stropha.tools.hook_install``.

Per RFC §6.2 / §9 (Fase 1.5c):
- install() é idempotente
- preserva conteúdo existente do post-commit (não sobrescreve)
- detecta marker version
- detecta coexistência com graphify hook
- uninstall() remove APENAS nosso bloco
- file removido se sobra só shebang
- core.hooksPath respeitado quando válido
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from stropha.tools.hook_install import (
    END_MARKER,
    GRAPHIFY_MARKER,
    HOOK_VERSION,
    START_MARKER,
    _resolve_hooks_dir,
    install,
    status,
    uninstall,
)

# --------------------------------------------------------------------------- helpers


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Initialise a bare git repo skeleton (just .git/) suitable for hook tests."""
    r = tmp_path / "repo"
    r.mkdir()
    (r / ".git").mkdir()
    (r / ".git" / "hooks").mkdir()
    return r


def _read_hook(repo: Path) -> str:
    return (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- install


def test_install_creates_new_hook(repo: Path) -> None:
    result = install(repo)
    assert result["action"] == "created"
    assert result["version"] == HOOK_VERSION

    hook_text = _read_hook(repo)
    assert hook_text.startswith("#!/bin/sh")
    assert START_MARKER in hook_text
    assert END_MARKER in hook_text
    assert f"v={HOOK_VERSION}" in hook_text

    # Executable
    mode = (repo / ".git" / "hooks" / "post-commit").stat().st_mode
    assert mode & 0o100  # owner-x


def test_install_is_idempotent(repo: Path) -> None:
    install(repo)
    first = _read_hook(repo)
    install(repo)
    second = _read_hook(repo)
    assert first == second


def test_install_appends_when_existing_file_has_no_markers(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        "#!/bin/sh\necho 'pre-existing user hook'\n",
        encoding="utf-8",
    )
    result = install(repo)
    assert result["action"] == "appended"
    body = _read_hook(repo)
    # Pre-existing content preserved
    assert "pre-existing user hook" in body
    # Our block added
    assert START_MARKER in body
    assert END_MARKER in body


def test_install_replaces_existing_block(repo: Path) -> None:
    install(repo)
    # Manually corrupt the version line to simulate an outdated install
    hook_path = repo / ".git" / "hooks" / "post-commit"
    body = hook_path.read_text(encoding="utf-8")
    body = body.replace(f"{START_MARKER} v={HOOK_VERSION}",
                        f"{START_MARKER} v=0")
    hook_path.write_text(body, encoding="utf-8")

    result = install(repo)
    assert result["action"] == "updated"

    new_body = _read_hook(repo)
    # The old block is gone, only the new (current) version line remains
    assert f"{START_MARKER} v=0" not in new_body
    assert f"{START_MARKER} v={HOOK_VERSION}" in new_body
    # Should appear exactly once
    assert new_body.count(START_MARKER) == 1


def test_install_detects_graphify_cohabit(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        f"#!/bin/sh\n{GRAPHIFY_MARKER}\necho 'graphify'\n# graphify-hook-end\n",
        encoding="utf-8",
    )
    result = install(repo)
    assert result["graphify_cohabit"] is True
    assert result["coexist_warning"] is True

    # With force=True, no warning
    result_force = install(repo, force=True)
    assert result_force["graphify_cohabit"] is True
    assert result_force["coexist_warning"] is False


def test_install_rejects_non_git_dir(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    with pytest.raises(ValueError, match="not a git repository"):
        install(not_a_repo)


def test_install_target_substituted_in_script(repo: Path) -> None:
    install(repo)
    body = _read_hook(repo)
    # The target absolute path appears in the EXPECTED= line
    assert f'EXPECTED="{repo.resolve()}"' in body


# --------------------------------------------------------------------------- status


def test_status_when_not_installed(repo: Path) -> None:
    s = status(repo)
    assert s.installed is False
    assert s.version is None
    assert s.graphify_cohabit is False


def test_status_after_install(repo: Path) -> None:
    install(repo)
    s = status(repo)
    assert s.installed is True
    assert s.version == HOOK_VERSION
    assert s.graphify_cohabit is False


def test_status_outdated_version(repo: Path) -> None:
    install(repo)
    hook_path = repo / ".git" / "hooks" / "post-commit"
    body = hook_path.read_text(encoding="utf-8")
    body = body.replace(f"{START_MARKER} v={HOOK_VERSION}",
                        f"{START_MARKER} v=0")
    hook_path.write_text(body, encoding="utf-8")
    s = status(repo)
    assert s.installed is True
    assert s.version == 0  # detected, but != HOOK_VERSION


# --------------------------------------------------------------------------- uninstall


def test_uninstall_when_not_installed(repo: Path) -> None:
    result = uninstall(repo)
    assert result["action"] == "noop"


def test_uninstall_removes_only_our_block(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text(
        "#!/bin/sh\necho 'user hook before'\n",
        encoding="utf-8",
    )
    install(repo)
    body_after_install = _read_hook(repo)
    assert "user hook before" in body_after_install
    assert START_MARKER in body_after_install

    result = uninstall(repo)
    assert result["action"] == "block_removed"
    body_after_uninstall = _read_hook(repo)
    assert "user hook before" in body_after_uninstall
    assert START_MARKER not in body_after_uninstall
    assert END_MARKER not in body_after_uninstall


def test_uninstall_removes_file_when_only_shebang_left(repo: Path) -> None:
    install(repo)
    result = uninstall(repo)
    assert result["action"] == "removed"
    assert not (repo / ".git" / "hooks" / "post-commit").exists()


def test_uninstall_noop_when_only_unrelated_content(repo: Path) -> None:
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.write_text("#!/bin/sh\necho hello\n", encoding="utf-8")
    result = uninstall(repo)
    assert result["action"] == "noop"
    # File preserved
    assert hook_path.read_text(encoding="utf-8") == "#!/bin/sh\necho hello\n"


# --------------------------------------------------------------------------- core.hooksPath


def test_resolve_hooks_dir_default(repo: Path) -> None:
    """No core.hooksPath configured → default `.git/hooks` (RFC §6.3)."""
    p = _resolve_hooks_dir(repo)
    assert p == (repo / ".git" / "hooks").resolve()


def test_resolve_hooks_dir_honours_core_hookspath(tmp_path: Path) -> None:
    """When git config core.hooksPath set inside repo, follow it."""
    # Initialise a real git repo so `git config` works
    repo = tmp_path / "withhooks"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})

    custom = repo / ".husky"
    custom.mkdir()
    subprocess.run(
        ["git", "config", "core.hooksPath", ".husky"],
        cwd=repo, check=True,
    )
    p = _resolve_hooks_dir(repo)
    assert p == custom.resolve()


def test_resolve_hooks_dir_rejects_unsafe_path(tmp_path: Path) -> None:
    """A core.hooksPath outside repo and HOME is rejected → falls back."""
    repo = tmp_path / "unsafe"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                   env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    # Point hooksPath to /tmp (outside both repo and HOME, on most systems)
    bad = Path("/private/tmp/totally-not-allowed")
    subprocess.run(
        ["git", "config", "core.hooksPath", str(bad)],
        cwd=repo, check=True,
    )
    p = _resolve_hooks_dir(repo)
    # When the configured path is rejected, we fall back to .git/hooks.
    # /tmp may legitimately resolve under HOME on some CI machines so we
    # only assert the fallback was chosen when the path was outside HOME.
    if not str(bad).startswith(str(Path.home().resolve())):
        assert p == (repo / ".git" / "hooks").resolve()


# --------------------------------------------------------------------------- v=3 cross-repo baked defaults


def test_install_without_new_args_uses_legacy_toplevel(repo: Path) -> None:
    """Back-compat regression guard: install() with no new flags must emit
    empty bakes and let runtime fall back to $TOPLEVEL."""
    install(repo)
    body = _read_hook(repo)
    assert 'PROJECT_DIR_DEFAULT=""' in body
    assert 'LOG_DEFAULT=""' in body
    # Runtime fallback chain must point at $TOPLEVEL when default is empty.
    assert 'PROJECT_DIR="${STROPHA_HOOK_PROJECT_DIR:-${PROJECT_DIR_DEFAULT:-$TOPLEVEL}}"' in body
    # And the CMD still uses $PROJECT_DIR for --directory (NOT hardcoded $TOPLEVEL).
    assert "--directory $PROJECT_DIR" in body
    # v=9: dead env-var plumbing is gone — the install must NOT emit any
    # of these markers.
    assert "INDEX_PATH_DEFAULT" not in body
    assert "INDEX_ENV" not in body
    assert "STROPHA_INDEX_PATH" not in body
    assert "STROPHA_HOOK_INDEX_PATH" not in body


def test_install_with_project_dir_bakes_it(repo: Path, tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv-host"
    venv_dir.mkdir()
    # Make pyproject.toml present so the bake is realistic
    (venv_dir / "pyproject.toml").write_text("[project]\nname = 'host'", encoding="utf-8")

    result = install(repo, project_dir=venv_dir)
    body = _read_hook(repo)
    assert f'PROJECT_DIR_DEFAULT="{venv_dir.resolve()}"' in body
    # The CMD line still references $PROJECT_DIR (not the literal path) —
    # the bake flows through the env-resolution chain.
    assert "--directory $PROJECT_DIR" in body
    # install() returns the resolved path
    assert result["project_dir"] == str(venv_dir.resolve())


def test_install_v9_drops_index_path_bake(repo: Path, tmp_path: Path) -> None:
    """v=9 regression guard: the dead INDEX_PATH bake / INDEX_ENV /
    STROPHA_(HOOK_)INDEX_PATH plumbing must not come back.

    These were silently no-ops in v<=8 (the env var was set in 3 sites
    and read in 0). Removing them simplifies the hook script and the
    cross-repo install API. Per-repo DB selection now lives in
    ``~/.stropha/config.yaml`` ``paths.index``.
    """
    install(repo)
    body = _read_hook(repo)
    assert "INDEX_PATH_DEFAULT" not in body
    assert "INDEX_ENV" not in body
    assert "STROPHA_INDEX_PATH" not in body
    assert "STROPHA_HOOK_INDEX_PATH" not in body


def test_install_with_log_path_bakes_it(repo: Path, tmp_path: Path) -> None:
    log = tmp_path / "logs" / "hook.log"

    result = install(repo, log_path=log)
    body = _read_hook(repo)
    assert f'LOG_DEFAULT="{log.expanduser()}"' in body
    # The LOG runtime resolution must consume LOG_DEFAULT.
    assert 'LOG="${STROPHA_HOOK_LOG:-${LOG_DEFAULT:-${HOME}/.cache/stropha-hook.log}}"' in body
    assert result["log_path"] == str(log.expanduser())


def test_status_reports_baked_defaults(repo: Path, tmp_path: Path) -> None:
    project = tmp_path / "host"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'host'", encoding="utf-8")
    logp = tmp_path / "h.log"

    install(repo, project_dir=project, log_path=logp)
    s = status(repo)
    assert s.installed is True
    assert s.version == HOOK_VERSION
    assert s.project_dir == project.resolve()
    assert s.log_path_default == logp.expanduser()
    # Dict serialisation also surfaces them.
    d = s.as_dict()
    assert d["project_dir"] == str(project.resolve())
    assert d["log_path_default"] == str(logp.expanduser())
    # v=9: index_path field removed from HookStatus.as_dict().
    assert "index_path" not in d


def test_install_replaces_older_block_with_current(repo: Path) -> None:
    """Upgrade path: a manually-written older-version block must be
    cleanly rewritten to ``HOOK_VERSION`` in place (no duplication, no
    leftover markers). Validates the v=2 → v=4 upgrade."""
    hook_path = repo / ".git" / "hooks" / "post-commit"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    # Synthetic v=2 block — content doesn't matter, just the markers.
    hook_path.write_text(
        "#!/bin/sh\n"
        "# user prelude\n"
        f"{START_MARKER} v=2\n"
        "echo legacy v=2 body\n"
        f"{END_MARKER}\n"
        "echo user epilog\n",
        encoding="utf-8",
    )

    result = install(repo)
    assert result["action"] == "updated"

    body = _read_hook(repo)
    # The new block reports the current HOOK_VERSION …
    assert f"{START_MARKER} v={HOOK_VERSION}" in body
    # … the old v=2 marker is gone …
    assert f"{START_MARKER} v=2" not in body
    # … and the surrounding user content is preserved.
    assert "# user prelude" in body
    assert "echo user epilog" in body
    # Block must appear exactly once.
    assert body.count(START_MARKER) == 1


def _exec_lines(body: str) -> list[str]:
    """Return the body with comments stripped — lines that actually run.

    Used to assert against the executable surface of the hook script
    without false positives from comments that mention the removed
    constructs by name (e.g. the v=9 docstring explains the flock and
    --incremental removal).
    """
    out: list[str] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line)
    return out


def test_install_v9_template_uses_auto_mode(repo: Path) -> None:
    """v=9: drop ``--incremental``. The CLI's default ``auto`` mode does
    the right thing (incremental when ``last_indexed_sha`` checkpoint
    exists, full otherwise), so the hook and the interactive CLI share
    one code path.
    """
    install(repo)
    body = _read_hook(repo)
    exec_only = "\n".join(_exec_lines(body))
    # `stropha index --repo $TOPLEVEL` (no --incremental, no --full) in
    # the executable surface of the script.
    assert "stropha index --repo $TOPLEVEL" in exec_only
    assert "--incremental" not in exec_only
    assert "--full" not in exec_only


def test_install_v9_uses_mkdir_lock(repo: Path) -> None:
    """v=9: replace flock with mkdir-based atomic lock for macOS portability.

    On macOS ``flock`` is not installed by default; v<=8 hooks silently
    skipped every commit because the missing binary returned non-zero,
    taking the "already running" branch. The new lock uses ``mkdir``
    (POSIX-atomic) + a PID file for stale-lock recovery.
    """
    install(repo)
    body = _read_hook(repo)
    exec_only = "\n".join(_exec_lines(body))
    # Old flock-based guard MUST be gone from the executable surface.
    assert "flock" not in exec_only
    # New mkdir-based lock MUST be present.
    assert "LOCK_DIR=" in exec_only
    assert "mkdir '" in exec_only  # mkdir attempt inside the nohup body
    assert "kill -0" in exec_only  # stale-PID detection
    assert "/pid" in exec_only  # PID file written to lock dir


# --------------------------------------------------------------------------- v=6 sleep-time meta loop (WIN-203)


class TestV6MetaIterations:
    """v=6 bakes a Meta-Harness bandit loop into the post-commit refresh.

    The meta step runs inside the same detached ``nohup`` block as the
    index refresh, so the foreground stays <100ms (CLAUDE.md §3.5).
    Default is disabled — the operator opts in via ``meta_iterations=N``
    when calling :func:`install`.
    """

    def test_default_install_disables_meta_loop(self, repo: Path) -> None:
        """No flag → ``META_ITERATIONS_DEFAULT=""`` and the meta step is
        a script no-op. Regression guard: shipping v=6 must not turn on
        the meta loop for every existing v=5 install on upgrade."""
        result = install(repo)
        assert result["meta_iterations"] == 0
        body = _read_hook(repo)
        assert 'META_ITERATIONS_DEFAULT=""' in body
        # The script defines META_CMD but it stays empty when the bake
        # is zero, so the runtime ``if [ -n '$META_CMD' ]`` short-circuits.
        assert "META_CMD=\"\"" in body

    def test_install_with_meta_iterations_bakes_the_count(
        self, repo: Path
    ) -> None:
        result = install(repo, meta_iterations=5)
        assert result["meta_iterations"] == 5
        body = _read_hook(repo)
        assert 'META_ITERATIONS_DEFAULT="5"' in body
        # Command line uses the resolved $META_ITERATIONS var, not the
        # literal — runtime env overrides must keep working.
        assert "--iterations $META_ITERATIONS --proposer bandit" in body

    def test_negative_meta_iterations_clamped_to_zero(
        self, repo: Path
    ) -> None:
        """Defensive: caller passes a negative integer (e.g. -1 from a
        misconfigured argparse default) — clamp to 0 instead of baking
        a nonsense value that breaks the runtime ``[ -gt 0 ]`` test."""
        result = install(repo, meta_iterations=-3)
        assert result["meta_iterations"] == 0
        body = _read_hook(repo)
        assert 'META_ITERATIONS_DEFAULT=""' in body

    def test_install_with_zero_meta_iterations_is_explicit_disable(
        self, repo: Path
    ) -> None:
        """Passing 0 explicitly is equivalent to omitting the flag —
        baked as empty string sentinel, meta step is a no-op."""
        result = install(repo, meta_iterations=0)
        assert result["meta_iterations"] == 0
        body = _read_hook(repo)
        assert 'META_ITERATIONS_DEFAULT=""' in body

    def test_install_with_meta_runs_after_backfill(
        self, repo: Path
    ) -> None:
        """The meta loop step must appear AFTER the backfill block so
        the candidate evaluation reads a freshly-labelled trace stream.
        Order matters: index → backfill → meta. We compare the
        execution log lines (inside the nohup block) since those are
        what the runtime hits — comment headers earlier in the file
        would confuse a naive substring search."""
        install(repo, meta_iterations=3)
        body = _read_hook(repo)
        backfill_pos = body.find("[stropha-hook] outcome backfill")
        meta_pos = body.find("[stropha-hook] meta loop")
        assert backfill_pos > -1, "backfill log line missing"
        assert meta_pos > -1, "meta loop log line missing"
        assert meta_pos > backfill_pos

    def test_install_with_meta_respects_no_meta_env(
        self, repo: Path
    ) -> None:
        """``STROPHA_HOOK_NO_META=1`` is the panic switch — even with a
        non-zero baked value, the runtime can disable the meta step
        without re-installing the hook."""
        install(repo, meta_iterations=3)
        body = _read_hook(repo)
        # The runtime resolution block must clear META_ITERATIONS when
        # STROPHA_HOOK_NO_META=1, before the META_CMD assignment.
        # After dedent, the inner block has 4-space indent.
        assert (
            'if [ "${STROPHA_HOOK_NO_META:-0}" = "1" ]; then\n'
            "    META_ITERATIONS=0"
            in body
        )

    def test_install_meta_command_uses_uv_run_directory(
        self, repo: Path, tmp_path: Path
    ) -> None:
        """The meta loop must reuse the same ``--directory $PROJECT_DIR``
        chain as the index step so cross-repo installs (target ≠ where
        stropha lives) keep working."""
        venv_dir = tmp_path / "host"
        venv_dir.mkdir()
        (venv_dir / "pyproject.toml").write_text(
            "[project]\nname='h'", encoding="utf-8"
        )
        install(repo, project_dir=venv_dir, meta_iterations=2)
        body = _read_hook(repo)
        # The bandit command must `uv run --directory $PROJECT_DIR`.
        assert (
            "$UV run --directory $PROJECT_DIR stropha meta run --iterations $META_ITERATIONS"
            in body
        )

    def test_install_meta_command_uses_timeout_when_available(
        self, repo: Path
    ) -> None:
        """When the host has ``timeout`` / ``gtimeout``, the meta step
        wraps the bandit call so a runaway proposer can't burn the
        bg slot indefinitely."""
        install(repo, meta_iterations=7)
        body = _read_hook(repo)
        # The TIMEOUT_BIN variable is reused — assert both branches of
        # the meta resolution exist.
        assert "if [ -n \"$TIMEOUT_BIN\" ]; then" in body
        # And the timeout-protected variant is present in the META_CMD
        # branch (this checks both the index CMD and meta CMD use it).
        assert body.count("$TIMEOUT_BIN ${STROPHA_HOOK_TIMEOUT:-600}") >= 2

    def test_status_reports_meta_iterations(
        self, repo: Path, tmp_path: Path
    ) -> None:
        install(repo, meta_iterations=4)
        s = status(repo)
        assert s.installed is True
        assert s.meta_iterations == 4
        # Dict serialisation surfaces it.
        d = s.as_dict()
        assert d["meta_iterations"] == 4

    def test_status_reports_zero_when_explicitly_disabled(
        self, repo: Path
    ) -> None:
        install(repo, meta_iterations=0)
        s = status(repo)
        assert s.meta_iterations == 0

    def test_status_on_older_hook_returns_none_for_meta(
        self, repo: Path
    ) -> None:
        """A hook installed by an older stropha (< v=6) does not have
        the ``META_ITERATIONS_DEFAULT`` line. ``status`` must return
        ``None`` (not 0) so the caller can distinguish "missing" from
        "explicitly disabled"."""
        # Synthesise an old-style hook block missing the meta line.
        hook_path = repo / ".git" / "hooks" / "post-commit"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text(
            "#!/bin/sh\n"
            f"{START_MARKER} v=5\n"
            "# legacy v=5 body, no META_ITERATIONS_DEFAULT line\n"
            f"{END_MARKER}\n",
            encoding="utf-8",
        )
        s = status(repo)
        assert s.installed is True
        assert s.version == 5
        assert s.meta_iterations is None

    def test_install_meta_log_lines_use_iters_count(
        self, repo: Path
    ) -> None:
        """The log message announcing the meta step must include the
        iteration count so ``stropha cost`` (WIN-205) can attribute
        TCO to the right run."""
        install(repo, meta_iterations=11)
        body = _read_hook(repo)
        # Look for the log line template — runtime substitutes the count.
        assert (
            "[stropha-hook] meta loop (commit ${SHORT_SHA}, "
            "$META_ITERATIONS iters)" in body
        )

    def test_install_preserves_existing_block_meta_setting_on_upgrade(
        self, repo: Path
    ) -> None:
        """Idempotency under upgrade: re-running install() with a fresh
        meta value cleanly overwrites the previous bake (no leftover
        old value, no duplicate META_ITERATIONS_DEFAULT line)."""
        install(repo, meta_iterations=2)
        install(repo, meta_iterations=8)
        body = _read_hook(repo)
        # Only the new value appears.
        assert 'META_ITERATIONS_DEFAULT="8"' in body
        assert 'META_ITERATIONS_DEFAULT="2"' not in body
        # Exactly one occurrence — no duplication.
        assert body.count("META_ITERATIONS_DEFAULT=") == 1
