"""Post-commit hook installer (RFC §6 / §7.1).

Manages the stropha post-commit hook that refreshes the graphify graph and
re-indexes the repo in detached background after every commit.

Public surface:
- :func:`install` — write or update the hook block between markers
- :func:`uninstall` — remove our block, leave the rest of the file intact
- :func:`status` — report whether the hook is installed and operational

Markers used: ``# stropha-hook-start v=N`` / ``# stropha-hook-end``.
The version number lets future installs detect outdated blocks and rewrite.
The hook itself runs in the background via ``nohup`` + ``flock``, never
blocks the commit by more than ~100 ms (the time to fork).

Coexistence: if the existing post-commit file already contains a graphify
hook (``# graphify-hook-start``), :func:`install` warns the caller. The
stropha hook itself already calls ``graphify update`` so keeping both
duplicates work; :func:`install` honours ``force=True`` to silence the warning.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from ..logging import get_logger

log = get_logger(__name__)

# Bump this when the embedded script changes in a backwards-incompatible way
# (e.g. added env var, renamed binary). `install --force` and `status` use it
# to detect outdated blocks.
#
# v=2: graphify update + stropha index in a single nohup block.
# v=3: cross-repo support. Bakes PROJECT_DIR / INDEX_PATH / LOG defaults at
#      install time so `--target /repo --project-dir /elsewhere` no longer
#      requires the user to set env vars on every commit.
# v=4: incremental indexing by default (Phase B). The `stropha index` CMD
#      gains `--incremental` so the post-commit refresh only re-chunks +
#      re-embeds the files that actually changed in this commit. Pipeline
#      auto-falls-back to full when no `last_indexed_sha` checkpoint exists
#      (first run after install), so existing v=3 installs upgrade cleanly.
# v=5: outcome backfill (WIN-164). After the index refresh, the hook calls
#      `stropha meta outcome backfill --since <prev_sha>` so the Meta-Harness
#      labeler turns the new trace stream into a T1-T7 outcome JSON. Runs
#      inside the same detached `nohup` block as the index refresh — git
#      commit stays <100ms in the foreground per CLAUDE.md §3.5.
# v=6: sleep-time meta iterations (WIN-203). `--meta-iterations N` bakes a
#      Meta-Harness bandit loop into the post-commit refresh. When N > 0,
#      the hook runs `stropha meta run --iterations N --proposer bandit`
#      AFTER the index refresh + backfill so every commit produces a
#      handful of candidate configs in the background. Disabled by default
#      (N=0); STROPHA_HOOK_META_ITERATIONS / STROPHA_HOOK_NO_META override
#      at commit time. Still inside the same detached `nohup` block — the
#      foreground stays <100ms per CLAUDE.md §3.5.
HOOK_VERSION = 6

START_MARKER = "# stropha-hook-start"
END_MARKER = "# stropha-hook-end"
GRAPHIFY_MARKER = "# graphify-hook-start"

# Regex matches `# stropha-hook-start v=<digits>` followed by anything up to
# `# stropha-hook-end`. DOTALL so it matches across lines.
_BLOCK_RE = re.compile(
    rf"{re.escape(START_MARKER)} v=\d+.*?{re.escape(END_MARKER)}\n?",
    re.DOTALL,
)
_VERSION_RE = re.compile(rf"{re.escape(START_MARKER)} v=(\d+)")

# Baked-default extractors (v=3). Empty string means "no bake — use legacy
# auto-resolution". We tolerate either ASCII " or fancier quotes that some
# editors might inject during manual fixes.
_PROJECT_DIR_RE = re.compile(r'PROJECT_DIR_DEFAULT="([^"]*)"')
_INDEX_PATH_RE = re.compile(r'INDEX_PATH_DEFAULT="([^"]*)"')
_LOG_DEFAULT_RE = re.compile(r'LOG_DEFAULT="([^"]*)"')
# v=6: integer count of sleep-time meta iterations baked into the script.
# "0" / empty = disabled (the default), the meta step is a no-op.
_META_ITERATIONS_RE = re.compile(r'META_ITERATIONS_DEFAULT="(\d*)"')


# --------------------------------------------------------------------- model


@dataclass(frozen=True)
class HookStatus:
    target_repo: Path
    hook_path: Path
    installed: bool
    version: int | None
    graphify_cohabit: bool
    log_path: Path
    # v=3 baked defaults parsed from the hook file. None when absent or
    # when an older hook version (≤2) lacks the bake.
    project_dir: Path | None = None
    index_path: Path | None = None
    log_path_default: Path | None = None
    # v=6: sleep-time meta iterations baked into the script. 0 (or None
    # on older hooks) means the meta step is disabled — no bandit loop
    # runs in the background.
    meta_iterations: int | None = None

    def as_dict(self) -> dict:
        return {
            "target_repo": str(self.target_repo),
            "hook_path": str(self.hook_path),
            "installed": self.installed,
            "version": self.version,
            "graphify_cohabit": self.graphify_cohabit,
            "log_path": str(self.log_path),
            "project_dir": str(self.project_dir) if self.project_dir else None,
            "index_path": str(self.index_path) if self.index_path else None,
            "log_path_default": (
                str(self.log_path_default) if self.log_path_default else None
            ),
            "meta_iterations": self.meta_iterations,
        }


# --------------------------------------------------------------------- helpers


def _resolve_hooks_dir(repo: Path) -> Path:
    """Honour ``core.hooksPath`` so husky / lefthook coexist (RFC §6.3).

    Validates the resolved path is rooted under ``repo`` or ``$HOME`` to
    block supply-chain attacks via malicious git config.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=repo, capture_output=True, text=True, check=False, timeout=5,
        )
        configured = result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        configured = ""

    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = repo / candidate
        try:
            candidate = candidate.resolve()
            repo_resolved = repo.resolve()
            home_resolved = Path.home().resolve()
            if str(candidate).startswith((str(repo_resolved), str(home_resolved))):
                return candidate
        except (OSError, RuntimeError):  # pragma: no cover — defensive
            pass

    return (repo / ".git" / "hooks").resolve()


def _hook_path(repo: Path) -> Path:
    return _resolve_hooks_dir(repo) / "post-commit"


def _log_path() -> Path:
    return Path(os.environ.get("STROPHA_HOOK_LOG", str(Path.home() / ".cache" / "stropha-hook.log")))


def _detect_uv_path() -> str:
    """Best-effort path to the user's ``uv`` binary (hooks have stripped PATH)."""
    candidates = [
        os.environ.get("STROPHA_HOOK_UV"),
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ]
    for c in candidates:
        if c and Path(str(c)).is_file():
            return str(c)
    # Fallback: use whatever PATH gives us at install time.
    try:
        result = subprocess.run(
            ["which", "uv"], capture_output=True, text=True, check=False, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "uv"  # let runtime resolution discover via STROPHA_HOOK_UV override


# --------------------------------------------------------------------- script template


def _render_hook_block(
    target: Path,
    *,
    project_dir: Path | None = None,
    index_path: Path | None = None,
    log_path: Path | None = None,
    meta_iterations: int | None = None,
) -> str:
    """The script body sandwiched between START and END markers.

    Same logic as RFC §6.1 + the manual install: graphify update FIRST
    (no LLM, fast), then stropha index — both inside one detached background
    nohup process protected by flock.

    v=3 additions: bake ``project_dir`` / ``index_path`` / ``log_path``
    defaults into the generated script so cross-repo installs (target ≠
    where stropha lives) work without commit-time env vars. Each baked
    value is honoured unless overridden by the matching
    ``STROPHA_HOOK_*`` env var.

    v=6 addition: bake ``meta_iterations`` so the post-commit refresh
    runs N sleep-time bandit iterations after the index/backfill steps
    (WIN-203). ``None`` / ``0`` means disabled — the meta step is a
    no-op and adds nothing to the foreground.
    """
    target_str = str(target)
    project_dir_default = str(project_dir.resolve()) if project_dir else ""
    index_path_default = str(index_path.expanduser()) if index_path else ""
    log_default = str(log_path.expanduser()) if log_path else ""
    # ``meta_iterations`` of 0 / None / negative is a no-op in the script.
    # Normalize at install time so the baked value is a plain non-negative
    # integer string; the runtime test stays a cheap `[ $N -gt 0 ]`.
    meta_iters = (
        max(0, int(meta_iterations))
        if meta_iterations is not None
        else 0
    )
    meta_iterations_default = str(meta_iters) if meta_iters > 0 else ""
    # Use single-quoted heredoc semantics: every $ in the template is a literal
    # for the generated shell file (no Python interpolation).
    return dedent(
        f"""\
        {START_MARKER} v={HOOK_VERSION}
        # Post-commit: refresh graphify graph + stropha index in background.
        # Generated by `stropha hook install`. Edit overrides via env vars only.
        # graphify update runs BEFORE stropha index so the index can pick up
        # the fresh graph.json on the same commit.
        #
        # Overrides (env vars beat the baked defaults below):
        #   STROPHA_HOOK_SKIP=1         -> skip entirely (rebases)
        #   STROPHA_HOOK_TIMEOUT=600    -> wall-clock seconds for bg job
        #   STROPHA_HOOK_LOG=<path>     -> override log file
        #   STROPHA_HOOK_UV=<path>      -> override `uv` binary
        #   STROPHA_HOOK_GRAPHIFY=<p>   -> override `graphify` binary
        #   STROPHA_HOOK_NO_GRAPHIFY=1  -> skip graphify update step
        #   STROPHA_HOOK_PROJECT_DIR=<p> -> `uv run --directory` target (where
        #                                  stropha is installed). Defaults to
        #                                  the baked PROJECT_DIR_DEFAULT, else
        #                                  $TOPLEVEL (legacy dogfooding case).
        #   STROPHA_HOOK_INDEX_PATH=<p>  -> override STROPHA_INDEX_PATH for the
        #                                  index step. Defaults to the baked
        #                                  INDEX_PATH_DEFAULT, else honours
        #                                  .env in PROJECT_DIR / Config defaults.
        #   STROPHA_HOOK_META_ITERATIONS=<n> -> override the baked sleep-time
        #                                  meta loop count. 0 disables.
        #   STROPHA_HOOK_NO_META=1       -> skip the sleep-time meta step
        #                                  even when iterations > 0 (panic
        #                                  switch for when the Meta-Harness
        #                                  is misbehaving).

        # Defaults baked at install time. Re-run `stropha hook install` with
        # the appropriate flags to change. Empty string = no default, fall
        # back to legacy auto-resolution.
        PROJECT_DIR_DEFAULT="{project_dir_default}"
        INDEX_PATH_DEFAULT="{index_path_default}"
        LOG_DEFAULT="{log_default}"
        META_ITERATIONS_DEFAULT="{meta_iterations_default}"

        # --- 1. Skip conditions ----------------------------------------------
        [ "${{STROPHA_HOOK_SKIP:-0}}" = "1" ] && exit 0

        GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || exit 0
        [ -d "$GIT_DIR/rebase-merge" ]      && exit 0
        [ -d "$GIT_DIR/rebase-apply" ]      && exit 0
        [ -f "$GIT_DIR/MERGE_HEAD" ]        && exit 0
        [ -f "$GIT_DIR/CHERRY_PICK_HEAD" ]  && exit 0

        TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
        EXPECTED="{target_str}"
        [ "$TOPLEVEL" = "$EXPECTED" ] || exit 0

        CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD)
        [ -z "$CHANGED" ] && exit 0

        # --- 2. Resolve `uv` (hooks have a stripped PATH) --------------------
        UV="${{STROPHA_HOOK_UV:-}}"
        if [ -z "$UV" ]; then
            if command -v uv >/dev/null 2>&1; then
                UV=$(command -v uv)
            elif [ -x "$HOME/.local/bin/uv" ]; then
                UV="$HOME/.local/bin/uv"
            elif [ -x "/opt/homebrew/bin/uv" ]; then
                UV="/opt/homebrew/bin/uv"
            elif [ -x "/usr/local/bin/uv" ]; then
                UV="/usr/local/bin/uv"
            fi
        fi

        # --- 3. Resolve optional `timeout` -----------------------------------
        TIMEOUT_BIN=""
        if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout"; fi
        if [ -z "$TIMEOUT_BIN" ] && command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"; fi

        # --- 3b. Resolve `graphify` ------------------------------------------
        GRAPHIFY="${{STROPHA_HOOK_GRAPHIFY:-}}"
        if [ -z "$GRAPHIFY" ] && [ "${{STROPHA_HOOK_NO_GRAPHIFY:-0}}" != "1" ]; then
            if command -v graphify >/dev/null 2>&1; then
                GRAPHIFY=$(command -v graphify)
            elif [ -x "$HOME/.local/bin/graphify" ]; then
                GRAPHIFY="$HOME/.local/bin/graphify"
            elif [ -x "/opt/homebrew/bin/graphify" ]; then
                GRAPHIFY="/opt/homebrew/bin/graphify"
            elif [ -x "/usr/local/bin/graphify" ]; then
                GRAPHIFY="/usr/local/bin/graphify"
            fi
        fi

        # --- 4. Detached background run --------------------------------------
        # Honour commit-time env > install-time baked default > legacy $TOPLEVEL.
        PROJECT_DIR="${{STROPHA_HOOK_PROJECT_DIR:-${{PROJECT_DIR_DEFAULT:-$TOPLEVEL}}}}"
        INDEX_PATH_OVERRIDE="${{STROPHA_HOOK_INDEX_PATH:-$INDEX_PATH_DEFAULT}}"

        LOG="${{STROPHA_HOOK_LOG:-${{LOG_DEFAULT:-${{HOME}}/.cache/stropha-hook.log}}}}"
        LOCK="/tmp/stropha-hook-$(printf '%s' "$TOPLEVEL" | md5).lock"
        mkdir -p "$(dirname "$LOG")"

        SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo "[$(date -u +%FT%TZ)] commit ${{SHORT_SHA}} - launching refresh" >> "$LOG"

        if [ -z "$UV" ]; then
            echo "[stropha-hook] uv not found on PATH; aborting (set STROPHA_HOOK_UV)" >> "$LOG"
            exit 0
        fi

        # Inject STROPHA_TARGET_REPO + STROPHA_INDEX_PATH into the index step
        # only when the user opted in via --index-path / STROPHA_HOOK_INDEX_PATH.
        # Empty INDEX_ENV preserves the legacy behaviour (honours .env / Config
        # defaults from PROJECT_DIR).
        INDEX_ENV=""
        if [ -n "$INDEX_PATH_OVERRIDE" ]; then
            INDEX_ENV="env STROPHA_TARGET_REPO=$TOPLEVEL STROPHA_INDEX_PATH=$INDEX_PATH_OVERRIDE"
        fi

        # v=4: `--incremental` makes the hook re-chunk + re-embed only the
        # files that this commit touched. The pipeline gracefully falls
        # back to a full walk when no `last_indexed_sha` checkpoint
        # exists yet (e.g. first run after install).
        if [ -n "$TIMEOUT_BIN" ]; then
            CMD="$TIMEOUT_BIN ${{STROPHA_HOOK_TIMEOUT:-600}} $INDEX_ENV $UV run --directory $PROJECT_DIR stropha index --repo $TOPLEVEL --incremental"
        else
            CMD="$INDEX_ENV $UV run --directory $PROJECT_DIR stropha index --repo $TOPLEVEL --incremental"
        fi

        # v=5: outcome backfill (WIN-164). The previous commit SHA is the
        # baseline for the diff the labeler computes against. Skipped when
        # there is no parent (initial commit) — git rev-parse HEAD~1 fails
        # cleanly and we just don't append the backfill step.
        PREV_SHA=$(git rev-parse HEAD~1 2>/dev/null || true)
        BACKFILL_CMD=""
        if [ -n "$PREV_SHA" ]; then
            BACKFILL_CMD="$INDEX_ENV $UV run --directory $PROJECT_DIR stropha meta outcome backfill --since $PREV_SHA --repo $TOPLEVEL"
        fi

        # v=6: sleep-time bandit iterations (WIN-203). Resolved at commit
        # time so STROPHA_HOOK_META_ITERATIONS / STROPHA_HOOK_NO_META win
        # over the baked default. ``META_ITERATIONS`` ends up either a
        # positive integer (run the loop) or 0/empty (skip).
        META_ITERATIONS="${{STROPHA_HOOK_META_ITERATIONS:-$META_ITERATIONS_DEFAULT}}"
        if [ "${{STROPHA_HOOK_NO_META:-0}}" = "1" ]; then
            META_ITERATIONS=0
        fi
        META_CMD=""
        if [ -n "$META_ITERATIONS" ] && [ "$META_ITERATIONS" -gt 0 ] 2>/dev/null; then
            if [ -n "$TIMEOUT_BIN" ]; then
                META_CMD="$TIMEOUT_BIN ${{STROPHA_HOOK_TIMEOUT:-600}} $INDEX_ENV $UV run --directory $PROJECT_DIR stropha meta run --iterations $META_ITERATIONS --proposer bandit"
            else
                META_CMD="$INDEX_ENV $UV run --directory $PROJECT_DIR stropha meta run --iterations $META_ITERATIONS --proposer bandit"
            fi
        fi

        nohup sh -c "
            exec 9>'$LOCK'
            if ! flock -n 9 2>/dev/null; then
                :
            fi
            cd '$TOPLEVEL' || exit 1

            if [ -n '$GRAPHIFY' ]; then
                if [ -f '$TOPLEVEL/graphify-out/graph.json' ]; then
                    echo '[stropha-hook] graphify update (commit ${{SHORT_SHA}})' >> '$LOG'
                    '$GRAPHIFY' update '$TOPLEVEL' --no-cluster >> '$LOG' 2>&1 \\
                        || echo '[stropha-hook] graphify update failed (commit ${{SHORT_SHA}})' >> '$LOG'
                else
                    echo '[stropha-hook] graphify-out/ missing - skipping graphify update' >> '$LOG'
                fi
            else
                echo '[stropha-hook] graphify binary not found - skipping graphify update' >> '$LOG'
            fi

            $CMD >> '$LOG' 2>&1 \\
                && echo '[stropha-hook] refresh done (commit ${{SHORT_SHA}})' >> '$LOG' \\
                || echo '[stropha-hook] failed (commit ${{SHORT_SHA}})' >> '$LOG'

            if [ -n '$BACKFILL_CMD' ]; then
                echo '[stropha-hook] outcome backfill (commit ${{SHORT_SHA}})' >> '$LOG'
                $BACKFILL_CMD >> '$LOG' 2>&1 \\
                    && echo '[stropha-hook] backfill done (commit ${{SHORT_SHA}})' >> '$LOG' \\
                    || echo '[stropha-hook] backfill failed (commit ${{SHORT_SHA}})' >> '$LOG'
            fi

            if [ -n '$META_CMD' ]; then
                echo '[stropha-hook] meta loop (commit ${{SHORT_SHA}}, $META_ITERATIONS iters)' >> '$LOG'
                $META_CMD >> '$LOG' 2>&1 \\
                    && echo '[stropha-hook] meta done (commit ${{SHORT_SHA}})' >> '$LOG' \\
                    || echo '[stropha-hook] meta failed (commit ${{SHORT_SHA}})' >> '$LOG'
            fi
        " >> "$LOG" 2>&1 < /dev/null &
        disown 2>/dev/null || true
        {END_MARKER}
        """
    )


def _atomic_write(path: Path, content: str, *, executable: bool = True) -> None:
    """Write ``content`` to ``path`` via tmpfile + rename (RFC §6.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    if executable:
        st = tmp.stat()
        tmp.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    tmp.replace(path)


# --------------------------------------------------------------------- public api


def install(
    target: Path,
    *,
    force: bool = False,
    project_dir: Path | None = None,
    index_path: Path | None = None,
    log_path: Path | None = None,
    meta_iterations: int | None = None,
) -> dict:
    """Install or update the post-commit hook for ``target`` repo.

    Behaviour matrix (RFC §6.2):
      - File missing → create with shebang + our block
      - File exists, no markers → append our block at end
      - File exists, our markers → replace block in-place
      - File exists, graphify markers → warn (or proceed silently when ``force``)

    Cross-repo installs (v=3): pass ``project_dir`` to point ``uv run
    --directory`` at the directory where ``stropha`` is actually installed
    (not the ``target`` repo). Pass ``index_path`` to bake a stable
    ``STROPHA_INDEX_PATH`` for the index step. Pass ``log_path`` to send
    refresh logs somewhere other than the default
    ``~/.cache/stropha-hook.log`` — handy when you maintain hooks for
    several repos and want per-repo trails.

    Sleep-time meta loop (v=6, WIN-203): pass ``meta_iterations=N``
    (N > 0) to bake a ``stropha meta run --iterations N --proposer
    bandit`` step into the post-commit refresh. The Meta-Harness loop
    runs in the same detached background as the index refresh, so the
    foreground (the actual ``git commit`` return) stays <100 ms per
    CLAUDE.md §3.5. Pass ``None`` / ``0`` to leave the meta step out.
    Negative values are clamped to 0.

    Returns a structured outcome dict suitable for CLI rendering or logging.
    """
    target = target.resolve()
    if not (target / ".git").exists():
        raise ValueError(f"{target} is not a git repository (no .git/)")

    hook = _hook_path(target)
    block = _render_hook_block(
        target,
        project_dir=project_dir,
        index_path=index_path,
        log_path=log_path,
        meta_iterations=meta_iterations,
    )
    existing = hook.read_text(encoding="utf-8") if hook.is_file() else ""

    has_graphify = GRAPHIFY_MARKER in existing
    coexist_warning = has_graphify and not force

    if not existing:
        # New file → shebang + block
        body = "#!/bin/sh\n" + block
        action = "created"
    elif _BLOCK_RE.search(existing):
        # Replace our block in-place — handles v=2 → v=3 upgrade.
        body = _BLOCK_RE.sub(block, existing, count=1)
        action = "updated"
    else:
        # Append after existing content
        sep = "" if existing.endswith("\n") else "\n"
        body = existing + sep + "\n" + block
        action = "appended"

    _atomic_write(hook, body, executable=True)
    # Normalize the meta-iterations report so the CLI / structlog surfaces
    # a non-negative int even when the caller passed None / a negative
    # value. The script itself already clamps via the bake.
    baked_meta = (
        max(0, int(meta_iterations)) if meta_iterations is not None else 0
    )
    log.info(
        "hook.install", target=str(target), hook=str(hook), action=action,
        version=HOOK_VERSION, graphify_cohabit=has_graphify,
        project_dir=str(project_dir.resolve()) if project_dir else None,
        index_path=str(index_path.expanduser()) if index_path else None,
        log_path_baked=str(log_path.expanduser()) if log_path else None,
        meta_iterations=baked_meta,
    )

    return {
        "action": action,
        "hook_path": str(hook),
        "version": HOOK_VERSION,
        "coexist_warning": coexist_warning,
        "graphify_cohabit": has_graphify,
        "log_path": str(log_path.expanduser()) if log_path else str(_log_path()),
        "project_dir": str(project_dir.resolve()) if project_dir else None,
        "index_path": str(index_path.expanduser()) if index_path else None,
        "meta_iterations": baked_meta,
    }


def uninstall(target: Path) -> dict:
    """Remove our block. Leave any other content intact.

    If the resulting file is just ``#!/bin/sh`` (whitespace-only otherwise),
    the file is removed.
    """
    target = target.resolve()
    hook = _hook_path(target)
    if not hook.is_file():
        return {"action": "noop", "hook_path": str(hook),
                "reason": "hook file not present"}

    existing = hook.read_text(encoding="utf-8")
    if not _BLOCK_RE.search(existing):
        return {"action": "noop", "hook_path": str(hook),
                "reason": "stropha block not found"}

    new_content = _BLOCK_RE.sub("", existing)
    # Strip trailing whitespace blocks after removal.
    new_content = re.sub(r"\n{3,}", "\n\n", new_content).rstrip() + "\n"

    # If only the shebang remains, drop the file entirely.
    if new_content.strip() in ("", "#!/bin/sh", "#!/usr/bin/env sh"):
        hook.unlink()
        log.info("hook.uninstall", target=str(target), hook=str(hook), action="removed")
        return {"action": "removed", "hook_path": str(hook)}

    _atomic_write(hook, new_content, executable=True)
    log.info("hook.uninstall", target=str(target), hook=str(hook), action="block_removed")
    return {"action": "block_removed", "hook_path": str(hook)}


def status(target: Path) -> HookStatus:
    """Inspect whether the hook is installed for ``target``.

    For v=3 hooks the baked defaults (``PROJECT_DIR_DEFAULT`` /
    ``INDEX_PATH_DEFAULT`` / ``LOG_DEFAULT``) are parsed out of the script
    body so the CLI can render them. Older hooks return ``None`` for
    those fields.
    """
    target = target.resolve()
    hook = _hook_path(target)
    if not hook.is_file():
        return HookStatus(
            target_repo=target, hook_path=hook, installed=False,
            version=None, graphify_cohabit=False, log_path=_log_path(),
        )
    existing = hook.read_text(encoding="utf-8")
    m = _VERSION_RE.search(existing)
    version = int(m.group(1)) if m else None

    def _parse_baked(rx: re.Pattern[str]) -> Path | None:
        m = rx.search(existing)
        if m is None:
            return None
        value = m.group(1).strip()
        return Path(value) if value else None

    def _parse_meta_iters() -> int | None:
        """``None`` when the marker is absent (older hook < v=6), else the
        baked integer (0 when the bake is the empty-string sentinel)."""
        m = _META_ITERATIONS_RE.search(existing)
        if m is None:
            return None
        raw = m.group(1).strip()
        return int(raw) if raw else 0

    return HookStatus(
        target_repo=target, hook_path=hook,
        installed=version is not None,
        version=version,
        graphify_cohabit=GRAPHIFY_MARKER in existing,
        log_path=_log_path(),
        project_dir=_parse_baked(_PROJECT_DIR_RE),
        index_path=_parse_baked(_INDEX_PATH_RE),
        log_path_default=_parse_baked(_LOG_DEFAULT_RE),
        meta_iterations=_parse_meta_iters(),
    )
