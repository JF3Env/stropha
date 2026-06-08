"""``index_health`` MCP tool (WIN-348, EPIC 4.6).

A self-check the calling agent can use to decide how much to trust the index
instead of assuming it blindly: per-repo coverage% (indexed files / tracked
files), last-index lag, commits behind HEAD, and whether the active
embedder/enricher config has drifted from what last wrote the index.

The work lives in the pure function :func:`index_health_impl` so it is
unit-test friendly — ``now_fn`` and ``git_runner`` are injectable. The MCP
tool in ``server.py`` is a thin wrapper.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

# A git runner takes (args, cwd) and returns stdout, or None on any failure
# (missing git, non-repo, bad sha, timeout). Injected for tests.
GitRunner = Callable[[list[str], str], "str | None"]


def _default_git_runner(args: list[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _lag_seconds(now: datetime, last_at: str | None) -> float | None:
    """Seconds between ``now`` and an ISO timestamp; None if unparseable."""
    if not last_at:
        return None
    try:
        parsed = datetime.fromisoformat(last_at)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return round((now - parsed).total_seconds(), 1)


def _tracked_file_count(local_path: str | None, run_git: GitRunner) -> int | None:
    if not local_path:
        return None
    out = run_git(["ls-files"], local_path)
    if out is None:
        return None
    return sum(1 for line in out.splitlines() if line.strip())


def _commits_behind(
    local_path: str | None, since_sha: str | None, run_git: GitRunner
) -> int | None:
    if not (local_path and since_sha):
        return None
    out = run_git(["rev-list", "--count", f"{since_sha}..HEAD"], local_path)
    if out is None:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def index_health_impl(
    app: Any,
    *,
    now_fn: Callable[[], datetime] | None = None,
    git_runner: GitRunner | None = None,
) -> dict[str, Any]:
    """Per-repo freshness/coverage + config-drift signals for the live index."""
    now = (now_fn or _utcnow)()
    run_git = git_runner or _default_git_runner
    storage = app.storage
    built = app.built

    active_enricher = getattr(built.enricher, "adapter_id", None)
    active_embedder = getattr(built.embedder, "adapter_id", None)
    stored_enricher = storage.get_meta("active_enricher_id")
    stored_embedder = storage.get_meta("active_embedder_id")
    # Drift only when something was previously recorded AND it differs — a
    # fresh index (no stored id) is not "drifted".
    enricher_drift = stored_enricher is not None and stored_enricher != active_enricher
    embedder_drift = stored_embedder is not None and stored_embedder != active_embedder

    repos_out: list[dict[str, Any]] = []
    for r in storage.list_repos():
        repo_id = storage.lookup_repo_id(r.normalized_key)
        indexed_sha = (
            storage.get_meta(f"last_indexed_sha_{repo_id}")
            if repo_id is not None
            else None
        )
        tracked = _tracked_file_count(r.local_path, run_git)
        coverage = None
        if tracked and r.files is not None and tracked > 0:
            coverage = round(100.0 * r.files / tracked, 1)
        repos_out.append(
            {
                "repo": r.normalized_key,
                "files_indexed": r.files,
                "chunks": r.chunks,
                "tracked_files": tracked,
                "coverage_pct": coverage,
                "last_indexed_at": r.last_indexed_at,
                "lag_seconds": _lag_seconds(now, r.last_indexed_at),
                "indexed_commit": indexed_sha,
                "head_commit": r.head_commit,
                "commits_behind": _commits_behind(r.local_path, indexed_sha, run_git),
            }
        )

    notes: list[str] = []
    if enricher_drift or embedder_drift:
        notes.append(
            "config drift: the active embedder/enricher differs from what last "
            "indexed — a reindex will re-embed/re-enrich affected chunks."
        )
    behind = [x for x in repos_out if (x["commits_behind"] or 0) > 0]
    if behind:
        notes.append(
            f"{len(behind)} repo(s) behind HEAD — run `stropha index` to catch up."
        )

    return {
        "storage": getattr(built.storage, "adapter_name", "?"),
        "embedder": active_embedder,
        "enricher": active_enricher,
        "config_drift": bool(enricher_drift or embedder_drift),
        "drift_detail": {
            "active_enricher_id": active_enricher,
            "indexed_enricher_id": stored_enricher,
            "enricher_drifted": enricher_drift,
            "active_embedder_id": active_embedder,
            "indexed_embedder_id": stored_embedder,
            "embedder_drifted": embedder_drift,
        },
        "repos": repos_out,
        "notes": notes,
    }


def build_index_watermark(
    storage: Any,
    repo_keys: Any,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> list[Any]:
    """Per-repo freshness stamps (WIN-349) for the repos in a search result set.

    Cheap by design: one ``list_repos()`` read, no git per query. Returns
    ``IndexWatermark`` rows only for the repos in ``repo_keys``; empty when
    there is nothing to stamp. The deep commits-behind check stays in
    ``index_health``.
    """
    wanted = {k for k in repo_keys if k}
    if not wanted:
        return []
    from .server_models import IndexWatermark

    now = (now_fn or _utcnow)()
    out: list[Any] = []
    for r in storage.list_repos():
        if r.normalized_key in wanted:
            out.append(
                IndexWatermark(
                    repo=r.normalized_key,
                    indexed_commit=r.head_commit,
                    last_indexed_at=r.last_indexed_at,
                    age_seconds=_lag_seconds(now, r.last_indexed_at),
                )
            )
    return out
