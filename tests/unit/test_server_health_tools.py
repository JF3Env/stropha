"""Tests for the index_health MCP tool impl (WIN-348)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from stropha.models import RepoStats
from stropha.server_health_tools import _lag_seconds, index_health_impl


def _repo(**kw: Any) -> RepoStats:
    base: dict[str, Any] = dict(
        normalized_key="github.com/x/repo",
        url=None,
        local_path="/repo",
        default_branch="main",
        head_commit="HEAD1",
        files=10,
        chunks=100,
        last_indexed_at="2026-06-08T07:00:00+00:00",
    )
    base.update(kw)
    return RepoStats(**base)


class _FakeStorage:
    def __init__(self, repos, meta, ids):
        self._repos, self._meta, self._ids = repos, meta, ids

    def list_repos(self):
        return self._repos

    def get_meta(self, key):
        return self._meta.get(key)

    def lookup_repo_id(self, key):
        return self._ids.get(key)


def _app(storage, *, enricher="contextual:1", embedder="voyage:512", storage_name="postgres"):
    built = SimpleNamespace(
        enricher=SimpleNamespace(adapter_id=enricher),
        embedder=SimpleNamespace(adapter_id=embedder),
        storage=SimpleNamespace(adapter_name=storage_name),
    )
    return SimpleNamespace(storage=storage, built=built)


def _git(ls_files_n: int = 10, behind: str = "0"):
    def run(args, cwd):
        if args[:1] == ["ls-files"]:
            return "\n".join(f"f{i}.py" for i in range(ls_files_n)) + "\n"
        if args[:2] == ["rev-list", "--count"]:
            return behind + "\n"
        return None

    return run


def _now() -> datetime:
    return datetime(2026, 6, 8, 8, 0, 0, tzinfo=UTC)


def test_index_health_fresh_repo() -> None:
    st = _FakeStorage(
        [_repo()],
        meta={
            "active_enricher_id": "contextual:1",
            "active_embedder_id": "voyage:512",
            "last_indexed_sha_1": "HEAD1",
        },
        ids={"github.com/x/repo": 1},
    )
    out = index_health_impl(_app(st), now_fn=_now, git_runner=_git(ls_files_n=10, behind="0"))
    assert out["config_drift"] is False
    assert out["storage"] == "postgres"
    repo = out["repos"][0]
    assert repo["files_indexed"] == 10
    assert repo["tracked_files"] == 10
    assert repo["coverage_pct"] == 100.0
    assert repo["lag_seconds"] == 3600.0
    assert repo["commits_behind"] == 0
    assert repo["indexed_commit"] == "HEAD1"
    assert out["notes"] == []


def test_index_health_config_drift_flagged() -> None:
    st = _FakeStorage(
        [_repo()],
        meta={
            "active_enricher_id": "contextual:OLD",
            "active_embedder_id": "voyage:512",
            "last_indexed_sha_1": "HEAD1",
        },
        ids={"github.com/x/repo": 1},
    )
    out = index_health_impl(_app(st, enricher="contextual:NEW"), now_fn=_now, git_runner=_git())
    assert out["config_drift"] is True
    assert out["drift_detail"]["enricher_drifted"] is True
    assert out["drift_detail"]["embedder_drifted"] is False
    assert any("config drift" in n for n in out["notes"])


def test_index_health_commits_behind() -> None:
    st = _FakeStorage(
        [_repo()],
        meta={"last_indexed_sha_1": "OLDSHA"},  # no active_* -> no drift
        ids={"github.com/x/repo": 1},
    )
    out = index_health_impl(_app(st), now_fn=_now, git_runner=_git(behind="3"))
    assert out["repos"][0]["commits_behind"] == 3
    assert any("behind HEAD" in n for n in out["notes"])
    assert out["config_drift"] is False


def test_index_health_git_unavailable_is_graceful() -> None:
    st = _FakeStorage([_repo(local_path=None)], meta={}, ids={"github.com/x/repo": 1})

    def no_git(args, cwd):
        return None

    out = index_health_impl(_app(st), now_fn=_now, git_runner=no_git)
    repo = out["repos"][0]
    assert repo["tracked_files"] is None
    assert repo["coverage_pct"] is None
    assert repo["commits_behind"] is None


def test_lag_seconds_handles_bad_input() -> None:
    assert _lag_seconds(_now(), None) is None
    assert _lag_seconds(_now(), "not-a-date") is None
    # naive timestamp is assumed UTC
    assert _lag_seconds(_now(), "2026-06-08T07:00:00") == 3600.0
