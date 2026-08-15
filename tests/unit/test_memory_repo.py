"""Multi-repo memory binding (Problem 2).

Memory tools accept an optional ``repo`` (normalized key / local path / None)
so a multi-repo agent can anchor memory to the repo it is actually reasoning
about instead of always the server's ``target_repo``. Staleness must diff the
*entry's own* repo tree, not the target.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from stropha.server_memory_tools import (
    _repo_root_path,
    _resolve_repo,
    check_staleness_impl,
    co_occurrence_impl,
    forget_impl,
    recall_episodes_impl,
    recall_impl,
    record_episode_impl,
    remember_impl,
)

REPO = "github.com/org/mimoria"


def _mem_rec() -> SimpleNamespace:
    return SimpleNamespace(
        kind="fact",
        key="billing",
        value="tenants map to stripe customers",
        source="sess",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        expires_at=None,
        provenance_commit="abc123",
    )


def _episode_rec() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        title="wired stripe webhook",
        summary="",
        session_id="sess",
        touched_files=["billing.py"],
        touched_symbols=["handle_webhook"],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        provenance_commit="abc123",
    )


def _capturing_resolve(captured: dict, repo_id: int | None = 7):
    def _resolve(app, *, repo=None):
        captured["repo"] = repo
        return repo_id

    return _resolve


def _capturing_head(captured: dict, head: str | None = "abc123"):
    def _head(app, *, repo=None):
        captured["head_repo"] = repo
        return head

    return _head


# --------------------------------------------------------------------------- #
# threading: every memory impl forwards ``repo`` to the resolver              #
# --------------------------------------------------------------------------- #
def test_remember_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.remember.return_value = _mem_rec()

    resp = remember_impl(
        "fact",
        "billing",
        "v",
        app,
        expires_at=None,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        current_head_commit_fn=_capturing_head(cap),
        repo=REPO,
    )

    assert resp.ok is True
    assert cap["repo"] == REPO
    assert cap["head_repo"] == REPO
    assert app.storage.remember.call_args.kwargs["repo_id"] == 7
    assert app.storage.remember.call_args.kwargs["provenance_commit"] == "abc123"


def test_recall_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.recall.return_value = [_mem_rec()]
    app.storage._conn.execute.return_value.fetchone.return_value = {
        "normalized_key": REPO
    }

    resp = recall_impl(
        app,
        kind=None,
        key=None,
        limit=10,
        include_expired=False,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        repo=REPO,
    )

    assert cap["repo"] == REPO
    assert app.storage.recall.call_args.kwargs["repo_id"] == 7
    assert len(resp.entries) == 1
    assert resp.repo == REPO


def test_forget_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.forget.return_value = True

    resp = forget_impl(
        "fact",
        "billing",
        app,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        repo=REPO,
    )

    assert cap["repo"] == REPO
    assert resp.removed is True
    assert app.storage.forget.call_args.kwargs["repo_id"] == 7


def test_record_episode_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.record_episode.return_value = _episode_rec()

    resp = record_episode_impl(
        "wired stripe webhook",
        "",
        ["billing.py"],
        ["handle_webhook"],
        app,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        current_head_commit_fn=_capturing_head(cap),
        repo=REPO,
    )

    assert resp.ok is True
    assert cap["repo"] == REPO
    assert cap["head_repo"] == REPO
    assert app.storage.record_episode.call_args.kwargs["repo_id"] == 7


def test_recall_episodes_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.recall_episodes.return_value = [_episode_rec()]
    app.storage._conn.execute.return_value.fetchone.return_value = {
        "normalized_key": REPO
    }

    resp = recall_episodes_impl(
        app,
        query="stripe",
        limit=10,
        since=None,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        repo=REPO,
    )

    assert cap["repo"] == REPO
    assert app.storage.recall_episodes.call_args.kwargs["repo_id"] == 7
    assert resp.repo == REPO


def test_co_occurrence_threads_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.episode_co_occurrence.return_value = {
        "files": {"billing.py": 3},
        "symbols": {},
    }
    app.storage._conn.execute.return_value.fetchone.return_value = {
        "normalized_key": REPO
    }

    resp = co_occurrence_impl(
        "handle_webhook",
        app,
        limit=10,
        in_files=True,
        in_symbols=True,
        resolve_target_repo_id_fn=_capturing_resolve(cap),
        repo=REPO,
    )

    assert cap["repo"] == REPO
    assert app.storage.episode_co_occurrence.call_args.kwargs["repo_id"] == 7
    assert resp.files == {"billing.py": 3}


# --------------------------------------------------------------------------- #
# staleness diffs the entry's OWN repo, not target_repo                       #
# --------------------------------------------------------------------------- #
def test_check_staleness_diffs_entry_repo() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.recall.return_value = [_mem_rec()]
    # _repo_root_path -> the entry's repo tree
    app.storage._conn.execute.return_value.fetchone.return_value = {
        "root_path": "/tmp/mimoria"
    }
    signal = SimpleNamespace(
        stale=True,
        since_commit="abc123",
        current_commit="def456",
        changed_files=["billing.py"],
        reason="anchor changed",
    )

    with patch("stropha.meta.provenance.staleness_signal", return_value=signal) as m:
        resp = check_staleness_impl(
            "fact",
            "billing",
            app,
            anchor_files=["billing.py"],
            resolve_target_repo_id_fn=_capturing_resolve(cap, repo_id=9),
            repo=REPO,
        )

    assert cap["repo"] == REPO
    # diffed the entry's repo tree + the entry's own commit
    called_path, kwargs = m.call_args.args[0], m.call_args.kwargs
    assert called_path == Path("/tmp/mimoria")
    assert kwargs["since_commit"] == "abc123"
    assert resp.found is True
    assert resp.stale is True
    assert resp.changed_files == ["billing.py"]


def test_check_staleness_unknown_repo_path() -> None:
    cap: dict = {}
    app = MagicMock()
    app.storage.recall.return_value = [_mem_rec()]
    app.storage._conn.execute.return_value.fetchone.return_value = None  # no root_path

    resp = check_staleness_impl(
        "fact",
        "billing",
        app,
        anchor_files=None,
        resolve_target_repo_id_fn=_capturing_resolve(cap, repo_id=9),
        repo=REPO,
    )

    assert resp.found is True
    assert resp.stale is None
    assert resp.since_commit == "abc123"


def test_check_staleness_no_repo() -> None:
    cap: dict = {}
    app = MagicMock()

    resp = check_staleness_impl(
        "fact",
        "billing",
        app,
        anchor_files=None,
        resolve_target_repo_id_fn=_capturing_resolve(cap, repo_id=None),
        repo="does/not/exist",
    )

    assert resp.found is False


# --------------------------------------------------------------------------- #
# _resolve_repo: key / path / default                                         #
# --------------------------------------------------------------------------- #
def test_resolve_repo_by_key() -> None:
    app = MagicMock()
    app.storage.lookup_repo_id.return_value = 5
    app.storage._conn.execute.return_value.fetchone.return_value = {
        "root_path": "/tmp/mimoria"
    }

    repo_id, local = _resolve_repo(app, REPO)

    assert repo_id == 5
    assert local == Path("/tmp/mimoria")
    app.storage.lookup_repo_id.assert_called_once_with(REPO)


def test_resolve_repo_by_path() -> None:
    app = MagicMock()
    app.storage.lookup_repo_id.return_value = 7

    with patch(
        "stropha.ingest.git_meta.detect",
        return_value=SimpleNamespace(normalized_key=REPO),
    ):
        repo_id, local = _resolve_repo(app, "/tmp/somewhere")

    assert repo_id == 7
    assert local == Path("/tmp/somewhere")
    app.storage.lookup_repo_id.assert_called_once_with(REPO)


def test_resolve_repo_treats_relative_path_as_exact_key() -> None:
    app = MagicMock()
    app.storage.lookup_repo_id.return_value = None

    with patch("stropha.ingest.git_meta.detect") as detect:
        repo_id, local = _resolve_repo(app, "src/stropha")

    assert repo_id is None
    assert local is None
    detect.assert_not_called()
    app.storage.lookup_repo_id.assert_called_once_with("src/stropha")


def test_resolve_repo_defaults_to_target() -> None:
    app = MagicMock()
    app.config.target_repo = "/tmp/target"
    app.storage.lookup_repo_id.return_value = 3

    with patch(
        "stropha.ingest.git_meta.detect",
        return_value=SimpleNamespace(normalized_key=REPO),
    ):
        repo_id, local = _resolve_repo(app, None)

    assert repo_id == 3
    assert local == Path("/tmp/target")


def test_resolve_repo_no_target() -> None:
    app = MagicMock()
    app.config.target_repo = None

    repo_id, local = _resolve_repo(app, None)

    assert repo_id is None
    assert local is None


def test_repo_root_path_reads_repos_table() -> None:
    storage = MagicMock()
    storage._conn.execute.return_value.fetchone.return_value = {"root_path": "/tmp/x"}

    assert _repo_root_path(storage, 4) == "/tmp/x"

    storage._conn.execute.return_value.fetchone.return_value = None
    assert _repo_root_path(storage, 4) is None
