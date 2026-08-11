"""Unit tests for git URL support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from stropha.tools.git_url import (
    CachedRepo,
    GitUrlError,
    GitUrlSearchResult,
    clone_repo,
    get_cache_dir,
    get_cached_repo,
    is_git_url,
    search_git_url,
    search_git_url_scoped,
    url_to_cache_key,
)


def _configure_git(
    monkeypatch,
    cache: Path,
    *,
    ttl: int = 86_400,
    clone_depth: int = 1,
    clone_timeout: int = 300,
) -> None:
    config = SimpleNamespace(
        git=SimpleNamespace(
            url_cache=cache,
            url_ttl=ttl,
            clone_depth=clone_depth,
            clone_timeout=clone_timeout,
        )
    )
    monkeypatch.setattr("stropha.tools.git_url.get_config", lambda: config)


class TestIsGitUrl:
    """Tests for is_git_url function."""

    def test_github_https(self):
        assert is_git_url("https://github.com/org/repo") is True
        assert is_git_url("https://github.com/org/repo.git") is True

    def test_gitlab_https(self):
        assert is_git_url("https://gitlab.com/org/repo") is True

    def test_bitbucket_https(self):
        assert is_git_url("https://bitbucket.org/org/repo") is True

    def test_git_protocol(self):
        assert is_git_url("git://github.com/org/repo.git") is True

    def test_ssh_url(self):
        assert is_git_url("git@github.com:org/repo.git") is True

    def test_not_git_url(self):
        assert is_git_url("https://example.com") is False
        assert is_git_url("/local/path") is False
        assert is_git_url("./relative") is False
        assert is_git_url("") is False

    def test_non_string(self):
        assert is_git_url(None) is False  # type: ignore
        assert is_git_url(123) is False  # type: ignore


class TestUrlToCacheKey:
    """Tests for url_to_cache_key function."""

    def test_consistent_hash(self):
        url = "https://github.com/org/repo"
        key1 = url_to_cache_key(url)
        key2 = url_to_cache_key(url)
        assert key1 == key2

    def test_normalizes_trailing_slash(self):
        key1 = url_to_cache_key("https://github.com/org/repo")
        key2 = url_to_cache_key("https://github.com/org/repo/")
        assert key1 == key2

    def test_normalizes_git_suffix(self):
        key1 = url_to_cache_key("https://github.com/org/repo")
        key2 = url_to_cache_key("https://github.com/org/repo.git")
        assert key1 == key2

    def test_case_insensitive(self):
        key1 = url_to_cache_key("https://github.com/Org/Repo")
        key2 = url_to_cache_key("https://github.com/org/repo")
        assert key1 == key2

    def test_different_urls_different_keys(self):
        key1 = url_to_cache_key("https://github.com/org/repo1")
        key2 = url_to_cache_key("https://github.com/org/repo2")
        assert key1 != key2


class TestCachedRepo:
    """Tests for CachedRepo dataclass."""

    def test_is_expired_within_ttl(self, tmp_path: Path, monkeypatch):
        import time

        _configure_git(monkeypatch, tmp_path, ttl=200)
        repo = CachedRepo(
            url="https://github.com/org/repo",
            local_path=tmp_path / "repo",
            index_path=tmp_path / "index.db",
            cloned_at=time.time() - 100,  # 100 seconds ago
            last_indexed=None,
        )
        assert repo.is_expired is False

    def test_is_expired_after_ttl(self, tmp_path: Path, monkeypatch):
        import time

        _configure_git(monkeypatch, tmp_path, ttl=50)
        repo = CachedRepo(
            url="https://github.com/org/repo",
            local_path=tmp_path / "repo",
            index_path=tmp_path / "index.db",
            cloned_at=time.time() - 100,
            last_indexed=None,
        )
        assert repo.is_expired is True

    def test_needs_index_no_last_indexed(self, tmp_path: Path):
        import time

        repo = CachedRepo(
            url="https://github.com/org/repo",
            local_path=tmp_path / "repo",
            index_path=tmp_path / "index.db",
            cloned_at=time.time(),
            last_indexed=None,
        )
        assert repo.needs_index is True

    def test_needs_index_no_db(self, tmp_path: Path):
        import time

        repo = CachedRepo(
            url="https://github.com/org/repo",
            local_path=tmp_path / "repo",
            index_path=tmp_path / "nonexistent.db",
            cloned_at=time.time(),
            last_indexed=time.time(),
        )
        assert repo.needs_index is True

    def test_does_not_need_index(self, tmp_path: Path):
        import time

        index_path = tmp_path / "index.db"
        index_path.touch()

        repo = CachedRepo(
            url="https://github.com/org/repo",
            local_path=tmp_path / "repo",
            index_path=index_path,
            cloned_at=time.time(),
            last_indexed=time.time(),
        )
        assert repo.needs_index is False


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_creates_directory(self, tmp_path: Path, monkeypatch):
        _configure_git(monkeypatch, tmp_path / "cache")
        cache_dir = get_cache_dir()
        assert cache_dir.is_dir()

    def test_returns_existing_directory(self, tmp_path: Path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()

        _configure_git(monkeypatch, cache)
        assert get_cache_dir() == cache


class TestGetCachedRepo:
    """Tests for get_cached_repo function."""

    def test_returns_none_when_not_cached(self, tmp_path: Path, monkeypatch):
        _configure_git(monkeypatch, tmp_path)
        assert get_cached_repo("https://github.com/org/repo") is None

    def test_returns_cached_repo(self, tmp_path: Path, monkeypatch):
        import time

        url = "https://github.com/org/repo"
        cache_key = url_to_cache_key(url)
        repo_dir = tmp_path / cache_key
        repo_dir.mkdir()
        (repo_dir / "repo").mkdir()

        meta = {
            "url": url,
            "cloned_at": time.time(),
            "last_indexed": time.time(),
        }
        (repo_dir / ".stropha-meta").write_text(json.dumps(meta))

        _configure_git(monkeypatch, tmp_path)
        result = get_cached_repo(url)

        assert result is not None
        assert result.url == url

    def test_returns_none_on_invalid_meta(self, tmp_path: Path, monkeypatch):
        url = "https://github.com/org/repo"
        cache_key = url_to_cache_key(url)
        repo_dir = tmp_path / cache_key
        repo_dir.mkdir()
        (repo_dir / ".stropha-meta").write_text("invalid json")

        _configure_git(monkeypatch, tmp_path)
        result = get_cached_repo(url)

        assert result is None


class TestCloneRepo:
    """Tests for clone_repo function."""

    def test_uses_cached_when_valid(self, tmp_path: Path, monkeypatch):
        import time

        url = "https://github.com/org/repo"
        cache_key = url_to_cache_key(url)
        repo_dir = tmp_path / cache_key
        repo_dir.mkdir()
        (repo_dir / "repo").mkdir()

        meta = {
            "url": url,
            "cloned_at": time.time(),
            "last_indexed": None,
        }
        (repo_dir / ".stropha-meta").write_text(json.dumps(meta))

        _configure_git(monkeypatch, tmp_path)
        result = clone_repo(url)

        assert result.url == url

    def test_clones_when_not_cached(self, tmp_path: Path, monkeypatch):
        url = "https://github.com/org/repo"

        mock_result = MagicMock()
        mock_result.returncode = 0

        _configure_git(monkeypatch, tmp_path, clone_depth=7, clone_timeout=42)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = clone_repo(url)

        assert result.url == url
        assert mock_run.call_args.args[0] == [
            "git",
            "clone",
            "--depth",
            "7",
            url,
            str(tmp_path / url_to_cache_key(url) / "repo"),
        ]
        assert mock_run.call_args.kwargs["timeout"] == 42

    def test_raises_on_clone_failure(self, tmp_path: Path, monkeypatch):
        url = "https://github.com/org/repo"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: repository not found"

        _configure_git(monkeypatch, tmp_path)
        with (
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(GitUrlError, match="git clone failed"),
        ):
            clone_repo(url)

    def test_refresh_reclones(self, tmp_path: Path, monkeypatch):
        import time

        url = "https://github.com/org/repo"
        cache_key = url_to_cache_key(url)
        repo_dir = tmp_path / cache_key
        repo_dir.mkdir()
        (repo_dir / "repo").mkdir()

        meta = {
            "url": url,
            "cloned_at": time.time(),
            "last_indexed": None,
        }
        (repo_dir / ".stropha-meta").write_text(json.dumps(meta))

        mock_result = MagicMock()
        mock_result.returncode = 0

        _configure_git(monkeypatch, tmp_path)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            clone_repo(url, refresh=True)

        # Should have called git clone because refresh=True
        mock_run.assert_called_once()


def test_search_git_url_scopes_retrieval_to_cached_repo(tmp_path: Path) -> None:
    cached = CachedRepo(
        url="https://github.com/org/repo",
        local_path=tmp_path / "repo",
        index_path=tmp_path / "index.db",
        cloned_at=0.0,
        last_indexed=1.0,
    )
    cached.local_path.mkdir()
    cached.index_path.touch()
    storage = MagicMock()
    storage.lookup_repo_id.return_value = 23
    retrieval = MagicMock()
    retrieval.search.return_value = []
    built = MagicMock(storage=storage, retrieval=retrieval)

    with (
        patch("stropha.tools.git_url.clone_repo", return_value=cached),
        patch("stropha.pipeline.config.load_pipeline_config", return_value=MagicMock()),
        patch("stropha.pipeline.builder.build_stages", return_value=built),
        patch(
            "stropha.ingest.git_meta.detect",
            return_value=SimpleNamespace(normalized_key="github.com/org/repo"),
        ),
    ):
        result = search_git_url_scoped("auth", cached.url, top_k=4)

    assert result.hits == []
    assert result.repo_id == 23
    retrieval.search.assert_called_once_with("auth", top_k=4, repo_id=23)


def test_search_git_url_preserves_public_hit_list_contract(tmp_path: Path) -> None:
    hits = [MagicMock()]
    with patch(
        "stropha.tools.git_url.search_git_url_scoped",
        return_value=GitUrlSearchResult(hits=hits, repo_id=23),
    ):
        result = search_git_url("auth", "https://github.com/org/repo")

    assert result is hits


def test_search_git_url_forwards_recursive_override() -> None:
    hits = [MagicMock()]
    with patch(
        "stropha.tools.git_url.search_git_url_scoped",
        return_value=GitUrlSearchResult(hits=hits, repo_id=23),
    ) as scoped:
        result = search_git_url(
            "auth",
            "https://github.com/org/repo",
            recursive=True,
        )

    assert result is hits
    scoped.assert_called_once_with(
        "auth",
        "https://github.com/org/repo",
        top_k=10,
        refresh=False,
        recursive=True,
    )
