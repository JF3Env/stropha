"""Tests for `indexing.blacklist_repos` — the never-index-again list."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stropha.config import repo_is_blacklisted
from stropha.config.models import Config, IndexingConfig
from stropha.errors import RepoBlacklistedError
from stropha.ingest.git_meta import RepoIdentity


def _config(*entries: str) -> Config:
    return dataclasses.replace(Config(), indexing=IndexingConfig(blacklist_repos=entries))


def _identity(key: str, root: str) -> RepoIdentity:
    return RepoIdentity(
        normalized_key=key,
        remote_url=f"https://{key}.git",
        root_path=Path(root),
        default_branch="main",
        head_commit="abc123",
    )


def test_blacklist_is_empty_by_default() -> None:
    assert Config().indexing.blacklist_repos == ()
    assert repo_is_blacklisted("github.com/acme/app", "/src/app", config=_config()) is False


@pytest.mark.parametrize(
    "entry",
    [
        "github.com/acme/app",  # normalized key
        "/src/app",  # repository root path
        "path:/src/app",  # fallback identity key
        "local:/src/app",  # legacy fallback identity key
        "/src/app/",  # same path, spelled with a trailing slash
    ],
)
def test_any_spelling_of_a_repository_blocks_it(entry) -> None:
    config = _config(entry)
    assert repo_is_blacklisted("github.com/acme/app", "/src/app", config=config) is True


@pytest.mark.parametrize("entry", ["/src/app", "path:/src/app", "/src/app/"])
def test_a_path_entry_blocks_the_checkout_under_any_identity(entry) -> None:
    """A renamed remote must not re-open a blocked checkout."""
    config = _config(entry)
    for key in ("github.com/acme/app", "github.com/acme/renamed", "path:/src/app"):
        assert repo_is_blacklisted(key, "/src/app", config=config) is True


def test_blacklist_leaves_other_repositories_alone() -> None:
    config = _config("github.com/acme/app")
    assert repo_is_blacklisted("github.com/acme/other", "/src/other", config=config) is False
    assert repo_is_blacklisted("github.com/acme/app-tools", "/src/tools", config=config) is False


def test_indexing_a_blocked_repository_raises() -> None:
    """The check sits at the funnel every entry point goes through."""
    from stropha.pipeline.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    identity = _identity("github.com/acme/app", "/src/app")
    with (
        patch("stropha.pipeline.pipeline.detect_repo", return_value=identity),
        patch("stropha.config.get_config", return_value=_config("github.com/acme/app")),
        pytest.raises(RepoBlacklistedError) as raised,
    ):
        pipeline._index_one_repo(Path("/src/app"))

    assert raised.value.repo == "github.com/acme/app"
    assert "blacklist_repos" in str(raised.value)


def test_a_blocked_repository_never_reaches_storage() -> None:
    from stropha.pipeline.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._storage = MagicMock()
    identity = _identity("github.com/acme/app", "/src/app")
    with (
        patch("stropha.pipeline.pipeline.detect_repo", return_value=identity),
        patch("stropha.config.get_config", return_value=_config("/src/app")),
        pytest.raises(RepoBlacklistedError),
    ):
        pipeline._index_one_repo(Path("/src/app"))

    pipeline._storage.register_repo.assert_not_called()


def test_blacklist_parses_from_the_configuration_document(tmp_path, monkeypatch) -> None:
    from stropha.config import load_config

    document = tmp_path / "config.yaml"
    document.write_text(
        "indexing:\n  blacklist_repos:\n    - github.com/acme/app\n    - '  '\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path.parent))
    monkeypatch.setattr("stropha.config._CONFIG_PATH", document, raising=False)
    with patch(
        "stropha.config._load_yaml",
        return_value={"indexing": {"blacklist_repos": ["github.com/acme/app", "  "]}},
    ):
        config = load_config()
    # Blank entries are dropped rather than blocking everything.
    assert config.indexing.blacklist_repos == ("github.com/acme/app",)
