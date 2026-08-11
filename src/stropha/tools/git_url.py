"""Git URL support for stropha search.

Allows searching remote repositories via git URL clone-and-index.

Usage:
    stropha search "auth" https://github.com/x/y
    stropha search "auth" --repo-url https://github.com/x/y --refresh

The repository is cloned to ~/.cache/stropha/repos/<url-hash>/ and indexed
on first access. Subsequent searches use the cached index unless --refresh
is passed or the cache TTL (24h) has expired.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from stropha.config import get_config
from stropha.errors import StrophaError
from stropha.models import SearchHit

logger = structlog.get_logger(__name__)


class GitUrlError(StrophaError):
    """Error handling git URL operations."""


@dataclass
class CachedRepo:
    """A cached repository from a git URL."""

    url: str
    local_path: Path
    index_path: Path
    cloned_at: float  # Unix timestamp
    last_indexed: float | None  # Unix timestamp

    @property
    def is_expired(self) -> bool:
        """Check if the cache has expired (TTL exceeded)."""
        ttl = get_config().git.url_ttl
        return (time.time() - self.cloned_at) > ttl

    @property
    def needs_index(self) -> bool:
        """Check if the repo needs indexing."""
        return self.last_indexed is None or not self.index_path.exists()


@dataclass(frozen=True)
class GitUrlSearchResult:
    """Remote hits plus the exact repository row used for retrieval."""

    hits: list[SearchHit]
    repo_id: int


def url_to_cache_key(url: str) -> str:
    """Generate a cache key from a git URL.

    Uses a truncated SHA256 hash of the normalized URL.
    """
    # Normalize URL: remove trailing .git, lowercase
    normalized = url.rstrip("/").removesuffix(".git").lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_cache_dir() -> Path:
    """Get the cache directory for git URL repos."""
    cache_dir = get_config().git.url_cache.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_repo(url: str) -> CachedRepo | None:
    """Get a cached repository if it exists and is valid.

    Returns None if:
    - The repo is not cached
    - The cache has expired
    """
    cache_key = url_to_cache_key(url)
    cache_dir = get_cache_dir()
    repo_dir = cache_dir / cache_key

    if not repo_dir.exists():
        return None

    # Read metadata
    meta_file = repo_dir / ".stropha-meta"
    if not meta_file.exists():
        return None

    try:
        import json

        meta = json.loads(meta_file.read_text())
        return CachedRepo(
            url=meta["url"],
            local_path=repo_dir / "repo",
            index_path=repo_dir / "index.db",
            cloned_at=meta["cloned_at"],
            last_indexed=meta.get("last_indexed"),
        )
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("invalid_cache_meta", url=url, error=str(exc))
        return None


def clone_repo(url: str, *, refresh: bool = False) -> CachedRepo:
    """Clone a git repository to the cache.

    Args:
        url: Git URL to clone
        refresh: If True, re-clone even if cached

    Returns:
        CachedRepo with the cloned repository

    Raises:
        GitUrlError: If cloning fails
    """
    cache_key = url_to_cache_key(url)
    cache_dir = get_cache_dir()
    repo_dir = cache_dir / cache_key
    local_path = repo_dir / "repo"
    index_path = repo_dir / "index.db"
    meta_file = repo_dir / ".stropha-meta"

    # Check if we have a valid cache
    existing = get_cached_repo(url)
    if existing and not refresh and not existing.is_expired:
        logger.info("using_cached_repo", url=url, path=str(existing.local_path))
        return existing

    # Clear existing cache if refreshing or expired
    if repo_dir.exists():
        logger.info("clearing_expired_cache", url=url)
        shutil.rmtree(repo_dir)

    # Create directory structure
    repo_dir.mkdir(parents=True, exist_ok=True)

    git_config = get_config().git

    # Clone the repository
    logger.info("cloning_repo", url=url, target=str(local_path))
    try:
        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                str(git_config.clone_depth),
                url,
                str(local_path),
            ],
            capture_output=True,
            text=True,
            timeout=git_config.clone_timeout,
        )
        if result.returncode != 0:
            raise GitUrlError(f"git clone failed: {result.stderr}")
    except subprocess.TimeoutExpired as exc:
        raise GitUrlError(f"git clone timed out after {git_config.clone_timeout} seconds") from exc
    except FileNotFoundError as exc:
        raise GitUrlError("git is not installed or not in PATH") from exc

    # Write metadata
    import json

    cloned_at = time.time()
    meta = {
        "url": url,
        "cloned_at": cloned_at,
        "last_indexed": None,
    }
    meta_file.write_text(json.dumps(meta))

    return CachedRepo(
        url=url,
        local_path=local_path,
        index_path=index_path,
        cloned_at=cloned_at,
        last_indexed=None,
    )


def index_cached_repo(cached: CachedRepo) -> None:
    """Index a cached repository.

    Updates the cache metadata with the indexing timestamp.
    """
    from stropha.config import get_config
    from stropha.pipeline.builder import build_stages
    from stropha.pipeline.config import load_pipeline_config
    from stropha.pipeline.pipeline import Pipeline

    logger.info("indexing_cached_repo", url=cached.url, path=str(cached.local_path))

    # Tell the unified config which repo to walk. The per-cache DB path
    # used to be propagated via STROPHA_INDEX_PATH but that env var was
    # never read by any loader (removed in v=9). The cached repo is now
    # indexed into the unified YAML `paths.index` DB; multi-repo
    # tracking inside that DB keeps the per-clone chunks separate.
    os.environ["STROPHA_TARGET_REPO"] = str(cached.local_path)

    try:
        resolved = load_pipeline_config()
        built = build_stages(resolved)
        inference = get_config().inference
        with built.storage as storage:  # type: ignore[union-attr]
            pipeline = Pipeline(
                storage=storage,
                embedder=built.embedder,
                secondary_embedder=built.secondary_embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=[cached.local_path],
                enricher_concurrency=inference.enricher_concurrency,
                enricher_batch_size=inference.effective_batch_size(),
            )
            stats = pipeline.run(rebuild=False)

        logger.info(
            "indexed_cached_repo",
            url=cached.url,
            files=stats.files_visited,
            chunks=stats.chunks_embedded,
        )
    finally:
        # Clean up environment
        os.environ.pop("STROPHA_TARGET_REPO", None)

    # Update metadata
    meta_file = cached.local_path.parent / ".stropha-meta"
    import json

    meta = json.loads(meta_file.read_text())
    meta["last_indexed"] = time.time()
    meta_file.write_text(json.dumps(meta))

    # Update the cached object
    object.__setattr__(cached, "last_indexed", meta["last_indexed"])


def search_git_url(
    query: str,
    url: str,
    *,
    top_k: int = 10,
    refresh: bool = False,
    recursive: bool = False,
) -> list[SearchHit]:
    """Search a remote repository and return the legacy hit list."""
    return search_git_url_scoped(
        query,
        url,
        top_k=top_k,
        refresh=refresh,
        recursive=recursive,
    ).hits


def search_git_url_scoped(
    query: str,
    url: str,
    *,
    top_k: int = 10,
    refresh: bool = False,
    recursive: bool = False,
) -> GitUrlSearchResult:
    """Search a remote repository and retain its exact repository id.

    Args:
        query: Search query
        url: Git URL to search
        top_k: Number of results to return
        refresh: If True, re-clone and re-index
        recursive: Enable sibling/adjacent chunk merging for this search

    Returns:
        Hits and the exact resolved repository id, including for zero hits.
    """
    from stropha.pipeline.builder import build_stages
    from stropha.pipeline.config import load_pipeline_config

    # Clone or get from cache
    cached = clone_repo(url, refresh=refresh)

    # Index if needed
    if cached.needs_index or refresh:
        index_cached_repo(cached)
        # Reload to pick up the updated ``last_indexed`` timestamp; the
        # ``None`` branch maps to a hard failure that no caller should see
        # in practice (we just wrote the cache entry).
        reloaded = get_cached_repo(url)
        if reloaded is None:
            raise GitUrlError("Failed to load cached repo after indexing")
        cached = reloaded

    # Search (STROPHA_INDEX_PATH removed in v=9 — see note in
    # index_cached_repo; the unified YAML `paths.index` DB is shared).
    os.environ["STROPHA_TARGET_REPO"] = str(cached.local_path)

    try:
        from stropha.ingest.git_meta import detect

        overrides = None
        if recursive:
            overrides = {"retrieval": {"config": {"recursive_enabled": True}}}
        resolved = load_pipeline_config(overrides=overrides)
        built = build_stages(resolved)
        with built.storage:  # type: ignore[union-attr]
            identity = detect(cached.local_path)
            repo_id = built.storage.lookup_repo_id(identity.normalized_key)
            if repo_id is None:
                raise GitUrlError(f"Cached repository is not indexed: {cached.url}")
            hits = built.retrieval.search(query, top_k=top_k, repo_id=repo_id)
        return GitUrlSearchResult(hits=hits, repo_id=repo_id)
    finally:
        os.environ.pop("STROPHA_TARGET_REPO", None)


def is_git_url(s: str) -> bool:
    """Check if a string looks like a git URL.

    Matches:
    - https://github.com/...
    - https://gitlab.com/...
    - git@github.com:...
    - git://...
    """
    if not isinstance(s, str):
        return False

    s = s.strip()

    # HTTPS URLs
    if s.startswith(("https://github.com/", "https://gitlab.com/", "https://bitbucket.org/")):
        return True

    # Generic https git URLs
    if s.startswith("https://") and (s.endswith(".git") or "/git/" in s):
        return True

    # SSH URLs
    if s.startswith("git@") and ":" in s:
        return True

    # Git protocol
    return bool(s.startswith("git://"))
