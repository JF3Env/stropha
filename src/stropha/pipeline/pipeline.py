"""Pipeline orchestrator — adapter-aware indexing.

Phase 2 wires walker, enricher, embedder, and storage as adapter
instances. Chunker remains the legacy ``stropha.ingest.chunker.Chunker``
until Phase 3 ships its dispatcher adapter.

Drift detection: a chunk is fresh only when its stored
``(content_hash, embedding_model, enricher_id)`` matches the active
triplet. Changing any of those triggers transparent re-processing. The
``enrichments`` cache table avoids redoing the enricher's work when the
same content+adapter combo has been seen before in this DB.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import (
    FullReconcileRequiredError,
    PipelineError,
    RepoBlacklistedError,
    StrophaError,
)
from ..ingest.git_meta import detect as detect_repo
from ..logging import get_logger
from ..models import SOURCE_SNAPSHOT_VERSION, Chunk, source_snapshot_meta_key
from ..stages.chunker import ChunkerStage
from ..stages.embedder import EmbedderStage
from ..stages.enricher import EnricherStage, effective_enricher_id
from ..stages.storage import FileSnapshotChunkWrite, StorageStage
from ..stages.walker import WalkerStage
from .base import StageContext

log = get_logger(__name__)

_SOURCE_SNAPSHOT_ATTEMPTS = 3
_INCOMPLETE_ENRICHER_ID = "__stropha_incomplete__"
_GRAPH_DIRTY_META_PREFIX = "graph_dirty:"
_CHUNKER_ID_META_PREFIX = "active_chunker_id:"
_WALKER_ID_META_PREFIX = "active_walker_id:"
_ADMISSION_RULE_FILES = frozenset({".gitignore", ".strophaignore", ".ragignore"})


def _repo_blocked(identity) -> bool:  # type: ignore[no-untyped-def]
    """Whether ``indexing.blacklist_repos`` blocks this repository.

    Imported lazily: this module is pulled in while the config package is
    still initializing on some entry points.
    """
    from ..config import repo_is_blacklisted

    return repo_is_blacklisted(identity.normalized_key, identity.root_path)


def _graph_dirty_meta_key(repo_id: int) -> str:
    return f"{_GRAPH_DIRTY_META_PREFIX}{repo_id}"


def _chunker_id_meta_key(repo_id: int) -> str:
    return f"{_CHUNKER_ID_META_PREFIX}{repo_id}"


def _walker_id_meta_key(repo_id: int) -> str:
    return f"{_WALKER_ID_META_PREFIX}{repo_id}"


@dataclass(frozen=True)
class _SourceSnapshot:
    """One stable read of a saved working-tree file."""

    content: str
    content_hash: str
    size_bytes: int
    mtime: float


def _capture_source_snapshot(path: Path, *, repo_root: Path) -> _SourceSnapshot | None:
    """Read bytes only when path identity and metadata stay stable."""
    try:
        root = repo_root.resolve()
        read_path = path.resolve()
        read_path.relative_to(root)
        before = read_path.stat()
        raw = read_path.read_bytes()
        after = read_path.stat()
        resolved_after = path.resolve()
        resolved_after.relative_to(root)
    except (OSError, ValueError):
        return None
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or len(raw) != after.st_size
        or resolved_after != read_path
    ):
        return None
    return _SourceSnapshot(
        content=raw.decode("utf-8", errors="replace"),
        content_hash=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mtime=after.st_mtime,
    )


@dataclass
class RepoStats:
    """Per-repo counters returned by :meth:`Pipeline.run`."""

    normalized_key: str
    url: str | None
    code_changed: bool = False
    """Whether source files were added, modified, renamed, or deleted."""
    files_visited: int = 0
    files_skipped_fresh: int = 0
    """Files whose full raw-byte hash, metadata, chunk count, and active
    adapters matched the saved source snapshot."""
    files_evicted: int = 0
    """Files that disappeared from the working tree since last index;
    their chunks have been dropped."""
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_skipped_fresh: int = 0
    chunks_enriched_from_cache: int = 0
    chunks_enriched_fresh: int = 0
    chunks_enriched_failed: int = 0
    """Chunks the enricher tried and failed (e.g. LLM rate-limit). The
    result is NOT cached so the next index run will retry."""


@dataclass
class PipelineStats:
    """Aggregate result of a :meth:`Pipeline.run`."""

    repos: list[RepoStats] = field(default_factory=list)
    enricher_id: str = ""
    embedder_id: str = ""
    secondary_embedder_id: str | None = None
    secondary_chunks_embedded: int = 0

    @property
    def files_visited(self) -> int:
        return sum(r.files_visited for r in self.repos)

    @property
    def files_skipped_fresh(self) -> int:
        return sum(r.files_skipped_fresh for r in self.repos)

    @property
    def files_evicted(self) -> int:
        return sum(r.files_evicted for r in self.repos)

    @property
    def chunks_seen(self) -> int:
        return sum(r.chunks_seen for r in self.repos)

    @property
    def chunks_embedded(self) -> int:
        return sum(r.chunks_embedded for r in self.repos)

    @property
    def chunks_skipped_fresh(self) -> int:
        return sum(r.chunks_skipped_fresh for r in self.repos)

    @property
    def chunks_enriched_from_cache(self) -> int:
        return sum(r.chunks_enriched_from_cache for r in self.repos)

    @property
    def chunks_enriched_fresh(self) -> int:
        return sum(r.chunks_enriched_fresh for r in self.repos)

    @property
    def chunks_enriched_failed(self) -> int:
        return sum(r.chunks_enriched_failed for r in self.repos)


class Pipeline:
    """Adapter-aware walker → chunker → enricher → embedder → storage."""

    def __init__(
        self,
        *,
        storage: StorageStage,
        embedder: EmbedderStage,
        secondary_embedder: EmbedderStage | None = None,
        enricher: EnricherStage,
        repos: Sequence[Path],
        walker: WalkerStage | None = None,
        chunker: ChunkerStage | None = None,
        enricher_concurrency: int = 1,
        enricher_batch_size: int = 1,
        graphify: bool = True,
        graph_fts_augment: bool | None = None,
    ) -> None:
        """Construct the indexing pipeline.

        Args:
            storage / embedder / enricher: the adapter instances.
            repos: target repository paths.
            walker / chunker: optional override adapters (default: git-ls-files
                + tree-sitter-dispatch).
            enricher_concurrency: number of concurrent enricher calls during
                indexing. Cloud LLM backends (DeepSeek / OpenAI / Anthropic)
                wait ~250-500ms per request on network round-trip — running
                8 in-flight pipelines that latency efficiently. Local
                backends (mlx / ollama) saturate a single GPU regardless, so
                concurrency >1 mostly adds overhead. Default 1 (serial) is
                safe for any backend; opt in via ~/.stropha/config.yaml's
                ``inference.enricher_concurrency`` or ``--concurrency N``
                CLI flag.
            enricher_batch_size: chunks packed into a single LLM call. 1 =
                per-chunk (legacy). N>1 multiplies effective throughput by
                N under the same rate-limit budget. Only ``contextual``
                implements true batching; other enrichers (noop,
                hierarchical, graph-aware, ollama, mlx) ignore the knob and
                stay per-chunk. Config: ``inference.enricher_batch_size``.
            graphify: run ``graphify update`` before loading the graph
                mirror.  When ``True`` (default) the pipeline invokes
                graphify as a subprocess for each repo — creating
                ``graphify-out/graph.json`` on first run and refreshing
                it on subsequent runs.  Gracefully skipped when the
                ``graphify`` binary is not installed.  Pass ``False``
                to disable (e.g. ``--no-graphify`` on the CLI).
            graph_fts_augment: explicit graph-term FTS augmentation setting.
                ``None`` preserves the unified global configuration default.
        """
        if not repos:
            raise ValueError("Pipeline requires at least one repo")
        if enricher_concurrency < 1:
            raise ValueError(f"enricher_concurrency must be >= 1, got {enricher_concurrency}")
        if enricher_batch_size < 1:
            raise ValueError(f"enricher_batch_size must be >= 1, got {enricher_batch_size}")
        if (
            enricher_concurrency > 1
            and getattr(enricher, "__dict__", {}).get("_storage") is not None
        ):
            log.info(
                "pipeline.storage_bound_enricher_serialized",
                requested_concurrency=enricher_concurrency,
                enricher=getattr(enricher, "adapter_name", type(enricher).__name__),
            )
            enricher_concurrency = 1
        self._storage = storage
        self._embedder = embedder
        self._secondary_embedder = secondary_embedder
        self._enricher = enricher
        self._repos = [p.resolve() for p in repos]
        self._enricher_concurrency = enricher_concurrency
        self._enricher_batch_size = enricher_batch_size
        self._effective_enricher_id = effective_enricher_id(self._enricher, enricher_batch_size)
        self._graphify = graphify
        self._graph_fts_augment = graph_fts_augment
        self._force_file_snapshot_refresh = False

        # Pre-warm the enricher's lazy state (e.g. inference backend
        # singleton) so the first batch of N parallel enrichment calls
        # doesn't race on backend instantiation. health() is the cheapest
        # way to force the warmup; any probe error is benign and the
        # first real enrich() call will retry naturally.
        if enricher_concurrency > 1:
            try:
                enricher.health()
            except Exception as exc:  # pragma: no cover  defensive
                log.debug("pipeline.enricher_prewarm_failed", error=str(exc))
        # Lazy import — keeps the framework decoupled from a specific
        # adapter until the caller passes one or we fall back here.
        if walker is None:
            from ..adapters.walker.git_ls_files import GitLsFilesWalker

            walker = GitLsFilesWalker()
        if chunker is None:
            from ..adapters.chunker.tree_sitter_dispatch import (
                TreeSitterDispatchChunker,
            )

            chunker = TreeSitterDispatchChunker()
        self._walker = walker
        self._chunker = chunker

    # ------------------------------------------------------------------ run
    def run(
        self,
        *,
        rebuild: bool = False,
        mode: str = "auto",
        since_sha: str | None = None,
    ) -> PipelineStats:
        """Execute one index pass while always releasing the writer lock."""
        acquire_lock = getattr(self._storage, "try_acquire_index_lock", None)
        if acquire_lock is not None and not acquire_lock():
            detail = ""
            describe = getattr(self._storage, "describe_index_lock_holder", None)
            if describe is not None:
                detail = str(describe() or "").strip()
            raise StrophaError(
                "Another `stropha index` run is already writing to this index "
                "(advisory lock held). Wait for it to finish — or stop it — "
                "before starting another. Tip: a `git commit` triggers the "
                "post-commit hook's index, which can collide with a manual "
                "`--rebuild`." + (f" {detail}" if detail else "")
            )
        try:
            return self._run_once(rebuild=rebuild, mode=mode, since_sha=since_sha)
        finally:
            release_lock = getattr(self._storage, "release_index_lock", None)
            if release_lock is not None:
                release_lock()

    def refresh_paths(
        self,
        *,
        repo: Path,
        changed_or_added: Iterable[str] = (),
        deleted: Iterable[str] = (),
    ) -> PipelineStats:
        """Refresh an explicit saved-worktree subset without walking the repo.

        A complete reconcile must have established compatible source and adapter
        identities first. Callers should catch :class:`FullReconcileRequiredError`
        and run ``run(mode="reconcile")`` before retrying future path batches.
        """
        acquire_lock = getattr(self._storage, "try_acquire_index_lock", None)
        if acquire_lock is not None and not acquire_lock():
            raise StrophaError(
                "Another `stropha index` run is already writing to this index "
                "(advisory lock held). Wait for it to finish before refreshing paths."
            )
        try:
            return self._refresh_paths_once(
                repo=repo,
                changed_or_added=changed_or_added,
                deleted=deleted,
            )
        finally:
            release_lock = getattr(self._storage, "release_index_lock", None)
            if release_lock is not None:
                release_lock()

    def _refresh_paths_once(
        self,
        *,
        repo: Path,
        changed_or_added: Iterable[str],
        deleted: Iterable[str],
    ) -> PipelineStats:
        repo_path = repo.expanduser().resolve()
        if repo_path not in self._repos:
            raise ValueError(f"repo is not configured on this pipeline: {repo_path}")

        requested = self._normalize_refresh_paths(changed_or_added, deleted)
        stats = PipelineStats(
            enricher_id=self._effective_enricher_id,
            embedder_id=self._embedder.adapter_id,
            secondary_embedder_id=(
                self._secondary_embedder.adapter_id
                if self._secondary_embedder is not None
                else None
            ),
        )
        if not requested:
            return stats

        identity = detect_repo(repo_path)
        repo_id = self._storage.lookup_repo_id(identity.normalized_key)
        if repo_id is None:
            raise FullReconcileRequiredError(
                "Path refresh requires an existing repository snapshot"
            )
        self._require_path_refresh_compatible(repo_id)
        stored_paths = set(self._storage.load_file_meta_map(repo_id))
        requested = self._expand_indexed_symlink_aliases(
            repo_path,
            requested,
            stored_paths,
        )

        current_files: set[str] = set()
        absent: set[str] = set()
        for rel_path in requested:
            if Path(rel_path).name in _ADMISSION_RULE_FILES:
                raise FullReconcileRequiredError(f"Admission rules changed at {rel_path!r}")
            path = repo_path / rel_path
            if path.is_dir():
                raise FullReconcileRequiredError(f"Path refresh received directory {rel_path!r}")
            if path.is_file():
                current_files.add(rel_path)
            else:
                absent.add(rel_path)

        source_files = list(self._walker.discover_paths(repo_path, sorted(current_files)))
        eligible_paths = {source.rel_path for source in source_files}
        eviction_roots = absent | (current_files - eligible_paths)
        # Build the descendant predicate without accepting sibling prefixes
        # such as ``pkg2`` when the deleted directory was ``pkg``.
        evicted_paths = {
            stored_path
            for stored_path in stored_paths
            if any(
                stored_path == root or stored_path.startswith(f"{root.rstrip('/')}/")
                for root in eviction_roots
            )
            or any(stored_path.startswith(f"{root.rstrip('/')}/") for root in eligible_paths)
        }

        rstats = RepoStats(normalized_key=identity.normalized_key, url=identity.remote_url)
        stats.repos.append(rstats)
        if evicted_paths:
            self._delete_file_snapshots(repo_id, sorted(evicted_paths))
            self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")
            rstats.files_evicted = len(evicted_paths)
            rstats.code_changed = True
            self._storage.commit()

        self._force_file_snapshot_refresh = False
        published_paths = self._index_files(
            source_files,
            rstats,
            repo_id=repo_id,
            repo_key=identity.normalized_key,
            repo_root=repo_path,
        )
        if rstats.chunks_enriched_failed:
            raise PipelineError(
                f"Index incomplete: {rstats.chunks_enriched_failed} chunks failed enrichment"
            )

        vanished_paths = eligible_paths - published_paths
        late_evictions = {
            stored_path
            for stored_path in stored_paths - evicted_paths
            if any(
                stored_path == root or stored_path.startswith(f"{root.rstrip('/')}/")
                for root in vanished_paths
            )
        }
        if late_evictions:
            self._delete_file_snapshots(repo_id, sorted(late_evictions))
            self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")
            rstats.files_evicted += len(late_evictions)
            rstats.code_changed = True
            self._storage.commit()

        if self._secondary_embedder is not None:
            from ..ingest.secondary_embedding_loader import SecondaryEmbeddingLoader

            secondary = SecondaryEmbeddingLoader(self._storage, self._secondary_embedder).load()
            stats.secondary_chunks_embedded = secondary.embedded

        self._storage.commit()
        log.info(
            "pipeline.paths_refreshed",
            repo=identity.normalized_key,
            requested=len(requested),
            files=rstats.files_visited,
            files_evicted=rstats.files_evicted,
            chunks=rstats.chunks_seen,
            embedded=rstats.chunks_embedded,
        )
        return stats

    @staticmethod
    def _normalize_refresh_paths(
        changed_or_added: Iterable[str], deleted: Iterable[str]
    ) -> set[str]:
        normalized: set[str] = set()
        for raw_path in (*changed_or_added, *deleted):
            candidate = Path(raw_path)
            if not raw_path or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"refresh paths must be repository-relative: {raw_path!r}")
            normalized.add(candidate.as_posix())
        return normalized

    @staticmethod
    def _expand_indexed_symlink_aliases(
        repo: Path,
        requested: set[str],
        stored_paths: set[str],
    ) -> set[str]:
        """Include indexed in-repo symlinks that resolve to a requested path."""
        targets: set[Path] = set()
        for rel_path in requested:
            try:
                target = (repo / rel_path).resolve()
                target.relative_to(repo)
            except (OSError, RuntimeError, ValueError):
                continue
            targets.add(target)

        if not targets:
            return requested

        expanded = set(requested)
        for target in targets:
            canonical = target.relative_to(repo).as_posix()
            if canonical in stored_paths:
                expanded.add(canonical)
        for stored_path in stored_paths:
            logical_path = repo / stored_path
            if not logical_path.is_symlink():
                continue
            try:
                target = logical_path.resolve()
            except (OSError, RuntimeError):
                continue
            if target in targets:
                expanded.add(stored_path)
        return expanded

    def _require_path_refresh_compatible(self, repo_id: int) -> None:
        checks = (
            (
                "source snapshot",
                self._storage.get_meta(source_snapshot_meta_key(repo_id)),
                SOURCE_SNAPSHOT_VERSION,
            ),
            (
                "walker",
                self._storage.get_meta(_walker_id_meta_key(repo_id)),
                self._walker.adapter_id,
            ),
            (
                "chunker",
                self._storage.get_meta(_chunker_id_meta_key(repo_id)),
                self._chunker.adapter_id,
            ),
            (
                "enricher",
                self._storage.get_meta("active_enricher_id"),
                self._effective_enricher_id,
            ),
            (
                "embedder",
                self._storage.get_meta("active_embedder_id"),
                self._embedder.adapter_id,
            ),
            ("chunk id", self._storage.get_meta("chunk_id_version"), "2"),
        )
        for label, stored, active in checks:
            if stored != active:
                raise FullReconcileRequiredError(
                    f"Path refresh requires full reconcile after {label} drift "
                    f"(stored={stored!r}, active={active!r})"
                )
        if self._storage.count_chunks_without_repo() > 0:
            raise FullReconcileRequiredError(
                "Path refresh requires full reconcile while legacy chunks are unscoped"
            )

    def _run_once(
        self,
        *,
        rebuild: bool = False,
        mode: str = "auto",
        since_sha: str | None = None,
    ) -> PipelineStats:
        """Execute the pipeline.

        Args:
            rebuild: Wipe chunks and file metadata for the selected repositories
                before indexing (does NOT drop repos or affect other repositories).
                Forces ``mode="full"`` internally.
            mode: One of ``"auto"`` (default), ``"reconcile"``, ``"full"``,
                or ``"incremental"``.

                - ``auto`` — hash-based reconcile (WIN-425). The default.
                  Git-independent and crash-resilient.
                - ``reconcile`` — walk the full file set and (re)index only
                  files whose full raw-byte hash changed vs the ``files``
                  cache; evict deleted. Source of truth is
                  the content hash, not git, so uncommitted edits and files
                  left behind by an interrupted index are always caught.
                - ``full`` — walk every file via the configured walker; the
                  per-file freshness cache still skips unchanged files.
                - ``incremental`` — git fast-path: ``GitDiffWalker`` against
                  ``since_sha`` (defaults to the stored last sha). Only sees
                  committed deltas; falls back to ``reconcile`` when no
                  checkpoint exists.
            since_sha: Override the stored last_indexed_sha. Useful for
                debugging or re-running against a known checkpoint.
        """
        if rebuild:
            mode = "full"
        if mode not in ("auto", "full", "incremental", "reconcile"):
            raise ValueError(f"unknown pipeline mode: {mode!r}")

        stats = PipelineStats(
            enricher_id=self._effective_enricher_id,
            embedder_id=self._embedder.adapter_id,
            secondary_embedder_id=(
                self._secondary_embedder.adapter_id
                if self._secondary_embedder is not None
                else None
            ),
        )
        # Drift detection: compare the stored adapter ids (written by the
        # previous run at the bottom of this method) against the current
        # ones. When they differ, every per-chunk freshness check that
        # follows will fail and re-embed — surface that intent up-front
        # so the operator sees the cause, not just the symptom (silent
        # mass re-embed and the cloud bill that comes with it).
        if not rebuild:
            stored_enricher = self._storage.get_meta("active_enricher_id")
            stored_embedder = self._storage.get_meta("active_embedder_id")
            drift_enricher = (
                stored_enricher is not None and stored_enricher != self._effective_enricher_id
            )
            drift_embedder = (
                stored_embedder is not None and stored_embedder != self._embedder.adapter_id
            )
            if drift_enricher or drift_embedder:
                log.warning(
                    "pipeline.config_drift_detected",
                    stored_enricher_id=stored_enricher,
                    active_enricher_id=self._effective_enricher_id,
                    stored_embedder_id=stored_embedder,
                    active_embedder_id=self._embedder.adapter_id,
                    enricher_changed=drift_enricher,
                    embedder_changed=drift_embedder,
                    hint=(
                        "every chunk that doesn't match the new triplet "
                        "will be re-enriched and/or re-embedded. Pass "
                        "--rebuild --vacuum for a clean recreate."
                    ),
                )

        # Auto-backfill orphans (v1→v2 upgrade path).
        if not rebuild and self._storage.count_chunks_without_repo() > 0:
            if len(self._repos) == 1:
                identity = detect_repo(self._repos[0])
                repo_id = self._storage.register_repo(identity, publish_watermark=False)
                n = self._storage.backfill_chunks_to_repo(repo_id, self._repos[0])
                if n > 0:
                    log.info("pipeline.chunks_backfilled", count=n)
                    self._storage.commit()
            else:
                log.warning(
                    "pipeline.legacy_chunks_present_multi_repo",
                    advice="run `stropha index --rebuild` to clean up",
                )

        if rebuild:
            for repo_path in self._repos:
                identity = detect_repo(repo_path)
                if _repo_blocked(identity):
                    log.warning(
                        "pipeline.repo_blacklisted",
                        repo=identity.normalized_key,
                        path=str(repo_path),
                        phase="rebuild",
                    )
                    continue
                repo_id = self._storage.register_repo(identity, publish_watermark=False)
                log.info(
                    "pipeline.rebuild_clear_repo",
                    repo=identity.normalized_key,
                    repo_id=repo_id,
                )
                self._storage.clear_repo(repo_id)
                self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")

        for repo_path in self._repos:
            try:
                rstats = self._index_one_repo(
                    repo_path,
                    mode=mode,
                    since_sha_override=since_sha,
                )
            except RepoBlacklistedError as exc:
                # A blocked repository is skipped, not fatal: a multi-repo run
                # must still index everything else it was asked for.
                log.warning(
                    "pipeline.repo_blacklisted", repo=exc.repo, path=str(repo_path), phase="index"
                )
                continue
            stats.repos.append(rstats)
            self._storage.commit()

        self._storage.set_meta("chunk_id_version", "2")
        self._storage.set_meta("active_enricher_id", self._effective_enricher_id)
        self._storage.set_meta("active_embedder_id", self._embedder.adapter_id)
        self._storage.commit()

        # The secondary vector space is independently incremental: storage
        # selects rows by its adapter identity and the current content hash.
        # Run after primary indexing has committed all chunks, but before the
        # graph lifecycle starts.
        if self._secondary_embedder is not None:
            from ..ingest.secondary_embedding_loader import SecondaryEmbeddingLoader

            secondary_result = SecondaryEmbeddingLoader(
                self._storage, self._secondary_embedder
            ).load()
            stats.secondary_chunks_embedded = secondary_result.embedded

        # Run graphify to produce / refresh graph.json BEFORE loading the
        # mirror.  This makes ``stropha index`` a self-contained entry point:
        # even on first run, the graph tools will have structural data.
        # Gracefully skipped when the graphify binary is not installed.
        # The graphify mirror writes to the backend's graph tables. A storage
        # backend that does not implement the graph subsystem can advertise
        # ``graph_capable = False`` to skip the mirror so indexing still
        # succeeds (graph-gated MCP tools then report no graph). Both shipped
        # backends (sqlite-vec, postgres) support it; this is a framework hook.
        graph_capable = getattr(self._storage, "graph_capable", True)
        if self._graphify and graph_capable:
            failed_repos = self._run_graphify(stats.repos)
            # Refreshing the mirror also embeds graph nodes and augments FTS.
            # Keep the entire lifecycle behind --graphify/--no-graphify so a
            # caller can explicitly avoid expensive post-index graph work.
            mirrored_repo_ids = self._refresh_graphify_mirror(skip_repos=failed_repos)
            for repo_id in mirrored_repo_ids:
                if self._storage.get_meta(_graph_dirty_meta_key(repo_id)) == "1":
                    self._storage.set_meta(_graph_dirty_meta_key(repo_id), "0")
            if mirrored_repo_ids:
                self._storage.commit()

        return stats

    def _run_graphify(self, repo_stats: Sequence[RepoStats]) -> set[Path]:
        """Invoke ``graphify update`` for repos whose graph may be outdated.

        Creates ``graphify-out/graph.json`` on first run; refreshes it
        after source changes. Unchanged repos are skipped only when their
        existing graph has already been imported. Repositories skipped or
        failed by graphify are returned so their dirty markers remain set.
        """
        from ..ingest.graphify_loader import GraphifyLoader
        from .graphify_runner import run_graphify_multi

        repos_to_update: list[Path] = []
        for repo_path, rstats in zip(self._repos, repo_stats, strict=True):
            identity = detect_repo(repo_path)
            repo_id = self._storage.register_repo(identity, publish_watermark=False)
            graph_dirty = self._storage.get_meta(_graph_dirty_meta_key(repo_id)) == "1"
            if not rstats.code_changed and not graph_dirty:
                try:
                    loader = GraphifyLoader(
                        self._storage,
                        repo_path,
                        repo_id=repo_id,
                    )  # type: ignore[arg-type]
                    graph_is_imported = loader.exists() and not loader.is_stale()
                except Exception as exc:  # pragma: no cover - custom storage fallback
                    log.debug(
                        "pipeline.graphify_freshness_check_failed",
                        repo=str(repo_path),
                        error=str(exc),
                    )
                else:
                    if graph_is_imported:
                        log.info(
                            "pipeline.graphify_skipped_fresh",
                            repo=str(repo_path),
                            repo_id=repo_id,
                        )
                        continue
            repos_to_update.append(repo_path)

        if not repos_to_update:
            return set()

        results = run_graphify_multi(repos_to_update)
        failed_repos = {repo.resolve() for repo in repos_to_update}
        for r in results:
            repo = r.repo.resolve()
            if r.error or r.skipped:
                log.warning(
                    "pipeline.graphify_error",
                    repo=str(r.repo),
                    error=r.error or "graphify skipped without refreshing graph.json",
                    elapsed_s=round(r.elapsed_s, 2),
                )
                continue
            failed_repos.discard(repo)
            if r.created:
                log.info(
                    "pipeline.graphify_created",
                    repo=str(r.repo),
                    elapsed_s=round(r.elapsed_s, 2),
                )
            elif r.updated:
                log.info(
                    "pipeline.graphify_updated",
                    repo=str(r.repo),
                    elapsed_s=round(r.elapsed_s, 2),
                )
        return failed_repos

    def _refresh_graphify_mirror(self, *, skip_repos: set[Path] | None = None) -> set[int]:
        """Reload the graphify mirror tables if the on-disk graph is newer.

        After the structural mirror is refreshed, embed graph node labels
        into ``graph_nodes.embedding`` so the ``graph-vec`` retrieval stream
        has fresh vectors. The vec loader is incremental: nodes whose stored
        embedding model matches the active embedder are skipped.

        Multi-repo mode (schema v7):
          Each repo gets its own namespace in graph_nodes/graph_edges via
          the repo_id column. Node IDs are prefixed with ``{repo_id}:`` to
          avoid collisions when multiple repos share the same index.
        """
        # Local import — keeps the loader module optional. If graphify
        # tables don't exist (older schema, custom storage adapter) just
        # skip silently.
        try:
            from ..ingest.graph_vec_loader import GraphVecLoader
            from ..ingest.graphify_loader import GraphifyLoader
        except ImportError:  # pragma: no cover — defensive
            return set()
        skipped = {path.resolve() for path in (skip_repos or set())}
        mirrored_repo_ids: set[int] = set()
        for repo_path in self._repos:
            if repo_path.resolve() in skipped:
                continue
            # Get repo_id for this repo (required for multi-repo isolation)
            identity = detect_repo(repo_path)
            repo_id = self._storage.register_repo(identity, publish_watermark=False)

            try:
                loader = GraphifyLoader(
                    self._storage,
                    repo_path,
                    repo_id=repo_id,
                )  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("graphify_loader.init_failed", error=str(exc))
                continue
            if not loader.exists():
                continue
            try:
                if not loader.is_stale():
                    log.debug(
                        "graphify_loader.fresh",
                        path=str(loader.graph_path),
                        repo_id=repo_id,
                    )
                    # Even if the structural graph is fresh, the embedder may have
                    # changed since last index — let the vec loader decide.
                else:
                    loader.load()
            except Exception as exc:
                log.warning(
                    "graphify_loader.load_failed",
                    path=str(loader.graph_path),
                    repo_id=repo_id,
                    error=str(exc),
                )
                continue
            mirrored_repo_ids.add(repo_id)

        # Embed any node whose stored model is not the active embedder.
        try:
            vec_loader = GraphVecLoader(self._storage, self._embedder)  # type: ignore[arg-type]
            if vec_loader.needs_run():
                vec_loader.load()
        except Exception as exc:
            log.warning("graph_vec_loader.failed", error=str(exc))
            return set()

        # L2 retroactive FTS augmentation. Toggle via graph.fts_augment in
        # ~/.stropha/config.yaml (default true). The augmentation is a no-op when
        # graph_nodes is empty.
        graph_fts_augment = self._graph_fts_augment
        if graph_fts_augment is None:
            from ..config import get_config as _get_config

            graph_fts_augment = _get_config().graph.fts_augment
        if graph_fts_augment:
            try:
                self._storage.augment_fts_with_graph()  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("fts_augment.failed", error=str(exc))
                return set()
        return mirrored_repo_ids

    # --------------------------------------------------------------- internals
    def _index_one_repo(
        self,
        repo_path: Path,
        *,
        mode: str = "auto",
        since_sha_override: str | None = None,
    ) -> RepoStats:
        identity = detect_repo(repo_path)
        # Enforced here, at the single funnel every indexing path goes through,
        # so a script asking for a blocked repository cannot slip past it.
        if _repo_blocked(identity):
            raise RepoBlacklistedError(identity.normalized_key)
        repo_id = self._storage.register_repo(identity, publish_watermark=False)
        snapshot_meta_key = source_snapshot_meta_key(repo_id)
        stored_snapshot_version = self._storage.get_meta(snapshot_meta_key)
        source_snapshot_changed = stored_snapshot_version != SOURCE_SNAPSHOT_VERSION
        chunker_meta_key = _chunker_id_meta_key(repo_id)
        stored_chunker_id = self._storage.get_meta(chunker_meta_key)
        active_chunker_id = self._chunker.adapter_id
        chunker_changed = stored_chunker_id != active_chunker_id and (
            stored_chunker_id is not None or stored_snapshot_version == SOURCE_SNAPSHOT_VERSION
        )
        walker_meta_key = _walker_id_meta_key(repo_id)
        stored_walker_id = self._storage.get_meta(walker_meta_key)
        active_walker_id = self._walker.adapter_id
        walker_changed = stored_walker_id != active_walker_id and (
            stored_walker_id is not None or stored_snapshot_version == SOURCE_SNAPSHOT_VERSION
        )
        self._force_file_snapshot_refresh = (
            source_snapshot_changed or chunker_changed or walker_changed
        )
        if chunker_changed or walker_changed:
            log.warning(
                "pipeline.source_adapter_drift_detected",
                repo=identity.normalized_key,
                stored_chunker_id=stored_chunker_id,
                active_chunker_id=active_chunker_id,
                stored_walker_id=stored_walker_id,
                active_walker_id=active_walker_id,
                action="forcing complete hash reconciliation",
            )
        # A checkout may first be indexed before it is a Git repo or before an
        # origin is configured. Once its stable remote identity is available,
        # those fallback identities are never selected again and cannot
        # reconcile deleted files, so discard their obsolete index data.
        root = identity.root_path
        for fallback_key in (f"path:{root}", f"local:{root}"):
            if fallback_key == identity.normalized_key:
                continue
            fallback_id = self._storage.lookup_repo_id(fallback_key)
            if fallback_id is None or fallback_id == repo_id:
                continue
            self._storage.clear_repo(fallback_id)
            log.info(
                "pipeline.repo_fallback_cleared",
                repo=identity.normalized_key,
                fallback_repo=fallback_key,
            )
        log.info(
            "pipeline.repo_registered",
            repo=identity.normalized_key,
            url=identity.remote_url,
            branch=identity.default_branch,
            commit=identity.head_commit,
        )
        rstats = RepoStats(
            normalized_key=identity.normalized_key,
            url=identity.remote_url,
        )

        # ---- Resolve effective mode (auto → full | incremental) -----------
        meta_key = f"last_indexed_sha_{repo_id}"
        stored_sha = self._storage.get_meta(meta_key)
        since_sha = since_sha_override or stored_sha
        head_sha = identity.head_commit

        effective_mode = mode
        if self._force_file_snapshot_refresh and mode == "incremental":
            log.info(
                "pipeline.incremental_snapshot_upgrade",
                repo=identity.normalized_key,
                action="forcing complete hash reconciliation",
            )
            effective_mode = "reconcile"
        elif mode == "auto":
            # WIN-425: hash-based reconcile is the default change detector.
            # Git is no longer the source of truth — full-file hashes are — so
            # uncommitted edits and files left
            # behind by an interrupted index are always picked up, and a
            # non-git directory still indexes incrementally.
            effective_mode = "reconcile"
        elif mode == "incremental" and not since_sha:
            # Caller explicitly asked for the git fast-path but we have no
            # checkpoint — expected on the first run after `stropha hook
            # install` (the hook always passes --incremental to self-document
            # intent). Reconcile is the safe bootstrap: it indexes everything
            # via hashes and leaves a checkpoint for later incremental runs.
            log.info(
                "pipeline.incremental_no_checkpoint",
                repo=identity.normalized_key,
                action="falling back to hash reconcile",
            )
            effective_mode = "reconcile"

        log.info(
            "pipeline.mode_resolved",
            repo=identity.normalized_key,
            mode=effective_mode,
            since_sha=since_sha,
            head_sha=head_sha,
        )

        if effective_mode == "incremental":
            self._index_one_repo_incremental(
                repo_path,
                identity,
                repo_id,
                rstats,
                since_sha,  # type: ignore[arg-type]
            )
        elif effective_mode == "reconcile":
            self._index_one_repo_reconcile(repo_path, identity, repo_id, rstats)
        else:
            self._index_one_repo_full(repo_path, identity, repo_id, rstats)

        if rstats.chunks_enriched_failed:
            raise PipelineError(
                f"Index incomplete: {rstats.chunks_enriched_failed} chunks failed enrichment"
            )

        self._storage.set_meta(snapshot_meta_key, SOURCE_SNAPSHOT_VERSION)
        self._storage.set_meta(chunker_meta_key, active_chunker_id)
        self._storage.set_meta(walker_meta_key, active_walker_id)
        # Record the new checkpoint when we have a HEAD sha (real git repo).
        if head_sha:
            self._storage.set_meta(meta_key, head_sha)
        self._storage.complete_repo_index(repo_id, head_sha)
        self._storage.commit()

        log.info(
            "pipeline.repo_done",
            repo=identity.normalized_key,
            mode=effective_mode,
            files=rstats.files_visited,
            files_skipped_fresh=rstats.files_skipped_fresh,
            files_evicted=rstats.files_evicted,
            chunks=rstats.chunks_seen,
            embedded=rstats.chunks_embedded,
            reused=rstats.chunks_skipped_fresh,
            enricher_cache_hits=rstats.chunks_enriched_from_cache,
        )
        return rstats

    def _index_one_repo_full(
        self,
        repo_path: Path,
        identity,
        repo_id: int,
        rstats: RepoStats,
    ) -> None:
        """Legacy path: walk every file via the configured walker."""
        # Materialise the walker output so we can both index it and use
        # it for stale-file detection.
        source_files = list(self._walker.discover(repo_path))
        current_paths = self._index_files(
            source_files,
            rstats,
            repo_id=repo_id,
            repo_key=identity.normalized_key,
            repo_root=repo_path,
        )
        # Passive stale cleanup: files that were tracked in a previous
        # run but no longer appear in the walker output get their
        # chunks AND file_meta evicted.
        stale = self._storage.list_stale_files(repo_id, sorted(current_paths))
        if stale:
            rstats.code_changed = True
            evicted = self._delete_file_snapshots(repo_id, stale)
            self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")
            rstats.files_evicted = len(stale)
            log.info(
                "pipeline.stale_files_evicted",
                repo=identity.normalized_key,
                evicted_files=len(stale),
                evicted_chunks=evicted,
                examples=stale[:5],
            )
            self._storage.commit()

    def _index_one_repo_reconcile(
        self,
        repo_path: Path,
        identity,
        repo_id: int,
        rstats: RepoStats,
    ) -> None:
        """Reconcile every saved file against its full raw-byte SHA-256."""
        source_files = list(self._walker.discover(repo_path))
        skipped_before = rstats.files_skipped_fresh
        visited_before = rstats.files_visited
        current_paths = self._index_files(
            source_files,
            rstats,
            repo_id=repo_id,
            repo_key=identity.normalized_key,
            repo_root=repo_path,
        )
        visited = rstats.files_visited - visited_before
        fresh = rstats.files_skipped_fresh - skipped_before

        log.info(
            "pipeline.reconcile_scan",
            repo=identity.normalized_key,
            total=len(source_files),
            changed=visited - fresh,
            fresh=fresh,
        )

        # Deletions: files tracked in the index but no longer on disk.
        stale = self._storage.list_stale_files(repo_id, sorted(current_paths))
        if stale:
            rstats.code_changed = True
            evicted = self._delete_file_snapshots(repo_id, stale)
            self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")
            rstats.files_evicted = len(stale)
            log.info(
                "pipeline.reconcile_evicted",
                repo=identity.normalized_key,
                evicted_files=len(stale),
                evicted_chunks=evicted,
                examples=stale[:5],
            )
            self._storage.commit()

    def _index_one_repo_incremental(
        self,
        repo_path: Path,
        identity,
        repo_id: int,
        rstats: RepoStats,
        since_sha: str,
    ) -> None:
        """Phase B path: ingest only the FileDeltas between since_sha..HEAD.

        Falls back to ``_index_one_repo_full`` on any git-side failure —
        we never want a transient git error to leave the index stale.
        """
        from ..adapters.walker.git_diff import GitDiffWalker, GitDiffWalkerConfig
        from ..config import get_config
        from ..errors import WalkerError

        rename_threshold = max(0, min(100, get_config().git.rename_threshold))
        diff_walker = GitDiffWalker(GitDiffWalkerConfig(rename_threshold=rename_threshold))
        try:
            deltas = diff_walker.discover_deltas(repo_path, since_sha=since_sha)
        except WalkerError as exc:
            log.warning(
                "pipeline.incremental_fallback",
                repo=identity.normalized_key,
                error=str(exc),
            )
            self._index_one_repo_full(repo_path, identity, repo_id, rstats)
            return

        rstats.code_changed = bool(deltas)
        if deltas:
            self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")

        adds_modifies: list[str] = []
        deletes: list[str] = []
        renames: list[tuple[str, str]] = []
        for d in deltas:
            if d.action == "delete":
                deletes.append(d.rel_path)
            elif d.action == "rename" and d.old_rel_path:
                renames.append((d.old_rel_path, d.rel_path))
                # If the rename has content changes we still need to
                # re-index the destination — chunks_is_fresh catches the
                # unchanged-chunk case downstream.
                adds_modifies.append(d.rel_path)
            else:  # add | modify
                adds_modifies.append(d.rel_path)

        log.info(
            "pipeline.incremental_start",
            repo=identity.normalized_key,
            since_sha=since_sha,
            adds_modifies=len(adds_modifies),
            renames=len(renames),
            deletes=len(deletes),
        )

        # ---- Apply renames first (zero re-embed when content stable) -----
        for old, new in renames:
            moved = self._storage.rename_chunks(repo_id, old, new)
            log.info(
                "pipeline.incremental_rename",
                repo=identity.normalized_key,
                old=old,
                new=new,
                chunks_moved=moved,
            )

        # ---- Apply deletes ------------------------------------------------
        if deletes:
            evicted = self._delete_file_snapshots(repo_id, deletes)
            rstats.files_evicted = len(deletes)
            log.info(
                "pipeline.incremental_deletes",
                repo=identity.normalized_key,
                files=len(deletes),
                chunks_evicted=evicted,
            )
        self._storage.commit()

        # ---- Index adds + modifies (full chunker + embedder pass) --------
        if adds_modifies:
            source_files = list(self._walker.discover_paths(repo_path, adds_modifies))
            self._index_files(
                source_files,
                rstats,
                repo_id=repo_id,
                repo_key=identity.normalized_key,
                repo_root=repo_path,
            )

    def _delete_file_snapshots(self, repo_id: int, rel_paths: Sequence[str]) -> int:
        return sum(
            self._storage.delete_file_snapshot(repo_id=repo_id, rel_path=rel_path)
            for rel_path in rel_paths
        )

    def _index_files(
        self,
        files: Iterable,
        stats: RepoStats,
        *,
        repo_id: int,
        repo_key: str,
        repo_root: Path,
    ) -> set[str]:
        batch_size = self._embedder.batch_size
        current_paths: set[str] = set()
        stored_files = self._storage.load_file_meta_map(repo_id)
        enricher_id = self._effective_enricher_id
        embedder_model = self._embedder.model_name

        for sf in files:
            log.debug("pipeline.file_start", repo=repo_key, rel_path=sf.rel_path)
            stats.files_visited += 1
            published = False
            vanished = False
            for attempt in range(1, _SOURCE_SNAPSHOT_ATTEMPTS + 1):
                source = _capture_source_snapshot(sf.path, repo_root=repo_root)
                if source is None:
                    vanished = not sf.path.exists()
                    if vanished:
                        break
                    if attempt < _SOURCE_SNAPSHOT_ATTEMPTS:
                        continue
                    break

                if not self._force_file_snapshot_refresh and self._storage.file_is_fresh(
                    repo_id=repo_id,
                    rel_path=sf.rel_path,
                    content_hash=source.content_hash,
                    mtime=source.mtime,
                    size_bytes=source.size_bytes,
                    enricher_id=enricher_id,
                    embedder_model=embedder_model,
                ):
                    current_paths.add(sf.rel_path)
                    stats.files_skipped_fresh += 1
                    published = True
                    break

                snapshot_file = sf.model_copy(
                    update={
                        "snapshot_content": source.content,
                        "size_bytes": source.size_bytes,
                    }
                )
                file_chunks = list(self._chunker.chunk([snapshot_file], repo_key=repo_key))
                desired_chunk_ids = [chunk.chunk_id for chunk in file_chunks]
                if len(desired_chunk_ids) != len(set(desired_chunk_ids)):
                    raise PipelineError(f"Chunker emitted duplicate IDs for {sf.rel_path!r}")
                by_id = {chunk.chunk_id: chunk for chunk in file_chunks}

                chunks_skipped = 0
                reused_chunks: list[Chunk] = []
                enriched_cached = 0
                enriched_fresh = 0
                enriched_failed = 0
                pending_chunks: list[Chunk] = []
                pending_indices: list[int] = []
                staged: list[tuple[Chunk, str | None]] = []
                enrichment_writes: list[tuple[str, str]] = []
                for chunk in file_chunks:
                    if self._storage.chunk_is_fresh(
                        chunk.chunk_id,
                        chunk.content_hash,
                        embedder_model,
                        enricher_id=enricher_id,
                    ):
                        chunks_skipped += 1
                        reused_chunks.append(chunk)
                        continue
                    cached = self._storage.get_enrichment(chunk.content_hash, enricher_id)
                    if cached is not None:
                        enriched_cached += 1
                        staged.append((chunk, cached))
                    else:
                        pending_indices.append(len(staged))
                        staged.append((chunk, None))
                        pending_chunks.append(chunk)

                if pending_chunks:
                    contexts = [
                        StageContext(
                            repo_key=repo_key,
                            parent_chunk=(
                                by_id.get(chunk.parent_chunk_id) if chunk.parent_chunk_id else None
                            ),
                        )
                        for chunk in pending_chunks
                    ]
                    results = self._enrich_chunks(pending_chunks, contexts)
                    for idx, (embedding_text, success) in zip(
                        pending_indices, results, strict=True
                    ):
                        chunk = staged[idx][0]
                        staged[idx] = (chunk, embedding_text)
                        if success:
                            enrichment_writes.append((chunk.content_hash, embedding_text))
                            enriched_fresh += 1
                        else:
                            enriched_failed += 1

                chunk_writes: list[FileSnapshotChunkWrite] = []
                for offset in range(0, len(staged), batch_size):
                    batch = staged[offset : offset + batch_size]
                    texts = [text for _chunk, text in batch]
                    if any(text is None for text in texts):
                        raise PipelineError(
                            f"Enricher returned incomplete results for {sf.rel_path!r}"
                        )
                    embedding_texts = [text for text in texts if text is not None]
                    vectors = self._embedder.embed_documents(embedding_texts)
                    for (chunk, _text), vector, embedding_text in zip(
                        batch, vectors, embedding_texts, strict=True
                    ):
                        chunk_writes.append(
                            {
                                "chunk": chunk,
                                "embedding": vector,
                                "embedding_text": embedding_text,
                            }
                        )

                confirmed = _capture_source_snapshot(sf.path, repo_root=repo_root)
                if confirmed is None or confirmed.content_hash != source.content_hash:
                    vanished = confirmed is None and not sf.path.exists()
                    if vanished:
                        break
                    log.info(
                        "pipeline.file_changed_during_index",
                        repo=repo_key,
                        rel_path=sf.rel_path,
                        attempt=attempt,
                    )
                    if attempt < _SOURCE_SNAPSHOT_ATTEMPTS:
                        continue
                    break

                for content_hash, embedding_text in enrichment_writes:
                    self._storage.put_enrichment(content_hash, enricher_id, embedding_text)
                self._storage.replace_file_snapshot(
                    repo_id=repo_id,
                    rel_path=sf.rel_path,
                    desired_chunk_ids=desired_chunk_ids,
                    chunk_writes=chunk_writes,
                    embedding_model=embedder_model,
                    embedding_dim=self._embedder.dim,
                    enricher_id=(enricher_id if enriched_failed == 0 else _INCOMPLETE_ENRICHER_ID),
                    file_content_hash=confirmed.content_hash,
                    size_bytes=confirmed.size_bytes,
                    mtime=confirmed.mtime,
                    reused_chunks=reused_chunks,
                )
                self._storage.set_meta(_graph_dirty_meta_key(repo_id), "1")
                self._storage.commit()

                previous = stored_files.get(sf.rel_path)
                if (
                    previous is None
                    or previous.content_hash != confirmed.content_hash
                    or previous.enricher_id == _INCOMPLETE_ENRICHER_ID
                ):
                    stats.code_changed = True
                stats.chunks_seen += len(file_chunks)
                stats.chunks_embedded += len(chunk_writes)
                stats.chunks_skipped_fresh += chunks_skipped
                stats.chunks_enriched_from_cache += enriched_cached
                stats.chunks_enriched_fresh += enriched_fresh
                stats.chunks_enriched_failed += enriched_failed
                current_paths.add(sf.rel_path)
                published = True
                log.info(
                    "pipeline.file_snapshot_replaced",
                    repo=repo_key,
                    rel_path=sf.rel_path,
                    chunks=len(file_chunks),
                    embedded=len(chunk_writes),
                    attempt=attempt,
                )
                break

            if not published and not vanished:
                raise PipelineError(
                    f"Could not capture a stable source snapshot for {sf.rel_path!r} "
                    f"after {_SOURCE_SNAPSHOT_ATTEMPTS} attempts"
                )

        return current_paths

    def _enrich_chunks(
        self,
        chunks: Sequence[Chunk],
        contexts: Sequence[StageContext],
    ) -> list[tuple[str, bool]]:
        """Run the enricher on ``chunks`` and return ``(embed_text, success)``
        in input order.

        The ``success`` bit is provided by
        :func:`stropha.stages.enricher.enrich_with_status`, which dispatches
        to ``enricher.enrich_with_status`` when the adapter implements it,
        otherwise wraps ``enricher.enrich`` and assumes success. Adapters
        that can fail gracefully (LLM-backed) MUST override
        ``enrich_with_status`` so the pipeline can skip caching failed
        results.

        Switches between serial and ThreadPool execution based on
        ``self._enricher_concurrency`` (default 1 = serial). The function
        is the only place that knows about parallelism; everything upstream
        (cache checks, freshness checks) and downstream (storage writes,
        embedding) stays serial — that keeps SQLite single-writer (it is
        NOT thread-safe per default) and embedder batching deterministic.

        Thread-safety notes:
          - ``enricher.enrich`` is a pure compute-or-network call. The
            ``contextual`` enricher routes through ``stropha.inference``
            which uses module-level stdlib urllib — safe under threads.
          - The ``get_backend`` singleton may race on first init, but
            after init it's a read-only lookup. Pre-warming by calling
            it serially before the first batch avoids the race entirely.
          - We do NOT touch ``self._storage`` here; all storage writes
            happen in the caller after this returns.

        Order preservation: ``ThreadPoolExecutor.map`` yields results in
        input order regardless of completion order, so the returned list
        zips 1:1 with ``chunks``.
        """
        if len(chunks) != len(contexts):
            raise ValueError(
                f"chunks ({len(chunks)}) and contexts ({len(contexts)}) must have the same length"
            )

        # Import lazily to avoid a circular import (stages.enricher imports
        # pipeline.base, pipeline.base is sibling of pipeline.pipeline).
        from ..stages.enricher import enrich_with_status

        # Decide whether to batch. The adapter must expose ``enrich_batch``
        # AND the configured batch size must be > 1. Other enrichers stay
        # per-chunk regardless of the knob.
        adapter_supports_batch = callable(getattr(self._enricher, "enrich_batch", None))
        batch_size = self._enricher_batch_size if adapter_supports_batch else 1

        if batch_size > 1 and len(chunks) > 1:
            return self._enrich_chunks_batched(chunks, contexts, batch_size)

        # Per-chunk path (legacy default).
        # Serial path: same behaviour as before, no thread overhead. Also
        # the right choice when there's only 1 chunk pending — a worker
        # pool for a single task is wasteful.
        if self._enricher_concurrency <= 1 or len(chunks) <= 1:
            return [
                enrich_with_status(self._enricher, chunk, ctx)
                for chunk, ctx in zip(chunks, contexts, strict=False)
            ]

        # Parallel path. max_workers is capped at len(chunks) so we never
        # spin up idle threads. The executor lifetime is scoped to this
        # call — we want clean teardown between files so a stuck network
        # call in one batch can't bleed into the next.
        max_workers = min(self._enricher_concurrency, len(chunks))
        log.debug(
            "pipeline.enricher_parallel",
            workers=max_workers,
            pending=len(chunks),
        )
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="stropha-enrich",
        ) as executor:
            return list(
                executor.map(
                    lambda pair: enrich_with_status(self._enricher, pair[0], pair[1]),
                    zip(chunks, contexts, strict=True),
                )
            )

    def _enrich_chunks_batched(
        self,
        chunks: Sequence[Chunk],
        contexts: Sequence[StageContext],
        batch_size: int,
    ) -> list[tuple[str, bool]]:
        """Split ``chunks`` into ``batch_size`` groups and dispatch each as
        one ``enricher.enrich_batch`` call.

        Combines with ``enricher_concurrency``: each batch is one LLM call
        (e.g. 10 chunks → 1 request), and the ThreadPool runs up to
        ``concurrency`` batches in parallel (e.g. 8 batches × 10 chunks =
        80 chunks per round-trip cycle). Order is preserved across batches.
        """
        from ..stages.enricher import enrich_batch_with_status

        # Slice chunks/contexts into batches of size ``batch_size``. The
        # last batch may be smaller; the enricher handles N<batch_size fine.
        batches: list[tuple[Sequence[Chunk], Sequence[StageContext]]] = []
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batches.append((chunks[start:end], contexts[start:end]))

        log.debug(
            "pipeline.enricher_batched",
            batches=len(batches),
            batch_size=batch_size,
            total_chunks=len(chunks),
            concurrency=self._enricher_concurrency,
        )

        # Serial dispatch (concurrency=1 OR a single batch).
        if self._enricher_concurrency <= 1 or len(batches) <= 1:
            results: list[tuple[str, bool]] = []
            for batch_chunks, batch_contexts in batches:
                results.extend(
                    enrich_batch_with_status(self._enricher, batch_chunks, batch_contexts)
                )
            return results

        # Parallel dispatch: ThreadPool runs up to ``concurrency`` batches
        # at once. Each worker holds a small slice of chunks, so the GIL
        # is released on the network call and the executor.map() output
        # preserves batch order (and therefore chunk order).
        max_workers = min(self._enricher_concurrency, len(batches))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="stropha-enrich-batch",
        ) as executor:
            batch_results = list(
                executor.map(
                    lambda pair: enrich_batch_with_status(self._enricher, pair[0], pair[1]),
                    batches,
                )
            )
        # Flatten while keeping order.
        return [item for batch_result in batch_results for item in batch_result]
