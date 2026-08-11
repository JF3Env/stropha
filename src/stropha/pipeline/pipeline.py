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

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import StrophaError
from ..ingest.git_meta import detect as detect_repo
from ..logging import get_logger
from ..models import Chunk
from ..stages.chunker import ChunkerStage
from ..stages.embedder import EmbedderStage
from ..stages.enricher import EnricherStage
from ..stages.storage import StorageStage
from ..stages.walker import WalkerStage
from .base import StageContext

log = get_logger(__name__)


@dataclass
class RepoStats:
    """Per-repo counters returned by :meth:`Pipeline.run`."""

    normalized_key: str
    url: str | None
    code_changed: bool = False
    """Whether source files were added, modified, renamed, or deleted."""
    files_visited: int = 0
    files_skipped_fresh: int = 0
    """Files whose stored ``(mtime, size_bytes)`` matched the on-disk
    state — the chunker + enricher + embedder were skipped entirely.
    Phase A incremental win."""
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
                safe for any backend; opt in via stropha.yaml's
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
        """
        if not repos:
            raise ValueError("Pipeline requires at least one repo")
        if enricher_concurrency < 1:
            raise ValueError(
                f"enricher_concurrency must be >= 1, got {enricher_concurrency}"
            )
        if enricher_batch_size < 1:
            raise ValueError(
                f"enricher_batch_size must be >= 1, got {enricher_batch_size}"
            )
        self._storage = storage
        self._embedder = embedder
        self._secondary_embedder = secondary_embedder
        self._enricher = enricher
        self._repos = [p.resolve() for p in repos]
        self._enricher_concurrency = enricher_concurrency
        self._enricher_batch_size = enricher_batch_size
        supports_batch = callable(getattr(self._enricher, "enrich_batch", None))
        effective_batch_size = enricher_batch_size if supports_batch else 1
        self._effective_enricher_id = (
            f"{self._enricher.adapter_id}:batch={effective_batch_size}"
        )
        self._graphify = graphify

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
                  files whose content hash (via mtime+size fast-path) changed
                  vs the ``files`` cache; evict deleted. Source of truth is
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

        # Serialize concurrent index writers. Two `stropha index` runs writing
        # to chunks/files at once can deadlock on Postgres (SQLite serialized
        # writers implicitly). The storage advisory lock makes the second run
        # abort cleanly instead of deadlocking. No-op for backends that don't
        # implement it; released in the matching block before ``return`` and
        # by ``storage.close()`` on the error path.
        _acquire_lock = getattr(self._storage, "try_acquire_index_lock", None)
        if _acquire_lock is not None and not _acquire_lock():
            raise StrophaError(
                "Another `stropha index` run is already writing to this index "
                "(advisory lock held). Wait for it to finish — or stop it — "
                "before starting another. Tip: a `git commit` triggers the "
                "post-commit hook's index, which can collide with a manual "
                "`--rebuild`."
            )

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
                stored_enricher is not None
                and stored_enricher != self._effective_enricher_id
            )
            drift_embedder = (
                stored_embedder is not None
                and stored_embedder != self._embedder.adapter_id
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
                repo_id = self._storage.register_repo(identity)
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
                repo_id = self._storage.register_repo(identity)
                log.info(
                    "pipeline.rebuild_clear_repo",
                    repo=identity.normalized_key,
                    repo_id=repo_id,
                )
                self._storage.clear_repo(repo_id)

        for repo_path in self._repos:
            rstats = self._index_one_repo(
                repo_path, mode=mode, since_sha_override=since_sha,
            )
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
            self._run_graphify(stats.repos)
            # Refreshing the mirror also embeds graph nodes and augments FTS.
            # Keep the entire lifecycle behind --graphify/--no-graphify so a
            # caller can explicitly avoid expensive post-index graph work.
            self._refresh_graphify_mirror()

        _release_lock = getattr(self._storage, "release_index_lock", None)
        if _release_lock is not None:
            _release_lock()
        return stats

    def _run_graphify(self, repo_stats: Sequence[RepoStats]) -> None:
        """Invoke ``graphify update`` for repos whose graph may be outdated.

        Creates ``graphify-out/graph.json`` on first run; refreshes it
        after source changes. Unchanged repos are skipped only when their
        existing graph has already been imported. Skipped silently when
        the ``graphify`` binary is not installed (no hard dependency).
        """
        from ..ingest.graphify_loader import GraphifyLoader
        from .graphify_runner import run_graphify_multi

        repos_to_update: list[Path] = []
        for repo_path, rstats in zip(self._repos, repo_stats, strict=True):
            if not rstats.code_changed:
                try:
                    identity = detect_repo(repo_path)
                    repo_id = self._storage.register_repo(identity)
                    loader = GraphifyLoader(
                        self._storage, repo_path, repo_id=repo_id,
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
            return

        results = run_graphify_multi(repos_to_update)
        for r in results:
            if r.error:
                log.warning(
                    "pipeline.graphify_error",
                    repo=str(r.repo),
                    error=r.error,
                    elapsed_s=round(r.elapsed_s, 2),
                )
            elif r.created:
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

    def _refresh_graphify_mirror(self) -> None:
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
            return
        for repo_path in self._repos:
            # Get repo_id for this repo (required for multi-repo isolation)
            identity = detect_repo(repo_path)
            repo_id = self._storage.register_repo(identity)

            try:
                loader = GraphifyLoader(
                    self._storage, repo_path, repo_id=repo_id,
                )  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("graphify_loader.init_failed", error=str(exc))
                continue
            if not loader.exists():
                continue
            if not loader.is_stale():
                log.debug(
                    "graphify_loader.fresh",
                    path=str(loader.graph_path),
                    repo_id=repo_id,
                )
                # Even if the structural graph is fresh, the embedder may have
                # changed since last index — let the vec loader decide.
            else:
                try:
                    loader.load()
                except Exception as exc:
                    log.warning(
                        "graphify_loader.load_failed",
                        path=str(loader.graph_path),
                        repo_id=repo_id,
                        error=str(exc),
                    )
                    continue

        # Embed any node whose stored model is not the active embedder.
        try:
            vec_loader = GraphVecLoader(self._storage, self._embedder)  # type: ignore[arg-type]
            if vec_loader.needs_run():
                vec_loader.load()
        except Exception as exc:
            log.warning("graph_vec_loader.failed", error=str(exc))

        # L2 retroactive FTS augmentation. Toggle via graph.fts_augment in
        # stropha.yaml (default true). The augmentation is a no-op when
        # graph_nodes is empty.
        from ..config import get_config as _get_config
        if _get_config().graph.fts_augment:
            try:
                self._storage.augment_fts_with_graph()  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("fts_augment.failed", error=str(exc))

    # --------------------------------------------------------------- internals
    def _index_one_repo(
        self,
        repo_path: Path,
        *,
        mode: str = "auto",
        since_sha_override: str | None = None,
    ) -> RepoStats:
        identity = detect_repo(repo_path)
        repo_id = self._storage.register_repo(identity)
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
        if mode == "auto":
            # WIN-425: hash-based reconcile is the default change detector.
            # Git is no longer the source of truth — content hashes (via the
            # mtime+size fast-path) are — so uncommitted edits and files left
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
                repo_path, identity, repo_id, rstats, since_sha,  # type: ignore[arg-type]
            )
        elif effective_mode == "reconcile":
            self._index_one_repo_reconcile(repo_path, identity, repo_id, rstats)
        else:
            self._index_one_repo_full(repo_path, identity, repo_id, rstats)

        # Record the new checkpoint when we have a HEAD sha (real git repo).
        if head_sha:
            self._storage.set_meta(meta_key, head_sha)

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
        self, repo_path: Path, identity, repo_id: int, rstats: RepoStats,
    ) -> None:
        """Legacy path: walk every file via the configured walker."""
        # Materialise the walker output so we can both index it and use
        # it for stale-file detection.
        source_files = list(self._walker.discover(repo_path))
        stored = self._storage.load_file_meta_map(repo_id)
        for sf in source_files:
            try:
                st = sf.path.stat()
            except OSError:
                continue
            meta = stored.get(sf.rel_path)
            if (
                meta is None
                or meta.mtime != st.st_mtime
                or meta.size_bytes != st.st_size
            ):
                rstats.code_changed = True
                break
        self._index_files(
            source_files,
            rstats,
            repo_id=repo_id,
            repo_key=identity.normalized_key,
        )
        # Passive stale cleanup: files that were tracked in a previous
        # run but no longer appear in the walker output get their
        # chunks AND file_meta evicted.
        current_paths = [sf.rel_path for sf in source_files]
        stale = self._storage.list_stale_files(repo_id, current_paths)
        if stale:
            rstats.code_changed = True
            evicted = self._storage.delete_chunks_by_repo_paths(repo_id, stale)
            self._storage.delete_file_meta(repo_id, stale)
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
        self, repo_path: Path, identity, repo_id: int, rstats: RepoStats,
    ) -> None:
        """Hash-based reconciliation (WIN-425) — git-independent change
        detection that is crash-resilient by construction.

        Walk the full current file set, then diff each file against the
        stored ``files`` cache in memory (one bulk read, not one SELECT
        per file). A file is (re)indexed when it is new OR its
        ``(mtime, size)`` changed OR — honouring the ADR-004 / CLAUDE.md
        §3.1 drift contract — the active ``(enricher_id, embedder_model)``
        differs from what last indexed it. So a config change that alters
        the enricher ``adapter_id`` or the embedder ``model_name`` forces
        every file to be reprocessed, exactly like ``file_is_fresh``.

        Crash-resilience: a run interrupted mid-pass leaves changed/new
        files without a fresh ``files`` row, so the next reconcile picks
        them up — regardless of the git checkpoint. Uncommitted edits are
        seen too, because detection never consults git.
        """
        source_files = list(self._walker.discover(repo_path))
        stored = self._storage.load_file_meta_map(repo_id)
        # Same identifiers file_is_fresh / _index_files use, so the drift
        # contract is enforced identically on the reconcile fast-path.
        # Batched enrichers use a different prompt shape, so the effective
        # runtime batch size is part of freshness even when supplied by CLI.
        enricher_id = self._effective_enricher_id
        embedder_model = self._embedder.model_name

        changed: list = []
        for sf in source_files:
            try:
                st = sf.path.stat()
            except OSError:
                # Vanished between walk and stat — stale-cleanup evicts it.
                continue
            meta = stored.get(sf.rel_path)
            source_is_fresh = (
                meta is not None
                and meta.mtime == st.st_mtime
                and meta.size_bytes == st.st_size
            )
            if not source_is_fresh:
                rstats.code_changed = True
            if (
                meta is not None
                and source_is_fresh
                and meta.enricher_id == enricher_id
                and meta.embedder_model == embedder_model
            ):
                rstats.files_skipped_fresh += 1
                continue
            changed.append(sf)

        log.info(
            "pipeline.reconcile_scan",
            repo=identity.normalized_key,
            total=len(source_files),
            changed=len(changed),
            fresh=rstats.files_skipped_fresh,
        )

        # Index new/changed files. ``_index_files`` re-checks chunk-level
        # freshness, so unchanged chunks inside a touched file are not
        # re-embedded.
        if changed:
            self._index_files(
                changed, rstats,
                repo_id=repo_id, repo_key=identity.normalized_key,
            )

        # Deletions: files tracked in the index but no longer on disk.
        current_paths = [sf.rel_path for sf in source_files]
        stale = self._storage.list_stale_files(repo_id, current_paths)
        if stale:
            rstats.code_changed = True
            evicted = self._storage.delete_chunks_by_repo_paths(repo_id, stale)
            self._storage.delete_file_meta(repo_id, stale)
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
        from ..adapters.walker.git_diff import GitDiffWalker
        from ..errors import WalkerError
        from ..ingest.walker import Walker

        diff_walker = GitDiffWalker()
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
                old=old, new=new, chunks_moved=moved,
            )

        # ---- Apply deletes ------------------------------------------------
        if deletes:
            evicted = self._storage.delete_chunks_by_repo_paths(repo_id, deletes)
            self._storage.delete_file_meta(repo_id, deletes)
            rstats.files_evicted = len(deletes)
            log.info(
                "pipeline.incremental_deletes",
                repo=identity.normalized_key,
                files=len(deletes),
                chunks_evicted=evicted,
            )
        self._storage.commit()

        # ---- Index adds + modifies (full chunker + embedder pass) --------
        # We need to feed the chunker SourceFile objects. The legacy Walker
        # has a `discover_paths` helper for exactly this case.
        if adds_modifies:
            walker = Walker(repo_path)
            source_files = list(walker.discover_paths(adds_modifies))
            self._index_files(
                source_files, rstats,
                repo_id=repo_id, repo_key=identity.normalized_key,
            )

    def _index_files(
        self,
        files: Iterable,
        stats: RepoStats,
        *,
        repo_id: int,
        repo_key: str,
    ) -> None:
        batch_size = self._embedder.batch_size
        batch_texts: list[str] = []
        batch_chunks: list[Chunk] = []
        batch_embed_texts: list[str] = []
        seen_paths: set[str] = set()
        enricher_id = self._effective_enricher_id
        embedder_model = self._embedder.model_name

        def flush() -> None:
            if not batch_texts:
                return
            batch_n = len(batch_chunks)
            vectors = self._embedder.embed_documents(batch_texts)
            for chunk, vec, embedded_text in zip(
                batch_chunks, vectors, batch_embed_texts, strict=True
            ):
                self._storage.upsert_chunk(
                    chunk,
                    vec,
                    embedder_model,
                    self._embedder.dim,
                    repo_id=repo_id,
                    embedding_text=embedded_text,
                    enricher_id=enricher_id,
                )
                stats.chunks_embedded += 1
            self._storage.commit()
            # Periodic progress signal. Each batch = 32-64 chunks ≈ a few
            # seconds of work (slower with the contextual enricher hitting
            # a remote LLM). Tail this with:
            #   tail -f /tmp/stropha.log | grep pipeline.batch_flushed
            log.info(
                "pipeline.batch_flushed",
                repo=repo_key,
                batch_size=batch_n,
                files_visited=stats.files_visited,
                chunks_seen=stats.chunks_seen,
                chunks_embedded=stats.chunks_embedded,
                enriched_fresh=stats.chunks_enriched_fresh,
                enriched_cached=stats.chunks_enriched_from_cache,
                skipped_fresh=stats.chunks_skipped_fresh,
            )
            batch_texts.clear()
            batch_chunks.clear()
            batch_embed_texts.clear()

        for sf in files:
            log.debug("pipeline.file_start", repo=repo_key, rel_path=sf.rel_path)
            seen_paths.add(sf.rel_path)
            stats.files_visited += 1

            # --- Phase A early-skip: file-level dirty cache ----------------
            # mtime + size_bytes are a cheap O(1) check that skips the
            # chunker + enricher + embedder entirely for files that did
            # not change since the last index. The chunk_is_fresh layer
            # below still catches the edge case where a file was touched
            # but its content (hash) stayed identical.
            try:
                fs_mtime = sf.path.stat().st_mtime
                fs_size = sf.path.stat().st_size
            except OSError:
                # File vanished between walker and pipeline — skip silently
                # and let the stale-cleanup pass evict its chunks.
                continue
            if self._storage.file_is_fresh(
                repo_id=repo_id,
                rel_path=sf.rel_path,
                mtime=fs_mtime,
                size_bytes=fs_size,
                enricher_id=enricher_id,
                embedder_model=embedder_model,
            ):
                stats.files_skipped_fresh += 1
                continue

            # Build a per-file index of chunks for parent-chunk lookup in
            # the hierarchical enricher. Cheap — chunker emits ≤ ~tens per file.
            file_chunks: list[Chunk] = list(
                self._chunker.chunk([sf], repo_key=repo_key)
            )
            by_id: dict[str, Chunk] = {c.chunk_id: c for c in file_chunks}
            # Capture the file-level content hash for the file_meta upsert.
            # Same hash function chunks use — guarantees a single read of
            # the file content is sufficient. We use the first chunk's
            # content_hash as a proxy when the chunker emits a file-level
            # chunk; otherwise we hash the on-disk content here.
            file_content_hash: str | None = None

            # Two-pass split to enable parallel enrichment:
            #   pass 1 (serial, fast): chunk freshness + enrichment cache
            #     checks. These are cheap SQL queries; running them in
            #     parallel would just contend on the SQLite connection.
            #   pass 2 (parallel-able): LLM enrichment for chunks that
            #     missed the cache. Each call waits ~250-500ms on network
            #     for cloud backends; ThreadPoolExecutor lets N requests
            #     fly concurrently. Order is preserved by ``map()``.
            pending_chunks: list[Chunk] = []
            pending_embed_text_indices: list[int] = []
            # ``staged`` is the per-file workbench: each entry is the chunk
            # plus its eventual embed_text (None when still to enrich).
            staged: list[tuple[Chunk, str | None]] = []
            for chunk in file_chunks:
                stats.chunks_seen += 1
                if self._storage.chunk_is_fresh(
                    chunk.chunk_id,
                    chunk.content_hash,
                    embedder_model,
                    enricher_id=enricher_id,
                ):
                    stats.chunks_skipped_fresh += 1
                    continue

                cached = self._storage.get_enrichment(chunk.content_hash, enricher_id)
                if cached is not None:
                    stats.chunks_enriched_from_cache += 1
                    staged.append((chunk, cached))
                else:
                    # Mark for the parallel enrichment pass.
                    pending_embed_text_indices.append(len(staged))
                    staged.append((chunk, None))
                    pending_chunks.append(chunk)

            # Pass 2: enrich the pending chunks (parallel when concurrency>1
            # and there are 2+ to amortize the executor overhead).
            if pending_chunks:
                # Build StageContext per chunk now — context construction is
                # local + serial, no benefit in parallelizing it.
                contexts = [
                    StageContext(
                        repo_key=repo_key,
                        parent_chunk=(
                            by_id.get(c.parent_chunk_id) if c.parent_chunk_id else None
                        ),
                    )
                    for c in pending_chunks
                ]
                results = self._enrich_chunks(pending_chunks, contexts)
                # Splice results back into ``staged`` (order-preserving) and
                # cache each *successful* result. Failed enrichments stage
                # raw content but skip the cache write — see ``enrich_with_status``
                # in ``stropha.stages.enricher`` for why poisoning the cache
                # is worse than re-attempting next run.
                for idx, (embed_text, success) in zip(
                    pending_embed_text_indices, results, strict=False
                ):
                    chunk = staged[idx][0]
                    staged[idx] = (chunk, embed_text)
                    if success:
                        self._storage.put_enrichment(
                            chunk.content_hash, enricher_id, embed_text
                        )
                        stats.chunks_enriched_fresh += 1
                    else:
                        stats.chunks_enriched_failed += 1

            # Drain ``staged`` into the embedder batch (order-preserving).
            for chunk, maybe_embed_text in staged:
                assert maybe_embed_text is not None  # set by pass 2 if previously None
                # Shadow with a non-Optional alias so the embedder typing is
                # satisfied without rebinding ``embed_text`` from the outer
                # ``zip(...)`` loop above (mypy ``no-redef``).
                drained_embed_text: str = maybe_embed_text
                batch_texts.append(drained_embed_text)
                batch_chunks.append(chunk)
                batch_embed_texts.append(drained_embed_text)
                if len(batch_texts) >= batch_size:
                    flush()
            # Record file_meta after the file is fully processed. We use
            # the content hash of the FIRST chunk as a stable proxy; for
            # multi-chunk files the (mtime, size) primary key is what
            # triggers freshness anyway. If chunking emitted zero chunks
            # (e.g. binary slipped through filter, parser crash) we still
            # record the visit so we don't re-attempt every run.
            if file_chunks:
                file_content_hash = file_chunks[0].content_hash
            else:
                import hashlib as _hashlib
                try:
                    file_content_hash = _hashlib.sha256(
                        sf.path.read_bytes()
                    ).hexdigest()
                except OSError:
                    file_content_hash = "(unreadable)"
            self._storage.upsert_file_meta(
                repo_id=repo_id,
                rel_path=sf.rel_path,
                content_hash=file_content_hash,
                size_bytes=fs_size,
                mtime=fs_mtime,
                chunk_count=len(file_chunks),
                enricher_id=enricher_id,
                embedder_model=embedder_model,
            )
        flush()

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
                f"chunks ({len(chunks)}) and contexts ({len(contexts)}) "
                "must have the same length"
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
                    lambda pair: enrich_with_status(
                        self._enricher, pair[0], pair[1]
                    ),
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
                    enrich_batch_with_status(
                        self._enricher, batch_chunks, batch_contexts
                    )
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
                    lambda pair: enrich_batch_with_status(
                        self._enricher, pair[0], pair[1]
                    ),
                    batches,
                )
            )
        # Flatten while keeping order.
        return [item for batch_result in batch_results for item in batch_result]
