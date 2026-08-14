"""Snapshot-backed, isolated indexes for Meta-Harness candidates."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import get_config
from ..errors import StrophaError
from ..evaluation.postgres import DatabaseRef, PostgresEvaluationDatabaseManager
from ..ingest.git_meta import detect
from ..pipeline.builder import BuiltStages, build_stages
from ..pipeline.config import pipeline_config_from_dict
from ..pipeline.pipeline import Pipeline
from ..storage.sqlite import SCHEMA_VERSION
from .index_cache import IndexCache
from .search_space import ResolvedCandidateConfig, SearchSpace, load_search_space
from .snapshot import Snapshot, create_snapshot, resolve_sha

CANDIDATE_INDEX_CACHE_FORMAT = f"candidate-sqlite-v1-schema-{SCHEMA_VERSION}"
POSTGRES_CANDIDATE_INDEX_CACHE_FORMAT = f"candidate-postgres-v1-schema-{SCHEMA_VERSION}"
_GRAPH_DISABLED_IDENTITY = "disabled"
_POSTGRES_IDENTITY_DSN = "postgresql://managed-evaluation/candidate-index"


class CandidateIndexError(StrophaError):
    """Raised when a candidate index cannot be prepared safely."""


def candidate_index_cache_format(storage_adapter: str) -> str:
    """Return the schema-bound candidate cache format for a storage backend."""
    if storage_adapter == "sqlite-vec":
        return CANDIDATE_INDEX_CACHE_FORMAT
    if storage_adapter == "postgres":
        return POSTGRES_CANDIDATE_INDEX_CACHE_FORMAT
    raise CandidateIndexError(f"unsupported candidate index storage adapter: {storage_adapter!r}")


@dataclass(frozen=True)
class CandidateIndexIdentity:
    """Immutable identity of one canonical candidate index."""

    repo_key: str
    snapshot_sha: str
    indexing_hash: str
    graph_identity: str
    graph_fts_augment: bool
    cache_format: str
    cache_key: str


@dataclass(frozen=True)
class CandidateIndex:
    """Open candidate stages backed by isolated writable storage."""

    built: BuiltStages
    repo_id: int
    identity: CandidateIndexIdentity
    index_path: Path | None
    cache_hit: bool
    resolved_config: ResolvedCandidateConfig
    database_name: str | None = None
    storage_ref: DatabaseRef | None = None


class CandidateIndexProvider:
    """Build, cache, and open snapshot-pinned candidate indexes."""

    def __init__(
        self,
        *,
        cache: IndexCache,
        workspace_root: Path | str,
        graphify: bool = False,
        graph_identity: str | None = None,
        cache_format: str | None = None,
        base_pipeline_config: Mapping[str, Any] | None = None,
        search_space: SearchSpace | None = None,
        postgres_manager: PostgresEvaluationDatabaseManager | None = None,
    ) -> None:
        if graphify:
            if not graph_identity:
                raise CandidateIndexError("graph_identity is required when graphify is enabled")
            resolved_graph_identity = graph_identity
        else:
            resolved_graph_identity = _GRAPH_DISABLED_IDENTITY
        if cache_format is not None and not cache_format:
            raise CandidateIndexError("cache_format must not be empty")
        self.cache = cache
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if postgres_manager is None:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.graphify = graphify
        self.graph_identity = resolved_graph_identity
        self.postgres_manager = postgres_manager
        storage_adapter = "postgres" if postgres_manager is not None else "sqlite-vec"
        self.cache_format = cache_format or candidate_index_cache_format(storage_adapter)
        self.base_pipeline_config = copy.deepcopy(dict(base_pipeline_config or {}))
        self.search_space = search_space or load_search_space(strict=True)
        self._build_lock = threading.Lock()

    def pin_snapshot(
        self,
        *,
        source_repo: Path,
        ref: str = "HEAD",
        snapshot_root: Path | None = None,
    ) -> Snapshot:
        """Create or reuse the caller-requested pinned source snapshot."""
        return create_snapshot(
            source_repo=source_repo,
            ref=ref,
            snapshot_root=snapshot_root,
        )

    def identity_for(
        self,
        *,
        snapshot: Snapshot,
        candidate_config: Mapping[str, Any],
    ) -> CandidateIndexIdentity:
        """Resolve the full cache identity without opening candidate storage."""
        self._validate_snapshot(snapshot)
        resolved = self._resolve_candidate(candidate_config)
        self._validate_resolved_candidate(resolved)
        storage_target: Path | str = (
            _POSTGRES_IDENTITY_DSN
            if self.postgres_manager is not None
            else self.workspace_root / ".identity.db"
        )
        config = self._config_for_index(resolved.pipeline, storage_target)
        built = build_stages(config, open_storage=False)
        repo_key = detect(snapshot.source_repo).normalized_key
        indexing_hash = built.indexing_hash()
        payload: dict[str, str] = {
            "cache_format": self.cache_format,
            "graph_identity": self.graph_identity,
            "graph_fts_augment": str(resolved.graph.get("fts_augment", True)),
            "indexing_hash": indexing_hash,
            "repo_key": repo_key,
            "snapshot_sha": snapshot.sha,
        }
        if self.postgres_manager is not None:
            payload["storage_config"] = json.dumps(
                config["storage"], sort_keys=True, separators=(",", ":")
            )
            cluster = self.postgres_manager.cluster_identity
            payload["storage_runtime"] = json.dumps(
                {
                    "database": cluster.database,
                    "host": cluster.host,
                    "port": cluster.port,
                    "server_version": cluster.server_version,
                    "template_database": cluster.template_database,
                    "template_extensions": list(cluster.template_extensions),
                    "user": cluster.user,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CandidateIndexIdentity(
            repo_key=repo_key,
            snapshot_sha=snapshot.sha,
            indexing_hash=indexing_hash,
            graph_identity=self.graph_identity,
            graph_fts_augment=bool(resolved.graph.get("fts_augment", True)),
            cache_format=self.cache_format,
            cache_key=cache_key,
        )

    @contextlib.contextmanager
    def open_candidate(
        self,
        *,
        snapshot: Snapshot,
        candidate_config: Mapping[str, Any],
        run_id: str,
        candidate_id: str,
    ) -> Iterator[CandidateIndex]:
        """Yield open candidate stages and remove their writable clone on exit."""
        resolved = self._resolve_candidate(candidate_config)
        identity = self.identity_for(snapshot=snapshot, candidate_config=candidate_config)
        if self.postgres_manager is not None:
            cache_built = False

            def build_cache(database: DatabaseRef) -> None:
                nonlocal cache_built
                cache_built = True
                self._build_postgres_cache_entry(
                    snapshot=snapshot,
                    resolved=resolved,
                    identity=identity,
                    database=database,
                )

            canonical = self.postgres_manager.get_or_create_cache(
                identity.cache_key,
                build_cache,
            )
            with self.postgres_manager.writable_clone(canonical) as clone:
                config = self._config_for_index(resolved.pipeline, clone.dsn)
                built = build_stages(config)
                with built.storage:
                    repo_id = self._repo_id(built, snapshot)
                    yield CandidateIndex(
                        built=built,
                        repo_id=repo_id,
                        identity=identity,
                        index_path=None,
                        cache_hit=not cache_built,
                        resolved_config=resolved,
                        database_name=clone.name,
                        storage_ref=clone,
                    )
            return

        cache_hit = (
            self.cache.get(
                identity.cache_key,
                source_repo=snapshot.source_repo,
                sha=snapshot.sha,
            )
            is not None
        )
        if not cache_hit:
            with self._build_lock:
                cache_hit = (
                    self.cache.get(
                        identity.cache_key,
                        source_repo=snapshot.source_repo,
                        sha=snapshot.sha,
                    )
                    is not None
                )
                if not cache_hit:
                    self._build_cache_entry(
                        snapshot=snapshot,
                        candidate_config=candidate_config,
                        identity=identity,
                    )

        prefix = _workspace_prefix(run_id=run_id, candidate_id=candidate_id)
        clone_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.workspace_root))
        clone_path = clone_dir / "index.db"
        try:
            self.cache.clone_writable(
                indexing_hash=identity.cache_key,
                destination=clone_path,
                source_repo=snapshot.source_repo,
                sha=snapshot.sha,
            )
            config = self._config_for_index(resolved.pipeline, clone_path)
            built = build_stages(config)
            with built.storage:
                repo_id = self._repo_id(built, snapshot)
                yield CandidateIndex(
                    built=built,
                    repo_id=repo_id,
                    identity=identity,
                    index_path=clone_path,
                    cache_hit=cache_hit,
                    resolved_config=resolved,
                )
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def _build_cache_entry(
        self,
        *,
        snapshot: Snapshot,
        candidate_config: Mapping[str, Any],
        identity: CandidateIndexIdentity,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=".build-", dir=self.workspace_root) as raw_dir:
            build_dir = Path(raw_dir)
            build_path = build_dir / "index.db"
            backup_path = build_dir / "cache.db"
            resolved = self._resolve_candidate(candidate_config)
            config = self._config_for_index(resolved.pipeline, build_path)
            built = build_stages(config, force_rebuild=True)
            inference = get_config().inference
            with built.storage as storage:
                pipeline = Pipeline(
                    storage=storage,
                    embedder=built.embedder,
                    secondary_embedder=built.secondary_embedder,
                    enricher=built.enricher,
                    walker=built.walker,
                    chunker=built.chunker,
                    repos=[snapshot.path],
                    graphify=self.graphify,
                    graph_fts_augment=identity.graph_fts_augment,
                    enricher_concurrency=inference.enricher_concurrency,
                    enricher_batch_size=inference.effective_batch_size(),
                )
                pipeline.run(rebuild=True, mode="full")
                backup_to = getattr(storage, "backup_to", None)
                if not callable(backup_to):
                    raise CandidateIndexError("sqlite-vec storage does not support WAL-safe backup")
                backup_to(backup_path)

            self.cache.put(
                indexing_hash=identity.cache_key,
                index_path=backup_path,
                adapter_ids=_indexing_adapter_ids(built),
                sha=snapshot.sha,
                source_repo=snapshot.source_repo,
            )

    def _build_postgres_cache_entry(
        self,
        *,
        snapshot: Snapshot,
        resolved: ResolvedCandidateConfig,
        identity: CandidateIndexIdentity,
        database: DatabaseRef,
    ) -> None:
        config = self._config_for_index(resolved.pipeline, database.dsn)
        built = build_stages(config, force_rebuild=True)
        inference = get_config().inference
        with built.storage as storage:
            pipeline = Pipeline(
                storage=storage,
                embedder=built.embedder,
                secondary_embedder=built.secondary_embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=[snapshot.path],
                graphify=self.graphify,
                graph_fts_augment=identity.graph_fts_augment,
                enricher_concurrency=inference.enricher_concurrency,
                enricher_batch_size=inference.effective_batch_size(),
            )
            stats = pipeline.run(rebuild=True, mode="full")
            if stats.chunks_enriched_failed:
                raise CandidateIndexError(
                    "PostgreSQL candidate index enrichment failed for "
                    f"{stats.chunks_enriched_failed} chunks"
                )
            storage_stats = storage.stats()
            chunks = int(storage_stats.get("chunks", 0))
            models = storage_stats.get("models") or []
            vectors = sum(int(model.get("n", 0)) for model in models if isinstance(model, Mapping))
            if chunks == 0 or vectors != chunks:
                raise CandidateIndexError(
                    f"PostgreSQL candidate index is incomplete: chunks={chunks}, vectors={vectors}"
                )
            self._repo_id(built, snapshot)
            storage.set_meta("candidate_index_complete", identity.cache_key)
            storage.commit()

    def _config_for_index(
        self,
        candidate_config: Mapping[str, Any],
        storage_target: Path | str,
    ) -> dict[str, Any]:
        config = pipeline_config_from_dict(dict(candidate_config), resolve_storage_path=False)
        if self.postgres_manager is not None:
            config["storage"] = {
                "adapter": "postgres",
                "config": {"dsn": str(storage_target)},
            }
        else:
            config["storage"] = {
                "adapter": "sqlite-vec",
                "config": {"path": str(storage_target)},
            }
        return config

    def _resolve_candidate(
        self,
        candidate_config: Mapping[str, Any],
    ) -> ResolvedCandidateConfig:
        dimension_names = set(self.search_space.by_name)
        is_flat = not candidate_config or (
            set(candidate_config) <= dimension_names
            and all(not isinstance(value, Mapping) for value in candidate_config.values())
        )
        if is_flat:
            resolved = self.search_space.resolve_candidate(
                candidate_config,
                base_pipeline_config=self.base_pipeline_config,
            )
            retrieval_config = resolved.pipeline.setdefault("retrieval", {}).setdefault(
                "config", {}
            )
            retrieval_config["inherit_global_features"] = False
            if "hyde_code_tuned" in resolved.retrieval_features:
                retrieval_config["hyde_code_tuned"] = resolved.retrieval_features["hyde_code_tuned"]
            return resolved
        return ResolvedCandidateConfig(
            pipeline=copy.deepcopy(dict(candidate_config)),
            retrieval_features={},
            graph={},
        )

    @staticmethod
    def _validate_resolved_candidate(resolved: ResolvedCandidateConfig) -> None:
        walker = resolved.pipeline.get("walker") or {}
        if walker.get("adapter") == "git-diff":
            raise CandidateIndexError(
                "git-diff walker cannot build a full snapshot candidate index"
            )

    @staticmethod
    def _repo_id(built: BuiltStages, snapshot: Snapshot) -> int:
        indexed_repo = detect(snapshot.path)
        repo_id = built.storage.lookup_repo_id(indexed_repo.normalized_key)
        if repo_id is None:
            raise CandidateIndexError(
                f"snapshot repository is absent from candidate index: {indexed_repo.normalized_key}"
            )
        return repo_id

    @staticmethod
    def _validate_snapshot(snapshot: Snapshot) -> None:
        if not snapshot.path.is_dir():
            raise CandidateIndexError(f"snapshot path does not exist: {snapshot.path}")
        actual_sha = resolve_sha("HEAD", snapshot.path)
        if actual_sha != snapshot.sha:
            raise CandidateIndexError(
                f"snapshot HEAD drifted: expected {snapshot.sha}, found {actual_sha}"
            )


def _indexing_adapter_ids(built: BuiltStages) -> dict[str, str]:
    return {
        "walker": built.walker.adapter_id,
        "chunker": built.chunker.adapter_id,
        "enricher": built.effective_enricher_id(),
        "embedder": built.embedder.adapter_id,
        "secondary_embedder": (
            built.secondary_embedder.adapter_id if built.secondary_embedder is not None else "none"
        ),
    }


def _workspace_prefix(*, run_id: str, candidate_id: str) -> str:
    label = hashlib.sha256(f"{run_id}\0{candidate_id}".encode()).hexdigest()[:12]
    return f"candidate-{label}-"


__all__ = [
    "CANDIDATE_INDEX_CACHE_FORMAT",
    "POSTGRES_CANDIDATE_INDEX_CACHE_FORMAT",
    "CandidateIndex",
    "CandidateIndexError",
    "CandidateIndexIdentity",
    "CandidateIndexProvider",
    "candidate_index_cache_format",
]
