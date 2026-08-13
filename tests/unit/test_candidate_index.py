"""Focused tests for snapshot-backed Meta-Harness candidate indexes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from stropha.meta import (
    CandidateIndexError,
    CandidateIndexProvider,
    IndexCache,
    Snapshot,
)


class _Stage:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id


class _Storage(_Stage):
    def __init__(self, path: Path) -> None:
        super().__init__("sqlite-vec:dim=1")
        self.path = path
        self.closed = False

    def __enter__(self) -> _Storage:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.closed = True

    def lookup_repo_id(self, _repo_key: str) -> int | None:
        return 17

    def backup_to(self, destination: Path) -> None:
        destination.write_bytes(self.path.read_bytes())


class _Built:
    def __init__(
        self,
        *,
        path: Path,
        indexing_hash: str,
        storage: bool,
        resolved: dict[str, Any],
    ) -> None:
        self.walker = _Stage("walker:v1")
        self.chunker = _Stage("chunker:v1")
        self.enricher = _Stage("enricher:v1")
        self.embedder = _Stage(indexing_hash)
        self.secondary_embedder = None
        self.storage = _Storage(path) if storage else None
        self.retrieval = _Stage("retrieval:v1") if storage else None
        self.resolved = resolved

    def indexing_hash(self) -> str:
        return self.embedder.adapter_id

    def effective_enricher_id(self) -> str:
        return self.enricher.adapter_id


class _Pipeline:
    builds = 0
    graph_fts_augment: ClassVar[list[bool | None]] = []

    def __init__(
        self,
        *,
        storage: _Storage,
        graph_fts_augment: bool | None = None,
        **_kwargs: object,
    ) -> None:
        self.storage = storage
        type(self).graph_fts_augment.append(graph_fts_augment)

    def run(self, *, rebuild: bool, mode: str) -> None:
        assert rebuild is True
        assert mode == "full"
        type(self).builds += 1
        self.storage.path.write_bytes(f"index-{type(self).builds}".encode())


@pytest.fixture
def provider_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import stropha.meta.candidate_index as mod

    source = tmp_path / "source"
    snapshot_path = tmp_path / "snapshot"
    source.mkdir()
    snapshot_path.mkdir()
    snapshot = Snapshot(sha="a" * 40, path=snapshot_path, source_repo=source)
    (snapshot_path / ".sha").write_text(snapshot.sha)

    def fake_build_stages(
        config: dict[str, Any], *, open_storage: bool = True, force_rebuild: bool = False
    ) -> _Built:
        del force_rebuild
        path = Path(config["storage"]["config"]["path"])
        embedder = config["embedder"]
        indexing_hash = f"{embedder['adapter']}:{embedder.get('config', {}).get('model', '')}"
        return _Built(
            path=path,
            indexing_hash=indexing_hash,
            storage=open_storage,
            resolved=config,
        )

    monkeypatch.setattr(mod, "build_stages", fake_build_stages)
    monkeypatch.setattr(mod, "Pipeline", _Pipeline)
    monkeypatch.setattr(mod, "resolve_sha", lambda _ref, path: (Path(path) / ".sha").read_text())
    monkeypatch.setattr(
        mod,
        "detect",
        lambda path: SimpleNamespace(normalized_key=f"example.test/org/{Path(path).name}"),
    )
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(
                enricher_concurrency=1,
                effective_batch_size=lambda: 1,
            )
        ),
    )
    _Pipeline.builds = 0
    _Pipeline.graph_fts_augment = []
    provider = CandidateIndexProvider(
        cache=IndexCache(tmp_path / "cache"),
        workspace_root=tmp_path / "workspaces",
    )
    return provider, snapshot


def test_pin_snapshot_delegates_caller_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stropha.meta.candidate_index as mod

    expected = Snapshot("b" * 40, tmp_path / "snapshot", tmp_path / "source")
    called: dict[str, object] = {}

    def fake_create_snapshot(**kwargs: object) -> Snapshot:
        called.update(kwargs)
        return expected

    monkeypatch.setattr(mod, "create_snapshot", fake_create_snapshot)
    provider = CandidateIndexProvider(
        cache=IndexCache(tmp_path / "cache"),
        workspace_root=tmp_path / "workspaces",
    )
    result = provider.pin_snapshot(
        source_repo=tmp_path / "source",
        ref="release",
        snapshot_root=tmp_path / "snapshots",
    )

    assert result is expected
    assert called == {
        "source_repo": tmp_path / "source",
        "ref": "release",
        "snapshot_root": tmp_path / "snapshots",
    }


def test_identity_contains_every_cache_dimension(provider_env) -> None:
    provider, snapshot = provider_env
    identity = provider.identity_for(snapshot=snapshot, candidate_config={})

    assert identity.repo_key == "example.test/org/source"
    assert identity.snapshot_sha == snapshot.sha
    assert identity.indexing_hash == "local:"
    assert identity.graph_identity == "disabled"
    assert identity.graph_fts_augment is True
    assert identity.cache_format
    assert len(identity.cache_key) == 64


def test_every_identity_dimension_changes_cache_key(provider_env, tmp_path: Path) -> None:
    provider, snapshot = provider_env
    base = provider.identity_for(snapshot=snapshot, candidate_config={})

    other_source = tmp_path / "other-source"
    other_source.mkdir()
    repo_snapshot_path = tmp_path / "repo-snapshot"
    repo_snapshot_path.mkdir()
    (repo_snapshot_path / ".sha").write_text(snapshot.sha)
    repo_snapshot = Snapshot(snapshot.sha, repo_snapshot_path, other_source)

    sha_snapshot_path = tmp_path / "sha-snapshot"
    sha_snapshot_path.mkdir()
    other_sha = "b" * 40
    (sha_snapshot_path / ".sha").write_text(other_sha)
    sha_snapshot = Snapshot(other_sha, sha_snapshot_path, snapshot.source_repo)

    graph_provider = CandidateIndexProvider(
        cache=provider.cache,
        workspace_root=tmp_path / "graph-workspaces",
        graphify=True,
        graph_identity="graphify:v2",
    )
    format_provider = CandidateIndexProvider(
        cache=provider.cache,
        workspace_root=tmp_path / "format-workspaces",
        cache_format="candidate-sqlite-v2",
    )
    identities = [
        base,
        provider.identity_for(snapshot=repo_snapshot, candidate_config={}),
        provider.identity_for(snapshot=sha_snapshot, candidate_config={}),
        provider.identity_for(
            snapshot=snapshot,
            candidate_config={
                "embedder": {"adapter": "local", "config": {"model": "other"}}
            },
        ),
        graph_provider.identity_for(snapshot=snapshot, candidate_config={}),
        format_provider.identity_for(snapshot=snapshot, candidate_config={}),
        provider.identity_for(
            snapshot=snapshot,
            candidate_config={"graph_fts_augment": False},
        ),
    ]

    assert len({identity.cache_key for identity in identities}) == len(identities)


def test_full_flat_candidate_is_resolved_for_index_and_evaluation(provider_env) -> None:
    provider, snapshot = provider_env
    candidate = {
        "walker": "filesystem",
        "chunker": "tree-sitter-dispatch",
        "enricher": "noop",
        "embedder": "model2vec",
        "storage": "sqlite-vec",
        "retrieval": "hybrid-rrf",
        "reranker": "code-aware",
        "hyde": True,
        "hyde_code_tuned": False,
        "query_rewrite": True,
        "multi_query": True,
        "multi_query_count": 5,
        "recursive": True,
        "recursive_adjacency": 20,
        "adaptive_weighting": False,
        "candidate_k": 50,
        "rrf_k": 100,
        "graph_fts_augment": False,
    }

    with provider.open_candidate(
        snapshot=snapshot,
        candidate_config=candidate,
        run_id="run-flat",
        candidate_id="candidate-flat",
    ) as opened:
        assert opened.built.resolved["walker"]["adapter"] == "filesystem"
        assert opened.built.resolved["embedder"]["adapter"] == "model2vec"
        retrieval = opened.built.resolved["retrieval"]["config"]
        assert retrieval["candidate_k"] == 50
        assert retrieval["rrf_k"] == 100
        assert retrieval["reranker"] == {"adapter": "code-aware"}
        assert opened.resolved_config.retrieval_features == {"hyde_code_tuned": False}
        assert opened.resolved_config.pipeline["retrieval"]["config"][
            "inherit_global_features"
        ] is False
        assert opened.resolved_config.pipeline["retrieval"]["config"][
            "hyde_code_tuned"
        ] is False
        assert opened.resolved_config.graph == {"fts_augment": False}
        assert opened.identity.graph_fts_augment is False

    assert _Pipeline.graph_fts_augment == [False]


def test_graphify_requires_explicit_identity(tmp_path: Path) -> None:
    with pytest.raises(CandidateIndexError, match="graph_identity"):
        CandidateIndexProvider(
            cache=IndexCache(tmp_path / "cache"),
            workspace_root=tmp_path / "workspaces",
            graphify=True,
        )


def test_retrieval_only_configs_reuse_cache_and_clones_are_cleaned(provider_env) -> None:
    provider, snapshot = provider_env
    first_config = {"retrieval": {"config": {"candidate_k": 20}}}
    second_config = {"retrieval": {"config": {"candidate_k": 80}}}

    with provider.open_candidate(
        snapshot=snapshot,
        candidate_config=first_config,
        run_id="run-1",
        candidate_id="candidate-a",
    ) as first:
        first_path = first.index_path
        assert first.cache_hit is False
        assert first.repo_id == 17
        assert first.built.retrieval.adapter_id == "retrieval:v1"
        first_path.write_bytes(b"candidate mutation")
    assert not first_path.exists()

    with provider.open_candidate(
        snapshot=snapshot,
        candidate_config=second_config,
        run_id="run-1",
        candidate_id="candidate-b",
    ) as second:
        second_path = second.index_path
        assert second.cache_hit is True
        assert second_path.read_bytes() == b"index-1"
    assert not second_path.exists()
    assert _Pipeline.builds == 1


def test_indexing_config_change_creates_cache_miss(provider_env) -> None:
    provider, snapshot = provider_env

    for model in ("model-a", "model-b"):
        with provider.open_candidate(
            snapshot=snapshot,
            candidate_config={"embedder": {"adapter": "local", "config": {"model": model}}},
            run_id="run-1",
            candidate_id=model,
        ) as candidate:
            assert candidate.cache_hit is False

    assert _Pipeline.builds == 2
    assert len(provider.cache.list_entries()) == 2


def test_candidate_clone_is_cleaned_when_caller_raises(provider_env) -> None:
    provider, snapshot = provider_env
    clone_path: Path | None = None

    with (
        pytest.raises(RuntimeError, match="evaluation failed"),
        provider.open_candidate(
            snapshot=snapshot,
            candidate_config={},
            run_id="run-1",
            candidate_id="candidate-a",
        ) as candidate,
    ):
        clone_path = candidate.index_path
        raise RuntimeError("evaluation failed")

    assert clone_path is not None
    assert not clone_path.exists()


def test_snapshot_head_drift_is_rejected(provider_env, monkeypatch: pytest.MonkeyPatch) -> None:
    import stropha.meta.candidate_index as mod

    provider, snapshot = provider_env
    monkeypatch.setattr(mod, "resolve_sha", lambda _ref, _path: "b" * 40)

    with pytest.raises(CandidateIndexError, match="HEAD drifted"):
        provider.identity_for(snapshot=snapshot, candidate_config={})
