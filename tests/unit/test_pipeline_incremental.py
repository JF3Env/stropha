"""Tests for Phase A — file-level dirty cache (schema v6).

Covers:
- `Storage.file_is_fresh` semantics (mtime + size + enricher + embedder)
- `Storage.upsert_file_meta` round-trip
- `Storage.delete_file_meta` returns count, removes row
- `Storage.list_stale_files` finds rows not in current set
- `Storage.delete_chunks_by_repo_paths` is repo-scoped
- Pipeline-level: file_skipped_fresh counters fire, chunker is bypassed
- Pipeline-level: file content change (mtime delta) invalidates the skip
- Pipeline-level: stale files get evicted at end of repo run
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pytest

from stropha.adapters.enricher.noop import NoopEnricher
from stropha.embeddings.base import Embedder
from stropha.errors import EmbeddingError, PipelineError
from stropha.ingest.chunkers.base import make_chunk_id
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import (
    SOURCE_SNAPSHOT_VERSION,
    Chunk,
    SourceFile,
    source_snapshot_meta_key,
)
from stropha.pipeline import Pipeline
from stropha.pipeline.pipeline import RepoStats
from stropha.storage import Storage

# --------------------------------------------------------------------------- helpers


class _StubEmbedder(Embedder):
    """4-dim deterministic embedder. Used in tests to avoid loading
    fastembed weights for what are really plumbing tests."""

    @property
    def model_name(self) -> str:
        return "stub:v1"

    @property
    def dim(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def adapter_id(self) -> str:
        return "stub:v1"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _LineChunker:
    """Deterministic test chunker that consumes the captured source text."""

    def chunk(
        self,
        files: Iterable[SourceFile],
        *,
        repo_key: str | None = None,
    ) -> Iterator[Chunk]:
        for source in files:
            assert source.snapshot_content is not None
            for line_number, line in enumerate(source.snapshot_content.splitlines(), 1):
                content_hash = hashlib.sha256(line.encode()).hexdigest()
                yield Chunk(
                    chunk_id=make_chunk_id(
                        source.rel_path,
                        line_number,
                        line_number,
                        content_hash,
                        repo_key=repo_key,
                    ),
                    rel_path=source.rel_path,
                    language=source.language,
                    start_line=line_number,
                    end_line=line_number,
                    content=line,
                    content_hash=content_hash,
                )


class _MutatingEmbedder(_StubEmbedder):
    def __init__(self, path: Path, replacement: str) -> None:
        self._path = path
        self._replacement = replacement
        self.mutated = False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.mutated:
            self._path.write_text(self._replacement, encoding="utf-8")
            self.mutated = True
        return super().embed_documents(texts)


class _FailingEmbedder(_StubEmbedder):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError("injected embedding failure")


class _FlakyEnricher(NoopEnricher):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def adapter_id(self) -> str:
        return "flaky:v1"

    def enrich_with_status(self, chunk: Chunk, _ctx: object) -> tuple[str, bool]:
        self.calls += 1
        if self.calls == 1:
            return chunk.content, False
        return f"enriched:{chunk.content}", True


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    s = Storage(tmp_path / "idx.db", embedding_dim=4)
    yield s
    s.close()


@pytest.fixture
def repo_id(storage: Storage) -> int:
    return storage.register_repo(
        RepoIdentity(
            normalized_key="local:test",
            remote_url=None,
            root_path="/tmp/test",
            default_branch=None,
            head_commit=None,
        )
    )


def _git_init_with_file(repo: Path, name: str, content: str) -> Path:
    import subprocess

    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@x.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    file_path = repo / name
    file_path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return file_path


# --------------------------------------------------------------------------- Storage methods


def test_file_is_fresh_returns_false_on_unknown_file(storage: Storage, repo_id: int) -> None:
    assert storage.file_is_fresh(
        repo_id, "missing.py", content_hash="abc", mtime=100.0, size_bytes=42,
    ) is False


def test_file_is_fresh_returns_true_when_state_matches(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=42,
        enricher_id="noop", embedder_model="m1",
    ) is True


def test_file_is_fresh_returns_false_when_mtime_changes(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=200.0, size_bytes=42,
    ) is False


def test_file_is_fresh_returns_false_when_size_changes(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=99,
    ) is False


def test_file_is_fresh_returns_false_when_enricher_differs(
    storage: Storage, repo_id: int,
) -> None:
    """Drift contract (ADR-004): switching enricher must invalidate fresh
    files even when (mtime, size) match exactly."""
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=42,
        enricher_id="hierarchical", embedder_model="m1",
    ) is False


def test_file_is_fresh_returns_false_when_embedder_differs(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
        enricher_id="noop", embedder_model="m1",
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=42,
        enricher_id="noop", embedder_model="m2",
    ) is False


def test_file_is_fresh_ignores_unspecified_pipeline_keys(
    storage: Storage, repo_id: int,
) -> None:
    """Callers that pass enricher_id=None / embedder_model=None opt out of
    that side of the comparison — useful for storage-only tests."""
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=0,
        enricher_id="ENRICHER", embedder_model="MODEL",
    )
    # Both None → source snapshot check without adapter drift comparison.
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=42,
    ) is True


def test_file_is_fresh_rejects_full_file_hash_mismatch(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="old",
        size_bytes=42, mtime=100.0, chunk_count=0,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="new", mtime=100.0, size_bytes=42,
    ) is False


def test_file_is_fresh_rejects_incomplete_chunk_snapshot(
    storage: Storage, repo_id: int,
) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=1,
    )
    assert storage.file_is_fresh(
        repo_id, "a.py", content_hash="abc", mtime=100.0, size_bytes=42,
    ) is False


@pytest.mark.parametrize("satellite", ["vec_chunks", "fts_chunks"])
def test_file_is_fresh_rejects_missing_primary_satellite(
    storage: Storage, repo_id: int, satellite: str,
) -> None:
    chunk = Chunk(
        chunk_id="satellite-test",
        rel_path="a.py",
        language="python",
        start_line=1,
        end_line=1,
        content="x = 1",
        content_hash="chunk-hash",
    )
    rowid = storage.upsert_chunk(
        chunk, [0.0] * 4, "m1", 4, repo_id=repo_id, enricher_id="noop"
    )
    storage.upsert_file_meta(
        repo_id=repo_id,
        rel_path="a.py",
        content_hash="file-hash",
        size_bytes=5,
        mtime=100.0,
        chunk_count=1,
        enricher_id="noop",
        embedder_model="m1",
    )
    storage.commit()
    assert storage.file_is_fresh(
        repo_id,
        "a.py",
        content_hash="file-hash",
        mtime=100.0,
        size_bytes=5,
        enricher_id="noop",
        embedder_model="m1",
    )

    storage._conn.execute(f"DELETE FROM {satellite} WHERE rowid = ?", (rowid,))

    assert not storage.chunk_is_fresh(
        chunk.chunk_id, chunk.content_hash, "m1", enricher_id="noop"
    )
    assert not storage.file_is_fresh(
        repo_id,
        "a.py",
        content_hash="file-hash",
        mtime=100.0,
        size_bytes=5,
        enricher_id="noop",
        embedder_model="m1",
    )


def test_legacy_file_hash_is_not_exposed_as_source_evidence(
    storage: Storage, repo_id: int,
) -> None:
    chunk = Chunk(
        chunk_id="legacy-source",
        rel_path="legacy.py",
        language="python",
        start_line=1,
        end_line=1,
        content="x = 1",
        content_hash="legacy-chunk-hash",
    )
    storage.upsert_chunk(chunk, [0.0] * 4, "m1", 4, repo_id=repo_id)
    storage.upsert_file_meta(
        repo_id=repo_id,
        rel_path="legacy.py",
        content_hash="legacy-first-chunk-hash",
        size_bytes=5,
        mtime=100.0,
        chunk_count=1,
    )
    storage.commit()

    hit = storage.lookup_by_chunk_id(chunk.chunk_id)

    assert hit is not None
    assert hit.source is None


def test_source_snapshot_marker_is_repository_scoped(
    storage: Storage, repo_id: int,
) -> None:
    legacy_repo_id = storage.register_repo(
        RepoIdentity(
            normalized_key="local:legacy",
            remote_url=None,
            root_path="/tmp/legacy",
            default_branch=None,
            head_commit=None,
        )
    )
    chunk = Chunk(
        chunk_id="legacy-other-repo",
        rel_path="legacy.py",
        language="python",
        start_line=1,
        end_line=1,
        content="x = 1",
        content_hash="legacy-chunk-hash",
    )
    storage.upsert_chunk(chunk, [0.0] * 4, "m1", 4, repo_id=legacy_repo_id)
    storage.upsert_file_meta(
        repo_id=legacy_repo_id,
        rel_path="legacy.py",
        content_hash="legacy-first-chunk-hash",
        size_bytes=5,
        mtime=100.0,
        chunk_count=1,
    )
    storage.set_meta(source_snapshot_meta_key(repo_id), SOURCE_SNAPSHOT_VERSION)
    storage.commit()

    hit = storage.lookup_by_chunk_id(chunk.chunk_id)

    assert hit is not None
    assert hit.source is None


def test_upsert_file_meta_idempotent(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="abc",
        size_bytes=42, mtime=100.0, chunk_count=3,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="def",
        size_bytes=43, mtime=200.0, chunk_count=4,
    )
    rows = storage._conn.execute(
        "SELECT * FROM files WHERE repo_id = ? AND rel_path = ?",
        (repo_id, "a.py"),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == "def"
    assert int(rows[0]["last_chunk_count"]) == 4


def test_delete_file_meta_removes_rows(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="h", size_bytes=1, mtime=1.0, chunk_count=1,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="b.py", content_hash="h", size_bytes=1, mtime=1.0, chunk_count=1,
    )
    count = storage.delete_file_meta(repo_id, ["a.py", "b.py", "missing.py"])
    assert count == 2  # missing.py reports 0


def test_list_stale_files_returns_missing(storage: Storage, repo_id: int) -> None:
    for p in ("a.py", "b.py", "c.py"):
        storage.upsert_file_meta(
            repo_id=repo_id, rel_path=p, content_hash="h",
            size_bytes=1, mtime=1.0, chunk_count=1,
        )
    stale = storage.list_stale_files(repo_id, ["a.py", "c.py"])
    assert set(stale) == {"b.py"}


def test_list_stale_files_empty_current_returns_all(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="x.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    assert storage.list_stale_files(repo_id, []) == ["x.py"]


def test_delete_chunks_by_repo_paths_is_scoped(storage: Storage) -> None:
    """Two repos with identically-named files; deletion in one must NOT
    touch the other."""
    repo_a = storage.register_repo(RepoIdentity(
        normalized_key="r-a", remote_url=None, root_path="/a",
        default_branch=None, head_commit=None,
    ))
    repo_b = storage.register_repo(RepoIdentity(
        normalized_key="r-b", remote_url=None, root_path="/b",
        default_branch=None, head_commit=None,
    ))
    for repo, cid in ((repo_a, "ca"), (repo_b, "cb")):
        c = Chunk(
            chunk_id=cid, rel_path="shared.py", language="python",
            kind="file", symbol=None, parent_chunk_id=None,
            start_line=1, end_line=5,
            content="x", content_hash=cid,
        )
        storage.upsert_chunk(c, [0.0] * 4, "stub", 4, repo_id=repo)
    storage.commit()

    deleted = storage.delete_chunks_by_repo_paths(repo_a, ["shared.py"])
    assert deleted == 1
    remaining = storage._conn.execute(
        "SELECT chunk_id FROM chunks WHERE rel_path = 'shared.py'"
    ).fetchall()
    assert {r["chunk_id"] for r in remaining} == {"cb"}


def test_stats_reports_files_tracked(storage: Storage, repo_id: int) -> None:
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="a.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    storage.upsert_file_meta(
        repo_id=repo_id, rel_path="b.py", content_hash="h",
        size_bytes=1, mtime=1.0, chunk_count=1,
    )
    s = storage.stats()
    assert s["files_tracked"] == 2


# --------------------------------------------------------------------------- Pipeline end-to-end


def test_pipeline_skips_unchanged_files_on_second_run(tmp_path: Path) -> None:
    """The user-visible Phase A win: second run of `stropha index` on an
    unchanged repo records files_skipped_fresh == files_visited and
    chunks_seen == 0 (chunker never ran).

    Pinning `mode="full"` keeps the Phase A semantic — every file goes
    through the walker, but the file-level cache short-circuits before
    chunking. Phase B `auto` mode would route through git-diff instead
    and skip even visiting unchanged files (covered by Phase B tests).
    """
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()

        s1 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")
        assert s1.chunks_embedded >= 1
        assert s1.files_skipped_fresh == 0
        assert s1.repos[0].code_changed is True

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_skipped_fresh == s1.files_visited
        assert s2.chunks_seen == 0  # chunker bypassed
        assert s2.chunks_embedded == 0
        assert s2.repos[0].code_changed is False


def test_pipeline_invalidates_skip_when_file_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")

        # Mutate the file → mtime + size change → must NOT be skipped.
        main.write_text("x = 2\ny = 3\n", encoding="utf-8")
        # Make sure mtime really changes (some filesystems quantize to 1s)
        os.utime(main, (1_000_000_000, 1_000_000_000))

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_skipped_fresh == 0
        assert s2.chunks_seen >= 1
        assert s2.repos[0].code_changed is True


def test_graph_dirty_survives_graph_disabled_run(tmp_path: Path) -> None:
    from unittest.mock import patch

    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run(rebuild=True, mode="full")
        assert storage.get_meta("graph_dirty:1") == "1"

        pipeline = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=True,
        )
        with patch(
            "stropha.pipeline.graphify_runner.run_graphify_multi",
            return_value=[],
        ) as run_graphify:
            failed = pipeline._run_graphify(
                [RepoStats(normalized_key="test", url=None)]
            )

        run_graphify.assert_called_once_with([repo.resolve()])
        assert failed == {repo.resolve()}
        assert storage.get_meta("graph_dirty:1") == "1"


def test_graph_dirty_clears_only_after_mirror_success(tmp_path: Path) -> None:
    from unittest.mock import patch

    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run(rebuild=True, mode="full")

        pipeline = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=True,
        )
        with patch.object(pipeline, "_run_graphify", return_value=set()), patch.object(
            pipeline, "_refresh_graphify_mirror", return_value={1}
        ):
            pipeline.run(mode="full")

        assert storage.get_meta("graph_dirty:1") == "0"


# --------------------------------------------------------------------------- Phase B — git-diff incremental


def test_incremental_falls_back_to_full_on_first_run(tmp_path: Path) -> None:
    """No `last_indexed_sha` stored yet → `--incremental` must NOT error;
    it walks everything and stores the checkpoint for next time."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        # Explicit incremental on a fresh DB.
        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="incremental")
        # Fallback happened — full pass embedded the file.
        assert stats.chunks_embedded >= 1
        # last_indexed_sha is now set so future runs go incremental.
        repos = s.list_repos()
        assert len(repos) == 1
        meta_key = f"last_indexed_sha_{1}"
        assert s.get_meta(meta_key)


def test_incremental_picks_up_new_commit(tmp_path: Path) -> None:
    """Add a file in a 2nd commit; incremental mode embeds just that one."""
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "a.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)

        # Add new file in a fresh commit.
        (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "add b"], cwd=repo, check=True)

        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")
        # Only b.py touched
        assert stats.files_visited == 1
        # First file (a.py) skipped — no work
        assert stats.files_skipped_fresh == 0  # a.py not even visited
        assert stats.files_evicted == 0


def test_incremental_evicts_deleted_file(tmp_path: Path) -> None:
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "keep.py", "k\n")
    (repo / "drop.py").write_text("d\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add drop"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)

        # Delete file in next commit.
        subprocess.run(["git", "rm", "drop.py", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "drop"], cwd=repo, check=True)

        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")
        assert stats.files_evicted == 1
        assert s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path='drop.py' LIMIT 1"
        ).fetchone() is None


def test_incremental_renames_chunks_without_re_embed(tmp_path: Path) -> None:
    """Pure rename → rename_chunks preserves the original chunks (zero
    re-embed work). The file_meta row's rel_path is updated."""
    import subprocess
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "old_name.py", "def foo():\n    return 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)

        # Pure rename, no content change.
        subprocess.run(["git", "mv", "old_name.py", "new_name.py"],
                       cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "rename"], cwd=repo, check=True)

        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="incremental")

        # The chunk_id today still hashes rel_path so the rename causes
        # at least some downstream re-processing. What MUST hold is that
        # the new path is present in chunks and the old is absent.
        rows = s._conn.execute(
            "SELECT rel_path FROM chunks ORDER BY rel_path"
        ).fetchall()
        paths = {r["rel_path"] for r in rows}
        assert "new_name.py" in paths
        assert "old_name.py" not in paths


# --------------------------------------------------------------------------- Phase C — rename-resilient chunk_id


def test_rename_chunks_recomputes_chunk_id(storage: Storage, repo_id: int) -> None:
    """After `rename_chunks`, the chunk row's chunk_id reflects the new
    rel_path (computed via `make_chunk_id`). This is what makes
    `chunk_is_fresh` find the renamed chunks under the new path."""
    from stropha.ingest.chunkers.base import make_chunk_id

    chunk = Chunk(
        chunk_id="placeholder",  # we'll compute the real one below
        rel_path="old.py", language="python",
        kind="function", symbol="foo", parent_chunk_id=None,
        start_line=1, end_line=10,
        content="def foo(): pass", content_hash="hash-foo",
    )
    real_old_id = make_chunk_id("old.py", 1, 10, "hash-foo", repo_key="local:test")
    chunk = chunk.model_copy(update={"chunk_id": real_old_id})
    storage.upsert_chunk(chunk, [0.0] * 4, "stub", 4, repo_id=repo_id)
    storage.commit()

    moved = storage.rename_chunks(repo_id, "old.py", "new.py")
    assert moved == 1

    row = storage._conn.execute(
        "SELECT chunk_id, rel_path FROM chunks WHERE rel_path = 'new.py'"
    ).fetchone()
    assert row is not None
    expected_new_id = make_chunk_id(
        "new.py", 1, 10, "hash-foo", repo_key="local:test",
    )
    assert row["chunk_id"] == expected_new_id
    assert row["chunk_id"] != real_old_id


def test_rename_chunks_regenerates_fts5_row(storage: Storage, repo_id: int) -> None:
    """FTS5 path tokens come from the chunk's rel_path. After rename,
    BM25 queries on the new path must find the chunk."""
    from stropha.ingest.chunkers.base import make_chunk_id

    chunk = Chunk(
        chunk_id=make_chunk_id("src/old_thing.py", 1, 5, "h", repo_key="local:test"),
        rel_path="src/old_thing.py", language="python",
        kind="function", symbol="thing", parent_chunk_id=None,
        start_line=1, end_line=5,
        content="def thing(): pass", content_hash="h",
    )
    storage.upsert_chunk(chunk, [0.0] * 4, "stub", 4, repo_id=repo_id)
    storage.commit()

    # Query for the old path token surfaces it.
    hits_old = storage.search_bm25("old_thing", k=5)
    assert any(h.rel_path == "src/old_thing.py" for h in hits_old)

    storage.rename_chunks(repo_id, "src/old_thing.py", "src/new_thing.py")
    storage.commit()

    # Now the new path token surfaces it.
    hits_new = storage.search_bm25("new_thing", k=5)
    assert any(h.rel_path == "src/new_thing.py" for h in hits_new)


def test_rename_preserves_chunk_is_fresh(storage: Storage, repo_id: int) -> None:
    """After rename, the new chunk_id matches what `make_chunk_id` would
    produce for the new path — so `chunk_is_fresh(new_id, ...)` returns
    True, which is what saves re-embed cost on the next index run."""
    from stropha.ingest.chunkers.base import make_chunk_id

    chunk = Chunk(
        chunk_id=make_chunk_id("a.py", 1, 5, "hc", repo_key="local:test"),
        rel_path="a.py", language="python",
        kind="function", symbol=None, parent_chunk_id=None,
        start_line=1, end_line=5,
        content="x", content_hash="hc",
    )
    storage.upsert_chunk(chunk, [0.0] * 4, "m", 4, repo_id=repo_id)
    storage.rename_chunks(repo_id, "a.py", "b.py")
    storage.commit()

    new_id = make_chunk_id("b.py", 1, 5, "hc", repo_key="local:test")
    assert storage.chunk_is_fresh(new_id, "hc", "m") is True


def test_rename_chunks_handles_empty_path(storage: Storage, repo_id: int) -> None:
    """Renaming a path with no chunks under it is a no-op."""
    moved = storage.rename_chunks(repo_id, "ghost.py", "ghost_renamed.py")
    assert moved == 0


def test_rename_same_path_is_noop(storage: Storage, repo_id: int) -> None:
    moved = storage.rename_chunks(repo_id, "x.py", "x.py")
    assert moved == 0


def test_full_flag_overrides_incremental_default(tmp_path: Path) -> None:
    """Even after a checkpoint exists, mode='full' walks everything."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "a.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True)
        # Touch the file mtime artificially so file_is_fresh fails:
        # we want to confirm that mode='full' calls discover() on every
        # file, not git diff. The simplest check is that files_visited
        # > 0 even when no commits happened since last index.
        stats = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        # Walker visited the file (it was just skipped via cache).
        assert stats.files_visited == 1


def test_pipeline_evicts_stale_files_at_end_of_run(tmp_path: Path) -> None:
    """A file present at run 1 but deleted before run 2 has its chunks +
    file_meta evicted (Phase A passive stale cleanup)."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "keep.py", "x = 1\n")
    transient = repo / "transient.py"
    transient.write_text("y = 2\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add transient"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        embedder = _StubEmbedder()
        s1 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(rebuild=True, mode="full")
        assert s1.files_visited == 2

        # Remove the transient file from the working tree AND from git.
        transient.unlink()
        subprocess.run(["git", "rm", "transient.py", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "drop transient"], cwd=repo, check=True)

        s2 = Pipeline(
            storage=s, embedder=embedder, enricher=NoopEnricher(),
            repos=[repo],
        ).run(mode="full")
        assert s2.files_evicted == 1
        # No chunks for transient.py remain.
        remaining = s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path = 'transient.py' LIMIT 1"
        ).fetchone()
        assert remaining is None
        # file_meta row also gone.
        assert s._conn.execute(
            "SELECT 1 FROM files WHERE rel_path = 'transient.py' LIMIT 1"
        ).fetchone() is None


# --------------------------------------------------------------------------- WIN-425 — hash-based reconcile (auto default)


class _StubEmbedderV2(_StubEmbedder):
    """Same dim as _StubEmbedder, different model_name — simulates an
    embedder *config change* (drift) that needs no schema rebuild."""

    @property
    def model_name(self) -> str:
        return "stub:v2"

    @property
    def adapter_id(self) -> str:
        return "stub:v2"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.5, 0.6, 0.7, 0.8] for _ in texts]


def test_auto_mode_routes_to_reconcile(tmp_path: Path) -> None:
    """auto (the default) reconciles by hash: first run indexes, second
    run on an unchanged tree skips every file as fresh (no re-chunk)."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")
    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        emb = _StubEmbedder()
        s1 = Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()  # default mode == "auto" -> reconcile
        assert s1.chunks_embedded >= 1
        s2 = Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()
        assert s2.files_skipped_fresh >= 1
        assert s2.chunks_seen == 0  # nothing re-chunked
        assert s2.repos[0].code_changed is False


def test_reconcile_detects_uncommitted_edit(tmp_path: Path) -> None:
    """Headline WIN-425 win: a tracked file edited but NOT committed is
    re-indexed by auto mode. git-diff incremental would never see it."""
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")
    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        emb = _StubEmbedder()
        Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()

        # Edit WITHOUT committing; bump mtime so the (mtime,size) fast-path fires.
        main.write_text("x = 2\ny = 3\n", encoding="utf-8")
        os.utime(main, (1_000_000_000, 1_000_000_000))

        stats = Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()
        assert stats.files_skipped_fresh == 0
        assert stats.chunks_seen >= 1
        assert stats.chunks_embedded >= 1
        assert stats.repos[0].code_changed is True


def test_reconcile_recovers_from_interrupted_index(tmp_path: Path) -> None:
    """Simulate a full index that died mid-pass: one file's chunks +
    files-meta row never got written. The next auto run re-indexes exactly
    that file — no --rebuild, independent of the git checkpoint."""
    import subprocess

    repo = tmp_path / "repo"
    _git_init_with_file(repo, "a.py", "a = 1\n")
    (repo / "b.py").write_text("b = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add b"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        emb = _StubEmbedder()
        Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()
        repo_id = s._conn.execute("SELECT id FROM repos LIMIT 1").fetchone()["id"]

        # Simulate the interruption: drop b.py's index state entirely.
        s.delete_chunks_by_repo_paths(repo_id, ["b.py"])
        s.delete_file_meta(repo_id, ["b.py"])
        s.commit()

        stats = Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()
        # a.py is fresh (skipped); b.py had no meta -> reconciled.
        assert stats.files_skipped_fresh == 1
        assert stats.chunks_seen >= 1
        assert s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path = 'b.py' LIMIT 1"
        ).fetchone() is not None


def test_reconcile_evicts_file_deleted_from_worktree(tmp_path: Path) -> None:
    """A tracked file deleted only on disk is evicted on the next auto run."""
    import subprocess

    repo = tmp_path / "repo"
    _git_init_with_file(repo, "keep.py", "k = 1\n")
    (repo / "drop.py").write_text("d = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add drop"], cwd=repo, check=True)

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        emb = _StubEmbedder()
        Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()

        (repo / "drop.py").unlink()

        stats = Pipeline(
            storage=s, embedder=emb, enricher=NoopEnricher(), repos=[repo],
        ).run()
        assert stats.files_evicted == 1
        assert stats.repos[0].code_changed is True
        assert s._conn.execute(
            "SELECT 1 FROM chunks WHERE rel_path = 'drop.py' LIMIT 1"
        ).fetchone() is None


def test_pipeline_clears_obsolete_path_identity(tmp_path: Path) -> None:
    """Fallback repo data cannot survive after a checkout gains a remote identity."""
    import subprocess

    repo = tmp_path / "repo"
    _git_init_with_file(repo, "current.py", "current = True\n")
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.com/acme/repo.git"],
        cwd=repo,
        check=True,
    )

    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        fallback_id = s.register_repo(
            RepoIdentity(
                normalized_key=f"path:{repo.resolve()}",
                remote_url=None,
                root_path=repo.resolve(),
                default_branch=None,
                head_commit=None,
            )
        )
        stale = Chunk(
            chunk_id="sha256:obsolete-path-identity",
            rel_path="deleted.py",
            language="python",
            kind="module",
            symbol=None,
            parent_chunk_id=None,
            start_line=1,
            end_line=1,
            content="deleted = True",
            content_hash="obsolete",
        )
        s.upsert_chunk(stale, [0.1] * 4, "stub:v1", 4, repo_id=fallback_id)
        s.commit()

        Pipeline(
            storage=s,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
        ).run()

        assert s._conn.execute(
            "SELECT 1 FROM chunks WHERE repo_id = ? LIMIT 1", (fallback_id,)
        ).fetchone() is None


def test_reconcile_reindexes_on_embedder_config_change(tmp_path: Path) -> None:
    """Config-drift contract (CLAUDE.md 3.1): swapping the embedder
    (model_name) forces reprocessing through reconcile even though the file
    content is byte-identical. Proves config changes still drive indexing."""
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")
    with Storage(tmp_path / "idx.db", embedding_dim=4) as s:
        Pipeline(
            storage=s, embedder=_StubEmbedder(), enricher=NoopEnricher(),
            repos=[repo],
        ).run()
        # Identical content, different embedder config -> must NOT skip.
        stats = Pipeline(
            storage=s, embedder=_StubEmbedderV2(), enricher=NoopEnricher(),
            repos=[repo],
        ).run()
        assert stats.files_skipped_fresh == 0
        assert stats.chunks_embedded >= 1
        assert stats.repos[0].code_changed is False


def test_reconcile_detects_same_size_edit_with_preserved_mtime(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()
        original = main.stat()
        main.write_text("x = 2\n", encoding="utf-8")
        os.utime(main, ns=(original.st_atime_ns, original.st_mtime_ns))

        stats = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()

        row = storage._conn.execute(
            "SELECT content_hash, size_bytes, mtime FROM files WHERE rel_path = 'main.py'"
        ).fetchone()
        assert row is not None
        assert stats.files_skipped_fresh == 0
        assert stats.repos[0].code_changed is True
        assert row["content_hash"] == hashlib.sha256(main.read_bytes()).hexdigest()
        assert int(row["size_bytes"]) == original.st_size
        assert float(row["mtime"]) == original.st_mtime
        chunk_row = storage._conn.execute(
            "SELECT chunk_id, content FROM chunks WHERE rel_path = 'main.py'"
        ).fetchone()
        assert "x = 2" in chunk_row["content"]
        hit = storage.lookup_by_chunk_id(chunk_row["chunk_id"])
        assert hit is not None
        assert hit.source is not None
        assert hit.source.source_version.layer == "saved_worktree"
        assert hit.source.source_version.file_sha256 == row["content_hash"]
        assert hit.source.indexed_file_sha256 == row["content_hash"]
        assert hit.source.source_slice_sha256 is None


def test_rechunk_removes_obsolete_chunks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "alpha\nbeta\ngamma\n")
    chunker = _LineChunker()

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            chunker=chunker,  # type: ignore[arg-type]
            graphify=False,
        ).run()
        old_ids = {
            row["chunk_id"]
            for row in storage._conn.execute(
                "SELECT chunk_id FROM chunks WHERE rel_path = 'main.py'"
            ).fetchall()
        }
        assert len(old_ids) == 3

        main.write_text("alpha\n", encoding="utf-8")
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            chunker=chunker,  # type: ignore[arg-type]
            graphify=False,
        ).run()

        rows = storage._conn.execute(
            "SELECT chunk_id, content FROM chunks WHERE rel_path = 'main.py'"
        ).fetchall()
        current_ids = {row["chunk_id"] for row in rows}
        assert [row["content"] for row in rows] == ["alpha"]
        assert len(current_ids) == 1
        for obsolete_id in old_ids - current_ids:
            assert storage.chunk_exists(obsolete_id) is False


def test_pipeline_retries_when_source_changes_during_embedding(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "first\n")
    chunker = _LineChunker()

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            chunker=chunker,  # type: ignore[arg-type]
            graphify=False,
        ).run()
        main.write_text("second\n", encoding="utf-8")
        embedder = _MutatingEmbedder(main, "latest\n")

        stats = Pipeline(
            storage=storage,
            embedder=embedder,
            enricher=NoopEnricher(),
            repos=[repo],
            chunker=chunker,  # type: ignore[arg-type]
            graphify=False,
        ).run()

        assert embedder.mutated is True
        row = storage._conn.execute(
            "SELECT content_hash FROM files WHERE rel_path = 'main.py'"
        ).fetchone()
        assert row["content_hash"] == hashlib.sha256(b"latest\n").hexdigest()
        assert [
            item["content"]
            for item in storage._conn.execute(
                "SELECT content FROM chunks WHERE rel_path = 'main.py'"
            ).fetchall()
        ] == ["latest"]
        assert stats.chunks_embedded == 1


def test_failed_index_preserves_snapshot_and_repository_watermark(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()
        before_repo = storage._conn.execute(
            "SELECT id, head_commit, last_indexed_at FROM repos LIMIT 1"
        ).fetchone()
        before_chunks = storage._conn.execute(
            "SELECT chunk_id, content FROM chunks ORDER BY chunk_id"
        ).fetchall()
        before_checkpoint = storage.get_meta(
            f"last_indexed_sha_{int(before_repo['id'])}"
        )

        main.write_text("x = 2\n", encoding="utf-8")
        subprocess.run(["git", "add", "main.py"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "change"], cwd=repo, check=True)

        with pytest.raises(EmbeddingError, match="injected embedding failure"):
            Pipeline(
                storage=storage,
                embedder=_FailingEmbedder(),
                enricher=NoopEnricher(),
                repos=[repo],
                graphify=False,
            ).run()

        after_repo = storage._conn.execute(
            "SELECT head_commit, last_indexed_at FROM repos LIMIT 1"
        ).fetchone()
        after_chunks = storage._conn.execute(
            "SELECT chunk_id, content FROM chunks ORDER BY chunk_id"
        ).fetchall()
        assert dict(after_repo) == {
            "head_commit": before_repo["head_commit"],
            "last_indexed_at": before_repo["last_indexed_at"],
        }
        assert [tuple(row) for row in after_chunks] == [
            tuple(row) for row in before_chunks
        ]
        assert storage.get_meta(
            f"last_indexed_sha_{int(before_repo['id'])}"
        ) == before_checkpoint


def test_failed_first_index_does_not_publish_repository_watermark(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        with pytest.raises(EmbeddingError, match="injected embedding failure"):
            Pipeline(
                storage=storage,
                embedder=_FailingEmbedder(),
                enricher=NoopEnricher(),
                repos=[repo],
                graphify=False,
            ).run()

        row = storage._conn.execute(
            "SELECT head_commit, last_indexed_at FROM repos LIMIT 1"
        ).fetchone()
        assert row["head_commit"] is None
        assert row["last_indexed_at"] == ""
        assert storage.list_repos()[0].last_indexed_at is None
        assert storage._conn.execute("SELECT 1 FROM files LIMIT 1").fetchone() is None
        assert storage._conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is None


def test_failed_index_releases_storage_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")
    released: list[bool] = []

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        monkeypatch.setattr(
            storage, "try_acquire_index_lock", lambda: True, raising=False
        )
        monkeypatch.setattr(
            storage,
            "release_index_lock",
            lambda: released.append(True),
            raising=False,
        )

        with pytest.raises(EmbeddingError, match="injected embedding failure"):
            Pipeline(
                storage=storage,
                embedder=_FailingEmbedder(),
                enricher=NoopEnricher(),
                repos=[repo],
                graphify=False,
            ).run()

    assert released == [True]


def test_zero_chunk_file_publishes_complete_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    empty = _git_init_with_file(repo, "empty.py", " \n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        first = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()
        row = storage._conn.execute(
            "SELECT content_hash, last_chunk_count FROM files WHERE rel_path = 'empty.py'"
        ).fetchone()
        second = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()

        assert first.chunks_seen == 0
        assert row["content_hash"] == hashlib.sha256(empty.read_bytes()).hexdigest()
        assert int(row["last_chunk_count"]) == 0
        assert second.files_skipped_fresh == 1


def test_snapshot_upgrade_forces_reconcile_in_incremental_mode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    main = _git_init_with_file(repo, "main.py", "x = 1\n")

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run()
        repo_id = int(
            storage._conn.execute("SELECT id FROM repos LIMIT 1").fetchone()["id"]
        )
        storage._conn.execute(
            "UPDATE files SET content_hash = 'legacy-first-chunk-hash' "
            "WHERE rel_path = 'main.py'"
        )
        storage._conn.execute(
            "DELETE FROM meta WHERE key = ?",
            (source_snapshot_meta_key(repo_id),),
        )
        storage.commit()

        stats = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=NoopEnricher(),
            repos=[repo],
            graphify=False,
        ).run(mode="incremental")

        row = storage._conn.execute(
            "SELECT content_hash FROM files WHERE rel_path = 'main.py'"
        ).fetchone()
        assert stats.files_visited == 1
        assert stats.files_skipped_fresh == 0
        assert row["content_hash"] == hashlib.sha256(main.read_bytes()).hexdigest()
        assert (
            storage.get_meta(source_snapshot_meta_key(repo_id))
            == SOURCE_SNAPSHOT_VERSION
        )


def test_failed_enrichment_stays_retryable_and_defers_watermark(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init_with_file(repo, "main.py", "x = 1\n")
    enricher = _FlakyEnricher()

    with Storage(tmp_path / "idx.db", embedding_dim=4) as storage:
        with pytest.raises(PipelineError, match="1 chunks failed enrichment"):
            Pipeline(
                storage=storage,
                embedder=_StubEmbedder(),
                enricher=enricher,
                repos=[repo],
                chunker=_LineChunker(),  # type: ignore[arg-type]
                graphify=False,
            ).run()

        failed_meta = storage._conn.execute(
            "SELECT last_enricher_id FROM files WHERE rel_path = 'main.py'"
        ).fetchone()
        repo_id = int(
            storage._conn.execute("SELECT id FROM repos LIMIT 1").fetchone()["id"]
        )
        assert failed_meta["last_enricher_id"] == "__stropha_incomplete__"
        assert storage.get_meta(source_snapshot_meta_key(repo_id)) is None
        assert storage.list_repos()[0].last_indexed_at is None

        retry_stats = Pipeline(
            storage=storage,
            embedder=_StubEmbedder(),
            enricher=enricher,
            repos=[repo],
            chunker=_LineChunker(),  # type: ignore[arg-type]
            graphify=False,
        ).run()

        completed_meta = storage._conn.execute(
            "SELECT last_enricher_id FROM files WHERE rel_path = 'main.py'"
        ).fetchone()
        assert enricher.calls == 2
        assert completed_meta["last_enricher_id"] == "flaky:v1"
        assert (
            storage.get_meta(source_snapshot_meta_key(repo_id))
            == SOURCE_SNAPSHOT_VERSION
        )
        assert storage.list_repos()[0].last_indexed_at is not None
        assert retry_stats.repos[0].code_changed is True
