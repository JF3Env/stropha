"""Shared pydantic models used across modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SOURCE_SNAPSHOT_VERSION = "1"
SOURCE_SNAPSHOT_META_PREFIX = "source_snapshot_version:"

MAX_AGENT_PATH_CHARS = 1024
MAX_AGENT_RECENT_FILES = 20
MAX_AGENT_CHANGED_PATHS = 128
MAX_RUNTIME_ARTIFACTS = 16
MAX_RUNTIME_ARTIFACT_ID_CHARS = 128
MAX_RUNTIME_ARTIFACT_CONTENT_BYTES = 32 * 1024
MAX_RUNTIME_ARTIFACT_TOTAL_BYTES = 128 * 1024
MAX_RUNTIME_SOURCE_LOCATIONS = 16

_RUNTIME_ARTIFACT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
_AgentPath = Annotated[
    str,
    Field(min_length=1, max_length=MAX_AGENT_PATH_CHARS),
]
_RuntimeArtifactId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_RUNTIME_ARTIFACT_ID_CHARS,
        pattern=_RUNTIME_ARTIFACT_ID_PATTERN,
    ),
]


def source_snapshot_meta_key(repo_id: int) -> str:
    """Return the repository-scoped source snapshot contract key."""
    return f"{SOURCE_SNAPSHOT_META_PREFIX}{repo_id}"


class SourceFile(BaseModel):
    """A file discovered by the walker, ready to be chunked."""

    path: Path
    rel_path: str
    language: str
    size_bytes: int
    snapshot_content: str | None = Field(default=None, exclude=True, repr=False)
    """Immutable text captured by the pipeline for this indexing attempt."""


@dataclass(frozen=True)
class FileDelta:
    """One file-level change emitted by the incremental walker.

    Maps to the four ``git diff --name-status`` actions:

    - ``add``     — file appeared since ``since_sha``
    - ``modify``  — file existed and was edited
    - ``delete``  — file removed from the working tree
    - ``rename``  — file renamed (possibly with edits — ``similarity`` is
      the git-reported percentage; identical-content renames are 100)

    ``rel_path`` is the path *as of HEAD* (i.e. post-rename). For renames
    ``old_rel_path`` is the pre-rename path so the pipeline can
    ``rename_chunks(old, new)`` instead of re-embedding.
    """

    action: Literal["add", "modify", "delete", "rename"]
    rel_path: str
    old_rel_path: str | None = None
    similarity: int | None = None


@dataclass(frozen=True)
class StoredFileMeta:
    """The freshness-relevant columns of one ``files`` row.

    Returned in bulk by ``Storage.load_file_meta_map`` so the hash-based
    reconcile path (WIN-425) can compare the saved source snapshot and active
    adapter identities without loading chunk bodies.
    """

    content_hash: str
    mtime: float
    size_bytes: int
    chunk_count: int
    enricher_id: str | None = None
    embedder_model: str | None = None


class SourceVersion(BaseModel):
    """Observed version of one repository-relative source file."""

    repository: str
    path: str
    layer: Literal["saved_worktree", "committed", "session_overlay"]
    file_sha256: str | None = None
    observed_at: str
    head_commit: str | None = None
    dirty: bool | None = Field(
        default=None,
        description="Whether live bytes differ from the persistent indexed snapshot.",
    )


class SourceEvidenceRef(BaseModel):
    """Line-range evidence tied to an explicitly versioned source file."""

    source_version: SourceVersion
    start_line: int
    end_line: int
    chunk_id: str | None = None
    source_slice_sha256: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the literal source bytes in this line range; null when "
            "those bytes were not available to the evidence producer."
        ),
    )
    indexed_file_sha256: str | None = None
    freshness: Literal["fresh", "stale", "missing", "unknown"]


class _StrictObservationModel(BaseModel):
    """Immutable, closed model for session-local host observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentSelectionV1(_StrictObservationModel):
    """Repository-relative line selection reported by the host editor."""

    path: _AgentPath
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("selection end_line must be greater than or equal to start_line")
        return self


class AgentStateSnapshotV1(_StrictObservationModel):
    """Complete replacement snapshot of one host agent's current editor state."""

    revision: int = Field(ge=1, strict=True)
    repository: str = Field(min_length=1, max_length=512)
    active_file: _AgentPath | None
    recent_files: tuple[_AgentPath, ...] = Field(max_length=MAX_AGENT_RECENT_FILES)
    selection: AgentSelectionV1 | None = None
    changed_paths: tuple[_AgentPath, ...] = Field(max_length=MAX_AGENT_CHANGED_PATHS)
    runtime_artifact_ids: tuple[_RuntimeArtifactId, ...] = Field(
        max_length=MAX_RUNTIME_ARTIFACTS
    )
    observed_at: AwareDatetime


class RuntimeSourceLocationV1(_StrictObservationModel):
    """Repository-relative source range associated with a runtime artifact."""

    repository: str = Field(min_length=1, max_length=512)
    path: _AgentPath
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("source end_line must be greater than or equal to start_line")
        return self


class RuntimeArtifactV1(_StrictObservationModel):
    """Bounded, session-local runtime evidence reported by an MCP host."""

    artifact_id: _RuntimeArtifactId
    kind: Literal["diagnostic", "test_failure", "stack_frame", "log_excerpt"]
    media_type: Literal["text/plain", "application/json"]
    producer: str = Field(min_length=1, max_length=128)
    trust_class: Literal["host_reported"] = "host_reported"
    content: str = Field(
        min_length=1,
        max_length=MAX_RUNTIME_ARTIFACT_CONTENT_BYTES,
        description="Runtime-limited to 32768 UTF-8 bytes.",
    )
    source_locations: tuple[RuntimeSourceLocationV1, ...] = Field(
        max_length=MAX_RUNTIME_SOURCE_LOCATIONS
    )
    redaction_status: Literal["unknown", "redacted", "not_required"]
    expires_at: AwareDatetime


class Chunk(BaseModel):
    """An indexable source unit, from a whole file to a structured AST chunk.

    The adapter-aware pipeline augments the row with the enricher's output
    (``embedding_text``) and the adapter id that produced it
    (``enricher_id``) so the pipeline can detect drift (config change →
    re-enrich + re-embed automatically, no ``--rebuild`` needed).
    """

    chunk_id: str  # sha256 of (rel_path + start_line + end_line + content_hash)
    rel_path: str
    language: str
    kind: str = "file"  # file | class | method | component | scenario | section
    symbol: str | None = None
    parent_chunk_id: str | None = None
    start_line: int
    end_line: int
    content: str
    content_hash: str  # sha256 of content
    # Embedding metadata is set after the embedder runs.
    embedding_model: str | None = None
    embedding_dim: int | None = None
    # Enricher metadata is set after the enricher runs.
    embedding_text: str | None = None
    """Exact text that was fed to the embedder. Equals ``content`` for
    ``enricher=noop``. Persisted so drift detection can compare against
    a fresh enricher's output."""
    enricher_id: str | None = None
    """``adapter_id`` of the enricher that produced ``embedding_text``."""


class RepoRef(BaseModel):
    """Stable identifier of the git repository a chunk belongs to.

    Attached to a ``SearchHit`` when repository metadata is available, allowing
    an MCP client to run ``git clone <url>`` to retrieve the source. See
    ``ingest/git_meta.py:RepoIdentity`` for the producer.

    Every file reference includes both the git origin URL (for remote
    access) and the local filesystem path (for direct access when the
    repo is available locally). File paths (``rel_path``) are always
    relative to the git repository root, not absolute filesystem paths.
    """

    normalized_key: str = Field(
        description=(
            "Stable cross-user key (lowercased host/path, no scheme/auth/.git). "
            "'local:<path>' for git repos without a remote; "
            "'path:<path>' for non-git directories."
        )
    )
    url: str | None = Field(
        default=None,
        description="HTTPS URL suitable for `git clone`. None for local-only.",
    )
    local_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the repository root on the indexer host. "
            "Combined with rel_path gives the full file path for local access."
        ),
    )
    default_branch: str | None = Field(
        default=None,
        description="Default branch for checkout (e.g. 'main').",
    )
    head_commit: str | None = Field(
        default=None,
        description="HEAD SHA at last index time.",
    )


class SearchHit(BaseModel):
    """A single result from retrieval."""

    rank: int
    score: float
    rel_path: str
    language: str
    kind: str
    symbol: str | None = None
    start_line: int
    end_line: int
    snippet: str = Field(description="Truncated content preview.")
    chunk_id: str
    repo: RepoRef | None = Field(
        default=None,
        description="Source repository identity when available.",
    )
    source: SourceEvidenceRef | None = Field(
        default=None,
        description="Versioned saved-source evidence for this indexed chunk.",
    )
    streams: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-stream RRF contributions when this hit came from a fused "
            "retrieval (e.g. {'dense': 0.0164, 'sparse': 0.0162}). None for "
            "hits that did not pass through fusion. Consumed by the "
            "Meta-Harness trace recorder (WIN-160)."
        ),
    )


class RepoStats(BaseModel):
    """Per-repo counters reported by Storage.stats() and `list_repos`.

    Includes both the git origin URL and the local filesystem path so
    callers can access the repository via either method.
    """

    normalized_key: str
    url: str | None
    local_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the repository root on the indexer host."
        ),
    )
    default_branch: str | None
    head_commit: str | None
    files: int
    chunks: int
    last_indexed_at: str | None


class SessionRecord(BaseModel):
    """One row of the schema v8 ``sessions`` table (WIN-221, EPIC 1.9).

    Persists per-session agent state across MCP-server restarts. The
    columns cover deduplication, pinned chunks, scratchpad state and
    extensible session metadata.

    JSON-encoded set fields live as ``frozenset[str]`` in memory so
    callers can use the usual ``in`` / ``add`` / ``update`` patterns;
    serialisation back to JSON happens at the Storage layer.
    """

    session_id: str
    client_id: str | None = None
    created_at: str
    updated_at: str
    seen_chunk_ids: set[str] = Field(
        default_factory=set,
        description="Chunks already served — drives WIN-217 dedup.",
    )
    pinned_chunk_ids: set[str] = Field(
        default_factory=set,
        description="Chunks the agent pinned — drives WIN-218 prepend.",
    )
    meta: dict[str, object] = Field(
        default_factory=dict,
        description="Extensible session-scoped metadata.",
    )
    scratchpad: str = Field(
        default="",
        description=(
            "Durable free-form working notes for the current task "
            "(WIN-229). Survives MCP-server restarts via the "
            "Schema-v12 ``sessions.scratchpad`` column."
        ),
    )


class EpisodeRecord(BaseModel):
    """One ``episodes`` row (Schema v10, WIN-225).

    Captures a per-session task summary plus the files/symbols the
    agent touched. Co-occurrence queries at recall time answer
    "the last time I touched FSRS I also updated migration" by
    scanning past episodes and ranking what was usually touched
    alongside a given file/symbol.

    Distinct from :class:`ProjectMemoryRecord` (timeline vs. facts).
    ``touched_files`` / ``touched_symbols`` are persisted as JSON
    arrays — sets are deduped at the Storage layer.
    """

    id: int | None = None
    repo_id: int
    session_id: str | None = None
    title: str
    summary: str = ""
    touched_files: list[str] = Field(default_factory=list)
    touched_symbols: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    provenance_commit: str | None = Field(
        default=None,
        description=(
            "Git HEAD SHA when the episode was recorded (WIN-228). "
            "Lets the staleness check diff against the current HEAD."
        ),
    )


class ProjectMemoryRecord(BaseModel):
    """One ``project_memory`` row (Schema v9, WIN-222).

    Per-repo long-term notes the agent curates so it doesn't have to
    re-explain anchor facts every session. ``key`` + ``kind`` +
    ``repo_id`` together form the natural UPSERT key, so the agent
    can write ``remember("decision", "fsrs-authoritative", "...")``
    and a follow-up call with the same key updates the existing
    row.

    ``expires_at`` is the TTL hook (WIN-228); ``None`` means
    perpetual.
    """

    id: int | None = None
    repo_id: int
    kind: str
    key: str
    value: str
    source: str | None = None
    created_at: str
    updated_at: str
    expires_at: str | None = None
    provenance_commit: str | None = Field(
        default=None,
        description=(
            "Git HEAD SHA at the time the entry was last upserted "
            "(WIN-228). Drives the staleness check."
        ),
    )
