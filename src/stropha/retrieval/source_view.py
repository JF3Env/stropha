"""Live saved-working-tree source evidence for retrieval responses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..models import SourceEvidenceRef, SourceVersion

_SOURCE_READ_ATTEMPTS = 3


@dataclass(frozen=True)
class SourceView:
    """A live source slice plus its explicit provenance."""

    text: str | None
    evidence: SourceEvidenceRef | None


def _read_stable_bytes(
    path: Path, *, logical_path: Path | None = None
) -> tuple[bytes | None, Literal["missing", "unknown"]]:
    """Read one path version, rejecting concurrent replacement or mutation."""
    missing = False
    for _attempt in range(_SOURCE_READ_ATTEMPTS):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            resolved_after = (logical_path or path).resolve()
        except OSError:
            missing = True
            continue
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            identity_before == identity_after
            and len(raw) == after.st_size
            and resolved_after == path
        ):
            return raw, "unknown"
        missing = False
    return None, "missing" if missing else "unknown"


def read_saved_worktree_source(
    hit: Any,
    *,
    indexed_file_sha256: str | None,
) -> SourceView:
    """Read a hit's current source slice without allowing path escape."""
    rel_path = getattr(hit, "rel_path", None)
    start_line = getattr(hit, "start_line", None)
    end_line = getattr(hit, "end_line", None)
    repo = getattr(hit, "repo", None)
    local_path = None
    repository = None
    if repo is not None:
        local_path = getattr(repo, "local_path", None) or getattr(
            repo, "root_path", None
        )
        repository = getattr(repo, "normalized_key", None)
    if not (
        isinstance(rel_path, str)
        and rel_path
        and isinstance(local_path, str)
        and local_path
        and isinstance(start_line, int)
        and isinstance(end_line, int)
        and start_line >= 1
        and end_line >= start_line
    ):
        return SourceView(text=None, evidence=None)

    root = Path(local_path).resolve()
    if not isinstance(repository, str) or not repository:
        repository = f"path:{root}"
    logical_path = root.joinpath(rel_path)
    source_path = logical_path.resolve()
    if not source_path.is_relative_to(root):
        return SourceView(text=None, evidence=None)

    observed_at = datetime.now(UTC).isoformat()
    raw, read_failure = _read_stable_bytes(source_path, logical_path=logical_path)
    if raw is None:
        version = SourceVersion(
            repository=repository,
            path=rel_path,
            layer="saved_worktree",
            observed_at=observed_at,
            head_commit=getattr(repo, "head_commit", None),
        )
        return SourceView(
            text=None,
            evidence=SourceEvidenceRef(
                source_version=version,
                start_line=start_line,
                end_line=end_line,
                chunk_id=getattr(hit, "chunk_id", None),
                indexed_file_sha256=indexed_file_sha256,
                freshness=read_failure,
            ),
        )

    file_sha256 = hashlib.sha256(raw).hexdigest()
    if indexed_file_sha256 is None:
        freshness: Literal["fresh", "stale", "missing", "unknown"] = "unknown"
        dirty = None
    else:
        dirty = file_sha256 != indexed_file_sha256
        freshness = "stale" if dirty else "fresh"

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    selected = "".join(lines[start_line - 1 : end_line])
    source_text = None
    if start_line <= len(lines):
        source_text = selected.rstrip()
    slice_hash = None
    if selected and text.encode("utf-8", errors="replace") == raw:
        slice_hash = hashlib.sha256(selected.encode("utf-8")).hexdigest()

    version = SourceVersion(
        repository=repository,
        path=rel_path,
        layer="saved_worktree",
        file_sha256=file_sha256,
        observed_at=observed_at,
        head_commit=getattr(repo, "head_commit", None),
        dirty=dirty,
    )
    return SourceView(
        text=source_text,
        evidence=SourceEvidenceRef(
            source_version=version,
            start_line=start_line,
            end_line=end_line,
            chunk_id=getattr(hit, "chunk_id", None),
            source_slice_sha256=slice_hash,
            indexed_file_sha256=indexed_file_sha256,
            freshness=freshness,
        ),
    )


__all__ = ["SourceView", "read_saved_worktree_source"]
