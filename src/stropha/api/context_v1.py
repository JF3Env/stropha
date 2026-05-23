"""Stable v1 context-retrieval contract (WIN-257, EPIC 1.12 #94).

Every harness that talks to stropha (Cursor, Zed, Continue,
opencode, …) expects a slightly different shape from the
underlying search engine. WIN-257 defines a single Pydantic
contract that every harness adapter (MCP, future HTTP) maps to.
The MCP ``search_code`` tool already returns rich responses; this
module normalises that shape into ``ContextResponseV1`` so a
non-MCP transport gets identical bytes.

Versioning
----------

The contract is versioned by module name (``context_v1``). A
breaking change ships as ``context_v2`` so callers can migrate at
their own pace. ``context_version`` is set on every response so
the client can sanity-check.

Surface
-------

``ContextRequestV1`` — what a caller sends.
``ContextResponseV1`` — what a caller receives.
``to_response(hits, *, query, ...)`` — adapter from internal
``SearchHit`` objects (or hit-like dicts) to the response payload.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


CONTEXT_VERSION = "v1"


class RepoInfoV1(BaseModel):
    """Repository identity carried on every hit."""

    model_config = ConfigDict(extra="ignore")

    normalized_key: str = Field(description="Stable cross-user key, e.g. github.com/user/repo.")
    url: str | None = Field(default=None, description="HTTPS clone URL.")
    local_path: str | None = Field(default=None, description="Filesystem path on the indexer.")
    default_branch: str | None = None
    head_commit: str | None = None


class HitV1(BaseModel):
    """One search result in the stable contract."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    rel_path: str
    start_line: int
    end_line: int
    score: float = 0.0
    rank: int = 0
    language: str | None = None
    kind: str | None = None
    symbol: str | None = None
    snippet: str = ""
    repo: RepoInfoV1 | None = None


class ContextRequestV1(BaseModel):
    """Input shape for harness adapters."""

    model_config = ConfigDict(extra="ignore")

    query: str
    top_k: int = 10
    session_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class ContextResponseV1(BaseModel):
    """Output shape every harness adapter returns."""

    model_config = ConfigDict(extra="ignore")

    context_version: str = CONTEXT_VERSION
    query: str
    session_id: str | None = None
    hits: list[HitV1] = Field(default_factory=list)
    total_hits: int = 0
    truncated: bool = False
    notes: list[str] = Field(default_factory=list)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hit_to_v1(hit: Any) -> HitV1:
    """Convert any hit-like object to ``HitV1``.

    Tolerates SearchHit objects, dicts, MagicMocks, and bare
    objects with attribute access.
    """
    def _get(name: str) -> Any:
        if isinstance(hit, dict):
            return hit.get(name)
        return getattr(hit, name, None)

    repo_raw = _get("repo")
    repo: RepoInfoV1 | None = None
    if repo_raw is not None:
        repo_get = (lambda k: repo_raw.get(k)) if isinstance(repo_raw, dict) else (
            lambda k: getattr(repo_raw, k, None)
        )
        nk = repo_get("normalized_key")
        if nk:
            repo = RepoInfoV1(
                normalized_key=_safe_str(nk),
                url=repo_get("url"),
                local_path=_safe_str(repo_get("local_path")) or None,
                default_branch=repo_get("default_branch"),
                head_commit=repo_get("head_commit"),
            )

    return HitV1(
        chunk_id=_safe_str(_get("chunk_id")),
        rel_path=_safe_str(_get("rel_path")),
        start_line=_safe_int(_get("start_line")),
        end_line=_safe_int(_get("end_line")),
        score=_safe_float(_get("score")),
        rank=_safe_int(_get("rank")),
        language=_get("language"),
        kind=_get("kind"),
        symbol=_get("symbol"),
        snippet=_safe_str(_get("snippet")),
        repo=repo,
    )


def to_response(
    hits: list[Any],
    *,
    query: str,
    session_id: str | None = None,
    truncated: bool = False,
    notes: list[str] | None = None,
) -> ContextResponseV1:
    """Build a :class:`ContextResponseV1` from internal hits.

    Args:
        hits: SearchHit-like objects from the engine.
        query: echo of the original request.
        session_id: optional session identity (WIN-217 / WIN-218
          state lives here).
        truncated: True when the search-engine capped the result
          list before this adapter could see them.
        notes: optional cross-cutting messages.

    Returns:
        Stable ``ContextResponseV1`` payload — same bytes
        regardless of the harness transport (MCP, HTTP, CLI).
    """
    v1_hits = [_hit_to_v1(h) for h in (hits or [])]
    return ContextResponseV1(
        query=query,
        session_id=session_id,
        hits=v1_hits,
        total_hits=len(v1_hits),
        truncated=bool(truncated),
        notes=list(notes or []),
    )


__all__ = [
    "CONTEXT_VERSION",
    "ContextRequestV1",
    "ContextResponseV1",
    "HitV1",
    "RepoInfoV1",
    "to_response",
]
