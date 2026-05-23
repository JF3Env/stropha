"""Multi-tenant scoping primitive (WIN-259, EPIC 1.12 #96).

One stropha process can already index many repos (schema v2 +
v7 — repo-namespaced chunk_ids + repo_id columns on graph
tables). What it doesn't do today is **enforce** per-tenant
isolation at query time — every search returns hits across all
repos.

WIN-259 ships the primitive that lets the operator boundary a
query to one or more tenants without modifying the search engine.
Given a ``TenantContext`` + a raw result list, the filter keeps
only the hits whose repo matches the tenant scope.

Concept
-------

A **tenant** is identified by either:

- ``normalized_key`` (e.g., ``github.com/team-a/service``), or
- A glob over the key (e.g., ``github.com/team-a/*``).

A ``TenantContext`` carries:

- ``tenant_id`` — the operator's label for billing / audit logs.
- ``allowed_repos`` — explicit normalized_keys.
- ``allowed_patterns`` — glob patterns.

The filter is whitelisting: an empty allow-list means "deny
all" (returns no hits). The operator must explicitly grant
access.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TenantContext:
    """Per-tenant scoping rules."""

    tenant_id: str
    allowed_repos: list[str] = field(default_factory=list)
    allowed_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TenantFilterResult:
    """Outcome of :func:`filter_hits`."""

    allowed: list[Any] = field(default_factory=list)
    rejected_count: int = 0
    rejected_repos: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _hit_repo_key(hit: Any) -> str | None:
    """Extract the repo's normalized_key from a hit-like object."""
    repo = getattr(hit, "repo", None)
    if repo is None and isinstance(hit, dict):
        repo = hit.get("repo")
    if repo is None:
        return None
    if isinstance(repo, dict):
        return repo.get("normalized_key")
    return getattr(repo, "normalized_key", None)


def is_repo_allowed(
    normalized_key: str | None, tenant: TenantContext,
) -> bool:
    """Return True when ``normalized_key`` matches the tenant's
    allow-list (explicit OR glob)."""
    if not normalized_key:
        return False
    if normalized_key in tenant.allowed_repos:
        return True
    for pattern in tenant.allowed_patterns:
        if pattern and fnmatch.fnmatchcase(normalized_key, pattern):
            return True
    return False


def filter_hits(
    hits: list[Any], tenant: TenantContext,
) -> TenantFilterResult:
    """Partition ``hits`` into allowed / rejected by tenant scope.

    Args:
        hits: SearchHit-like objects to filter.
        tenant: scoping rules. An empty allow-list rejects every
            hit (whitelisting model — operator must explicitly
            grant access).

    Returns:
        :class:`TenantFilterResult` with the allowed subset, the
        rejected count, and the unique list of rejected repo keys.
    """
    notes: list[str] = []
    if not tenant.allowed_repos and not tenant.allowed_patterns:
        notes.append(
            f"tenant {tenant.tenant_id!r} has no allow-list — "
            "rejecting every hit"
        )

    allowed: list[Any] = []
    rejected: list[str] = []
    rejected_count = 0
    seen_rejects: set[str] = set()
    for hit in hits or []:
        repo_key = _hit_repo_key(hit)
        if is_repo_allowed(repo_key, tenant):
            allowed.append(hit)
            continue
        rejected_count += 1
        key = repo_key or "<missing>"
        if key not in seen_rejects:
            seen_rejects.add(key)
            rejected.append(key)

    return TenantFilterResult(
        allowed=allowed,
        rejected_count=rejected_count,
        rejected_repos=rejected,
        notes=notes,
    )


__all__ = [
    "TenantContext",
    "TenantFilterResult",
    "filter_hits",
    "is_repo_allowed",
]
