"""Unit tests for multi-tenant scoping (WIN-259, EPIC 1.12 #96)."""

from __future__ import annotations

import pytest

from stropha.api.tenant import (
    TenantContext,
    TenantFilterResult,
    _hit_repo_key,
    filter_hits,
    is_repo_allowed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Repo:
    def __init__(self, normalized_key: str) -> None:
        self.normalized_key = normalized_key


class _Hit:
    def __init__(self, repo: object | None = None) -> None:
        self.repo = repo


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_context_defaults(self) -> None:
        ctx = TenantContext(tenant_id="team-a")
        assert ctx.allowed_repos == []
        assert ctx.allowed_patterns == []

    def test_result_defaults(self) -> None:
        r = TenantFilterResult()
        assert r.allowed == []
        assert r.rejected_count == 0
        assert r.rejected_repos == []
        assert r.notes == []

    def test_frozen(self) -> None:
        ctx = TenantContext(tenant_id="t")
        with pytest.raises((AttributeError, Exception)):
            ctx.tenant_id = "u"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _hit_repo_key
# ---------------------------------------------------------------------------


class TestHitRepoKey:
    def test_object_repo(self) -> None:
        hit = _Hit(repo=_Repo("github.com/u/r"))
        assert _hit_repo_key(hit) == "github.com/u/r"

    def test_dict_repo(self) -> None:
        hit = _Hit(repo={"normalized_key": "github.com/x/y"})
        assert _hit_repo_key(hit) == "github.com/x/y"

    def test_dict_hit_with_dict_repo(self) -> None:
        assert _hit_repo_key({
            "repo": {"normalized_key": "github.com/a/b"},
        }) == "github.com/a/b"

    def test_no_repo(self) -> None:
        assert _hit_repo_key(_Hit()) is None
        assert _hit_repo_key({}) is None


# ---------------------------------------------------------------------------
# is_repo_allowed
# ---------------------------------------------------------------------------


class TestIsRepoAllowed:
    def test_explicit_match(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["github.com/u/r"],
        )
        assert is_repo_allowed("github.com/u/r", tenant) is True

    def test_glob_match(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_patterns=["github.com/team-a/*"],
        )
        assert is_repo_allowed("github.com/team-a/svc", tenant) is True

    def test_no_match(self) -> None:
        tenant = TenantContext(
            tenant_id="t",
            allowed_repos=["github.com/u/r"],
            allowed_patterns=["github.com/team-a/*"],
        )
        assert is_repo_allowed("github.com/u/other", tenant) is False

    def test_none_key_rejected(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["github.com/u/r"],
        )
        assert is_repo_allowed(None, tenant) is False

    def test_empty_allow_list_rejects_all(self) -> None:
        tenant = TenantContext(tenant_id="t")
        assert is_repo_allowed("github.com/u/r", tenant) is False

    def test_empty_pattern_skipped(self) -> None:
        """An empty-string entry in allowed_patterns must NOT match
        every key (fnmatch would otherwise treat '' as a strict
        literal but defensively we skip falsy patterns)."""
        tenant = TenantContext(
            tenant_id="t", allowed_patterns=["", "github.com/u/*"],
        )
        # Empty pattern shouldn't match arbitrary keys.
        assert is_repo_allowed("github.com/x/y", tenant) is False
        # But the real pattern works.
        assert is_repo_allowed("github.com/u/r", tenant) is True


# ---------------------------------------------------------------------------
# filter_hits
# ---------------------------------------------------------------------------


def _hit_for(key: str) -> _Hit:
    return _Hit(repo=_Repo(key))


class TestFilterHits:
    def test_empty_allow_list_rejects_all(self) -> None:
        tenant = TenantContext(tenant_id="t")
        result = filter_hits(
            [_hit_for("github.com/u/r")], tenant,
        )
        assert result.allowed == []
        assert result.rejected_count == 1
        assert any("rejecting every" in n for n in result.notes)

    def test_allow_list_filters(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["github.com/u/r"],
        )
        hits = [
            _hit_for("github.com/u/r"),
            _hit_for("github.com/x/y"),
            _hit_for("github.com/u/r"),
        ]
        result = filter_hits(hits, tenant)
        assert len(result.allowed) == 2
        assert result.rejected_count == 1
        assert "github.com/x/y" in result.rejected_repos

    def test_pattern_filter(self) -> None:
        tenant = TenantContext(
            tenant_id="t",
            allowed_patterns=["github.com/team-a/*"],
        )
        hits = [
            _hit_for("github.com/team-a/svc1"),
            _hit_for("github.com/team-a/svc2"),
            _hit_for("github.com/team-b/svc1"),
        ]
        result = filter_hits(hits, tenant)
        assert len(result.allowed) == 2

    def test_hit_without_repo_rejected(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["github.com/u/r"],
        )
        result = filter_hits([_Hit()], tenant)
        assert result.allowed == []
        assert "<missing>" in result.rejected_repos

    def test_rejected_repos_deduplicated(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["allowed"],
        )
        hits = [
            _hit_for("rejected"),
            _hit_for("rejected"),
            _hit_for("rejected"),
        ]
        result = filter_hits(hits, tenant)
        assert result.rejected_count == 3
        # But the rejected_repos list is deduplicated.
        assert result.rejected_repos == ["rejected"]

    def test_none_hits_treated_as_empty(self) -> None:
        tenant = TenantContext(
            tenant_id="t", allowed_repos=["github.com/u/r"],
        )
        # type-checker would complain but defensively we tolerate None
        result = filter_hits(None, tenant)  # type: ignore[arg-type]
        assert result.allowed == []
        assert result.rejected_count == 0


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.api import tenant as mod
        assert set(mod.__all__) == {
            "TenantContext",
            "TenantFilterResult",
            "filter_hits",
            "is_repo_allowed",
        }
