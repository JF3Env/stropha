"""Unit tests for harness-agnostic v1 context contract
(WIN-257, EPIC 1.12 #94)."""

from __future__ import annotations

from stropha.api.context_v1 import (
    CONTEXT_VERSION,
    ContextRequestV1,
    ContextResponseV1,
    HitV1,
    RepoInfoV1,
    _hit_to_v1,
    _safe_float,
    _safe_int,
    _safe_str,
    to_response,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestVersion:
    def test_context_version_v1(self) -> None:
        assert CONTEXT_VERSION == "v1"


# ---------------------------------------------------------------------------
# Pydantic shapes
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_request_defaults(self) -> None:
        r = ContextRequestV1(query="x")
        assert r.top_k == 10
        assert r.session_id is None
        assert r.filters == {}
        assert r.options == {}

    def test_response_defaults(self) -> None:
        r = ContextResponseV1(query="x")
        assert r.context_version == "v1"
        assert r.hits == []
        assert r.total_hits == 0
        assert r.truncated is False
        assert r.notes == []

    def test_hit_defaults(self) -> None:
        h = HitV1(
            chunk_id="c", rel_path="x.py", start_line=1, end_line=2,
        )
        assert h.score == 0.0
        assert h.rank == 0
        assert h.snippet == ""
        assert h.repo is None

    def test_repo_defaults(self) -> None:
        r = RepoInfoV1(normalized_key="github.com/u/r")
        assert r.url is None
        assert r.local_path is None

    def test_extra_fields_ignored(self) -> None:
        # ConfigDict(extra="ignore") — unknown fields don't raise.
        r = ContextRequestV1(query="x", unknown="ignored")  # type: ignore[call-arg]
        assert r.query == "x"


# ---------------------------------------------------------------------------
# Safe coercion helpers
# ---------------------------------------------------------------------------


class TestSafeCoercion:
    def test_safe_str_none(self) -> None:
        assert _safe_str(None) == ""
        assert _safe_str(None, default="x") == "x"

    def test_safe_str_value(self) -> None:
        assert _safe_str("abc") == "abc"
        assert _safe_str(42) == "42"

    def test_safe_int_valid(self) -> None:
        assert _safe_int("42") == 42
        assert _safe_int(3.7) == 3

    def test_safe_int_invalid(self) -> None:
        assert _safe_int("not-a-number") == 0
        assert _safe_int(None) == 0
        assert _safe_int("x", default=-1) == -1

    def test_safe_float_valid(self) -> None:
        assert _safe_float("3.14") == 3.14
        assert _safe_float(2) == 2.0

    def test_safe_float_invalid(self) -> None:
        assert _safe_float("x") == 0.0
        assert _safe_float(None) == 0.0


# ---------------------------------------------------------------------------
# _hit_to_v1
# ---------------------------------------------------------------------------


class _Repo:
    def __init__(self, normalized_key: str = "github.com/u/r") -> None:
        self.normalized_key = normalized_key
        self.url = "https://github.com/u/r.git"
        self.local_path = "/path/to/repo"
        self.default_branch = "main"
        self.head_commit = "abc"


class _Hit:
    def __init__(
        self,
        chunk_id: str = "c1",
        rel_path: str = "x.py",
        start_line: int = 1,
        end_line: int = 10,
        score: float = 0.5,
        rank: int = 0,
        language: str = "python",
        kind: str = "function",
        symbol: str = "foo",
        snippet: str = "def foo(): pass",
        repo: object | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.rel_path = rel_path
        self.start_line = start_line
        self.end_line = end_line
        self.score = score
        self.rank = rank
        self.language = language
        self.kind = kind
        self.symbol = symbol
        self.snippet = snippet
        self.repo = repo


class TestHitToV1:
    def test_object_hit_full(self) -> None:
        hit = _Hit(repo=_Repo())
        out = _hit_to_v1(hit)
        assert out.chunk_id == "c1"
        assert out.start_line == 1
        assert out.repo is not None
        assert out.repo.normalized_key == "github.com/u/r"

    def test_dict_hit(self) -> None:
        out = _hit_to_v1({
            "chunk_id": "c2",
            "rel_path": "y.py",
            "start_line": 5,
            "end_line": 15,
            "score": 0.9,
        })
        assert out.chunk_id == "c2"
        assert out.start_line == 5
        assert out.score == 0.9

    def test_object_without_repo(self) -> None:
        hit = _Hit(repo=None)
        out = _hit_to_v1(hit)
        assert out.repo is None

    def test_dict_repo(self) -> None:
        hit = _Hit(repo={"normalized_key": "github.com/x/y"})
        out = _hit_to_v1(hit)
        assert out.repo is not None
        assert out.repo.normalized_key == "github.com/x/y"

    def test_repo_without_normalized_key_dropped(self) -> None:
        """Repo info with no normalized_key isn't useful — dropped."""
        hit = _Hit(repo={"url": "https://x"})
        out = _hit_to_v1(hit)
        assert out.repo is None

    def test_missing_fields_defaulted(self) -> None:
        class _Bare:
            pass

        out = _hit_to_v1(_Bare())
        assert out.chunk_id == ""
        assert out.start_line == 0
        assert out.score == 0.0

    def test_garbage_score_coerced_to_zero(self) -> None:
        out = _hit_to_v1({"score": "not-a-number"})
        assert out.score == 0.0


# ---------------------------------------------------------------------------
# to_response
# ---------------------------------------------------------------------------


class TestToResponse:
    def test_basic(self) -> None:
        resp = to_response(
            [_Hit(chunk_id="a"), _Hit(chunk_id="b")],
            query="how does X work",
        )
        assert resp.query == "how does X work"
        assert resp.total_hits == 2
        assert resp.context_version == "v1"
        assert resp.truncated is False

    def test_session_id_propagated(self) -> None:
        resp = to_response([], query="x", session_id="sess-1")
        assert resp.session_id == "sess-1"

    def test_truncated_flag(self) -> None:
        resp = to_response([], query="x", truncated=True)
        assert resp.truncated is True

    def test_notes_propagated(self) -> None:
        resp = to_response([], query="x", notes=["a", "b"])
        assert resp.notes == ["a", "b"]

    def test_empty_hits_returns_empty_list(self) -> None:
        resp = to_response([], query="x")
        assert resp.hits == []
        assert resp.total_hits == 0

    def test_none_hits_treated_as_empty(self) -> None:
        # type-checker would complain but defensively we tolerate None
        resp = to_response(None, query="x")  # type: ignore[arg-type]
        assert resp.hits == []


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.api import context_v1 as mod
        assert set(mod.__all__) == {
            "CONTEXT_VERSION",
            "ContextRequestV1",
            "ContextResponseV1",
            "HitV1",
            "RepoInfoV1",
            "to_response",
        }
