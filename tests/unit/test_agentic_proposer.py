"""Unit tests for the agentic proposers (WIN-190 local, WIN-191 cloud).

The base class (``_agentic_base.AgenticProposerBase``) owns prompt
assembly + JSON parsing + fallback-to-bandit. Each subclass picks a
stropha.inference backend. We test the base via a fake LLM subclass,
then verify each concrete subclass picks the right backend.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from stropha.meta import Dimension, ScoreCard, SearchSpace
from stropha.meta.proposer import (
    CloudAgenticProposer,
    load_skill,
    make_cloud_agentic_proposer,
    make_local_agentic_proposer,
)
from stropha.meta.proposer._agentic_base import (
    MAX_PROMPT_BYTES,
    AgenticProposerBase,
    _clamp_to_space,
    _extract_json,
    _summarise_archive,
    _summarise_card,
    _summarise_manifest,
)
from stropha.meta.store import CandidateRecord


def _record(idx: int, config: dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        id=f"c{idx:02d}",
        parent_id=None,
        created_at="2026-05-22T00:00:00+00:00",
        label=f"iter-{idx}",
        config=config,
    )


def _stub_space(*, with_constraints: bool = False) -> SearchSpace:
    from stropha.meta import Constraint
    constraints = []
    if with_constraints:
        constraints.append(
            Constraint(
                kind="gated_by",
                when={"multi_query": False},
                set={"multi_query_count": 3},
            )
        )
    return SearchSpace(
        dimensions=[
            Dimension(name="walker", kind="stage", indexing=True,
                      type="choice", choices=("git-ls-files", "filesystem")),
            Dimension(name="enricher", kind="stage", indexing=True,
                      type="choice", choices=("noop", "contextual")),
            Dimension(name="multi_query", kind="feature", indexing=False,
                      type="bool", default=False),
            Dimension(name="multi_query_count", kind="feature", indexing=False,
                      type="int", range=(1, 5), default=3),
        ],
        constraints=constraints,
    )


class _FakeProposer(AgenticProposerBase):
    """Test double — replays a canned LLM response."""

    def __init__(self, response: str | None, **kwargs: Any) -> None:
        self._response = response
        super().__init__(**kwargs)

    def _call_llm(self, prompt: str) -> str | None:
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_strips_plain_fence(self) -> None:
        assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_extracts_first_object_after_prose(self) -> None:
        text = "Here is my answer:\n\n{\"x\": 2, \"y\": 3}\n\nThanks."
        assert _extract_json(text) == {"x": 2, "y": 3}

    def test_trailing_comma_tolerated(self) -> None:
        # Strict JSON disallows trailing commas; the helper trims them.
        text = '{"a": 1, "b": 2,}'
        assert _extract_json(text) == {"a": 1, "b": 2}

    def test_no_object_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("no braces here")

    def test_unbalanced_raises(self) -> None:
        with pytest.raises(ValueError, match="unbalanced"):
            _extract_json("{open but never closes")

    def test_root_not_dict_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defensive 'not a dict' branch fires when json.loads
        returns a non-dict for a brace-bounded substring. We force it
        via a monkeypatched parser since vanilla json never produces
        a non-dict from a '{...}' substring."""
        import stropha.meta.proposer._agentic_base as base
        monkeypatch.setattr(base.json, "loads", lambda _s: [1, 2, 3])
        with pytest.raises(ValueError, match="not a dict"):
            _extract_json('{"a": 1}')

    def test_response_with_no_dict_at_all(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("[1, 2, 3]")


class TestClampToSpace:
    def test_unknown_choice_falls_back_to_default(self) -> None:
        space = _stub_space()
        out = _clamp_to_space(
            {"walker": "totally-made-up", "enricher": "noop"}, space
        )
        # walker.default is None per the Dimension dataclass default,
        # so falls through to choices[0].
        assert out["walker"] == "git-ls-files"
        assert out["enricher"] == "noop"

    def test_int_clamped_to_range(self) -> None:
        space = _stub_space()
        out = _clamp_to_space({"multi_query_count": 999}, space)
        assert out["multi_query_count"] == 5  # clamped to hi
        out = _clamp_to_space({"multi_query_count": -5}, space)
        assert out["multi_query_count"] == 1  # clamped to lo

    def test_int_non_numeric_falls_back_to_default(self) -> None:
        space = _stub_space()
        out = _clamp_to_space({"multi_query_count": "not-a-number"}, space)
        assert out["multi_query_count"] == 3

    def test_bool_coerced(self) -> None:
        space = _stub_space()
        out = _clamp_to_space({"multi_query": 1}, space)
        assert out["multi_query"] is True

    def test_missing_dim_uses_default(self) -> None:
        space = _stub_space()
        out = _clamp_to_space({}, space)
        # default for walker = None → fall through to first choice.
        assert out["walker"] == "git-ls-files"
        assert out["multi_query_count"] == 3


class TestSummariseArchive:
    def test_returns_last_n(self) -> None:
        archive = [
            (_record(i, {"v": i}), ScoreCard(quality=i / 10.0))
            for i in range(20)
        ]
        out = _summarise_archive(archive, limit=5)
        assert len(out) == 5
        # Last 5 ids preserved.
        assert [e["id"] for e in out] == ["c15", "c16", "c17", "c18", "c19"]

    def test_card_summary_fields_round(self) -> None:
        card = ScoreCard(
            quality=0.123456, context_efficiency=0.5,
            latency_ms=42.78901, outcome_score=0.9,
            tier_counts={"T1": 5},
        )
        out = _summarise_card(card)
        assert out["quality"] == 0.1235
        assert out["latency_ms"] == 42.8
        assert out["tier_counts"] == {"T1": 5}


class TestSummariseManifest:
    def test_segregates_stages_and_features(self) -> None:
        space = _stub_space()
        out = _summarise_manifest(space)
        assert "walker" in out["stages"]
        assert "multi_query" in out["features"]
        assert "choices" in out["stages"]["walker"]

    def test_features_carry_type_and_range(self) -> None:
        space = _stub_space()
        out = _summarise_manifest(space)
        # multi_query_count is int with a range.
        f = out["features"]["multi_query_count"]
        assert f["type"] == "int"
        assert f["range"] == [1, 5]

    def test_int_with_step_carries_step(self) -> None:
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="cache_size", kind="feature", indexing=False,
                    type="int", range=(100, 1000), step=100, default=500,
                ),
            ],
        )
        out = _summarise_manifest(space)
        assert out["features"]["cache_size"]["step"] == 100


class TestBaseNotImplemented:
    def test_direct_instantiation_call_raises(self) -> None:
        """The base class is meant to be subclassed; calling _call_llm
        directly must raise NotImplementedError so a buggy subclass
        that forgets to override surfaces the gap loudly."""
        space = _stub_space()
        base = AgenticProposerBase(
            space=space, skill="", backend_name="abstract",
        )
        with pytest.raises(NotImplementedError):
            base._call_llm("anything")


# ---------------------------------------------------------------------------
# Base proposer behaviour
# ---------------------------------------------------------------------------


class TestBaseHappyPath:
    def test_valid_llm_response_passes_through(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response=json.dumps({"walker": "filesystem", "enricher": "contextual"}),
            space=space, skill=load_skill(), backend_name="fake",
        )
        out = p([])
        assert out["walker"] == "filesystem"
        assert out["enricher"] == "contextual"

    def test_constraints_applied_after_parse(self) -> None:
        space = _stub_space(with_constraints=True)
        p = _FakeProposer(
            response=json.dumps({
                "walker": "git-ls-files", "enricher": "noop",
                "multi_query": False, "multi_query_count": 99,
            }),
            space=space, skill=load_skill(), backend_name="fake",
        )
        out = p([])
        # multi_query=False → gated_by clamps multi_query_count back to 3.
        assert out["multi_query_count"] == 3

    def test_unknown_choice_clamped(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response=json.dumps({"walker": "made-up-walker"}),
            space=space, skill=load_skill(), backend_name="fake",
        )
        out = p([])
        assert out["walker"] in ("git-ls-files", "filesystem")


class TestBaseFallback:
    def test_empty_response_falls_back_to_bandit(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response="",  # empty → fallback
            space=space, skill=load_skill(), backend_name="fake",
            fallback_seed=42,
        )
        out = p([])
        # Bandit produces a valid config.
        assert out["walker"] in ("git-ls-files", "filesystem")

    def test_none_response_falls_back(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response=None, space=space, skill=load_skill(),
            backend_name="fake", fallback_seed=42,
        )
        out = p([])
        assert out["walker"] in ("git-ls-files", "filesystem")

    def test_malformed_json_falls_back(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response="not json at all", space=space, skill=load_skill(),
            backend_name="fake", fallback_seed=42,
        )
        out = p([])
        assert "walker" in out

    def test_llm_exception_falls_back(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response=RuntimeError("simulated"),  # type: ignore[arg-type]
            space=space, skill=load_skill(),
            backend_name="fake", fallback_seed=42,
        )
        out = p([])
        assert "walker" in out


class TestPromptAssembly:
    def test_prompt_includes_skill_manifest_archive(self) -> None:
        space = _stub_space()
        p = _FakeProposer(
            response=json.dumps({"walker": "git-ls-files"}),
            space=space, skill=load_skill(), backend_name="fake",
        )
        archive = [(_record(0, {"walker": "git-ls-files"}), ScoreCard(quality=0.5))]
        prompt = p._build_prompt(archive)
        # All three sections present.
        assert "Manifest" in prompt
        assert "archive" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt
        # Skill content (first heading from skill.md).
        assert "Meta-Harness" in prompt

    def test_prompt_truncates_when_oversized(self) -> None:
        space = _stub_space()
        # _summarise_archive caps at 10 entries — so each entry must
        # be huge to bust the 50 KB cap. 10 entries x ~8 KB each = 80 KB.
        huge_archive = [
            (_record(i, {"walker": "git-ls-files", "x": "a" * 8000}),
             ScoreCard(quality=0.5))
            for i in range(10)
        ]
        p = _FakeProposer(
            response=json.dumps({"walker": "git-ls-files"}),
            space=space, skill=load_skill(), backend_name="fake",
        )
        prompt = p._build_prompt(huge_archive)
        assert len(prompt) <= MAX_PROMPT_BYTES
        assert "truncated" in prompt


# ---------------------------------------------------------------------------
# LocalAgenticProposer (WIN-190)
# ---------------------------------------------------------------------------


class TestLocalAgenticProposer:
    def test_constructed_with_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stropha.inference import InferenceBackend

        class _StubBackend(InferenceBackend):
            @property
            def name(self) -> str: return "stub"
            @property
            def is_available(self) -> bool: return True
            def generate(self, prompt: str, **_kw: Any) -> str | None:
                return json.dumps({"walker": "git-ls-files"})

        monkeypatch.setattr(
            "stropha.meta.proposer.inference_helper.get_backend",
            lambda **_: _StubBackend(),
        )
        space = _stub_space()
        p = make_local_agentic_proposer(space, prefer_backend="stub")
        out = p([])
        assert out["walker"] == "git-ls-files"

    def test_backend_init_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_kw):  # type: ignore[no-untyped-def]
            raise RuntimeError("no local backend available")

        monkeypatch.setattr(
            "stropha.meta.proposer.inference_helper.get_backend", boom
        )
        space = _stub_space()
        p = make_local_agentic_proposer(space, fallback_seed=42)
        # No backend → _call_llm returns None → fallback to bandit.
        out = p([])
        assert out["walker"] in ("git-ls-files", "filesystem")


# ---------------------------------------------------------------------------
# CloudAgenticProposer (WIN-191)
# ---------------------------------------------------------------------------


class TestCloudAgenticProposer:
    def test_invalid_prefer_backend_raises(self) -> None:
        space = _stub_space()
        with pytest.raises(ValueError, match="prefer_backend"):
            CloudAgenticProposer(
                space=space, skill="", backend_name="ollama",
                prefer_backend="ollama",  # not in (anthropic, deepseek)
            )

    def test_constructed_with_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stropha.inference import InferenceBackend

        class _Anthropic(InferenceBackend):
            @property
            def name(self) -> str: return "anthropic"
            @property
            def is_available(self) -> bool: return True
            def generate(self, prompt: str, **_kw: Any) -> str | None:
                return json.dumps({"enricher": "contextual"})

        monkeypatch.setattr(
            "stropha.meta.proposer.agentic.get_backend", lambda **_: _Anthropic()
        )
        space = _stub_space()
        p = make_cloud_agentic_proposer(space, prefer_backend="anthropic")
        out = p([])
        assert out["enricher"] == "contextual"

    def test_backend_init_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_kw):  # type: ignore[no-untyped-def]
            raise RuntimeError("ANTHROPIC_API_KEY missing")

        monkeypatch.setattr(
            "stropha.meta.proposer.agentic.get_backend", boom
        )
        space = _stub_space()
        p = make_cloud_agentic_proposer(
            space, prefer_backend="deepseek", fallback_seed=42,
        )
        out = p([])
        assert "walker" in out


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_proposer_re_exports(self) -> None:
        from stropha.meta import proposer
        for name in (
            "LocalAgenticProposer", "CloudAgenticProposer",
            "make_local_agentic_proposer", "make_cloud_agentic_proposer",
        ):
            assert hasattr(proposer, name)