"""Unit tests for the proposer skill prompt (WIN-188)."""

from __future__ import annotations

from stropha.meta.proposer import SKILL_PATH, load_skill


class TestSkillPath:
    def test_path_exists(self) -> None:
        assert SKILL_PATH.is_file()

    def test_path_is_markdown(self) -> None:
        assert SKILL_PATH.suffix == ".md"

    def test_path_lives_under_proposer_package(self) -> None:
        assert SKILL_PATH.parent.name == "proposer"


class TestLoadSkill:
    def test_returns_non_empty_string(self) -> None:
        text = load_skill()
        assert isinstance(text, str)
        assert len(text) > 1000  # the doc is substantial

    def test_idempotent_reads(self) -> None:
        assert load_skill() == load_skill()

    def test_mentions_key_invariants(self) -> None:
        """The skill must surface the invariants the proposer needs:
        T7=neutral, Pareto-not-scalar, cache-aware, constraints."""
        text = load_skill()
        for needle in (
            "T7",  # read-only neutrality
            "Pareto",  # frontier-aware
            "cache",  # indexing-vs-query-time
            "constraint",  # requires_env / gated_by
            "search_space",  # manifest reference
            "DIMENSION_SIGNS",  # the four dims
        ):
            assert needle in text, f"skill missing key concept: {needle!r}"

    def test_mentions_anti_patterns(self) -> None:
        """The 'Anti-patterns to avoid' section must be present so the
        proposer knows what NOT to do."""
        text = load_skill()
        assert "Anti-patterns" in text or "anti-pattern" in text.lower()