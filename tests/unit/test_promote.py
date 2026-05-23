"""Unit tests for ``stropha.meta.promote`` (WIN-202).

Covers the pure-function promotion planner — diff rendering, no-op
detection, target validation, and atomic apply. The CLI integration
lives in ``test_meta_cli.py::TestMetaPromote``; this file proves the
library surface is correct independently of Typer wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stropha.meta.promote import (
    PromotionError,
    PromotionPlan,
    apply_plan,
    build_plan,
    render_summary,
)
from stropha.meta.store import CandidateRecord


def _make_candidate(config: dict, cid: str = "cand-001") -> CandidateRecord:
    """Build a minimal CandidateRecord — no filesystem touch required."""
    return CandidateRecord(
        id=cid,
        parent_id=None,
        created_at="2026-05-23T00:00:00+00:00",
        label="test",
        config=config,
    )


# ---------------------------------------------------------------------------
# build_plan
# ---------------------------------------------------------------------------


class TestBuildPlan:
    def test_missing_target_marks_current_config_none(
        self, tmp_path: Path
    ) -> None:
        """When the target file is absent, ``current_config`` is None and
        the diff shows the entire candidate as an addition."""
        cand = _make_candidate({"retrieval_features": {"hyde_enabled": True}})
        target = tmp_path / "stropha.yaml"  # does NOT exist
        plan = build_plan(cand, target)
        assert plan.current_config is None
        assert plan.candidate_config == {"retrieval_features": {"hyde_enabled": True}}
        assert plan.is_noop is False
        # Diff header should call out the missing file.
        assert "missing" in plan.diff_text

    def test_existing_target_loaded_into_current_config(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "stropha.yaml"
        target.write_text(
            "retrieval_features:\n  hyde_enabled: false\n", encoding="utf-8"
        )
        cand = _make_candidate({"retrieval_features": {"hyde_enabled": True}})
        plan = build_plan(cand, target)
        assert plan.current_config == {"retrieval_features": {"hyde_enabled": False}}
        assert plan.is_noop is False
        assert "hyde_enabled: false" in plan.diff_text
        assert "hyde_enabled: true" in plan.diff_text

    def test_identical_configs_flagged_as_noop(self, tmp_path: Path) -> None:
        target = tmp_path / "stropha.yaml"
        config = {"retrieval_features": {"hyde_enabled": True}}
        target.write_text(
            yaml.safe_dump(config, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        cand = _make_candidate(config)
        plan = build_plan(cand, target)
        assert plan.is_noop is True
        # No-op plans still produce an (empty) diff string.
        assert plan.diff_text == ""

    def test_empty_yaml_target_treated_as_empty_config(
        self, tmp_path: Path
    ) -> None:
        """A file containing only whitespace / a YAML ``null`` document is
        treated as an empty mapping — promotion still proceeds and shows
        the candidate as additions."""
        target = tmp_path / "stropha.yaml"
        target.write_text("\n", encoding="utf-8")
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, target)
        assert plan.current_config == {}
        assert plan.is_noop is False

    def test_malformed_yaml_raises_promotion_error(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "stropha.yaml"
        target.write_text("this is: { not valid: ]]]", encoding="utf-8")
        cand = _make_candidate({"a": 1})
        with pytest.raises(PromotionError, match="not valid YAML"):
            build_plan(cand, target)

    def test_non_mapping_top_level_yaml_raises(self, tmp_path: Path) -> None:
        """A top-level list / scalar makes no sense as a stropha config —
        the planner refuses rather than silently overwriting."""
        target = tmp_path / "stropha.yaml"
        target.write_text("- item1\n- item2\n", encoding="utf-8")
        cand = _make_candidate({"a": 1})
        with pytest.raises(PromotionError, match="YAML mapping"):
            build_plan(cand, target)

    def test_diff_is_deterministic_across_dict_orderings(
        self, tmp_path: Path
    ) -> None:
        """``_render_yaml`` sorts keys so two semantically-equal dicts in
        different insertion orders produce identical diffs — no churn."""
        target = tmp_path / "stropha.yaml"
        target.write_text(
            yaml.safe_dump(
                {"b": 2, "a": 1}, sort_keys=True, default_flow_style=False
            ),
            encoding="utf-8",
        )
        cand_ordered_one_way = _make_candidate({"a": 1, "b": 2})
        cand_ordered_other_way = _make_candidate({"b": 2, "a": 1})
        plan_a = build_plan(cand_ordered_one_way, target)
        plan_b = build_plan(cand_ordered_other_way, target)
        assert plan_a.is_noop is True
        assert plan_b.is_noop is True

    def test_diff_header_includes_candidate_id(
        self, tmp_path: Path
    ) -> None:
        """The unified diff's ``+++`` label includes the candidate id so
        an operator reviewing piped output knows which candidate they
        are looking at."""
        target = tmp_path / "stropha.yaml"
        target.write_text("a: 1\n", encoding="utf-8")
        cand = _make_candidate({"a": 2}, cid="my-cand-id")
        plan = build_plan(cand, target)
        assert "my-cand-id" in plan.diff_text

    def test_empty_candidate_config_diffs_as_full_removal(
        self, tmp_path: Path
    ) -> None:
        """A candidate with an empty config (no sampled dimensions) still
        produces a plan — the diff is a full removal of every key in the
        current target."""
        target = tmp_path / "stropha.yaml"
        target.write_text("a: 1\nb: 2\n", encoding="utf-8")
        cand = _make_candidate({})
        plan = build_plan(cand, target)
        assert plan.candidate_config == {}
        assert plan.is_noop is False
        # Diff shows the removals.
        assert "-a: 1" in plan.diff_text
        assert "-b: 2" in plan.diff_text


# ---------------------------------------------------------------------------
# apply_plan
# ---------------------------------------------------------------------------


class TestApplyPlan:
    def test_writes_yaml_to_target(self, tmp_path: Path) -> None:
        target = tmp_path / "stropha.yaml"
        cand = _make_candidate({"hyde_enabled": True})
        plan = build_plan(cand, target)
        apply_plan(plan)
        assert target.is_file()
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert loaded == {"hyde_enabled": True}

    def test_creates_parent_directories_if_missing(
        self, tmp_path: Path
    ) -> None:
        """``--target`` can point into a directory that doesn't exist
        yet — apply must mkdir it instead of crashing."""
        target = tmp_path / "nested" / "sub" / "stropha.yaml"
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, target)
        apply_plan(plan)
        assert target.is_file()

    def test_overwrites_existing_target(self, tmp_path: Path) -> None:
        target = tmp_path / "stropha.yaml"
        target.write_text("retrieval_features:\n  hyde_enabled: false\n",
                          encoding="utf-8")
        cand = _make_candidate({"retrieval_features": {"hyde_enabled": True}})
        plan = build_plan(cand, target)
        apply_plan(plan)
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert loaded == {"retrieval_features": {"hyde_enabled": True}}

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path: Path) -> None:
        """The ``.promote.tmp`` shadow file must not linger after a
        successful apply — the rename is what completes the swap."""
        target = tmp_path / "stropha.yaml"
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, target)
        apply_plan(plan)
        leftovers = list(tmp_path.glob("*.promote.tmp"))
        assert leftovers == []

    def test_noop_apply_still_writes_target(self, tmp_path: Path) -> None:
        """When the plan is a no-op we still re-write — recovery from a
        partially-written target requires the rewrite, and idempotency
        keeps the operation cheap."""
        target = tmp_path / "stropha.yaml"
        cand_config = {"a": 1}
        target.write_text(
            yaml.safe_dump(cand_config, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        cand = _make_candidate(cand_config)
        plan = build_plan(cand, target)
        assert plan.is_noop is True
        # Should NOT raise even on a no-op plan.
        apply_plan(plan)
        assert target.is_file()


# ---------------------------------------------------------------------------
# render_summary
# ---------------------------------------------------------------------------


class TestRenderSummary:
    def test_noop_summary(self, tmp_path: Path) -> None:
        target = tmp_path / "stropha.yaml"
        target.write_text("a: 1\n", encoding="utf-8")
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, target)
        summary = render_summary(plan)
        assert "no-op" in summary

    def test_create_summary_for_missing_target(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "stropha.yaml"
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, target)
        summary = render_summary(plan)
        assert "create" in summary

    def test_modify_summary_includes_diff_lines(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "stropha.yaml"
        target.write_text("a: 1\n", encoding="utf-8")
        cand = _make_candidate({"a": 2})
        plan = build_plan(cand, target)
        summary = render_summary(plan)
        assert "modify" in summary
        assert "lines" in summary


# ---------------------------------------------------------------------------
# PromotionPlan dataclass
# ---------------------------------------------------------------------------


class TestPromotionPlan:
    def test_frozen_dataclass_rejects_mutation(self, tmp_path: Path) -> None:
        """The plan is immutable so it can be passed between layers
        without surprise mutation."""
        cand = _make_candidate({"a": 1})
        plan = build_plan(cand, tmp_path / "stropha.yaml")
        with pytest.raises((AttributeError, TypeError)):
            plan.is_noop = True  # type: ignore[misc]

    def test_promotion_plan_fields_are_complete(
        self, tmp_path: Path
    ) -> None:
        cand = _make_candidate({"a": 1}, cid="abc")
        target = tmp_path / "stropha.yaml"
        plan = build_plan(cand, target)
        assert plan.candidate_id == "abc"
        assert plan.target_path == target
        assert plan.current_config is None
        assert plan.candidate_config == {"a": 1}
        assert isinstance(plan.diff_text, str)
        assert isinstance(plan.is_noop, bool)


# ---------------------------------------------------------------------------
# Public surface (__init__ re-exports)
# ---------------------------------------------------------------------------


class TestPublicReexports:
    def test_symbols_exported_from_stropha_meta(self) -> None:
        from stropha import meta

        assert meta.PromotionError is PromotionError
        assert meta.PromotionPlan is PromotionPlan
        assert meta.build_plan is build_plan
        assert meta.apply_plan is apply_plan
        assert meta.render_summary is render_summary
