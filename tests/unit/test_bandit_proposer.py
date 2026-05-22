"""Unit tests for the zero-LLM Thompson-sampling bandit proposer (WIN-189)."""

from __future__ import annotations

import pytest

from stropha.meta import (
    Constraint,
    Dimension,
    ScoreCard,
    SearchSpace,
    SearchSpaceError,
    load_search_space,
)
from stropha.meta.proposer import BanditProposer, make_bandit_proposer
from stropha.meta.proposer.bandit import _Stats
from stropha.meta.store import CandidateRecord


def _record(idx: int, config: dict) -> CandidateRecord:
    return CandidateRecord(
        id=f"c-{idx:02d}",
        parent_id=None,
        created_at="2026-05-22T00:00:00+00:00",
        label=f"iter-{idx}",
        config=config,
    )


# ---------------------------------------------------------------------------
# _Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_default_is_uniform_prior(self) -> None:
        s = _Stats()
        assert s.successes == 0.0
        assert s.total == 0.0

    def test_update_above_mean(self) -> None:
        s = _Stats()
        s.update(0.8, above_mean=True)
        assert s.total == 1
        assert s.successes == 0.8

    def test_update_below_mean(self) -> None:
        s = _Stats()
        s.update(0.3, above_mean=False)
        assert s.total == 1
        assert s.successes == 0.0  # only above_mean contributes

    def test_negative_reward_clamped(self) -> None:
        s = _Stats()
        s.update(-0.5, above_mean=True)
        assert s.successes == 0.0

    def test_sample_returns_float_in_unit_interval(self) -> None:
        import random as _random

        s = _Stats(successes=2.0, total=5.0)
        v = s.sample(_random.Random(42))
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# BanditProposer — protocol
# ---------------------------------------------------------------------------


class TestBanditProposer:
    def test_deterministic_with_seed(self) -> None:
        # Use a stub space without env-gated constraints so the
        # determinism test isn't held hostage to which env is set.
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="walker", kind="stage", indexing=True,
                    type="choice", choices=("a", "b", "c"),
                ),
                Dimension(
                    name="cache_size", kind="feature", indexing=False,
                    type="int", range=(100, 1000), step=100, default=500,
                ),
                Dimension(
                    name="hyde", kind="feature", indexing=False,
                    type="bool", default=False,
                ),
            ],
        )
        # Seed two bandits identically; first proposal must match.
        a = BanditProposer(space=space, seed=42)
        b = BanditProposer(space=space, seed=42)
        cfg_a = a([])
        cfg_b = b([])
        assert cfg_a == cfg_b

    def test_no_seed_uses_os_entropy(self) -> None:
        """Without a seed the proposer still produces a valid config.

        Use a constraint-free stub space so we don't depend on which
        env vars are set in the test environment.
        """
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="walker", kind="stage", indexing=True,
                    type="choice", choices=("a", "b"),
                ),
                Dimension(
                    name="embedder", kind="stage", indexing=True,
                    type="choice", choices=("local",),
                ),
            ],
        )
        p = BanditProposer(space=space)
        cfg = p([])
        assert "walker" in cfg
        assert "embedder" in cfg

    def test_proposal_respects_search_space_choices(self) -> None:
        # Use a stub space without env-gated constraints — the live
        # manifest's voyage requirement makes this test environment-
        # dependent otherwise.
        space = SearchSpace(
            dimensions=[
                Dimension(name="walker", kind="stage", indexing=True,
                          type="choice", choices=("git-ls-files", "filesystem")),
                Dimension(name="enricher", kind="stage", indexing=True,
                          type="choice", choices=("noop", "hierarchical")),
            ],
        )
        p = BanditProposer(space=space, seed=7)
        cfg = p([])
        for dim in space.enumerate_dimensions(kind="stage"):
            assert cfg[dim.name] in dim.choices

    def test_feature_int_in_range_with_step(self) -> None:
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="cache_size", kind="feature", indexing=False,
                    type="int", range=(100, 1000), step=100, default=500,
                ),
            ],
        )
        p = BanditProposer(space=space, seed=7)
        for _ in range(10):
            cfg = p([])
            assert 100 <= cfg["cache_size"] <= 1000
            assert cfg["cache_size"] % 100 == 0

    def test_feature_bool_drawn(self) -> None:
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="hyde", kind="feature", indexing=False,
                    type="bool", default=False,
                ),
            ],
        )
        p = BanditProposer(space=space, seed=7)
        results = {p([])["hyde"] for _ in range(30)}
        # Both branches visited with 30 draws (extremely high prob).
        assert results == {True, False}

    def test_feature_int_no_range_uses_default(self) -> None:
        """An int dim without a range falls back to its default."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="x", kind="feature", indexing=False,
                    type="int", range=None, default=42,
                ),
            ],
        )
        p = BanditProposer(space=space, seed=7)
        assert p([])["x"] == 42

    def test_archive_pushes_winning_choice_higher(self) -> None:
        """Thompson sampling biases toward choices that won in the past.
        We can't assert determinism but can assert the winning choice
        appears at higher frequency over N draws."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="walker", kind="stage", indexing=True,
                    type="choice", choices=("alpha", "beta"),
                ),
            ],
        )
        # 5 archive entries: 'alpha' got quality 0.9, 'beta' got 0.1.
        archive = []
        for i in range(5):
            archive.append((
                _record(i, {"walker": "alpha"}),
                ScoreCard(quality=0.9),
            ))
            archive.append((
                _record(i + 5, {"walker": "beta"}),
                ScoreCard(quality=0.1),
            ))
        p = BanditProposer(space=space, seed=99)
        picks = [p(archive)["walker"] for _ in range(40)]
        # Alpha should dominate.
        assert picks.count("alpha") > picks.count("beta")

    def test_archive_entry_without_dim_value_ignored(self) -> None:
        """A record whose config is missing a dimension must not crash."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="walker", kind="stage", indexing=True,
                    type="choice", choices=("a", "b"),
                ),
            ],
        )
        archive = [
            (_record(0, {}), ScoreCard(quality=0.5)),  # no 'walker' key
        ]
        p = BanditProposer(space=space, seed=1)
        # Must still produce a valid config.
        cfg = p(archive)
        assert cfg["walker"] in ("a", "b")

    def test_proposal_applies_constraints(self) -> None:
        """``gated_by`` constraints must fire after sampling."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="multi_query", kind="feature", indexing=False,
                    type="bool", default=False,
                ),
                Dimension(
                    name="multi_query_count", kind="feature", indexing=False,
                    type="int", range=(1, 5), default=3,
                ),
            ],
            constraints=[
                Constraint(
                    kind="gated_by",
                    when={"multi_query": False},
                    set={"multi_query_count": 3},
                ),
            ],
        )
        p = BanditProposer(space=space, seed=42)
        # Run many proposals; whenever multi_query is False the count
        # must be the default 3.
        for _ in range(20):
            cfg = p([])
            if cfg["multi_query"] is False:
                assert cfg["multi_query_count"] == 3

    def test_requires_env_unsatisfied_raises(self) -> None:
        """When a sampled config triggers a `requires_env` constraint and
        the env var is not set, apply_constraints raises."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="embedder", kind="stage", indexing=True,
                    type="choice", choices=("voyage",),  # only voyage
                ),
            ],
            constraints=[
                Constraint(
                    kind="requires_env",
                    when={"embedder": "voyage"},
                    require_env="VOYAGE_API_KEY",
                ),
            ],
        )
        p = BanditProposer(space=space, seed=1, env={})
        with pytest.raises(SearchSpaceError):
            p([])

    def test_empty_choices_raises(self) -> None:
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="walker", kind="stage", indexing=True,
                    type="choice", choices=(),
                ),
            ],
        )
        p = BanditProposer(space=space, seed=1)
        with pytest.raises(ValueError, match="no choices"):
            p([])

    def test_unknown_feature_type_returns_default(self) -> None:
        """Defensive — a dimension whose ``type`` is neither bool nor int
        is treated as choice but the helper drops back to default when
        choices are empty."""
        space = SearchSpace(
            dimensions=[
                Dimension(
                    name="weird", kind="feature", indexing=False,
                    type="choice", default="default-value",
                ),
            ],
        )
        # The feature path doesn't go through _sample_stage; it falls
        # through to the dim.default branch since type isn't bool or int.
        p = BanditProposer(space=space, seed=1)
        cfg = p([])
        assert cfg["weird"] == "default-value"


# ---------------------------------------------------------------------------
# make_bandit_proposer factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_returns_bandit_proposer(self) -> None:
        space = load_search_space()
        p = make_bandit_proposer(space, seed=42)
        assert isinstance(p, BanditProposer)
        assert p.seed == 42

    def test_passes_env_through(self) -> None:
        space = load_search_space()
        p = make_bandit_proposer(space, env={"VOYAGE_API_KEY": "sk-fake"})
        assert p.env == {"VOYAGE_API_KEY": "sk-fake"}


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_proposer_re_exports(self) -> None:
        from stropha.meta import proposer
        assert hasattr(proposer, "BanditProposer")
        assert hasattr(proposer, "make_bandit_proposer")