"""Zero-LLM Thompson-sampling bandit proposer (WIN-189).

Acts as the control proposer for the Meta-Harness loop. The agentic
proposers (WIN-190 local, WIN-191 cloud) are smarter on average but
the bandit:

- Has zero LLM cost — every proposal is a stdlib RNG draw.
- Gives a hard baseline to compare agentic against. If the agentic
  proposer cannot beat random Thompson sampling, it's not earning its
  LLM tokens.
- Always reproducible given a seed — useful for CI regression tests.

Algorithm: per-dimension Beta-Bernoulli Thompson sampling on choice
dimensions, Gaussian sampling on int dimensions. Each candidate
contributes its observed `quality` score as the reward signal. A
candidate's choice gets a +reward for each draw above the running
mean and a -reward below.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..fitness import ScoreCard
from ..search_space import Dimension, SearchSpace
from ..store import CandidateRecord


@dataclass
class _Stats:
    """Running success / total counts for one (dimension, choice) pair."""

    successes: float = 0.0
    total: float = 0.0

    def update(self, reward: float, *, above_mean: bool) -> None:
        self.total += 1
        if above_mean:
            self.successes += max(0.0, reward)

    def sample(self, rng: random.Random) -> float:
        """Beta(alpha, beta) draw with Bayesian uniform prior."""
        alpha = 1.0 + self.successes
        beta = 1.0 + max(0.0, self.total - self.successes)
        return rng.betavariate(alpha, beta)


@dataclass
class BanditProposer:
    """Thompson sampling over the search-space manifest.

    Args:
        space: parsed :class:`SearchSpace` (WIN-183).
        seed: optional RNG seed for reproducibility. None = OS entropy.
        env: env-var snapshot for ``apply_constraints``. None = empty.
    """

    space: SearchSpace
    seed: int | None = None
    env: dict[str, str] | None = None
    _rng: random.Random = field(init=False, repr=False)
    _stats: dict[tuple[str, Any], _Stats] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

    # ------------------------------------------------------------------ #
    # ProposerFn protocol                                                #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        archive: Iterable[tuple[CandidateRecord, ScoreCard]],
    ) -> dict[str, Any]:
        """Sample the next candidate config given the archive."""
        archive_list = list(archive)
        self._update_stats(archive_list)
        return self._sample_config()

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _update_stats(
        self,
        archive: list[tuple[CandidateRecord, ScoreCard]],
    ) -> None:
        """Recompute Beta posteriors from the archive's quality scores."""
        self._stats.clear()
        if not archive:
            return
        # Quality score is the reward signal. Above-mean candidates
        # contribute positive evidence to their per-dimension choice.
        scores = [card.quality for _, card in archive]
        mean = sum(scores) / len(scores)
        for record, card in archive:
            above = card.quality > mean
            for dim in self.space.enumerate_dimensions(kind="stage"):
                choice = record.config.get(dim.name)
                if choice is None:
                    continue
                key = (dim.name, choice)
                stat = self._stats.setdefault(key, _Stats())
                stat.update(card.quality, above_mean=above)

    def _sample_config(self) -> dict[str, Any]:
        """Draw one config: Thompson-sample stage choices + sample features."""
        config: dict[str, Any] = {}
        for dim in self.space.dimensions:
            if dim.kind == "stage":
                config[dim.name] = self._sample_stage(dim)
            else:
                config[dim.name] = self._sample_feature(dim)
        # WIN-185 constraints (requires_env, gated_by) applied last.
        return self.space.apply_constraints(config, env=self.env or {})

    def _sample_stage(self, dim: Dimension) -> Any:
        if not dim.choices:
            raise ValueError(f"stage {dim.name!r} has no choices")
        best_choice = None
        best_draw = -1.0
        for choice in dim.choices:
            stat = self._stats.get((dim.name, choice), _Stats())
            draw = stat.sample(self._rng)
            if draw > best_draw:
                best_draw = draw
                best_choice = choice
        return best_choice

    def _sample_feature(self, dim: Dimension) -> Any:
        if dim.type == "bool":
            return self._rng.choice([True, False])
        if dim.type == "int":
            if dim.range is None:
                return dim.default
            lo, hi = dim.range
            step = dim.step or 1
            n_steps = (hi - lo) // step
            return lo + self._rng.randint(0, n_steps) * step
        return dim.default


def make_bandit_proposer(
    space: SearchSpace,
    *,
    seed: int | None = None,
    env: dict[str, str] | None = None,
) -> BanditProposer:
    """Factory wrapper — preferred entry point.

    The plain class is also fine; the factory exists so future tweaks
    (e.g. swapping the prior) have a single call site to update.
    """
    return BanditProposer(space=space, seed=seed, env=env)


__all__ = ["BanditProposer", "make_bandit_proposer"]
