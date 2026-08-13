"""Zero-LLM Thompson-sampling bandit proposer (WIN-189).

Acts as the control proposer for the Meta-Harness loop. The agentic
proposers (WIN-190 local, WIN-191 cloud) are smarter on average but
the bandit:

- Has zero LLM cost — every proposal is a stdlib RNG draw.
- Gives a hard baseline to compare agentic against. If the agentic
  proposer cannot beat random Thompson sampling, it's not earning its
  LLM tokens.
- Always reproducible given a seed — useful for CI regression tests.

Algorithm: per-arm Beta Thompson sampling across every finite dimension.
Each candidate contributes its signed scalar fitness and every signed
objective. One signal is sampled per proposal so multi-objective trade-offs
remain explorable instead of collapsing permanently to quality.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice, product
from typing import Any

from ..fitness import DIMENSION_SIGNS, ScoreCard, scalar_fitness
from ..search_space import Dimension, SearchSpace, SearchSpaceError
from ..store import CandidateRecord


@dataclass
class _Stats:
    """Running success / total counts for one (dimension, choice) pair."""

    successes: float = 0.0
    total: float = 0.0
    failures: float = 0.0

    def update(self, reward: float, *, above_mean: bool | None = None) -> None:
        signal = reward if above_mean is None or above_mean else -abs(reward)
        self.total += 1
        self.successes += max(0.0, signal)
        self.failures += max(0.0, -signal)

    def sample(self, rng: random.Random) -> float:
        """Beta(alpha, beta) draw with Bayesian uniform prior."""
        alpha = 1.0 + self.successes
        beta = 1.0 + self.failures + max(
            0.0, self.total - self.successes - self.failures
        )
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
    _stats: dict[tuple[str, str, Any], _Stats] = field(default_factory=dict, init=False)
    _proposed: set[tuple[Any, ...]] = field(default_factory=set, init=False, repr=False)
    _archive_ids: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

    def dump_state(self) -> dict[str, Any]:
        """Return JSON-safe state for exact deterministic resume."""
        return {
            "version": 1,
            "seed": self.seed,
            "rng_state": _lists(self._rng.getstate()),
            "proposed": [_lists(value) for value in sorted(self._proposed, key=repr)],
            "archive_ids": list(self._archive_ids),
        }

    def load_state(self, value: Mapping[str, Any]) -> None:
        """Restore state emitted by :meth:`dump_state`."""
        if value.get("version") != 1:
            raise ValueError("unsupported bandit state version")
        if value.get("seed") != self.seed:
            raise ValueError("bandit state seed does not match proposer seed")
        try:
            self._rng.setstate(_tuples(value["rng_state"]))
            self._proposed = {
                tuple(item) for item in value.get("proposed", [])
            }
            self._archive_ids = tuple(str(item) for item in value.get("archive_ids", []))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bandit state: {exc}") from exc

    # ------------------------------------------------------------------ #
    # ProposerFn protocol                                                #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        archive: Iterable[tuple[CandidateRecord, ScoreCard]],
    ) -> dict[str, Any]:
        """Sample the next candidate config given the archive."""
        archive_list = list(archive)
        archive_ids = tuple(record.id for record, _ in archive_list)
        if archive_ids != self._archive_ids:
            self._proposed.clear()
            self._archive_ids = archive_ids
        self._update_stats(archive_list)
        existing = {self._config_key(record.config) for record, _ in archive_list}
        last: dict[str, Any] | None = None
        for _ in range(128):
            candidate = self._sample_config()
            last = candidate
            key = self._config_key(candidate)
            if key not in existing and key not in self._proposed:
                self._proposed.add(key)
                return candidate
        assert last is not None
        unseen = self._first_unseen(existing | self._proposed)
        if unseen is not None:
            self._proposed.add(self._config_key(unseen))
            return unseen
        return last

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _update_stats(
        self,
        archive: list[tuple[CandidateRecord, ScoreCard]],
    ) -> None:
        """Recompute per-signal arm posteriors from the complete archive."""
        self._stats.clear()
        if not archive:
            return
        signals: dict[str, list[float]] = {
            "scalar": [scalar_fitness(card.to_dict()) for _, card in archive],
        }
        for objective, sign in DIMENSION_SIGNS.items():
            signals[objective] = [getattr(card, objective) * sign for _, card in archive]

        for signal_name, values in signals.items():
            mean = sum(values) / len(values)
            deviations = [value - mean for value in values]
            scale = max((abs(value) for value in deviations), default=0.0) or 1.0
            for (record, _), deviation in zip(archive, deviations, strict=True):
                reward = deviation / scale
                for dim in self.space.dimensions:
                    value = record.config.get(dim.name)
                    if value not in self._domain(dim):
                        continue
                    key = (signal_name, dim.name, value)
                    self._stats.setdefault(key, _Stats()).update(reward)

    def _sample_config(self) -> dict[str, Any]:
        """Draw one config using one deterministic objective preference."""
        signals = ("scalar", *DIMENSION_SIGNS)
        signal = self._rng.choice(signals)
        config: dict[str, Any] = {}
        for dim in self.space.dimensions:
            config[dim.name] = self._sample_dimension(dim, signal)
        # WIN-185 constraints (requires_env, gated_by) applied last.
        return self.space.apply_constraints(config, env=self.env or {})

    def _sample_dimension(self, dim: Dimension, signal: str) -> Any:
        domain = self._domain(dim)
        if not domain:
            raise ValueError(f"dimension {dim.name!r} has no choices")
        best_choice = None
        best_draw = -1.0
        for choice in domain:
            stat = self._stats.get((signal, dim.name, choice), _Stats())
            draw = stat.sample(self._rng)
            if draw > best_draw:
                best_draw = draw
                best_choice = choice
        return best_choice

    @staticmethod
    def _domain(dim: Dimension) -> tuple[Any, ...]:
        if dim.type == "choice":
            return tuple(dim.choices) or (() if dim.default is None else (dim.default,))
        if dim.type == "bool":
            return (False, True)
        if dim.type == "int":
            if dim.range is None:
                return (() if dim.default is None else (dim.default,))
            lo, hi = dim.range
            step = dim.step or 1
            return tuple(range(lo, hi + 1, step))
        return (() if dim.default is None else (dim.default,))

    def _config_key(self, config: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(config.get(dim.name) for dim in self.space.dimensions)

    def _first_unseen(self, seen: set[tuple[Any, ...]]) -> dict[str, Any] | None:
        """Deterministically find an unseen constrained config, if one exists."""
        domains = [self._domain(dim) for dim in self.space.dimensions]
        if any(not domain for domain in domains):
            return None
        max_checks = max(128, min(4096, len(seen) + 128))
        for values in islice(product(*domains), max_checks):
            raw = dict(zip((dim.name for dim in self.space.dimensions), values, strict=True))
            try:
                candidate = self.space.apply_constraints(raw, env=self.env or {})
            except SearchSpaceError:
                continue
            key = self._config_key(candidate)
            if key not in seen:
                return candidate
        return None

    def _sample_stage(self, dim: Dimension) -> Any:
        return self._sample_dimension(dim, "scalar")

    def _sample_feature(self, dim: Dimension) -> Any:
        return self._sample_dimension(dim, "scalar")


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


def _lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


__all__ = ["BanditProposer", "make_bandit_proposer"]
