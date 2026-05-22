"""Search-space manifest loader + sampler (WIN-183).

Consumes the declarative ``search_space.yaml`` (WIN-182) and gives the
Meta-Harness proposer three primitives:

- :func:`load_search_space` — parse the YAML, validate the shape, hydrate
  into :class:`SearchSpace`.
- :meth:`SearchSpace.sample` — generate a candidate config dict. Honors
  the constraint engine (WIN-185) when ``apply_constraints=True``.
- :meth:`SearchSpace.enumerate_dimensions` — list every dimension the
  proposer is allowed to vary, with its type / range / choices.

The manifest's split between *indexing* and *query-time* dimensions
matters because WIN-177's index cache only invalidates on indexing
changes. The loader surfaces that split so a bandit proposer can prefer
query-time tweaks first (cache hits cheap) before paying for re-index.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


# Path to the bundled default manifest.
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "search_space.yaml"


class SearchSpaceError(StrophaError):
    """Raised when the search-space manifest cannot be parsed or sampled."""


@dataclass
class Dimension:
    """One sample-able knob.

    ``kind`` is ``"stage"`` for adapter-choice dimensions (walker /
    chunker / …) and ``"feature"`` for ``retrieval_features`` knobs.
    """

    name: str
    kind: Literal["stage", "feature"]
    indexing: bool
    type: Literal["choice", "bool", "int"] = "choice"
    choices: tuple[str, ...] = ()
    range: tuple[int, int] | None = None
    step: int | None = None
    default: Any = None


@dataclass
class Constraint:
    """One declarative rule the sampler MUST respect.

    Three kinds map to fields of the dataclass:
    - ``requires_env`` — ``when`` + ``require_env``
    - ``gated_by``    — ``when`` + ``set``
    """

    kind: Literal["requires_env", "gated_by"]
    when: dict[str, Any] = field(default_factory=dict)
    require_env: str | None = None
    set: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchSpace:
    """Parsed manifest. Use :meth:`sample` to draw a candidate config."""

    dimensions: list[Dimension] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)

    # Indexable views — built once at construction.
    by_name: dict[str, Dimension] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.by_name = {d.name: d for d in self.dimensions}

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    def enumerate_dimensions(
        self,
        *,
        indexing: bool | None = None,
        kind: Literal["stage", "feature"] | None = None,
    ) -> list[Dimension]:
        """All dimensions, optionally filtered.

        ``indexing=True`` returns only dimensions whose change forces a
        re-index (cache miss); ``indexing=False`` returns query-time-only
        dimensions (cache hit).
        """
        out = list(self.dimensions)
        if indexing is not None:
            out = [d for d in out if d.indexing == indexing]
        if kind is not None:
            out = [d for d in out if d.kind == kind]
        return out

    def indexing_dimensions(self) -> list[Dimension]:
        return self.enumerate_dimensions(indexing=True)

    def query_dimensions(self) -> list[Dimension]:
        return self.enumerate_dimensions(indexing=False)

    # ------------------------------------------------------------------ #
    # Sampling                                                           #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        *,
        rng: random.Random | None = None,
        apply_constraints: bool = True,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Draw one candidate configuration.

        Returns a flat ``{dimension_name: value}`` dict. The caller
        translates that into a ``pipeline_config_from_dict`` overrides
        shape when needed.
        """
        r = rng or random.Random()
        chosen: dict[str, Any] = {}
        for dim in self.dimensions:
            chosen[dim.name] = _sample_dimension(dim, r)
        if apply_constraints:
            chosen = self.apply_constraints(chosen, env=env)
        return chosen

    # ------------------------------------------------------------------ #
    # Constraint engine (WIN-185)                                        #
    # ------------------------------------------------------------------ #

    def apply_constraints(
        self,
        config: dict[str, Any],
        *,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run every constraint over ``config`` and return the fixed dict.

        ``requires_env`` constraints raise :class:`SearchSpaceError`
        when the gated env var is unset — the proposer should retry
        with a different sample. ``gated_by`` constraints quietly
        rewrite the dependent keys back to their default.
        """
        env_map = env if env is not None else {}
        out = dict(config)
        for constraint in self.constraints:
            if not _when_matches(constraint.when, out):
                continue
            if constraint.kind == "requires_env":
                key = constraint.require_env or ""
                if not env_map.get(key, "").strip():
                    raise SearchSpaceError(
                        f"constraint require_env={key!r} unsatisfied "
                        f"for {constraint.when!r}"
                    )
            elif constraint.kind == "gated_by":
                for k, v in constraint.set.items():
                    out[k] = v
        return out


def _when_matches(when: dict[str, Any], config: dict[str, Any]) -> bool:
    """All key/value pairs in ``when`` must match ``config``."""
    return all(config.get(k) == v for k, v in when.items())


def _sample_dimension(dim: Dimension, rng: random.Random) -> Any:
    """Draw one value from ``dim``'s domain."""
    if dim.type == "bool":
        return rng.choice([True, False])
    if dim.type == "int":
        if dim.range is None:
            raise SearchSpaceError(
                f"dimension {dim.name!r} is type=int but has no range"
            )
        lo, hi = dim.range
        step = dim.step or 1
        n_steps = (hi - lo) // step
        return lo + rng.randint(0, n_steps) * step
    # choice
    if not dim.choices:
        raise SearchSpaceError(
            f"dimension {dim.name!r} is type=choice but has no choices"
        )
    return rng.choice(list(dim.choices))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_search_space(path: Path | str | None = None) -> SearchSpace:
    """Parse a search-space YAML into a :class:`SearchSpace`.

    Args:
        path: optional manifest path. Defaults to the bundled
            ``meta/search_space.yaml``.

    Raises :class:`SearchSpaceError` for any structural issue.
    """
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    if not manifest_path.is_file():
        raise SearchSpaceError(f"manifest does not exist: {manifest_path}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SearchSpaceError(f"manifest malformed: {exc}") from exc
    if not isinstance(raw, dict):
        raise SearchSpaceError(
            f"manifest root must be a mapping, got {type(raw).__name__}"
        )

    dimensions: list[Dimension] = []

    stages = raw.get("stages") or {}
    if not isinstance(stages, dict):
        raise SearchSpaceError("manifest.stages must be a mapping")
    for stage_name, stage_spec in stages.items():
        if not isinstance(stage_spec, dict):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name} must be a mapping"
            )
        choices = tuple(stage_spec.get("choices") or ())
        if not choices:
            raise SearchSpaceError(
                f"manifest.stages.{stage_name} has no `choices`"
            )
        dimensions.append(
            Dimension(
                name=stage_name,
                kind="stage",
                indexing=bool(stage_spec.get("indexing", True)),
                type="choice",
                choices=choices,
                default=choices[0],
            )
        )

    features = raw.get("features") or {}
    if not isinstance(features, dict):
        raise SearchSpaceError("manifest.features must be a mapping")
    for feat_name, feat_spec in features.items():
        if not isinstance(feat_spec, dict):
            raise SearchSpaceError(
                f"manifest.features.{feat_name} must be a mapping"
            )
        feat_type = str(feat_spec.get("type") or "bool")
        if feat_type not in ("bool", "int"):
            raise SearchSpaceError(
                f"manifest.features.{feat_name}.type must be 'bool' or 'int', "
                f"got {feat_type!r}"
            )
        rng_spec = feat_spec.get("range")
        rng_tuple: tuple[int, int] | None = None
        if rng_spec is not None:
            if not (isinstance(rng_spec, list) and len(rng_spec) == 2):
                raise SearchSpaceError(
                    f"manifest.features.{feat_name}.range must be [lo, hi]"
                )
            rng_tuple = (int(rng_spec[0]), int(rng_spec[1]))
        dimensions.append(
            Dimension(
                name=feat_name,
                kind="feature",
                indexing=bool(feat_spec.get("indexing", False)),
                type=feat_type,  # type: ignore[arg-type]
                range=rng_tuple,
                step=int(feat_spec["step"]) if "step" in feat_spec else None,
                default=feat_spec.get("default"),
            )
        )

    constraints: list[Constraint] = []
    raw_constraints = raw.get("constraints") or []
    if not isinstance(raw_constraints, list):
        raise SearchSpaceError("manifest.constraints must be a list")
    for entry in raw_constraints:
        if not isinstance(entry, dict):
            raise SearchSpaceError("each constraint must be a mapping")
        kind = str(entry.get("kind") or "")
        if kind not in ("requires_env", "gated_by"):
            raise SearchSpaceError(
                f"constraint.kind must be 'requires_env' or 'gated_by', "
                f"got {kind!r}"
            )
        constraints.append(
            Constraint(
                kind=kind,  # type: ignore[arg-type]
                when=dict(entry.get("when") or {}),
                require_env=entry.get("require_env"),
                set=dict(entry.get("set") or {}),
            )
        )

    return SearchSpace(dimensions=dimensions, constraints=constraints)


def discover_gaps(
    space: SearchSpace,
    *,
    available_adapters: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Surface registry vs manifest drift (WIN-184).

    Compares the manifest's per-stage ``choices`` against the live
    ``stropha.pipeline.registry`` and returns a dict::

        {
          "missing_from_manifest": {stage: [adapter, ...]},
          "missing_from_registry": {stage: [adapter, ...]},
        }

    A stage with no drift contributes nothing. The CLI surfaces this
    via ``stropha pipeline validate`` (future); the proposer uses it
    to abort when a new adapter has been added without manifest
    annotation (because the manifest is the source of truth on
    indexing-vs-query-time + choice ranges).

    Args:
        space: parsed search space.
        available_adapters: optional ``{stage_name: [adapter_id]}`` map
            (so tests can stub the registry). Defaults to a live
            registry lookup.
    """
    if available_adapters is None:
        from ..pipeline.registry import available_for_stage
        available_adapters = {}
        for dim in space.enumerate_dimensions(kind="stage"):
            try:
                available_adapters[dim.name] = list(available_for_stage(dim.name))
            except Exception:
                available_adapters[dim.name] = []

    missing_from_manifest: dict[str, list[str]] = {}
    missing_from_registry: dict[str, list[str]] = {}

    for dim in space.enumerate_dimensions(kind="stage"):
        manifest_set = set(dim.choices)
        registry_set = set(available_adapters.get(dim.name, []))
        only_registry = sorted(registry_set - manifest_set)
        only_manifest = sorted(manifest_set - registry_set)
        if only_registry:
            missing_from_manifest[dim.name] = only_registry
        if only_manifest:
            missing_from_registry[dim.name] = only_manifest

    return {
        "missing_from_manifest": missing_from_manifest,
        "missing_from_registry": missing_from_registry,
    }


__all__ = [
    "DEFAULT_MANIFEST",
    "Constraint",
    "Dimension",
    "SearchSpace",
    "SearchSpaceError",
    "discover_gaps",
    "load_search_space",
]
