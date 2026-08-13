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

import copy
import random
from collections.abc import Iterable, Mapping
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
    binding: str | None = None


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


@dataclass(frozen=True)
class ResolvedCandidateConfig:
    """One flat candidate resolved into its runtime configuration sections."""

    pipeline: dict[str, Any]
    retrieval_features: dict[str, Any]
    graph: dict[str, Any]


@dataclass
class SearchSpace:
    """Parsed manifest. Use :meth:`sample` to draw a candidate config."""

    dimensions: list[Dimension] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    excluded_adapters: dict[str, dict[str, str]] = field(default_factory=dict)

    # Indexable views — built once at construction.
    by_name: dict[str, Dimension] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        names = [dimension.name for dimension in self.dimensions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise SearchSpaceError(f"duplicate dimension names: {duplicates}")
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

    def resolve_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        base_pipeline_config: Mapping[str, Any],
    ) -> ResolvedCandidateConfig:
        """Merge a flat candidate into a supplied pipeline configuration.

        Bindings outside ``pipeline`` are returned separately so callers can
        apply them explicitly without changing the process-global ``Config``.
        """
        unknown = sorted(set(candidate) - set(self.by_name))
        if unknown:
            raise SearchSpaceError(f"candidate has unknown dimensions: {unknown}")

        resolved = ResolvedCandidateConfig(
            pipeline=copy.deepcopy(dict(base_pipeline_config)),
            retrieval_features={},
            graph={},
        )
        sections = {
            "pipeline": resolved.pipeline,
            "retrieval_features": resolved.retrieval_features,
            "graph": resolved.graph,
        }
        for name, value in candidate.items():
            dimension = self.by_name[name]
            if value not in _dimension_domain(dimension):
                raise SearchSpaceError(
                    f"candidate dimension {name!r} has value outside its domain: {value!r}"
                )
            if dimension.binding is None:
                raise SearchSpaceError(f"candidate dimension {name!r} has no runtime binding")
            section, separator, path = dimension.binding.partition(".")
            if not separator or section not in sections or not path:
                raise SearchSpaceError(
                    f"candidate dimension {name!r} has unsupported binding "
                    f"{dimension.binding!r}"
                )
            _set_binding(
                sections[section],
                path.split("."),
                value,
                replace_adapter=dimension.kind == "stage",
            )
        return resolved

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


def load_search_space(
    path: Path | str | None = None,
    *,
    strict: bool = False,
) -> SearchSpace:
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
        raw = yaml.load(
            manifest_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        ) or {}
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
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise SearchSpaceError("manifest stage names must be non-empty strings")
        if not isinstance(stage_spec, dict):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name} must be a mapping"
            )
        if not stage_spec.get("choices"):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name} has no `choices`"
            )
        raw_choices = stage_spec["choices"]
        if not isinstance(raw_choices, list) or not all(
            isinstance(choice, str) and choice.strip() for choice in raw_choices
        ):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name}.choices must be a list of strings"
            )
        choices = tuple(raw_choices)
        if len(set(choices)) != len(choices):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name}.choices contains duplicates"
            )
        indexing = stage_spec.get("indexing", True)
        if not isinstance(indexing, bool):
            raise SearchSpaceError(
                f"manifest.stages.{stage_name}.indexing must be a bool"
            )
        dimensions.append(
            Dimension(
                name=stage_name,
                kind="stage",
                indexing=indexing,
                type="choice",
                choices=choices,
                default=choices[0],
                binding=_parse_binding(
                    stage_spec.get("binding"),
                    location=f"manifest.stages.{stage_name}.binding",
                    strict=strict,
                ),
            )
        )

    features = raw.get("features") or {}
    if not isinstance(features, dict):
        raise SearchSpaceError("manifest.features must be a mapping")
    for feat_name, feat_spec in features.items():
        if not isinstance(feat_name, str) or not feat_name.strip():
            raise SearchSpaceError("manifest feature names must be non-empty strings")
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
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in rng_spec):
                raise SearchSpaceError(
                    f"manifest.features.{feat_name}.range values must be integers"
                )
            rng_tuple = (rng_spec[0], rng_spec[1])
            if rng_tuple[0] > rng_tuple[1]:
                raise SearchSpaceError(
                    f"manifest.features.{feat_name}.range lower bound exceeds upper bound"
                )
        indexing = feat_spec.get("indexing", False)
        if not isinstance(indexing, bool):
            raise SearchSpaceError(
                f"manifest.features.{feat_name}.indexing must be a bool"
            )
        step = feat_spec.get("step")
        if step is not None and (
            not isinstance(step, int) or isinstance(step, bool) or step <= 0
        ):
            raise SearchSpaceError(
                f"manifest.features.{feat_name}.step must be a positive integer"
            )
        default = feat_spec.get("default")
        if feat_type == "bool" and not isinstance(default, bool):
            raise SearchSpaceError(
                f"manifest.features.{feat_name}.default must be a bool"
            )
        if feat_type == "int":
            if rng_tuple is None:
                raise SearchSpaceError(
                    f"manifest.features.{feat_name} is type=int but has no range"
                )
            if not isinstance(default, int) or isinstance(default, bool):
                raise SearchSpaceError(
                    f"manifest.features.{feat_name}.default must be an integer"
                )
            lo, hi = rng_tuple
            effective_step = step or 1
            if not lo <= default <= hi or (default - lo) % effective_step:
                raise SearchSpaceError(
                    f"manifest.features.{feat_name}.default is outside its stepped range"
                )
        binding = _parse_binding(
            feat_spec.get("binding"),
            location=f"manifest.features.{feat_name}.binding",
            strict=strict,
        )
        dimensions.append(
            Dimension(
                name=feat_name,
                kind="feature",
                indexing=indexing,
                type=feat_type,  # type: ignore[arg-type]
                range=rng_tuple,
                step=step,
                default=default,
                binding=binding,
            )
        )

    constraints: list[Constraint] = []
    raw_constraints = raw.get("constraints") or []
    if not isinstance(raw_constraints, list):
        raise SearchSpaceError("manifest.constraints must be a list")
    for index, entry in enumerate(raw_constraints):
        if not isinstance(entry, dict):
            raise SearchSpaceError("each constraint must be a mapping")
        kind = str(entry.get("kind") or "")
        if kind not in ("requires_env", "gated_by"):
            raise SearchSpaceError(
                f"constraint.kind must be 'requires_env' or 'gated_by', "
                f"got {kind!r}"
            )
        when = entry.get("when") or {}
        set_values = entry.get("set") or {}
        if not isinstance(when, dict):
            raise SearchSpaceError(f"constraint[{index}].when must be a mapping")
        if not isinstance(set_values, dict):
            raise SearchSpaceError(f"constraint[{index}].set must be a mapping")
        require_env = entry.get("require_env")
        if kind == "requires_env" and (
            not isinstance(require_env, str) or not require_env.strip()
        ):
            raise SearchSpaceError(
                f"constraint[{index}].require_env must be a non-empty string"
            )
        if kind == "gated_by" and not set_values:
            raise SearchSpaceError(f"constraint[{index}].set must not be empty")
        constraints.append(
            Constraint(
                kind=kind,  # type: ignore[arg-type]
                when=dict(when),
                require_env=require_env,
                set=dict(set_values),
            )
        )

    excluded_raw = raw.get("excluded_adapters") or {}
    if not isinstance(excluded_raw, dict):
        raise SearchSpaceError("manifest.excluded_adapters must be a mapping")
    excluded: dict[str, dict[str, str]] = {}
    for stage, adapters in excluded_raw.items():
        if not isinstance(stage, str) or not isinstance(adapters, dict):
            raise SearchSpaceError("excluded adapters must be grouped by stage")
        excluded[stage] = {}
        for adapter, reason in adapters.items():
            if not isinstance(adapter, str) or not isinstance(reason, str) or not reason.strip():
                raise SearchSpaceError(
                    f"excluded adapter {stage}.{adapter} must have a non-empty reason"
                )
            excluded[stage][adapter] = reason

    if strict:
        bindings = [dimension.binding for dimension in dimensions]
        duplicate_bindings = sorted({
            binding for binding in bindings
            if binding is not None and bindings.count(binding) > 1
        })
        if duplicate_bindings:
            raise SearchSpaceError(
                f"feature runtime config bindings must be unique: {duplicate_bindings}"
            )
    stage_choices = {
        dimension.name: set(dimension.choices)
        for dimension in dimensions
        if dimension.kind == "stage"
    }
    for stage, adapters in excluded.items():
        overlap = sorted(set(adapters) & stage_choices.get(stage, set()))
        if overlap:
            raise SearchSpaceError(
                f"adapters cannot be both eligible and excluded for {stage}: {overlap}"
            )

    space = SearchSpace(
        dimensions=dimensions,
        constraints=constraints,
        excluded_adapters=excluded,
    )
    _validate_constraints(space)
    return space


def resolve_candidate_config(
    candidate: Mapping[str, Any],
    *,
    base_pipeline_config: Mapping[str, Any],
    search_space: SearchSpace | None = None,
) -> ResolvedCandidateConfig:
    """Resolve a flat candidate using manifest-declared runtime bindings."""
    space = search_space or load_search_space(strict=True)
    return space.resolve_candidate(
        candidate,
        base_pipeline_config=base_pipeline_config,
    )


def discover_gaps(
    space: SearchSpace,
    *,
    available_adapters: dict[str, list[str]] | None = None,
    supported_dimensions: Iterable[str] | None = None,
    strict: bool = False,
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
        supported_dimensions: optional evaluator/consumer dimension names.
            Differences from the manifest are reported and fail in strict mode.
        strict: raise :class:`SearchSpaceError` for registry failures or drift.
    """
    registry_error: Exception | None = None
    if available_adapters is None:
        if strict:
            try:
                from ..pipeline.registry import all_adapters
                available_adapters = all_adapters()
            except Exception as exc:
                registry_error = exc
                available_adapters = {}
        else:
            from ..pipeline.registry import available_for_stage
            available_adapters = {}
            for dim in space.enumerate_dimensions(kind="stage"):
                try:
                    available_adapters[dim.name] = list(available_for_stage(dim.name))
                except Exception:
                    available_adapters[dim.name] = []

    missing_from_manifest: dict[str, list[str]] = {}
    missing_from_registry: dict[str, list[str]] = {}
    stale_exclusions: dict[str, list[str]] = {}

    manifest_stages = {
        dim.name: set(dim.choices)
        for dim in space.enumerate_dimensions(kind="stage")
    }
    for stage in sorted(
        set(manifest_stages) | set(available_adapters) | set(space.excluded_adapters)
    ):
        manifest_set = manifest_stages.get(stage, set())
        registry_set = set(available_adapters.get(stage, []))
        exclusions = set(space.excluded_adapters.get(stage, {}))
        only_registry = sorted(registry_set - manifest_set - exclusions)
        only_manifest = sorted(manifest_set - registry_set)
        stale = sorted(exclusions - registry_set)
        if only_registry:
            missing_from_manifest[stage] = only_registry
        if only_manifest:
            missing_from_registry[stage] = only_manifest
        if stale:
            stale_exclusions[stage] = stale

    gaps = {
        "missing_from_manifest": missing_from_manifest,
        "missing_from_registry": missing_from_registry,
    }
    dimension_gaps: dict[str, dict[str, list[str]]] = {}
    if supported_dimensions is not None:
        manifest_dimensions = set(space.by_name)
        consumer_dimensions = set(supported_dimensions)
        dimension_gaps = {
            "missing_from_consumer": {
                "dimensions": sorted(manifest_dimensions - consumer_dimensions),
            },
            "missing_from_manifest_dimensions": {
                "dimensions": sorted(consumer_dimensions - manifest_dimensions),
            },
        }
        gaps.update({
            name: values
            for name, values in dimension_gaps.items()
            if values["dimensions"]
        })
    if strict and registry_error is not None:
        raise SearchSpaceError(f"adapter registry discovery failed: {registry_error}")
    if strict and stale_exclusions:
        raise SearchSpaceError(
            f"search-space preflight has stale adapter exclusions: {stale_exclusions}"
        )
    if strict and any(values["dimensions"] for values in dimension_gaps.values()):
        raise SearchSpaceError(
            f"search-space consumer dimension disagreement: {dimension_gaps}"
        )
    if strict and (missing_from_manifest or missing_from_registry):
        raise SearchSpaceError(f"search-space preflight failed: {gaps}")
    return gaps


def _validate_constraints(space: SearchSpace) -> None:
    for index, constraint in enumerate(space.constraints):
        for field_name, values in (("when", constraint.when), ("set", constraint.set)):
            for name, value in values.items():
                dim = space.by_name.get(name)
                if dim is None:
                    raise SearchSpaceError(
                        f"constraint[{index}].{field_name} references unknown dimension {name!r}"
                    )
                if value not in _dimension_domain(dim):
                    raise SearchSpaceError(
                        f"constraint[{index}].{field_name}.{name}={value!r} is outside the domain"
                    )


def _dimension_domain(dim: Dimension) -> tuple[Any, ...]:
    if dim.type == "choice":
        return tuple(dim.choices)
    if dim.type == "bool":
        return (False, True)
    if dim.range is None:
        return ()
    lo, hi = dim.range
    return tuple(range(lo, hi + 1, dim.step or 1))


def _parse_binding(value: Any, *, location: str, strict: bool) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise SearchSpaceError(f"{location} must be a non-empty string")
    if strict and value is None:
        raise SearchSpaceError(f"{location.removesuffix('.binding')} has no runtime config binding")
    return value


def _set_binding(
    target: dict[str, Any],
    path: list[str],
    value: Any,
    *,
    replace_adapter: bool,
) -> None:
    current = target
    for part in path[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise SearchSpaceError(f"candidate binding crosses non-mapping field {part!r}")
        current = child
    leaf = path[-1]
    if replace_adapter and leaf == "adapter" and current.get(leaf) != value:
        current.clear()
    current[leaf] = copy.deepcopy(value)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SearchSpaceError(f"duplicate manifest key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


__all__ = [
    "DEFAULT_MANIFEST",
    "Constraint",
    "Dimension",
    "ResolvedCandidateConfig",
    "SearchSpace",
    "SearchSpaceError",
    "discover_gaps",
    "load_search_space",
    "resolve_candidate_config",
]
