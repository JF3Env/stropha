"""Machine-local policy restrictions for full-snapshot experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .bench.matrix import Matrix, MatrixError
from .config import Config, ExperimentsConfig
from .errors import StrophaError
from .meta.search_space import SearchSpace

_STAGE_AXES = frozenset(
    {"walker", "chunker", "enricher", "embedder", "storage", "retrieval", "reranker"}
)
_FULL_SNAPSHOT_GIT_DIFF_REASON = (
    "git-diff is incremental-only and cannot produce a full-snapshot experiment index"
)
_BLACKLIST_REASON = "excluded by the machine-local experiment adapter blacklist"


class ExperimentPolicyError(StrophaError):
    """Raised when local experiment policy leaves no valid experiment."""


@dataclass(frozen=True)
class ExperimentPolicy:
    """Immutable view of experiment settings used to restrict declared inputs."""

    config: ExperimentsConfig

    def __post_init__(self) -> None:
        self.config.validate_for_run()

    @classmethod
    def from_config(cls, config: Config | ExperimentsConfig) -> ExperimentPolicy:
        """Construct policy from the unified config or its experiments section."""
        experiments = config.experiments if isinstance(config, Config) else config
        return cls(config=experiments)

    @property
    def fingerprint(self) -> str:
        """Stable policy identity that deliberately excludes the PostgreSQL DSN."""
        payload = {
            "version": 1,
            "storage": self.config.storage,
            "postgres": {
                "admin_dsn_configured": bool(self.config.postgres.admin_dsn),
                "database_prefix": self.config.postgres.database_prefix,
                "template_database": self.config.postgres.template_database,
            },
            "blacklist": {
                stage: sorted(adapters)
                for stage, adapters in sorted(self.config.blacklist.adapters.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def exclusion_reason(self, stage: str, adapter: str) -> str | None:
        """Return the policy reason for excluding one adapter, if any."""
        if adapter in self.config.blacklist.adapters.get(stage, ()):
            return _BLACKLIST_REASON
        if stage == "walker" and adapter == "git-diff":
            return _FULL_SNAPSHOT_GIT_DIFF_REASON
        return None

    def restrict_search_space(
        self,
        space: SearchSpace,
        *,
        storage_adapter: str | None = None,
    ) -> SearchSpace:
        """Copy and restrict a search space after its static manifest preflight.

        The caller performs static manifest preflight first, then passes the
        selected experiment ``storage_adapter``. That adapter may have been a
        reasoned exclusion in the static search space.
        """
        source_storage = space.by_name.get("storage")
        if source_storage is None or source_storage.kind != "stage":
            raise ExperimentPolicyError("search space has no storage stage domain")
        selected_storage = storage_adapter or self.config.storage
        if selected_storage != self.config.storage:
            raise ExperimentPolicyError(
                f"storage adapter {selected_storage!r} does not match configured experiment "
                f"storage {self.config.storage!r}"
            )

        declared_storage = set(source_storage.choices) | set(
            space.excluded_adapters.get("storage", {})
        )
        if selected_storage not in declared_storage:
            raise ExperimentPolicyError(
                f"experiment storage {selected_storage!r} is not declared by the static manifest"
            )
        reason = self.exclusion_reason("storage", selected_storage)
        if reason is not None:
            raise ExperimentPolicyError(
                f"selected experiment storage {selected_storage!r} is excluded: {reason}"
            )

        restricted = copy.deepcopy(space)
        for dimension in restricted.enumerate_dimensions(kind="stage"):
            if dimension.name == "storage":
                continue
            choices: list[str] = []
            for adapter in dimension.choices:
                reason = self.exclusion_reason(dimension.name, adapter)
                if reason is None:
                    choices.append(adapter)
                    continue
                restricted.excluded_adapters.setdefault(dimension.name, {})[adapter] = reason
            if not choices:
                raise ExperimentPolicyError(
                    f"experiment policy leaves stage {dimension.name!r} with an empty domain"
                )
            dimension.choices = tuple(choices)
            if dimension.default not in dimension.choices:
                dimension.default = dimension.choices[0]

        storage = restricted.by_name["storage"]
        storage.choices = (selected_storage,)
        storage.default = selected_storage
        restricted.excluded_adapters.get("storage", {}).pop(selected_storage, None)
        return restricted

    def filter_matrix(self, matrix: Matrix) -> Matrix:
        """Copy a benchmark matrix and remove policy-excluded stage values."""
        filtered = copy.deepcopy(matrix)

        if filtered.baseline:
            for axis, value in filtered.baseline.items():
                if axis not in _STAGE_AXES:
                    continue
                adapter = _adapter_name(value)
                if adapter is None:
                    continue
                reason = self.exclusion_reason(axis, adapter)
                if reason is not None:
                    raise ExperimentPolicyError(
                        f"declared baseline {axis}={adapter!r} is excluded: {reason}"
                    )

        for axis, values in filtered.axes.items():
            if axis not in _STAGE_AXES:
                continue
            allowed = [
                value
                for value in values
                if (adapter := _adapter_name(value)) is None
                or self.exclusion_reason(axis, adapter) is None
            ]
            if not allowed:
                raise ExperimentPolicyError(
                    f"experiment policy leaves matrix axis {axis!r} with an empty domain"
                )
            filtered.axes[axis] = allowed

        for axis, values in filtered.axes.items():
            if not values:
                raise ExperimentPolicyError(
                    f"experiment policy leaves matrix axis {axis!r} with an empty domain"
                )

        try:
            filtered.expand()
        except MatrixError as exc:
            raise ExperimentPolicyError(
                f"experiment policy leaves zero matrix cells: {exc}"
            ) from exc
        if filtered.baseline and filtered.baseline_cell() is None:
            raise ExperimentPolicyError(
                "declared baseline is not present after experiment policy filtering"
            )
        return filtered


def _adapter_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("adapter"), str):
        return value["adapter"]
    return None


__all__ = ["ExperimentPolicy", "ExperimentPolicyError"]
