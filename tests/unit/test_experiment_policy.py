"""Focused tests for machine-local experiment restrictions."""

from __future__ import annotations

import copy

import pytest

from stropha.bench.matrix import Matrix
from stropha.config import (
    Config,
    ExperimentBlacklistConfig,
    ExperimentPostgresConfig,
    ExperimentsConfig,
    ExperimentStorage,
)
from stropha.errors import ConfigError
from stropha.experiment_policy import ExperimentPolicy, ExperimentPolicyError
from stropha.meta.search_space import Dimension, SearchSpace


def _policy(
    *,
    storage: ExperimentStorage = "postgres",
    blacklist: dict[str, tuple[str, ...]] | None = None,
    dsn: str | None = "postgresql://admin:secret@localhost/stropha_bench",
) -> ExperimentPolicy:
    experiments = ExperimentsConfig(
        storage=storage,
        postgres=ExperimentPostgresConfig(admin_dsn=dsn),
        blacklist=ExperimentBlacklistConfig(adapters=blacklist or {}),
    )
    return ExperimentPolicy.from_config(Config(experiments=experiments))


def _search_space() -> SearchSpace:
    return SearchSpace(
        dimensions=[
            Dimension(
                name="walker",
                kind="stage",
                indexing=True,
                choices=("git-ls-files", "git-diff"),
                default="git-diff",
            ),
            Dimension(
                name="embedder",
                kind="stage",
                indexing=True,
                choices=("voyage", "local"),
                default="voyage",
            ),
            Dimension(
                name="storage",
                kind="stage",
                indexing=False,
                choices=("sqlite-vec",),
                default="sqlite-vec",
            ),
        ],
        excluded_adapters={"storage": {"postgres": "isolated experiments opt in explicitly"}},
    )


def test_policy_requires_postgres_dsn_only_when_constructed_for_a_run() -> None:
    experiments = ExperimentsConfig(storage="postgres")

    assert experiments.postgres.admin_dsn is None
    with pytest.raises(ConfigError, match="admin_dsn is required"):
        ExperimentPolicy.from_config(experiments)


def test_fingerprint_is_stable_and_contains_no_secret_material() -> None:
    first = _policy(
        dsn="postgresql://admin:first-password@localhost/stropha_bench",
        blacklist={"embedder": ("voyage", "local")},
    )
    second = _policy(
        dsn="postgresql://admin:second-password@localhost/stropha_bench",
        blacklist={"embedder": ("local", "voyage")},
    )

    assert first.fingerprint == first.fingerprint
    assert first.fingerprint == second.fingerprint
    assert "first-password" not in first.fingerprint
    assert len(first.fingerprint) == 64


def test_restrict_search_space_filters_and_pins_storage_without_mutation() -> None:
    source = _search_space()
    original = copy.deepcopy(source)
    policy = _policy(blacklist={"embedder": ("voyage",)})

    restricted = policy.restrict_search_space(source, storage_adapter="postgres")

    assert restricted.by_name["walker"].choices == ("git-ls-files",)
    assert restricted.by_name["walker"].default == "git-ls-files"
    assert "full-snapshot" in restricted.excluded_adapters["walker"]["git-diff"]
    assert restricted.by_name["embedder"].choices == ("local",)
    assert restricted.by_name["embedder"].default == "local"
    assert restricted.by_name["storage"].choices == ("postgres",)
    assert restricted.by_name["storage"].default == "postgres"
    assert "postgres" not in restricted.excluded_adapters["storage"]
    assert source.dimensions == original.dimensions
    assert source.excluded_adapters == original.excluded_adapters


def test_restrict_search_space_requires_configured_and_declared_storage() -> None:
    source = _search_space()
    policy = _policy()

    with pytest.raises(ExperimentPolicyError, match="does not match configured"):
        policy.restrict_search_space(source, storage_adapter="sqlite-vec")

    source.excluded_adapters["storage"].pop("postgres")
    with pytest.raises(ExperimentPolicyError, match="not declared by the static manifest"):
        policy.restrict_search_space(source, storage_adapter="postgres")


def test_restrict_search_space_fails_when_a_stage_domain_becomes_empty() -> None:
    source = _search_space()
    policy = _policy(blacklist={"embedder": ("voyage", "local")})

    with pytest.raises(ExperimentPolicyError, match=r"embedder.*empty domain"):
        policy.restrict_search_space(source, storage_adapter="postgres")


def test_filter_matrix_handles_strings_and_adapter_mappings_without_mutation() -> None:
    matrix = Matrix(
        axes={
            "embedder": [{"adapter": "voyage", "dim": 512}, {"adapter": "local"}],
            "enricher": ["noop", "contextual"],
            "walker": ["git-ls-files", "git-diff"],
            "hyde": [False, True],
        },
        baseline={
            "embedder": {"adapter": "local"},
            "enricher": "noop",
            "walker": "git-ls-files",
            "hyde": False,
        },
    )
    original = copy.deepcopy(matrix)
    policy = _policy(blacklist={"embedder": ("voyage",), "enricher": ("contextual",)})

    filtered = policy.filter_matrix(matrix)

    assert filtered.axes["embedder"] == [{"adapter": "local"}]
    assert filtered.axes["enricher"] == ["noop"]
    assert filtered.axes["walker"] == ["git-ls-files"]
    assert filtered.axes["hyde"] == [False, True]
    assert filtered.baseline == matrix.baseline
    assert filtered.baseline_cell() is not None
    assert matrix == original


def test_filter_matrix_rejects_blacklisted_baseline() -> None:
    matrix = Matrix(
        axes={"embedder": ["voyage", "local"]},
        baseline={"embedder": "voyage"},
    )

    with pytest.raises(ExperimentPolicyError, match=r"declared baseline.*excluded"):
        _policy(blacklist={"embedder": ("voyage",)}).filter_matrix(matrix)


def test_filter_matrix_rejects_empty_domain_and_zero_cells() -> None:
    policy = _policy(blacklist={"embedder": ("voyage",)})
    with pytest.raises(ExperimentPolicyError, match=r"embedder.*empty domain"):
        policy.filter_matrix(Matrix(axes={"embedder": ["voyage"]}))

    constrained = Matrix(
        axes={"embedder": ["voyage", "local"]},
        constraints=[{"skip_when": {"embedder": "local"}}],
    )
    with pytest.raises(ExperimentPolicyError, match="zero matrix cells"):
        policy.filter_matrix(constrained)
