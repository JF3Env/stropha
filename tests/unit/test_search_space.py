"""Unit tests for the search-space manifest + loader + sampler (WIN-182/183/185)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from stropha.meta import (
    DEFAULT_MANIFEST,
    Constraint,
    Dimension,
    SearchSpace,
    SearchSpaceError,
    load_search_space,
)
from stropha.meta.search_space import (
    _sample_dimension,
    _when_matches,
    resolve_candidate_config,
)

# ---------------------------------------------------------------------------
# Bundled manifest sanity (WIN-182)
# ---------------------------------------------------------------------------


class TestBundledManifest:
    def test_default_manifest_path_exists(self) -> None:
        assert DEFAULT_MANIFEST.is_file()

    def test_loads_with_no_args(self) -> None:
        space = load_search_space()
        assert isinstance(space, SearchSpace)
        assert len(space.dimensions) > 0
        assert len(space.constraints) > 0

    def test_every_stage_has_choices(self) -> None:
        space = load_search_space()
        for dim in space.enumerate_dimensions(kind="stage"):
            assert dim.choices, f"stage {dim.name!r} has no choices"

    def test_indexing_split_matches_intent(self) -> None:
        """walker/chunker/enricher/embedder must be indexing=True;
        storage/retrieval/reranker must be indexing=False."""
        space = load_search_space()
        by_name = space.by_name
        for stage in ("walker", "chunker", "enricher", "embedder"):
            assert by_name[stage].indexing is True
        for stage in ("storage", "retrieval", "reranker"):
            assert by_name[stage].indexing is False

    def test_bundled_manifest_passes_strict_shape_validation(self) -> None:
        space = load_search_space(strict=True)
        assert all(
            dimension.binding
            for dimension in space.enumerate_dimensions()
        )


class TestResolveCandidateConfig:
    def test_resolves_every_bundled_dimension_without_mutating_base(self) -> None:
        space = load_search_space(strict=True)
        candidate = {dimension.name: dimension.default for dimension in space.dimensions}
        candidate.update(
            walker="filesystem",
            enricher="noop",
            embedder="model2vec",
            reranker="code-aware",
            hyde=True,
            hyde_code_tuned=False,
            query_rewrite=True,
            multi_query=True,
            multi_query_count=5,
            recursive=True,
            recursive_adjacency=20,
            adaptive_weighting=False,
            candidate_k=50,
            rrf_k=100,
            graph_fts_augment=False,
        )
        base = {
            "walker": {"adapter": "git-ls-files", "config": {"tracked": True}},
            "chunker": {"adapter": "tree-sitter-dispatch", "config": {"keep": True}},
            "enricher": {"adapter": "contextual", "config": {"model": "old"}},
            "embedder": {"adapter": "local", "config": {"model": "old"}},
            "storage": {"adapter": "sqlite-vec", "config": {"path": "index.db"}},
            "retrieval": {
                "adapter": "hybrid-rrf",
                "config": {"reranker": {"adapter": "llm", "config": {"model": "old"}}},
            },
        }

        resolved = resolve_candidate_config(
            candidate,
            base_pipeline_config=base,
            search_space=space,
        )

        assert base["walker"] == {
            "adapter": "git-ls-files",
            "config": {"tracked": True},
        }
        assert resolved.pipeline["walker"] == {"adapter": "filesystem"}
        assert resolved.pipeline["chunker"]["config"] == {"keep": True}
        assert resolved.pipeline["enricher"] == {"adapter": "noop"}
        assert resolved.pipeline["embedder"] == {"adapter": "model2vec"}
        assert resolved.pipeline["retrieval"]["config"] == {
            "reranker": {"adapter": "code-aware"},
            "hyde_enabled": True,
            "query_rewrite_enabled": True,
            "query_rewrite_max_terms": 6,
            "multi_query_enabled": True,
            "multi_query_count": 5,
            "multi_query_min_terms": 7,
            "recursive_enabled": True,
            "recursive_adjacency": 20,
            "adaptive_weighting": False,
            "candidate_k": 50,
            "rrf_k": 100,
        }
        assert resolved.retrieval_features == {"hyde_code_tuned": False}
        assert resolved.graph == {"fts_augment": False}

    def test_rejects_unknown_and_out_of_domain_dimensions(self) -> None:
        space = load_search_space(strict=True)
        with pytest.raises(SearchSpaceError, match="unknown dimensions"):
            space.resolve_candidate({"unknown": True}, base_pipeline_config={})
        with pytest.raises(SearchSpaceError, match="outside its domain"):
            space.resolve_candidate({"candidate_k": 11}, base_pipeline_config={})


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SearchSpaceError, match="does not exist"):
            load_search_space(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("not: valid:\n  - yaml:\n: torn")
        with pytest.raises(SearchSpaceError, match="malformed"):
            load_search_space(p)

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- one\n- two\n")
        with pytest.raises(SearchSpaceError, match="root must be a mapping"):
            load_search_space(p)

    def test_stages_not_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("stages: [one, two]\n")
        with pytest.raises(SearchSpaceError, match="stages must be a mapping"):
            load_search_space(p)

    def test_stage_not_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("stages:\n  walker: just-a-string\n")
        with pytest.raises(SearchSpaceError, match=r"stages\.walker must be a mapping"):
            load_search_space(p)

    def test_stage_missing_choices_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("stages:\n  walker:\n    indexing: true\n")
        with pytest.raises(SearchSpaceError, match="no `choices`"):
            load_search_space(p)

    def test_features_not_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("features: [hyde]\n")
        with pytest.raises(SearchSpaceError, match="features must be a mapping"):
            load_search_space(p)

    def test_feature_not_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("features:\n  hyde: just-a-string\n")
        with pytest.raises(SearchSpaceError, match=r"features\.hyde must be a mapping"):
            load_search_space(p)

    def test_feature_bad_type_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("features:\n  hyde:\n    type: float\n")
        with pytest.raises(SearchSpaceError, match=r"\.type must be 'bool' or 'int'"):
            load_search_space(p)

    def test_feature_bad_range_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "features:\n  multi:\n    type: int\n    range: [1, 2, 3]\n"
        )
        with pytest.raises(SearchSpaceError, match="range must be"):
            load_search_space(p)

    def test_constraints_not_list_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("constraints: not-a-list\n")
        with pytest.raises(SearchSpaceError, match="constraints must be a list"):
            load_search_space(p)

    def test_constraint_not_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("constraints:\n  - just-a-string\n")
        with pytest.raises(SearchSpaceError, match="constraint must be a mapping"):
            load_search_space(p)

    def test_constraint_bad_kind_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "constraints:\n  - kind: forbidden\n    when: {}\n"
        )
        with pytest.raises(SearchSpaceError, match=r"constraint\.kind must be"):
            load_search_space(p)

    def test_duplicate_yaml_key_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("features:\n  hyde: {type: bool, default: false}\n"
                     "  hyde: {type: bool, default: true}\n")
        with pytest.raises(SearchSpaceError, match="duplicate manifest key"):
            load_search_space(p)

    def test_invalid_int_default_and_step_raise(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "features:\n  k:\n    type: int\n    range: [10, 30]\n"
            "    step: 10\n    default: 15\n"
        )
        with pytest.raises(SearchSpaceError, match="outside its stepped range"):
            load_search_space(p)

    def test_dangling_constraint_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "features:\n  enabled: {type: bool, default: false}\n"
            "constraints:\n  - kind: gated_by\n    when: {missing: true}\n"
            "    set: {enabled: false}\n"
        )
        with pytest.raises(SearchSpaceError, match="unknown dimension 'missing'"):
            load_search_space(p)

    def test_strict_requires_feature_binding(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text("features:\n  hyde: {type: bool, default: false}\n")
        with pytest.raises(SearchSpaceError, match="runtime config binding"):
            load_search_space(p, strict=True)

    def test_strict_accepts_feature_binding(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "features:\n  hyde:\n    type: bool\n    default: false\n"
            "    binding: pipeline.retrieval.config.hyde_enabled\n"
        )
        assert load_search_space(p, strict=True).by_name["hyde"].binding

    def test_strict_rejects_duplicate_feature_bindings(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        p.write_text(
            "features:\n"
            "  hyde: {type: bool, default: false, binding: retrieval.enabled}\n"
            "  recursive: {type: bool, default: false, binding: retrieval.enabled}\n"
        )
        with pytest.raises(SearchSpaceError, match="bindings must be unique"):
            load_search_space(p, strict=True)


# ---------------------------------------------------------------------------
# enumerate_dimensions
# ---------------------------------------------------------------------------


class TestEnumerateDimensions:
    def test_filter_indexing_true(self) -> None:
        space = load_search_space()
        indexing = space.indexing_dimensions()
        # walker / chunker / enricher / embedder are all indexing.
        names = {d.name for d in indexing}
        assert {"walker", "chunker", "enricher", "embedder"}.issubset(names)
        # storage / retrieval / reranker are NOT indexing.
        assert "storage" not in names
        assert "retrieval" not in names
        assert "reranker" not in names

    def test_filter_indexing_false(self) -> None:
        space = load_search_space()
        query = space.query_dimensions()
        names = {d.name for d in query}
        assert {"storage", "retrieval", "reranker"}.issubset(names)
        assert "walker" not in names

    def test_filter_kind(self) -> None:
        space = load_search_space()
        stages = space.enumerate_dimensions(kind="stage")
        features = space.enumerate_dimensions(kind="feature")
        assert all(d.kind == "stage" for d in stages)
        assert all(d.kind == "feature" for d in features)
        assert len(stages) + len(features) == len(space.dimensions)

    def test_no_filter_returns_all(self) -> None:
        space = load_search_space()
        assert len(space.enumerate_dimensions()) == len(space.dimensions)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_is_deterministic_with_seeded_rng(self) -> None:
        space = load_search_space()
        a = space.sample(rng=random.Random(42), apply_constraints=False)
        b = space.sample(rng=random.Random(42), apply_constraints=False)
        assert a == b

    def test_sample_covers_every_dimension(self) -> None:
        space = load_search_space()
        cfg = space.sample(apply_constraints=False)
        assert set(cfg) == {d.name for d in space.dimensions}

    def test_bool_sample_in_domain(self) -> None:
        dim = Dimension(name="x", kind="feature", indexing=False, type="bool")
        v = _sample_dimension(dim, random.Random(7))
        assert v in (True, False)

    def test_int_sample_in_range_with_step(self) -> None:
        dim = Dimension(
            name="x", kind="feature", indexing=False, type="int",
            range=(100, 1000), step=100,
        )
        for seed in range(20):
            v = _sample_dimension(dim, random.Random(seed))
            assert 100 <= v <= 1000
            assert v % 100 == 0

    def test_int_no_range_raises(self) -> None:
        dim = Dimension(name="x", kind="feature", indexing=False, type="int")
        with pytest.raises(SearchSpaceError, match="no range"):
            _sample_dimension(dim, random.Random())

    def test_choice_no_choices_raises(self) -> None:
        dim = Dimension(name="x", kind="stage", indexing=True, type="choice")
        with pytest.raises(SearchSpaceError, match="no choices"):
            _sample_dimension(dim, random.Random())


# ---------------------------------------------------------------------------
# Constraint engine (WIN-185)
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_when_matches_full_match(self) -> None:
        assert _when_matches({"embedder": "voyage"}, {"embedder": "voyage", "x": 1}) is True

    def test_when_matches_no_match(self) -> None:
        assert _when_matches({"embedder": "voyage"}, {"embedder": "local"}) is False

    def test_when_matches_partial_dict(self) -> None:
        # `when` requires both keys to match.
        assert _when_matches({"a": 1, "b": 2}, {"a": 1, "b": 99}) is False

    def test_when_matches_empty_when_always_matches(self) -> None:
        assert _when_matches({}, {"a": 1}) is True

    def test_requires_env_raises_when_unset(self) -> None:
        space = SearchSpace(
            dimensions=[
                Dimension(name="embedder", kind="stage", indexing=True,
                          type="choice", choices=("voyage",)),
            ],
            constraints=[
                Constraint(
                    kind="requires_env",
                    when={"embedder": "voyage"},
                    require_env="VOYAGE_API_KEY",
                ),
            ],
        )
        # Passing an empty env triggers the constraint.
        with pytest.raises(SearchSpaceError, match="require_env"):
            space.apply_constraints({"embedder": "voyage"}, env={})

    def test_requires_env_pass_when_set(self) -> None:
        space = SearchSpace(
            dimensions=[],
            constraints=[
                Constraint(
                    kind="requires_env",
                    when={"embedder": "voyage"},
                    require_env="VOYAGE_API_KEY",
                ),
            ],
        )
        out = space.apply_constraints(
            {"embedder": "voyage"},
            env={"VOYAGE_API_KEY": "sk-fake"},
        )
        assert out == {"embedder": "voyage"}

    def test_requires_env_skipped_when_when_does_not_match(self) -> None:
        space = SearchSpace(
            dimensions=[],
            constraints=[
                Constraint(
                    kind="requires_env",
                    when={"embedder": "voyage"},
                    require_env="VOYAGE_API_KEY",
                ),
            ],
        )
        # embedder=local doesn't trigger the voyage constraint.
        out = space.apply_constraints({"embedder": "local"}, env={})
        assert out == {"embedder": "local"}

    def test_gated_by_overrides_dependent_keys(self) -> None:
        space = SearchSpace(
            dimensions=[],
            constraints=[
                Constraint(
                    kind="gated_by",
                    when={"multi_query": False},
                    set={"multi_query_count": 3},
                ),
            ],
        )
        out = space.apply_constraints(
            {"multi_query": False, "multi_query_count": 99}
        )
        assert out["multi_query_count"] == 3

    def test_gated_by_skipped_when_when_does_not_match(self) -> None:
        space = SearchSpace(
            dimensions=[],
            constraints=[
                Constraint(
                    kind="gated_by",
                    when={"multi_query": False},
                    set={"multi_query_count": 3},
                ),
            ],
        )
        out = space.apply_constraints(
            {"multi_query": True, "multi_query_count": 5}
        )
        assert out["multi_query_count"] == 5  # unchanged

    def test_sample_with_constraints_uses_default_env(self) -> None:
        """``apply_constraints=True`` with default env=None forces env_map={};
        a voyage sample is rejected with SearchSpaceError on retry."""
        # Use a constrained mini-space so we can force the rejection.
        mini = SearchSpace(
            dimensions=[
                Dimension(
                    name="embedder", kind="stage", indexing=True,
                    type="choice", choices=("voyage",),  # only choice
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
        with pytest.raises(SearchSpaceError):
            mini.sample(apply_constraints=True, env={})

    def test_apply_constraints_does_not_mutate_input(self) -> None:
        space = load_search_space()
        original = {"multi_query": False, "multi_query_count": 99}
        snapshot = dict(original)
        space.apply_constraints(original, env={})
        assert original == snapshot


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_meta_re_exports(self) -> None:
        from stropha import meta
        for name in (
            "DEFAULT_MANIFEST",
            "Constraint",
            "Dimension",
            "SearchSpace",
            "SearchSpaceError",
            "load_search_space",
            "discover_gaps",
        ):
            assert hasattr(meta, name), f"missing export: {name}"


# ---------------------------------------------------------------------------
# WIN-184 — registry vs manifest drift
# ---------------------------------------------------------------------------


class TestDiscoverGaps:
    def test_no_drift_returns_empty_dicts(self) -> None:
        from stropha.meta import discover_gaps

        space = load_search_space()
        # Fake registry == manifest for every stage.
        fake_registry = {
            dim.name: list(dim.choices)
            for dim in space.enumerate_dimensions(kind="stage")
        }
        gaps = discover_gaps(space, available_adapters=fake_registry)
        assert gaps == {"missing_from_manifest": {}, "missing_from_registry": {}}

    def test_registry_has_extra_adapter_reported(self) -> None:
        from stropha.meta import discover_gaps

        space = load_search_space()
        fake_registry = {
            dim.name: list(dim.choices)
            for dim in space.enumerate_dimensions(kind="stage")
        }
        # Pretend the registry registered a new walker adapter that
        # the manifest hasn't been updated for.
        fake_registry["walker"].append("brand-new-walker")
        gaps = discover_gaps(space, available_adapters=fake_registry)
        assert gaps["missing_from_manifest"]["walker"] == ["brand-new-walker"]
        assert gaps["missing_from_registry"] == {}

    def test_manifest_has_extra_adapter_reported(self) -> None:
        from stropha.meta import discover_gaps

        space = load_search_space()
        # Drop one of the walker choices from the registry.
        fake_registry = {
            dim.name: list(dim.choices)
            for dim in space.enumerate_dimensions(kind="stage")
        }
        removed = fake_registry["walker"].pop()
        gaps = discover_gaps(space, available_adapters=fake_registry)
        assert removed in gaps["missing_from_registry"]["walker"]
        assert gaps["missing_from_manifest"] == {}

    def test_default_uses_real_registry(self) -> None:
        """Without `available_adapters`, the helper queries the live
        ``stropha.pipeline.registry``. Should not raise even if no drift."""
        from stropha.meta import discover_gaps

        space = load_search_space()
        gaps = discover_gaps(space)
        # The shape is always the same — two dicts, possibly empty.
        assert set(gaps) == {"missing_from_manifest", "missing_from_registry"}

    def test_registry_lookup_failure_yields_empty_list_for_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``available_for_stage`` raises for an unknown stage, the
        defensive branch treats it as empty and surfaces every manifest
        choice as 'missing from registry'."""
        from stropha.meta import discover_gaps

        space = load_search_space()

        def boom(_stage: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated registry lookup failure")

        monkeypatch.setattr(
            "stropha.pipeline.registry.available_for_stage", boom
        )
        gaps = discover_gaps(space)
        # Every stage's manifest choices land in missing_from_registry
        # because we treated the registry as empty.
        for dim in space.enumerate_dimensions(kind="stage"):
            assert dim.name in gaps["missing_from_registry"]

    def test_complete_registry_reports_unmanifested_stage(self) -> None:
        from stropha.meta import discover_gaps

        space = SearchSpace(dimensions=[
            Dimension(name="walker", kind="stage", indexing=True,
                      choices=("filesystem",)),
        ])
        gaps = discover_gaps(space, available_adapters={
            "walker": ["filesystem"],
            "router": ["pattern"],
        })
        assert gaps["missing_from_manifest"] == {"router": ["pattern"]}

    def test_excluded_adapter_with_reason_is_not_a_gap(self) -> None:
        from stropha.meta import discover_gaps

        space = SearchSpace(
            dimensions=[Dimension(name="storage", kind="stage", indexing=False,
                                  choices=("sqlite-vec",))],
            excluded_adapters={"storage": {"postgres": "not isolated in search"}},
        )
        gaps = discover_gaps(space, available_adapters={
            "storage": ["sqlite-vec", "postgres"],
        })
        assert gaps == {"missing_from_manifest": {}, "missing_from_registry": {}}

    def test_strict_drift_raises(self) -> None:
        from stropha.meta import discover_gaps

        space = SearchSpace(dimensions=[
            Dimension(name="walker", kind="stage", indexing=True, choices=("a",)),
        ])
        with pytest.raises(SearchSpaceError, match="preflight failed"):
            discover_gaps(space, available_adapters={"walker": ["a", "b"]}, strict=True)

    def test_strict_registry_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stropha.meta import discover_gaps

        monkeypatch.setattr(
            "stropha.pipeline.registry.all_adapters",
            lambda: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
        )
        with pytest.raises(SearchSpaceError, match="registry discovery failed"):
            discover_gaps(load_search_space(), strict=True)

    def test_consumer_dimension_disagreement_reported_and_strictly_rejected(self) -> None:
        from stropha.meta import discover_gaps

        space = SearchSpace(dimensions=[
            Dimension(name="walker", kind="stage", indexing=True, choices=("a",)),
            Dimension(name="hyde", kind="feature", indexing=False,
                      type="bool", default=False),
        ])
        adapters = {"walker": ["a"]}
        gaps = discover_gaps(
            space,
            available_adapters=adapters,
            supported_dimensions={"walker", "unknown"},
        )
        assert gaps["missing_from_consumer"] == {"dimensions": ["hyde"]}
        assert gaps["missing_from_manifest_dimensions"] == {"dimensions": ["unknown"]}
        with pytest.raises(SearchSpaceError, match="dimension disagreement"):
            discover_gaps(
                space,
                available_adapters=adapters,
                supported_dimensions={"walker"},
                strict=True,
            )
