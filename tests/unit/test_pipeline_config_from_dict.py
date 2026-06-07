"""Unit tests for ``pipeline_config_from_dict`` (WIN-179).

The Meta-Harness loop must not be contaminated by ``~/.stropha/config.yaml``
or ``./stropha.yaml`` or ``VOYAGE_API_KEY``. These tests verify the helper
gives the same answer on every host.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stropha.pipeline import pipeline_config_from_dict


class TestIsolation:
    def test_ignores_user_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when ~/.stropha/config.yaml exists, the helper ignores it."""
        fake_home = tmp_path / "home"
        (fake_home / ".stropha").mkdir(parents=True)
        (fake_home / ".stropha" / "config.yaml").write_text(
            "embedder:\n  adapter: voyage\n"
        )
        monkeypatch.setenv("HOME", str(fake_home))
        cfg = pipeline_config_from_dict({})
        # Default embedder is 'local', not the 'voyage' the user YAML
        # would otherwise force.
        assert cfg["embedder"]["adapter"] == "local"

    def test_ignores_project_yaml(self, tmp_path: Path) -> None:
        # Even when called from a cwd containing stropha.yaml, the
        # project file is not read.
        (tmp_path / "stropha.yaml").write_text(
            "enricher:\n  adapter: contextual\n"
        )
        old_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            cfg = pipeline_config_from_dict({})
        finally:
            os.chdir(old_cwd)
        # Default enricher is 'hierarchical', not the 'contextual' the project YAML asks for.
        assert cfg["enricher"]["adapter"] == "hierarchical"

    def test_ignores_voyage_api_key_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VOYAGE_API_KEY set in env must NOT push the embedder to voyage."""
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-fake")
        cfg = pipeline_config_from_dict({})
        assert cfg["embedder"]["adapter"] == "local"


class TestOverridesApplied:
    def test_explicit_adapter_overrides_default(self) -> None:
        cfg = pipeline_config_from_dict({
            "embedder": {"adapter": "voyage", "config": {"model": "voyage-code-3"}},
        })
        assert cfg["embedder"]["adapter"] == "voyage"
        assert cfg["embedder"]["config"]["model"] == "voyage-code-3"

    def test_deep_merge_preserves_defaults(self) -> None:
        cfg = pipeline_config_from_dict({
            "enricher": {"config": {"flag": True}},
        })
        # enricher.adapter wasn't set so it stays default ('hierarchical').
        assert cfg["enricher"]["adapter"] == "hierarchical"
        assert cfg["enricher"]["config"] == {"flag": True}

    def test_storage_path_materialised_by_default(self) -> None:
        cfg = pipeline_config_from_dict({})
        assert cfg["storage"]["config"]["path"]  # non-empty

    def test_storage_path_skipped_when_flag_false(self) -> None:
        cfg = pipeline_config_from_dict({}, resolve_storage_path=False)
        assert cfg["storage"]["config"] == {}

    def test_explicit_storage_path_kept(self) -> None:
        cfg = pipeline_config_from_dict({
            "storage": {"config": {"path": "/tmp/explicit.db"}},
        })
        assert cfg["storage"]["config"]["path"] == "/tmp/explicit.db"

    def test_empty_overrides_returns_pure_defaults(self) -> None:
        cfg = pipeline_config_from_dict({})
        # The full default skeleton is materialised even without overrides.
        assert cfg["walker"]["adapter"] == "git-ls-files"
        assert cfg["chunker"]["adapter"] == "tree-sitter-dispatch"
        assert cfg["enricher"]["adapter"] == "hierarchical"
        assert cfg["embedder"]["adapter"] == "local"
        assert cfg["storage"]["adapter"] == "sqlite-vec"
        assert cfg["retrieval"]["adapter"] == "hybrid-rrf"


class TestDeterminism:
    def test_same_overrides_same_output(self) -> None:
        a = pipeline_config_from_dict({"embedder": {"adapter": "local"}})
        b = pipeline_config_from_dict({"embedder": {"adapter": "local"}})
        assert a == b

    def test_no_mutation_of_overrides_arg(self) -> None:
        overrides = {"embedder": {"adapter": "voyage"}}
        original = dict(overrides)
        pipeline_config_from_dict(overrides)
        assert overrides == original


class TestPackageExports:
    def test_pipeline_re_exports(self) -> None:
        from stropha import pipeline
        assert hasattr(pipeline, "pipeline_config_from_dict")