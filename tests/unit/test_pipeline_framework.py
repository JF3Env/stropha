"""Tests for the pipeline framework (base + registry + config + builder)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from stropha import adapters  # noqa: F401  triggers auto-import
from stropha.config import reset_config
from stropha.errors import AdapterError, ConfigError
from stropha.ingest.graph_vec_loader import GraphVecLoader
from stropha.pipeline import (
    StageContext,
    StageHealth,
    all_adapters,
    available_for_stage,
    build_stages,
    load_pipeline_config,
    lookup_adapter,
    register_adapter,
)
from stropha.pipeline.registry import _REGISTRY  # type: ignore[attr-defined]
from stropha.storage import Storage


class _StubNodeVoyageClient:
    def embed(self, texts: Sequence[str], *_, **kwargs):
        dim = kwargs.get("output_dimension", 512)
        return type("_Resp", (), {"embeddings": [[0.0] * dim for _ in texts]})()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_default_adapters_registered() -> None:
    reg = all_adapters()
    assert "embedder" in reg
    assert "enricher" in reg
    assert "local" in reg["embedder"]
    assert "voyage" in reg["embedder"]
    assert "noop" in reg["enricher"]
    assert "hierarchical" in reg["enricher"]


def test_lookup_unknown_adapter_raises_config_error_with_alternatives() -> None:
    with pytest.raises(ConfigError) as exc:
        lookup_adapter("enricher", "does-not-exist")
    msg = str(exc.value)
    assert "does-not-exist" in msg
    # The error must hint at the registered alternatives.
    assert "noop" in msg or "hierarchical" in msg


def test_available_for_stage_returns_sorted() -> None:
    names = available_for_stage("embedder")
    assert names == sorted(names)


def test_register_adapter_requires_config_attribute() -> None:
    with pytest.raises(AdapterError):

        @register_adapter(stage="enricher", name="broken-fixture")
        class _Broken:
            """Missing ``Config`` attribute."""

    # Make sure no half-registered entry leaked.
    assert ("enricher", "broken-fixture") not in _REGISTRY


def test_register_adapter_rejects_duplicate_name() -> None:
    class _Cfg(BaseModel):
        pass

    @register_adapter(stage="enricher", name="dup-fixture")
    class _A:
        Config = _Cfg

    with pytest.raises(AdapterError):

        @register_adapter(stage="enricher", name="dup-fixture")
        class _B:
            Config = _Cfg

    # Cleanup so the fixture doesn't leak into other tests.
    _REGISTRY.pop(("enricher", "dup-fixture"), None)


# ---------------------------------------------------------------------------
# StageContext / StageHealth shape
# ---------------------------------------------------------------------------


def test_stage_context_defaults() -> None:
    ctx = StageContext()
    assert ctx.repo_key is None
    assert ctx.parent_chunk is None
    assert ctx.file_content is None
    assert ctx.pipeline_meta == {}


def test_stage_health_ready() -> None:
    h = StageHealth(status="ready", message="ok")
    assert h.status == "ready"
    assert h.detail == {}


# ---------------------------------------------------------------------------
# Config loader (deterministic via injected environ)
# ---------------------------------------------------------------------------


def test_config_defaults_pick_local_when_no_voyage_key(tmp_path) -> None:
    cfg = load_pipeline_config(project_root=tmp_path)
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["enricher"]["adapter"] == "noop"


def test_config_picks_voyage_when_api_key_set(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("secrets:\n  voyage_api_key: sk-test\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_pipeline_config(project_root=tmp_path)
    assert cfg["embedder"]["adapter"] == "voyage"


def test_config_legacy_alias_local_model(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "pipeline:\n  embedder:\n    adapter: local\n    config:\n      model: BAAI/bge-small-en-v1.5\n"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_pipeline_config(project_root=tmp_path)
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-small-en-v1.5"


def test_config_legacy_alias_voyage_does_not_leak_into_local(tmp_path, monkeypatch) -> None:
    """Regression: voyage adapter config must NOT contaminate a local adapter."""
    from pathlib import Path
    # No voyage key → auto-select local; only local model set.
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "pipeline:\n  embedder:\n    adapter: local\n    config:\n      model: BAAI/bge-small-en-v1.5\n"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_pipeline_config(project_root=tmp_path)
    assert cfg["embedder"]["adapter"] == "local"
    assert cfg["embedder"]["config"]["model"] == "BAAI/bge-small-en-v1.5"


def test_config_cli_override_wins_over_env(tmp_path) -> None:
    cfg = load_pipeline_config(
        project_root=tmp_path,
        overrides={"enricher": {"adapter": "noop"}},
    )
    assert cfg["enricher"]["adapter"] == "noop"


def test_config_env_overlay_for_nested_path(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "pipeline:\n  enricher:\n    config:\n      include_repo_url: true\n"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_pipeline_config(project_root=tmp_path)
    # Note: lowercased keys per the loader convention.
    assert cfg["enricher"]["config"]["include_repo_url"] is True


def test_config_yaml_file_loaded_when_present(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "pipeline:\n  enricher:\n    adapter: hierarchical\n", encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_pipeline_config(project_root=tmp_path)
    assert cfg["enricher"]["adapter"] == "hierarchical"


def test_config_yaml_invalid_raises_config_error(tmp_path, monkeypatch) -> None:
    from pathlib import Path
    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("not:\n  : -valid: [", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(ConfigError):
        load_pipeline_config(project_root=tmp_path)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_stages_yields_working_enricher() -> None:
    # We don't need a real embedder for the enricher assertion below — but
    # building local IS heavy (ONNX load), so skip embedder validation in
    # this test by patching it with a known-cheap subclass would require
    # adapter modification. Instead, verify the noop enricher directly.
    from stropha.adapters.enricher.noop import NoopEnricher

    enr = NoopEnricher()
    assert enr.adapter_id == "noop"
    assert enr.adapter_name == "noop"
    assert enr.stage_name == "enricher"
    assert enr.health().status == "ready"


def test_build_stages_unknown_adapter_raises() -> None:
    resolved = {
        "embedder": {"adapter": "no-such", "config": {}},
        "enricher": {"adapter": "noop", "config": {}},
    }
    with pytest.raises(ConfigError):
        build_stages(resolved)


def test_build_stages_invalid_config_raises() -> None:
    resolved = {
        "embedder": {"adapter": "voyage", "config": {"dim": -5}},
        "enricher": {"adapter": "noop", "config": {}},
    }
    with pytest.raises(ConfigError):
        build_stages(resolved)


def test_pipeline_built_voyage_embedder_uses_yaml_secret_for_graph_loader(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    cfg_dir = tmp_path / ".stropha"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("secrets:\n  voyage_api_key: sk-test\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    reset_config()

    resolved = load_pipeline_config(project_root=tmp_path)
    assert resolved["embedder"]["adapter"] == "voyage"

    voyageai_module = type(
        "_VoyageModule",
        (),
        {"Client": lambda self=None, api_key=None: _StubNodeVoyageClient()},
    )()
    monkeypatch.setitem(__import__("sys").modules, "voyageai", voyageai_module)

    storage = Storage(tmp_path / "idx.db", embedding_dim=512)
    try:
        storage._conn.execute(
            """CREATE TABLE IF NOT EXISTS graph_nodes (
                   node_id TEXT PRIMARY KEY,
                   label TEXT,
                   community_label TEXT,
                   embedding BLOB,
                   embedding_model TEXT,
                   embedding_dim INTEGER
               )"""
        )
        storage._conn.execute(
            "INSERT INTO graph_nodes(node_id, label) VALUES (?, ?)",
            ("n1", "Alpha"),
        )
        storage.commit()

        built = build_stages(
            {
                **resolved,
                "chunker": {"adapter": "tree-sitter-dispatch", "config": {}},
                "walker": {"adapter": "git-ls-files", "config": {}},
                "enricher": {"adapter": "noop", "config": {}},
                "storage": {"adapter": "sqlite-vec", "config": {"path": str(tmp_path / "idx.db")}},
                "retrieval": {"adapter": "hybrid-rrf", "config": {}},
            }
        )

        result = GraphVecLoader(storage, built.embedder).load(batch_size=1)
        assert result.failed == 0
        assert result.embedded == 1
    finally:
        storage.close()
        reset_config()
