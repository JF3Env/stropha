"""Tests for the centralized config introspection used by the MCP ``get_config`` tool.

We care about three guarantees:

1. ``get_config_info()`` exposes the env vars still in the registry
   (STROPHA_TARGET_REPO + STROPHA_HOOK_*) and the YAML-backed settings.
2. Secrets loaded from YAML are redacted in introspection output.
3. Back-compat keys are preserved for existing callers.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from stropha.config import _ENV_REGISTRY, get_config_info, load_config, reset_config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Each test gets an isolated home dir so ~/.stropha/config.yaml is fresh."""
    stropha_dir = tmp_path / ".stropha"
    stropha_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    # Remove any stale STROPHA_TARGET_REPO so tests start clean.
    monkeypatch.delenv("STROPHA_TARGET_REPO", raising=False)
    reset_config()
    yield
    reset_config()


def _write_config(tmp_path, content: str) -> None:
    (tmp_path / ".stropha" / "config.yaml").write_text(textwrap.dedent(content))


def test_registry_covers_hook_and_target_repo(tmp_path):
    """The registry must cover STROPHA_TARGET_REPO and all STROPHA_HOOK_* vars."""
    _write_config(tmp_path, "")
    info = get_config_info()

    expected = {
        "STROPHA_TARGET_REPO",
        "STROPHA_HOOK_LOG", "STROPHA_HOOK_TIMEOUT", "STROPHA_HOOK_SKIP",
        "STROPHA_HOOK_UV", "STROPHA_HOOK_GRAPHIFY", "STROPHA_HOOK_NO_GRAPHIFY",
        "STROPHA_HOOK_PROJECT_DIR", "STROPHA_HOOK_INDEX_PATH",
    }
    missing = expected - set(info["sources"])
    assert not missing, f"Missing from registry: {sorted(missing)}"


def test_secrets_from_yaml_are_redacted(tmp_path):
    """API keys in YAML must NEVER appear verbatim in the introspection output."""
    _write_config(tmp_path, """
        secrets:
          voyage_api_key: "pa-secret-do-not-leak"
          anthropic_api_key: "sk-also-secret"
    """)

    info = get_config_info()
    assert info["settings"]["voyage_api_key"] == "(set)"
    assert info["settings"]["anthropic_api_key"] == "(set)"
    # Defensive: the raw secret never appears anywhere in the structure.
    serialised = repr(info)
    assert "pa-secret-do-not-leak" not in serialised
    assert "sk-also-secret" not in serialised


def test_secrets_absent_when_not_set(tmp_path):
    """When secrets are not in YAML, introspection reports None."""
    _write_config(tmp_path, "")
    info = get_config_info()
    assert info["settings"]["voyage_api_key"] is None
    assert info["settings"]["anthropic_api_key"] is None


def test_use_voyage_true_when_key_present(tmp_path):
    """use_voyage must be True when voyage_api_key is set in YAML."""
    _write_config(tmp_path, """
        secrets:
          voyage_api_key: "pa-test"
    """)
    info = get_config_info()
    assert info["settings"]["use_voyage"] is True
    assert info["config"]["use_voyage"] is True


def test_use_voyage_false_without_key(tmp_path):
    """use_voyage must be False when no voyage_api_key in YAML."""
    _write_config(tmp_path, "")
    info = get_config_info()
    assert info["settings"]["use_voyage"] is False


def test_paths_from_yaml(tmp_path):
    """index_path and target_repo must come from YAML paths section."""
    _write_config(tmp_path, f"""
        paths:
          index: {tmp_path}/.stropha/custom.db
    """)
    info = get_config_info()
    assert "custom.db" in info["config"]["index_path"]


def test_target_repo_env_override(tmp_path, monkeypatch):
    """STROPHA_TARGET_REPO env var must still override the default target."""
    _write_config(tmp_path, "")
    monkeypatch.setenv("STROPHA_TARGET_REPO", str(tmp_path))
    reset_config()

    info = get_config_info()
    assert info["config"]["target_repo"] == str(tmp_path)
    assert info["sources"]["STROPHA_TARGET_REPO"] == "env"


def test_back_compat_keys_preserved(tmp_path):
    """Existing callers reading ``info['config']['…']`` must keep working."""
    _write_config(tmp_path, "")
    info = get_config_info()
    legacy = info["config"]
    for key in (
        "index_path", "target_repo", "log_level", "max_file_bytes",
        "use_voyage", "local_embed_model", "voyage_embed_model",
        "voyage_embed_dim",
    ):
        assert key in legacy, f"Back-compat key {key!r} missing from info['config']"


def test_settings_dump_has_mcp_fields(tmp_path):
    """The ``settings`` block must expose MCP watcher fields."""
    _write_config(tmp_path, "")
    info = get_config_info()
    for key in ("mcp_watch", "mcp_watch_interval", "mcp_watch_debounce"):
        assert key in info["settings"], f"Field {key!r} missing from settings dump"


def test_env_groups_partition_registry(tmp_path):
    """Every registered var must be reachable through ``env_groups`` exactly once."""
    _write_config(tmp_path, "")
    info = get_config_info()
    flattened: list[str] = []
    for group, vars_ in info["env_groups"].items():
        assert isinstance(group, str)
        for var_name, meta in vars_.items():
            flattened.append(var_name)
            assert {"value", "default", "source", "description", "secret"} <= meta.keys()

    assert sorted(flattened) == sorted(info["sources"].keys())


def test_embedder_from_yaml(tmp_path):
    """Embedder model settings are read from YAML, not env vars."""
    _write_config(tmp_path, """
        embedder:
          voyage_model: voyage-code-3
          voyage_dim: 1024
          local_model: BAAI/bge-small-en-v1.5
    """)
    cfg = load_config()
    assert cfg.embedder.voyage_model == "voyage-code-3"
    assert cfg.embedder.voyage_dim == 1024
    assert cfg.embedder.local_model == "BAAI/bge-small-en-v1.5"


def test_logging_from_yaml(tmp_path):
    """Log level is read from YAML logging.level."""
    _write_config(tmp_path, """
        logging:
          level: DEBUG
    """)
    cfg = load_config()
    assert cfg.log_level == "DEBUG"


def test_mcp_from_yaml(tmp_path):
    """MCP watch settings are read from YAML."""
    _write_config(tmp_path, """
        mcp:
          watch: false
          watch_interval: 5.0
    """)
    cfg = load_config()
    assert cfg.mcp_watch is False
    assert cfg.mcp_watch_interval == 5.0
