"""Tests for the centralized config introspection used by the MCP ``get_config`` tool.

We care about three guarantees:

1. ``get_config_info()`` exposes every env var listed in §2.5 of CLAUDE.md, not
   just the legacy four. New env vars must be registered.
2. Source detection (``env`` vs ``dotenv`` vs ``default``) is accurate.
3. Secrets are never leaked — they show as ``(set)`` or ``null``.
"""

from __future__ import annotations

import os

import pytest

from stropha.config import _ENV_REGISTRY, get_config_info, reset_config


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Each test starts with a clean slate of stropha env vars."""
    for var in _ENV_REGISTRY:
        monkeypatch.delenv(var.name, raising=False)
    reset_config()
    yield
    reset_config()


def test_every_known_env_var_is_in_sources(monkeypatch, tmp_path):
    """The registry MUST cover every env var the runtime consumes.

    If you add a new ``os.environ.get("STROPHA_FOO")`` somewhere in the code,
    add a matching ``_EnvVar(...)`` row to the registry. Catch the drift here.
    """
    # Set one var so .env loading is skipped (deterministic test).
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))

    info = get_config_info()

    # At minimum the categories users care about must be present.
    expected = {
        # Core
        "STROPHA_INDEX_PATH", "STROPHA_TARGET_REPO", "STROPHA_LOG_LEVEL",
        "STROPHA_MAX_FILE_BYTES",
        # Embedders
        "STROPHA_LOCAL_EMBED_MODEL", "STROPHA_VOYAGE_EMBED_MODEL",
        "STROPHA_VOYAGE_EMBED_DIM", "VOYAGE_API_KEY", "ANTHROPIC_API_KEY",
        # Inference
        "STROPHA_INFERENCE_BACKEND", "STROPHA_OLLAMA_MODEL", "OLLAMA_HOST",
        # Query intelligence
        "STROPHA_HYDE_ENABLED", "STROPHA_HYDE_CODE_TUNED",
        "STROPHA_QUERY_ROUTER_LLM",
        "STROPHA_QUERY_DECOMPOSITION_ENABLED", "STROPHA_QUERY_REWRITE_ENABLED",
        "STROPHA_MULTI_QUERY_ENABLED", "STROPHA_MULTI_QUERY_COUNT",
        "STROPHA_ADAPTIVE_WEIGHTING",
        # Cache
        "STROPHA_QUERY_CACHE_ENABLED", "STROPHA_QUERY_CACHE_SIZE",
        "STROPHA_QUERY_CACHE_TTL",
        # Retrieval
        "STROPHA_RECURSIVE_RETRIEVAL", "STROPHA_GRAPH_FTS_AUGMENT",
        "STROPHA_GRAPH_CONFIDENCE", "STROPHA_GRAPHIFY_OUT",
        # MCP watcher
        "STROPHA_MCP_WATCH", "STROPHA_MCP_WATCH_INTERVAL",
        "STROPHA_MCP_WATCH_DEBOUNCE",
        # Git url cache
        "STROPHA_GIT_URL_CACHE", "STROPHA_GIT_URL_TTL",
        "STROPHA_RENAME_THRESHOLD",
        # Hook
        "STROPHA_HOOK_LOG", "STROPHA_HOOK_TIMEOUT", "STROPHA_HOOK_SKIP",
        "STROPHA_HOOK_UV", "STROPHA_HOOK_GRAPHIFY", "STROPHA_HOOK_NO_GRAPHIFY",
        "STROPHA_HOOK_PROJECT_DIR", "STROPHA_HOOK_INDEX_PATH",
    }
    missing = expected - set(info["sources"])
    assert not missing, f"Missing from registry: {sorted(missing)}"


def test_source_env_vs_default(monkeypatch, tmp_path):
    """A var explicitly set before stropha starts MUST report source=env."""
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("STROPHA_HYDE_ENABLED", "1")

    info = get_config_info()
    assert info["sources"]["STROPHA_HYDE_ENABLED"] == "env"
    assert info["sources"]["STROPHA_QUERY_CACHE_ENABLED"] == "default"


def test_env_values_fill_in_defaults(monkeypatch, tmp_path):
    """When an env var is unset, ``env_values`` exposes the documented default.

    This is the point of the registry — the agent calling the MCP tool should
    see what the runtime will actually do, not just what was set.
    """
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))

    info = get_config_info()
    assert info["env_values"]["STROPHA_HYDE_ENABLED"] == "0"
    assert info["env_values"]["STROPHA_MULTI_QUERY_COUNT"] == "3"
    assert info["env_values"]["STROPHA_QUERY_CACHE_TTL"] == "3600"
    assert info["env_values"]["OLLAMA_HOST"] == "http://localhost:11434"


def test_secrets_are_redacted(monkeypatch, tmp_path):
    """API keys must NEVER appear verbatim in the introspection output."""
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-secret-do-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-also-secret")

    info = get_config_info()
    assert info["env_values"]["VOYAGE_API_KEY"] == "(set)"
    assert info["env_values"]["ANTHROPIC_API_KEY"] == "(set)"
    # And in the settings dump.
    assert info["settings"]["voyage_api_key"] == "(set)"
    assert info["settings"]["anthropic_api_key"] == "(set)"
    # Defensive: the raw secret never appears anywhere in the structure.
    serialised = repr(info)
    assert "pa-secret-do-not-leak" not in serialised
    assert "sk-also-secret" not in serialised


def test_env_groups_partition_registry(monkeypatch, tmp_path):
    """Every registered var must be reachable through ``env_groups`` exactly once."""
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))

    info = get_config_info()
    flattened: list[str] = []
    for group, vars_ in info["env_groups"].items():
        assert isinstance(group, str)
        for var_name, meta in vars_.items():
            flattened.append(var_name)
            assert {"value", "default", "source", "description", "secret"} <= meta.keys()

    assert sorted(flattened) == sorted(info["sources"].keys())


def test_back_compat_keys_preserved(monkeypatch, tmp_path):
    """Existing callers reading ``info['config']['…']`` must keep working."""
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))

    info = get_config_info()
    legacy = info["config"]
    for key in (
        "index_path", "target_repo", "log_level", "max_file_bytes",
        "use_voyage", "local_embed_model", "voyage_embed_model",
        "voyage_embed_dim",
    ):
        assert key in legacy


def test_settings_dump_mirrors_pydantic_model(monkeypatch, tmp_path):
    """The ``settings`` block exposes every Config field — future-proof."""
    monkeypatch.setenv("STROPHA_INDEX_PATH", str(tmp_path / "x.db"))

    info = get_config_info()
    # MCP watcher fields are on Config but were missing from the legacy block.
    for key in ("mcp_watch", "mcp_watch_interval", "mcp_watch_debounce"):
        assert key in info["settings"], f"Pydantic field {key!r} missing from settings dump"
