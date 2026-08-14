"""Tests for the centralized config introspection used by the MCP ``get_config`` tool.

We care about three guarantees:

1. ``get_config_info()`` exposes registered process arguments and YAML-backed
   settings without reviving setting environment fallbacks.
2. Secrets loaded from YAML are redacted in introspection output.
3. Back-compat keys are preserved for existing callers.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from stropha.config import get_config_info, load_config, reset_config
from stropha.errors import ConfigError


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


def test_registry_covers_runtime_process_variables(tmp_path):
    """The registry covers runtime, launcher, session, and hook process vars.

    Note: STROPHA_HOOK_INDEX_PATH was removed in v=9 — it was wired to
    ``cfg.hook.index_path`` but never propagated to the storage adapter,
    so the override silently did nothing. The unified YAML
    ``paths.index`` is the single source.
    """
    _write_config(tmp_path, "")
    info = get_config_info()

    expected = {
        "STROPHA_TARGET_REPO",
        "STROPHA_HOOK_LOG",
        "STROPHA_HOOK_TIMEOUT",
        "STROPHA_HOOK_SKIP",
        "STROPHA_HOOK_UV",
        "STROPHA_HOOK_GRAPHIFY",
        "STROPHA_HOOK_PROJECT_DIR",
        "STROPHA_HOOK_META_ITERATIONS",
        "STROPHA_HOOK_NO_META",
        "STROPHA_HOOK_NO_EVAL",
        "STROPHA_MCP_HOST",
        "STROPHA_MCP_PORT",
        "STROPHA_MCP_PROJECT_DIR",
        "STROPHA_MCP_PROBE_TIMEOUT",
        "STROPHA_CLIENT_ID",
        "STROPHA_TRACES_DIR",
        "STROPHA_USE_POSTGRES",
        "STROPHA_PG_CONTAINER",
        "STROPHA_PG_IMAGE",
        "STROPHA_PG_PORT",
        "STROPHA_PG_USER",
        "STROPHA_PG_PASSWORD",
        "STROPHA_PG_DB",
        "STROPHA_PG_VOLUME",
    }
    missing = expected - set(info["sources"])
    assert not missing, f"Missing from registry: {sorted(missing)}"
    # Regression guard: the dead env var must NOT come back.
    assert "STROPHA_HOOK_INDEX_PATH" not in info["sources"]


def test_registry_excludes_yaml_backed_settings(tmp_path):
    _write_config(tmp_path, "")
    names = set(get_config_info()["sources"])

    assert names.isdisjoint(
        {
            "STROPHA_ADAPTIVE_WEIGHTING",
            "STROPHA_RECURSIVE_RETRIEVAL",
            "STROPHA_RECURSIVE_ADJACENCY",
            "STROPHA_GRAPHIFY_OUT",
            "STROPHA_GIT_URL_CACHE",
            "STROPHA_GIT_URL_TTL",
            "STROPHA_RENAME_THRESHOLD",
            "STROPHA_PG_DSN",
            "VOYAGE_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
        }
    )


def test_registry_redacts_launcher_password(tmp_path, monkeypatch):
    _write_config(tmp_path, "")
    monkeypatch.setenv("STROPHA_PG_PASSWORD", "do-not-leak")

    info = get_config_info()

    assert info["env_values"]["STROPHA_PG_PASSWORD"] == "(set)"
    assert "do-not-leak" not in repr(info)


def test_secrets_from_yaml_are_redacted(tmp_path):
    """API keys in YAML must NEVER appear verbatim in the introspection output."""
    _write_config(
        tmp_path,
        """
        secrets:
          voyage_api_key: "pa-secret-do-not-leak"
          anthropic_api_key: "sk-also-secret"
    """,
    )

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
    _write_config(
        tmp_path,
        """
        secrets:
          voyage_api_key: "pa-test"
    """,
    )
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
    _write_config(
        tmp_path,
        f"""
        paths:
          index: {tmp_path}/.stropha/custom.db
    """,
    )
    info = get_config_info()
    assert "custom.db" in info["config"]["index_path"]


def test_diversity_packing_settings_from_yaml(tmp_path):
    _write_config(
        tmp_path,
        """
        retrieval_features:
          diversity_enabled: true
          diversity_max_chunks_per_file: 3
    """,
    )

    features = load_config(force_reload=True).retrieval_features

    assert features.diversity_enabled is True
    assert features.diversity_max_chunks_per_file == 3


def test_git_clone_settings_from_yaml(tmp_path):
    _write_config(
        tmp_path,
        """
        git:
          clone_depth: 7
          clone_timeout: 42
    """,
    )

    cfg = load_config()

    assert cfg.git.clone_depth == 7
    assert cfg.git.clone_timeout == 42


def test_experiment_settings_load_from_yaml(tmp_path):
    _write_config(
        tmp_path,
        """
        experiments:
          storage: postgres
          postgres:
            admin_dsn: postgresql://admin:secret@localhost/stropha_bench
            database_prefix: local_bench
            template_database: local_bench_template
          blacklist:
            adapters:
              embedder: [voyage]
              walker: [git-diff]
        """,
    )

    experiments = load_config().experiments

    assert experiments.storage == "postgres"
    assert experiments.postgres.database_prefix == "local_bench"
    assert experiments.postgres.template_database == "local_bench_template"
    assert experiments.blacklist.adapters == {
        "embedder": ("voyage",),
        "walker": ("git-diff",),
    }
    experiments.validate_for_run()


def test_postgres_experiment_missing_dsn_loads_but_fails_run_validation(tmp_path):
    _write_config(
        tmp_path,
        """
        experiments:
          storage: postgres
        """,
    )

    experiments = load_config().experiments

    assert experiments.postgres.admin_dsn is None
    with pytest.raises(ConfigError, match="admin_dsn is required"):
        experiments.validate_for_run()


@pytest.mark.parametrize(
    "experiments, match",
    [
        ({"storage": "memory"}, "storage must be"),
        ({"postgres": {"database_prefix": "Upper"}}, "database_prefix must match"),
        ({"postgres": {"database_prefix": "a" * 33}}, "at most 32"),
        ({"postgres": {"template_database": "Bad-Template"}}, "template_database must be"),
        ({"blacklist": {"adapters": []}}, "adapters must be a mapping"),
        (
            {"blacklist": {"adapters": {"embedder": ["voyage", "voyage"]}}},
            "duplicate adapters",
        ),
        (
            {"blacklist": {"adapters": {"embedder": [""]}}},
            "entries must be non-empty strings",
        ),
    ],
)
def test_invalid_experiment_settings_are_rejected(tmp_path, experiments, match):
    _write_config(tmp_path, "experiments:\n" + textwrap.indent(yaml.safe_dump(experiments), "  "))

    with pytest.raises(ConfigError, match=match):
        load_config()


def test_experiment_introspection_redacts_postgres_dsn(tmp_path):
    dsn = "postgresql://admin:do-not-leak@localhost/stropha_bench"
    _write_config(
        tmp_path,
        f"""
        experiments:
          storage: postgres
          postgres:
            admin_dsn: {dsn}
            database_prefix: local_bench
        """,
    )

    info = get_config_info()

    assert info["settings"]["experiments"]["postgres"] == {
        "admin_dsn": "(set)",
        "database_prefix": "local_bench",
        "template_database": "template1",
    }
    assert info["experiments"] == info["settings"]["experiments"]
    assert dsn not in repr(info)
    assert "do-not-leak" not in repr(info)


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
        "index_path",
        "target_repo",
        "log_level",
        "max_file_bytes",
        "use_voyage",
        "local_embed_model",
        "voyage_embed_model",
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
    _write_config(
        tmp_path,
        """
        embedder:
          voyage_model: voyage-code-3
          voyage_dim: 1024
          local_model: BAAI/bge-small-en-v1.5
    """,
    )
    cfg = load_config()
    assert cfg.embedder.voyage_model == "voyage-code-3"
    assert cfg.embedder.voyage_dim == 1024
    assert cfg.embedder.local_model == "BAAI/bge-small-en-v1.5"


def test_logging_from_yaml(tmp_path):
    """Log level is read from YAML logging.level."""
    _write_config(
        tmp_path,
        """
        logging:
          level: DEBUG
    """,
    )
    cfg = load_config()
    assert cfg.log_level == "DEBUG"


def test_mcp_from_yaml(tmp_path):
    """MCP watch settings are read from YAML."""
    _write_config(
        tmp_path,
        """
        mcp:
          watch: false
          watch_interval: 5.0
    """,
    )
    cfg = load_config()
    assert cfg.mcp_watch is False
    assert cfg.mcp_watch_interval == 5.0
