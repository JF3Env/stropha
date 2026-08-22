"""Unified stropha configuration package.

Single source of truth: ``~/.stropha/config.yaml``.

All configuration lives in that one file — pipeline adapters, retrieval
features, inference, paths, secrets, embedder model selection, logging,
MCP settings, graph, git.  No ``.env`` file, no project-level override
file, no environment variables for settings.

Environment variables are reserved for process/runtime arguments: the target
repository, MCP launcher controls, session identity and tracing, hook escape
hatches, and opt-in local PostgreSQL container management.

The resolved ``Config`` dataclass (see ``models.py``) exposes nested sections
(``retrieval_features``, ``inference``, ``graph``, ``paths``, ``secrets``,
``embedder``, ``walker``, ``logging``, ``mcp``, ``git``, ``hook``) plus
flat back-compat accessors so legacy call sites in ``cli.py`` / ``server.py``
/ ``embeddings/factory.py`` keep working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..errors import ConfigError
from ..logging import get_logger
from .models import (
    MCP_PROFILES,
    Config,
    EmbedderConfig,
    ExperimentBlacklistConfig,
    ExperimentPostgresConfig,
    ExperimentsConfig,
    ExperimentStorage,
    GitConfig,
    GraphConfig,
    HookConfig,
    InferenceConfig,
    LoggingConfig,
    McpConfig,
    McpProfile,
    PathsConfig,
    RetrievalFeatures,
    SecretsConfig,
    UnifiedConfig,
    WalkerConfig,
    _default_index_path,
)

__all__ = [
    "MCP_PROFILES",
    # Introspection metadata (consumed by tests + MCP get_config tool)
    "_ENV_REGISTRY",
    # Models (re-exported)
    "Config",
    "EmbedderConfig",
    "ExperimentBlacklistConfig",
    "ExperimentPostgresConfig",
    "ExperimentStorage",
    "ExperimentsConfig",
    "GitConfig",
    "GraphConfig",
    "HookConfig",
    "InferenceConfig",
    "LoggingConfig",
    "McpConfig",
    "McpProfile",
    "PathsConfig",
    "RetrievalFeatures",
    "SecretsConfig",
    "UnifiedConfig",
    "WalkerConfig",
    # Loader API
    "get_config",
    "get_config_info",
    "load_config",
    "reset_config",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Environment variable registry
# ---------------------------------------------------------------------------
#
# Only the vars the runtime *still* reads from the environment are listed
# here.  Everything else moved to ~/.stropha/config.yaml.


@dataclass(frozen=True)
class _EnvVar:
    """A single configurable env var (introspection metadata)."""

    name: str
    group: str
    default: str | None
    description: str
    secret: bool = False


_ENV_REGISTRY: tuple[_EnvVar, ...] = (
    # --- runtime path argument (not configuration) --------------------------
    _EnvVar("STROPHA_TARGET_REPO", "core", "<cwd>", "Repo to index / serve; runtime argument"),
    # --- shared MCP launcher -------------------------------------------------
    _EnvVar("STROPHA_MCP_HOST", "mcp", "127.0.0.1", "Shared MCP daemon bind host"),
    _EnvVar("STROPHA_MCP_PORT", "mcp", "9741", "Shared MCP daemon bind port"),
    _EnvVar("STROPHA_MCP_PROJECT_DIR", "mcp", "<workspace root>", "Launcher working directory"),
    _EnvVar("STROPHA_MCP_PROBE_TIMEOUT", "mcp", "8", "MCP protocol probe timeout (s)"),
    # --- per-client session/runtime state -----------------------------------
    _EnvVar("STROPHA_CLIENT_ID", "session", None, "Stable client identity for session resume"),
    _EnvVar(
        "STROPHA_TRACES_DIR",
        "session",
        "<index parent>/meta/traces",
        "Runtime trace output directory",
    ),
    # --- post-commit hook subprocess overrides ------------------------------
    _EnvVar("STROPHA_HOOK_SKIP", "hook", None, "Bypass post-commit hook (rebases)"),
    _EnvVar("STROPHA_HOOK_TIMEOUT", "hook", "600", "Hook bg-process wall-clock (s)"),
    _EnvVar("STROPHA_HOOK_LOG", "hook", "~/.cache/stropha-hook.log", "Hook log path"),
    _EnvVar("STROPHA_HOOK_UV", "hook", None, "Override `uv` binary path"),
    _EnvVar("STROPHA_HOOK_GRAPHIFY", "hook", None, "Override `graphify` binary path"),
    _EnvVar("STROPHA_HOOK_PROJECT_DIR", "hook", None, "`uv run --directory` target"),
    _EnvVar("STROPHA_HOOK_META_ITERATIONS", "hook", None, "Override installed meta iterations"),
    _EnvVar("STROPHA_HOOK_NO_META", "hook", None, "Skip hook meta-harness step"),
    _EnvVar("STROPHA_HOOK_NO_EVAL", "hook", None, "Skip hook evaluation step"),
    # --- opt-in launcher-managed PostgreSQL container -----------------------
    _EnvVar("STROPHA_USE_POSTGRES", "postgres-launcher", "0", "Manage local PostgreSQL container"),
    _EnvVar("STROPHA_PG_CONTAINER", "postgres-launcher", "stropha-pg", "Container name"),
    _EnvVar("STROPHA_PG_IMAGE", "postgres-launcher", "pgvector/pgvector:pg17", "Container image"),
    _EnvVar("STROPHA_PG_PORT", "postgres-launcher", "<YAML DSN or 54329>", "Published port"),
    _EnvVar("STROPHA_PG_USER", "postgres-launcher", "<YAML DSN or stropha>", "Database user"),
    _EnvVar(
        "STROPHA_PG_PASSWORD",
        "postgres-launcher",
        "<YAML DSN or stropha>",
        "Database password",
        secret=True,
    ),
    _EnvVar("STROPHA_PG_DB", "postgres-launcher", "<YAML DSN or stropha>", "Database name"),
    _EnvVar("STROPHA_PG_VOLUME", "postgres-launcher", "stropha-pgdata", "Data volume name"),
    # STROPHA_HOOK_INDEX_PATH removed in v=9 (see HookConfig docstring) —
    # was wired to cfg.hook.index_path but never propagated to the
    # storage adapter. Edit ~/.stropha/config.yaml `paths.index` instead.
)


def _registry_names() -> tuple[str, ...]:
    return tuple(v.name for v in _ENV_REGISTRY)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _int(val: Any, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _int_optional(val: Any) -> int | None:
    """Parse an optional integer YAML field.

    Returns ``None`` when the key is absent or unparseable — sentinel that
    callers (e.g. ``InferenceConfig.effective_batch_size``) interpret as
    "fall back to the related global default".
    """
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# YAML loading — single file only
# ---------------------------------------------------------------------------


def _load_yaml() -> dict[str, Any]:
    """Load ``~/.stropha/config.yaml`` — the one and only config file."""
    user_yaml = Path.home() / ".stropha" / "config.yaml"
    if not user_yaml.is_file():
        log.warning("config.missing_yaml", path=str(user_yaml))
        return {}
    try:
        raw = yaml.safe_load(user_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            log.warning(
                "config.invalid_yaml", path=str(user_yaml), error="top-level must be a mapping"
            )
            return {}
        log.debug("config.loaded", path=str(user_yaml))
        return raw
    except yaml.YAMLError as exc:
        log.warning("config.invalid_yaml", path=str(user_yaml), error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# Section parsers — all read from YAML only
# ---------------------------------------------------------------------------


def _parse_retrieval_features(raw: dict[str, Any]) -> RetrievalFeatures:
    return RetrievalFeatures(
        hyde_enabled=_bool(raw.get("hyde_enabled"), False),
        hyde_code_tuned=_bool(raw.get("hyde_code_tuned"), True),
        query_decomposition_enabled=_bool(raw.get("query_decomposition_enabled"), False),
        query_decomposition_llm=_bool(raw.get("query_decomposition_llm"), False),
        query_router_llm=_bool(raw.get("query_router_llm"), False),
        query_rewrite_enabled=_bool(raw.get("query_rewrite_enabled"), False),
        query_rewrite_max_terms=_int(raw.get("query_rewrite_max_terms"), 6),
        multi_query_enabled=_bool(raw.get("multi_query_enabled"), False),
        multi_query_count=_int(raw.get("multi_query_count"), 3),
        multi_query_min_terms=_int(raw.get("multi_query_min_terms"), 7),
        recursive_enabled=_bool(raw.get("recursive_enabled"), False),
        recursive_adjacency=_int(raw.get("recursive_adjacency"), 5),
        diversity_enabled=_bool(raw.get("diversity_enabled"), False),
        diversity_max_chunks_per_file=_int(raw.get("diversity_max_chunks_per_file"), 2),
        cache_enabled=_bool(raw.get("cache_enabled"), False),
        cache_size=_int(raw.get("cache_size"), 500),
        cache_ttl=_float(raw.get("cache_ttl"), 3600.0),
        adaptive_weighting=_bool(raw.get("adaptive_weighting"), True),
    )


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    return InferenceConfig(
        backend=raw.get("backend"),
        mlx_model=str(raw.get("mlx_model", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")),
        ollama_model=str(raw.get("ollama_model", "qwen2.5-coder:1.5b")),
        ollama_host=str(raw.get("ollama_host", "http://localhost:11434")),
        deepseek_model=str(raw.get("deepseek_model", "deepseek-chat")),
        deepseek_base_url=str(raw.get("deepseek_base_url", "https://api.deepseek.com/v1")),
        deepseek_timeout=_float(raw.get("deepseek_timeout"), 120.0),
        deepseek_max_retries=_int(raw.get("deepseek_max_retries"), 2),
        lms_model=str(raw.get("lms_model", "Qwen/Qwen3-Coder-Next-FP8")),
        lms_base_url=str(raw.get("lms_base_url", "https://lms.example.com/openai")),
        lms_timeout=_float(raw.get("lms_timeout"), 180.0),
        lms_max_retries=_int(raw.get("lms_max_retries"), 2),
        lms_think=_bool(raw.get("lms_think"), False),
        lms_max_connections=_int(raw.get("lms_max_connections"), 8),
        lms_max_keepalive_connections=_int(raw.get("lms_max_keepalive_connections"), 4),
        moa_model=str(raw.get("moa_model", "moa-code")),
        moa_base_url=str(raw.get("moa_base_url", "http://127.0.0.1:14598/v1")),
        moa_timeout=_float(raw.get("moa_timeout"), 300.0),
        moa_max_retries=_int(raw.get("moa_max_retries"), 1),
        anthropic_model=str(raw.get("anthropic_model", "claude-haiku-4-5")),
        anthropic_base_url=str(raw.get("anthropic_base_url", "https://api.anthropic.com/v1")),
        anthropic_timeout=_float(raw.get("anthropic_timeout"), 60.0),
        anthropic_max_retries=_int(raw.get("anthropic_max_retries"), 2),
        anthropic_version=str(raw.get("anthropic_version", "2023-06-01")),
        enricher_concurrency=_int(raw.get("enricher_concurrency"), 1),
        enricher_batch_size=_int(raw.get("enricher_batch_size"), 1),
        deepseek_batch_size=_int_optional(raw.get("deepseek_batch_size")),
        lms_batch_size=_int_optional(raw.get("lms_batch_size")),
        moa_batch_size=_int_optional(raw.get("moa_batch_size")),
        anthropic_batch_size=_int_optional(raw.get("anthropic_batch_size")),
        ollama_batch_size=_int_optional(raw.get("ollama_batch_size")),
        mlx_batch_size=_int_optional(raw.get("mlx_batch_size")),
    )


def _parse_graph(raw: dict[str, Any]) -> GraphConfig:
    graphify_out_raw = raw.get("graphify_out")
    graphify_out = Path(graphify_out_raw).expanduser() if graphify_out_raw else None
    return GraphConfig(
        confidence=str(raw.get("confidence", "EXTRACTED")),
        fts_augment=_bool(raw.get("fts_augment"), True),
        graphify_out=graphify_out,
    )


def _parse_paths(raw: dict[str, Any]) -> PathsConfig:
    # STROPHA_TARGET_REPO stays as a runtime env var — it's an argument
    # (which repo to index), not a configuration setting.
    target_env = os.environ.get("STROPHA_TARGET_REPO")
    target_repo = Path(target_env).expanduser() if target_env else Path.cwd()

    # Index path comes from YAML paths.index; fall back to built-in default.
    index_raw = raw.get("index")
    index_path = Path(index_raw).expanduser() if index_raw else _default_index_path()
    return PathsConfig(target_repo=target_repo, index_path=index_path)


def _parse_secrets(raw: dict[str, Any]) -> SecretsConfig:
    return SecretsConfig(
        voyage_api_key=raw.get("voyage_api_key") or None,
        anthropic_api_key=raw.get("anthropic_api_key") or None,
        deepseek_api_key=raw.get("deepseek_api_key") or None,
        lms_api_key=(raw.get("lms_api_key") if isinstance(raw.get("lms_api_key"), str) else None),
        moa_api_key=(raw.get("moa_api_key") if isinstance(raw.get("moa_api_key"), str) else None),
    )


def _parse_embedder(raw: dict[str, Any]) -> EmbedderConfig:
    return EmbedderConfig(
        voyage_model=str(raw.get("voyage_model", "voyage-4-large")),
        voyage_dim=_int(raw.get("voyage_dim"), 512),
        local_model=str(raw.get("local_model", "mixedbread-ai/mxbai-embed-large-v1")),
    )


def _parse_walker(raw: dict[str, Any]) -> WalkerConfig:
    return WalkerConfig(
        max_file_bytes=_int(raw.get("max_file_bytes"), 524_288),
    )


def _parse_logging(raw: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(level=str(raw.get("level", "INFO")))


def _parse_mcp(raw: dict[str, Any]) -> McpConfig:
    return McpConfig(
        profile=raw.get("profile", "full"),
        watch=_bool(raw.get("watch"), True),
        watch_interval=_float(raw.get("watch_interval"), 1.0),
        watch_debounce=_float(raw.get("watch_debounce"), 2.0),
    )


def _parse_git(raw: dict[str, Any]) -> GitConfig:
    cache_raw = raw.get("url_cache")
    cache_path = (
        Path(cache_raw).expanduser() if cache_raw else Path.home() / ".cache" / "stropha" / "repos"
    )
    return GitConfig(
        url_cache=cache_path,
        url_ttl=_int(raw.get("url_ttl"), 86_400),
        clone_depth=_int(raw.get("clone_depth"), 1),
        clone_timeout=_int(raw.get("clone_timeout"), 300),
        rename_threshold=_int(raw.get("rename_threshold"), 80),
    )


def _parse_experiments(raw: Any) -> ExperimentsConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("experiments must be a mapping")

    postgres_raw = raw.get("postgres")
    if postgres_raw is None:
        postgres_raw = {}
    if not isinstance(postgres_raw, dict):
        raise ConfigError("experiments.postgres must be a mapping")

    blacklist_raw = raw.get("blacklist")
    if blacklist_raw is None:
        blacklist_raw = {}
    if not isinstance(blacklist_raw, dict):
        raise ConfigError("experiments.blacklist must be a mapping")
    adapters_raw = blacklist_raw.get("adapters")
    if adapters_raw is None:
        adapters_raw = {}

    return ExperimentsConfig(
        storage=raw.get("storage", "sqlite-vec"),
        postgres=ExperimentPostgresConfig(
            admin_dsn=postgres_raw.get("admin_dsn"),
            database_prefix=postgres_raw.get("database_prefix", "stropha_bench"),
            template_database=postgres_raw.get("template_database", "template1"),
        ),
        blacklist=ExperimentBlacklistConfig(adapters=adapters_raw),
    )


def _parse_hook() -> HookConfig:
    """Hook config is read from env only — the hook is a detached subprocess."""
    log_env = os.environ.get("STROPHA_HOOK_LOG")
    log_path = (
        Path(log_env).expanduser() if log_env else Path.home() / ".cache" / "stropha-hook.log"
    )
    proj_env = os.environ.get("STROPHA_HOOK_PROJECT_DIR")
    return HookConfig(
        log_path=log_path,
        timeout=_int(os.environ.get("STROPHA_HOOK_TIMEOUT"), 600),
        uv_path=os.environ.get("STROPHA_HOOK_UV"),
        graphify_path=os.environ.get("STROPHA_HOOK_GRAPHIFY"),
        skip=os.environ.get("STROPHA_HOOK_SKIP") is not None,
        project_dir=Path(proj_env).expanduser() if proj_env else None,
    )


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


_config_instance: Config | None = None


def load_config(project_root: Path | None = None, force_reload: bool = False) -> Config:
    """Load unified configuration from ``~/.stropha/config.yaml``. Cached.

    Args:
        project_root: Ignored — kept for call-site compatibility only.
                      All config comes from ``~/.stropha/config.yaml``.
        force_reload: Bypass cache and reload from disk.

    Returns:
        Resolved :class:`Config` dataclass.
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    raw = _load_yaml()

    _config_instance = Config(
        retrieval_features=_parse_retrieval_features(raw.get("retrieval_features", {})),
        inference=_parse_inference(raw.get("inference", {})),
        graph=_parse_graph(raw.get("graph", {})),
        paths=_parse_paths(raw.get("paths", {})),
        secrets=_parse_secrets(raw.get("secrets", {})),
        embedder=_parse_embedder(raw.get("embedder", {})),
        walker=_parse_walker(raw.get("walker", {})),
        logging=_parse_logging(raw.get("logging", {})),
        mcp=_parse_mcp(raw.get("mcp", {})),
        git=_parse_git(raw.get("git", {})),
        hook=_parse_hook(),
        experiments=_parse_experiments(raw.get("experiments", {})),
    )

    log.debug(
        "config.loaded",
        index=str(_config_instance.paths.index_path),
        target=str(_config_instance.paths.target_repo),
        hyde=_config_instance.retrieval_features.hyde_enabled,
        inference_backend=_config_instance.inference.backend,
    )

    return _config_instance


def get_config(*, reload: bool = False) -> Config:
    """Return the singleton :class:`Config` instance, loading on first call."""
    return load_config(force_reload=reload)


def reset_config() -> None:
    """Drop the cached Config (forces reload on next call). For tests."""
    global _config_instance
    _config_instance = None


# ---------------------------------------------------------------------------
# Introspection (backs the MCP `get_config` tool)
# ---------------------------------------------------------------------------


def get_config_info() -> dict[str, Any]:
    """Return a debug snapshot of the resolved configuration.

    Used by the MCP ``get_config`` tool. The returned dict is JSON-safe:
    paths are stringified and secrets are redacted to ``(set)``.
    """
    cfg = get_config()

    # Only registered process/runtime arguments are represented here.
    sources: dict[str, str] = {
        var.name: ("env" if os.environ.get(var.name) is not None else "default")
        for var in _ENV_REGISTRY
    }
    env_values: dict[str, str | None] = {
        var.name: (
            "(set)"
            if var.secret and os.environ.get(var.name)
            else os.environ.get(var.name, var.default)
        )
        for var in _ENV_REGISTRY
    }

    env_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for var in _ENV_REGISTRY:
        env_groups.setdefault(var.group, {})[var.name] = {
            "value": env_values[var.name],
            "default": var.default,
            "source": sources[var.name],
            "description": var.description,
            "secret": var.secret,
        }

    # Secrets: redact when set, null when absent.
    voyage_set = bool(cfg.secrets.voyage_api_key)
    anthropic_set = bool(cfg.secrets.anthropic_api_key)
    deepseek_set = bool(cfg.secrets.deepseek_api_key)
    lms_set = bool(cfg.secrets.lms_api_key)
    moa_set = bool(cfg.secrets.moa_api_key)

    experiment_summary = {
        "storage": cfg.experiments.storage,
        "postgres": {
            "admin_dsn": "(set)" if cfg.experiments.postgres.admin_dsn else None,
            "database_prefix": cfg.experiments.postgres.database_prefix,
            "template_database": cfg.experiments.postgres.template_database,
        },
        "blacklist": {
            "adapters": {
                stage: list(adapters)
                for stage, adapters in cfg.experiments.blacklist.adapters.items()
            }
        },
    }

    settings_dump: dict[str, Any] = {
        "voyage_api_key": "(set)" if voyage_set else None,
        "anthropic_api_key": "(set)" if anthropic_set else None,
        "deepseek_api_key": "(set)" if deepseek_set else None,
        "lms_api_key": "(set)" if lms_set else None,
        "moa_api_key": "(set)" if moa_set else None,
        "target_repo": str(cfg.target_repo),
        "index_path": str(cfg.index_path),
        "voyage_embed_model": cfg.voyage_embed_model,
        "voyage_embed_dim": cfg.voyage_embed_dim,
        "local_embed_model": cfg.local_embed_model,
        "log_level": cfg.log_level,
        "max_file_bytes": cfg.max_file_bytes,
        "mcp_watch": cfg.mcp_watch,
        "mcp_profile": cfg.mcp.profile,
        "mcp_watch_interval": cfg.mcp_watch_interval,
        "mcp_watch_debounce": cfg.mcp_watch_debounce,
        "use_voyage": cfg.use_voyage,
        "experiments": experiment_summary,
    }

    return {
        # Legacy "config" block — back-compat for callers that read it directly.
        "config": {
            "index_path": str(cfg.resolve_index_path()),
            "target_repo": str(cfg.target_repo),
            "log_level": cfg.log_level,
            "max_file_bytes": cfg.max_file_bytes,
            "use_voyage": cfg.use_voyage,
            "local_embed_model": cfg.local_embed_model,
            "voyage_embed_model": cfg.voyage_embed_model,
            "voyage_embed_dim": cfg.voyage_embed_dim,
        },
        "settings": settings_dump,
        "experiments": experiment_summary,
        "sources": sources,
        "env_values": env_values,
        "env_groups": env_groups,
        "runtime": {
            "cwd": str(Path.cwd()),
            "index_exists": cfg.resolve_index_path().exists(),
            "target_exists": cfg.target_repo.exists(),
        },
    }
