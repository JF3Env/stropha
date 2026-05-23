"""Unified stropha configuration package.

This is the single source of truth for runtime configuration. It loads from
two layers:

1. **YAML** (project-level ``./stropha.yaml`` overlaid on user
   ``~/.stropha/config.yaml``) — for query-time features, inference, graph,
   pipeline-stage adapters.
2. **Environment variables** — for paths, secrets, hook subprocess settings,
   embedder model overrides, MCP watcher knobs, and other process-level
   plumbing that must work in subprocess contexts (the hook can't read YAML).

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

import yaml

from ..logging import get_logger
from .models import (
    Config,
    EmbedderConfig,
    GitConfig,
    GraphConfig,
    HookConfig,
    InferenceConfig,
    LoggingConfig,
    McpConfig,
    PathsConfig,
    RetrievalFeatures,
    SecretsConfig,
    UnifiedConfig,
    WalkerConfig,
    _default_index_path,
)

__all__ = [
    # Models (re-exported)
    "Config",
    "EmbedderConfig",
    "GitConfig",
    "GraphConfig",
    "HookConfig",
    "InferenceConfig",
    "LoggingConfig",
    "McpConfig",
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
    # Introspection metadata (consumed by tests + MCP get_config tool)
    "_ENV_REGISTRY",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Environment variable registry
# ---------------------------------------------------------------------------
#
# Every env var the runtime consumes (or used to consume) is declared here so
# the MCP ``get_config`` tool, CLI debug output, and §2.5 of CLAUDE.md stay in
# lockstep. Vars marked ``[deprecated]`` are still listed for documentation
# but their behaviour now lives in ``stropha.yaml``.


@dataclass(frozen=True)
class _EnvVar:
    """A single configurable env var (introspection metadata)."""

    name: str
    group: str
    default: str | None
    description: str
    secret: bool = False


_ENV_REGISTRY: tuple[_EnvVar, ...] = (
    # --- paths / target -----------------------------------------------------
    _EnvVar("STROPHA_INDEX_PATH", "core", ".stropha/index.db", "SQLite index path"),
    _EnvVar("STROPHA_TARGET_REPO", "core", "<cwd>", "Default repo for index / MCP server"),
    _EnvVar("STROPHA_LOG_LEVEL", "core", "INFO", "structlog level"),
    _EnvVar("STROPHA_MAX_FILE_BYTES", "core", "524288", "Max file size walked (bytes)"),
    # --- embedders ----------------------------------------------------------
    _EnvVar(
        "STROPHA_LOCAL_EMBED_MODEL",
        "embedder",
        "mixedbread-ai/mxbai-embed-large-v1",
        "Local fastembed model id",
    ),
    _EnvVar("STROPHA_VOYAGE_EMBED_MODEL", "embedder", "voyage-4-large", "Voyage embedder model id"),
    _EnvVar("STROPHA_VOYAGE_EMBED_DIM", "embedder", "512", "Voyage embedding dim (Matryoshka)"),
    _EnvVar("VOYAGE_API_KEY", "embedder", None, "Voyage AI API key", secret=True),
    # --- inference backends -------------------------------------------------
    _EnvVar(
        "STROPHA_INFERENCE_BACKEND",
        "inference",
        None,
        "Preferred LLM backend (ollama/mlx/deepseek/anthropic); YAML inference.backend is canonical",
    ),
    _EnvVar(
        "ANTHROPIC_API_KEY",
        "inference",
        None,
        "Anthropic API key (required when inference.backend=anthropic)",
        secret=True,
    ),
    _EnvVar(
        "STROPHA_OLLAMA_MODEL",
        "inference",
        "qwen2.5-coder:1.5b",
        "[deprecated] use stropha.yaml inference.ollama_model",
    ),
    _EnvVar("OLLAMA_HOST", "inference", "http://localhost:11434", "Ollama HTTP endpoint"),
    _EnvVar(
        "DEEPSEEK_API_KEY",
        "inference",
        None,
        "DeepSeek API key (required when inference.backend=deepseek)",
        secret=True,
    ),
    _EnvVar(
        "DEEPSEEK_BASE_URL",
        "inference",
        "https://api.deepseek.com/v1",
        "DeepSeek API base URL (override for proxies / self-hosted gateways)",
    ),
    # --- query intelligence (deprecated — YAML retrieval_features.* canonical)
    _EnvVar(
        "STROPHA_HYDE_ENABLED",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.hyde_enabled",
    ),
    _EnvVar(
        "STROPHA_HYDE_CODE_TUNED",
        "query",
        "1",
        "[deprecated] use stropha.yaml retrieval_features.hyde_code_tuned",
    ),
    _EnvVar("STROPHA_HYDE_PROMPT", "query", None, "Custom HyDE prompt template"),
    _EnvVar(
        "STROPHA_QUERY_ROUTER_LLM",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.query_router_llm",
    ),
    _EnvVar(
        "STROPHA_QUERY_DECOMPOSITION_ENABLED",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.query_decomposition_enabled",
    ),
    _EnvVar(
        "STROPHA_QUERY_DECOMPOSITION_LLM",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.query_decomposition_llm",
    ),
    _EnvVar(
        "STROPHA_QUERY_REWRITE_ENABLED",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.query_rewrite_enabled",
    ),
    _EnvVar("STROPHA_QUERY_REWRITE_PROMPT", "query", None, "Custom rewrite prompt template"),
    _EnvVar(
        "STROPHA_MULTI_QUERY_ENABLED",
        "query",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.multi_query_enabled",
    ),
    _EnvVar(
        "STROPHA_MULTI_QUERY_COUNT",
        "query",
        "3",
        "[deprecated] use stropha.yaml retrieval_features.multi_query_count",
    ),
    _EnvVar("STROPHA_MULTI_QUERY_PROMPT", "query", None, "Custom paraphrase prompt template"),
    _EnvVar("STROPHA_ADAPTIVE_WEIGHTING", "query", "1", "Adaptive symbol/natural query weighting"),
    # --- query cache (deprecated — YAML retrieval_features.* canonical) ----
    _EnvVar(
        "STROPHA_QUERY_CACHE_ENABLED",
        "cache",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.cache_enabled",
    ),
    _EnvVar(
        "STROPHA_QUERY_CACHE_SIZE",
        "cache",
        "500",
        "[deprecated] use stropha.yaml retrieval_features.cache_size",
    ),
    _EnvVar(
        "STROPHA_QUERY_CACHE_TTL",
        "cache",
        "3600",
        "[deprecated] use stropha.yaml retrieval_features.cache_ttl",
    ),
    # --- retrieval behaviour -----------------------------------------------
    _EnvVar(
        "STROPHA_RECURSIVE_RETRIEVAL",
        "retrieval",
        "0",
        "[deprecated] use stropha.yaml retrieval_features.recursive_enabled",
    ),
    _EnvVar(
        "STROPHA_RECURSIVE_ADJACENCY",
        "retrieval",
        "5",
        "[deprecated] use stropha.yaml retrieval_features.recursive_adjacency",
    ),
    _EnvVar(
        "STROPHA_GRAPH_FTS_AUGMENT",
        "retrieval",
        "1",
        "[deprecated] use stropha.yaml graph.fts_augment",
    ),
    _EnvVar(
        "STROPHA_GRAPH_CONFIDENCE",
        "retrieval",
        "EXTRACTED",
        "[deprecated] use stropha.yaml graph.confidence",
    ),
    _EnvVar("STROPHA_GRAPHIFY_OUT", "retrieval", "<repo>/graphify-out", "Override graphify output path"),
    # --- MCP server watcher ------------------------------------------------
    _EnvVar("STROPHA_MCP_WATCH", "mcp", "1", "Auto-reindex file watcher in MCP server"),
    _EnvVar("STROPHA_MCP_WATCH_INTERVAL", "mcp", "1.0", "Watcher polling interval (s)"),
    _EnvVar("STROPHA_MCP_WATCH_DEBOUNCE", "mcp", "2.0", "Watcher debounce window (s)"),
    # --- git url cache / walker --------------------------------------------
    _EnvVar("STROPHA_GIT_URL_CACHE", "git", "~/.cache/stropha/repos", "Cache dir for cloned remote repos"),
    _EnvVar("STROPHA_GIT_URL_TTL", "git", "86400", "Cache TTL for remote repos (s)"),
    _EnvVar("STROPHA_RENAME_THRESHOLD", "git", "80", "Git rename detection similarity (0-100)"),
    # --- post-commit hook --------------------------------------------------
    _EnvVar("STROPHA_HOOK_SKIP", "hook", None, "Bypass post-commit hook (rebases)"),
    _EnvVar("STROPHA_HOOK_TIMEOUT", "hook", "600", "Hook bg-process wall-clock (s)"),
    _EnvVar("STROPHA_HOOK_LOG", "hook", "~/.cache/stropha-hook.log", "Hook log path"),
    _EnvVar("STROPHA_HOOK_UV", "hook", None, "Override `uv` binary path"),
    _EnvVar("STROPHA_HOOK_GRAPHIFY", "hook", None, "Override `graphify` binary path"),
    _EnvVar("STROPHA_HOOK_NO_GRAPHIFY", "hook", None, "Skip graphify step in hook"),
    _EnvVar("STROPHA_HOOK_PROJECT_DIR", "hook", None, "`uv run --directory` target"),
    _EnvVar("STROPHA_HOOK_INDEX_PATH", "hook", None, "Per-repo STROPHA_INDEX_PATH for the hook"),
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


def _float(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """In-place deep merge: ``src`` keys win over ``dst`` for scalars; dicts recurse."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _load_yaml(project_root: Path | None) -> dict[str, Any]:
    raw: dict[str, Any] = {}

    # 1. User config (~/.stropha/config.yaml)
    user_yaml = Path.home() / ".stropha" / "config.yaml"
    if user_yaml.is_file():
        try:
            user_raw = yaml.safe_load(user_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(user_raw, dict):
                raw.update(user_raw)
                log.debug("config.loaded_user", path=str(user_yaml))
        except yaml.YAMLError as exc:
            log.warning("config.invalid_yaml", path=str(user_yaml), error=str(exc))

    # 2. Project config (./stropha.yaml) — overlays user config.
    project = project_root or Path.cwd()
    project_yaml = project / "stropha.yaml"
    if project_yaml.is_file():
        try:
            proj_raw = yaml.safe_load(project_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(proj_raw, dict):
                _deep_merge(raw, proj_raw)
                log.debug("config.loaded_project", path=str(project_yaml))
        except yaml.YAMLError as exc:
            log.warning("config.invalid_yaml", path=str(project_yaml), error=str(exc))

    return raw


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_retrieval_features(raw: dict[str, Any]) -> RetrievalFeatures:
    return RetrievalFeatures(
        hyde_enabled=_bool(raw.get("hyde_enabled"), False),
        hyde_code_tuned=_bool(raw.get("hyde_code_tuned"), True),
        query_decomposition_enabled=_bool(raw.get("query_decomposition_enabled"), False),
        query_decomposition_llm=_bool(raw.get("query_decomposition_llm"), False),
        query_router_llm=_bool(raw.get("query_router_llm"), False),
        query_rewrite_enabled=_bool(raw.get("query_rewrite_enabled"), False),
        multi_query_enabled=_bool(raw.get("multi_query_enabled"), False),
        multi_query_count=_int(raw.get("multi_query_count"), 3),
        recursive_enabled=_bool(raw.get("recursive_enabled"), False),
        recursive_adjacency=_int(raw.get("recursive_adjacency"), 5),
        cache_enabled=_bool(raw.get("cache_enabled"), False),
        cache_size=_int(raw.get("cache_size"), 500),
        cache_ttl=_float(raw.get("cache_ttl"), 3600.0),
        # Adaptive weighting is still ENV-driven (no YAML key today).
        adaptive_weighting=_bool_env("STROPHA_ADAPTIVE_WEIGHTING", True),
    )


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    # OLLAMA_HOST stays ENV-supported for ecosystem compatibility.
    ollama_host = raw.get("ollama_host") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )
    # DeepSeek base URL: YAML wins; env override for ecosystem compat.
    deepseek_base_url = raw.get("deepseek_base_url") or os.environ.get(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )
    # Anthropic base URL: YAML wins; env override for ecosystem compat
    # (proxies, self-hosted gateways).
    anthropic_base_url = raw.get("anthropic_base_url") or os.environ.get(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
    )
    return InferenceConfig(
        backend=raw.get("backend"),
        mlx_model=str(raw.get("mlx_model", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")),
        ollama_model=str(raw.get("ollama_model", "qwen2.5-coder:1.5b")),
        ollama_host=str(ollama_host),
        deepseek_model=str(raw.get("deepseek_model", "deepseek-chat")),
        deepseek_base_url=str(deepseek_base_url),
        deepseek_timeout=_float(raw.get("deepseek_timeout"), 120.0),
        deepseek_max_retries=_int(raw.get("deepseek_max_retries"), 2),
        anthropic_model=str(raw.get("anthropic_model", "claude-haiku-4-5")),
        anthropic_base_url=str(anthropic_base_url),
        anthropic_timeout=_float(raw.get("anthropic_timeout"), 60.0),
        anthropic_max_retries=_int(raw.get("anthropic_max_retries"), 2),
        anthropic_version=str(raw.get("anthropic_version", "2023-06-01")),
        enricher_concurrency=_int(raw.get("enricher_concurrency"), 1),
        enricher_batch_size=_int(raw.get("enricher_batch_size"), 1),
    )


def _parse_graph(raw: dict[str, Any]) -> GraphConfig:
    graphify_out_env = os.environ.get("STROPHA_GRAPHIFY_OUT")
    graphify_out = Path(graphify_out_env).expanduser() if graphify_out_env else None
    return GraphConfig(
        confidence=str(raw.get("confidence", "EXTRACTED")),
        fts_augment=_bool(raw.get("fts_augment"), True),
        graphify_out=graphify_out,
    )


def _parse_paths() -> PathsConfig:
    target_env = os.environ.get("STROPHA_TARGET_REPO")
    target_repo = Path(target_env).expanduser() if target_env else Path.cwd()
    index_env = os.environ.get("STROPHA_INDEX_PATH")
    index_path = Path(index_env).expanduser() if index_env else _default_index_path()
    return PathsConfig(target_repo=target_repo, index_path=index_path)


def _parse_secrets() -> SecretsConfig:
    return SecretsConfig(
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )


def _parse_embedder() -> EmbedderConfig:
    return EmbedderConfig(
        voyage_model=os.environ.get("STROPHA_VOYAGE_EMBED_MODEL", "voyage-4-large"),
        voyage_dim=_int(os.environ.get("STROPHA_VOYAGE_EMBED_DIM"), 512),
        local_model=os.environ.get(
            "STROPHA_LOCAL_EMBED_MODEL", "mixedbread-ai/mxbai-embed-large-v1"
        ),
    )


def _parse_walker() -> WalkerConfig:
    return WalkerConfig(
        max_file_bytes=_int(os.environ.get("STROPHA_MAX_FILE_BYTES"), 524_288),
    )


def _parse_logging() -> LoggingConfig:
    return LoggingConfig(level=os.environ.get("STROPHA_LOG_LEVEL", "INFO"))


def _parse_mcp() -> McpConfig:
    return McpConfig(
        watch=_bool_env("STROPHA_MCP_WATCH", True),
        watch_interval=_float(os.environ.get("STROPHA_MCP_WATCH_INTERVAL"), 1.0),
        watch_debounce=_float(os.environ.get("STROPHA_MCP_WATCH_DEBOUNCE"), 2.0),
    )


def _parse_git() -> GitConfig:
    cache_env = os.environ.get("STROPHA_GIT_URL_CACHE")
    cache_path = (
        Path(cache_env).expanduser()
        if cache_env
        else Path.home() / ".cache" / "stropha" / "repos"
    )
    return GitConfig(
        url_cache=cache_path,
        url_ttl=_int(os.environ.get("STROPHA_GIT_URL_TTL"), 86_400),
        rename_threshold=_int(os.environ.get("STROPHA_RENAME_THRESHOLD"), 80),
    )


def _parse_hook() -> HookConfig:
    log_env = os.environ.get("STROPHA_HOOK_LOG")
    log_path = (
        Path(log_env).expanduser()
        if log_env
        else Path.home() / ".cache" / "stropha-hook.log"
    )
    proj_env = os.environ.get("STROPHA_HOOK_PROJECT_DIR")
    idx_env = os.environ.get("STROPHA_HOOK_INDEX_PATH")
    return HookConfig(
        log_path=log_path,
        timeout=_int(os.environ.get("STROPHA_HOOK_TIMEOUT"), 600),
        uv_path=os.environ.get("STROPHA_HOOK_UV"),
        graphify_path=os.environ.get("STROPHA_HOOK_GRAPHIFY"),
        skip=os.environ.get("STROPHA_HOOK_SKIP") is not None,
        no_graphify=os.environ.get("STROPHA_HOOK_NO_GRAPHIFY") is not None,
        project_dir=Path(proj_env).expanduser() if proj_env else None,
        index_path=Path(idx_env).expanduser() if idx_env else None,
    )


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


_config_instance: Config | None = None
# Pre-`.env`-load snapshot. Captures every var in the registry so we can
# distinguish "set in process env" from "set via .env file" from "default".
_config_env_snapshot: dict[str, str | None] = {}


def load_config(
    project_root: Path | None = None, force_reload: bool = False
) -> Config:
    """Load unified configuration from YAML + ENV. Cached.

    Args:
        project_root: Directory containing ``stropha.yaml``. Defaults to cwd.
        force_reload: Bypass cache and reload from disk + env.

    Returns:
        Resolved :class:`Config` dataclass.
    """
    global _config_instance, _config_env_snapshot

    if _config_instance is not None and not force_reload:
        return _config_instance

    # Snapshot every registered env var BEFORE loading .env so source
    # detection can tell process-env from dotenv from default.
    _config_env_snapshot = {name: os.environ.get(name) for name in _registry_names()}

    # Only load .env when STROPHA_INDEX_PATH isn't already in the process env.
    # MCP clients (opencode) set STROPHA_INDEX_PATH explicitly, so this guard
    # prevents the stropha project's .env from clobbering caller-provided values.
    if not os.environ.get("STROPHA_INDEX_PATH"):
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass

    raw = _load_yaml(project_root)

    _config_instance = Config(
        retrieval_features=_parse_retrieval_features(raw.get("retrieval_features", {})),
        inference=_parse_inference(raw.get("inference", {})),
        graph=_parse_graph(raw.get("graph", {})),
        paths=_parse_paths(),
        secrets=_parse_secrets(),
        embedder=_parse_embedder(),
        walker=_parse_walker(),
        logging=_parse_logging(),
        mcp=_parse_mcp(),
        git=_parse_git(),
        hook=_parse_hook(),
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
    global _config_instance, _config_env_snapshot
    _config_instance = None
    _config_env_snapshot = {}


# ---------------------------------------------------------------------------
# Introspection (backs the MCP `get_config` tool)
# ---------------------------------------------------------------------------


def _source_of(name: str) -> str:
    """Tell where an env var's value came from.

    - ``env``: set in the process env before stropha started.
    - ``dotenv``: appeared in ``os.environ`` only after stropha loaded ``.env``.
    - ``default``: never set; the registry default applies.
    """
    snapshot_val = _config_env_snapshot.get(name)
    if snapshot_val is not None:
        return "env"
    if os.environ.get(name) is not None:
        return "dotenv"
    return "default"


def _effective_value(var: _EnvVar) -> str | None:
    """The value the runtime will actually observe for ``var``.

    Returns ``"(set)"`` for secrets when present (never the raw value).
    """
    raw = os.environ.get(var.name)
    if raw is None:
        return var.default
    if var.secret:
        return "(set)"
    return raw


def get_config_info() -> dict[str, Any]:
    """Return a debug snapshot of the resolved configuration.

    Used by the MCP ``get_config`` tool. The returned dict is JSON-safe:
    paths are stringified and secrets are redacted to ``(set)``.
    """
    cfg = get_config()

    sources: dict[str, str] = {var.name: _source_of(var.name) for var in _ENV_REGISTRY}
    env_values: dict[str, str | None] = {
        var.name: _effective_value(var) for var in _ENV_REGISTRY
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

    # Flattened back-compat settings dump (mirrors the legacy pydantic Config
    # fields). Secrets redacted; Paths stringified.
    settings_dump: dict[str, Any] = {
        "voyage_api_key": "(set)" if cfg.secrets.voyage_api_key else None,
        "anthropic_api_key": "(set)" if cfg.secrets.anthropic_api_key else None,
        "target_repo": str(cfg.target_repo),
        "index_path": str(cfg.index_path),
        "voyage_embed_model": cfg.voyage_embed_model,
        "voyage_embed_dim": cfg.voyage_embed_dim,
        "local_embed_model": cfg.local_embed_model,
        "log_level": cfg.log_level,
        "max_file_bytes": cfg.max_file_bytes,
        "mcp_watch": cfg.mcp_watch,
        "mcp_watch_interval": cfg.mcp_watch_interval,
        "mcp_watch_debounce": cfg.mcp_watch_debounce,
        "use_voyage": cfg.use_voyage,
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
        "sources": sources,
        "env_values": env_values,
        "env_groups": env_groups,
        "runtime": {
            "cwd": str(Path.cwd()),
            "index_exists": cfg.resolve_index_path().exists(),
            "target_exists": cfg.target_repo.exists(),
        },
    }
