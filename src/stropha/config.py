"""Configuration loaded from environment / .env via pydantic-settings.

This module provides a centralized configuration singleton that all parts of
the system should use. The configuration is loaded once and cached, ensuring
consistent behavior across CLI, MCP server, and pipeline components.

Usage:
    from stropha.config import get_config
    cfg = get_config()  # Returns the singleton instance

    # For debugging / MCP tool:
    from stropha.config import get_config_info
    info = get_config_info()  # Returns dict with config + source info
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_index_path() -> Path:
    """Default index path: .stropha/index.db relative to cwd, or ~/.stropha/index.db."""
    local = Path.cwd() / ".stropha" / "index.db"
    if local.parent.exists():
        return local
    return Path.home() / ".stropha" / "index.db"


def _default_target_repo() -> Path:
    """Default target = current working directory."""
    return Path.cwd()


class Config(BaseSettings):
    """Runtime configuration. All product env vars are prefixed `STROPHA_`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # API keys (provider names, no STROPHA_ prefix).
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Paths.
    target_repo: Path = Field(default_factory=_default_target_repo, alias="STROPHA_TARGET_REPO")
    index_path: Path = Field(default_factory=_default_index_path, alias="STROPHA_INDEX_PATH")

    # Models.
    voyage_embed_model: str = Field(
        default="voyage-code-3", alias="STROPHA_VOYAGE_EMBED_MODEL"
    )
    voyage_embed_dim: int = Field(default=512, alias="STROPHA_VOYAGE_EMBED_DIM")
    # Default per ADR-008 (docs/architecture/stropha-graphify-integration.md):
    # mxbai-embed-large-v1 — top open-source English MTEB at this scale,
    # 1024-dim, ~0.64 GB. Chosen over jina-embeddings-v2-base-code because
    # the latter has an ONNX-runtime instability on macOS aarch64 that
    # causes hang/crash on the second consecutive embed call.
    local_embed_model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1", alias="STROPHA_LOCAL_EMBED_MODEL"
    )

    # Behavior.
    log_level: str = Field(default="INFO", alias="STROPHA_LOG_LEVEL")
    max_file_bytes: int = Field(default=524_288, alias="STROPHA_MAX_FILE_BYTES")

    # MCP server watch integration (auto-reindex on file changes).
    mcp_watch: bool = Field(default=True, alias="STROPHA_MCP_WATCH")
    mcp_watch_interval: float = Field(default=1.0, alias="STROPHA_MCP_WATCH_INTERVAL")
    mcp_watch_debounce: float = Field(default=2.0, alias="STROPHA_MCP_WATCH_DEBOUNCE")

    @property
    def use_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    def resolve_index_path(self) -> Path:
        """Expand ~ and env vars in the configured index path."""
        return Path(os.path.expandvars(str(self.index_path))).expanduser()


# ---------------------------------------------------------------------------
# Environment variable registry
# ---------------------------------------------------------------------------
#
# Every env var the runtime consumes is declared here so the MCP ``get_config``
# tool and CLI debug output stay in lockstep with the codebase. Keep this list
# in sync with §2.5 of CLAUDE.md.


@dataclass(frozen=True)
class _EnvVar:
    """A single configurable env var."""

    name: str
    group: str
    default: str | None
    description: str
    secret: bool = False


# Ordered so the resulting dict is human-readable.
_ENV_REGISTRY: tuple[_EnvVar, ...] = (
    # --- paths / target -----------------------------------------------------
    _EnvVar("STROPHA_INDEX_PATH", "core", ".stropha/index.db", "SQLite index path"),
    _EnvVar("STROPHA_TARGET_REPO", "core", "<cwd>", "Default repo for index / MCP server"),
    _EnvVar("STROPHA_LOG_LEVEL", "core", "INFO", "structlog level"),
    _EnvVar("STROPHA_MAX_FILE_BYTES", "core", "524288", "Max file size walked (bytes)"),
    # --- embedders ----------------------------------------------------------
    _EnvVar("STROPHA_LOCAL_EMBED_MODEL", "embedder", "mixedbread-ai/mxbai-embed-large-v1", "Local fastembed model id"),
    _EnvVar("STROPHA_VOYAGE_EMBED_MODEL", "embedder", "voyage-code-3", "Voyage embedder model id"),
    _EnvVar("STROPHA_VOYAGE_EMBED_DIM", "embedder", "512", "Voyage embedding dim (Matryoshka)"),
    _EnvVar("VOYAGE_API_KEY", "embedder", None, "Voyage AI API key", secret=True),
    _EnvVar("ANTHROPIC_API_KEY", "embedder", None, "Anthropic API key (optional)", secret=True),
    # --- inference backends -------------------------------------------------
    _EnvVar("STROPHA_INFERENCE_BACKEND", "inference", None, "Preferred LLM backend (ollama/mlx)"),
    _EnvVar("STROPHA_OLLAMA_MODEL", "inference", "qwen2.5-coder:1.5b", "Default Ollama model"),
    _EnvVar("OLLAMA_HOST", "inference", "http://localhost:11434", "Ollama HTTP endpoint"),
    # --- query intelligence -------------------------------------------------
    _EnvVar("STROPHA_HYDE_ENABLED", "query", "0", "Route dense query through Ollama (HyDE rewrite)"),
    _EnvVar("STROPHA_HYDE_CODE_TUNED", "query", "1", "Auto-select code-tuned HyDE prompt"),
    _EnvVar("STROPHA_HYDE_PROMPT", "query", None, "Custom HyDE prompt template"),
    _EnvVar("STROPHA_QUERY_ROUTER_LLM", "query", "0", "Use LLM for ambiguous query routing"),
    _EnvVar("STROPHA_QUERY_DECOMPOSITION_ENABLED", "query", "0", "Split compound queries + RRF fuse"),
    _EnvVar("STROPHA_QUERY_DECOMPOSITION_LLM", "query", "0", "Use LLM for complex query decomposition"),
    _EnvVar("STROPHA_QUERY_REWRITE_ENABLED", "query", "0", "LLM expands natural language into code terms"),
    _EnvVar("STROPHA_QUERY_REWRITE_PROMPT", "query", None, "Custom rewrite prompt template"),
    _EnvVar("STROPHA_MULTI_QUERY_ENABLED", "query", "0", "Generate N paraphrases + RRF fuse"),
    _EnvVar("STROPHA_MULTI_QUERY_COUNT", "query", "3", "Number of paraphrases (1-5)"),
    _EnvVar("STROPHA_MULTI_QUERY_PROMPT", "query", None, "Custom paraphrase prompt template"),
    _EnvVar("STROPHA_ADAPTIVE_WEIGHTING", "query", "1", "Adaptive symbol/natural query weighting"),
    # --- query cache --------------------------------------------------------
    _EnvVar("STROPHA_QUERY_CACHE_ENABLED", "cache", "0", "Enable semantic query cache (LRU)"),
    _EnvVar("STROPHA_QUERY_CACHE_SIZE", "cache", "500", "Max cache entries"),
    _EnvVar("STROPHA_QUERY_CACHE_TTL", "cache", "3600", "Cache entry TTL (seconds)"),
    # --- retrieval behaviour ------------------------------------------------
    _EnvVar("STROPHA_RECURSIVE_RETRIEVAL", "retrieval", "0", "Parent + adjacent auto-merge"),
    _EnvVar("STROPHA_RECURSIVE_ADJACENCY", "retrieval", "5", "Line gap treated as 'adjacent'"),
    _EnvVar("STROPHA_GRAPH_FTS_AUGMENT", "retrieval", "1", "Retroactive FTS5 augmentation w/ graph"),
    _EnvVar("STROPHA_GRAPH_CONFIDENCE", "retrieval", "EXTRACTED", "Graph edge confidences to load"),
    _EnvVar("STROPHA_GRAPHIFY_OUT", "retrieval", "<repo>/graphify-out", "Override graphify output path"),
    # --- MCP server watcher -------------------------------------------------
    _EnvVar("STROPHA_MCP_WATCH", "mcp", "1", "Auto-reindex file watcher in MCP server"),
    _EnvVar("STROPHA_MCP_WATCH_INTERVAL", "mcp", "1.0", "Watcher polling interval (s)"),
    _EnvVar("STROPHA_MCP_WATCH_DEBOUNCE", "mcp", "2.0", "Watcher debounce window (s)"),
    # --- git url cache / walker --------------------------------------------
    _EnvVar("STROPHA_GIT_URL_CACHE", "git", "~/.cache/stropha/repos", "Cache dir for cloned remote repos"),
    _EnvVar("STROPHA_GIT_URL_TTL", "git", "86400", "Cache TTL for remote repos (s)"),
    _EnvVar("STROPHA_RENAME_THRESHOLD", "git", "80", "Git rename detection similarity (0-100)"),
    # --- post-commit hook ---------------------------------------------------
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
# Singleton management
# ---------------------------------------------------------------------------

_config_instance: Config | None = None
# Pre-`.env`-load snapshot. Captures every var in the registry so we can tell
# "process env" apart from ".env file" apart from "default".
_config_env_snapshot: dict[str, str | None] = {}


def get_config(*, reload: bool = False) -> Config:
    """Get the singleton Config instance.

    The config is loaded once from environment variables (with optional .env
    fallback). Subsequent calls return the cached instance unless reload=True.

    Environment variables take precedence over .env file values. The .env file
    is only consulted if STROPHA_INDEX_PATH is not already set in the environment
    (this prevents the stropha project's .env from overriding MCP client config).

    Args:
        reload: Force reload from environment. Use sparingly — typically only
            in tests or after programmatically changing os.environ.

    Returns:
        The singleton Config instance.
    """
    global _config_instance, _config_env_snapshot

    if _config_instance is None or reload:
        # Snapshot every registered env var BEFORE loading .env so we can
        # accurately report whether a value came from the process env or
        # bled in via .env afterwards.
        _config_env_snapshot = {name: os.environ.get(name) for name in _registry_names()}

        # Only load .env if critical env vars are not already set.
        # This ensures MCP clients (opencode) can override via env without
        # the stropha project's .env clobbering the values.
        if not os.environ.get("STROPHA_INDEX_PATH"):
            from dotenv import load_dotenv
            load_dotenv(override=False)

        _config_instance = Config()  # type: ignore[call-arg]

    return _config_instance


def _source_of(key: str) -> str:
    """Determine where a config value came from.

    - ``env``: set in the process env before stropha started
    - ``dotenv``: appeared in os.environ after stropha loaded .env
    - ``default``: never set; the registry/built-in default applies
    """
    snapshot_val = _config_env_snapshot.get(key)
    if snapshot_val is not None:
        return "env"
    if os.environ.get(key) is not None:
        return "dotenv"
    return "default"


def _effective_value(var: _EnvVar) -> str | None:
    """The value the runtime will actually observe for this env var.

    Returns ``(set)`` placeholder for secrets when present.
    """
    raw = os.environ.get(var.name)
    if raw is None:
        return var.default
    if var.secret:
        return "(set)"
    return raw


def get_config_info() -> dict[str, Any]:
    """Get configuration details for debugging.

    Returns a dict with:
    - ``config``: resolved values for the legacy core knobs (back-compat).
    - ``sources``: per-var origin (``env`` / ``dotenv`` / ``default``) for every
      env var in the registry — not just the core four.
    - ``env_values``: effective resolved value per var (secrets shown as
      ``(set)``; unset secrets shown as ``null``). Defaults are filled in so
      a caller can see exactly what the runtime will use.
    - ``settings``: full Pydantic Config dump (mirrors the BaseSettings model).
    - ``env_groups``: env vars bucketed by feature area for human readability.
    - ``runtime``: cwd + index/target existence flags.

    This powers the MCP ``get_config`` tool — keep it in sync with §2.5 of
    CLAUDE.md (the env var reference table).
    """
    cfg = get_config()

    # Per-var origin for every registered env var.
    sources: dict[str, str] = {var.name: _source_of(var.name) for var in _ENV_REGISTRY}

    # Effective value per var (with secret redaction). ``None`` means the var
    # has no value AND no default — i.e. the feature is off / opt-in.
    env_values: dict[str, str | None] = {
        var.name: _effective_value(var) for var in _ENV_REGISTRY
    }

    # Group by feature area so the LLM / human can navigate quickly.
    env_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for var in _ENV_REGISTRY:
        env_groups.setdefault(var.group, {})[var.name] = {
            "value": env_values[var.name],
            "default": var.default,
            "source": sources[var.name],
            "description": var.description,
            "secret": var.secret,
        }

    # Full pydantic dump — covers any Config field added in the future without
    # having to touch this function. ``Path`` fields stringified for JSON.
    settings_dump: dict[str, Any] = {}
    for field_name in type(cfg).model_fields:
        raw = getattr(cfg, field_name)
        if isinstance(raw, Path):
            settings_dump[field_name] = str(raw)
        elif field_name.endswith("_api_key") and raw:
            settings_dump[field_name] = "(set)"
        else:
            settings_dump[field_name] = raw
    # Computed properties not in model_fields.
    settings_dump["use_voyage"] = cfg.use_voyage

    return {
        # Back-compat: callers that read the legacy keys keep working.
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


def reset_config() -> None:
    """Reset the singleton (for tests). Forces reload on next get_config()."""
    global _config_instance, _config_env_snapshot
    _config_instance = None
    _config_env_snapshot = {}
