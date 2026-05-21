"""Unified configuration for retrieval features, inference, and graph settings.

This module provides a centralized configuration system that loads from:
1. stropha.yaml (project-level)
2. ~/.stropha/config.yaml (user-level)

BREAKING CHANGE: The following ENV vars are no longer supported:
- STROPHA_HYDE_ENABLED, STROPHA_HYDE_CODE_TUNED
- STROPHA_QUERY_DECOMPOSITION_ENABLED, STROPHA_QUERY_DECOMPOSITION_LLM
- STROPHA_QUERY_ROUTER_LLM
- STROPHA_QUERY_REWRITE_ENABLED
- STROPHA_MULTI_QUERY_ENABLED, STROPHA_MULTI_QUERY_COUNT
- STROPHA_RECURSIVE_RETRIEVAL, STROPHA_RECURSIVE_ADJACENCY
- STROPHA_QUERY_CACHE_ENABLED, STROPHA_QUERY_CACHE_SIZE, STROPHA_QUERY_CACHE_TTL
- STROPHA_MLX_MODEL, STROPHA_OLLAMA_MODEL
- STROPHA_GRAPH_CONFIDENCE, STROPHA_GRAPH_FTS_AUGMENT

These are now configured via stropha.yaml sections:
- retrieval_features: query-time feature toggles
- inference: LLM backend settings
- graph: graph integration settings

ENV vars still supported:
- VOYAGE_API_KEY, ANTHROPIC_API_KEY (secrets)
- STROPHA_INDEX_PATH, STROPHA_TARGET_REPO (paths)
- STROPHA_HOOK_* (subprocess settings)
- OLLAMA_HOST (server URL, kept for compatibility with Ollama ecosystem)
- STROPHA_GRAPHIFY_OUT (override for graphify output path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RetrievalFeatures:
    """Query-time retrieval feature configuration."""

    # HyDE
    hyde_enabled: bool = False
    hyde_code_tuned: bool = True

    # Query decomposition
    query_decomposition_enabled: bool = False
    query_decomposition_llm: bool = False

    # Query routing
    query_router_llm: bool = False

    # Query rewrite
    query_rewrite_enabled: bool = False

    # Multi-query
    multi_query_enabled: bool = False
    multi_query_count: int = 3

    # Recursive retrieval
    recursive_enabled: bool = False
    recursive_adjacency: int = 5

    # Query cache
    cache_enabled: bool = False
    cache_size: int = 500
    cache_ttl: float = 3600.0


@dataclass(frozen=True)
class InferenceConfig:
    """LLM inference backend configuration."""

    backend: str | None = None  # "mlx", "ollama", or None for auto-detect
    mlx_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    ollama_model: str = "qwen2.5-coder:1.5b"
    ollama_host: str = "http://localhost:11434"


@dataclass(frozen=True)
class GraphConfig:
    """Graph integration configuration."""

    confidence: str = "EXTRACTED"  # EXTRACTED, INFERRED, or comma-separated
    fts_augment: bool = True


@dataclass(frozen=True)
class UnifiedConfig:
    """Top-level unified configuration."""

    retrieval_features: RetrievalFeatures = field(default_factory=RetrievalFeatures)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


# Module-level cached config
_cached_config: UnifiedConfig | None = None


def load_config(project_root: Path | None = None, force_reload: bool = False) -> UnifiedConfig:
    """Load unified configuration from YAML files.

    Args:
        project_root: Directory containing stropha.yaml (defaults to cwd).
        force_reload: Bypass cache and reload from disk.

    Returns:
        UnifiedConfig with all sections populated from YAML or defaults.
    """
    global _cached_config

    if _cached_config is not None and not force_reload:
        return _cached_config

    raw: dict[str, Any] = {}

    # 1. User config (~/.stropha/config.yaml)
    user_yaml = Path.home() / ".stropha" / "config.yaml"
    if user_yaml.is_file():
        try:
            user_raw = yaml.safe_load(user_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(user_raw, dict):
                raw.update(user_raw)
                log.debug("config.features.loaded_user", path=str(user_yaml))
        except yaml.YAMLError as exc:
            log.warning("config.features.invalid_yaml", path=str(user_yaml), error=str(exc))

    # 2. Project config (./stropha.yaml)
    project = project_root or Path.cwd()
    project_yaml = project / "stropha.yaml"
    if project_yaml.is_file():
        try:
            proj_raw = yaml.safe_load(project_yaml.read_text(encoding="utf-8")) or {}
            if isinstance(proj_raw, dict):
                # Deep merge project config over user config
                _deep_merge(raw, proj_raw)
                log.debug("config.features.loaded_project", path=str(project_yaml))
        except yaml.YAMLError as exc:
            log.warning("config.features.invalid_yaml", path=str(project_yaml), error=str(exc))

    # Parse sections
    retrieval_features = _parse_retrieval_features(raw.get("retrieval_features", {}))
    inference = _parse_inference(raw.get("inference", {}))
    graph = _parse_graph(raw.get("graph", {}))

    _cached_config = UnifiedConfig(
        retrieval_features=retrieval_features,
        inference=inference,
        graph=graph,
    )

    log.info(
        "config.features.loaded",
        hyde=retrieval_features.hyde_enabled,
        recursive=retrieval_features.recursive_enabled,
        cache=retrieval_features.cache_enabled,
        inference_backend=inference.backend,
    )

    return _cached_config


def get_config() -> UnifiedConfig:
    """Get the cached config, loading if needed."""
    return load_config()


def reset_config() -> None:
    """Reset the cached config. Useful for testing."""
    global _cached_config
    _cached_config = None


def _parse_retrieval_features(raw: dict[str, Any]) -> RetrievalFeatures:
    """Parse retrieval_features section."""
    return RetrievalFeatures(
        hyde_enabled=_bool(raw.get("hyde_enabled", False)),
        hyde_code_tuned=_bool(raw.get("hyde_code_tuned", True)),
        query_decomposition_enabled=_bool(raw.get("query_decomposition_enabled", False)),
        query_decomposition_llm=_bool(raw.get("query_decomposition_llm", False)),
        query_router_llm=_bool(raw.get("query_router_llm", False)),
        query_rewrite_enabled=_bool(raw.get("query_rewrite_enabled", False)),
        multi_query_enabled=_bool(raw.get("multi_query_enabled", False)),
        multi_query_count=_int(raw.get("multi_query_count", 3)),
        recursive_enabled=_bool(raw.get("recursive_enabled", False)),
        recursive_adjacency=_int(raw.get("recursive_adjacency", 5)),
        cache_enabled=_bool(raw.get("cache_enabled", False)),
        cache_size=_int(raw.get("cache_size", 500)),
        cache_ttl=_float(raw.get("cache_ttl", 3600.0)),
    )


def _parse_inference(raw: dict[str, Any]) -> InferenceConfig:
    """Parse inference section."""
    # Allow OLLAMA_HOST env var for ecosystem compatibility
    ollama_host = raw.get("ollama_host") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )

    return InferenceConfig(
        backend=raw.get("backend"),
        mlx_model=str(raw.get("mlx_model", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")),
        ollama_model=str(raw.get("ollama_model", "qwen2.5-coder:1.5b")),
        ollama_host=str(ollama_host),
    )


def _parse_graph(raw: dict[str, Any]) -> GraphConfig:
    """Parse graph section."""
    return GraphConfig(
        confidence=str(raw.get("confidence", "EXTRACTED")),
        fts_augment=_bool(raw.get("fts_augment", True)),
    )


def _bool(val: Any) -> bool:
    """Coerce value to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "on", "1")
    return bool(val)


def _int(val: Any) -> int:
    """Coerce value to int."""
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _float(val: Any) -> float:
    """Coerce value to float."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Deep merge src into dst in-place."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
