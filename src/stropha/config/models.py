"""Configuration dataclasses for the unified stropha config.

This is the single source of truth for all configuration types in stropha.
No other module should define configuration dataclasses or read environment
variables directly — everything goes through the unified ``Config`` object
returned by :func:`stropha.config.get_config`.

The structure is intentionally nested by concern so that adapters and
modules ask for the section they care about (``cfg.retrieval_features``,
``cfg.paths``, …) rather than reaching for flat attributes.

Backwards compatibility: a few flat properties (``log_level``,
``target_repo``, ``resolve_index_path``, ``use_voyage``, …) are provided
on the top-level ``Config`` so the existing call sites in ``cli.py`` /
``server.py`` continue to work unchanged. These are thin delegates to the
nested sections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


# ---------------------------------------------------------------------------
# Section: retrieval features (query-time toggles, no re-index required)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalFeatures:
    """Query-time retrieval feature configuration (YAML: ``retrieval_features``)."""

    # HyDE — hypothetical document embedding for the dense stream.
    hyde_enabled: bool = False
    hyde_code_tuned: bool = True

    # Query decomposition — split compound queries, search each, RRF-fuse.
    query_decomposition_enabled: bool = False
    query_decomposition_llm: bool = False

    # Query routing — auto-dispatch to graph tools for structural questions.
    query_router_llm: bool = False

    # Query rewrite — LLM expands natural language into code terms.
    query_rewrite_enabled: bool = False

    # Multi-query — N paraphrases of the query, RRF-fused.
    multi_query_enabled: bool = False
    multi_query_count: int = 3

    # Recursive retrieval — parent promotion + adjacency merge.
    recursive_enabled: bool = False
    recursive_adjacency: int = 5

    # Semantic query cache (LRU, in-memory).
    cache_enabled: bool = False
    cache_size: int = 500
    cache_ttl: float = 3600.0

    # Adaptive stream weighting (symbol queries boost lexical streams,
    # natural-language queries boost dense streams).
    adaptive_weighting: bool = True


# ---------------------------------------------------------------------------
# Section: inference (LLM backend)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """LLM inference backend configuration (YAML: ``inference``).

    The ``deepseek_*`` fields are opt-in cloud knobs — the backend is
    only selected when ``backend == "deepseek"``. The API key lives in
    :class:`SecretsConfig` (ENV only, never YAML).
    """

    backend: str | None = None  # "mlx", "ollama", "deepseek", or None for auto-detect
    mlx_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    ollama_model: str = "qwen2.5-coder:1.5b"
    ollama_host: str = "http://localhost:11434"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # DeepSeek is hosted abroad and goes through transcontinental links.
    # 60s is tight under load — 120s gives generous headroom while still
    # bounding the worst-case per-chunk wait.
    deepseek_timeout: float = 120.0
    # Retry on URLError / OSError / TimeoutError / HTTP 5xx with
    # exponential backoff (1s, 2s, 4s). HTTP 4xx (bad key, bad request,
    # rate limit) is permanent and short-circuits without retry.
    deepseek_max_retries: int = 2
    # Concurrent in-flight LLM requests during indexing. 1 = serial
    # (legacy behaviour, no thread overhead). 8 is a good default for
    # cloud backends (DeepSeek / OpenAI / Anthropic) — each chunk waits
    # ~5s on network, so 8 in-flight uses the available concurrency without
    # tripping rate limits at the basic-tier 60 req/min. Local backends
    # (mlx, ollama) saturate a single GPU regardless — keep at 1.
    enricher_concurrency: int = 1


# ---------------------------------------------------------------------------
# Section: graph integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    """Graph integration configuration (YAML: ``graph``)."""

    confidence: str = "EXTRACTED"  # EXTRACTED, INFERRED, or comma-separated
    fts_augment: bool = True
    graphify_out: Path | None = None  # override for graphify output dir


# ---------------------------------------------------------------------------
# Section: paths (index, target repo)
# ---------------------------------------------------------------------------


def _default_index_path() -> Path:
    """Default index path: ``.stropha/index.db`` relative to cwd, or ``~/.stropha/index.db``."""
    local = Path.cwd() / ".stropha" / "index.db"
    if local.parent.exists():
        return local
    return Path.home() / ".stropha" / "index.db"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths (YAML: ``paths``; ENV fallback)."""

    target_repo: Path = field(default_factory=Path.cwd)
    index_path: Path = field(default_factory=_default_index_path)

    def resolve_index_path(self) -> Path:
        """Expand ``~`` and env vars in the configured index path."""
        return Path(os.path.expandvars(str(self.index_path))).expanduser()


# ---------------------------------------------------------------------------
# Section: secrets (API keys — ENV only, never YAML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretsConfig:
    """API keys. ENV only — never persisted to YAML."""

    voyage_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None

    @property
    def has_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key.strip())


# ---------------------------------------------------------------------------
# Section: embedder selection (model ids)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedderConfig:
    """Embedder model selection (YAML: ``embedder``)."""

    voyage_model: str = "voyage-code-3"
    voyage_dim: int = 512
    local_model: str = "mixedbread-ai/mxbai-embed-large-v1"


# ---------------------------------------------------------------------------
# Section: walker (file discovery)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkerConfig:
    """Walker configuration (YAML: ``walker``)."""

    max_file_bytes: int = 524_288


# ---------------------------------------------------------------------------
# Section: logging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration (YAML: ``logging``)."""

    level: str = "INFO"


# ---------------------------------------------------------------------------
# Section: MCP server
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpConfig:
    """MCP server configuration (YAML: ``mcp``)."""

    watch: bool = True
    watch_interval: float = 1.0
    watch_debounce: float = 2.0


# ---------------------------------------------------------------------------
# Section: git (URL caching, rename detection)
# ---------------------------------------------------------------------------


def _default_git_url_cache() -> Path:
    return Path.home() / ".cache" / "stropha" / "repos"


@dataclass(frozen=True)
class GitConfig:
    """Git-related configuration (YAML: ``git``)."""

    url_cache: Path = field(default_factory=_default_git_url_cache)
    url_ttl: int = 86_400  # 24 hours
    rename_threshold: int = 80  # 0-100, git rename similarity


# ---------------------------------------------------------------------------
# Section: hook (post-commit hook settings — subprocess so ENV only)
# ---------------------------------------------------------------------------


def _default_hook_log() -> Path:
    return Path.home() / ".cache" / "stropha-hook.log"


@dataclass(frozen=True)
class HookConfig:
    """Post-commit hook configuration (YAML: ``hook``).

    These values are baked into the generated hook script at install time
    or read from ENV at runtime by the hook itself (which runs as a
    subprocess and cannot read the YAML config).
    """

    log_path: Path = field(default_factory=_default_hook_log)
    timeout: int = 600
    uv_path: str | None = None
    graphify_path: str | None = None
    skip: bool = False
    no_graphify: bool = False
    project_dir: Path | None = None
    index_path: Path | None = None


# ---------------------------------------------------------------------------
# Top-level unified Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Unified top-level configuration.

    All sections are populated with defaults; explicit values come from
    YAML and ENV via :func:`stropha.config.load_config`.

    The flat properties at the bottom of this class exist for backwards
    compatibility with the legacy pydantic ``Config`` API. New code
    should use the nested sections directly.
    """

    # Nested sections.
    retrieval_features: RetrievalFeatures = field(default_factory=RetrievalFeatures)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    walker: WalkerConfig = field(default_factory=WalkerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    git: GitConfig = field(default_factory=GitConfig)
    hook: HookConfig = field(default_factory=HookConfig)

    # -----------------------------------------------------------------
    # Backwards-compatibility flat accessors. Keep these stable —
    # external code (cli.py, server.py, embeddings/factory.py) reads
    # them directly. Internal new code should use the nested sections.
    # -----------------------------------------------------------------

    @property
    def log_level(self) -> str:
        return self.logging.level

    @property
    def max_file_bytes(self) -> int:
        return self.walker.max_file_bytes

    @property
    def target_repo(self) -> Path:
        return self.paths.target_repo

    @property
    def index_path(self) -> Path:
        return self.paths.index_path

    @property
    def voyage_api_key(self) -> str | None:
        return self.secrets.voyage_api_key

    @property
    def anthropic_api_key(self) -> str | None:
        return self.secrets.anthropic_api_key

    @property
    def use_voyage(self) -> bool:
        return self.secrets.has_voyage

    @property
    def voyage_embed_model(self) -> str:
        return self.embedder.voyage_model

    @property
    def voyage_embed_dim(self) -> int:
        return self.embedder.voyage_dim

    @property
    def local_embed_model(self) -> str:
        return self.embedder.local_model

    @property
    def mcp_watch(self) -> bool:
        return self.mcp.watch

    @property
    def mcp_watch_interval(self) -> float:
        return self.mcp.watch_interval

    @property
    def mcp_watch_debounce(self) -> float:
        return self.mcp.watch_debounce

    def resolve_index_path(self) -> Path:
        """Expand ``~`` and env vars in the configured index path."""
        return self.paths.resolve_index_path()

    # -----------------------------------------------------------------
    # Functional updates (used by config_override context manager).
    # -----------------------------------------------------------------

    def merge(self, **section_overrides: object) -> "Config":
        """Return a new Config with the given sections replaced.

        Each kwarg must be one of the section names (e.g.
        ``retrieval_features``) and the value must be a dataclass of the
        matching type. Used by :func:`config_override`.
        """
        return replace(self, **section_overrides)  # type: ignore[arg-type]


# Convenience type aliases (back-compat).
UnifiedConfig = Config
