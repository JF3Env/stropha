"""Configuration dataclasses for the unified stropha config.

This is the single source of truth for all configuration types in stropha.
No other module should define configuration dataclasses or read environment
variables directly — everything goes through the unified ``Config`` object
returned by :func:`stropha.config.get_config`.

All settings are read from ``~/.stropha/config.yaml``, including secrets.
Environment variables are reserved for process/runtime arguments registered in
``stropha.config._ENV_REGISTRY``: target selection, the shared MCP launcher,
session/tracing identity, hook escape hatches, and opt-in launcher-managed
PostgreSQL.

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

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ..errors import ConfigError

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
    query_rewrite_max_terms: int = 6

    # Multi-query — N paraphrases of the query, RRF-fused.
    multi_query_enabled: bool = False
    multi_query_count: int = 3
    multi_query_min_terms: int = 7

    # Recursive retrieval — parent promotion + adjacency merge.
    recursive_enabled: bool = False
    recursive_adjacency: int = 5

    # Diversity-aware budget packing — prefer distinct evidence before
    # filling from deferred same-file/parent-symbol hits.
    diversity_enabled: bool = False
    diversity_max_chunks_per_file: int = 2

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

    The network backend fields are opt-in knobs. They are selected only when
    ``backend`` names them explicitly. API keys live in :class:`SecretsConfig`
    (YAML: ``secrets``).
    """

    backend: str | None = None
    mlx_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    ollama_model: str = "qwen2.5-coder:1.5b"
    ollama_host: str = "http://localhost:11434"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # DeepSeek is hosted abroad and goes through transcontinental links.
    # 60s is tight under load — 120s gives generous headroom while still
    # bounding the worst-case per-chunk wait.
    deepseek_timeout: float = 120.0
    # Retry on URLError / OSError / TimeoutError / HTTP 429 (rate limit) /
    # HTTP 5xx with exponential backoff (1s, 2s, 4s). 429 honours the
    # ``retry-after`` header when present, capped at 60s. Other HTTP 4xx
    # (bad key, bad request, wrong model) is permanent and short-circuits
    # without retry.
    deepseek_max_retries: int = 2
    # LMS OpenAI-compatible endpoint. Like other network backends, LMS is
    # explicit opt-in and is never selected by auto-detection. The endpoint has
    # no built-in value: it must be set in ~/.stropha/config.yaml.
    lms_model: str = "Qwen/Qwen3-Coder-Next-FP8"
    lms_base_url: str | None = None
    lms_timeout: float = 180.0
    lms_max_retries: int = 2
    lms_think: bool = False
    lms_max_connections: int = 8
    lms_max_keepalive_connections: int = 4
    # Local MoA Gateway. It exposes an OpenAI-compatible API and may run
    # without authentication when bound to loopback.
    moa_model: str = "moa-code"
    moa_base_url: str = "http://127.0.0.1:14598/v1"
    moa_timeout: float = 300.0
    moa_max_retries: int = 1
    # Anthropic Messages API (https://api.anthropic.com/v1/messages).
    # Default is claude-haiku-4-5 — fast + cheap, good fit for the
    # contextual enricher's per-chunk pattern (thousands of short calls).
    # Use claude-sonnet-4-5 or claude-opus-4-7 for higher quality at higher cost.
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    # Anthropic's US endpoint is fast — 60s is plenty for the Messages API,
    # which streams responses internally. Match DeepSeek's retry policy.
    anthropic_timeout: float = 60.0
    anthropic_max_retries: int = 2
    # Anthropic-Version header (required by the API). Pinned to the
    # last GA version; bump when Anthropic releases a new stable.
    anthropic_version: str = "2023-06-01"
    # Concurrent in-flight LLM requests during indexing. 1 is serial.
    # Higher values may improve remote-backend throughput, subject to provider
    # capacity; useful local concurrency depends on the available hardware.
    enricher_concurrency: int = 1
    # Maximum chunks packed into one call by enrichers that implement true
    # batching. 1 keeps per-chunk requests; larger values reduce request count
    # but increase prompt and response size. Other enrichers ignore this knob.
    enricher_batch_size: int = 1
    # Per-backend batch-size overrides. ``None`` = use ``enricher_batch_size``.
    # Set when a specific backend needs a tighter batch than the global
    # default, for example when a model's output becomes unreliable at larger
    # prompt or response sizes.
    deepseek_batch_size: int | None = None
    lms_batch_size: int | None = None
    moa_batch_size: int | None = None
    anthropic_batch_size: int | None = None
    ollama_batch_size: int | None = None
    mlx_batch_size: int | None = None

    def effective_batch_size(self) -> int:
        """Resolve the batch size for the currently-selected backend.

        Precedence:
          1. Per-backend override (``<backend>_batch_size``) when set.
          2. ``enricher_batch_size`` (global default).

        Used by the CLI to pass the right value into the Pipeline. The
        CLI ``--batch-size`` flag still wins above both (handled in
        ``cli.py``).
        """
        override = {
            "deepseek": self.deepseek_batch_size,
            "lms": self.lms_batch_size,
            "moa": self.moa_batch_size,
            "anthropic": self.anthropic_batch_size,
            "ollama": self.ollama_batch_size,
            "mlx": self.mlx_batch_size,
        }.get(self.backend or "")
        if override is not None:
            return override
        return self.enricher_batch_size


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
    """Default index path: always ``~/.stropha/index.db``.

    Historically this inspected ``Path.cwd() / ".stropha"`` first and only
    fell back to ``~`` when absent. That branch was a silent landmine: the
    CLI ran in the user's shell cwd while the post-commit hook ran in
    ``uv run --directory $PROJECT_DIR`` cwd, so the two could resolve to
    different DB files. Now both paths converge on the same user-level
    index. To use a project-local DB, set ``paths.index`` explicitly in
    ``~/.stropha/config.yaml`` (or ``pipeline.storage.config.path``).
    """
    return Path.home() / ".stropha" / "index.db"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths (YAML: ``paths``)."""

    target_repo: Path = field(default_factory=Path.cwd)
    index_path: Path = field(default_factory=_default_index_path)

    def resolve_index_path(self) -> Path:
        """Expand ``~`` in the configured index path."""
        return self.index_path.expanduser()


# ---------------------------------------------------------------------------
# Section: secrets (API keys — YAML: ``secrets``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretsConfig:
    """API keys. Read from ``~/.stropha/config.yaml`` under ``secrets:``."""

    voyage_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None
    lms_api_key: str | None = None
    moa_api_key: str | None = None

    @property
    def has_voyage(self) -> bool:
        return bool(self.voyage_api_key and self.voyage_api_key.strip())

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key and self.deepseek_api_key.strip())

    @property
    def has_lms(self) -> bool:
        return bool(self.lms_api_key and self.lms_api_key.strip())

    @property
    def has_moa(self) -> bool:
        return bool(self.moa_api_key and self.moa_api_key.strip())

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.strip())


# ---------------------------------------------------------------------------
# Section: embedder selection (model ids)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedderConfig:
    """Embedder model selection (YAML: ``embedder``)."""

    voyage_model: str = "voyage-4-large"
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


McpProfile = Literal["core", "standard", "full"]
MCP_PROFILES: tuple[McpProfile, ...] = ("core", "standard", "full")


@dataclass(frozen=True)
class McpConfig:
    """MCP server configuration (YAML: ``mcp``)."""

    profile: McpProfile = "full"
    watch: bool = True
    watch_interval: float = 1.0
    watch_debounce: float = 2.0

    def __post_init__(self) -> None:
        if self.profile not in MCP_PROFILES:
            choices = ", ".join(MCP_PROFILES)
            raise ConfigError(f"mcp.profile must be one of: {choices}")


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
    clone_depth: int = 1
    clone_timeout: int = 300
    rename_threshold: int = 80  # 0-100, git rename similarity


# ---------------------------------------------------------------------------
# Section: hook (post-commit hook settings)
# ---------------------------------------------------------------------------


def _default_hook_log() -> Path:
    return Path.home() / ".cache" / "stropha-hook.log"


@dataclass(frozen=True)
class HookConfig:
    """Post-commit hook configuration (YAML: ``hook``).

    These values come from the unified YAML configuration. The detached hook
    subprocess may apply its supported runtime overrides.

    Note: ``index_path`` removed in v=9. The hook always uses the YAML
    ``paths.index`` (resp. ``pipeline.storage.config.path``) — there is
    no separate per-hook DB. ``STROPHA_HOOK_INDEX_PATH`` is no longer
    honoured.
    """

    log_path: Path = field(default_factory=_default_hook_log)
    timeout: int = 600
    uv_path: str | None = None
    graphify_path: str | None = None
    skip: bool = False
    project_dir: Path | None = None


# ---------------------------------------------------------------------------
# Section: machine-local experiment policy
# ---------------------------------------------------------------------------


ExperimentStorage = Literal["sqlite-vec", "postgres"]
_DATABASE_PREFIX_RE = re.compile(r"[a-z][a-z0-9_]*")
_MAX_DATABASE_PREFIX_LENGTH = 32


@dataclass(frozen=True)
class ExperimentPostgresConfig:
    """PostgreSQL database provisioning settings for isolated experiments."""

    admin_dsn: str | None = None
    database_prefix: str = "stropha_bench"
    template_database: str = "template1"

    def __post_init__(self) -> None:
        if self.admin_dsn is not None:
            if not isinstance(self.admin_dsn, str):
                raise ConfigError("experiments.postgres.admin_dsn must be a string")
            if not self.admin_dsn.strip():
                object.__setattr__(self, "admin_dsn", None)
        prefix = self.database_prefix
        if not isinstance(prefix, str) or not prefix:
            raise ConfigError("experiments.postgres.database_prefix must be a non-empty string")
        if len(prefix) > _MAX_DATABASE_PREFIX_LENGTH or not _DATABASE_PREFIX_RE.fullmatch(prefix):
            raise ConfigError(
                "experiments.postgres.database_prefix must match [a-z][a-z0-9_]* "
                f"and be at most {_MAX_DATABASE_PREFIX_LENGTH} characters"
            )
        template = self.template_database
        if (
            not isinstance(template, str)
            or not template
            or len(template.encode()) > 63
            or not _DATABASE_PREFIX_RE.fullmatch(template)
        ):
            raise ConfigError(
                "experiments.postgres.template_database must be a lowercase PostgreSQL "
                "identifier of at most 63 bytes"
            )


@dataclass(frozen=True)
class ExperimentBlacklistConfig:
    """Machine-local adapter exclusions grouped by pipeline stage."""

    adapters: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.adapters, Mapping):
            raise ConfigError("experiments.blacklist.adapters must be a mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_stage, raw_adapters in self.adapters.items():
            if not isinstance(raw_stage, str) or not raw_stage.strip():
                raise ConfigError(
                    "experiments.blacklist.adapters stage names must be non-empty strings"
                )
            if not isinstance(raw_adapters, (list, tuple)):
                raise ConfigError(f"experiments.blacklist.adapters.{raw_stage} must be a list")
            adapters: list[str] = []
            for raw_adapter in raw_adapters:
                if not isinstance(raw_adapter, str) or not raw_adapter.strip():
                    raise ConfigError(
                        f"experiments.blacklist.adapters.{raw_stage} entries must be "
                        "non-empty strings"
                    )
                adapters.append(raw_adapter.strip())
            if len(set(adapters)) != len(adapters):
                raise ConfigError(
                    f"experiments.blacklist.adapters.{raw_stage} contains duplicate adapters"
                )
            stage = raw_stage.strip()
            if stage in normalized:
                raise ConfigError(
                    f"experiments.blacklist.adapters contains duplicate stage {stage!r}"
                )
            normalized[stage] = tuple(adapters)
        object.__setattr__(self, "adapters", MappingProxyType(normalized))


@dataclass(frozen=True)
class ExperimentsConfig:
    """Machine-local experiment storage and adapter eligibility policy."""

    storage: ExperimentStorage = "sqlite-vec"
    postgres: ExperimentPostgresConfig = field(default_factory=ExperimentPostgresConfig)
    blacklist: ExperimentBlacklistConfig = field(default_factory=ExperimentBlacklistConfig)

    def __post_init__(self) -> None:
        if self.storage not in ("sqlite-vec", "postgres"):
            raise ConfigError("experiments.storage must be 'sqlite-vec' or 'postgres'")

    def validate_for_run(self) -> None:
        """Validate requirements needed only when an experiment is about to run."""
        if self.storage == "postgres" and not self.postgres.admin_dsn:
            raise ConfigError(
                "experiments.postgres.admin_dsn is required when experiments.storage=postgres"
            )


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
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)

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

    def merge(self, **section_overrides: object) -> Config:
        """Return a new Config with the given sections replaced.

        Each kwarg must be one of the section names (e.g.
        ``retrieval_features``) and the value must be a dataclass of the
        matching type. Used by :func:`config_override`.
        """
        return replace(self, **section_overrides)  # type: ignore[arg-type]


# Convenience type aliases (back-compat).
UnifiedConfig = Config
