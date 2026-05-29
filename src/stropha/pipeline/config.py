"""Pipeline configuration loader.

Cascading precedence:

    CLI overrides  >  YAML (~/.stropha/config.yaml)  >  Built-in defaults

Single config file: ``~/.stropha/config.yaml``.  No project-level override,
no env-var overlay (``STROPHA_<STAGE>__<KEY>`` format removed), no legacy
aliases.

For embedder auto-resolution (adapter=None), the YAML ``secrets.voyage_api_key``
field determines whether voyage or local is selected.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError
from ..logging import get_logger

log = get_logger(__name__)

# Top-level config skeleton. Adapter sections that omit ``adapter`` fall back
# to these defaults. Adding a stage here in a future phase is the only place
# the framework needs touching.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "walker": {"adapter": "git-ls-files", "config": {}},
    "chunker": {"adapter": "tree-sitter-dispatch", "config": {}},
    "enricher": {"adapter": "noop", "config": {}},
    "embedder": {"adapter": None, "config": {}},  # `None` => auto (voyage if key, else local)
    "storage": {"adapter": "sqlite-vec", "config": {}},
    "retrieval": {
        "adapter": "hybrid-rrf",
        "config": {
            # WIN-296: cross-encoder reranker is default. Benchmark shows +26.7%
            # Recall@10 (0.50 → 0.63) over noop, with ~1.4s added latency per
            # query. Model is local ONNX (~1GB, downloaded on first use). Set
            # reranker: null in ~/.stropha/config.yaml to disable.
            "reranker": {
                "adapter": "cross-encoder",
                "config": {"model": "BAAI/bge-reranker-base"},
            },
            "candidate_k": 50,
        },
    },
}


def pipeline_config_from_dict(
    overrides: dict[str, Any],
    *,
    resolve_storage_path: bool = True,
) -> dict[str, Any]:
    """Build a resolved pipeline config from ``overrides`` ONLY.

    WIN-179: the Meta-Harness loop needs to score candidate configs in
    full isolation. ``load_pipeline_config`` reads
    ``~/.stropha/config.yaml``, so two candidate evaluations on the
    same host would silently inherit whatever the operator has in that
    file — fitness comparisons become non-reproducible.

    This helper bypasses every external source. It deep-merges
    ``overrides`` onto the built-in ``_DEFAULTS`` skeleton, resolves
    embedder=None to ``local`` (deterministic, no key lookup), and
    optionally materialises the storage path.

    Args:
        overrides: the candidate's pipeline config. Same shape as
            ``load_pipeline_config`` returns.
        resolve_storage_path: when True (default), fills
            ``storage.config.path`` with the built-in default if the
            caller didn't set one. Set to False when the caller has
            already chosen an explicit path (e.g. from the WIN-177
            index cache).

    Returns the fully-resolved config dict.
    """
    cfg = copy.deepcopy(_DEFAULTS)
    if overrides:
        _merge(cfg, overrides)

    # Auto-resolve embedder=None to local (deterministic, no secrets lookup).
    if cfg["embedder"].get("adapter") is None:
        cfg["embedder"]["adapter"] = "local"

    if resolve_storage_path:
        storage_cfg = cfg.setdefault("storage", {}).setdefault("config", {})
        if not storage_cfg.get("path"):
            from ..config.models import _default_index_path
            storage_cfg["path"] = str(_default_index_path())

    log.debug("pipeline.config.from_dict", resolved=cfg)
    return cfg


def load_pipeline_config(
    *,
    project_root: Path | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the full pipeline config dict.

    Args:
        project_root: Ignored — kept for call-site compatibility only.
                      All config comes from ``~/.stropha/config.yaml``.
        overrides: CLI-level overrides, deep-merged last (highest priority).
        environ: Ignored — kept for call-site compatibility only.
                 Env vars no longer influence pipeline config.

    Returns the resolved config in the form ``{"embedder": {...}, "enricher": {...}}``,
    where each section has ``adapter`` (str|None) and ``config`` (dict).
    """
    if environ is not None:
        log.debug("pipeline.config.environ_ignored", reason="env vars no longer used for pipeline config")

    cfg = copy.deepcopy(_DEFAULTS)
    sources: list[str] = []

    # 1. User YAML (~/.stropha/config.yaml) — the only config file.
    user_yaml = Path.home() / ".stropha" / "config.yaml"
    if user_yaml.is_file():
        _merge(cfg, _read_yaml(user_yaml))
        sources.append(str(user_yaml))

    # 2. CLI overrides (highest priority).
    if overrides:
        _merge(cfg, overrides)
        sources.append("cli")

    # 3. Resolve embedder=None (auto pick voyage vs local).
    #    Uses secrets.voyage_api_key from YAML (already loaded above via
    #    the unified Config). We import here to avoid circular imports.
    if cfg["embedder"].get("adapter") is None:
        try:
            from ..config import get_config
            has_voyage = get_config().secrets.has_voyage
        except Exception:
            has_voyage = False
        cfg["embedder"]["adapter"] = "voyage" if has_voyage else "local"

    # 4. Materialize the storage path so it shows up in logs / `pipeline show`.
    storage_cfg = cfg.setdefault("storage", {}).setdefault("config", {})
    if not storage_cfg.get("path"):
        from ..config.models import _default_index_path
        storage_cfg["path"] = str(_default_index_path())
        sources.append("default(_default_index_path)")

    log.debug("pipeline.config.load", sources=sources, resolved=cfg)
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    # Accept either {"pipeline": {...}} (canonical) or the bare sections.
    inner = raw.get("pipeline", raw)
    if not isinstance(inner, dict):
        raise ConfigError(f"{path}: `pipeline` must be a mapping")
    return inner


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Deep merge ``src`` into ``dst`` in-place."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
