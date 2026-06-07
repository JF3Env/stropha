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

import yaml  # type: ignore[import-untyped]

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
            # Reranking is OFF by default.
            #
            # WIN-296 made the cross-encoder (bge-reranker-base) the default on
            # a +26.7% Recall@10 win (0.50 -> 0.63). That win held when
            # first-stage recall was weak (~0.50). With the current setup
            # first-stage recall@10 is ~0.80-0.87, and re-measuring on the
            # golden set across three embedders (voyage / mlx / model2vec,
            # HyDE off, deterministic) shows the cross-encoder now HURTS code
            # retrieval — it drops recall@10 by 0.13-0.20 and roughly halves
            # MRR by demoting exact code matches (it is NL-trained). The
            # `code-aware` heuristic reranker is less harmful but still trailed
            # no-reranker on every metric on the voyage embedder. So the
            # net-best default is no reranker.
            #
            # Re-enable per-repo in ~/.stropha/config.yaml, e.g.:
            #   pipeline.retrieval.config.reranker:
            #     adapter: code-aware        # or: cross-encoder
            # (the adapters are unchanged; only the default is flipped). Also
            # adds ~1.4s/query latency for the cross-encoder.
            "reranker": None,
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
    yaml_storage_path_explicit: str | None = None
    yaml_paths_index_explicit: str | None = None
    if user_yaml.is_file():
        # Read the raw YAML once so we can distinguish "user explicitly
        # set the storage path key" from "the storage path arrived via
        # CLI overrides / built-in default". The conflict guard below
        # only fires when BOTH come from YAML — we don't want to break
        # the standard CLI / test override path that explicitly passes
        # a storage path via `overrides`.
        raw_yaml = _read_raw_yaml(user_yaml)
        yaml_paths_index_explicit = (
            (raw_yaml.get("paths") or {}).get("index")
        )
        yaml_pipeline_section = raw_yaml.get("pipeline", raw_yaml)
        if isinstance(yaml_pipeline_section, dict):
            yaml_storage_path_explicit = (
                ((yaml_pipeline_section.get("storage") or {}).get("config") or {}).get("path")
            )
        _merge(cfg, _read_yaml(user_yaml))
        sources.append(str(user_yaml))

    # 1a. Conflict guard at YAML level only. If the user wrote BOTH
    # `paths.index` AND `pipeline.storage.config.path` in the YAML and
    # they disagree, fail loudly — that's the silent split landmine we
    # hit before v=9 (CLI panel showed one path, storage opened another).
    if yaml_paths_index_explicit and yaml_storage_path_explicit:
        p1 = Path(yaml_paths_index_explicit).expanduser().resolve()
        p2 = Path(yaml_storage_path_explicit).expanduser().resolve()
        if p1 != p2:
            raise ConfigError(
                "Index path conflict in ~/.stropha/config.yaml: "
                f"paths.index resolves to {p1} but "
                f"pipeline.storage.config.path resolves to {p2}. "
                "Pick one (paths.index is preferred; drop the storage override)."
            )

    # 2. CLI overrides (highest priority). May legitimately override the
    # storage path for tests, per-repo evaluation, or one-off runs.
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

    # 4. Materialize the storage path. Default = unified config's
    # `paths.index` (sourced from YAML or, in its absence, from
    # `_default_index_path()` which is now cwd-independent).
    storage_cfg = cfg.setdefault("storage", {}).setdefault("config", {})
    if not storage_cfg.get("path"):
        try:
            from ..config import get_config
            unified_index = get_config().resolve_index_path()
        except Exception:
            from ..config.models import _default_index_path
            unified_index = _default_index_path()
        storage_cfg["path"] = str(unified_index)
        sources.append("default(paths.index)")

    log.debug("pipeline.config.load", sources=sources, resolved=cfg)
    return cfg


def _read_raw_yaml(path: Path) -> dict[str, Any]:
    """Read ``path`` as a YAML mapping with no transformation.

    Used by the index-path conflict guard to peek at the raw user YAML
    BEFORE the ``pipeline:`` section is unwrapped. Returns ``{}`` on any
    error so the guard fails open (rather than blocking the load on
    transient YAML problems that the main loader already handles).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


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
