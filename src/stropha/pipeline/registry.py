"""Adapter registry — populated at import-time via the ``@register_adapter`` decorator.

Per ADR-003 (``docs/architecture/stropha-pipeline-adapters.md``): adding a new
adapter is adding a single file under ``src/stropha/adapters/<stage>/``. The
package's ``__init__`` walks every submodule on import, which triggers the
decorator side-effect that populates the registry.

WIN-199 (Phase 5 code search) introduces **namespaced registries**: when the
loop is evaluating a candidate that ships ``code_artifacts`` (WIN-196), it
can wrap the candidate's adapter loading in :func:`namespace_scope` so the
``@register_adapter`` writes go to a per-candidate dictionary, not the global
production registry. This lets two candidates with same-named adapters live
side-by-side without collision, and guarantees that nothing the candidate
registers leaks into production state once the scope exits.

Public surface
--------------

- :func:`register_adapter` — the decorator. Honours the active namespace.
- :func:`lookup_adapter` — resolve ``(stage, name)``. Checks the active
  namespace first, falls back to the global registry.
- :func:`available_for_stage` / :func:`all_adapters` — introspection. Union
  of namespace + global.
- :func:`namespace_scope` — context manager that activates a candidate
  namespace.
- :func:`drop_namespace` — discard a namespace's adapters (free memory after
  a candidate evaluation completes).
- :func:`get_active_namespace`, :func:`get_registry` — readers for advanced
  callers (the sandbox runner uses :func:`get_registry` to introspect what
  the candidate registered).

Imports are intentionally cheap: heavy work (ONNX model load, HTTP clients)
happens in the adapter's ``__init__``, not at module import.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any

from ..errors import AdapterError, ConfigError

# The production / default registry. Every ``@register_adapter`` call that
# runs OUTSIDE a :func:`namespace_scope` writes here.
_GLOBAL_REGISTRY: dict[tuple[str, str], type[Any]] = {}

# Back-compat alias. Older tests + the module docstring referred to
# ``_REGISTRY`` directly; keeping the symbol pointing at the same dict
# avoids a sweeping rename across the test suite.
_REGISTRY = _GLOBAL_REGISTRY

# Per-namespace registries. Keyed by candidate id (or any caller-chosen
# string). Created lazily on first :func:`register_adapter` write inside
# the scope OR via :func:`get_registry`.
_NAMESPACED_REGISTRIES: dict[str, dict[tuple[str, str], type[Any]]] = {}

# Tracks the currently-active namespace for the running task. Using a
# ``ContextVar`` (not a plain global) means async tasks and threads stay
# isolated — exactly what the Meta-Harness needs once it ships
# concurrent candidate evaluation.
_active_namespace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "stropha_active_registry_namespace",
    default=None,
)


def register_adapter(*, stage: str, name: str):
    """Class decorator. Registers ``cls`` under ``(stage, name)``.

    Writes go to the **active namespace** when one is set via
    :func:`namespace_scope`, otherwise to the global registry. The adapter
    class MUST expose a ``Config`` class attribute (a pydantic
    ``BaseModel`` subtype) so the builder can validate user-supplied config.

    Duplicate-registration semantics: registering the same class twice
    under the same key is idempotent. Registering a *different* class
    under an already-taken key raises :class:`AdapterError`. Collision
    is checked within the active namespace only — a candidate registering
    ``(enricher, foo)`` does NOT clash with a global ``(enricher, foo)``;
    that's the whole point of namespacing.
    """

    def deco(cls: type[Any]) -> type[Any]:
        if not hasattr(cls, "Config"):
            raise AdapterError(
                f"{cls.__module__}.{cls.__name__} is missing the `Config` "
                f"class attribute required by register_adapter"
            )
        registry = _registry_for_writes()
        key = (stage, name)
        if key in registry and registry[key] is not cls:
            raise AdapterError(
                f"Duplicate adapter registration for {stage!r}/{name!r}: "
                f"{registry[key].__module__} vs {cls.__module__}"
            )
        registry[key] = cls
        return cls

    return deco


def lookup_adapter(stage: str, name: str) -> type[Any]:
    """Return the adapter class registered for ``(stage, name)``.

    Lookup order: active namespace → global registry. A namespace
    registration overrides a global registration with the same key — that
    lets a candidate experimentally replace a production adapter without
    touching the global.

    Raises :class:`ConfigError` with the list of valid alternatives when
    the requested adapter does not exist in either layer.
    """
    namespace = _active_namespace.get()
    if namespace is not None:
        cls = _NAMESPACED_REGISTRIES.get(namespace, {}).get((stage, name))
        if cls is not None:
            return cls
    cls = _GLOBAL_REGISTRY.get((stage, name))
    if cls is None:
        available = available_for_stage(stage)
        hint = (
            f"Available {stage} adapters: {available}" if available else (
                f"No {stage} adapters registered. Did you forget to import "
                f"`stropha.adapters` to trigger auto-registration?"
            )
        )
        raise ConfigError(f"Unknown {stage} adapter {name!r}. {hint}")
    return cls


def available_for_stage(stage: str) -> list[str]:
    """Return sorted adapter names registered for ``stage``.

    Union of the active namespace + global registry, de-duplicated. A
    namespace-only adapter and a global adapter with the same name appear
    once.
    """
    names: set[str] = set()
    namespace = _active_namespace.get()
    if namespace is not None:
        for s, n in _NAMESPACED_REGISTRIES.get(namespace, {}):
            if s == stage:
                names.add(n)
    for s, n in _GLOBAL_REGISTRY:
        if s == stage:
            names.add(n)
    return sorted(names)


def all_adapters() -> dict[str, list[str]]:
    """Snapshot of the registry, grouped by stage. Used by ``adapters list``.

    Union of the active namespace + global. Duplicate adapter names
    (same name in both layers) collapse to a single entry.
    """
    out: dict[str, set[str]] = {}
    namespace = _active_namespace.get()
    if namespace is not None:
        for s, n in _NAMESPACED_REGISTRIES.get(namespace, {}):
            out.setdefault(s, set()).add(n)
    for s, n in _GLOBAL_REGISTRY:
        out.setdefault(s, set()).add(n)
    return {s: sorted(v) for s, v in out.items()}


@contextlib.contextmanager
def namespace_scope(namespace: str) -> Iterator[dict[tuple[str, str], type[Any]]]:
    """Activate a per-candidate registry namespace for the enclosed block.

    All ``@register_adapter`` calls inside the ``with`` body write to the
    namespace's dictionary. On exit, the previous namespace (usually None)
    is restored — global registration state is untouched.

    Yields the namespace dict so the caller can introspect what got
    registered::

        with namespace_scope(candidate_id) as ns_reg:
            # ... import candidate adapter files ...
            assert ("enricher", "candidate-only") in ns_reg

    The namespace dict is **persisted** across scope-exit so a follow-up
    ``lookup_adapter`` from a different scope can find the candidate's
    adapters. Use :func:`drop_namespace` to discard.
    """
    if not isinstance(namespace, str) or not namespace:
        raise AdapterError(
            f"namespace must be a non-empty string; got {namespace!r}"
        )
    registry = _NAMESPACED_REGISTRIES.setdefault(namespace, {})
    token = _active_namespace.set(namespace)
    try:
        yield registry
    finally:
        _active_namespace.reset(token)


def drop_namespace(namespace: str) -> None:
    """Discard a namespace's registry.

    Idempotent: dropping a namespace that doesn't exist is a no-op.
    Call this once a candidate evaluation completes so the per-candidate
    adapters don't accumulate in memory across a long-running loop.
    """
    _NAMESPACED_REGISTRIES.pop(namespace, None)


def get_active_namespace() -> str | None:
    """Return the namespace currently active in this context, or ``None``."""
    return _active_namespace.get()


def get_registry(
    namespace: str | None = None,
) -> dict[tuple[str, str], type[Any]]:
    """Return the registry dict for ``namespace`` (or the global one).

    Passing ``None`` (the default) returns the global registry. Passing
    a namespace creates the dict lazily — useful for tests that want to
    pre-populate a namespace before activating it via
    :func:`namespace_scope`.
    """
    if namespace is None:
        return _GLOBAL_REGISTRY
    return _NAMESPACED_REGISTRIES.setdefault(namespace, {})


def _registry_for_writes() -> dict[tuple[str, str], type[Any]]:
    """Return the dict ``@register_adapter`` should write into.

    Namespace if one is active, otherwise the global.
    """
    namespace = _active_namespace.get()
    if namespace is None:
        return _GLOBAL_REGISTRY
    return _NAMESPACED_REGISTRIES.setdefault(namespace, {})


def _reset_for_tests() -> None:
    """Test helper: clear EVERY registry. Do NOT call from production code.

    Both the global registry and all namespaces are wiped. Use sparingly —
    many production paths depend on auto-registration that happens once at
    import time; clearing the registry mid-run leaves the pipeline unable
    to resolve adapters until the modules are re-imported.
    """
    _GLOBAL_REGISTRY.clear()
    _NAMESPACED_REGISTRIES.clear()
