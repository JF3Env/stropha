"""Unit tests for the Phase-5 namespaced registry (WIN-199).

The namespaced registry lets two candidates with same-named adapters
coexist without a global collision. WIN-196 ships the artifacts;
WIN-198 ships the subprocess sandbox that imports each candidate in
isolation. WIN-199 ships the in-process layer: when two candidates
import their adapters in the same Python process (e.g. concurrent
evaluators in a future async-loop), :func:`namespace_scope` keeps
their ``(stage, name)`` mappings apart.

Coverage targets:

- Back-compat: existing global ``register_adapter`` / ``lookup_adapter``
  / ``available_for_stage`` / ``all_adapters`` keep working.
- ``namespace_scope``: routes writes to the namespace, restores on
  exit, persists across exits, raises on empty/non-string.
- ``drop_namespace``: removes the namespace state, idempotent.
- ``lookup_adapter``: namespace-first lookup, falls back to global,
  raises ConfigError when neither has the adapter.
- Namespace adapter overrides global with same key (per-candidate
  experiments). Two parallel candidates with same adapter name do NOT
  collide.
- ``available_for_stage`` + ``all_adapters``: union semantics, de-dup.
- ``get_active_namespace`` / ``get_registry`` read paths.
- ``ContextVar`` semantics: nested scopes restore the outer namespace.
- ``_reset_for_tests``: nukes everything.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from stropha.errors import AdapterError, ConfigError
from stropha.pipeline import registry as reg


class _DummyConfig(BaseModel):
    """Minimal Pydantic config so the decorator's hasattr(cls, 'Config') passes."""


def _adapter_class(name_hint: str) -> type:
    """Build a fresh class on demand — lets us register N variants in tests."""

    class _Adapter:
        Config = _DummyConfig
        __module__ = f"test_namespaced_registry.{name_hint}"

    _Adapter.__qualname__ = f"_Adapter_{name_hint}"
    return _Adapter


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot + restore the registry across every test.

    We can't call ``_reset_for_tests`` because it would also wipe the
    production registrations the rest of the suite depends on. Instead
    we snapshot the dicts on setup and restore on teardown.
    """
    global_snapshot = dict(reg._GLOBAL_REGISTRY)
    ns_snapshot = {k: dict(v) for k, v in reg._NAMESPACED_REGISTRIES.items()}
    try:
        yield
    finally:
        reg._GLOBAL_REGISTRY.clear()
        reg._GLOBAL_REGISTRY.update(global_snapshot)
        reg._NAMESPACED_REGISTRIES.clear()
        reg._NAMESPACED_REGISTRIES.update(ns_snapshot)


# ---------------------------------------------------------------------------
# Global (back-compat) path
# ---------------------------------------------------------------------------


class TestGlobalRegistration:
    def test_register_writes_to_global(self) -> None:
        cls = _adapter_class("g1")
        reg.register_adapter(stage="enricher", name="g1")(cls)
        assert reg._GLOBAL_REGISTRY[("enricher", "g1")] is cls
        # And the namespaced dict is untouched.
        assert reg._NAMESPACED_REGISTRIES == {}

    def test_lookup_finds_global(self) -> None:
        cls = _adapter_class("g2")
        reg.register_adapter(stage="walker", name="g2")(cls)
        assert reg.lookup_adapter("walker", "g2") is cls

    def test_lookup_unknown_raises_with_hint(self) -> None:
        with pytest.raises(ConfigError, match="Unknown enricher adapter"):
            reg.lookup_adapter("enricher", "nope")

    def test_register_without_config_raises(self) -> None:
        class _NoConfig:
            __module__ = "x"

        with pytest.raises(AdapterError, match="missing the `Config`"):
            reg.register_adapter(stage="enricher", name="noc")(_NoConfig)

    def test_idempotent_re_register_same_class(self) -> None:
        cls = _adapter_class("dup")
        deco = reg.register_adapter(stage="enricher", name="dup")
        deco(cls)
        # Re-register the same class — must NOT raise.
        deco(cls)
        assert reg._GLOBAL_REGISTRY[("enricher", "dup")] is cls

    def test_different_class_same_key_raises(self) -> None:
        a = _adapter_class("a")
        b = _adapter_class("b")
        reg.register_adapter(stage="enricher", name="x")(a)
        with pytest.raises(AdapterError, match="Duplicate"):
            reg.register_adapter(stage="enricher", name="x")(b)

    def test_legacy_REGISTRY_alias(self) -> None:
        """The pre-WIN-199 _REGISTRY symbol still resolves to the global."""
        assert reg._REGISTRY is reg._GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# namespace_scope
# ---------------------------------------------------------------------------


class TestNamespaceScope:
    def test_register_routes_to_namespace(self) -> None:
        cls = _adapter_class("nsa")
        with reg.namespace_scope("cand-A") as ns_reg:
            reg.register_adapter(stage="enricher", name="nsa")(cls)
            assert ns_reg[("enricher", "nsa")] is cls
        # Outside the scope, global is untouched.
        assert ("enricher", "nsa") not in reg._GLOBAL_REGISTRY
        # Namespace dict persists for later lookup.
        assert reg._NAMESPACED_REGISTRIES["cand-A"][("enricher", "nsa")] is cls

    def test_scope_restores_previous_namespace(self) -> None:
        assert reg.get_active_namespace() is None
        with reg.namespace_scope("cand-A"):
            assert reg.get_active_namespace() == "cand-A"
        assert reg.get_active_namespace() is None

    def test_nested_scopes(self) -> None:
        with reg.namespace_scope("outer"):
            assert reg.get_active_namespace() == "outer"
            with reg.namespace_scope("inner"):
                assert reg.get_active_namespace() == "inner"
            assert reg.get_active_namespace() == "outer"
        assert reg.get_active_namespace() is None

    def test_empty_namespace_rejected(self) -> None:
        with pytest.raises(AdapterError, match="non-empty"), reg.namespace_scope(""):
            ...

    def test_non_string_namespace_rejected(self) -> None:
        with (
            pytest.raises(AdapterError, match="non-empty"),
            reg.namespace_scope(123),  # type: ignore[arg-type]
        ):
            ...

    def test_two_candidates_dont_collide(self) -> None:
        a_cls = _adapter_class("A")
        b_cls = _adapter_class("B")
        with reg.namespace_scope("cand-A"):
            reg.register_adapter(stage="enricher", name="shared")(a_cls)
        with reg.namespace_scope("cand-B"):
            reg.register_adapter(stage="enricher", name="shared")(b_cls)
        # Both candidates have their own ``shared`` adapter.
        assert reg._NAMESPACED_REGISTRIES["cand-A"][("enricher", "shared")] is a_cls
        assert reg._NAMESPACED_REGISTRIES["cand-B"][("enricher", "shared")] is b_cls
        # Global is empty.
        assert ("enricher", "shared") not in reg._GLOBAL_REGISTRY

    def test_collision_within_same_namespace_raises(self) -> None:
        a = _adapter_class("a")
        b = _adapter_class("b")
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="dup")(a)
            with pytest.raises(AdapterError, match="Duplicate"):
                reg.register_adapter(stage="enricher", name="dup")(b)


# ---------------------------------------------------------------------------
# Lookup precedence
# ---------------------------------------------------------------------------


class TestLookupPrecedence:
    def test_namespace_overrides_global(self) -> None:
        global_cls = _adapter_class("global_v")
        ns_cls = _adapter_class("ns_v")
        reg.register_adapter(stage="enricher", name="thing")(global_cls)
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="thing")(ns_cls)
            # Inside the scope, lookup returns the namespace version.
            assert reg.lookup_adapter("enricher", "thing") is ns_cls
        # Outside the scope, the global version is back.
        assert reg.lookup_adapter("enricher", "thing") is global_cls

    def test_namespace_falls_back_to_global(self) -> None:
        global_cls = _adapter_class("g")
        reg.register_adapter(stage="enricher", name="only_in_global")(global_cls)
        with reg.namespace_scope("ns"):
            assert reg.lookup_adapter("enricher", "only_in_global") is global_cls

    def test_namespace_only_invisible_outside(self) -> None:
        ns_cls = _adapter_class("nsx")
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="nsx")(ns_cls)
        # Outside the scope, the candidate adapter is unreachable.
        with pytest.raises(ConfigError, match="Unknown enricher"):
            reg.lookup_adapter("enricher", "nsx")

    def test_neither_layer_has_it_raises(self) -> None:
        with reg.namespace_scope("ns"), pytest.raises(ConfigError, match="Unknown"):
            reg.lookup_adapter("enricher", "vanished")

    def test_lookup_hint_lists_namespace_adapters(self) -> None:
        """The ``Available`` hint must include namespace-only adapters."""
        ns_cls = _adapter_class("hinted")
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="hinted")(ns_cls)
            try:
                reg.lookup_adapter("enricher", "missing")
            except ConfigError as exc:
                assert "hinted" in str(exc)
            else:  # pragma: no cover
                pytest.fail("expected ConfigError")


# ---------------------------------------------------------------------------
# Introspection (available_for_stage / all_adapters)
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_available_union_of_layers(self) -> None:
        g = _adapter_class("g")
        n = _adapter_class("n")
        reg.register_adapter(stage="enricher", name="only_global")(g)
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="only_ns")(n)
            names = reg.available_for_stage("enricher")
        # Outside the scope, namespace is gone but global stays.
        outside_names = reg.available_for_stage("enricher")
        assert "only_global" in names
        assert "only_ns" in names
        assert "only_global" in outside_names
        assert "only_ns" not in outside_names

    def test_available_dedup_when_both_layers(self) -> None:
        g = _adapter_class("g")
        n = _adapter_class("n")
        reg.register_adapter(stage="enricher", name="shared")(g)
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="shared")(n)
            names = reg.available_for_stage("enricher")
        assert names.count("shared") == 1

    def test_all_adapters_groups_by_stage(self) -> None:
        e_cls = _adapter_class("e")
        w_cls = _adapter_class("w")
        reg.register_adapter(stage="enricher", name="e_global")(e_cls)
        reg.register_adapter(stage="walker", name="w_global")(w_cls)
        snapshot = reg.all_adapters()
        assert "e_global" in snapshot["enricher"]
        assert "w_global" in snapshot["walker"]

    def test_all_adapters_includes_namespace(self) -> None:
        n_cls = _adapter_class("n")
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="walker", name="ns_only")(n_cls)
            snap = reg.all_adapters()
        assert "ns_only" in snap["walker"]


# ---------------------------------------------------------------------------
# drop_namespace
# ---------------------------------------------------------------------------


class TestDropNamespace:
    def test_drops_existing_namespace(self) -> None:
        cls = _adapter_class("d1")
        with reg.namespace_scope("ns"):
            reg.register_adapter(stage="enricher", name="d1")(cls)
        assert "ns" in reg._NAMESPACED_REGISTRIES
        reg.drop_namespace("ns")
        assert "ns" not in reg._NAMESPACED_REGISTRIES

    def test_drop_missing_is_noop(self) -> None:
        # No exception, even though the namespace was never created.
        reg.drop_namespace("never_existed")


# ---------------------------------------------------------------------------
# get_active_namespace / get_registry
# ---------------------------------------------------------------------------


class TestReaders:
    def test_get_active_namespace_default_none(self) -> None:
        assert reg.get_active_namespace() is None

    def test_get_registry_none_returns_global(self) -> None:
        assert reg.get_registry(None) is reg._GLOBAL_REGISTRY

    def test_get_registry_creates_namespace_lazily(self) -> None:
        assert "fresh-ns" not in reg._NAMESPACED_REGISTRIES
        d = reg.get_registry("fresh-ns")
        assert "fresh-ns" in reg._NAMESPACED_REGISTRIES
        # Returns the same dict on subsequent calls.
        assert reg.get_registry("fresh-ns") is d


# ---------------------------------------------------------------------------
# _reset_for_tests
# ---------------------------------------------------------------------------


class TestResetForTests:
    def test_clears_global_and_namespaces(self) -> None:
        cls = _adapter_class("r")
        reg.register_adapter(stage="enricher", name="r_global")(cls)
        with reg.namespace_scope("rns"):
            reg.register_adapter(stage="enricher", name="r_ns")(cls)
        assert reg._GLOBAL_REGISTRY
        assert reg._NAMESPACED_REGISTRIES
        reg._reset_for_tests()
        assert not reg._GLOBAL_REGISTRY
        assert not reg._NAMESPACED_REGISTRIES
