"""Canonical MCP tool inventory and static profile membership."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .config.models import MCP_PROFILES, McpProfile
from .errors import ConfigError

_PROFILE_RANK = {profile: rank for rank, profile in enumerate(MCP_PROFILES)}


@dataclass(frozen=True)
class ToolManifestEntry:
    """One registered tool and the smallest profile that exposes it."""

    name: str
    minimum_profile: McpProfile

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("MCP tool manifest names must be non-empty")
        if self.minimum_profile not in MCP_PROFILES:
            choices = ", ".join(MCP_PROFILES)
            raise ConfigError(f"MCP tool profile must be one of: {choices}")


# Registration order is intentional: ``tools/list`` follows this sequence.
MCP_TOOL_MANIFEST: tuple[ToolManifestEntry, ...] = (
    ToolManifestEntry("capabilities", "core"),
    ToolManifestEntry("search_code", "core"),
    ToolManifestEntry("reset_dedup", "standard"),
    ToolManifestEntry("pin_chunks", "standard"),
    ToolManifestEntry("unpin_chunks", "standard"),
    ToolManifestEntry("list_pins", "standard"),
    ToolManifestEntry("smart_search_tool", "standard"),
    ToolManifestEntry("get_symbol", "core"),
    ToolManifestEntry("list_repos", "core"),
    ToolManifestEntry("get_file_outline", "core"),
    ToolManifestEntry("expand_chunk", "standard"),
    ToolManifestEntry("expand_chunks", "standard"),
    ToolManifestEntry("remember", "full"),
    ToolManifestEntry("recall", "full"),
    ToolManifestEntry("forget", "full"),
    ToolManifestEntry("plan_investigation", "standard"),
    ToolManifestEntry("cache_stats", "full"),
    ToolManifestEntry("cache_prune", "full"),
    ToolManifestEntry("scratchpad_read", "full"),
    ToolManifestEntry("scratchpad_write", "full"),
    ToolManifestEntry("scratchpad_append", "full"),
    ToolManifestEntry("scratchpad_clear", "full"),
    ToolManifestEntry("check_staleness", "full"),
    ToolManifestEntry("bootstrap_environment", "core"),
    ToolManifestEntry("record_episode", "full"),
    ToolManifestEntry("recall_episodes", "full"),
    ToolManifestEntry("co_occurrence", "full"),
    ToolManifestEntry("find_callers", "core"),
    ToolManifestEntry("find_related", "core"),
    ToolManifestEntry("get_community", "standard"),
    ToolManifestEntry("find_rationale", "core"),
    ToolManifestEntry("find_tests_for", "core"),
    ToolManifestEntry("trace_feature", "standard"),
    ToolManifestEntry("report_bad_result", "full"),
    ToolManifestEntry("assemble_context", "core"),
    ToolManifestEntry("execute_investigation", "standard"),
    ToolManifestEntry("get_config", "full"),
    ToolManifestEntry("index_health", "core"),
    ToolManifestEntry("suggest_next_tool_mcp", "standard"),
    ToolManifestEntry("list_task_recipes", "standard"),
    ToolManifestEntry("apply_task_recipe", "standard"),
    ToolManifestEntry("verify_claim_tool", "standard"),
    ToolManifestEntry("check_symbols_tool", "standard"),
    ToolManifestEntry("blast_radius_tool", "standard"),
    ToolManifestEntry("assert_snippet_freshness_tool", "standard"),
    ToolManifestEntry("test_impact_tool", "standard"),
    ToolManifestEntry("audit_citations_tool", "standard"),
    ToolManifestEntry("spec_consistency_tool", "standard"),
    ToolManifestEntry("contract_tests_tool", "standard"),
    ToolManifestEntry("working_set_tool", "standard"),
)

_manifest_by_name = {entry.name: entry for entry in MCP_TOOL_MANIFEST}
if len(_manifest_by_name) != len(MCP_TOOL_MANIFEST):
    raise ConfigError("MCP tool manifest contains duplicate names")

MCP_TOOL_MANIFEST_BY_NAME = MappingProxyType(_manifest_by_name)


def tool_names_for_profile(profile: McpProfile) -> tuple[str, ...]:
    """Return manifest-ordered tool names visible in ``profile``."""
    if profile not in MCP_PROFILES:
        choices = ", ".join(MCP_PROFILES)
        raise ConfigError(f"mcp.profile must be one of: {choices}")
    rank = _PROFILE_RANK[profile]
    return tuple(
        entry.name
        for entry in MCP_TOOL_MANIFEST
        if _PROFILE_RANK[entry.minimum_profile] <= rank
    )


def profile_allows_tool(profile: McpProfile, name: str) -> bool:
    """Whether ``name`` is visible and invocable in ``profile``."""
    entry = MCP_TOOL_MANIFEST_BY_NAME.get(name)
    return entry is not None and _PROFILE_RANK[entry.minimum_profile] <= _PROFILE_RANK[profile]


__all__ = [
    "MCP_TOOL_MANIFEST",
    "MCP_TOOL_MANIFEST_BY_NAME",
    "ToolManifestEntry",
    "profile_allows_tool",
    "tool_names_for_profile",
]
