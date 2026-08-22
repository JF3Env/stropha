from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from stropha.config.models import MCP_PROFILES, McpProfile
from stropha.server import capabilities, mcp
from stropha.server_tool_manifest import MCP_TOOL_MANIFEST, tool_names_for_profile

_FULL_COMPATIBILITY_TOOLS = {
    "apply_task_recipe",
    "assemble_context",
    "assert_snippet_freshness_tool",
    "audit_citations_tool",
    "blast_radius_tool",
    "bootstrap_environment",
    "cache_prune",
    "cache_stats",
    "check_staleness",
    "check_symbols_tool",
    "co_occurrence",
    "contract_tests_tool",
    "execute_investigation",
    "expand_chunk",
    "expand_chunks",
    "find_callers",
    "find_rationale",
    "find_related",
    "find_tests_for",
    "forget",
    "get_community",
    "get_config",
    "get_file_outline",
    "get_symbol",
    "index_health",
    "list_pins",
    "list_repos",
    "list_task_recipes",
    "observe_agent_state",
    "pin_chunks",
    "plan_investigation",
    "recall",
    "recall_episodes",
    "record_episode",
    "record_runtime_artifact",
    "remember",
    "report_bad_result",
    "reset_dedup",
    "scratchpad_append",
    "scratchpad_clear",
    "scratchpad_read",
    "scratchpad_write",
    "search_code",
    "smart_search_tool",
    "spec_consistency_tool",
    "suggest_next_tool_mcp",
    "test_impact_tool",
    "trace_feature",
    "unpin_chunks",
    "verify_claim_tool",
    "working_set_tool",
}

_CORE_TOOLS = {
    "assemble_context",
    "bootstrap_environment",
    "capabilities",
    "find_callers",
    "find_rationale",
    "find_related",
    "find_tests_for",
    "get_file_outline",
    "get_symbol",
    "index_health",
    "list_repos",
    "search_code",
}

_STANDARD_ADDITIONAL_TOOLS = {
    "apply_task_recipe",
    "assert_snippet_freshness_tool",
    "audit_citations_tool",
    "blast_radius_tool",
    "check_symbols_tool",
    "contract_tests_tool",
    "execute_investigation",
    "expand_chunk",
    "expand_chunks",
    "get_community",
    "list_pins",
    "list_task_recipes",
    "observe_agent_state",
    "pin_chunks",
    "plan_investigation",
    "reset_dedup",
    "record_runtime_artifact",
    "smart_search_tool",
    "spec_consistency_tool",
    "suggest_next_tool_mcp",
    "test_impact_tool",
    "trace_feature",
    "unpin_chunks",
    "verify_claim_tool",
    "working_set_tool",
}

_NEW_STRUCTURED_TOOLS = {
    "observe_agent_state": "AgentObservationResponse",
    "record_runtime_artifact": "RuntimeArtifactResponse",
    "find_callers": "FindCallersResponse",
    "find_related": "FindRelatedResponse",
    "get_community": "CommunityResponse",
    "find_rationale": "FindRationaleResponse",
    "find_tests_for": "FindTestsResponse",
    "trace_feature": "TraceFeatureResponse",
    "report_bad_result": "BadResultResponse",
    "get_config": "RuntimeConfigResponse",
    "index_health": "IndexHealthResponse",
}

_STRUCTURED_REQUIRED_KEYS = {
    "observe_agent_state": {"snapshot", "runtime_artifacts"},
    "record_runtime_artifact": {
        "artifact",
        "stored_artifacts",
        "stored_content_bytes",
    },
    "find_callers": {"symbol"},
    "find_related": {"symbol"},
    "get_community": {"query"},
    "find_rationale": {"symbol"},
    "find_tests_for": {"symbol"},
    "trace_feature": {"feature"},
    "report_bad_result": {"ok"},
}


@pytest.fixture(autouse=True)
def _restore_profile():
    original = mcp.tool_profile
    yield
    mcp.set_tool_profile(original)


def _listed_tools(profile: McpProfile):
    mcp.set_tool_profile(profile)
    return asyncio.run(mcp.list_tools())


def test_manifest_exactly_matches_registration_order() -> None:
    registered = tuple(tool.name for tool in mcp._tool_manager.list_tools())
    declared = tuple(entry.name for entry in MCP_TOOL_MANIFEST)

    assert registered == declared


def test_profiles_are_nested_and_full_preserves_compatibility_surface() -> None:
    surfaces = {
        profile: {tool.name for tool in _listed_tools(profile)}
        for profile in MCP_PROFILES
    }

    assert surfaces["core"] == _CORE_TOOLS
    assert surfaces["standard"] == _CORE_TOOLS | _STANDARD_ADDITIONAL_TOOLS
    assert surfaces["core"] < surfaces["standard"] < surfaces["full"]
    assert surfaces["full"] == _FULL_COMPATIBILITY_TOOLS | {"capabilities"}
    assert len(surfaces["core"]) <= 14


def test_core_tools_list_stays_within_context_budget() -> None:
    tools = _listed_tools("core")
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "tools-1",
            "result": {
                "tools": [
                    tool.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for tool in tools
                ]
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    assert len(payload.encode("utf-8")) <= 40 * 1024
    assert len(payload) <= 99_451 / 2


def test_capabilities_reports_only_active_surface() -> None:
    mcp.set_tool_profile("core")

    response = capabilities()

    assert response.active_profile == "core"
    assert response.available_profiles == list(MCP_PROFILES)
    assert response.tool_count == len(_CORE_TOOLS)
    assert set(response.tools) == _CORE_TOOLS
    assert response.profile_is_static is True


def test_hidden_tool_cannot_be_invoked_directly() -> None:
    mcp.set_tool_profile("core")

    with pytest.raises(ToolError, match="not available in MCP profile 'core'"):
        asyncio.run(mcp.call_tool("remember", {}))


def test_every_registered_tool_has_a_structured_output_schema() -> None:
    tools = {tool.name: tool for tool in _listed_tools("full")}

    assert all(tool.outputSchema is not None for tool in tools.values())
    for name, title in _NEW_STRUCTURED_TOOLS.items():
        assert tools[name].outputSchema is not None
        assert tools[name].outputSchema["title"] == title
    for name, required in _STRUCTURED_REQUIRED_KEYS.items():
        assert set(tools[name].outputSchema.get("required", [])) == required


def test_manifest_profile_helper_matches_discovery() -> None:
    for profile in MCP_PROFILES:
        assert tuple(tool.name for tool in _listed_tools(profile)) == tool_names_for_profile(
            profile
        )


def test_main_selects_configured_profile_before_server_start() -> None:
    from stropha.server import main

    original_host = mcp.settings.host
    original_port = mcp.settings.port

    def assert_profile_at_run(*, transport: str) -> None:
        assert transport == "streamable-http"
        assert mcp.tool_profile == "core"

    try:
        with (
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=SimpleNamespace(host="127.0.0.1", port=9741),
            ),
            patch(
                "stropha.server.get_config",
                return_value=SimpleNamespace(mcp=SimpleNamespace(profile="core")),
            ),
            patch.object(mcp, "run", side_effect=assert_profile_at_run) as run,
        ):
            main()

        run.assert_called_once_with(transport="streamable-http")
    finally:
        mcp.settings.host = original_host
        mcp.settings.port = original_port
