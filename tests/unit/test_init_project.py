"""Unit tests for stropha init command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.tools.init_project import (
    DEFAULT_STROPHAIGNORE,
    MCPClient,
    configure_mcp_client,
    create_stropha_dir,
    detect_mcp_clients,
    generate_mcp_entry,
    print_next_steps,
)


class TestCreateStrophaDir:
    """Tests for create_stropha_dir function."""

    def test_creates_stropha_dir(self, tmp_path: Path):
        created, messages = create_stropha_dir(tmp_path)
        
        assert created is True
        assert (tmp_path / ".stropha").is_dir()
        assert any(".stropha" in m for m in messages)

    def test_creates_strophaignore(self, tmp_path: Path):
        create_stropha_dir(tmp_path)
        
        ignore_file = tmp_path / ".strophaignore"
        assert ignore_file.is_file()
        content = ignore_file.read_text()
        assert "node_modules" in content
        assert ".env" in content

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        # Create existing .strophaignore with custom content
        (tmp_path / ".strophaignore").write_text("custom content")
        
        create_stropha_dir(tmp_path)
        
        # Should not overwrite
        assert (tmp_path / ".strophaignore").read_text() == "custom content"

    def test_idempotent(self, tmp_path: Path):
        create_stropha_dir(tmp_path)
        created1, _ = create_stropha_dir(tmp_path)
        
        # Second call should not recreate
        assert created1 is False


class TestGenerateMCPEntry:
    """Tests for generate_mcp_entry function."""

    def test_basic_entry(self, tmp_path: Path):
        entry = generate_mcp_entry(tmp_path)
        
        assert entry["command"] == "uvx"
        assert "stropha-mcp" in entry["args"]
        assert "STROPHA_TARGET_REPO" in entry["env"]
        assert "STROPHA_INDEX_PATH" in entry["env"]

    def test_with_stropha_project_dir(self, tmp_path: Path):
        stropha_dir = tmp_path / "stropha"
        stropha_dir.mkdir()
        
        entry = generate_mcp_entry(tmp_path, stropha_project_dir=stropha_dir)
        
        assert entry["command"] == "uv"
        assert "--directory" in entry["args"]
        assert str(stropha_dir.resolve()) in entry["args"]


class TestConfigureMCPClient:
    """Tests for configure_mcp_client function."""

    def test_creates_new_config(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )
        
        success, msg = configure_mcp_client(client, tmp_path)
        
        assert success is True
        assert config_path.is_file()
        
        config = json.loads(config_path.read_text())
        assert "stropha_rag" in config["mcpServers"]

    def test_updates_existing_config(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text(json.dumps({"mcpServers": {"other": {}}}))
        
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )
        
        success, msg = configure_mcp_client(client, tmp_path)
        
        assert success is True
        config = json.loads(config_path.read_text())
        assert "other" in config["mcpServers"]  # Preserved
        assert "stropha_rag" in config["mcpServers"]  # Added

    def test_does_not_overwrite_existing_entry(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text(json.dumps({
            "mcpServers": {
                "stropha_rag": {"command": "original"}
            }
        }))
        
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )
        
        success, msg = configure_mcp_client(client, tmp_path)
        
        assert success is False
        assert "already configured" in msg

    def test_handles_invalid_json(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text("not json")
        
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )
        
        success, msg = configure_mcp_client(client, tmp_path)
        
        assert success is False


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_find_config_exists(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.touch()
        
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )
        
        assert client.find_config() == config_path

    def test_find_config_not_exists(self, tmp_path: Path):
        client = MCPClient(
            name="Test",
            config_paths=[tmp_path / "nonexistent.json"],
            key="mcpServers",
        )
        
        assert client.find_config() is None

    def test_find_config_first_match(self, tmp_path: Path):
        # Create second config but not first
        second = tmp_path / "second.json"
        second.touch()
        
        client = MCPClient(
            name="Test",
            config_paths=[
                tmp_path / "first.json",
                second,
            ],
            key="mcpServers",
        )
        
        assert client.find_config() == second


class TestPrintNextSteps:
    """Tests for print_next_steps function."""

    def test_not_indexed(self, tmp_path: Path):
        steps = print_next_steps(tmp_path, indexed=False)
        
        assert any("stropha index" in s for s in steps)
        assert any("hook install" in s for s in steps)

    def test_indexed(self, tmp_path: Path):
        steps = print_next_steps(tmp_path, indexed=True)
        
        # Should not include index step
        assert not any("stropha index --repo" in s for s in steps)
        assert any("hook install" in s for s in steps)
