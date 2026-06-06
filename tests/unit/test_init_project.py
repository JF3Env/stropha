"""Unit tests for stropha init command."""

from __future__ import annotations

import json
from pathlib import Path

from stropha.tools.init_project import (
    MCPClient,
    configure_mcp_client,
    create_stropha_dir,
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
    """Tests for generate_mcp_entry function (HTTP-only)."""

    def test_mcpservers_schema(self, tmp_path: Path):
        client = MCPClient(name="Test", config_paths=[tmp_path], key="mcpServers")
        entry = generate_mcp_entry(client)

        assert entry == {
            "url": "http://127.0.0.1:9741/mcp",
            "transport": "streamable-http",
        }

    def test_opencode_schema(self, tmp_path: Path):
        client = MCPClient(name="Test", config_paths=[tmp_path], key="mcp")
        entry = generate_mcp_entry(client)

        assert entry == {
            "type": "remote",
            "url": "http://127.0.0.1:9741/mcp",
            "enabled": True,
        }

    def test_custom_url(self, tmp_path: Path):
        client = MCPClient(name="Test", config_paths=[tmp_path], key="mcpServers")
        entry = generate_mcp_entry(client, url="http://10.0.0.1:8000/mcp")

        assert entry["url"] == "http://10.0.0.1:8000/mcp"

    def test_no_subprocess_spawning(self, tmp_path: Path):
        """HTTP-only: never spawn stropha-mcp as a subprocess."""
        for key in ("mcpServers", "mcp"):
            client = MCPClient(name="Test", config_paths=[tmp_path], key=key)
            entry = generate_mcp_entry(client)
            assert "command" not in entry
            assert "args" not in entry
            assert "env" not in entry


class TestConfigureMCPClient:
    """Tests for configure_mcp_client function."""

    def test_creates_new_config(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )

        success, _msg = configure_mcp_client(client)

        assert success is True
        assert config_path.is_file()

        config = json.loads(config_path.read_text())
        assert "stropha_rag" in config["mcpServers"]
        assert config["mcpServers"]["stropha_rag"]["url"].startswith("http://")

    def test_updates_existing_config(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text(json.dumps({"mcpServers": {"other": {}}}))

        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )

        success, _msg = configure_mcp_client(client)

        assert success is True
        config = json.loads(config_path.read_text())
        assert "other" in config["mcpServers"]  # Preserved
        assert "stropha_rag" in config["mcpServers"]  # Added

    def test_does_not_overwrite_existing_entry(self, tmp_path: Path):
        config_path = tmp_path / ".mcp.json"
        config_path.write_text(json.dumps({
            "mcpServers": {
                "stropha_rag": {"url": "http://existing/mcp"}
            }
        }))

        client = MCPClient(
            name="Test",
            config_paths=[config_path],
            key="mcpServers",
        )

        success, msg = configure_mcp_client(client)

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

        success, _msg = configure_mcp_client(client)

        assert success is False

    def test_opencode_uses_remote_schema(self, tmp_path: Path):
        config_path = tmp_path / "opencode.json"
        client = MCPClient(name="OpenCode", config_paths=[config_path], key="mcp")

        success, _ = configure_mcp_client(client)

        assert success is True
        config = json.loads(config_path.read_text())
        entry = config["mcp"]["stropha_rag"]
        assert entry["type"] == "remote"
        assert entry["enabled"] is True


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
