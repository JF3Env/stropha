"""Project initialization for stropha.

The `stropha init` command sets up a new project for use with stropha:
1. Creates `.stropha/` directory with default `.strophaignore`
2. Detects MCP client and offers to configure it
3. Optionally runs first index
4. Prints next steps (hook install, graphify)

Supported MCP clients:
- Claude Code / Claude Desktop: .mcp.json or ~/.config/claude/mcp.json
- Cursor: ~/.cursor/mcp.json
- OpenCode: opencode.json or ~/.config/opencode/config.json
- Continue: ~/.continue/config.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class MCPClient:
    """Metadata about an MCP client."""
    
    name: str
    config_paths: list[Path]  # Possible config file locations
    key: str  # Key in the config file where servers go
    
    def find_config(self) -> Path | None:
        """Find the first existing config file."""
        for p in self.config_paths:
            if p.exists():
                return p
        return None


# Known MCP clients and their config locations
MCP_CLIENTS = [
    MCPClient(
        name="Claude Code",
        config_paths=[
            Path.cwd() / ".mcp.json",
            Path.home() / ".config" / "claude" / "mcp.json",
        ],
        key="mcpServers",
    ),
    MCPClient(
        name="Cursor",
        config_paths=[
            Path.home() / ".cursor" / "mcp.json",
        ],
        key="mcpServers",
    ),
    MCPClient(
        name="OpenCode",
        config_paths=[
            Path.cwd() / "opencode.json",
            Path.home() / ".config" / "opencode" / "config.json",
        ],
        key="mcp",
    ),
    MCPClient(
        name="Continue",
        config_paths=[
            Path.home() / ".continue" / "config.json",
        ],
        key="mcpServers",
    ),
]


DEFAULT_STROPHAIGNORE = """\
# stropha ignore patterns (like .gitignore)
# Files matching these patterns will not be indexed

# Build outputs
build/
dist/
*.pyc
__pycache__/
*.egg-info/
node_modules/
target/

# IDE/editor files
.idea/
.vscode/
*.swp
*.swo
*~

# Environment files (secrets)
.env
.env.*
*.pem
*.key

# Large/binary files
*.zip
*.tar.gz
*.jar
*.war
*.dll
*.exe
*.so
*.dylib

# Generated files
*.min.js
*.min.css
*.map
coverage/
.coverage
htmlcov/
"""


def create_stropha_dir(target: Path) -> tuple[bool, list[str]]:
    """Create .stropha/ directory with default config.
    
    Returns:
        Tuple of (created, messages)
    """
    stropha_dir = target / ".stropha"
    messages: list[str] = []
    created = False
    
    if not stropha_dir.exists():
        stropha_dir.mkdir(parents=True)
        messages.append(f"Created {stropha_dir}/")
        created = True
    
    # Create .strophaignore if it doesn't exist
    ignore_file = target / ".strophaignore"
    if not ignore_file.exists():
        ignore_file.write_text(DEFAULT_STROPHAIGNORE)
        messages.append(f"Created {ignore_file}")
        created = True
    else:
        messages.append(f"Found existing {ignore_file}")
    
    return created, messages


def detect_mcp_clients() -> list[MCPClient]:
    """Find installed MCP clients."""
    found: list[MCPClient] = []
    for client in MCP_CLIENTS:
        if client.find_config():
            found.append(client)
    return found


DEFAULT_MCP_URL = "http://127.0.0.1:9741/mcp"


def generate_mcp_entry(
    client: MCPClient,
    url: str = DEFAULT_MCP_URL,
) -> dict[str, Any]:
    """Generate the HTTP-remote MCP entry for ``client``.

    stropha-mcp is HTTP-only: every client points at the same shared
    singleton (started by ``bin/stropha-mcp-ensure``). The entry shape
    differs by client schema:

    - OpenCode (``key="mcp"``):
      ``{"type": "remote", "url": ..., "enabled": true}``
    - Claude Code / Cursor / Continue (``key="mcpServers"``):
      ``{"url": ..., "transport": "streamable-http"}``
    """
    if client.key == "mcp":
        return {"type": "remote", "url": url, "enabled": True}
    return {"url": url, "transport": "streamable-http"}


def configure_mcp_client(
    client: MCPClient,
    server_name: str = "stropha_rag",
    url: str = DEFAULT_MCP_URL,
) -> tuple[bool, str]:
    """Add stropha (HTTP remote) to an MCP client's configuration.

    Returns:
        Tuple of (success, message)
    """
    config_path = client.find_config()
    if not config_path:
        # Create new config file in first location
        config_path = client.config_paths[0]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {}
    else:
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Failed to read {config_path}: {e}"
    
    # Get or create the servers section
    servers = config.setdefault(client.key, {})
    
    # Check if already configured
    if server_name in servers:
        return False, f"{server_name} already configured in {config_path}"
    
    # Add stropha entry (HTTP remote)
    servers[server_name] = generate_mcp_entry(client, url)
    
    # Write back
    try:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        return True, f"Added {server_name} to {config_path}"
    except OSError as e:
        return False, f"Failed to write {config_path}: {e}"


def print_next_steps(target: Path, indexed: bool) -> list[str]:
    """Generate next steps guidance."""
    steps: list[str] = []

    if not indexed:
        steps.append(
            f"Run initial index: stropha index --repo {target}"
        )

    steps.append(
        "Start the shared MCP server: bin/stropha-mcp-ensure"
    )

    steps.append(
        "Install post-commit hook: stropha hook install"
    )

    steps.append(
        "Generate symbol graph (optional): graphify ."
    )

    steps.append(
        "Search: stropha search \"how does authentication work\""
    )

    return steps
