#!/usr/bin/env python3
"""Generate opencode.json for a target repository to use stropha MCP server.

Usage:
    python generate_opencode_config.py <target_repo_path>
    
Example:
    python generate_opencode_config.py C:\sources\my_project
    
This will create/update C:\sources\my_project\opencode.json with the correct
stropha_rag MCP configuration pointing to that repository's .stropha/index.db.
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    target_repo = Path(sys.argv[1]).resolve()
    if not target_repo.is_dir():
        print(f"Error: {target_repo} is not a directory")
        sys.exit(1)
    
    # Stropha project directory (where this script lives)
    stropha_dir = Path(__file__).parent.parent.resolve()
    
    # Target index path
    index_path = target_repo / ".stropha" / "index.db"
    
    # OpenCode config
    config = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "stropha_rag": {
                "type": "local",
                "command": [
                    "uv",
                    "--directory",
                    str(stropha_dir),
                    "run",
                    "stropha-mcp"
                ],
                "enabled": True,
                "env": {
                    "STROPHA_TARGET_REPO": str(target_repo),
                    "STROPHA_INDEX_PATH": str(index_path),
                    "STROPHA_LOCAL_EMBED_MODEL": "BAAI/bge-small-en-v1.5",
                    "STROPHA_LOG_LEVEL": "WARNING"
                }
            }
        }
    }
    
    # Output path
    output_path = target_repo / "opencode.json"
    
    # If file exists, merge with existing config
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            # Merge MCP configs
            if "mcp" not in existing:
                existing["mcp"] = {}
            existing["mcp"]["stropha_rag"] = config["mcp"]["stropha_rag"]
            config = existing
        except json.JSONDecodeError:
            pass  # Overwrite invalid JSON
    
    # Write config
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created {output_path}")
    print(f"  Target repo: {target_repo}")
    print(f"  Index path:  {index_path}")
    print(f"  Stropha dir: {stropha_dir}")
    print()
    print("Next steps:")
    print(f"  1. Ensure .stropha/index.db exists: cd {target_repo} && uv --directory {stropha_dir} run stropha index")
    print(f"  2. Restart OpenCode to load the MCP server")


if __name__ == "__main__":
    main()
