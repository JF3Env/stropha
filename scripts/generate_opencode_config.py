#!/usr/bin/env python3
"""Generate opencode.json for a target repository to use stropha MCP server.

stropha-mcp is HTTP-only: every client points at the same shared singleton
(started by ``bin/stropha-mcp-ensure``). This script just writes the remote
MCP entry — there is no per-repo subprocess command, no env block.

Usage:
    python generate_opencode_config.py <target_repo_path> [--url URL]

Example:
    python generate_opencode_config.py ~/sources/my_project
    python generate_opencode_config.py ~/sources/my_project --url http://10.0.0.1:9741/mcp
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_URL = "http://127.0.0.1:9741/mcp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Target repository path")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL of the shared stropha-mcp HTTP server (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    target_repo = args.target.expanduser().resolve()
    if not target_repo.is_dir():
        print(f"Error: {target_repo} is not a directory", file=sys.stderr)
        sys.exit(1)

    entry = {
        "type": "remote",
        "url": args.url,
        "enabled": True,
    }

    output_path = target_repo / "opencode.json"

    config: dict
    if output_path.exists():
        try:
            config = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            config = {}
    else:
        config = {}

    config.setdefault("$schema", "https://opencode.ai/config.json")
    config.setdefault("mcp", {})["stropha_rag"] = entry

    output_path.write_text(json.dumps(config, indent=2) + "\n")

    print(f"Wrote {output_path}")
    print(f"  MCP URL: {args.url}")
    print()
    print("Next steps:")
    print("  1. Start the shared server: bin/stropha-mcp-ensure")
    print("  2. Restart OpenCode to load the MCP server")


if __name__ == "__main__":
    main()
