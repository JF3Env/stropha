"""Behavioral tests for the shared MCP daemon launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "bin" / "stropha-mcp-ensure"


def _fake_docker(bin_dir: Path, marker: Path) -> None:
    docker = bin_dir / "docker"
    docker.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    docker.chmod(0o755)


def test_postgres_container_is_disabled_by_default(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "docker-called"
    _fake_docker(bin_dir, marker)
    config = tmp_path / "config.yaml"
    config.write_text("pipeline:\n  storage:\n    adapter: postgres\n")

    result = subprocess.run(
        ["sh", str(SCRIPT), "--status"],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STROPHA_CONFIG_FILE": str(config),
            "STROPHA_MCP_PORT": "57991",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert not marker.exists()
    assert "[stropha-pg]" not in result.stdout


def test_postgres_container_management_is_explicitly_opt_in(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "docker-called"
    _fake_docker(bin_dir, marker)

    result = subprocess.run(
        ["sh", str(SCRIPT), "--status"],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STROPHA_USE_POSTGRES": "1",
            "STROPHA_MCP_PORT": "57992",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert marker.exists()
    assert "[stropha-pg] not running" in result.stdout


def test_launcher_tracks_the_real_listener_after_uv_handoff() -> None:
    source = SCRIPT.read_text()

    ready_branch = source.split("for i in $(seq 1 30); do", maxsplit=1)[1]
    assert "_track_listener_pid || true" in ready_branch
    assert 'echo "$1" > "$PIDFILE"' in source
    assert "if ! _wait_for_port_release" in source
    status_branch = source.split("_status() {", maxsplit=1)[1]
    assert "if ! _is_running" in status_branch
    assert "&& _mcp_responsive" in status_branch
