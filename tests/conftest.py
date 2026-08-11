"""Pytest fixtures and configuration for stropha tests.

This module sets up common test fixtures including:
- Config cache reset (ensures tests don't share config state)
- Inference backend reset
- Isolated HOME (so tests don't pick up ~/.stropha/config.yaml from the
  developer's machine — see docstring on isolated_test_env for the rationale)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_test_env(tmp_path, monkeypatch):
    """Isolate tests from the user's ``~/.stropha/config.yaml``.

    Pre-2026-05 the fixture only changed to ``tmp_path`` but left the
    developer's user configuration visible. Tests like
    ``test_auto_detect_never_picks_deepseek`` that asserted "no YAML present"
    were silently broken whenever the developer had `inference.backend: …`
    set in their personal config. Patching ``Path.home()`` closes that hole.

    The replacement HOME is the test's ``tmp_path``, so a test that needs
    to write a user-level config can do::

        (tmp_path / ".stropha").mkdir()
        (tmp_path / ".stropha" / "config.yaml").write_text("...")
    """
    # Reset config cache before changing directory
    from stropha.config import reset_config

    reset_config()

    # Change to temp directory + isolate HOME from real ~/.stropha/
    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Pathlib's Path.home() consults HOME on POSIX but caches via os.path.expanduser.
    # On macOS / Linux this monkeypatch is enough; pathlib re-reads HOME per call.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    yield tmp_path

    # Restore original directory and reset config
    monkeypatch.chdir(original_cwd)
    reset_config()


@pytest.fixture(autouse=True)
def reset_inference_backend():
    """Reset cached inference backend before each test."""
    from stropha.inference.backend import reset_backend

    reset_backend()

    yield

    reset_backend()
