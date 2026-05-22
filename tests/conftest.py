"""Pytest fixtures and configuration for stropha tests.

This module sets up common test fixtures including:
- Config cache reset (ensures tests don't share config state)
- Inference backend reset
- Isolated working directory (so tests don't pick up stropha.yaml from project root)
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_test_env(tmp_path, monkeypatch):
    """Isolate tests from the project's stropha.yaml.
    
    This changes cwd to a temp directory so tests get default config values
    instead of project-specific settings.
    """
    # Reset config cache before changing directory
    from stropha.config import reset_config
    reset_config()
    
    # Change to temp directory
    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    
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
