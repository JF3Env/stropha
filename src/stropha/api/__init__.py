"""Harness-agnostic API contract (WIN-257).

Modules under ``stropha.api`` define stable JSON shapes that any
harness (MCP, HTTP, CLI) can produce or consume. The contract is
versioned (``context_v1``) so future breaking changes ship as
``context_v2`` without forcing every caller to upgrade in
lock-step.
"""

from __future__ import annotations
