"""Defense-in-depth secret scanning for MCP search responses."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .server_models import SearchResult

REDACTED_SNIPPET = "[REDACTED: potential secret detected]"

_SIGNATURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "aws-access-key",
        re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b"),
    ),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82,255}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,255}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,255}\b")),
    ("stripe-key", re.compile(r"\b[rs]k_live_[A-Za-z0-9]{16,255}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "sendgrid-key",
        re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{7,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    (
        "credential-url",
        re.compile(
            r"\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)"
            r"://[^\s:/@]+:[^\s/@]+@",
            re.IGNORECASE,
        ),
    ),
)

_SENSITIVE_ASSIGNMENT = re.compile(
    r"""
    \b(?:
        api[_-]?key | access[_-]?key | auth[_-]?token | bearer[_-]?token |
        client[_-]?secret | private[_-]?key | password | passwd | secret | token
    )\b
    [ \t]*(?:=|:)[ \t]*
    ["'`]?
    (?P<value>[A-Za-z0-9_./+=$:@-]{8,})
    """,
    re.IGNORECASE | re.VERBOSE,
)
_QUOTED_TOKEN = re.compile(
    r"(?P<quote>[\"'`])(?P<value>[A-Za-z0-9+/=_-]{32,})(?P=quote)"
)
_PLACEHOLDER_MARKERS = (
    "changeme",
    "dummy",
    "example",
    "fake",
    "placeholder",
    "redacted",
    "replace",
    "sample",
    "test",
    "your_",
    "your-",
)


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]+", value))


def _shannon_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _looks_high_entropy(value: str) -> bool:
    if _is_placeholder(value) or re.fullmatch(r"[0-9a-fA-F]+", value):
        return False
    character_classes = sum(
        bool(pattern.search(value))
        for pattern in (re.compile(r"[a-z]"), re.compile(r"[A-Z]"), re.compile(r"[0-9]"))
    )
    return character_classes >= 3 and _shannon_entropy(value) >= 4.3


def detect_secret_rules(text: str) -> tuple[str, ...]:
    """Return matched rule names without retaining or returning secret values."""
    rules: list[str] = []

    for name, pattern in _SIGNATURES:
        if pattern.search(text) and name not in rules:
            rules.append(name)

    for match in _SENSITIVE_ASSIGNMENT.finditer(text):
        value = match.group("value")
        suffix = text[match.end():].lstrip()
        if suffix.startswith(("(", "[", "=", "|")):
            continue
        if value.isidentifier() and not value.islower():
            continue
        if not _is_placeholder(value) and "sensitive-assignment" not in rules:
            rules.append("sensitive-assignment")

    for match in _QUOTED_TOKEN.finditer(text):
        if _looks_high_entropy(match.group("value")):
            rules.append("high-entropy-token")
            break

    return tuple(rules)


def guard_search_results(
    results: list[SearchResult],
    *,
    log: Any,
) -> tuple[list[SearchResult], int]:
    """Replace suspect snippets before MCP serialization and log safe metadata."""
    guarded: list[SearchResult] = []
    redacted_count = 0

    for result in results:
        rules = detect_secret_rules(result.snippet)
        if not rules:
            guarded.append(result)
            continue

        redacted_count += 1
        guarded.append(result.model_copy(update={"snippet": REDACTED_SNIPPET}))
        log.warning(
            "mcp.search_code.secret_redacted",
            chunk_id=result.chunk_id,
            path=result.path,
            rules=list(rules),
        )

    return guarded, redacted_count
