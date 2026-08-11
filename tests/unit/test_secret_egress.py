"""WIN-340: secret scanning at the search_code MCP egress boundary."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from stropha.models import SearchHit
from stropha.secret_egress import REDACTED_SNIPPET, detect_secret_rules


@pytest.mark.parametrize(
    ("snippet", "rule"),
    [
        ("token = '" + "ghp_" + "A" * 36 + "'", "github-token"),
        ("key = '" + "AKIA" + "A" * 16 + "'", "aws-access-key"),
        ("-----BEGIN PRIVATE KEY-----", "private-key"),
        ("dsn = 'postgresql://service:real-password@db/app'", "credential-url"),
        ("api_key = 'correct-horse-battery-staple'", "sensitive-assignment"),
    ],
)
def test_detects_gitleaks_style_signatures(snippet: str, rule: str) -> None:
    assert rule in detect_secret_rules(snippet)


def test_detects_high_entropy_quoted_token() -> None:
    value = "aB3dE5fG7hJ9kL2mN4pQ6rS8tU1vW3xY5z"
    assert "high-entropy-token" in detect_secret_rules(f'payload = "{value}"')


@pytest.mark.parametrize(
    "snippet",
    [
        "api_key = os.getenv('API_KEY')",
        "client_secret: SecretStr = Field(default='')",
        "token: Optional[str] = None",
        "api_key = 'your-api-key'",
        "token = 'test-token'",
        "def calculate_total(items): return sum(items)",
        "digest = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'",
    ],
)
def test_ignores_placeholders_references_and_normal_code(snippet: str) -> None:
    assert detect_secret_rules(snippet) == ()


def _hit(chunk_id: str, snippet: str) -> SearchHit:
    return SearchHit(
        rank=1,
        score=1.0,
        rel_path="src/settings.py",
        language="python",
        kind="file",
        start_line=1,
        end_line=1,
        snippet=snippet,
        chunk_id=chunk_id,
    )


def _context(hits: list[SearchHit]) -> MagicMock:
    engine = MagicMock()
    engine.search.return_value = hits
    storage = MagicMock()
    storage.lookup_repo_id.return_value = 1
    storage.get_chunk_by_id.side_effect = lambda chunk_id: {
        "chunk_id": chunk_id,
        "repo_id": 1,
    }
    app = SimpleNamespace(
        config=SimpleNamespace(target_repo=Path.cwd()),
        search_engine=engine,
        storage=storage,
        pinned_chunk_ids=set(),
        seen_chunk_ids=set(),
        recent_queries=deque(maxlen=5),
        session_id="",
        client_id="",
        trace_recorder=None,
    )
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


def test_search_code_redacts_suspect_snippet_before_response() -> None:
    from stropha.server import search_code

    value = "aB3dE5fG7hJ9kL2mN4pQ6rS8tU1vW3xY5z"
    suspect = f'client_secret = "{value}"'
    ctx = _context([_hit("secret", suspect), _hit("safe", "def safe(): pass")])

    with patch("stropha.server.log") as mock_log:
        response = search_code(query="settings", ctx=ctx)

    assert response.redacted_count == 1
    assert response.results[0].snippet == REDACTED_SNIPPET
    assert response.results[1].snippet == "def safe(): pass"
    assert value not in response.model_dump_json()

    warning = next(
        call
        for call in mock_log.warning.call_args_list
        if call.args == ("mcp.search_code.secret_redacted",)
    )
    assert value not in repr(warning)
    assert warning.kwargs["chunk_id"] == "secret"
    assert set(warning.kwargs) == {"chunk_id", "path", "rules"}
