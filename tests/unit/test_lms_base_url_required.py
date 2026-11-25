"""The LMS endpoint comes only from configuration: no built-in URL."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stropha.config import get_config, reset_config
from stropha.errors import InferenceError
from stropha.inference import LmsBackend
from stropha.server_frontend import configuration_catalog


def test_lms_base_url_has_no_builtin_default() -> None:
    reset_config()
    assert get_config().inference.lms_base_url is None


def test_lms_base_url_is_read_from_config(isolated_test_env) -> None:
    cfg_dir = isolated_test_env / ".stropha"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        "inference:\n  lms_base_url: https://lms.example/openai\n"
    )
    reset_config()
    assert get_config().inference.lms_base_url == "https://lms.example/openai"


def test_lms_without_base_url_is_unavailable() -> None:
    reset_config()
    backend = LmsBackend(api_key="secret")
    assert backend.is_available is False
    ok, message = backend.health_check()
    assert ok is False
    assert "inference.lms_base_url" in message


def test_generate_without_base_url_sends_no_request() -> None:
    reset_config()
    backend = LmsBackend(api_key="secret")
    with patch("httpx.Client.post") as post:
        assert backend.generate("hello") is None
        with pytest.raises(InferenceError, match=r"inference\.lms_base_url"):
            backend.generate("hello", raise_errors=True)
    post.assert_not_called()


def test_dashboard_offers_no_default_lms_url() -> None:
    catalog = configuration_catalog()
    fields = [field for section in catalog["sections"] for field in section["fields"]]
    lms_url = next(field for field in fields if field["path"] == "inference.lms_base_url")
    assert lms_url["default"] is None
