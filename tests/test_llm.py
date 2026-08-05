from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest

from casefile.config import get_settings
from casefile.llm import (
    AnthropicJSONClient,
    LLMUnavailable,
    build_anthropic_client,
)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_complete_json_uses_configured_anthropic_base_url(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        response = {
            "content": [{"type": "text", "text": '{"status": "ready"}'}]
        }
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = AnthropicJSONClient(
        api_key="test-key",
        model="claude-test-deployment",
        base_url="https://example.services.ai.azure.com/anthropic/",
        timeout=12,
    )

    result = client.complete_json(system="Return JSON.", user="Report status.")

    request = captured["request"]
    headers = {name.lower(): value for name, value in request.header_items()}
    body = json.loads(request.data)
    assert result == {"status": "ready"}
    assert request.full_url == (
        "https://example.services.ai.azure.com/anthropic/v1/messages"
    )
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-test-deployment"
    assert captured["timeout"] == 12


def test_anthropic_base_url_rejects_non_https_or_markdown() -> None:
    for base_url in (
        "http://example.com/anthropic",
        "[https://example.com/anthropic](https://example.com/anthropic)",
        "https://example.com/anthropic?api-version=test",
    ):
        with pytest.raises(LLMUnavailable, match="ANTHROPIC_BASE_URL"):
            _ = AnthropicJSONClient("test-key", base_url=base_url).messages_url


def test_build_anthropic_client_copies_settings() -> None:
    settings = replace(
        get_settings(),
        anthropic_api_key="test-key",
        anthropic_base_url="https://example.services.ai.azure.com/anthropic",
        model="claude-test-deployment",
    )

    client = build_anthropic_client(settings)

    assert client.api_key == "test-key"
    assert client.model == "claude-test-deployment"
    assert client.messages_url.endswith("/anthropic/v1/messages")
