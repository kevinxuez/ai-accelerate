from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.config import get_settings
from casefile.llm import (
    AnthropicJSONClient,
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
            "content": [{"type": "text", "text": '{"status": "ready"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 5},
        }
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = AnthropicJSONClient(
        api_key="test-key",
        model="claude-test-deployment",
        base_url="https://example.services.ai.azure.com/anthropic/",
        timeout=12,
    )

    result = client.complete_json(
        system="Return JSON.",
        user="Report status.",
        agent="supervisor",
        prompt_template="supervisor/classify.md",
        prompt_version="1",
    )

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
    trace = client.calls[0]
    assert trace.agent == "supervisor"
    assert trace.prompt_template == "supervisor/classify.md"
    assert trace.prompt_version == "1"
    assert trace.stop_reason == "end_turn"
    assert (trace.input_tokens, trace.output_tokens) == (12, 5)
    assert trace.status == "completed"
    assert trace.prompt_sha256 and trace.response_sha256


def test_anthropic_base_url_rejects_non_https_or_markdown() -> None:
    for base_url in (
        "http://example.com/anthropic",
        "[https://example.com/anthropic](https://example.com/anthropic)",
        "https://example.com/anthropic?api-version=test",
    ):
        with pytest.raises(CaseFileError, match="ANTHROPIC_BASE_URL") as caught:
            _ = AnthropicJSONClient("test-key", base_url=base_url).messages_url
        assert caught.value.code == ErrorCode.MODEL_CONFIGURATION_ERROR


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
