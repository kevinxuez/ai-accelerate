"""Small Anthropic JSON client used only for bounded judgment tasks.

The ingestion passes ask the model for indices or labels, never reproduced evidence.
Keeping the HTTP client here also lets the core app run without an SDK installation.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


class LLMUnavailable(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


@dataclass
class AnthropicJSONClient:
    api_key: str | None
    model: str = "claude-sonnet-4-6"
    timeout: int = 90

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4000,
        schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not configured")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
            ).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailable(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Model did not return valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise LLMResponseError("Model JSON must be an object")
        if schema is not None:
            try:
                return schema.model_validate(value).model_dump(mode="json")
            except ValidationError as exc:
                raise LLMResponseError(
                    f"Model JSON failed the {schema.__name__} schema"
                ) from exc
        return value
