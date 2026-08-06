"""Required Anthropic-compatible JSON client with observable, single-shot calls."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from casefile.agents.errors import CaseFileError, ErrorCode

if TYPE_CHECKING:
    from casefile.config import Settings


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


@dataclass(frozen=True)
class ModelCallRecord:
    agent: str
    model: str
    prompt_template: str
    prompt_version: str
    prompt_sha256: str
    response_sha256: str | None
    schema_name: str
    started_at: datetime
    latency_ms: int
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    status: str
    error_code: str | None
    rendered_system_prompt: str | None = None
    rendered_user_payload: str | None = None
    model_response: str | None = None


@dataclass
class AnthropicJSONClient:
    api_key: str | None
    model: str = "claude-sonnet-4-6"
    base_url: str = "https://api.anthropic.com"
    timeout: float = 90
    expose_prompts: bool = False
    calls: list[ModelCallRecord] = field(default_factory=list, init=False)

    @property
    def messages_url(self) -> str:
        base_url = self.base_url.strip().rstrip("/")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise CaseFileError(
                ErrorCode.MODEL_CONFIGURATION_ERROR,
                "ANTHROPIC_BASE_URL must be an HTTPS base URL without a query or fragment.",
                stage="model.configuration",
            )
        return f"{base_url}/v1/messages"

    def validate_configuration(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise CaseFileError(
                ErrorCode.MODEL_CONFIGURATION_ERROR,
                "ANTHROPIC_API_KEY is not configured.",
                stage="model.configuration",
            )
        if not self.model.strip():
            raise CaseFileError(
                ErrorCode.MODEL_CONFIGURATION_ERROR,
                "CASEFILE_MODEL is not configured.",
                stage="model.configuration",
            )
        if self.timeout <= 0:
            raise CaseFileError(
                ErrorCode.MODEL_CONFIGURATION_ERROR,
                "CASEFILE_MODEL_TIMEOUT_SECONDS must be positive.",
                stage="model.configuration",
            )
        _ = self.messages_url

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4000,
        schema: type[BaseModel] | None = None,
        agent: str = "unassigned",
        prompt_template: str = "inline",
        prompt_version: str = "1",
    ) -> dict[str, Any]:
        """Make exactly one model request and validate its exact JSON response."""

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        schema_name = schema.__name__ if schema is not None else "object"
        raw_response: str | None = None
        payload: dict[str, Any] | None = None
        try:
            self.validate_configuration()
            request = urllib.request.Request(
                self.messages_url,
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
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
            except TimeoutError as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_TIMEOUT,
                    "The configured model timed out.",
                    stage="model.request",
                    agent=agent,
                    cause=exc,
                ) from exc
            except urllib.error.HTTPError as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_UPSTREAM_ERROR,
                    "The configured model returned an HTTP error.",
                    stage="model.request",
                    agent=agent,
                    cause=exc,
                ) from exc
            except urllib.error.URLError as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_UPSTREAM_ERROR,
                    "The configured model could not be reached.",
                    stage="model.request",
                    agent=agent,
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_UPSTREAM_ERROR,
                    "The configured model transport failed.",
                    stage="model.request",
                    agent=agent,
                    cause=exc,
                ) from exc
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model returned an invalid response envelope.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                    cause=exc,
                ) from exc

            if not isinstance(payload, dict):
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model returned an invalid response envelope.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                )
            content = payload.get("content")
            if not isinstance(content, list):
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model response did not contain text content.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                )
            text_blocks = [
                block.get("text")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if not all(isinstance(text, str) for text in text_blocks):
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model returned malformed text content.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                )
            raw_response = "".join(text_blocks).strip()
            try:
                value = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model did not return the required JSON object.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                    cause=exc,
                ) from exc
            if not isinstance(value, dict):
                raise CaseFileError(
                    ErrorCode.MODEL_OUTPUT_INVALID,
                    "The configured model JSON must be an object.",
                    stage="model.response",
                    agent=agent,
                    safe_details={"schema": schema_name},
                )
            if schema is not None:
                try:
                    value = schema.model_validate(value).model_dump(mode="json")
                except ValidationError as exc:
                    raise CaseFileError(
                        ErrorCode.MODEL_OUTPUT_INVALID,
                        "The configured model output did not match its required schema.",
                        stage="model.response",
                        agent=agent,
                        safe_details={"schema": schema_name},
                        cause=exc,
                    ) from exc
        except CaseFileError as error:
            self._record(
                agent=agent,
                prompt_template=prompt_template,
                prompt_version=prompt_version,
                schema_name=schema_name,
                system=system,
                user=user,
                response=raw_response,
                payload=payload,
                started_at=started_at,
                started=started,
                error=error,
            )
            raise

        self._record(
            agent=agent,
            prompt_template=prompt_template,
            prompt_version=prompt_version,
            schema_name=schema_name,
            system=system,
            user=user,
            response=raw_response,
            payload=payload,
            started_at=started_at,
            started=started,
            error=None,
        )
        return value

    def _record(
        self,
        *,
        agent: str,
        prompt_template: str,
        prompt_version: str,
        schema_name: str,
        system: str,
        user: str,
        response: str | None,
        payload: dict[str, Any] | None,
        started_at: datetime,
        started: float,
        error: CaseFileError | None,
    ) -> None:
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        self.calls.append(
            ModelCallRecord(
                agent=agent,
                model=self.model,
                prompt_template=prompt_template,
                prompt_version=prompt_version,
                prompt_sha256=_sha256(f"{system}\n{user}"),
                response_sha256=_sha256(response) if response is not None else None,
                schema_name=schema_name,
                started_at=started_at,
                latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
                stop_reason=(
                    str(payload.get("stop_reason"))
                    if isinstance(payload, dict) and payload.get("stop_reason")
                    else None
                ),
                input_tokens=(
                    _token_count(usage.get("input_tokens"))
                    if isinstance(usage, dict)
                    else None
                ),
                output_tokens=(
                    _token_count(usage.get("output_tokens"))
                    if isinstance(usage, dict)
                    else None
                ),
                status="failed" if error else "completed",
                error_code=error.code.value if error else None,
                rendered_system_prompt=system if self.expose_prompts else None,
                rendered_user_payload=user if self.expose_prompts else None,
                model_response=response if self.expose_prompts else None,
            )
        )


def build_anthropic_client(settings: Settings) -> AnthropicJSONClient:
    return AnthropicJSONClient(
        api_key=settings.anthropic_api_key,
        model=settings.model,
        base_url=settings.anthropic_base_url,
        timeout=settings.model_timeout_seconds,
        expose_prompts=settings.expose_model_prompts,
    )
