"""Typed, public-safe application errors for the CaseFile runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    CAPABILITY_DISABLED = "CAPABILITY_DISABLED"
    AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_INVALID = "REQUEST_INVALID"
    STATE_LIMIT_EXCEEDED = "STATE_LIMIT_EXCEEDED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_CORRUPT = "SESSION_CORRUPT"
    SESSION_VERSION_UNSUPPORTED = "SESSION_VERSION_UNSUPPORTED"
    MODEL_CONFIGURATION_ERROR = "MODEL_CONFIGURATION_ERROR"
    MODEL_UPSTREAM_ERROR = "MODEL_UPSTREAM_ERROR"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    AGENT_OUTPUT_INVALID = "AGENT_OUTPUT_INVALID"
    AGENT_STEP_LIMIT_EXCEEDED = "AGENT_STEP_LIMIT_EXCEEDED"
    DOCUMENT_PARSE_FAILED = "DOCUMENT_PARSE_FAILED"
    DOCUMENT_UNSAFE = "DOCUMENT_UNSAFE"
    INGESTION_BOUNDARY_INVALID = "INGESTION_BOUNDARY_INVALID"
    INGESTION_CARD_INVALID = "INGESTION_CARD_INVALID"
    INGESTION_SOURCE_CHANGED = "INGESTION_SOURCE_CHANGED"
    CONFIRMATION_INVALID = "CONFIRMATION_INVALID"
    STORAGE_READ_FAILED = "STORAGE_READ_FAILED"
    STORAGE_WRITE_FAILED = "STORAGE_WRITE_FAILED"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    RETRIEVAL_INDEX_MISMATCH = "RETRIEVAL_INDEX_MISMATCH"
    INDEX_REBUILD_FAILED = "INDEX_REBUILD_FAILED"
    ARGUMENT_VALIDATION_FAILED = "ARGUMENT_VALIDATION_FAILED"
    NSDA_UPSTREAM_ERROR = "NSDA_UPSTREAM_ERROR"
    CALENDAR_NOT_CONFIGURED = "CALENDAR_NOT_CONFIGURED"
    CALENDAR_UPSTREAM_ERROR = "CALENDAR_UPSTREAM_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


HTTP_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.CONFIGURATION_ERROR: 503,
    ErrorCode.CAPABILITY_DISABLED: 503,
    ErrorCode.AUTHORIZATION_DENIED: 403,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.REQUEST_INVALID: 422,
    ErrorCode.STATE_LIMIT_EXCEEDED: 422,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_CORRUPT: 409,
    ErrorCode.SESSION_VERSION_UNSUPPORTED: 409,
    ErrorCode.MODEL_CONFIGURATION_ERROR: 503,
    ErrorCode.MODEL_UPSTREAM_ERROR: 502,
    ErrorCode.MODEL_TIMEOUT: 504,
    ErrorCode.MODEL_OUTPUT_INVALID: 502,
    ErrorCode.AGENT_OUTPUT_INVALID: 502,
    ErrorCode.AGENT_STEP_LIMIT_EXCEEDED: 500,
    ErrorCode.DOCUMENT_PARSE_FAILED: 422,
    ErrorCode.DOCUMENT_UNSAFE: 403,
    ErrorCode.INGESTION_BOUNDARY_INVALID: 422,
    ErrorCode.INGESTION_CARD_INVALID: 422,
    ErrorCode.INGESTION_SOURCE_CHANGED: 409,
    ErrorCode.CONFIRMATION_INVALID: 409,
    ErrorCode.STORAGE_READ_FAILED: 500,
    ErrorCode.STORAGE_WRITE_FAILED: 500,
    ErrorCode.RETRIEVAL_UNAVAILABLE: 503,
    ErrorCode.RETRIEVAL_INDEX_MISMATCH: 500,
    ErrorCode.INDEX_REBUILD_FAILED: 500,
    ErrorCode.ARGUMENT_VALIDATION_FAILED: 422,
    ErrorCode.NSDA_UPSTREAM_ERROR: 502,
    ErrorCode.CALENDAR_NOT_CONFIGURED: 503,
    ErrorCode.CALENDAR_UPSTREAM_ERROR: 502,
    ErrorCode.INTERNAL_ERROR: 500,
}

RETRYABLE_CODES = {
    ErrorCode.MODEL_UPSTREAM_ERROR,
    ErrorCode.MODEL_TIMEOUT,
    ErrorCode.RETRIEVAL_UNAVAILABLE,
    ErrorCode.NSDA_UPSTREAM_ERROR,
    ErrorCode.CALENDAR_UPSTREAM_ERROR,
}


class ErrorDetail(BaseModel):
    """The error object exposed by API responses and persisted graph state."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: ErrorCode
    message: str = Field(min_length=1, max_length=1000)
    stage: str = Field(min_length=1, max_length=200)
    agent: str | None = Field(default=None, max_length=100)
    tool: str | None = Field(default=None, max_length=100)
    retryable: bool
    request_id: str | None = Field(default=None, max_length=200, exclude=True)
    details: dict[str, Any] = Field(default_factory=dict)


class CaseFileError(Exception):
    """An expected application failure with a stable public contract.

    ``safe_details`` must contain only values that are safe to return to the caller.
    The originating exception type is retained for logs but is never emitted by
    :meth:`public_detail` or :meth:`public_envelope`.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        stage: str,
        agent: str | None = None,
        tool: str | None = None,
        retryable: bool | None = None,
        request_id: str | None = None,
        safe_details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message
        self.http_status = http_status or HTTP_STATUS_BY_CODE[self.code]
        self.stage = stage
        self.agent = agent
        self.tool = tool
        self.retryable = (
            self.code in RETRYABLE_CODES if retryable is None else retryable
        )
        self.request_id = request_id
        self.safe_details = dict(safe_details or {})
        self.cause_type = type(cause).__name__ if cause is not None else None
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause

    def public_detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=self.message,
            stage=self.stage,
            agent=self.agent,
            tool=self.tool,
            retryable=self.retryable,
            request_id=self.request_id,
            details=self.safe_details,
        )

    def public_envelope(
        self,
        *,
        request_id: str,
        session_id: str | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        model_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        detail = self.public_detail().model_dump(mode="json", exclude={"request_id"})
        return {
            "status": "failed",
            "request_id": request_id,
            "session_id": session_id,
            "error": detail,
            "agent_trace": list(agent_trace or []),
            "tool_trace": list(tool_trace or []),
            "model_trace": list(model_trace or []),
        }

    def with_request_id(self, request_id: str) -> "CaseFileError":
        if self.request_id is None:
            self.request_id = request_id
        return self
