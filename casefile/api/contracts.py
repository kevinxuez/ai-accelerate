"""Transport contracts for the four-agent API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from casefile.agents.contracts import (
    ActiveGoal,
    AgentName,
    AgentTraceEntry,
    Artifact,
    ModelTraceEntry,
    StrictContract,
    ToolTraceEntry,
)
from casefile.agents.errors import ErrorDetail


class StrictRequest(StrictContract):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ChatRequest(StrictRequest):
    message: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    role: Literal["student", "coach"]
    user_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    idempotency_key: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    ) = None
    session_id: (
        Annotated[
            str,
            StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{15,127}$"),
        ]
        | None
    ) = None


class IngestionConfirmRequest(StrictRequest):
    confirmation_token: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    role: Literal["student", "coach"]
    user_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    idempotency_key: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    ) = None


class QuarantineApprovalRequest(StrictRequest):
    card_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    role: Literal["student", "coach"]
    user_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    idempotency_key: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    ) = None


class CalendarConfirmationRequest(StrictRequest):
    confirmation_token: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    role: Literal["student", "coach"]
    user_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    idempotency_key: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    ) = None


class ChatSuccessResponse(StrictContract):
    status: Literal["running", "needs_input", "needs_confirmation", "completed"]
    response: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    request_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    session_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    active_agent: AgentName
    active_goal: ActiveGoal | None
    awaiting_input: bool
    awaiting_confirmation: bool
    artifacts: list[Artifact] = Field(default_factory=list, max_length=20)
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list, max_length=256)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list, max_length=512)
    model_trace: list[ModelTraceEntry] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def validate_waiting_status(self) -> "ChatSuccessResponse":
        if self.awaiting_input != (self.status == "needs_input"):
            raise ValueError("awaiting_input must match needs_input status")
        if self.awaiting_confirmation != (self.status == "needs_confirmation"):
            raise ValueError(
                "awaiting_confirmation must match needs_confirmation status"
            )
        return self


class ErrorResponse(StrictContract):
    status: Literal["failed"] = "failed"
    request_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    session_id: (
        Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    ) = None
    error: ErrorDetail
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list, max_length=256)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list, max_length=512)
    model_trace: list[ModelTraceEntry] = Field(default_factory=list, max_length=256)
