"""Versioned, bounded state contract for the four-agent graph."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .contracts import (
    MAX_ARTIFACTS,
    MAX_COACHING_TURNS,
    MAX_GRAPH_STEPS,
    MAX_MESSAGES,
    MAX_PROGRESS_RECORDS,
    ActiveGoal,
    AgentName,
    AgentTraceEntry,
    Artifact,
    ArgumentDraft,
    ArgumentRequest,
    ClarificationRequest,
    CoachingTask,
    ConfirmationRequest,
    ConversationMessage,
    EvidencePacket,
    EvidenceQueryPlan,
    EvidenceRequest,
    ModelTraceEntry,
    ProgressEntry,
    RequestContext,
    RuntimeStatus,
    ScheduleToolCall,
    StrictContract,
    SupervisorDecision,
    ToolTraceEntry,
)
from .errors import ErrorDetail


CASEFILE_STATE_SCHEMA_VERSION = 1


class IngestionJobState(StrictContract):
    job_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    stage: Literal[
        "uploaded",
        "inspected",
        "extracted",
        "screened",
        "awaiting_metadata",
        "segmenting",
        "labeling",
        "preview_staged",
        "awaiting_confirmation",
        "committing",
        "indexing",
        "completed",
        "cancelled",
        "failed",
    ]
    source_filename: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    resolution: Annotated[str, StringConstraints(max_length=500)] | None = None
    side: Literal["pro", "con", "unknown"] | None = None


class CoachingState(StrictContract):
    student_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    speech_position: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    side: Literal["pro", "con"]
    focus: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    turns: list[ConversationMessage] = Field(
        default_factory=list, max_length=MAX_COACHING_TURNS
    )
    evidence_packet: EvidencePacket | None = None


class CaseFileState(StrictContract):
    schema_version: Literal[CASEFILE_STATE_SCHEMA_VERSION] = (
        CASEFILE_STATE_SCHEMA_VERSION
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list, max_length=MAX_MESSAGES
    )
    request: RequestContext
    status: RuntimeStatus = "running"
    active_goal: ActiveGoal | None = None
    active_agent: AgentName = "supervisor"
    supervisor_decision: SupervisorDecision | None = None
    pending_question: ClarificationRequest | None = None
    pending_confirmation: ConfirmationRequest | None = None
    evidence_request: EvidenceRequest | EvidenceQueryPlan | None = None
    evidence_packet: EvidencePacket | None = None
    argument_request: ArgumentRequest | None = None
    argument_draft: ArgumentDraft | None = None
    ingestion_job: IngestionJobState | None = None
    coaching_state: CoachingState | None = None
    coaching_task: CoachingTask | None = None
    schedule_request: ScheduleToolCall | None = None
    progress_context: list[ProgressEntry] = Field(
        default_factory=list,
        max_length=MAX_PROGRESS_RECORDS,
    )
    artifacts: list[Artifact] = Field(default_factory=list, max_length=MAX_ARTIFACTS)
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list, max_length=256)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list, max_length=512)
    model_trace: list[ModelTraceEntry] = Field(default_factory=list, max_length=256)
    error: ErrorDetail | None = None
    step_count: int = Field(default=0, ge=0, le=MAX_GRAPH_STEPS)

    @model_validator(mode="after")
    def validate_status_artifacts(self) -> "CaseFileState":
        if self.status == "needs_input" and self.pending_question is None:
            raise ValueError("needs_input requires pending_question")
        if self.status != "needs_input" and self.pending_question is not None:
            raise ValueError("pending_question requires needs_input status")
        if self.status == "needs_confirmation" and self.pending_confirmation is None:
            raise ValueError("needs_confirmation requires pending_confirmation")
        if (
            self.status != "needs_confirmation"
            and self.pending_confirmation is not None
        ):
            raise ValueError("pending_confirmation requires needs_confirmation status")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed state requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("state error requires failed status")
        return self
