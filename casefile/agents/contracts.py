"""Strict handoff and artifact contracts for the four-agent architecture."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


ShortText = Annotated[str, StringConstraints(min_length=1, max_length=500)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200)]
CardId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AgentName = Literal[
    "supervisor",
    "evidence_librarian",
    "argument_strategist",
    "skills_coach",
]

BoundaryFlag = Literal[
    "no_header",
    "no_tag",
    "no_body",
    "cite_is_bare_url",
    "cite_is_bare_headline",
    "tag_merged_into_cite",
    "cite_body_same_paragraph",
    "pdf_paste_fragmented",
    "text_corrupt",
    "duplicate_source",
]
CardLabelFlag = Literal[
    *BoundaryFlag.__args__,
    "html_entity",
    "no_marking",
    "fully_marked",
    "paraphrase_no_source",
    "do_not_ingest",
    "body_truncated",
]
SpecialistAgent = Literal["evidence_librarian", "argument_strategist", "skills_coach"]
Role = Literal["student", "coach"]
Side = Literal["pro", "con"]
SupportStatus = Literal["supported", "partially_supported", "unsupported"]
RuntimeStatus = Literal[
    "running", "needs_input", "needs_confirmation", "completed", "failed"
]

MAX_MESSAGES = 64
MAX_EVIDENCE_CARDS = 25
MAX_COACHING_TURNS = 32
MAX_PROGRESS_RECORDS = 100
MAX_GRAPH_STEPS = 64
MAX_ARTIFACTS = 20


class StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ModelContract(StrictContract):
    """Tolerant parser for model JSON before deterministic boundary checks."""

    model_config = ConfigDict(extra="ignore", strict=False)

    @model_validator(mode="before")
    @classmethod
    def normalize_side_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if isinstance(side := normalized.get("side"), str):
            side_aliases = {
                "pro": "pro",
                "affirmative": "pro",
                "aff": "pro",
                "con": "con",
                "negative": "con",
                "neg": "con",
            }
            normalized["side"] = side_aliases.get(side.strip().lower(), side)
        return normalized


class ConversationMessage(StrictContract):
    role: Literal["user", "assistant", "system"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    created_at: datetime | None = None


class AttachmentHandle(StrictContract):
    attachment_id: Identifier
    filename: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    media_type: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    size_bytes: int = Field(ge=1, le=10 * 1024 * 1024)
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RequestContext(StrictContract):
    request_id: Identifier
    session_id: Identifier
    role: Role
    user_id: Identifier
    active_resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    attachments: list[AttachmentHandle] = Field(default_factory=list, max_length=4)


class ActiveGoal(StrictContract):
    summary: ShortText
    completion_criteria: list[ShortText] = Field(min_length=1, max_length=12)


class ClarificationRequest(StrictContract):
    question: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    missing_fields: list[
        Annotated[str, StringConstraints(min_length=1, max_length=100)]
    ] = Field(default_factory=list, max_length=12)
    reason_code: Annotated[str, StringConstraints(min_length=1, max_length=100)]


class ConfirmationRequest(StrictContract):
    operation: Literal["commit_ingestion", "schedule_session", "log_assessment"]
    token: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    summary: ShortText
    expires_at: datetime | None = None


class SupervisorDecision(ModelContract):
    action: Literal[
        "delegate", "ask_clarification", "call_schedule", "finish", "refuse"
    ]
    target_agent: SpecialistAgent | None = None
    goal: str | None = None
    task: Annotated[str, StringConstraints(max_length=2000)] | None = None
    required_artifact: (
        Literal[
            "evidence_packet",
            "rule_packet",
            "topic_packet",
            "ingestion_preview",
            "ingestion_commit_result",
            "argument_draft",
            "drill_plan",
            "coach_turn",
            "progress_summary",
            "assessment_proposal",
            "calendar_event",
        ]
        | None
    ) = None
    clarification_question: (
        Annotated[str, StringConstraints(min_length=1, max_length=1000)] | None
    ) = None
    reason_code: Annotated[str, StringConstraints(min_length=1, max_length=100)] = (
        "model_decision"
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "SupervisorDecision":
        if self.action == "delegate":
            if (
                self.target_agent is None
                or self.required_artifact is None
                or not self.task
            ):
                raise ValueError(
                    "delegate requires target_agent, task, and required_artifact"
                )
        elif self.action in {"finish", "refuse"} and not self.task:
            raise ValueError("finish and refuse require a user-facing task")
        elif self.target_agent is not None:
            raise ValueError("target_agent is allowed only for delegate")
        if self.action == "ask_clarification":
            if not self.clarification_question:
                raise ValueError("ask_clarification requires clarification_question")
        elif self.clarification_question is not None:
            raise ValueError(
                "clarification_question is allowed only for ask_clarification"
            )
        return self


class ScheduleToolCall(ModelContract):
    """Typed arguments prepared by the Supervisor for its only tool."""

    student_id: Identifier
    start: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    duration_minutes: int = Field(default=45, ge=15, le=180)
    attendee_email: Annotated[str, StringConstraints(max_length=320)] | None = None
    timezone_name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    confirmation_token: (
        Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")] | None
    ) = None
    idempotency_key: ShortText | None = None


class CoachingTask(ModelContract):
    """Typed Supervisor-to-Coach execution context for a bounded handoff."""

    operation: str | None = None
    student_id: Identifier
    speech_position: ShortText | None = None
    side: Side | None = None
    focus: ShortText | None = None
    needs_evidence: bool = False
    source_files: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=20)

class IngestionMetadataPlan(ModelContract):
    """Librarian validation of user-supplied ingestion metadata."""

    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side | None = None
    clarification_needed: bool
    clarification_question: (
        Annotated[str, StringConstraints(min_length=1, max_length=1000)] | None
    ) = None


class EvidenceRequest(ModelContract):
    request_summary: ShortText
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side
    subject: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    entities: list[ShortText] = Field(default_factory=list, max_length=12)
    source_files: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=20)
    intended_use: Literal["drill", "coaching"]

class EvidenceQueryPlan(ModelContract):
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side | None = None
    subject: Annotated[str, StringConstraints(max_length=1000)] = ""
    entities: list[ShortText] = Field(default_factory=list, max_length=12)
    source_files: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=20)
    queries: list[Annotated[str, StringConstraints(min_length=1, max_length=2000)]] = (
        Field(default_factory=list, max_length=6)
    )
    result_limit: int = Field(default=10, ge=1, le=MAX_EVIDENCE_CARDS)
    clarification_needed: bool
    clarification_question: (
        Annotated[str, StringConstraints(min_length=1, max_length=1000)] | None
    ) = None


class BoundaryCardOutput(ModelContract):
    header: int | None = Field(ge=0)
    tag: list[int] = Field(default_factory=list, max_length=64)
    cite: list[int] = Field(min_length=1, max_length=64)
    body: list[int] = Field(default_factory=list, max_length=128)
    flags: list[BoundaryFlag] = Field(default_factory=list, max_length=16)

    @field_validator("tag", "cite", "body")
    @classmethod
    def paragraph_ids_are_nonnegative(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("paragraph ids must be nonnegative")
        if len(set(values)) != len(values):
            raise ValueError("paragraph ids within a field must be unique")
        return values


class BoundaryOutput(ModelContract):
    cards: list[BoundaryCardOutput] = Field(default_factory=list, max_length=200)
    junk: list[int] = Field(default_factory=list, max_length=5000)

    @field_validator("junk")
    @classmethod
    def junk_ids_are_valid(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("paragraph ids must be nonnegative")
        if len(set(values)) != len(values):
            raise ValueError("junk paragraph ids must be unique")
        return values

    def validate_ids(self, allowed_ids: set[int]) -> None:
        returned = set(self.junk)
        for card in self.cards:
            if card.header is not None:
                returned.add(card.header)
            returned.update(card.tag)
            returned.update(card.cite)
            returned.update(card.body)
        invalid = returned - allowed_ids
        if invalid:
            raise ValueError(
                f"Model returned paragraph ids outside its window: {sorted(invalid)}"
            )


class CardLabelOutput(ModelContract):
    evidence_type: Literal["quoted", "paraphrased", "unknown"]
    source_text_present: bool
    side: Literal["pro", "con", "unknown"]
    topic_tags: list[Annotated[str, StringConstraints(min_length=1, max_length=40)]] = (
        Field(default_factory=list, max_length=6)
    )
    flags: list[CardLabelFlag] = Field(default_factory=list, max_length=20)
    explanation: Annotated[str, StringConstraints(max_length=1000)]

    @model_validator(mode="after")
    def flagged_cards_are_explained(self) -> "CardLabelOutput":
        if self.flags and not self.explanation:
            raise ValueError("flagged card labels require an explanation")
        return self


class TextSpan(StrictContract):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TextSpan":
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")
        return self


class EvidenceCard(StrictContract):
    model_config = ConfigDict(str_strip_whitespace=False)

    card_id: CardId
    source_filename: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    citation: Annotated[str, StringConstraints(min_length=1, max_length=5000)]
    header: Annotated[str, StringConstraints(max_length=2000)]
    tag: Annotated[str, StringConstraints(max_length=2000)]
    body: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    read_spans: list[TextSpan] = Field(default_factory=list, max_length=500)
    emphasis_spans: list[TextSpan] = Field(default_factory=list, max_length=500)
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side
    retrieval_score: float = Field(ge=-1, le=1)

    @field_validator("read_spans", "emphasis_spans")
    @classmethod
    def spans_fit_body(cls, spans: list[TextSpan], info: Any) -> list[TextSpan]:
        body = info.data.get("body", "")
        if body and any(span.end > len(body) for span in spans):
            raise ValueError("span exceeds preserved card body")
        return spans


class EvidenceProvenance(StrictContract):
    ledger_schema_version: int = Field(ge=1)
    retrieval_backend: Literal["in_memory", "chroma"]
    embedding_model: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    confirmed_only: Literal[True]


class EvidencePacket(StrictContract):
    artifact_type: Literal["evidence_packet"] = "evidence_packet"
    request_summary: ShortText
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side | None
    confirmed_source_files_considered: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=100)
    queries_executed: list[
        Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    ] = Field(min_length=1, max_length=6)
    cards: list[EvidenceCard] = Field(
        default_factory=list, max_length=MAX_EVIDENCE_CARDS
    )
    empty_result: bool
    provenance: EvidenceProvenance

    @model_validator(mode="after")
    def validate_packet(self) -> "EvidencePacket":
        if self.empty_result != (not self.cards):
            raise ValueError(
                "empty_result must exactly reflect whether cards are present"
            )
        seen: set[str] = set()
        for card in self.cards:
            if card.card_id in seen:
                raise ValueError("evidence card ids must be unique")
            seen.add(card.card_id)
            if card.resolution != self.resolution:
                raise ValueError("card resolution must match the packet")
            if self.side is not None and card.side != self.side:
                raise ValueError("card side must match the packet")
        return self


class RuleChunk(StrictContract):
    chunk_id: Identifier
    section_number: Annotated[str, StringConstraints(min_length=1, max_length=50)]
    section_title: ShortText
    text: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    document: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    score: float = Field(ge=-1, le=1)


class RulePacket(StrictContract):
    artifact_type: Literal["rule_packet"] = "rule_packet"
    request_summary: ShortText
    event: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    chunks: list[RuleChunk] = Field(default_factory=list, max_length=20)
    empty_result: bool

    @model_validator(mode="after")
    def validate_empty_result(self) -> "RulePacket":
        if self.empty_result != (not self.chunks):
            raise ValueError(
                "empty_result must exactly reflect whether chunks are present"
            )
        return self


class TopicPacket(StrictContract):
    artifact_type: Literal["topic_packet"] = "topic_packet"
    event: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    effective_from: date | None = None
    effective_to: date | None = None
    provider: ShortText
    backend: Literal["http", "fixture"]
    synthetic: bool
    source_ref: Annotated[str, StringConstraints(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def fixture_is_synthetic(self) -> "TopicPacket":
        if self.backend == "fixture" and not self.synthetic:
            raise ValueError("fixture topic packets must be labeled synthetic")
        return self


class IngestionPreviewCard(StrictContract):
    card_id: CardId
    header: Annotated[str, StringConstraints(max_length=2000)]
    tag: Annotated[str, StringConstraints(max_length=2000)]
    citation: Annotated[str, StringConstraints(max_length=5000)]
    body: Annotated[str, StringConstraints(max_length=100_000)]
    read_spans: list[TextSpan] = Field(default_factory=list, max_length=500)
    emphasis_spans: list[TextSpan] = Field(default_factory=list, max_length=500)
    side: Literal["pro", "con", "unknown"]
    evidence_type: Literal["quoted", "paraphrased", "unknown"]
    topic_tags: list[ShortText] = Field(default_factory=list, max_length=12)
    indexable: bool
    flags: list[ShortText] = Field(default_factory=list, max_length=30)
    explanation: Annotated[str, StringConstraints(max_length=2000)]


class IngestionProvenance(StrictContract):
    extraction_backend: Literal["ooxml"] = "ooxml"
    boundary_method: Literal["model"] = "model"
    labeling_method: Literal["model"] = "model"
    model: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    boundary_prompt: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    labeling_prompt: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    confirmed: Literal[False] = False


class IngestionPreview(StrictContract):
    artifact_type: Literal["ingestion_preview"] = "ingestion_preview"
    job_id: Identifier
    confirmation_token: Identifier
    source_filename: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Literal["pro", "con", "unknown"]
    cards: list[IngestionPreviewCard] = Field(max_length=200)
    warnings: list[ShortText] = Field(default_factory=list, max_length=100)
    provenance: IngestionProvenance


class IngestionCommitResult(StrictContract):
    artifact_type: Literal["ingestion_commit_result"] = "ingestion_commit_result"
    job_id: Identifier
    source_filename: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    written_cards: int = Field(ge=0)
    searchable_cards: int = Field(ge=0)
    ledger_schema_version: int = Field(ge=1)
    index_rebuilt: Literal[True]

    @model_validator(mode="after")
    def validate_counts(self) -> "IngestionCommitResult":
        if self.searchable_cards > self.written_cards:
            raise ValueError("searchable_cards cannot exceed written_cards")
        return self


class ArgumentRequest(ModelContract):
    original_request: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side
    subject: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    entities: list[ShortText] = Field(default_factory=list, max_length=12)
    requested_sections: list[
        Literal[
            "claim",
            "warrant",
            "evidence",
            "impact",
            "resolution_link",
            "likely_response",
        ]
    ] = Field(default_factory=list, max_length=6)
    speech_position: ShortText | None = None
    length_seconds: int | None = Field(default=None, ge=10, le=600)
    source_files: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=20)
    constraints: list[ShortText] = Field(default_factory=list, max_length=12)
    preserve_citations: bool = False
    revision_instruction: (
        Annotated[str, StringConstraints(min_length=1, max_length=5000)] | None
    ) = None

class ArgumentSection(ModelContract):
    text: Annotated[str, StringConstraints(min_length=1, max_length=5000)]
    support: SupportStatus
    card_ids: list[CardId] = Field(default_factory=list, max_length=MAX_EVIDENCE_CARDS)

    @model_validator(mode="after")
    def validate_support(self) -> "ArgumentSection":
        if self.support in {"supported", "partially_supported"} and not self.card_ids:
            raise ValueError("supported sections require at least one card id")
        if self.support == "unsupported" and self.card_ids:
            raise ValueError("unsupported sections cannot cite a card")
        return self


class ArgumentDraft(ModelContract):
    artifact_type: Literal["argument_draft"] = "argument_draft"
    title: ShortText
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side
    format: Literal["structured_argument"] = "structured_argument"
    claim: ArgumentSection
    warrant: ArgumentSection
    evidence: ArgumentSection
    impact: ArgumentSection
    resolution_link: ArgumentSection
    likely_response: ArgumentSection
    source_card_ids: list[CardId] = Field(
        default_factory=list, max_length=MAX_EVIDENCE_CARDS
    )
    unsupported_facts: list[
        Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    ] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_citation_union(self) -> "ArgumentDraft":
        cited = {
            card_id
            for section in (
                self.claim,
                self.warrant,
                self.evidence,
                self.impact,
                self.resolution_link,
                self.likely_response,
            )
            for card_id in section.card_ids
        }
        if cited != set(self.source_card_ids):
            raise ValueError(
                "source_card_ids must exactly match the section citation union"
            )
        return self


class DrillPlan(ModelContract):
    artifact_type: Literal["drill_plan"] = "drill_plan"
    student_id: Identifier
    speech_position: ShortText
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    side: Side
    title: ShortText
    focus: list[ShortText] = Field(min_length=1, max_length=8)
    instructions: list[
        Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    ] = Field(min_length=1, max_length=20)
    duration_minutes: int = Field(ge=1, le=180)
    evidence_card_ids: list[CardId] = Field(default_factory=list, max_length=20)
    personalization_summary: Annotated[str, StringConstraints(max_length=2000)]

class CoachTurn(ModelContract):
    artifact_type: Literal["coach_turn"] = "coach_turn"
    label: Literal["simulated_coach"] = "simulated_coach"
    student_id: Identifier
    speech_position: ShortText
    side: Side
    focus: ShortText
    feedback: Annotated[str, StringConstraints(min_length=1, max_length=5000)]
    question: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    evidence_card_ids: list[CardId] = Field(default_factory=list, max_length=20)
    continue_session: bool = True


class ProgressEntry(StrictContract):
    student_id: Identifier
    date: date
    speech_position: ShortText
    resolution: Annotated[str, StringConstraints(max_length=500)]
    weakness_tags: list[ShortText] = Field(default_factory=list, max_length=20)
    assessment_text: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    author_id: Identifier

    @field_validator("date", mode="before")
    @classmethod
    def parse_progress_date(cls, value: Any) -> Any:
        if isinstance(value, str):
            return date.fromisoformat(value)
        return value


class ProgressSummary(ModelContract):
    artifact_type: Literal["progress_summary"] = "progress_summary"
    student_id: Identifier
    records: list[ProgressEntry] = Field(
        default_factory=list,
        max_length=MAX_PROGRESS_RECORDS,
    )
    summary: Annotated[str, StringConstraints(min_length=1, max_length=5000)]


class AssessmentProposal(ModelContract):
    artifact_type: Literal["assessment_proposal"] = "assessment_proposal"
    student_id: Identifier
    speech_position: ShortText
    resolution: Annotated[str, StringConstraints(max_length=500)]
    weakness_tags: list[ShortText] = Field(default_factory=list, max_length=20)
    assessment_text: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    confirmation_required: bool = True


class CalendarEvent(StrictContract):
    artifact_type: Literal["calendar_event"] = "calendar_event"
    event_id: Identifier
    student_id: Identifier
    start: datetime
    end: datetime
    timezone: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    attendee_email: Annotated[str, StringConstraints(max_length=320)] | None = None
    backend: Literal["google", "fixture"]
    synthetic: bool

    @model_validator(mode="after")
    def validate_event(self) -> "CalendarEvent":
        if self.end <= self.start:
            raise ValueError("calendar event end must follow start")
        if self.backend == "fixture" and not self.synthetic:
            raise ValueError("fixture calendar events must be labeled synthetic")
        return self


Artifact: TypeAlias = Annotated[
    EvidencePacket
    | RulePacket
    | TopicPacket
    | IngestionPreview
    | IngestionCommitResult
    | ArgumentDraft
    | DrillPlan
    | CoachTurn
    | ProgressSummary
    | AssessmentProposal
    | CalendarEvent,
    Field(discriminator="artifact_type"),
]


class AgentTraceEntry(StrictContract):
    sequence: int = Field(ge=0)
    agent: AgentName
    event: Literal["activated", "decision", "handoff", "returned", "finished"]
    from_agent: AgentName | None = None
    to_agent: AgentName | None = None
    reason_code: Annotated[str, StringConstraints(max_length=100)] | None = None
    summary: Annotated[str, StringConstraints(max_length=1000)]


class ToolTraceEntry(StrictContract):
    sequence: int = Field(ge=0)
    agent: AgentName
    tool: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    stage: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    status: Literal["started", "completed", "failed"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: Annotated[str, StringConstraints(max_length=1000)] | None = None
    error_code: Annotated[str, StringConstraints(max_length=100)] | None = None


class ModelTraceEntry(StrictContract):
    sequence: int = Field(ge=0)
    agent: AgentName
    model: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    prompt_template: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    prompt_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    response_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None
    schema_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    started_at: datetime
    latency_ms: int | None = Field(default=None, ge=0)
    stop_reason: Annotated[str, StringConstraints(max_length=100)] | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    status: Literal["completed", "failed"]
    error_code: Annotated[str, StringConstraints(max_length=100)] | None = None
    rendered_system_prompt: (
        Annotated[str, StringConstraints(max_length=100_000)] | None
    ) = None
    rendered_user_payload: (
        Annotated[str, StringConstraints(max_length=100_000)] | None
    ) = None
    model_response: Annotated[str, StringConstraints(max_length=100_000)] | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "ModelTraceEntry":
        if self.status == "completed":
            if self.response_sha256 is None or self.error_code is not None:
                raise ValueError(
                    "completed model calls require response_sha256 and no error_code"
                )
        elif self.response_sha256 is not None or self.error_code is None:
            raise ValueError(
                "failed model calls require error_code and no response_sha256"
            )
        return self
