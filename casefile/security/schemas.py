"""Strict schemas for all model and sensitive-tool boundaries."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


Identifier = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ClassifierOutput(StrictModel):
    intent: Literal[
        "retrieve_evidence",
        "explain_rule",
        "generate_drill",
        "coach_simulation",
        "progress",
        "ingest_cards",
        "schedule_session",
        "integrity_refusal",
        "unknown",
    ]
    side: Literal["pro", "con", "unknown"]
    student_id: Annotated[str, StringConstraints(max_length=100)] | None
    speech_position: Annotated[str, StringConstraints(max_length=100)] | None
    file_path: Annotated[str, StringConstraints(max_length=1000)] | None
    confirmation_token: Annotated[
        str, StringConstraints(pattern=r"^[0-9a-fA-F]{32}$")
    ] | None
    start: Annotated[str, StringConstraints(max_length=100)] | None
    clarification_needed: bool
    clarification_question: Annotated[str, StringConstraints(max_length=300)] | None


BoundaryFlag = Literal[
    "no_header",
    "no_body",
    "cite_is_bare_url",
    "cite_is_bare_headline",
    "tag_merged_into_cite",
    "cite_body_same_paragraph",
    "pdf_paste_fragmented",
    "text_corrupt",
    "duplicate_source",
]


class BoundaryCardOutput(StrictModel):
    header: Annotated[int, Field(ge=0)] | None
    tag: list[Annotated[int, Field(ge=0)]] = Field(max_length=64)
    cite: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=64)
    body: list[Annotated[int, Field(ge=0)]] = Field(max_length=128)
    flags: list[BoundaryFlag] = Field(max_length=16)


class BoundaryOutput(StrictModel):
    cards: list[BoundaryCardOutput] = Field(max_length=200)
    junk: list[Annotated[int, Field(ge=0)]] = Field(max_length=5000)

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
            raise ValueError(f"Model returned paragraph ids outside its window: {sorted(invalid)}")


FieldFlag = Literal[
    "no_header",
    "no_body",
    "cite_is_bare_url",
    "cite_is_bare_headline",
    "tag_merged_into_cite",
    "cite_body_same_paragraph",
    "pdf_paste_fragmented",
    "text_corrupt",
    "html_entity",
    "duplicate_source",
    "no_marking",
    "fully_marked",
    "paraphrase_no_source",
    "do_not_ingest",
    "body_truncated",
]


class FieldOutput(StrictModel):
    evidence_type: Literal["quoted", "paraphrased", "unknown"]
    source_text_present: bool
    side: Literal["pro", "con", "unknown"]
    topic_tags: list[
        Annotated[str, StringConstraints(min_length=1, max_length=40)]
    ] = Field(max_length=6)
    flags: list[FieldFlag] = Field(max_length=20)


class SearchCardsArgs(StrictModel):
    query: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    side: Literal["pro", "con"]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    n: int = Field(default=5, ge=1, le=10)


class SearchRulesArgs(StrictModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    n: int = Field(default=3, ge=1, le=8)


class DrillArgs(StrictModel):
    student_id: Identifier
    speech_position: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    side: Literal["pro", "con"]


class CoachTurnOutput(StrictModel):
    feedback: Annotated[str, StringConstraints(min_length=1, max_length=600)]
    question: Annotated[str, StringConstraints(min_length=1, max_length=400)]
    focus: Annotated[str, StringConstraints(min_length=1, max_length=100)]


class EvidenceArgumentOutput(StrictModel):
    claim: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    warrant: Annotated[str, StringConstraints(min_length=1, max_length=900)]
    impact: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    citations_used: list[
        Annotated[str, StringConstraints(min_length=1, max_length=200)]
    ] = Field(min_length=1, max_length=5)


class ProgressArgs(StrictModel):
    student_id: Identifier


class ApproveCardArgs(StrictModel):
    card_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None


class IngestArgs(StrictModel):
    file_path: Annotated[str, StringConstraints(max_length=2000)] | None = None
    resolution: Annotated[str, StringConstraints(max_length=200)] | None = None
    side: Literal["pro", "con", "unknown"] | None = None
    dry_run: bool = True
    confirmation_token: Annotated[
        str, StringConstraints(pattern=r"^[0-9a-f]{32}$")
    ] | None = None
    use_model: bool = True
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None

    @field_validator("file_path")
    @classmethod
    def blank_path_is_none(cls, value: str | None) -> str | None:
        return value or None


class ScheduleArgs(StrictModel):
    student_id: Identifier
    start: Annotated[str, StringConstraints(max_length=100)]
    duration_minutes: int = Field(default=45, ge=15, le=180)
    attendee_email: Annotated[str, StringConstraints(max_length=320)] | None = None
    timezone_name: Annotated[str, StringConstraints(min_length=1, max_length=100)] = (
        "America/Chicago"
    )
    confirmation_token: Annotated[
        str, StringConstraints(pattern=r"^[0-9a-f]{32}$")
    ] | None = None
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None

    @field_validator("attendee_email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", value
        ):
            raise ValueError("attendee_email is not valid")
        return value


class AssessmentArgs(StrictModel):
    student_id: Identifier
    speech_position: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(max_length=200)]
    weakness_tags: list[
        Annotated[str, StringConstraints(min_length=1, max_length=50)]
    ] = Field(max_length=20)
    assessment_text: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    date: Annotated[str, StringConstraints(max_length=30)] | None = None
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None
