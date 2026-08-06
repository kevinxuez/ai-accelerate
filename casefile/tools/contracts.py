"""Strict input contracts for registered CaseFile tools."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from casefile.agents.contracts import StrictContract


Identifier = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class SearchCardsArgs(StrictContract):
    query: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    side: Literal["pro", "con"]
    resolution: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    n: int = Field(default=5, ge=1, le=10)
    source_files: list[
        Annotated[str, StringConstraints(min_length=1, max_length=300)]
    ] = Field(default_factory=list, max_length=20)


class SearchRulesArgs(StrictContract):
    question: Annotated[str, StringConstraints(min_length=1, max_length=20_000)]
    n: int = Field(default=3, ge=1, le=8)


class CurrentTopicArgs(StrictContract):
    event: Annotated[str, StringConstraints(min_length=1, max_length=100)] = (
        "Public Forum"
    )
    as_of: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = (
        None
    )


class ProgressArgs(StrictContract):
    student_id: Identifier


class ApproveCardArgs(StrictContract):
    card_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None


class ScheduleArgs(StrictContract):
    student_id: Identifier
    start: Annotated[str, StringConstraints(max_length=100)]
    duration_minutes: int = Field(default=45, ge=15, le=180)
    attendee_email: Annotated[str, StringConstraints(max_length=320)] | None = None
    timezone_name: Annotated[str, StringConstraints(min_length=1, max_length=100)] = (
        "America/Chicago"
    )
    confirmation_token: (
        Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")] | None
    ) = None
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None

    @field_validator("attendee_email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", value
        ):
            raise ValueError("attendee_email is not valid")
        return value


class AssessmentArgs(StrictContract):
    student_id: Identifier
    speech_position: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    resolution: Annotated[str, StringConstraints(max_length=500)]
    weakness_tags: list[
        Annotated[str, StringConstraints(min_length=1, max_length=50)]
    ] = Field(max_length=20)
    assessment_text: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    date: Annotated[str, StringConstraints(max_length=30)] | None = None
    idempotency_key: Annotated[str, StringConstraints(max_length=200)] | None = None
