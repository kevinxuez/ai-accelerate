"""Authorized progress reads and explicitly confirmed assessment writes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import ValidationError

from casefile.security.prompt_guard import inspect_text

from .contracts import AssessmentArgs, ProgressArgs
from .context import WRITE_LOCK, ToolContext, ToolRuntime


SKILLS_COACH = frozenset({"skills_coach"})


@dataclass
class ProgressRecord:
    student_id: str
    date: str
    speech_position: str
    resolution: str
    weakness_tags: list[str]
    assessment_text: str
    author_role: Literal["student", "coach"]
    author_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressTools:
    def __init__(self, runtime: ToolRuntime) -> None:
        self.runtime = runtime

    def log_assessment(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        resolution: str,
        weakness_tags: list[str],
        assessment_text: str,
        date: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.authorize(
            context,
            tool="log_assessment",
            action="log assessment records",
            roles=frozenset({"coach"}),
            agents=SKILLS_COACH,
        )
        try:
            args = AssessmentArgs(
                student_id=student_id,
                speech_position=speech_position,
                resolution=resolution,
                weakness_tags=weakness_tags,
                assessment_text=assessment_text,
                date=date,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "log_assessment", exc)
        decision = inspect_text(args.assessment_text, trust="untrusted_user")
        if not decision.safe_for_write_tools:
            self.runtime.blocked(
                context,
                "log_assessment",
                {"assessment_text": assessment_text},
                signals=decision.signals,
            )
        prior = self.runtime.idempotent_result("log_assessment", args.idempotency_key)
        if prior is not None:
            return prior
        record = ProgressRecord(
            student_id=args.student_id,
            date=args.date or datetime.now(timezone.utc).date().isoformat(),
            speech_position=args.speech_position,
            resolution=args.resolution,
            weakness_tags=sorted({tag.lower() for tag in args.weakness_tags}),
            assessment_text=args.assessment_text,
            author_role="coach",
            author_id=context.user_id,
        ).to_dict()
        with WRITE_LOCK:
            records = self.runtime.read_progress()
            records.append(record)
            self.runtime.atomic_json(self.runtime.settings.progress_path, records)
            self.runtime.save_idempotent_result(
                "log_assessment", args.idempotency_key, record
            )
        self.runtime.audit(context, "log_assessment", record, {"written": True})
        return record

    def get_progress(
        self, context: ToolContext, student_id: str
    ) -> list[dict[str, Any]]:
        self.runtime.authorize(
            context,
            tool="get_progress",
            action="read progress records",
            roles=frozenset({"student", "coach"}),
            agents=SKILLS_COACH,
        )
        if context.role == "student" and student_id != context.user_id:
            self.runtime.denied(
                context,
                "read progress records for another student",
                "get_progress",
            )
        try:
            args = ProgressArgs(student_id=student_id)
        except ValidationError as exc:
            self.runtime.invalid(context, "get_progress", exc)
        result = [
            record
            for record in self.runtime.read_progress()
            if record.get("student_id") == args.student_id
        ]
        self.runtime.audit(
            context,
            "get_progress",
            {"student_id": student_id},
            result,
        )
        return result
