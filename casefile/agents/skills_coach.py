"""Model-driven drills, simulated coaching, progress, and assessments."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import ValidationError

from casefile.tools.context import ToolContext

from .contracts import (
    MAX_COACHING_TURNS,
    MAX_PROGRESS_RECORDS,
    AssessmentProposal,
    CoachTurn,
    ConversationMessage,
    DrillPlan,
    EvidencePacket,
    EvidenceRequest,
    ProgressEntry,
    ProgressSummary,
    Side,
    StrictContract,
)
from .errors import CaseFileError, ErrorCode
from .prompt_registry import load_prompt


SKILLS_COACH = "skills_coach"
ContractT = TypeVar("ContractT", bound=StrictContract)


class SkillsCoach:
    """Run coach model operations while keeping tools and writes policy-bound."""

    def __init__(self, tools: Any, model: Any) -> None:
        self.tools = tools
        self.model = model

    def request_evidence(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        side: Side,
        focus: str,
        intended_use: str,
        progress_summary: ProgressSummary | None = None,
        source_files: list[str] | None = None,
    ) -> EvidenceRequest:
        """Return a typed handoff; the coach never invokes retrieval itself."""

        self._authorize_student(context, student_id)
        if intended_use not in {"drill", "coaching"}:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "Evidence intended_use must be drill or coaching.",
                stage="skills_coach.request_evidence",
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"field": "intended_use"},
            )
        allowed_files = list(dict.fromkeys(source_files or []))
        request = self._complete(
            context,
            schema=EvidenceRequest,
            operation="request_evidence",
            payload={
                "student_id": student_id,
                "speech_position": speech_position,
                "active_resolution": context.resolution,
                "required_side": side,
                "focus": focus,
                "intended_use": intended_use,
                "allowed_source_files": allowed_files,
                "progress_summary": (
                    progress_summary.model_dump(mode="json")
                    if progress_summary is not None
                    else None
                ),
            },
        )
        invalid_field: str | None = None
        if request.resolution != context.resolution:
            invalid_field = "resolution"
        elif request.side != side:
            invalid_field = "side"
        elif request.intended_use != intended_use:
            invalid_field = "intended_use"
        elif set(request.source_files) - set(allowed_files):
            invalid_field = "source_files"
        if invalid_field is not None:
            self._invalid_output(
                context,
                "The Skills Coach changed a protected evidence request constraint.",
                field=invalid_field,
            )
        return request

    def summarize_progress(
        self,
        context: ToolContext,
        *,
        student_id: str,
    ) -> ProgressSummary:
        """Read authorized progress and produce a model-written typed summary."""

        raw_records = self.tools.registry.invoke(
            SKILLS_COACH,
            "get_progress",
            self._context(context),
            {"student_id": student_id},
        )
        if isinstance(raw_records, list) and len(raw_records) > MAX_PROGRESS_RECORDS:
            raise CaseFileError(
                ErrorCode.STATE_LIMIT_EXCEEDED,
                "The authorized progress history exceeds the model context limit.",
                stage="skills_coach.summarize_progress",
                agent=SKILLS_COACH,
                tool="get_progress",
                request_id=context.request_id,
                safe_details={"limit": MAX_PROGRESS_RECORDS},
            )
        records = self._progress_entries(
            raw_records,
            context=context,
        )
        summary = self._complete(
            context,
            schema=ProgressSummary,
            operation="summarize_progress",
            payload={
                "student_id": student_id,
                "records": [record.model_dump(mode="json") for record in records],
            },
        )
        if summary.student_id != student_id:
            self._invalid_output(
                context,
                "The Skills Coach changed the progress student.",
                field="student_id",
            )
        if summary.records != records:
            self._invalid_output(
                context,
                "The Skills Coach changed the stored progress records.",
                field="records",
            )
        return summary

    def generate_drill(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        side: Side,
        focus: str,
        progress_summary: ProgressSummary | None = None,
        evidence_packet: EvidencePacket | None = None,
    ) -> DrillPlan:
        """Generate a personalized drill; never synthesize deterministic instructions."""

        self._authorize_student(context, student_id)
        progress = progress_summary or self.summarize_progress(
            context,
            student_id=student_id,
        )
        self._validate_progress(progress, context=context, student_id=student_id)
        available_ids = self._validate_evidence(
            evidence_packet,
            context=context,
            side=side,
        )
        drill = self._complete(
            context,
            schema=DrillPlan,
            operation="generate_drill",
            payload={
                "student_id": student_id,
                "speech_position": speech_position,
                "active_resolution": context.resolution,
                "required_side": side,
                "focus": focus,
                "progress_summary": progress.model_dump(mode="json"),
                "evidence_packet": (
                    evidence_packet.model_dump(mode="json")
                    if evidence_packet is not None
                    else None
                ),
            },
            max_tokens=6_000,
        )
        invalid_field: str | None = None
        if drill.student_id != student_id:
            invalid_field = "student_id"
        elif drill.speech_position != speech_position:
            invalid_field = "speech_position"
        elif drill.resolution != context.resolution:
            invalid_field = "resolution"
        elif drill.side != side:
            invalid_field = "side"
        elif set(drill.evidence_card_ids) - available_ids:
            invalid_field = "evidence_card_ids"
        if invalid_field is not None:
            self._invalid_output(
                context,
                "The Skills Coach returned a drill outside its authorized context.",
                field=invalid_field,
            )
        return drill

    def coach_turn(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        side: Side,
        focus: str,
        student_message: str,
        prior_turns: list[ConversationMessage] | None = None,
        progress_summary: ProgressSummary | None = None,
        evidence_packet: EvidencePacket | None = None,
    ) -> CoachTurn:
        """Return one explicitly simulated, interactive coaching turn."""

        self._authorize_student(context, student_id)
        turns = self._messages(
            list(prior_turns or []),
            context=context,
            stage="skills_coach.coach_turn",
        )
        if len(turns) >= MAX_COACHING_TURNS:
            raise CaseFileError(
                ErrorCode.STATE_LIMIT_EXCEEDED,
                "The coaching session reached its turn limit.",
                stage="skills_coach.coach_turn",
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"limit": MAX_COACHING_TURNS},
            )
        current_message = self._messages(
            [{"role": "user", "content": student_message}],
            context=context,
            stage="skills_coach.coach_turn",
        )[0]
        progress = progress_summary or self.summarize_progress(
            context,
            student_id=student_id,
        )
        self._validate_progress(progress, context=context, student_id=student_id)
        available_ids = self._validate_evidence(
            evidence_packet,
            context=context,
            side=side,
        )
        turn = self._complete(
            context,
            schema=CoachTurn,
            operation="coach_turn",
            payload={
                "student_id": student_id,
                "speech_position": speech_position,
                "active_resolution": context.resolution,
                "required_side": side,
                "focus": focus,
                "student_message": current_message.content,
                "prior_turns": [item.model_dump(mode="json") for item in turns],
                "progress_summary": progress.model_dump(mode="json"),
                "evidence_packet": (
                    evidence_packet.model_dump(mode="json")
                    if evidence_packet is not None
                    else None
                ),
            },
            max_tokens=3_000,
        )
        invalid_field: str | None = None
        if turn.student_id != student_id:
            invalid_field = "student_id"
        elif turn.speech_position != speech_position:
            invalid_field = "speech_position"
        elif turn.side != side:
            invalid_field = "side"
        elif set(turn.evidence_card_ids) - available_ids:
            invalid_field = "evidence_card_ids"
        if invalid_field is not None:
            self._invalid_output(
                context,
                "The Skills Coach returned a coaching turn outside its authorized context.",
                field=invalid_field,
            )
        return turn

    def propose_assessment(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        coaching_turns: list[ConversationMessage],
    ) -> AssessmentProposal:
        """Create a non-writing proposal from an explicit coaching transcript."""

        self._authorize_student(context, student_id)
        turns = self._messages(
            coaching_turns,
            context=context,
            stage="skills_coach.propose_assessment",
        )
        if not turns:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "An assessment proposal requires a coaching transcript.",
                stage="skills_coach.propose_assessment",
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"field": "coaching_turns"},
            )
        if len(turns) > MAX_COACHING_TURNS:
            raise CaseFileError(
                ErrorCode.STATE_LIMIT_EXCEEDED,
                "The coaching transcript exceeds the assessment limit.",
                stage="skills_coach.propose_assessment",
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"limit": MAX_COACHING_TURNS},
            )
        proposal = self._complete(
            context,
            schema=AssessmentProposal,
            operation="propose_assessment",
            payload={
                "student_id": student_id,
                "speech_position": speech_position,
                "active_resolution": context.resolution,
                "coaching_turns": [item.model_dump(mode="json") for item in turns],
            },
            max_tokens=3_000,
        )
        invalid_field: str | None = None
        if proposal.student_id != student_id:
            invalid_field = "student_id"
        elif proposal.speech_position != speech_position:
            invalid_field = "speech_position"
        elif proposal.resolution != context.resolution:
            invalid_field = "resolution"
        if invalid_field is not None:
            self._invalid_output(
                context,
                "The Skills Coach changed a protected assessment field.",
                field=invalid_field,
            )
        return proposal

    def log_assessment(
        self,
        context: ToolContext,
        *,
        proposal: AssessmentProposal,
        confirmed: bool,
        idempotency_key: str | None = None,
    ) -> ProgressEntry:
        """Write only after a separate, explicit coach-authorized confirmation."""

        if not confirmed:
            raise CaseFileError(
                ErrorCode.CONFIRMATION_INVALID,
                "Assessment logging requires an explicit coach confirmation.",
                stage="skills_coach.log_assessment",
                agent=SKILLS_COACH,
                request_id=context.request_id,
            )
        raw = self.tools.registry.invoke(
            SKILLS_COACH,
            "log_assessment",
            self._context(context),
            {
                "student_id": proposal.student_id,
                "speech_position": proposal.speech_position,
                "resolution": proposal.resolution,
                "weakness_tags": proposal.weakness_tags,
                "assessment_text": proposal.assessment_text,
                "date": None,
                "idempotency_key": idempotency_key,
            },
        )
        entries = self._progress_entries([raw], context=context)
        return entries[0]

    def confirm_assessment(
        self,
        context: ToolContext,
        *,
        proposal: AssessmentProposal,
        idempotency_key: str | None = None,
    ) -> ProgressEntry:
        return self.log_assessment(
            context,
            proposal=proposal,
            confirmed=True,
            idempotency_key=idempotency_key,
        )

    def _complete(
        self,
        context: ToolContext,
        *,
        schema: type[ContractT],
        operation: str,
        payload: dict[str, Any],
        max_tokens: int = 3_000,
    ) -> ContractT:
        prompt = load_prompt("skills_coach")
        try:
            raw = self.model.complete_json(
                system=prompt.content,
                user=json.dumps(
                    {"operation": operation, **payload},
                    ensure_ascii=False,
                ),
                max_tokens=max_tokens,
                schema=schema,
                agent=SKILLS_COACH,
                prompt_template=prompt.template_name,
                prompt_version=prompt.version,
            )
        except CaseFileError as error:
            if context.request_id is not None:
                error.with_request_id(context.request_id)
            raise
        try:
            return schema.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.MODEL_OUTPUT_INVALID,
                f"The Skills Coach returned invalid {operation} output.",
                stage=f"skills_coach.{operation}",
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"schema": schema.__name__},
                cause=exc,
            ) from exc

    def _validate_evidence(
        self,
        packet: EvidencePacket | None,
        *,
        context: ToolContext,
        side: Side,
    ) -> set[str]:
        if packet is None:
            return set()
        invalid_field: str | None = None
        if packet.resolution != context.resolution:
            invalid_field = "evidence_packet.resolution"
        elif packet.side != side:
            invalid_field = "evidence_packet.side"
        if invalid_field is not None:
            self._invalid_output(
                context,
                "The supplied EvidencePacket does not match the coaching context.",
                field=invalid_field,
            )
        return {card.card_id for card in packet.cards}

    def _validate_progress(
        self,
        summary: ProgressSummary,
        *,
        context: ToolContext,
        student_id: str,
    ) -> None:
        if summary.student_id != student_id or any(
            record.student_id != student_id for record in summary.records
        ):
            self._invalid_output(
                context,
                "The progress summary does not match the coaching student.",
                field="progress_summary.student_id",
            )

    @staticmethod
    def _messages(
        values: list[Any],
        *,
        context: ToolContext,
        stage: str,
    ) -> list[ConversationMessage]:
        try:
            return [ConversationMessage.model_validate(value) for value in values]
        except (TypeError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "The coaching conversation contains an invalid message.",
                stage=stage,
                agent=SKILLS_COACH,
                request_id=context.request_id,
                safe_details={"field": "messages"},
                cause=exc,
            ) from exc

    @staticmethod
    def _progress_entries(
        raw_records: Any,
        *,
        context: ToolContext,
    ) -> list[ProgressEntry]:
        if not isinstance(raw_records, list):
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The progress tool returned malformed records.",
                stage="skills_coach.progress_records",
                agent=SKILLS_COACH,
                tool="get_progress",
                request_id=context.request_id,
            )
        try:
            return [
                ProgressEntry(
                    student_id=str(record["student_id"]),
                    date=record["date"],
                    speech_position=str(record["speech_position"]),
                    resolution=str(record.get("resolution") or ""),
                    weakness_tags=[str(tag) for tag in record.get("weakness_tags", [])],
                    assessment_text=str(record["assessment_text"]),
                    author_id=str(record["author_id"]),
                )
                for record in raw_records
            ]
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The progress tool returned malformed records.",
                stage="skills_coach.progress_records",
                agent=SKILLS_COACH,
                tool="get_progress",
                request_id=context.request_id,
                cause=exc,
            ) from exc

    @staticmethod
    def _authorize_student(context: ToolContext, student_id: str) -> None:
        if context.role == "student" and student_id != context.user_id:
            raise CaseFileError(
                ErrorCode.AUTHORIZATION_DENIED,
                "Students may coach only within their own progress context.",
                stage="skills_coach.authorization",
                agent=SKILLS_COACH,
                request_id=context.request_id,
            )

    @staticmethod
    def _context(context: ToolContext) -> ToolContext:
        if context.agent == SKILLS_COACH:
            return context
        return ToolContext(
            role=context.role,
            user_id=context.user_id,
            resolution=context.resolution,
            request_id=context.request_id,
            agent=SKILLS_COACH,
        )

    @staticmethod
    def _invalid_output(
        context: ToolContext,
        message: str,
        *,
        field: str,
    ) -> None:
        raise CaseFileError(
            ErrorCode.AGENT_OUTPUT_INVALID,
            message,
            stage="skills_coach.validate",
            agent=SKILLS_COACH,
            request_id=context.request_id,
            safe_details={"field": field},
        )
