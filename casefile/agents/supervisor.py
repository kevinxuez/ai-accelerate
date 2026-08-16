"""Model-driven Supervisor decisions and typed execution preparation."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import ValidationError

from .contracts import (
    ActiveGoal,
    CoachingTask,
    ScheduleToolCall,
    StrictContract,
    SupervisorDecision,
)
from .errors import CaseFileError, ErrorCode
from .prompt_registry import load_prompt
from .state import CaseFileState


SUPERVISOR = "supervisor"
ContractT = TypeVar("ContractT", bound=StrictContract)

_ARTIFACTS_BY_SPECIALIST = {
    "evidence_librarian": {
        "evidence_packet",
        "rule_packet",
        "topic_packet",
        "ingestion_preview",
        "ingestion_commit_result",
    },
    "argument_strategist": {"argument_draft"},
    "skills_coach": {
        "drill_plan",
        "coach_turn",
        "progress_summary",
        "assessment_proposal",
    },
}

_COACH_OPERATION_BY_ARTIFACT = {
    "drill_plan": "generate_drill",
    "coach_turn": "coach_turn",
    "progress_summary": "progress_summary",
    "assessment_proposal": "assessment_proposal",
}


class Supervisor:
    """Use the configured model to choose the next typed graph action."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def decide(self, state: CaseFileState) -> SupervisorDecision:
        decision = self._complete(
            state,
            schema=SupervisorDecision,
            operation="decide",
            payload={"state": self._public_state(state)},
            max_tokens=3_000,
        )
        self._validate_decision(state, decision)
        return decision

    def prepare_schedule(
        self,
        state: CaseFileState,
        decision: SupervisorDecision,
        *,
        idempotency_key: str | None = None,
    ) -> ScheduleToolCall:
        call = self._complete(
            state,
            schema=ScheduleToolCall,
            operation="prepare_schedule",
            payload={
                "decision": decision.model_dump(mode="json"),
                "state": self._public_state(state),
            },
            max_tokens=1_500,
        )
        if state.request.role == "student" and call.student_id != state.request.user_id:
            self._invalid(
                state,
                "The Supervisor changed the student identity for a scheduling call.",
                field="student_id",
            )
        if call.confirmation_token is not None or call.idempotency_key is not None:
            self._invalid(
                state,
                "The Supervisor supplied a protected scheduling credential.",
                field=(
                    "confirmation_token"
                    if call.confirmation_token is not None
                    else "idempotency_key"
                ),
            )
        return call.model_copy(update={"idempotency_key": idempotency_key})

    def prepare_coaching(
        self,
        state: CaseFileState,
        decision: SupervisorDecision,
    ) -> CoachingTask:
        task = self._complete(
            state,
            schema=CoachingTask,
            operation="prepare_coaching",
            payload={
                "decision": decision.model_dump(mode="json"),
                "state": self._public_state(state),
            },
            max_tokens=1_500,
        )
        expected = _COACH_OPERATION_BY_ARTIFACT.get(decision.required_artifact or "")
        task = task.model_copy(update={"operation": expected})
        if state.request.role == "student" and task.student_id != state.request.user_id:
            self._invalid(
                state,
                "The Supervisor changed the student identity for a coaching handoff.",
                field="student_id",
            )
        allowed_files = (
            set(state.evidence_packet.confirmed_source_files_considered)
            if state.evidence_packet is not None
            else set()
        )
        if set(task.source_files) - allowed_files:
            self._invalid(
                state,
                "The Supervisor selected a coaching source outside the confirmed state.",
                field="source_files",
            )
        return task

    @staticmethod
    def active_goal(decision: SupervisorDecision) -> ActiveGoal:
        if decision.required_artifact is not None:
            criterion = f"Return a validated {decision.required_artifact}."
        elif decision.action == "ask_clarification":
            criterion = "Obtain the missing user input."
        elif decision.action == "call_schedule":
            criterion = "Return a confirmed calendar event or confirmation request."
        else:
            criterion = "Present the completed result to the user."
        summary = decision.goal or decision.task or "Complete the user's request."
        return ActiveGoal(summary=summary[:500], completion_criteria=[criterion])

    def _complete(
        self,
        state: CaseFileState,
        *,
        schema: type[ContractT],
        operation: str,
        payload: dict[str, Any],
        max_tokens: int,
    ) -> ContractT:
        prompt = load_prompt("supervisor")
        try:
            raw = self.model.complete_json(
                system=prompt.content,
                user=json.dumps(
                    {"operation": operation, **payload},
                    ensure_ascii=False,
                ),
                max_tokens=max_tokens,
                schema=schema,
                agent=SUPERVISOR,
                prompt_template=prompt.template_name,
                prompt_version=prompt.version,
            )
        except CaseFileError as error:
            if error.code == ErrorCode.MODEL_OUTPUT_INVALID:
                raise CaseFileError(
                    ErrorCode.AGENT_OUTPUT_INVALID,
                    f"The Supervisor returned invalid {operation} output.",
                    stage=f"supervisor.{operation}",
                    agent=SUPERVISOR,
                    request_id=state.request.request_id,
                    safe_details=error.safe_details,
                    cause=error,
                ) from error
            raise error.with_request_id(state.request.request_id)
        try:
            return schema.model_validate(raw)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                f"The Supervisor returned invalid {operation} output.",
                stage=f"supervisor.{operation}",
                agent=SUPERVISOR,
                request_id=state.request.request_id,
                safe_details={"schema": schema.__name__},
                cause=exc,
            ) from exc

    def _validate_decision(
        self,
        state: CaseFileState,
        decision: SupervisorDecision,
    ) -> None:
        if decision.action == "delegate":
            allowed = _ARTIFACTS_BY_SPECIALIST.get(decision.target_agent or "", set())
            if decision.required_artifact not in allowed:
                self._invalid(
                    state,
                    "The Supervisor selected an artifact outside the specialist contract.",
                    field="required_artifact",
                )
            if (
                decision.target_agent == "argument_strategist"
                and state.evidence_packet is None
            ):
                self._invalid(
                    state,
                    "The Supervisor delegated argument generation without an EvidencePacket.",
                    field="target_agent",
                )
        if decision.action in {"finish", "refuse"} and not (
            decision.task and decision.task.strip()
        ):
            self._invalid(
                state,
                "The Supervisor completed a turn without a user-facing response.",
                field="task",
            )

    @staticmethod
    def _public_state(state: CaseFileState) -> dict[str, Any]:
        value = state.model_dump(
            mode="json",
            exclude={"agent_trace", "tool_trace", "model_trace", "error"},
        )
        for packet in [value.get("evidence_packet"), *value.get("artifacts", [])]:
            if (
                not isinstance(packet, dict)
                or packet.get("artifact_type") != "evidence_packet"
            ):
                continue
            for card in packet.get("cards", []):
                if isinstance(card, dict):
                    for field in ("body", "citation", "read_spans", "emphasis_spans"):
                        card.pop(field, None)
        pending = value.get("pending_confirmation")
        if isinstance(pending, dict):
            pending.pop("token", None)
        schedule = value.get("schedule_request")
        if isinstance(schedule, dict):
            schedule.pop("confirmation_token", None)
            schedule.pop("idempotency_key", None)
        return value

    @staticmethod
    def _invalid(state: CaseFileState, message: str, *, field: str) -> None:
        raise CaseFileError(
            ErrorCode.AGENT_OUTPUT_INVALID,
            message,
            stage="supervisor.validate",
            agent=SUPERVISOR,
            request_id=state.request.request_id,
            safe_details={"field": field},
        )
