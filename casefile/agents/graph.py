"""The required LangGraph orchestration for the four-agent runtime."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from casefile.security.audit import RateLimiter, SecurityAuditor
from casefile.security.prompt_guard import inspect_text
from casefile.tools import CaseFileTools, ToolContext
from casefile.tools.registry import capture_tool_invocations

from .argument_strategist import ArgumentStrategist
from .contracts import (
    MAX_ARTIFACTS,
    MAX_COACHING_TURNS,
    MAX_GRAPH_STEPS,
    MAX_MESSAGES,
    MAX_PROGRESS_RECORDS,
    AgentName,
    AgentTraceEntry,
    Artifact,
    AssessmentProposal,
    CalendarEvent,
    ClarificationRequest,
    CoachingTask,
    ConfirmationRequest,
    ConversationMessage,
    EvidencePacket,
    EvidenceQueryPlan,
    EvidenceRequest,
    IngestionCommitResult,
    IngestionPreview,
    ModelTraceEntry,
    ProgressEntry,
    ProgressSummary,
    ScheduleToolCall,
    ToolTraceEntry,
)
from .errors import CaseFileError, ErrorCode
from .evidence_librarian import EvidenceLibrarian
from .skills_coach import SkillsCoach
from .state import CaseFileState, CoachingState, IngestionJobState
from .supervisor import Supervisor


END_ROUTE = "end"


class FourAgentGraphNodes:
    """Small graph nodes whose only routing authority is SupervisorDecision."""

    def __init__(
        self,
        *,
        supervisor: Supervisor,
        evidence_librarian: EvidenceLibrarian,
        argument_strategist: ArgumentStrategist,
        skills_coach: SkillsCoach,
        tools: CaseFileTools,
        model: Any,
        security_auditor: SecurityAuditor,
        rate_limiter: RateLimiter,
        max_steps: int = MAX_GRAPH_STEPS,
        attachment_resolver: Callable[[Any], str] | None = None,
    ) -> None:
        if not 1 <= max_steps <= MAX_GRAPH_STEPS:
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                f"Graph max_steps must be between 1 and {MAX_GRAPH_STEPS}.",
                stage="runtime.graph_configuration",
            )
        self.supervisor_agent = supervisor
        self.evidence_librarian_agent = evidence_librarian
        self.argument_strategist_agent = argument_strategist
        self.skills_coach_agent = skills_coach
        self.tools = tools
        self.model = model
        self.security_auditor = security_auditor
        self.rate_limiter = rate_limiter
        self.max_steps = max_steps
        self.attachment_resolver = attachment_resolver or (
            lambda attachment: attachment.attachment_id
        )

    def screen_request(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "supervisor")
        if limited is not None:
            return limited
        message = self._latest_user_message(state)
        decision = inspect_text(message, trust="untrusted_user")
        allowed = self.rate_limiter.allow(
            f"four-agent:{state.request.user_id}:{state.request.role}"
        )
        self.security_auditor.record(
            "request_screened",
            decision=decision,
            request_id=state.request.request_id,
            user_id=state.request.user_id,
            raw_text=message,
            details={"role": state.request.role},
        )
        if not allowed:
            error = CaseFileError(
                ErrorCode.RATE_LIMITED,
                "The request rate limit was exceeded.",
                stage="security.request_screen",
                agent="supervisor",
                request_id=state.request.request_id,
            )
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
            )
        if decision.action != "allow" or not decision.safe_for_model:
            error = CaseFileError(
                ErrorCode.AUTHORIZATION_DENIED,
                "Security policy blocked unsafe request content.",
                stage="security.request_screen",
                agent="supervisor",
                request_id=state.request.request_id,
                safe_details={"signals": decision.signals},
            )
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
            )
        return {"step_count": state.step_count + 1, "active_agent": "supervisor"}

    def supervisor(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "supervisor")
        if limited is not None:
            return limited
        model_start = self._model_call_count()
        activated = self._agent_event(
            state,
            "supervisor",
            "activated",
            summary="Supervisor is evaluating the active conversation goal.",
        )
        try:
            decision = self.supervisor_agent.decide(state)
            coaching_task: CoachingTask | None = None
            schedule_request: ScheduleToolCall | None = state.schedule_request
            if (
                decision.action == "delegate"
                and decision.target_agent == "skills_coach"
            ):
                coaching_task = self._coaching_task_for_decision(state, decision)
            if decision.action == "call_schedule":
                if not (
                    state.pending_confirmation is not None
                    and state.pending_confirmation.operation == "schedule_session"
                    and state.schedule_request is not None
                ):
                    schedule_request = self.supervisor_agent.prepare_schedule(
                        state,
                        decision,
                        idempotency_key=state.request.request_id,
                    )
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated],
                model_trace=self._new_model_traces(state, model_start),
            )

        model_trace = self._new_model_traces(state, model_start)
        traces = [
            activated,
            self._agent_event(
                state,
                "supervisor",
                "decision",
                reason_code=decision.reason_code,
                summary=f"Supervisor selected {decision.action}.",
                offset=1,
            ),
        ]
        update: dict[str, Any] = {
            "step_count": state.step_count + 1,
            "active_agent": "supervisor",
            "active_goal": self.supervisor_agent.active_goal(decision),
            "supervisor_decision": decision,
            "agent_trace": [*state.agent_trace, *traces],
            "model_trace": [*state.model_trace, *model_trace],
            "error": None,
        }
        if coaching_task is not None:
            update["coaching_task"] = coaching_task
        if schedule_request is not None:
            update["schedule_request"] = schedule_request

        if decision.action == "delegate":
            target = decision.target_agent
            handoff = self._agent_event(
                state,
                "supervisor",
                "handoff",
                from_agent="supervisor",
                to_agent=target,
                reason_code=decision.reason_code,
                summary=f"Supervisor handed off {decision.required_artifact}.",
                offset=2,
            )
            update["agent_trace"] = [*state.agent_trace, *traces, handoff]
            update["active_agent"] = target
            return update

        if decision.action == "ask_clarification":
            question = ClarificationRequest(
                question=decision.clarification_question
                or "What information is missing?",
                missing_fields=[],
                reason_code=decision.reason_code,
            )
            try:
                messages = self._append_message(state, "assistant", question.question)
            except CaseFileError as error:
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=traces,
                    model_trace=model_trace,
                )
            update.update(
                status="needs_input",
                pending_question=question,
                pending_confirmation=None,
                messages=messages,
            )
            return update

        if decision.action in {"finish", "refuse"}:
            try:
                messages = self._append_message(state, "assistant", decision.task)
            except CaseFileError as error:
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=traces,
                    model_trace=model_trace,
                )
            finished = self._agent_event(
                state,
                "supervisor",
                "finished",
                reason_code=decision.reason_code,
                summary="Supervisor completed the conversation turn.",
                offset=2,
            )
            update.update(
                status="completed",
                pending_question=None,
                pending_confirmation=None,
                messages=messages,
                agent_trace=[*state.agent_trace, *traces, finished],
            )
            return update

        return update

    def evidence_librarian(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "evidence_librarian")
        if limited is not None:
            return limited
        decision = self._required_delegation(state, "evidence_librarian")
        activated = self._agent_event(
            state,
            "evidence_librarian",
            "activated",
            from_agent="supervisor",
            summary="Evidence Librarian accepted the typed handoff.",
        )
        context = self._tool_context(state, "evidence_librarian")
        model_start = self._model_call_count()
        tool_records: list[dict[str, Any]] = []
        try:
            with capture_tool_invocations(tool_records):
                result = self._run_librarian(
                    state,
                    decision.required_artifact,
                    decision.task,
                    context,
                )
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated],
                tool_trace=self._tool_traces(state, tool_records),
                model_trace=self._new_model_traces(state, model_start),
            )

        tool_trace = self._tool_traces(state, tool_records)
        model_trace = self._new_model_traces(state, model_start)
        if isinstance(result, ClarificationRequest):
            try:
                messages = self._append_message(state, "assistant", result.question)
            except CaseFileError as error:
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=[activated],
                    tool_trace=tool_trace,
                    model_trace=model_trace,
                )
            returned = self._returned_event(
                state,
                "evidence_librarian",
                "Evidence Librarian returned a clarification request.",
                offset=1,
            )
            return {
                "step_count": state.step_count + 1,
                "active_agent": "supervisor",
                "status": "needs_input",
                "pending_question": result,
                "pending_confirmation": None,
                "messages": messages,
                "agent_trace": [*state.agent_trace, activated, returned],
                "tool_trace": [*state.tool_trace, *tool_trace],
                "model_trace": [*state.model_trace, *model_trace],
                "error": None,
            }

        try:
            artifacts = self._append_artifact(state, result)
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated],
                tool_trace=tool_trace,
                model_trace=model_trace,
            )
        returned = self._returned_event(
            state,
            "evidence_librarian",
            f"Evidence Librarian returned {result.artifact_type}.",
            offset=1,
        )
        update: dict[str, Any] = {
            "step_count": state.step_count + 1,
            "active_agent": "supervisor",
            "status": "running",
            "pending_question": None,
            "pending_confirmation": None,
            "artifacts": artifacts,
            "agent_trace": [*state.agent_trace, activated, returned],
            "tool_trace": [*state.tool_trace, *tool_trace],
            "model_trace": [*state.model_trace, *model_trace],
            "error": None,
        }
        if isinstance(result, EvidencePacket):
            update["evidence_packet"] = result
            if state.coaching_state is not None:
                update["coaching_state"] = state.coaching_state.model_copy(
                    update={"evidence_packet": result}
                )
        elif isinstance(result, IngestionPreview):
            confirmation = ConfirmationRequest(
                operation="commit_ingestion",
                token=result.confirmation_token,
                summary=f"Commit the staged preview for {result.source_filename}.",
            )
            try:
                messages = self._append_message(
                    state,
                    "assistant",
                    confirmation.summary,
                )
            except CaseFileError as error:
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=[activated, returned],
                    tool_trace=tool_trace,
                    model_trace=model_trace,
                )
            update.update(
                status="needs_confirmation",
                pending_confirmation=confirmation,
                ingestion_job=IngestionJobState(
                    job_id=result.job_id,
                    stage="awaiting_confirmation",
                    source_filename=result.source_filename,
                    source_sha256=result.source_sha256,
                    resolution=result.resolution,
                    side=result.side,
                ),
                messages=messages,
            )
        elif (
            isinstance(result, IngestionCommitResult)
            and state.ingestion_job is not None
        ):
            update["ingestion_job"] = state.ingestion_job.model_copy(
                update={"stage": "completed"}
            )
        return update

    def argument_strategist(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "argument_strategist")
        if limited is not None:
            return limited
        decision = self._required_delegation(state, "argument_strategist")
        activated = self._agent_event(
            state,
            "argument_strategist",
            "activated",
            from_agent="supervisor",
            summary="Argument Strategist accepted the EvidencePacket handoff.",
        )
        context = self._tool_context(state, "argument_strategist")
        model_start = self._model_call_count()
        try:
            if state.evidence_packet is None:
                raise CaseFileError(
                    ErrorCode.AGENT_OUTPUT_INVALID,
                    "Argument generation requires an EvidencePacket handoff.",
                    stage="argument_strategist.handoff",
                    agent="argument_strategist",
                    request_id=state.request.request_id,
                )
            if state.argument_request is not None and state.argument_draft is not None:
                request, draft = self.argument_strategist_agent.revise_argument(
                    context,
                    instruction=decision.task,
                    evidence_packet=state.evidence_packet,
                    previous_request=state.argument_request,
                    previous_draft=state.argument_draft,
                )
            else:
                request, draft = self.argument_strategist_agent.create_argument(
                    context,
                    original_request=decision.task,
                    evidence_packet=state.evidence_packet,
                    requested_side=state.evidence_packet.side,
                )
            artifacts = self._append_artifact(state, draft)
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated],
                model_trace=self._new_model_traces(state, model_start),
            )
        model_trace = self._new_model_traces(state, model_start)
        returned = self._returned_event(
            state,
            "argument_strategist",
            "Argument Strategist returned a validated argument_draft.",
            offset=1,
        )
        return {
            "step_count": state.step_count + 1,
            "active_agent": "supervisor",
            "status": "running",
            "pending_question": None,
            "pending_confirmation": None,
            "argument_request": request,
            "argument_draft": draft,
            "artifacts": artifacts,
            "agent_trace": [*state.agent_trace, activated, returned],
            "model_trace": [*state.model_trace, *model_trace],
            "error": None,
        }

    def skills_coach(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "skills_coach")
        if limited is not None:
            return limited
        decision = self._required_delegation(state, "skills_coach")
        activated = self._agent_event(
            state,
            "skills_coach",
            "activated",
            from_agent="supervisor",
            summary="Skills Coach accepted the typed practice handoff.",
        )
        context = self._tool_context(state, "skills_coach")
        model_start = self._model_call_count()
        tool_records: list[dict[str, Any]] = []
        try:
            with capture_tool_invocations(tool_records):
                result, extra = self._run_coach(
                    state, decision.required_artifact, context
                )
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated],
                tool_trace=self._tool_traces(state, tool_records),
                model_trace=self._new_model_traces(state, model_start),
            )
        tool_trace = self._tool_traces(state, tool_records)
        model_trace = self._new_model_traces(state, model_start)
        returned = self._returned_event(
            state,
            "skills_coach",
            (
                "Skills Coach returned an EvidenceRequest to the Supervisor."
                if isinstance(result, EvidenceRequest)
                else "Skills Coach returned its typed practice artifact."
            ),
            offset=1,
        )
        update: dict[str, Any] = {
            "step_count": state.step_count + 1,
            "active_agent": "supervisor",
            "status": "running",
            "pending_question": None,
            "pending_confirmation": None,
            "agent_trace": [*state.agent_trace, activated, returned],
            "tool_trace": [*state.tool_trace, *tool_trace],
            "model_trace": [*state.model_trace, *model_trace],
            "error": None,
            **extra,
        }
        if isinstance(result, EvidenceRequest):
            update["evidence_request"] = result
            update["evidence_packet"] = None
            if state.coaching_state is not None:
                update["coaching_state"] = state.coaching_state.model_copy(
                    update={"evidence_packet": None}
                )
            return update
        if isinstance(result, ProgressEntry):
            progress = [*state.progress_context, result]
            if len(progress) > MAX_PROGRESS_RECORDS:
                error = self._limit_error(
                    state,
                    "The progress context exceeds its configured limit.",
                    MAX_PROGRESS_RECORDS,
                )
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=[activated, returned],
                    tool_trace=tool_trace,
                    model_trace=model_trace,
                )
            update["progress_context"] = progress
            return update
        try:
            update["artifacts"] = self._append_artifact(state, result)
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                agent_trace=[activated, returned],
                tool_trace=tool_trace,
                model_trace=model_trace,
            )
        if isinstance(result, ProgressSummary):
            update["progress_context"] = result.records
        if isinstance(result, AssessmentProposal):
            confirmation = ConfirmationRequest(
                operation="log_assessment",
                token=uuid.uuid4().hex,
                summary=f"Log the proposed assessment for {result.student_id}.",
            )
            try:
                messages = self._append_message(
                    state,
                    "assistant",
                    confirmation.summary,
                )
            except CaseFileError as error:
                return self._failure_update(
                    state,
                    error,
                    step_count=state.step_count + 1,
                    agent_trace=[activated, returned],
                    tool_trace=tool_trace,
                    model_trace=model_trace,
                )
            update.update(
                status="needs_confirmation",
                pending_confirmation=confirmation,
                messages=messages,
            )
        return update

    def schedule(self, raw_state: CaseFileState) -> dict[str, Any]:
        state = self._state(raw_state)
        limited = self._start_step(state, "supervisor")
        if limited is not None:
            return limited
        call = state.schedule_request
        if call is None:
            error = CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "The Supervisor called scheduling without typed arguments.",
                stage="supervisor.schedule",
                agent="supervisor",
                request_id=state.request.request_id,
            )
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
            )
        if (
            state.pending_confirmation is not None
            and state.pending_confirmation.operation == "schedule_session"
        ):
            call = call.model_copy(
                update={"confirmation_token": state.pending_confirmation.token}
            )
        context = self._tool_context(state, "supervisor")
        tool_records: list[dict[str, Any]] = []
        try:
            with capture_tool_invocations(tool_records):
                raw = self.tools.registry.invoke(
                    "supervisor",
                    "schedule_session",
                    context,
                    call.model_dump(mode="python"),
                )
            tool_trace = self._tool_traces(state, tool_records)
            if not isinstance(raw, dict):
                raise CaseFileError(
                    ErrorCode.CALENDAR_UPSTREAM_ERROR,
                    "The calendar tool returned a malformed result.",
                    stage="supervisor.schedule",
                    agent="supervisor",
                    tool="schedule_session",
                    request_id=state.request.request_id,
                )
            if raw.get("confirmation_required"):
                token = str(raw.get("confirmation_token") or "")
                confirmation = ConfirmationRequest(
                    operation="schedule_session",
                    token=token,
                    summary=str(raw.get("summary") or "Confirm the calendar write."),
                )
                messages = self._append_message(
                    state,
                    "assistant",
                    confirmation.summary,
                )
                return {
                    "step_count": state.step_count + 1,
                    "active_agent": "supervisor",
                    "status": "needs_confirmation",
                    "pending_question": None,
                    "pending_confirmation": confirmation,
                    "messages": messages,
                    "tool_trace": [*state.tool_trace, *tool_trace],
                    "error": None,
                }
            event = self._calendar_event(raw, call)
            artifacts = self._append_artifact(state, event)
        except CaseFileError as error:
            return self._failure_update(
                state,
                error,
                step_count=state.step_count + 1,
                tool_trace=self._tool_traces(state, tool_records),
            )
        return {
            "step_count": state.step_count + 1,
            "active_agent": "supervisor",
            "status": "running",
            "pending_question": None,
            "pending_confirmation": None,
            "artifacts": artifacts,
            "tool_trace": [*state.tool_trace, *tool_trace],
            "error": None,
        }

    def route_after_screen(self, raw_state: CaseFileState) -> str:
        return END_ROUTE if self._state(raw_state).status == "failed" else "supervisor"

    def route_after_supervisor(self, raw_state: CaseFileState) -> str:
        state = self._state(raw_state)
        if state.status in {"failed", "completed"}:
            return END_ROUTE
        decision = state.supervisor_decision
        if decision is None:
            return END_ROUTE
        if decision.action == "delegate":
            return decision.target_agent or END_ROUTE
        if decision.action == "call_schedule":
            return "schedule"
        return END_ROUTE

    def route_after_specialist(self, raw_state: CaseFileState) -> str:
        return "supervisor" if self._state(raw_state).status == "running" else END_ROUTE

    def _run_librarian(
        self,
        state: CaseFileState,
        artifact: str | None,
        task: str,
        context: ToolContext,
    ) -> Artifact | ClarificationRequest:
        if artifact == "evidence_packet":
            evidence_request = state.evidence_request
            requested_side = None
            request = task
            if isinstance(evidence_request, (EvidenceRequest, EvidenceQueryPlan)) and (
                state.evidence_packet is None
            ):
                requested_side = evidence_request.side
                request = (
                    evidence_request.request_summary
                    if isinstance(evidence_request, EvidenceRequest)
                    else task
                )
            return self.evidence_librarian_agent.retrieve_evidence(
                context,
                request=request,
                requested_side=requested_side,
            )
        if artifact == "rule_packet":
            return self.evidence_librarian_agent.retrieve_rules(context, question=task)
        if artifact == "topic_packet":
            return self.evidence_librarian_agent.retrieve_topic(context)
        if artifact == "ingestion_preview":
            if not state.request.attachments:
                return ClarificationRequest(
                    question="Please attach the DOCX file to ingest.",
                    missing_fields=["attachment"],
                    reason_code="ingestion_attachment_required",
                )
            metadata = self.evidence_librarian_agent.plan_ingestion_metadata(
                context,
                request=task,
            )
            if metadata.clarification_needed:
                return ClarificationRequest(
                    question=metadata.clarification_question
                    or "Is this Pro or Con evidence?",
                    missing_fields=["side"],
                    reason_code="ingestion_metadata_required",
                )
            return self.evidence_librarian_agent.stage_ingestion(
                context,
                file_path=self.attachment_resolver(state.request.attachments[-1]),
                resolution=metadata.resolution,
                side=metadata.side,
            )
        if artifact == "ingestion_commit_result":
            pending = state.pending_confirmation
            if pending is None or pending.operation != "commit_ingestion":
                raise CaseFileError(
                    ErrorCode.CONFIRMATION_INVALID,
                    "No staged ingestion is awaiting confirmation.",
                    stage="evidence_librarian.ingestion_commit",
                    agent="evidence_librarian",
                    request_id=state.request.request_id,
                )
            return self.evidence_librarian_agent.commit_ingestion(
                context,
                confirmation_token=pending.token,
            )
        raise CaseFileError(
            ErrorCode.AGENT_OUTPUT_INVALID,
            "The Evidence Librarian received an unsupported artifact handoff.",
            stage="evidence_librarian.handoff",
            agent="evidence_librarian",
            request_id=state.request.request_id,
            safe_details={"artifact": artifact},
        )

    def _run_coach(
        self,
        state: CaseFileState,
        artifact: str | None,
        context: ToolContext,
    ) -> tuple[Artifact | EvidenceRequest | ProgressEntry, dict[str, Any]]:
        pending = state.pending_confirmation
        if pending is not None and pending.operation == "log_assessment":
            proposal = self._latest_artifact(state, AssessmentProposal)
            if proposal is None:
                raise CaseFileError(
                    ErrorCode.CONFIRMATION_INVALID,
                    "No assessment proposal is awaiting confirmation.",
                    stage="skills_coach.log_assessment",
                    agent="skills_coach",
                    request_id=state.request.request_id,
                )
            entry = self.skills_coach_agent.confirm_assessment(
                context,
                proposal=proposal,
            )
            return entry, {}

        task = state.coaching_task
        if task is None:
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "The Skills Coach handoff is missing its typed task context.",
                stage="skills_coach.handoff",
                agent="skills_coach",
                request_id=state.request.request_id,
            )
        coaching_state = state.coaching_state
        if (
            coaching_state is None
            and task.side is not None
            and task.speech_position
            and task.focus
        ):
            coaching_state = CoachingState(
                student_id=task.student_id,
                speech_position=task.speech_position,
                side=task.side,
                focus=task.focus,
                turns=[],
                evidence_packet=state.evidence_packet,
            )

        if artifact in {"drill_plan", "coach_turn"} and task.needs_evidence:
            if state.evidence_packet is None:
                request = self.skills_coach_agent.request_evidence(
                    context,
                    student_id=task.student_id,
                    speech_position=task.speech_position or "general",
                    side=task.side or "pro",
                    focus=task.focus or "debate technique",
                    intended_use="drill" if artifact == "drill_plan" else "coaching",
                    source_files=task.source_files,
                )
                return request, {"coaching_state": coaching_state}

        progress = self._latest_artifact(state, ProgressSummary)
        if artifact == "progress_summary":
            result = self.skills_coach_agent.summarize_progress(
                context,
                student_id=task.student_id,
            )
            return result, {"coaching_state": coaching_state}
        if artifact == "drill_plan":
            result = self.skills_coach_agent.generate_drill(
                context,
                student_id=task.student_id,
                speech_position=task.speech_position or "general",
                side=task.side or "pro",
                focus=task.focus or "debate technique",
                progress_summary=progress,
                evidence_packet=state.evidence_packet,
            )
            return result, {"coaching_state": coaching_state}
        if artifact == "coach_turn":
            if coaching_state is None:
                raise CaseFileError(
                    ErrorCode.AGENT_OUTPUT_INVALID,
                    "A coaching turn requires typed coaching state.",
                    stage="skills_coach.handoff",
                    agent="skills_coach",
                    request_id=state.request.request_id,
                )
            student_message = self._latest_user_message(state)
            result = self.skills_coach_agent.coach_turn(
                context,
                student_id=coaching_state.student_id,
                speech_position=coaching_state.speech_position,
                side=coaching_state.side,
                focus=coaching_state.focus,
                student_message=student_message,
                prior_turns=coaching_state.turns,
                progress_summary=progress,
                evidence_packet=state.evidence_packet,
            )
            turns = [
                *coaching_state.turns,
                ConversationMessage(role="user", content=student_message),
                ConversationMessage(
                    role="assistant",
                    content=f"{result.feedback}\n\n{result.question}",
                ),
            ]
            if len(turns) > MAX_COACHING_TURNS:
                raise self._limit_error(
                    state,
                    "The coaching session exceeds its configured turn limit.",
                    MAX_COACHING_TURNS,
                )
            return result, {
                "coaching_state": coaching_state.model_copy(
                    update={"turns": turns, "evidence_packet": state.evidence_packet}
                )
            }
        if artifact == "assessment_proposal":
            if coaching_state is None:
                raise CaseFileError(
                    ErrorCode.REQUEST_INVALID,
                    "An assessment proposal requires an active coaching session.",
                    stage="skills_coach.propose_assessment",
                    agent="skills_coach",
                    request_id=state.request.request_id,
                )
            result = self.skills_coach_agent.propose_assessment(
                context,
                student_id=coaching_state.student_id,
                speech_position=coaching_state.speech_position,
                coaching_turns=coaching_state.turns,
            )
            return result, {"coaching_state": coaching_state}
        raise CaseFileError(
            ErrorCode.AGENT_OUTPUT_INVALID,
            "The Skills Coach received an unsupported artifact handoff.",
            stage="skills_coach.handoff",
            agent="skills_coach",
            request_id=state.request.request_id,
            safe_details={"artifact": artifact},
        )

    def _coaching_task_for_decision(
        self, state: CaseFileState, decision: Any
    ) -> CoachingTask:
        desired_operation = {
            "drill_plan": "generate_drill",
            "coach_turn": "coach_turn",
            "progress_summary": "progress_summary",
            "assessment_proposal": "assessment_proposal",
        }.get(decision.required_artifact)
        if (
            state.coaching_task is not None
            and state.evidence_request is not None
            and state.evidence_packet is not None
            and state.coaching_task.operation == desired_operation
        ):
            return state.coaching_task
        if state.coaching_state is not None and decision.required_artifact in {
            "coach_turn",
            "assessment_proposal",
        }:
            operation = (
                "coach_turn"
                if decision.required_artifact == "coach_turn"
                else "assessment_proposal"
            )
            return CoachingTask(
                operation=operation,
                student_id=state.coaching_state.student_id,
                speech_position=state.coaching_state.speech_position,
                side=state.coaching_state.side,
                focus=state.coaching_state.focus,
                needs_evidence=False,
                source_files=[],
            )
        return self.supervisor_agent.prepare_coaching(state, decision)

    def _required_delegation(self, state: CaseFileState, agent: AgentName) -> Any:
        decision = state.supervisor_decision
        if (
            decision is None
            or decision.action != "delegate"
            or decision.target_agent != agent
        ):
            raise CaseFileError(
                ErrorCode.AGENT_OUTPUT_INVALID,
                "A specialist was invoked without a matching Supervisor handoff.",
                stage="runtime.handoff",
                agent=agent,
                request_id=state.request.request_id,
            )
        return decision

    def _start_step(
        self, state: CaseFileState, agent: AgentName
    ) -> dict[str, Any] | None:
        if state.step_count < self.max_steps:
            return None
        error = CaseFileError(
            ErrorCode.AGENT_STEP_LIMIT_EXCEEDED,
            "The agent graph reached its configured global step limit.",
            stage="runtime.step_limit",
            agent=agent,
            request_id=state.request.request_id,
            safe_details={"limit": self.max_steps},
        )
        return self._failure_update(state, error, step_count=state.step_count)

    def _failure_update(
        self,
        state: CaseFileState,
        error: CaseFileError,
        *,
        step_count: int,
        agent_trace: list[AgentTraceEntry] | None = None,
        tool_trace: list[ToolTraceEntry] | None = None,
        model_trace: list[ModelTraceEntry] | None = None,
    ) -> dict[str, Any]:
        error.with_request_id(state.request.request_id)
        active: AgentName = (
            error.agent
            if error.agent
            in {
                "supervisor",
                "evidence_librarian",
                "argument_strategist",
                "skills_coach",
            }
            else state.active_agent
        )
        return {
            "step_count": step_count,
            "active_agent": active,
            "status": "failed",
            "pending_question": None,
            "pending_confirmation": None,
            "error": error.public_detail(),
            "agent_trace": [*state.agent_trace, *(agent_trace or [])],
            "tool_trace": [*state.tool_trace, *(tool_trace or [])],
            "model_trace": [*state.model_trace, *(model_trace or [])],
        }

    def _agent_event(
        self,
        state: CaseFileState,
        agent: AgentName,
        event: str,
        *,
        summary: str,
        from_agent: AgentName | None = None,
        to_agent: AgentName | None = None,
        reason_code: str | None = None,
        offset: int = 0,
    ) -> AgentTraceEntry:
        return AgentTraceEntry(
            sequence=len(state.agent_trace) + offset,
            agent=agent,
            event=event,
            from_agent=from_agent,
            to_agent=to_agent,
            reason_code=reason_code,
            summary=summary,
        )

    def _returned_event(
        self,
        state: CaseFileState,
        agent: AgentName,
        summary: str,
        *,
        offset: int,
    ) -> AgentTraceEntry:
        return self._agent_event(
            state,
            agent,
            "returned",
            from_agent=agent,
            to_agent="supervisor",
            summary=summary,
            offset=offset,
        )

    def _model_call_count(self) -> int:
        calls = getattr(self.model, "calls", None)
        return len(calls) if isinstance(calls, list) else 0

    def _new_model_traces(
        self,
        state: CaseFileState,
        start: int,
    ) -> list[ModelTraceEntry]:
        calls = getattr(self.model, "calls", None)
        if not isinstance(calls, list):
            return []
        entries: list[ModelTraceEntry] = []
        for raw in calls[start:]:
            get = (
                raw.get
                if isinstance(raw, dict)
                else lambda key, default=None: getattr(raw, key, default)
            )
            started_at = get("started_at")
            if not isinstance(started_at, datetime):
                continue
            agent = get("agent", "supervisor")
            if agent not in {
                "supervisor",
                "evidence_librarian",
                "argument_strategist",
                "skills_coach",
            }:
                agent = "supervisor"
            try:
                entries.append(
                    ModelTraceEntry(
                        sequence=len(state.model_trace) + len(entries),
                        agent=agent,
                        model=str(get("model", "unknown")),
                        prompt_template=str(get("prompt_template", "unknown")),
                        prompt_sha256=str(get("prompt_sha256", "")),
                        response_sha256=get("response_sha256"),
                        schema_name=str(get("schema_name", "object")),
                        started_at=started_at,
                        latency_ms=get("latency_ms"),
                        stop_reason=get("stop_reason"),
                        input_tokens=get("input_tokens"),
                        output_tokens=get("output_tokens"),
                        status=get("status", "failed"),
                        error_code=get("error_code"),
                        rendered_system_prompt=get("rendered_system_prompt"),
                        rendered_user_payload=get("rendered_user_payload"),
                        model_response=get("model_response"),
                    )
                )
            except ValidationError as exc:
                raise CaseFileError(
                    ErrorCode.INTERNAL_ERROR,
                    "A model trace record was malformed.",
                    stage="runtime.model_trace",
                    request_id=state.request.request_id,
                    cause=exc,
                ) from exc
        return entries

    @staticmethod
    def _tool_traces(
        state: CaseFileState,
        records: list[dict[str, Any]],
    ) -> list[ToolTraceEntry]:
        return [
            ToolTraceEntry(
                sequence=len(state.tool_trace) + index,
                **record,
            )
            for index, record in enumerate(records)
        ]

    @staticmethod
    def _state(raw_state: CaseFileState | dict[str, Any]) -> CaseFileState:
        return (
            raw_state
            if isinstance(raw_state, CaseFileState)
            else CaseFileState.model_validate(raw_state)
        )

    @staticmethod
    def _latest_user_message(state: CaseFileState) -> str:
        for message in reversed(state.messages):
            if message.role == "user":
                return message.content
        raise CaseFileError(
            ErrorCode.REQUEST_INVALID,
            "The conversation does not contain a user message.",
            stage="runtime.message",
            request_id=state.request.request_id,
        )

    @staticmethod
    def _append_message(
        state: CaseFileState,
        role: str,
        content: str,
    ) -> list[ConversationMessage]:
        try:
            messages = [
                *state.messages,
                ConversationMessage(role=role, content=content),
            ]
            if len(messages) > MAX_MESSAGES:
                raise ValueError("message limit exceeded")
            return messages
        except (IndexError, TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.STATE_LIMIT_EXCEEDED,
                "The conversation message limit was exceeded.",
                stage="runtime.state.messages",
                request_id=state.request.request_id,
                cause=exc,
            ) from exc

    @staticmethod
    def _append_artifact(state: CaseFileState, artifact: Artifact) -> list[Artifact]:
        if len(state.artifacts) >= MAX_ARTIFACTS:
            raise CaseFileError(
                ErrorCode.STATE_LIMIT_EXCEEDED,
                "The session artifact limit was exceeded.",
                stage="runtime.state.artifacts",
                request_id=state.request.request_id,
                safe_details={"limit": MAX_ARTIFACTS},
            )
        return [*state.artifacts, artifact]

    @staticmethod
    def _latest_artifact(state: CaseFileState, kind: type[Any]) -> Any | None:
        return next(
            (
                artifact
                for artifact in reversed(state.artifacts)
                if isinstance(artifact, kind)
            ),
            None,
        )

    @staticmethod
    def _tool_context(state: CaseFileState, agent: AgentName) -> ToolContext:
        return ToolContext(
            role=state.request.role,
            user_id=state.request.user_id,
            resolution=state.request.active_resolution,
            request_id=state.request.request_id,
            agent=agent,
        )

    @staticmethod
    def _calendar_event(raw: Any, call: ScheduleToolCall) -> CalendarEvent:
        if isinstance(raw, CalendarEvent):
            return raw
        if not isinstance(raw, dict):
            raise CaseFileError(
                ErrorCode.CALENDAR_UPSTREAM_ERROR,
                "The calendar tool returned a malformed event.",
                stage="supervisor.schedule",
                agent="supervisor",
                tool="schedule_session",
            )
        try:
            start_value = raw.get("start")
            end_value = raw.get("end")
            start = (
                start_value.get("dateTime")
                if isinstance(start_value, dict)
                else start_value
            )
            end = (
                end_value.get("dateTime") if isinstance(end_value, dict) else end_value
            )
            timezone_name = (
                start_value.get("timeZone")
                if isinstance(start_value, dict)
                else call.timezone_name
            ) or call.timezone_name
            attendees = raw.get("attendees") or []
            attendee = call.attendee_email
            if attendee is None and attendees and isinstance(attendees[0], dict):
                attendee = attendees[0].get("email")
            fixture = bool(raw.get("synthetic"))
            return CalendarEvent(
                event_id=str(raw.get("id") or raw.get("event_id") or ""),
                student_id=call.student_id,
                start=datetime.fromisoformat(str(start).replace("Z", "+00:00")),
                end=datetime.fromisoformat(str(end).replace("Z", "+00:00")),
                timezone=str(timezone_name),
                attendee_email=attendee,
                backend="fixture" if fixture else "google",
                synthetic=fixture,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise CaseFileError(
                ErrorCode.CALENDAR_UPSTREAM_ERROR,
                "The calendar tool returned a malformed event.",
                stage="supervisor.schedule",
                agent="supervisor",
                tool="schedule_session",
                cause=exc,
            ) from exc

    @staticmethod
    def _limit_error(state: CaseFileState, message: str, limit: int) -> CaseFileError:
        return CaseFileError(
            ErrorCode.STATE_LIMIT_EXCEEDED,
            message,
            stage="runtime.state",
            request_id=state.request.request_id,
            safe_details={"limit": limit},
        )


def compile_four_agent_graph(nodes: FourAgentGraphNodes) -> Any:
    """Compile the sole production orchestration path."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise CaseFileError(
            ErrorCode.CONFIGURATION_ERROR,
            "LangGraph is required but is not installed.",
            stage="runtime.langgraph",
            cause=exc,
        ) from exc

    graph = StateGraph(CaseFileState)
    graph.add_node("screen_request", nodes.screen_request)
    graph.add_node("supervisor", nodes.supervisor)
    graph.add_node("evidence_librarian", nodes.evidence_librarian)
    graph.add_node("argument_strategist", nodes.argument_strategist)
    graph.add_node("skills_coach", nodes.skills_coach)
    graph.add_node("schedule", nodes.schedule)
    graph.add_edge(START, "screen_request")
    graph.add_conditional_edges(
        "screen_request",
        nodes.route_after_screen,
        {"supervisor": "supervisor", END_ROUTE: END},
    )
    graph.add_conditional_edges(
        "supervisor",
        nodes.route_after_supervisor,
        {
            "evidence_librarian": "evidence_librarian",
            "argument_strategist": "argument_strategist",
            "skills_coach": "skills_coach",
            "schedule": "schedule",
            END_ROUTE: END,
        },
    )
    for specialist in (
        "evidence_librarian",
        "argument_strategist",
        "skills_coach",
        "schedule",
    ):
        graph.add_conditional_edges(
            specialist,
            nodes.route_after_specialist,
            {"supervisor": "supervisor", END_ROUTE: END},
        )
    return graph.compile()


__all__ = ["FourAgentGraphNodes", "compile_four_agent_graph"]
