from __future__ import annotations

import json
from typing import Any

import pytest

from casefile.agents.contracts import (
    CoachingTask,
    DrillPlan,
    EvidenceQueryPlan,
    EvidenceRequest,
    ProgressSummary,
    ScheduleToolCall,
    SupervisorDecision,
)
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.agents.runtime import CaseFileRuntime
from casefile.agents.session import CaseFileSessionStore


def _decision(
    action: str,
    *,
    target: str | None = None,
    artifact: str | None = None,
    task: str = "",
    question: str | None = None,
    reason: str = "test",
) -> dict[str, Any]:
    return {
        "action": action,
        "target_agent": target,
        "goal": "Complete the user's request.",
        "task": task,
        "required_artifact": artifact,
        "clarification_question": question,
        "reason_code": reason,
    }


class ScriptedModel:
    model = "scripted-supervisor"

    def __init__(self, responses: list[tuple[type[Any], dict[str, Any]]]) -> None:
        self.responses = list(responses)

    def validate_configuration(self) -> None:
        return None

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        expected, response = self.responses.pop(0)
        assert kwargs["schema"] is expected
        return response


class ScheduleRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, agent, name, context, arguments):
        assert agent == "supervisor"
        assert name == "schedule_session"
        self.calls.append(arguments)
        if arguments.get("confirmation_token") is None:
            return {
                "confirmation_required": True,
                "confirmation_token": "f" * 32,
                "summary": "Confirm the staged calendar event.",
            }
        return {
            "id": "fixture-1",
            "synthetic": True,
            "start": {
                "dateTime": "2026-08-10T16:00:00-07:00",
                "timeZone": "America/Los_Angeles",
            },
            "end": {
                "dateTime": "2026-08-10T16:45:00-07:00",
                "timeZone": "America/Los_Angeles",
            },
        }


class ScheduleTools:
    def __init__(self) -> None:
        self.registry = ScheduleRegistry()


def test_supervisor_delegates_research_and_persists_full_session(
    isolated_settings,
) -> None:
    model = ScriptedModel(
        [
            (
                SupervisorDecision,
                _decision(
                    "delegate",
                    target="evidence_librarian",
                    artifact="evidence_packet",
                    task="Find Pro evidence about poverty.",
                    reason="research_required",
                ),
            ),
            (
                EvidenceQueryPlan,
                {
                    "resolution": "Resolved: Example",
                    "side": "pro",
                    "subject": "poverty",
                    "entities": [],
                    "source_files": [],
                    "queries": ["poverty"],
                    "result_limit": 5,
                    "clarification_needed": False,
                    "clarification_question": None,
                },
            ),
            (
                SupervisorDecision,
                _decision(
                    "finish",
                    task="No confirmed matching cards were found.",
                    reason="research_complete",
                ),
            ),
            (
                SupervisorDecision,
                _decision(
                    "finish",
                    task="The prior evidence result remains available in this session.",
                    reason="session_continued",
                ),
            ),
        ]
    )
    runtime = CaseFileRuntime(isolated_settings, model=model)
    first = runtime.ask(
        "Find Pro evidence about poverty.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-research-1",
        session_id="session-research-0001",
    )

    assert first.status == "completed"
    assert first.evidence_packet is not None
    assert first.evidence_packet.empty_result is True
    assert [entry.event for entry in first.agent_trace] == [
        "activated",
        "decision",
        "handoff",
        "activated",
        "returned",
        "activated",
        "decision",
        "finished",
    ]
    assert first.agent_trace[2].to_agent == "evidence_librarian"

    second = runtime.ask(
        "Keep that result in this conversation.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-research-2",
        session_id=first.request.session_id,
    )
    assert second.evidence_packet == first.evidence_packet
    assert len(second.messages) == len(first.messages) + 2
    assert second.step_count > first.step_count
    assert model.responses == []


def test_schedule_clarification_and_confirmation_continue_in_one_session(
    isolated_settings,
) -> None:
    tools = ScheduleTools()
    model = ScriptedModel(
        [
            (
                SupervisorDecision,
                _decision(
                    "ask_clarification",
                    question="What start time and timezone should I use?",
                    reason="schedule_time_required",
                ),
            ),
            (
                SupervisorDecision,
                _decision(
                    "call_schedule",
                    artifact="calendar_event",
                    task="Stage the requested coaching session.",
                    reason="schedule_ready",
                ),
            ),
            (
                ScheduleToolCall,
                {
                    "student_id": "student-1",
                    "start": "2026-08-10T16:00:00-07:00",
                    "duration_minutes": 45,
                    "attendee_email": None,
                    "timezone_name": "America/Los_Angeles",
                    "confirmation_token": None,
                    "idempotency_key": None,
                },
            ),
            (
                SupervisorDecision,
                _decision(
                    "call_schedule",
                    artifact="calendar_event",
                    task="Confirm the staged coaching session.",
                    reason="schedule_confirmed",
                ),
            ),
            (
                SupervisorDecision,
                _decision(
                    "finish",
                    task="The coaching session is scheduled.",
                    reason="schedule_complete",
                ),
            ),
        ]
    )
    runtime = CaseFileRuntime(
        isolated_settings,
        model=model,
        tools=tools,
    )
    session_id = "session-schedule-0001"
    missing = runtime.ask(
        "Schedule a coaching session for me.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-schedule-1",
        session_id=session_id,
    )
    assert missing.status == "needs_input"
    assert missing.pending_question is not None

    staged = runtime.ask(
        "Use 2026-08-10 at 4 PM America/Los_Angeles for 45 minutes.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-schedule-2",
        session_id=session_id,
    )
    assert staged.status == "needs_confirmation"
    assert staged.pending_confirmation is not None
    assert staged.pending_confirmation.operation == "schedule_session"

    confirmed = runtime.ask(
        "Confirm that calendar event.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-schedule-3",
        session_id=session_id,
    )
    assert confirmed.status == "completed"
    assert confirmed.pending_confirmation is None
    assert confirmed.artifacts[-1].artifact_type == "calendar_event"
    assert tools.registry.calls[0]["confirmation_token"] is None
    assert tools.registry.calls[1]["confirmation_token"] == "f" * 32
    assert model.responses == []


def test_coach_evidence_request_returns_through_supervisor_and_librarian(
    isolated_settings,
) -> None:
    model = ScriptedModel(
        [
            (
                SupervisorDecision,
                _decision(
                    "delegate",
                    target="skills_coach",
                    artifact="drill_plan",
                    task="Build a Pro summary drill about poverty using evidence.",
                    reason="practice_required",
                ),
            ),
            (
                CoachingTask,
                {
                    "operation": "generate_drill",
                    "student_id": "student-1",
                    "speech_position": "summary",
                    "side": "pro",
                    "focus": "poverty impact weighing",
                    "needs_evidence": True,
                    "source_files": [],
                },
            ),
            (
                EvidenceRequest,
                {
                    "request_summary": "Find Pro poverty impact evidence.",
                    "resolution": "Resolved: Example",
                    "side": "pro",
                    "subject": "poverty impacts",
                    "entities": [],
                    "source_files": [],
                    "intended_use": "drill",
                },
            ),
            (
                SupervisorDecision,
                _decision(
                    "delegate",
                    target="evidence_librarian",
                    artifact="evidence_packet",
                    task="Retrieve the Skills Coach EvidenceRequest.",
                    reason="coach_evidence_required",
                ),
            ),
            (
                EvidenceQueryPlan,
                {
                    "resolution": "Resolved: Example",
                    "side": "pro",
                    "subject": "poverty impacts",
                    "entities": [],
                    "source_files": [],
                    "queries": ["poverty impacts"],
                    "result_limit": 5,
                    "clarification_needed": False,
                    "clarification_question": None,
                },
            ),
            (
                SupervisorDecision,
                _decision(
                    "delegate",
                    target="skills_coach",
                    artifact="drill_plan",
                    task="Complete the evidence-aware drill.",
                    reason="coach_resume",
                ),
            ),
            (
                ProgressSummary,
                {
                    "artifact_type": "progress_summary",
                    "student_id": "student-1",
                    "records": [],
                    "summary": "No recorded progress history is available.",
                },
            ),
            (
                DrillPlan,
                {
                    "artifact_type": "drill_plan",
                    "student_id": "student-1",
                    "speech_position": "summary",
                    "resolution": "Resolved: Example",
                    "side": "pro",
                    "title": "Poverty impact weighing",
                    "focus": ["impact weighing"],
                    "instructions": [
                        "Explain the impact comparison in your own words."
                    ],
                    "duration_minutes": 10,
                    "evidence_card_ids": [],
                    "personalization_summary": "No recorded history was available.",
                },
            ),
            (
                SupervisorDecision,
                _decision(
                    "finish",
                    task="Your personalized drill is ready.",
                    reason="practice_complete",
                ),
            ),
        ]
    )
    runtime = CaseFileRuntime(isolated_settings, model=model)
    state = runtime.ask(
        "Give me a Pro summary drill about poverty using evidence.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        session_id="session-coach-handoff-1",
    )

    assert state.status == "completed"
    assert [artifact.artifact_type for artifact in state.artifacts] == [
        "evidence_packet",
        "drill_plan",
    ]
    handoff_targets = [
        entry.to_agent for entry in state.agent_trace if entry.event == "handoff"
    ]
    assert handoff_targets == [
        "skills_coach",
        "evidence_librarian",
        "skills_coach",
    ]
    assert model.responses == []


def test_global_step_limit_fails_visibly_without_invoking_specialist(
    isolated_settings,
) -> None:
    model = ScriptedModel(
        [
            (
                SupervisorDecision,
                _decision(
                    "delegate",
                    target="evidence_librarian",
                    artifact="evidence_packet",
                    task="Find Pro evidence.",
                    reason="research_required",
                ),
            )
        ]
    )
    runtime = CaseFileRuntime(isolated_settings, model=model, max_steps=2)
    state = runtime.ask(
        "Find Pro evidence.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        session_id="session-step-limit-01",
    )

    assert state.status == "failed"
    assert state.error is not None
    assert state.error.code == ErrorCode.AGENT_STEP_LIMIT_EXCEEDED
    assert state.step_count == 2


def test_invalid_supervisor_output_fails_without_a_router_substitute(
    isolated_settings,
) -> None:
    model = ScriptedModel([(SupervisorDecision, {"action": "delegate"})])
    runtime = CaseFileRuntime(isolated_settings, model=model)
    state = runtime.ask(
        "Find Pro evidence.",
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        session_id="session-invalid-supervisor",
    )

    assert state.status == "failed"
    assert state.error is not None
    assert state.error.code == ErrorCode.AGENT_OUTPUT_INVALID
    assert state.error.agent == "supervisor"
    assert model.responses == []


def test_corrupt_and_unversioned_sessions_are_visible_and_not_deleted(
    isolated_settings,
) -> None:
    store = CaseFileSessionStore(isolated_settings)
    session_id = "session-corrupt-00001"
    path = store._path(session_id)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(CaseFileError) as corrupt:
        store.load(session_id, required=True)
    assert corrupt.value.code == ErrorCode.SESSION_CORRUPT
    assert path.exists()

    path.write_text(json.dumps({"state": {}}), encoding="utf-8")
    with pytest.raises(CaseFileError) as unsupported:
        store.load(session_id, required=True)
    assert unsupported.value.code == ErrorCode.SESSION_VERSION_UNSUPPORTED
    assert path.exists()
