from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from casefile.agents.contracts import (
    ConversationMessage,
    EvidenceCard,
    EvidencePacket,
    EvidenceProvenance,
    EvidenceRequest,
    ProgressEntry,
    ProgressSummary,
)
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.agents.skills_coach import SkillsCoach
from casefile.tools import CaseFileTools, ToolContext


CARD_ID = "c" * 64
RESOLUTION = "Resolved: Example"


class ScriptedModel:
    def __init__(self, *responses: dict[str, Any] | CaseFileError) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, CaseFileError):
            raise response
        return response


class StubRegistry:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = list(records or [])
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def invoke(self, agent, name, context, arguments):
        self.calls.append((agent, name, arguments))
        if name == "get_progress":
            return list(self.records)
        raise AssertionError(f"Unexpected tool call: {name}")


class StubTools:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.registry = StubRegistry(records)


def _context(*, role: str = "student", user_id: str = "student-1") -> ToolContext:
    return ToolContext(
        role=role,
        user_id=user_id,
        resolution=RESOLUTION,
        request_id="request-7",
        agent="skills_coach",
    )


def _raw_record() -> dict[str, Any]:
    return {
        "student_id": "student-1",
        "date": "2026-08-01",
        "speech_position": "summary",
        "resolution": RESOLUTION,
        "weakness_tags": ["collapse", "comparison"],
        "assessment_text": "Make the internal link and comparison explicit.",
        "author_role": "coach",
        "author_id": "coach-1",
    }


def _entry() -> ProgressEntry:
    return ProgressEntry(
        student_id="student-1",
        date=date(2026, 8, 1),
        speech_position="summary",
        resolution=RESOLUTION,
        weakness_tags=["collapse", "comparison"],
        assessment_text="Make the internal link and comparison explicit.",
        author_id="coach-1",
    )


def _summary() -> ProgressSummary:
    return ProgressSummary(
        student_id="student-1",
        records=[_entry()],
        summary="Prior feedback prioritizes collapse and comparison.",
    )


def _packet() -> EvidencePacket:
    return EvidencePacket(
        request_summary="Find evidence for summary practice.",
        resolution=RESOLUTION,
        side="pro",
        confirmed_source_files_considered=["confirmed.docx"],
        queries_executed=["consumer protection impact"],
        cards=[
            EvidenceCard(
                card_id=CARD_ID,
                source_filename="confirmed.docx",
                citation="Research Institute, 2026.",
                header="Research Institute '26",
                tag="Consumer protections reduce losses.",
                body="The preserved source body documents lower consumer losses.",
                read_spans=[],
                emphasis_spans=[],
                resolution=RESOLUTION,
                side="pro",
                retrieval_score=0.87,
            )
        ],
        empty_result=False,
        provenance=EvidenceProvenance(
            ledger_schema_version=1,
            retrieval_backend="chroma",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            confirmed_only=True,
        ),
    )


def test_progress_summary_uses_authorized_records_without_changing_them() -> None:
    tools = StubTools([_raw_record()])
    response = {
        "artifact_type": "progress_summary",
        "student_id": "student-1",
        "records": [_entry().model_dump(mode="json")],
        "summary": "Prior feedback prioritizes collapse and comparison.",
    }
    model = ScriptedModel(response)

    summary = SkillsCoach(tools, model).summarize_progress(
        _context(),
        student_id="student-1",
    )

    assert summary.records == [_entry()]
    assert tools.registry.calls == [
        ("skills_coach", "get_progress", {"student_id": "student-1"})
    ]
    assert model.calls[0]["schema"] is ProgressSummary
    assert model.calls[0]["prompt_template"] == "agents/prompts/skills_coach.md"


def test_evidence_request_is_a_handoff_and_never_calls_retrieval() -> None:
    tools = StubTools()
    response = {
        "request_summary": "Find Pro evidence for impact comparison practice.",
        "resolution": RESOLUTION,
        "side": "pro",
        "subject": "consumer impact comparison",
        "entities": ["consumers"],
        "source_files": ["confirmed.docx"],
        "intended_use": "drill",
    }
    model = ScriptedModel(response)

    request = SkillsCoach(tools, model).request_evidence(
        _context(),
        student_id="student-1",
        speech_position="summary",
        side="pro",
        focus="impact comparison",
        intended_use="drill",
        progress_summary=_summary(),
        source_files=["confirmed.docx"],
    )

    assert request.subject == "consumer impact comparison"
    assert tools.registry.calls == []
    assert model.calls[0]["schema"] is EvidenceRequest


def test_drill_and_coach_turn_are_personalized_and_evidence_bounded() -> None:
    drill_response = {
        "artifact_type": "drill_plan",
        "student_id": "student-1",
        "speech_position": "summary",
        "resolution": RESOLUTION,
        "side": "pro",
        "title": "Collapse and comparison drill",
        "focus": ["collapse", "impact comparison"],
        "instructions": [
            "State the extension yourself, cite the supplied card, and compare impacts."
        ],
        "duration_minutes": 12,
        "evidence_card_ids": [CARD_ID],
        "personalization_summary": "Targets the recorded collapse weakness.",
    }
    turn_response = {
        "artifact_type": "coach_turn",
        "label": "simulated_coach",
        "student_id": "student-1",
        "speech_position": "summary",
        "side": "pro",
        "focus": "impact comparison",
        "feedback": "Your extension is clear, but the comparison needs a metric.",
        "question": "Which impact is faster and why?",
        "evidence_card_ids": [CARD_ID],
        "continue_session": True,
    }
    model = ScriptedModel(drill_response, turn_response)
    coach = SkillsCoach(StubTools(), model)

    drill = coach.generate_drill(
        _context(),
        student_id="student-1",
        speech_position="summary",
        side="pro",
        focus="collapse and impact comparison",
        progress_summary=_summary(),
        evidence_packet=_packet(),
    )
    turn = coach.coach_turn(
        _context(),
        student_id="student-1",
        speech_position="summary",
        side="pro",
        focus="impact comparison",
        student_message="Our impact happens sooner.",
        prior_turns=[
            ConversationMessage(role="assistant", content="Which impact comes first?")
        ],
        progress_summary=_summary(),
        evidence_packet=_packet(),
    )

    assert drill.evidence_card_ids == [CARD_ID]
    assert "collapse" in drill.personalization_summary
    assert turn.label == "simulated_coach"
    assert turn.continue_session is True
    assert turn.evidence_card_ids == [CARD_ID]


def test_invented_coaching_card_id_is_rejected() -> None:
    response = {
        "artifact_type": "coach_turn",
        "label": "simulated_coach",
        "student_id": "student-1",
        "speech_position": "summary",
        "side": "pro",
        "focus": "impact comparison",
        "feedback": "Use the cited card.",
        "question": "What does it establish?",
        "evidence_card_ids": ["d" * 64],
        "continue_session": True,
    }

    with pytest.raises(CaseFileError) as caught:
        SkillsCoach(StubTools(), ScriptedModel(response)).coach_turn(
            _context(),
            student_id="student-1",
            speech_position="summary",
            side="pro",
            focus="impact comparison",
            student_message="I would use the evidence.",
            progress_summary=_summary(),
            evidence_packet=_packet(),
        )

    assert caught.value.code == ErrorCode.AGENT_OUTPUT_INVALID
    assert caught.value.safe_details == {"field": "evidence_card_ids"}


def test_assessment_is_proposed_before_a_separate_coach_only_write(
    isolated_settings,
) -> None:
    proposal_response = {
        "artifact_type": "assessment_proposal",
        "student_id": "student-1",
        "speech_position": "summary",
        "resolution": RESOLUTION,
        "weakness_tags": ["comparison"],
        "assessment_text": "The student needs a consistent comparison metric.",
        "confirmation_required": True,
    }
    tools = CaseFileTools(isolated_settings)
    coach = SkillsCoach(tools, ScriptedModel(proposal_response))
    coach_context = _context(role="coach", user_id="coach-1")
    transcript = [
        ConversationMessage(role="user", content="My impact happens sooner."),
        ConversationMessage(
            role="assistant",
            content="Which comparison metric establishes that?",
        ),
    ]

    proposal = coach.propose_assessment(
        coach_context,
        student_id="student-1",
        speech_position="summary",
        coaching_turns=transcript,
    )
    assert json.loads(isolated_settings.progress_path.read_text(encoding="utf-8")) == []

    with pytest.raises(CaseFileError) as unconfirmed:
        coach.log_assessment(
            coach_context,
            proposal=proposal,
            confirmed=False,
        )
    assert unconfirmed.value.code == ErrorCode.CONFIRMATION_INVALID
    assert json.loads(isolated_settings.progress_path.read_text(encoding="utf-8")) == []

    with pytest.raises(CaseFileError) as unauthorized:
        coach.confirm_assessment(
            _context(),
            proposal=proposal,
        )
    assert unauthorized.value.code == ErrorCode.AUTHORIZATION_DENIED

    written = coach.confirm_assessment(
        coach_context,
        proposal=proposal,
        idempotency_key="assessment-1",
    )
    assert written.student_id == "student-1"
    assert written.author_id == "coach-1"
    assert (
        len(json.loads(isolated_settings.progress_path.read_text(encoding="utf-8")))
        == 1
    )


@pytest.mark.parametrize("operation", ["drill", "coaching"])
def test_model_failure_never_produces_a_deterministic_coaching_artifact(
    operation: str,
) -> None:
    failure = CaseFileError(
        ErrorCode.MODEL_UPSTREAM_ERROR,
        "The configured model could not be reached.",
        stage="model.request",
        agent="skills_coach",
    )
    coach = SkillsCoach(StubTools(), ScriptedModel(failure))

    with pytest.raises(CaseFileError) as caught:
        if operation == "drill":
            coach.generate_drill(
                _context(),
                student_id="student-1",
                speech_position="summary",
                side="pro",
                focus="collapse",
                progress_summary=_summary(),
                evidence_packet=_packet(),
            )
        else:
            coach.coach_turn(
                _context(),
                student_id="student-1",
                speech_position="summary",
                side="pro",
                focus="collapse",
                student_message="Here is my comparison.",
                progress_summary=_summary(),
                evidence_packet=_packet(),
            )

    assert caught.value is failure
    assert caught.value.request_id == "request-7"
