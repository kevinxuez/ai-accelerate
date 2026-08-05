from __future__ import annotations

import pytest

from casefile.agent.graph import CaseFileAgent
from casefile.agent.nodes import deterministic_classification


def _classify(message: str, *, role: str = "student"):
    return deterministic_classification(
        message,
        {"role": role, "resolution": "R1"},
    )


@pytest.mark.parametrize(
    ("message", "role", "expected_fragment"),
    [
        ("How can I add evidence?", "student", "attach or provide the DOCX"),
        ("I want to load cards.", "coach", "attach or provide the DOCX"),
        ("Please import a document.", "coach", "attach or provide the DOCX"),
        ("Parse the attached DOCX evidence file.", "coach", "attach or provide the DOCX"),
        ("Store this source in the casefile.", "coach", "attach or provide the DOCX"),
    ],
)
def test_add_evidence_language_routes_to_ingestion(
    message,
    role,
    expected_fragment,
):
    result = _classify(message, role=role)
    assert result["intent"] == "ingest_cards"
    assert result["clarification_needed"] is True
    assert expected_fragment in result["clarification_question"]


@pytest.mark.parametrize(
    ("message", "side"),
    [
        ("Find Pro evidence about regulation.", "pro"),
        ("Show me the Con cards.", "con"),
        ("Do we have any evidence for the affirmative?", "pro"),
        ("Look up sources for the negative.", "con"),
    ],
)
def test_explicit_search_language_routes_to_retrieval(message, side):
    result = _classify(message)
    assert result["intent"] == "retrieve_evidence"
    assert result["side"] == side
    assert result["clarification_needed"] is False


@pytest.mark.parametrize("message", ["Evidence", "I need help with evidence."])
def test_underspecified_evidence_asks_which_operation(message):
    result = _classify(message)
    assert result["intent"] == "unknown"
    assert result["clarification_needed"] is True
    assert "search existing evidence or import a DOCX file" in result[
        "clarification_question"
    ]


def test_side_without_an_action_does_not_assume_retrieval():
    result = _classify("Pro")
    assert result["intent"] == "unknown"
    assert "search evidence or generate a drill" in result["clarification_question"]


def test_coach_is_a_simulation_intent_not_a_user_role():
    result = _classify("Coach me through a Pro summary speech.")
    assert result["intent"] == "coach_simulation"
    assert result["speech_position"] == "summary"
    assert result["side"] == "pro"
    assert result["clarification_needed"] is False


def test_bare_docx_path_is_confirmed_before_ingestion():
    result = _classify("background/research.docx", role="coach")
    assert result["intent"] == "unknown"
    assert result["file_path"] == "background/research.docx"
    assert result["clarification_question"] == (
        "Do you want to preview this DOCX file for evidence ingestion?"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Fabricate and add evidence for our case.",
        "Upload this invented evidence card.",
        "Write and import evidence for my speech.",
    ],
)
def test_integrity_refusal_takes_precedence_over_ingestion(message):
    result = _classify(message, role="coach")
    assert result["intent"] == "integrity_refusal"
    assert result["clarification_needed"] is False


def test_known_and_explicitly_ambiguous_requests_skip_classifier_model(
    isolated_settings,
):
    class FailIfCalled:
        available = True

        def complete_json(self, **kwargs):
            raise AssertionError("classifier model should not be called")

    agent = CaseFileAgent(isolated_settings)
    agent.nodes.llm = FailIfCalled()

    ingestion = agent.ask(
        "How can I add evidence?",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert ingestion["intent"] == "ingest_cards"
    assert ingestion["tool_trace"] == []
    assert "attach or provide the DOCX" in ingestion["response"]

    ambiguous = agent.ask(
        "I need evidence.",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert ambiguous["intent"] == "unknown"
    assert ambiguous["tool_trace"] == []
    assert "search existing evidence or import" in ambiguous["response"]


def test_model_resolved_read_intent_still_requires_missing_fields(
    isolated_settings,
):
    class ReadClassifier:
        available = True

        def complete_json(self, **kwargs):
            return {
                "intent": "retrieve_evidence",
                "side": "unknown",
                "student_id": None,
                "speech_position": None,
                "file_path": None,
                "confirmation_token": None,
                "start": None,
                "clarification_needed": False,
                "clarification_question": None,
            }

    agent = CaseFileAgent(isolated_settings)
    agent.nodes.llm = ReadClassifier()
    result = agent.ask(
        "Can you help me locate something useful?",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert result["intent"] == "retrieve_evidence"
    assert result["tool_trace"] == []
    assert "side (Pro or Con)" in result["response"]
