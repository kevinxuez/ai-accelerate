from __future__ import annotations

import json
from typing import Any

from casefile.agent.graph import CaseFileAgent
from casefile.security.schemas import CoachTurnOutput, EvidenceArgumentOutput


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "card-1",
            "header": "Morrison '21",
            "cite_full": "Morrison, Vox, 2021.",
            "tag": "Regulators can investigate major exchanges.",
            "body": "Coinbase faces an SEC investigation with broad market effects.",
            "read_spans": [[0, 35]],
            "emphasis_spans": [[18, 35]],
        },
        {
            "id": "card-2",
            "header": "Lai '21",
            "cite_full": "Lai, Consumer Finance Review, 2021.",
            "tag": "Payment safeguards affect consumers.",
            "body": "Payment safeguards can limit losses for ordinary consumers.",
            "read_spans": [[0, 18]],
            "emphasis_spans": [[40, 59]],
        },
    ]


class RecordingLLM:
    available = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        payload = json.loads(kwargs["user"])
        self.calls.append({"schema": kwargs["schema"], "payload": payload})
        if kwargs["schema"] is CoachTurnOutput:
            return {
                "focus": "impact comparison",
                "feedback": "Compare what each cited card establishes.",
                "question": "Which marked warrant should the judge prioritize?",
            }
        if kwargs["schema"] is EvidenceArgumentOutput:
            return {
                "claim": "Consumer safeguards support the Pro position.",
                "warrant": "Morrison and Lai identify regulatory and payment effects.",
                "impact": "These effects give the judge a concrete risk comparison.",
                "citations_used": ["Morrison '21", "Lai '21"],
            }
        raise AssertionError(f"Unexpected schema: {kwargs['schema']}")


def test_coach_gets_bounded_marked_excerpts_and_returns_both_cards(
    monkeypatch,
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    recording_llm = RecordingLLM()
    agent.nodes.llm = recording_llm
    cards = _cards()
    monkeypatch.setattr(agent.tools, "search_cards", lambda *args, **kwargs: cards)

    result = agent.ask(
        "Coach me through a Pro summary speech.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id="coach-grounding-0001",
    )

    assert result["coach_turn"]["focus"] == "impact comparison"
    assert len(result["grounding_cards"]) == 2
    assert result["grounding_cards"][0]["body"] == cards[0]["body"]
    assert result["grounding_cards"][1]["emphasis_spans"] == [[40, 59]]
    assert "Evidence 1: Morrison, Vox, 2021." in result["response"]
    assert "Evidence 2: Lai, Consumer Finance Review, 2021." in result["response"]

    coach_payload = next(
        call["payload"]
        for call in recording_llm.calls
        if call["schema"] is CoachTurnOutput
    )
    assert len(coach_payload["grounded_cards"]) == 2
    assert coach_payload["grounded_cards"][0]["read_excerpt"] == (
        "Coinbase faces an SEC investigation"
    )
    assert coach_payload["grounded_cards"][0]["emphasis_excerpt"] == (
        "SEC investigation"
    )
    assert coach_payload["grounded_cards"][1]["read_excerpt"] == (
        "Payment safeguards"
    )


def test_evidence_search_formats_a_grounded_argument_and_preserves_source_cards(
    monkeypatch,
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    recording_llm = RecordingLLM()
    agent.nodes.llm = recording_llm
    cards = _cards()
    monkeypatch.setattr(agent.tools, "search_cards", lambda *args, **kwargs: cards)

    result = agent.ask(
        "Find Pro evidence about consumer payment safeguards.",
        role="student",
        user_id="student-1",
        resolution="R1",
    )

    assert result["intent"] == "retrieve_evidence"
    assert result["evidence_argument"] == {
        "claim": "Consumer safeguards support the Pro position.",
        "warrant": "Morrison and Lai identify regulatory and payment effects.",
        "impact": "These effects give the judge a concrete risk comparison.",
        "citations_used": ["Morrison '21", "Lai '21"],
        "generated_by": "model",
    }
    assert [card["header"] for card in result["grounding_cards"]] == [
        "Morrison '21",
        "Lai '21",
    ]
    assert cards[0]["body"] in result["response"]
    assert cards[1]["body"] in result["response"]

    argument_payload = next(
        call["payload"]
        for call in recording_llm.calls
        if call["schema"] is EvidenceArgumentOutput
    )
    assert len(argument_payload["grounded_cards"]) == 2
    assert argument_payload["grounded_cards"][1]["emphasis_excerpt"] == (
        "ordinary consumers."
    )
