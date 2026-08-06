from __future__ import annotations

import json
from typing import Any

import pytest

from casefile.agents.argument_strategist import (
    ArgumentStrategist,
    validate_argument_draft,
)
from casefile.agents.contracts import (
    ArgumentDraft,
    ArgumentRequest,
    EvidenceCard,
    EvidencePacket,
    EvidenceProvenance,
)
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.tools import ToolContext


CARD_ID = "a" * 64
UNKNOWN_CARD_ID = "b" * 64
RESOLUTION = "Resolved: Example"
ORIGINAL_REQUEST = "Build a Pro contention about poverty using economics.docx."


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


def _context() -> ToolContext:
    return ToolContext(
        role="student",
        user_id="student-1",
        resolution=RESOLUTION,
        request_id="request-1",
        agent="argument_strategist",
    )


def _packet(*, empty: bool = False) -> EvidencePacket:
    cards = []
    if not empty:
        cards.append(
            EvidenceCard(
                card_id=CARD_ID,
                source_filename="economics.docx",
                citation="Research Institute, 2026, Household Costs.",
                header="Research Institute '26",
                tag="Higher costs burden low-income households.",
                body="Essential costs consume a larger share of low-income budgets.",
                read_spans=[],
                emphasis_spans=[],
                resolution=RESOLUTION,
                side="pro",
                retrieval_score=0.84,
            )
        )
    return EvidencePacket(
        request_summary="Find poverty evidence.",
        resolution=RESOLUTION,
        side="pro",
        confirmed_source_files_considered=["economics.docx"],
        queries_executed=["poverty household cost burden"],
        cards=cards,
        empty_result=empty,
        provenance=EvidenceProvenance(
            ledger_schema_version=1,
            retrieval_backend="chroma",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            confirmed_only=True,
        ),
    )


def _request(**changes: Any) -> dict[str, Any]:
    value = {
        "original_request": ORIGINAL_REQUEST,
        "resolution": RESOLUTION,
        "side": "pro",
        "subject": "poverty and household cost burdens",
        "entities": ["low-income households"],
        "requested_sections": [],
        "speech_position": None,
        "length_seconds": None,
        "source_files": ["economics.docx"],
        "constraints": ["Use only economics.docx"],
        "preserve_citations": False,
        "revision_instruction": None,
    }
    value.update(changes)
    return value


def _section(
    text: str,
    *,
    support: str = "supported",
    card_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "support": support,
        "card_ids": [CARD_ID] if card_ids is None else card_ids,
    }


def _draft(**changes: Any) -> dict[str, Any]:
    value = {
        "artifact_type": "argument_draft",
        "title": "Household cost poverty contention",
        "resolution": RESOLUTION,
        "side": "pro",
        "format": "structured_argument",
        "claim": _section("The policy reduces poverty pressure."),
        "warrant": _section("Lower essential costs free household resources."),
        "evidence": _section("The supplied card documents disproportionate costs."),
        "impact": _section("Reducing the burden improves material security."),
        "resolution_link": _section("That mechanism supports the Pro side."),
        "likely_response": _section("Opponents may contest the causal magnitude."),
        "source_card_ids": [CARD_ID],
        "unsupported_facts": [],
    }
    value.update(changes)
    return value


def test_open_ended_constraints_and_generation_use_only_evidence_packet() -> None:
    packet = _packet()
    model = ScriptedModel(_request(), _draft())
    request, draft = ArgumentStrategist(model).create_argument(
        _context(),
        original_request=ORIGINAL_REQUEST,
        evidence_packet=packet,
        requested_side="pro",
    )

    assert request.subject == "poverty and household cost burdens"
    assert draft.artifact_type == "argument_draft"
    assert draft.source_card_ids == [CARD_ID]
    assert [call["schema"] for call in model.calls] == [
        ArgumentRequest,
        ArgumentDraft,
    ]
    assert all(
        call["prompt_template"] == "agents/prompts/argument_strategist.md"
        for call in model.calls
    )
    generation_payload = json.loads(model.calls[1]["user"])
    assert generation_payload["evidence_packet"] == packet.model_dump(mode="json")
    assert generation_payload["previous_draft"] is None


def test_unknown_card_id_fails_deterministic_validation() -> None:
    request = ArgumentRequest.model_validate(_request())
    invalid = _draft(
        claim=_section(
            "Invented support.",
            card_ids=[UNKNOWN_CARD_ID],
        ),
        source_card_ids=[CARD_ID, UNKNOWN_CARD_ID],
    )
    model = ScriptedModel(invalid)

    with pytest.raises(CaseFileError) as caught:
        ArgumentStrategist(model).generate_argument(
            _context(),
            request=request,
            evidence_packet=_packet(),
        )

    assert caught.value.code == ErrorCode.ARGUMENT_VALIDATION_FAILED
    assert caught.value.stage == "argument_strategist.validate"
    assert caught.value.safe_details["unknown_card_count"] == 1


def test_empty_evidence_produces_an_explicit_partial_argument() -> None:
    unsupported = lambda text: _section(  # noqa: E731
        text,
        support="unsupported",
        card_ids=[],
    )
    partial = _draft(
        claim=unsupported("A Pro claim can be framed, but no card supports it."),
        warrant=unsupported("The causal mechanism is unsupported."),
        evidence=unsupported("No confirmed evidence was retrieved."),
        impact=unsupported("The impact magnitude is unsupported."),
        resolution_link=unsupported("The resolution link is unsupported."),
        likely_response=unsupported("No response evidence was retrieved."),
        source_card_ids=[],
        unsupported_facts=[
            "The claimed mechanism and impact require confirmed evidence."
        ],
    )
    request = ArgumentRequest.model_validate(_request())
    draft = ArgumentStrategist(ScriptedModel(partial)).generate_argument(
        _context(),
        request=request,
        evidence_packet=_packet(empty=True),
    )

    assert draft.source_card_ids == []
    assert all(
        getattr(draft, field).support == "unsupported"
        for field in (
            "claim",
            "warrant",
            "evidence",
            "impact",
            "resolution_link",
            "likely_response",
        )
    )
    assert draft.unsupported_facts


def test_revision_preserves_session_constraints_citations_and_other_sections() -> None:
    packet = _packet()
    previous_request = ArgumentRequest.model_validate(_request())
    previous_draft = ArgumentDraft.model_validate(_draft())
    instruction = "Make the impact shorter for final focus and keep the citations."
    revised_request = _request(
        requested_sections=["impact"],
        speech_position="final focus",
        constraints=["Use only economics.docx", "Shorten the impact"],
        preserve_citations=True,
        revision_instruction=instruction,
    )
    revised_draft = _draft(
        impact=_section("Lower costs improve material security."),
    )
    model = ScriptedModel(revised_request, revised_draft)

    request, draft = ArgumentStrategist(model).revise_argument(
        _context(),
        instruction=instruction,
        evidence_packet=packet,
        previous_request=previous_request,
        previous_draft=previous_draft,
    )

    assert request.requested_sections == ["impact"]
    assert request.preserve_citations is True
    assert draft.impact.text == "Lower costs improve material security."
    assert draft.warrant == previous_draft.warrant
    generation_payload = json.loads(model.calls[1]["user"])
    assert generation_payload["previous_draft"] == previous_draft.model_dump(
        mode="json"
    )


def test_revision_cannot_change_an_unrequested_section() -> None:
    previous = ArgumentDraft.model_validate(_draft())
    request = ArgumentRequest.model_validate(
        _request(
            requested_sections=["impact"],
            preserve_citations=True,
            revision_instruction="Shorten only the impact.",
        )
    )
    changed = ArgumentDraft.model_validate(
        _draft(
            warrant=_section("A different warrant."),
            impact=_section("Short impact."),
        )
    )

    with pytest.raises(CaseFileError) as caught:
        validate_argument_draft(
            changed,
            request=request,
            evidence_packet=_packet(),
            request_id="request-1",
            previous_draft=previous,
        )

    assert caught.value.code == ErrorCode.ARGUMENT_VALIDATION_FAILED
    assert caught.value.safe_details["changed_sections"] == ["warrant"]


def test_model_failure_is_not_replaced_with_an_argument_outline() -> None:
    failure = CaseFileError(
        ErrorCode.MODEL_UPSTREAM_ERROR,
        "The configured model could not be reached.",
        stage="model.request",
        agent="argument_strategist",
    )
    model = ScriptedModel(failure)

    with pytest.raises(CaseFileError) as caught:
        ArgumentStrategist(model).generate_argument(
            _context(),
            request=ArgumentRequest.model_validate(_request()),
            evidence_packet=_packet(),
        )

    assert caught.value is failure
    assert caught.value.request_id == "request-1"
    assert len(model.calls) == 1
