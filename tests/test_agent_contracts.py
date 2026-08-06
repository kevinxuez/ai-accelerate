from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from casefile.agents.contracts import (
    MAX_GRAPH_STEPS,
    MAX_MESSAGES,
    ActiveGoal,
    ArgumentDraft,
    ArgumentSection,
    ConversationMessage,
    EvidenceCard,
    EvidencePacket,
    EvidenceProvenance,
    RequestContext,
    StrictContract,
    SupervisorDecision,
    TextSpan,
)
from casefile.agents.errors import (
    HTTP_STATUS_BY_CODE,
    CaseFileError,
    ErrorCode,
)
from casefile.agents.state import CaseFileState
from casefile.api.contracts import ChatSuccessResponse


def _subclasses(model: type[StrictContract]) -> set[type[StrictContract]]:
    direct = set(model.__subclasses__())
    return direct | {nested for child in direct for nested in _subclasses(child)}


def _evidence_card() -> EvidenceCard:
    return EvidenceCard(
        card_id="a" * 64,
        source_filename="confirmed.docx",
        citation="Author, Source, 2026.",
        header="Poverty reduction",
        tag="The policy reduces poverty.",
        body="Preserved source body.",
        read_spans=[TextSpan(start=0, end=9)],
        emphasis_spans=[],
        resolution="Resolved: Example",
        side="pro",
        retrieval_score=0.75,
    )


def _evidence_packet() -> EvidencePacket:
    return EvidencePacket(
        request_summary="Find poverty evidence.",
        resolution="Resolved: Example",
        side="pro",
        confirmed_source_files_considered=["confirmed.docx"],
        queries_executed=["poverty reduction"],
        cards=[_evidence_card()],
        empty_result=False,
        provenance=EvidenceProvenance(
            ledger_schema_version=1,
            retrieval_backend="chroma",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            confirmed_only=True,
        ),
    )


def _unsupported(text: str) -> ArgumentSection:
    return ArgumentSection(text=text, support="unsupported", card_ids=[])


def test_all_four_agent_contracts_are_strict_and_forbid_extra_fields() -> None:
    models = _subclasses(StrictContract)
    assert models
    assert all(model.model_config.get("strict") is True for model in models)
    assert all(model.model_config.get("extra") == "forbid" for model in models)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ActiveGoal(
            summary="Research the topic.",
            completion_criteria=["Return evidence."],
            unexpected=True,
        )


def test_supervisor_decision_enforces_typed_handoff_fields() -> None:
    decision = SupervisorDecision(
        action="delegate",
        target_agent="evidence_librarian",
        goal="Find confirmed evidence.",
        task="Search confirmed Pro cards about poverty.",
        required_artifact="evidence_packet",
        clarification_question=None,
        reason_code="research_required",
    )
    assert decision.target_agent == "evidence_librarian"

    with pytest.raises(ValidationError, match="delegate requires"):
        SupervisorDecision(
            action="delegate",
            target_agent=None,
            goal="Find evidence.",
            task="",
            required_artifact=None,
            clarification_question=None,
            reason_code="invalid_handoff",
        )


def test_evidence_packet_enforces_provenance_scope_and_empty_semantics() -> None:
    packet = _evidence_packet()
    assert packet.provenance.confirmed_only is True
    assert packet.cards[0].body == "Preserved source body."

    with pytest.raises(ValidationError, match="empty_result"):
        EvidencePacket(
            **{
                **packet.model_dump(mode="python"),
                "empty_result": True,
            }
        )

    with pytest.raises(ValidationError, match="span exceeds"):
        EvidenceCard(
            **{
                **packet.cards[0].model_dump(mode="python"),
                "read_spans": [{"start": 0, "end": 999}],
            }
        )


def test_argument_contract_enforces_support_and_exact_citation_union() -> None:
    claim = ArgumentSection(
        text="The policy reduces poverty.",
        support="supported",
        card_ids=["a" * 64],
    )
    draft = ArgumentDraft(
        title="Poverty contention",
        resolution="Resolved: Example",
        side="pro",
        claim=claim,
        warrant=_unsupported("The causal mechanism lacks direct support."),
        evidence=claim,
        impact=_unsupported("The magnitude is not established."),
        resolution_link=_unsupported("The exact resolution link needs support."),
        likely_response=_unsupported("No response evidence was supplied."),
        source_card_ids=["a" * 64],
        unsupported_facts=["The magnitude is not established."],
    )
    assert draft.format == "structured_argument"

    with pytest.raises(ValidationError, match="cannot cite"):
        ArgumentSection(
            text="Unsupported claim.",
            support="unsupported",
            card_ids=["b" * 64],
        )
    with pytest.raises(ValidationError, match="exactly match"):
        ArgumentDraft(
            **{
                **draft.model_dump(mode="python"),
                "source_card_ids": ["b" * 64],
            }
        )


def test_state_and_api_contracts_are_bounded_and_artifact_typed() -> None:
    request = RequestContext(
        request_id="request-1",
        session_id="session-123456789",
        role="student",
        user_id="student-1",
        active_resolution="Resolved: Example",
        attachments=[],
    )
    state = CaseFileState(
        request=request,
        messages=[ConversationMessage(role="user", content="Find evidence.")],
        evidence_packet=_evidence_packet(),
        step_count=1,
    )
    restored = CaseFileState.model_validate_json(state.model_dump_json())
    assert restored.schema_version == 1
    assert restored.evidence_packet is not None

    response = ChatSuccessResponse(
        status="completed",
        response="Found one confirmed card.",
        request_id=request.request_id,
        session_id=request.session_id,
        active_agent="supervisor",
        active_goal=ActiveGoal(
            summary="Find evidence.",
            completion_criteria=["Return an EvidencePacket."],
        ),
        awaiting_input=False,
        awaiting_confirmation=False,
        artifacts=[_evidence_packet()],
    )
    assert response.artifacts[0].artifact_type == "evidence_packet"

    with pytest.raises(ValidationError):
        CaseFileState(
            request=request,
            messages=[
                ConversationMessage(role="user", content="x")
                for _ in range(MAX_MESSAGES + 1)
            ],
        )
    with pytest.raises(ValidationError):
        CaseFileState(request=request, step_count=MAX_GRAPH_STEPS + 1)


def test_error_codes_have_http_mappings_and_never_expose_causes() -> None:
    assert set(HTTP_STATUS_BY_CODE) == set(ErrorCode)
    upstream = RuntimeError("secret upstream response")
    error = CaseFileError(
        ErrorCode.MODEL_OUTPUT_INVALID,
        "The Evidence Librarian output did not match BoundaryOutput.",
        stage="ingestion.segment_cards",
        agent="evidence_librarian",
        request_id="request-1",
        safe_details={"schema": "BoundaryOutput"},
        cause=upstream,
    )
    envelope = error.public_envelope(request_id="request-1")
    assert envelope["error"]["code"] == "MODEL_OUTPUT_INVALID"
    assert envelope["error"]["details"] == {"schema": "BoundaryOutput"}
    assert "cause_type" not in envelope["error"]
    assert "secret upstream response" not in str(envelope)
    assert error.cause_type == "RuntimeError"


def test_tool_modules_contain_no_string_prefixed_errors() -> None:
    tools_dir = Path(__file__).parents[1] / "casefile" / "tools"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in tools_dir.glob("*.py")
    )
    for code in (
        "DENIED",
        "INVALID",
        "UNAVAILABLE",
        "MODEL_FAILED",
        "TASK_FAILED",
    ):
        prefix = f"[{code}]"
        assert prefix not in source
