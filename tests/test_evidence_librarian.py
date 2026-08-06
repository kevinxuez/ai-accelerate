from __future__ import annotations

from typing import Any

import pytest

from casefile.agents.contracts import EvidenceQueryPlan
from casefile.agents.evidence_librarian import EvidenceLibrarian
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.tools import ToolContext


CARD = {
    "id": "a" * 64,
    "source_file": "economics.docx",
    "cite_full": "Research Institute, 2026, Household Costs.",
    "header": "Research Institute '26",
    "tag": "Higher household costs burden low-income families.",
    "body": "Higher essential costs consume a larger share of low-income household budgets.",
    "read_spans": [[0, 22]],
    "emphasis_spans": [[52, 62]],
    "resolution": "Resolved: Example",
    "side": "pro",
    "score": 0.84,
}


class StubIndex:
    backend = "chroma"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"


class StubRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke(self, agent, name, context, arguments):
        self.calls.append((name, arguments))
        if name == "list_confirmed_files":
            return ["economics.docx", "general.docx"]
        if name == "search_cards":
            return [CARD]
        if name == "search_rules":
            return [
                {
                    "id": "rule-1",
                    "_chunk_id": "rule-1",
                    "section_number": "7.2",
                    "section_title": "Evidence Integrity",
                    "text": "Quoted evidence must remain attached to its citation.",
                    "document": "rules.md",
                    "score": 0.91,
                }
            ]
        if name == "get_current_topic":
            return {
                "topic": {
                    "id": "pf-example",
                    "event": "Public Forum",
                    "resolution": "Resolved: Example topic.",
                    "effective_from": "2026-09-01",
                    "effective_to": "2026-10-31",
                    "source_ref": "fixture://topics/pf-example",
                    "synthetic": True,
                },
                "provider": "Synthetic topic fixture",
                "backend": "fixture",
                "synthetic": True,
            }
        raise AssertionError(name)


class StubTools:
    def __init__(self) -> None:
        self.registry = StubRegistry()
        self.index = StubIndex()


class StubModel:
    def __init__(self, plan: dict[str, Any] | None = None) -> None:
        self.plan = plan
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.plan is None:
            raise CaseFileError(
                ErrorCode.MODEL_UPSTREAM_ERROR,
                "The configured model could not be reached.",
                stage="model.request",
                agent="evidence_librarian",
            )
        return self.plan


def _plan(**changes: Any) -> dict[str, Any]:
    value = {
        "resolution": "Resolved: Example",
        "side": "pro",
        "subject": "household costs for low-income people",
        "entities": ["low-income households"],
        "source_files": ["economics.docx"],
        "queries": ["household cost burden", "low-income essential spending"],
        "result_limit": 4,
        "clarification_needed": False,
        "clarification_question": None,
    }
    value.update(changes)
    return value


def _context() -> ToolContext:
    return ToolContext(
        role="student",
        user_id="student-1",
        resolution="Resolved: Example",
        request_id="request-1",
        agent="evidence_librarian",
    )


def test_open_ended_request_becomes_filtered_deduplicated_evidence_packet() -> None:
    tools = StubTools()
    model = StubModel(_plan())
    packet = EvidenceLibrarian(tools, model).retrieve_evidence(
        _context(),
        request="Use whatever evidence best explains why this affects low-income people.",
        requested_side="pro",
    )

    assert packet.artifact_type == "evidence_packet"
    assert packet.confirmed_source_files_considered == ["economics.docx"]
    assert packet.queries_executed == [
        "household cost burden",
        "low-income essential spending",
    ]
    assert [card.card_id for card in packet.cards] == ["a" * 64]
    assert packet.cards[0].body == CARD["body"]
    assert packet.provenance.confirmed_only is True
    searches = [
        arguments for name, arguments in tools.registry.calls if name == "search_cards"
    ]
    assert len(searches) == 2
    assert all(
        arguments["source_files"] == ["economics.docx"] for arguments in searches
    )
    assert model.calls[0]["schema"] is EvidenceQueryPlan
    assert model.calls[0]["prompt_template"] == "agents/prompts/evidence_librarian.md"


def test_unconfirmed_model_selected_file_is_rejected_before_search() -> None:
    tools = StubTools()
    librarian = EvidenceLibrarian(
        tools,
        StubModel(_plan(source_files=["pending-upload.docx"])),
    )

    with pytest.raises(CaseFileError) as caught:
        librarian.retrieve_evidence(
            _context(),
            request="Use only pending-upload.docx.",
            requested_side="pro",
        )

    assert caught.value.code == ErrorCode.AGENT_OUTPUT_INVALID
    assert caught.value.stage == "evidence_librarian.plan"
    assert [name for name, _ in tools.registry.calls] == ["list_confirmed_files"]


def test_rule_and_topic_tools_return_typed_packets() -> None:
    tools = StubTools()
    librarian = EvidenceLibrarian(tools, StubModel(_plan()))

    rules = librarian.retrieve_rules(_context(), question="Can evidence be altered?")
    topic = librarian.retrieve_topic(_context(), event="Public Forum")

    assert rules.artifact_type == "rule_packet"
    assert rules.chunks[0].chunk_id == "rule-1"
    assert topic.artifact_type == "topic_packet"
    assert topic.backend == "fixture"
    assert topic.synthetic is True


def test_model_failure_is_exposed_without_running_search() -> None:
    tools = StubTools()
    with pytest.raises(CaseFileError) as caught:
        EvidenceLibrarian(tools, StubModel()).retrieve_evidence(
            _context(),
            request="Find related Pro evidence.",
            requested_side="pro",
        )

    assert caught.value.code == ErrorCode.MODEL_UPSTREAM_ERROR
    assert [name for name, _ in tools.registry.calls] == ["list_confirmed_files"]
