from __future__ import annotations

import json

import pytest

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.retrieval import CaseFileIndex
from casefile.tools import CaseFileTools, ToolContext
from casefile.tools.context import ToolRuntime
from casefile.tools.evidence import EvidenceTools


def test_registry_exposes_only_agent_authorized_tools(isolated_settings) -> None:
    tools = CaseFileTools(isolated_settings)

    assert tools.registry.names_for_agent("supervisor") == {"schedule_session"}
    assert tools.registry.names_for_agent("argument_strategist") == set()
    assert tools.registry.names_for_agent("skills_coach") == {
        "get_progress",
        "log_assessment",
    }
    assert tools.registry.names_for_agent("evidence_librarian") == {
        "list_confirmed_files",
        "search_cards",
        "search_rules",
        "get_current_topic",
        "stage_ingestion_preview",
        "commit_ingestion",
    }
    assert all(
        definition.input_model is not None and definition.output_type is not None
        for agent in (
            "supervisor",
            "evidence_librarian",
            "skills_coach",
        )
        for definition in tools.registry.for_agent(agent)
    )
    assert (
        tools.registry.invoke(
            "evidence_librarian",
            "list_confirmed_files",
            ToolContext("student", "student-1", "R1"),
            {"resolution": "R1", "side": "pro"},
        )
        == []
    )


def test_confirmed_files_and_source_filters_use_only_indexable_ledger_cards(
    isolated_settings,
) -> None:
    cards = [
        {
            "id": "confirmed-pro",
            "resolution": "R1",
            "side": "pro",
            "source_file": "confirmed.docx",
            "embedding_text": "consumer protection",
            "returned_document": "Citation\nConsumer protection evidence.",
            "body": "Consumer protection evidence.",
            "ingest_status": "ok",
            "flags": [],
        },
        {
            "id": "incomplete-pro",
            "resolution": "R1",
            "side": "pro",
            "source_file": "incomplete.docx",
            "embedding_text": "consumer protection",
            "returned_document": "Citation",
            "body": "",
            "ingest_status": "incomplete",
            "flags": ["no_body"],
        },
        {
            "id": "confirmed-con",
            "resolution": "R1",
            "side": "con",
            "source_file": "con.docx",
            "embedding_text": "privacy",
            "returned_document": "Citation\nPrivacy evidence.",
            "body": "Privacy evidence.",
            "ingest_status": "ok",
            "flags": [],
        },
    ]
    isolated_settings.cards_path.write_text(
        json.dumps(cards) + "\n",
        encoding="utf-8",
    )
    index = CaseFileIndex(isolated_settings)
    index.rebuild_cards()
    evidence = EvidenceTools(ToolRuntime(isolated_settings), index)
    context = ToolContext(
        "student",
        "student-1",
        "R1",
        agent="evidence_librarian",
    )

    assert evidence.list_confirmed_files(context, side="pro") == ["confirmed.docx"]
    assert (
        evidence.search_cards(
            context,
            "consumer protection",
            "pro",
            source_files=["incomplete.docx"],
        )
        == []
    )
    assert evidence.search_cards(
        context,
        "consumer protection",
        "pro",
        source_files=["confirmed.docx"],
    )

    with pytest.raises(CaseFileError) as caught:
        evidence.list_confirmed_files(
            ToolContext(
                "student",
                "student-1",
                "R1",
                agent="skills_coach",
            )
        )
    assert caught.value.code == ErrorCode.AUTHORIZATION_DENIED


def test_disabled_calendar_capability_fails_visibly(isolated_settings) -> None:
    from dataclasses import replace

    tools = CaseFileTools(replace(isolated_settings, calendar_provider="disabled"))
    context = ToolContext(
        "student",
        "student-1",
        "R1",
        agent="supervisor",
    )

    with pytest.raises(CaseFileError) as caught:
        tools.registry.invoke(
            "supervisor",
            "schedule_session",
            context,
            {
                "student_id": "student-1",
                "start": "2026-08-10T16:00:00-07:00",
                "duration_minutes": 45,
                "attendee_email": None,
                "timezone_name": "America/Los_Angeles",
                "confirmation_token": None,
                "idempotency_key": None,
            },
        )

    assert caught.value.code == ErrorCode.CAPABILITY_DISABLED
