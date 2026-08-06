from __future__ import annotations

import json

import pytest

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.ingest.pipeline import IngestionPipeline
from casefile.retrieval import CaseFileIndex
from casefile.tools import CaseFileTools, ToolContext


def _ingest(sample_docx, settings):
    pipeline = IngestionPipeline(settings)
    preview = pipeline.preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
    )
    pipeline.confirm(preview.confirmation_token)


def test_filtered_retrieval_and_explicit_empty_branch(sample_docx, isolated_settings):
    _ingest(sample_docx, isolated_settings)
    index = CaseFileIndex(isolated_settings)
    cards = index.search_cards(
        "consumer accounts government protection",
        resolution="2026-09-CRYPTO",
        side="pro",
        n=3,
    )
    assert cards
    assert cards[0]["cite_full"].startswith("Federal Trade Commission")
    assert cards[0]["returned_document"].startswith(cards[0]["cite_full"])
    assert cards[0]["content_trust"] == "untrusted_document"
    assert cards[0]["retrieval_trust"] == "untrusted_retrieval"
    assert cards[0]["injection_risk"] == "low"
    assert (
        index.search_cards(
            "consumer protection", resolution="WRONG-RESOLUTION", side="pro"
        )
        == []
    )
    assert (
        index.search_cards(
            "consumer protection", resolution="2026-09-CRYPTO", side="con"
        )
        == []
    )


def test_role_checks_live_inside_tools(isolated_settings):
    tools = CaseFileTools(isolated_settings)
    student = ToolContext("student", "alice", "R1")
    coach = ToolContext("coach", "coach-1", "R1")
    with pytest.raises(CaseFileError) as progress_error:
        tools.progress.get_progress(student, "bob")
    assert progress_error.value.code == ErrorCode.AUTHORIZATION_DENIED
    with pytest.raises(CaseFileError) as assessment_error:
        tools.progress.log_assessment(
            student,
            student_id="alice",
            speech_position="summary",
            resolution="R1",
            weakness_tags=["collapse"],
            assessment_text="Needs work.",
        )
    assert assessment_error.value.code == ErrorCode.AUTHORIZATION_DENIED
    for context in (student, coach):
        with pytest.raises(CaseFileError) as confirmation_error:
            tools.ingestion_tools.commit_ingestion(
                context,
                confirmation_token="0" * 32,
            )
        assert confirmation_error.value.code == ErrorCode.CONFIRMATION_INVALID
    written = tools.progress.log_assessment(
        coach,
        student_id="alice",
        speech_position="summary",
        resolution="R1",
        weakness_tags=["collapse"],
        assessment_text="Needs work.",
    )
    assert written["author_role"] == "coach"
    assert tools.progress.get_progress(student, "alice")[0]["weakness_tags"] == [
        "collapse"
    ]


def test_audit_log_captures_role_args_and_chunk_ids(sample_docx, isolated_settings):
    _ingest(sample_docx, isolated_settings)
    tools = CaseFileTools(isolated_settings)
    result = tools.evidence.search_cards(
        ToolContext("student", "alice", "2026-09-CRYPTO"),
        "consumer accounts government protection",
        "pro",
    )
    assert result
    entries = [
        json.loads(line)
        for line in isolated_settings.audit_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert entries[-1]["caller_role"] == "student"
    assert entries[-1]["arguments"]["resolution"] == "2026-09-CRYPTO"
    assert entries[-1]["retrieved_chunk_ids"]
