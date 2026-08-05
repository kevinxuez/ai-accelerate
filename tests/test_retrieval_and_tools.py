from __future__ import annotations

import json

from casefile.agent.graph import CaseFileAgent
from casefile.agent.tools import CaseFileTools, ToolContext
from casefile.ingest.pipeline import IngestionPipeline
from casefile.retrieval import CaseFileIndex


def _ingest(sample_docx, settings):
    pipeline = IngestionPipeline(settings)
    preview = pipeline.preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
        use_model=False,
    )
    pipeline.confirm(preview.token)


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
    assert index.search_cards(
        "consumer protection", resolution="WRONG-RESOLUTION", side="pro"
    ) == []
    assert index.search_cards(
        "consumer protection", resolution="2026-09-CRYPTO", side="con"
    ) == []


def test_role_checks_live_inside_tools(isolated_settings):
    tools = CaseFileTools(isolated_settings)
    student = ToolContext("student", "alice", "R1")
    coach = ToolContext("coach", "coach-1", "R1")
    assert tools.get_progress(student, "bob") == (
        "[DENIED] role 'student' cannot read progress records for another student."
    )
    assert tools.log_assessment(
        student,
        student_id="alice",
        speech_position="summary",
        resolution="R1",
        weakness_tags=["collapse"],
        assessment_text="Needs work.",
    ) == "[DENIED] role 'student' cannot log assessment records."
    assert tools.ingest_cards(
        student,
        file_path="anything.docx",
        dry_run=False,
    ) == "[INVALID] a preview confirmation_token is required before writing."
    assert tools.ingest_cards(
        coach,
        file_path="anything.docx",
        dry_run=False,
    ) == "[INVALID] a preview confirmation_token is required before writing."
    written = tools.log_assessment(
        coach,
        student_id="alice",
        speech_position="summary",
        resolution="R1",
        weakness_tags=["collapse"],
        assessment_text="Needs work.",
    )
    assert written["author_role"] == "coach"
    assert tools.get_progress(student, "alice")[0]["weakness_tags"] == ["collapse"]


def test_agent_clarifies_refuses_and_denies(isolated_settings):
    agent = CaseFileAgent(isolated_settings)
    bare = agent.ask(
        "give me a drill", role="student", user_id="alice", resolution="R1"
    )
    assert bare["intent"] == "generate_drill"
    assert bare["tool_trace"] == []
    assert "side (Pro or Con)" in bare["response"]

    con_drill = agent.ask(
        "give me a drill, con",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert con_drill["intent"] == "generate_drill"
    assert con_drill["tool_trace"][0]["tool"] == "generate_drill"
    assert "general claim-evidence-warrant drill" in con_drill["response"]

    refusal = agent.ask(
        "write my final focus speech",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert refusal["intent"] == "integrity_refusal"
    assert "cannot fabricate citations" in refusal["response"]

    denied = agent.ask(
        "show progress for bob",
        role="student",
        user_id="alice",
        resolution="R1",
    )
    assert denied["response"].startswith("[DENIED]")


def test_audit_log_captures_role_args_and_chunk_ids(sample_docx, isolated_settings):
    _ingest(sample_docx, isolated_settings)
    tools = CaseFileTools(isolated_settings)
    result = tools.search_cards(
        ToolContext("student", "alice", "2026-09-CRYPTO"),
        "consumer accounts government protection",
        "pro",
    )
    assert result
    entries = [
        json.loads(line)
        for line in isolated_settings.audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert entries[-1]["caller_role"] == "student"
    assert entries[-1]["arguments"]["resolution"] == "2026-09-CRYPTO"
    assert entries[-1]["retrieved_chunk_ids"]
