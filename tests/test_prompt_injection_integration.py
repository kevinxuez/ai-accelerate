from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.ingest.pipeline import IngestionPipeline
from casefile.retrieval import CaseFileIndex
from casefile.tools import CaseFileTools, ToolContext


FIXTURES = Path(__file__).parent / "fixtures"


def _injected_docx(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (FIXTURES / "injected_card_document.xml").read_bytes(),
        )
    return path


class NeverCalledClient:
    def complete_json(self, **kwargs):
        raise AssertionError("quarantined document text reached a model")


def test_unsafe_document_is_rejected_before_models_or_staging(
    tmp_path, isolated_settings
):
    path = _injected_docx(tmp_path / "injected_card.docx")
    source_before = path.read_bytes()
    pipeline = IngestionPipeline(isolated_settings, llm=NeverCalledClient())
    with pytest.raises(CaseFileError) as caught:
        pipeline.preview(
            path,
            resolution="R-INJECT",
            default_side="pro",
        )
    assert caught.value.code == ErrorCode.DOCUMENT_UNSAFE
    assert json.loads(isolated_settings.cards_path.read_text(encoding="utf-8")) == []
    assert list(isolated_settings.pending_dir.glob("*.json")) == []
    assert path.read_bytes() == source_before


def test_ingest_paths_are_restricted_and_writes_are_idempotent(
    tmp_path, isolated_settings
):
    outside = _injected_docx(tmp_path / "outside.docx")
    tools = CaseFileTools(isolated_settings)
    coach = ToolContext("coach", "coach-1", "R1")
    with pytest.raises(CaseFileError) as caught:
        tools.ingestion_tools.stage_ingestion_preview(
            coach,
            file_path=str(outside),
            resolution="R1",
            side="pro",
        )
    assert caught.value.code == ErrorCode.AUTHORIZATION_DENIED

    first = tools.progress.log_assessment(
        coach,
        student_id="alice",
        speech_position="summary",
        resolution="R1",
        weakness_tags=["collapse"],
        assessment_text="Practice comparison.",
        idempotency_key="assessment-1",
    )
    second = tools.progress.log_assessment(
        coach,
        student_id="alice",
        speech_position="summary",
        resolution="R1",
        weakness_tags=["collapse"],
        assessment_text="Practice comparison.",
        idempotency_key="assessment-1",
    )
    assert first == second
    progress = json.loads(isolated_settings.progress_path.read_text(encoding="utf-8"))
    assert len(progress) == 1

    calendar_first = tools.calendar.schedule_session(
        coach,
        student_id="alice",
        start="2026-08-01T12:00:00",
        idempotency_key="calendar-1",
    )
    calendar_second = tools.calendar.schedule_session(
        coach,
        student_id="alice",
        start="2026-08-01T12:00:00",
        idempotency_key="calendar-1",
    )
    assert calendar_first == calendar_second
    events = json.loads(
        (isolated_settings.data_dir / "calendar_events.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(events) == 1


def test_injected_rule_chunk_is_not_searchable(tmp_path, isolated_settings):
    source = tmp_path / "injected_rules.md"
    shutil.copy(FIXTURES / "injected_rules.md", source)
    settings = replace(isolated_settings, rules_dir=tmp_path)
    index = CaseFileIndex(settings)
    assert index.rebuild_rules() == 0
    chunks = json.loads(
        (settings.data_dir / "rules_chunks.json").read_text(encoding="utf-8")
    )
    assert chunks[0]["injection_risk"] == "high"
    assert index.search_rules("security") == []


def test_real_calendar_requires_confirmation_and_replay_is_idempotent(
    isolated_settings,
):
    settings = replace(isolated_settings, calendar_provider="google")
    tools = CaseFileTools(settings)
    coach = ToolContext("coach", "coach-1", "R1")
    calls = []
    tools.calendar._google_calendar_event = lambda event: (  # type: ignore[method-assign]
        calls.append(event) or {"id": "google-1", "status": "confirmed", **event}
    )

    preview = tools.calendar.schedule_session(
        coach,
        student_id="alice",
        start="2026-08-01T12:00:00",
    )
    assert preview["confirmation_required"] is True
    assert calls == []

    confirmed = tools.calendar.schedule_session(
        coach,
        student_id="alice",
        start="",
        confirmation_token=preview["confirmation_token"],
        idempotency_key="real-calendar-1",
    )
    replayed = tools.calendar.schedule_session(
        coach,
        student_id="alice",
        start="",
        confirmation_token=preview["confirmation_token"],
        idempotency_key="real-calendar-1",
    )
    assert confirmed == replayed
    assert confirmed["id"] == "google-1"
    assert len(calls) == 1
