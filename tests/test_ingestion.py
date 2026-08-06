from __future__ import annotations

import json

import pytest

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.ingest.pipeline import IngestionPipeline
from casefile.ingest.ooxml import detect_convention, paragraph_records


def test_serializer_preserves_actual_sample_shape(sample_docx):
    records = paragraph_records(sample_docx)
    assert len(records) == 26
    assert [record.i for record in records[:5]] == [1, 2, 3, 4, 5]
    assert records[3].text.startswith("Throughout the opinion piece")
    convention, votes = detect_convention(records)
    assert convention == "bold"
    assert votes == {"bold": 5, "underline": 0, "YELLOW": 3}


def test_pipeline_preserves_original_body_and_spans(sample_docx, isolated_settings):
    records = {record.i: record for record in paragraph_records(sample_docx)}
    preview = IngestionPipeline(isolated_settings).preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
        stage=False,
    )
    assert preview.provenance.boundary_method == "model"
    assert preview.provenance.labeling_method == "model"
    assert len(preview.cards) == 8
    assert preview.cards[0].body == records[4].text
    assert preview.cards[1].read_spans
    assert preview.cards[1].emphasis_spans
    assert "paraphrase_no_source" in preview.cards[0].flags


def test_confirmation_writes_once_and_rebuilds_search_index(
    sample_docx, isolated_settings
):
    pipeline = IngestionPipeline(isolated_settings)
    preview = pipeline.preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
    )
    result = pipeline.confirm(preview.confirmation_token)
    assert result.written_cards == 8
    assert result.searchable_cards == 7
    assert result.index_rebuilt is True
    stored = json.loads(isolated_settings.cards_path.read_text(encoding="utf-8"))
    assert len(stored) == 8
    with pytest.raises(CaseFileError) as caught:
        pipeline.confirm(preview.confirmation_token)
    assert caught.value.code == ErrorCode.CONFIRMATION_INVALID


def test_invalid_model_output_fails_without_substitution(
    sample_docx, isolated_settings
):
    class InvalidModel:
        def complete_json(self, **kwargs):
            return {"unexpected": "schema smuggling"}

    with pytest.raises(CaseFileError) as caught:
        IngestionPipeline(isolated_settings, llm=InvalidModel()).preview(
            sample_docx,
            resolution="2026-09-CRYPTO",
            default_side="pro",
            stage=False,
        )
    assert caught.value.code == ErrorCode.MODEL_OUTPUT_INVALID
