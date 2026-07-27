from __future__ import annotations

import json

import pytest

from casefile.ingest.baseline_heuristic import heuristic_boundary_pass
from casefile.ingest.pipeline import IngestionPipeline
from casefile.ingest.score_boundaries import score
from casefile.ingest.serialize_index import detect_convention, paragraph_records


def test_serializer_preserves_actual_sample_shape(sample_docx):
    records = paragraph_records(sample_docx)
    assert len(records) == 26
    assert [record.i for record in records[:5]] == [1, 2, 3, 4, 5]
    assert records[3].text.startswith("Throughout the opinion piece")
    convention, votes = detect_convention(records)
    assert convention == "bold"
    assert votes == {"bold": 5, "underline": 0, "YELLOW": 3}


def test_offline_boundaries_match_supplied_document(sample_docx):
    records = paragraph_records(sample_docx)
    prediction = heuristic_boundary_pass(records)
    prediction["source_file"] = sample_docx.name
    truth = json.loads(
        (sample_docx.parents[1] / "casefile/ingest/ground_truth_sample.json").read_text()
    )
    result = score(prediction, truth)
    assert result["exact"] == result["ground_truth_cards"] == 8
    massad = prediction["cards"][5]
    assert massad["cite"] == [25, 26]
    assert massad["body"] == [26]


def test_scorer_rejects_the_legacy_benchmark_for_the_copy(sample_docx):
    prediction = heuristic_boundary_pass(paragraph_records(sample_docx))
    prediction["source_file"] = sample_docx.name
    legacy = json.loads(
        (sample_docx.parent / "ground_truth_crypto.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="source mismatch"):
        score(prediction, legacy)


def test_pipeline_preserves_original_body_and_spans(
    sample_docx, isolated_settings
):
    records = {record.i: record for record in paragraph_records(sample_docx)}
    preview = IngestionPipeline(isolated_settings).preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
        use_model=False,
        stage=False,
    )
    assert preview.validation["valid"] is True
    assert preview.boundary_method == "heuristic"
    assert len(preview.cards) == 8
    assert preview.cards[0]["body"] == records[4].text
    assert preview.cards[0]["source_paragraphs"] == [2, 3, 4]
    assert preview.cards[1]["read_spans"]
    assert preview.cards[1]["emphasis_spans"]
    assert "paraphrase_no_source" in preview.cards[0]["flags"]


def test_confirmation_writes_once_and_rebuilds_search_index(
    sample_docx, isolated_settings
):
    pipeline = IngestionPipeline(isolated_settings)
    preview = pipeline.preview(
        sample_docx,
        resolution="2026-09-CRYPTO",
        default_side="pro",
        use_model=False,
    )
    result = pipeline.confirm(preview.token)
    assert result == {
        "written": 8,
        "searchable": 7,
        "total_records": 8,
        "cards_path": str(isolated_settings.cards_path),
    }
    stored = json.loads(isolated_settings.cards_path.read_text(encoding="utf-8"))
    assert len(stored) == 8
    with pytest.raises(FileNotFoundError):
        pipeline.confirm(preview.token)

