"""Evidence Librarian boundary segmentation with deterministic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from casefile.agents.contracts import BoundaryOutput
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.agents.prompt_registry import load_prompt
from casefile.config import get_settings
from casefile.llm import AnthropicJSONClient, build_anthropic_client
from .contracts import ParagraphRecord
from .ooxml import render_index
from .validate import require_valid_boundaries, validate_boundaries


def model_boundary_pass(
    records: list[ParagraphRecord], client: AnthropicJSONClient
) -> dict[str, Any]:
    prompt = load_prompt("evidence_librarian_boundaries")
    value = client.complete_json(
        system=prompt.content,
        user=json.dumps(
            {"paragraph_index": render_index(records)},
            ensure_ascii=False,
        ),
        max_tokens=8_000,
        schema=BoundaryOutput,
        agent="evidence_librarian",
        prompt_template=prompt.template_name,
        prompt_version=prompt.version,
    )
    try:
        parsed = BoundaryOutput.model_validate(value)
    except ValidationError as exc:
        raise CaseFileError(
            ErrorCode.MODEL_OUTPUT_INVALID,
            "The Evidence Librarian returned card boundaries that did not match BoundaryOutput.",
            stage="ingestion.segment_cards",
            agent="evidence_librarian",
            safe_details={"schema": "BoundaryOutput"},
            cause=exc,
        ) from exc
    try:
        parsed.validate_ids({record.i for record in records})
    except ValueError as exc:
        raise CaseFileError(
            ErrorCode.INGESTION_BOUNDARY_INVALID,
            "The Evidence Librarian returned unknown paragraph IDs.",
            stage="ingestion.validate_boundaries",
            agent="evidence_librarian",
            cause=exc,
        ) from exc
    return parsed.model_dump(mode="json")


def run_boundary_pass(
    records: list[ParagraphRecord],
    *,
    client: AnthropicJSONClient,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Run exactly one segmentation judgment; never retry or substitute."""

    result = model_boundary_pass(records, client)
    validation = validate_boundaries(result, records)
    require_valid_boundaries(validation)
    return result, validation, "model"


def main() -> None:
    import argparse

    from casefile.security.audit import SecurityAuditor

    from .extract import extract_document

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx")
    args = parser.parse_args()
    settings = get_settings()
    extracted = extract_document(
        Path(args.docx),
        max_bytes=settings.max_upload_bytes,
        audit=SecurityAuditor(settings.security_audit_path),
    )
    result, validation, method = run_boundary_pass(
        extracted.records,
        client=build_anthropic_client(settings),
    )
    print(
        json.dumps(
            {
                "source_filename": extracted.source_filename,
                "result": result,
                "validation": validation,
                "boundary_method": method,
                "marking_convention": extracted.marking_convention,
                "marking_votes": extracted.marking_votes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
