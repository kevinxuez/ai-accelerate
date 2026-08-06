"""Evidence Librarian-managed, confirmation-gated DOCX ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from casefile.agents.contracts import IngestionCommitResult, IngestionPreview
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.agents.prompt_registry import load_prompt
from casefile.config import Settings, get_settings
from casefile.llm import AnthropicJSONClient, build_anthropic_client
from casefile.security.audit import SecurityAuditor

from .boundary_pass import run_boundary_pass
from .commit import (
    approve_quarantined_card,
    commit_ingestion,
    stage_ingestion,
)
from .extract import ExtractedDocument, extract_document
from .field_pass import label_card, parse_citation
from .contracts import CardBoundary, CardRecord, ParagraphRecord
from .preview import (
    STAGED_INGESTION_SCHEMA_VERSION,
    StagedIngestion,
    preview_summary,
)
from .slice_spans import (
    card_convention,
    join_paragraphs,
    span_text,
    spans_for_paragraphs,
)
from .validate import (
    CORRUPTION_FLAGS,
    boundary_flags,
    ingest_status,
    is_indexable,
    preview_card,
    source_key,
)


def _card_id(header: str, body: str) -> str:
    normalized = re.sub(r"\s+", " ", body).strip()
    return hashlib.sha256(f"{header.strip()}\n{normalized}".encode("utf-8")).hexdigest()


def _extracted_card(
    boundary: CardBoundary,
    *,
    by_id: dict[int, ParagraphRecord],
    document: ExtractedDocument,
) -> dict[str, Any]:
    header = by_id[boundary.header].text if boundary.header in by_id else ""
    tag = join_paragraphs(boundary.tag, by_id)
    citation = join_paragraphs(boundary.cite, by_id)
    marking = card_convention(
        boundary.body,
        by_id,
        document.marking_convention,
    )
    body, read_spans, emphasis_spans = spans_for_paragraphs(
        boundary.body,
        by_id,
        marking,
    )
    flags = boundary_flags(boundary, body, citation, read_spans)
    source_paragraphs = sorted(
        set(
            ([] if boundary.header is None else [boundary.header])
            + boundary.tag
            + boundary.cite
            + boundary.body
        )
    )
    return {
        "header": header,
        "tag": tag,
        "citation": citation,
        "body": body,
        "read_spans": read_spans,
        "emphasis_spans": emphasis_spans,
        "marking_convention": marking,
        "flags": flags,
        "source_paragraphs": source_paragraphs,
        "citation_fields": parse_citation(citation, header),
    }


class IngestionPipeline:
    """Orchestrate required model judgments around deterministic integrity steps."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: AnthropicJSONClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self.llm = llm or build_anthropic_client(self.settings)
        self.security_audit = SecurityAuditor(self.settings.security_audit_path)

    def inspect(self, docx: str | Path) -> ExtractedDocument:
        """Inspect, extract, and screen a DOCX without making a model call."""

        return extract_document(
            docx,
            max_bytes=self.settings.max_upload_bytes,
            audit=self.security_audit,
        )

    def preview(
        self,
        docx: str | Path,
        *,
        resolution: str,
        default_side: str,
        stage: bool = True,
    ) -> IngestionPreview:
        document = self.inspect(docx)
        resolved = resolution.strip()
        if not resolved or default_side not in {"pro", "con"}:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "A resolution and Pro or Con side are required before ingestion.",
                stage="ingestion.validate_metadata",
                agent="evidence_librarian",
            )
        boundary_result, _, _ = run_boundary_pass(
            document.records,
            client=self.llm,
        )
        by_id = {record.i: record for record in document.records}
        extracted = [
            _extracted_card(
                CardBoundary.from_dict(raw),
                by_id=by_id,
                document=document,
            )
            for raw in boundary_result["cards"]
        ]

        seen_sources: dict[str, str] = {}
        for card in extracted:
            candidate = {**card["citation_fields"], "body": card["body"]}
            key = source_key(candidate)
            identity = _card_id(card["header"], card["body"])
            if key and key in seen_sources and seen_sources[key] != identity:
                card["flags"].add("duplicate_source")
            elif key:
                seen_sources[key] = identity

        cards: list[dict[str, Any]] = []
        for position, card in enumerate(extracted, start=1):
            labels = label_card(
                header=card["header"],
                tag=card["tag"],
                citation=card["citation"],
                body=card["body"],
                source_filename=document.source_filename,
                default_side=default_side,
                validation_flags=sorted(card["flags"]),
                client=self.llm,
            )
            flags = set(card["flags"]) | set(labels.flags)
            if labels.evidence_type == "paraphrased" and not labels.source_text_present:
                flags.add("paraphrase_no_source")
            if flags & CORRUPTION_FLAGS:
                flags.add("do_not_ingest")
            if flags and not labels.explanation:
                raise CaseFileError(
                    ErrorCode.INGESTION_CARD_INVALID,
                    "The Evidence Librarian did not explain a flagged or excluded card.",
                    stage="ingestion.validate_cards",
                    agent="evidence_librarian",
                    safe_details={"card_position": position},
                )

            read_text = (
                span_text(card["body"], card["read_spans"])
                if card["read_spans"]
                else card["body"]
            )
            emphasis_text = span_text(card["body"], card["emphasis_spans"])
            embedding_text = "\n".join(
                part
                for part in (
                    card["header"],
                    card["tag"],
                    read_text,
                    emphasis_text,
                )
                if part
            )
            returned_document = "\n".join(
                part
                for part in (
                    card["citation"],
                    card["header"],
                    card["tag"],
                    card["body"],
                )
                if part
            )
            record = CardRecord(
                id=_card_id(card["header"], card["body"]),
                header=card["header"],
                tag=card["tag"],
                cite_full=card["citation"],
                body=card["body"],
                read_spans=card["read_spans"],
                emphasis_spans=card["emphasis_spans"],
                marking_convention=card["marking_convention"],
                evidence_type=labels.evidence_type,
                source_text_present=labels.source_text_present,
                resolution=resolved,
                resolution_confidence="high",
                side=labels.side,
                topic_tags=labels.topic_tags,
                ingest_status=ingest_status(card["body"], flags),
                flags=sorted(flags),
                source_file=document.source_filename,
                source_paragraphs=card["source_paragraphs"],
                embedding_text=embedding_text,
                returned_document=returned_document,
                explanation=labels.explanation,
                injection_risk=document.screening.risk,
                injection_signals=document.screening.signals,
                model_processing_skipped=False,
                **card["citation_fields"],
            ).to_dict()
            preview_card(record, position=position)
            cards.append(record)

        boundary_prompt = load_prompt("evidence_librarian_boundaries")
        labeling_prompt = load_prompt("evidence_librarian_labels")
        warnings = [
            f"Card {position} is excluded or flagged: {card['explanation']}"[:500]
            for position, card in enumerate(cards, start=1)
            if not is_indexable(card) or card.get("flags")
        ]
        staged = StagedIngestion(
            schema_version=STAGED_INGESTION_SCHEMA_VERSION,
            job_id=uuid.uuid4().hex,
            confirmation_token=uuid.uuid4().hex,
            source_path=str(document.source_path),
            source_filename=document.source_filename,
            source_sha256=document.source_sha256,
            resolution=resolved,
            side=default_side,
            marking_convention=document.marking_convention,
            marking_votes=document.marking_votes,
            cards=cards,
            warnings=warnings,
            model=str(getattr(self.llm, "model", type(self.llm).__name__)),
            boundary_prompt=boundary_prompt.template_name,
            labeling_prompt=labeling_prompt.template_name,
        )
        artifact = staged.artifact()
        if stage:
            stage_ingestion(self.settings, staged)
        return artifact

    def confirm(self, token: str) -> IngestionCommitResult:
        return commit_ingestion(self.settings, token)

    def approve_quarantined_card(self, card_id: str) -> dict[str, Any]:
        return approve_quarantined_card(
            self.settings,
            self.security_audit,
            card_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", nargs="?")
    parser.add_argument("--resolution")
    parser.add_argument("--side", choices=["pro", "con"])
    parser.add_argument("--confirm", metavar="TOKEN")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    pipeline = IngestionPipeline()
    if args.confirm:
        result = pipeline.confirm(args.confirm).model_dump(mode="json")
        print(json.dumps(result, indent=2) if args.as_json else result)
        return
    if not args.docx or not args.resolution or not args.side:
        parser.error("docx, --resolution, and --side are required for a preview")
    preview = pipeline.preview(
        args.docx,
        resolution=args.resolution,
        default_side=args.side,
    )
    value = preview.model_dump(mode="json")
    print(
        json.dumps(value, ensure_ascii=False, indent=2)
        if args.as_json
        else preview_summary(preview)
    )


if __name__ == "__main__":
    main()


__all__ = ["IngestionPipeline", "is_indexable"]
