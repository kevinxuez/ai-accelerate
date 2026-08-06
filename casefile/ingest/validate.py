"""Deterministic boundary, card, and indexability validation."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from casefile.agents.contracts import IngestionPreviewCard, TextSpan
from casefile.agents.errors import CaseFileError, ErrorCode
from .contracts import CardBoundary, ParagraphRecord


BARE_URL = re.compile(r"^\s*(?:https?://|www\.)\S+[\s./]*$", re.I)
HTML_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);")
CORRUPTION_FLAGS = {"pdf_paste_fragmented", "text_corrupt", "html_entity"}
NON_INDEXABLE_FLAGS = CORRUPTION_FLAGS | {"do_not_ingest", "paraphrase_no_source"}


def validate_boundaries(
    result: dict[str, Any], records: list[ParagraphRecord]
) -> dict[str, Any]:
    ids = {record.i for record in records}
    owners: dict[int, set[tuple[int, str]]] = {}
    invalid_ids: set[int] = set()
    malformed: list[int] = []
    cards = [CardBoundary.from_dict(card) for card in result.get("cards", [])]
    for number, card in enumerate(cards, start=1):
        if not card.cite:
            malformed.append(number)
        fields = {
            "header": [] if card.header is None else [card.header],
            "tag": card.tag,
            "cite": card.cite,
            "body": card.body,
        }
        for field, values in fields.items():
            for value in values:
                if value not in ids:
                    invalid_ids.add(value)
                owners.setdefault(value, set()).add((number, field))
    for value in result.get("junk", []):
        if value not in ids:
            invalid_ids.add(value)
        owners.setdefault(value, set()).add((0, "junk"))

    duplicates: list[int] = []
    for paragraph_id, assignments in owners.items():
        card_numbers = {number for number, _ in assignments}
        same_card_cite_body = (
            len(card_numbers) == 1
            and 0 not in card_numbers
            and {field for _, field in assignments} <= {"cite", "body"}
        )
        if len(assignments) > 1 and not same_card_cite_body:
            duplicates.append(paragraph_id)
    validation: dict[str, Any] = {
        "unassigned": sorted(ids - set(owners)),
        "duplicate_assignment": sorted(duplicates),
        "invalid_paragraph_ids": sorted(invalid_ids),
        "cards_without_citation": malformed,
        "cards_found": len(cards),
    }
    validation["valid"] = bool(cards) and not any(
        validation[key]
        for key in (
            "unassigned",
            "duplicate_assignment",
            "invalid_paragraph_ids",
            "cards_without_citation",
        )
    )
    return validation


def require_valid_boundaries(validation: dict[str, Any]) -> None:
    if validation.get("valid"):
        return
    raise CaseFileError(
        ErrorCode.INGESTION_BOUNDARY_INVALID,
        "The Evidence Librarian returned boundaries that failed deterministic validation.",
        stage="ingestion.validate_boundaries",
        agent="evidence_librarian",
        safe_details={
            key: len(value) if isinstance(value, list) else value
            for key, value in validation.items()
        },
    )


def boundary_flags(
    boundary: CardBoundary,
    body: str,
    citation: str,
    read_spans: list[list[int]],
) -> set[str]:
    flags = set(boundary.flags)
    if boundary.header is None:
        flags.add("no_header")
    if not boundary.body or not body.strip():
        flags.add("no_body")
    if BARE_URL.fullmatch(citation.strip()):
        flags.add("cite_is_bare_url")
    if (
        not re.search(r"https?://|\b(?:19|20)\d{2}\b", citation, re.I)
        and len(citation) < 180
    ):
        flags.add("cite_is_bare_headline")
    if set(boundary.cite) & set(boundary.body):
        flags.add("cite_body_same_paragraph")
    if body and not read_spans:
        flags.add("no_marking")
    if body and read_spans == [[0, len(body)]]:
        flags.add("fully_marked")
    opening = body.lstrip()[:1]
    if opening and (opening.islower() or opening in ".,;:)]}"):
        flags.add("body_truncated")
    if HTML_ENTITY.search(body):
        flags.update({"html_entity", "text_corrupt", "do_not_ingest"})
    if len(boundary.body) >= 3:
        flags.update({"pdf_paste_fragmented", "text_corrupt", "do_not_ingest"})
    if flags & CORRUPTION_FLAGS:
        flags.add("do_not_ingest")
    return flags


def ingest_status(body: str, flags: set[str]) -> str:
    if not body.strip() or "no_body" in flags:
        return "incomplete"
    return "flagged" if flags else "ok"


def source_key(card: dict[str, Any]) -> str:
    if card.get("url"):
        return re.sub(r"[/?#]+$", "", str(card["url"]).lower())
    return re.sub(r"\s+", " ", str(card.get("body", "")).lower()).strip()


def is_indexable(card: dict[str, Any]) -> bool:
    return (
        card.get("ingest_status") != "incomplete"
        and bool(card.get("body"))
        and not (set(card.get("flags", [])) & NON_INDEXABLE_FLAGS)
        and not (
            card.get("injection_risk") == "high"
            and not bool(card.get("injection_approved"))
        )
    )


def preview_card(card: dict[str, Any], *, position: int) -> IngestionPreviewCard:
    try:
        return IngestionPreviewCard(
            card_id=str(card.get("id") or ""),
            header=str(card.get("header") or ""),
            tag=str(card.get("tag") or ""),
            citation=str(card.get("cite_full") or ""),
            body=str(card.get("body") or ""),
            read_spans=[
                TextSpan(start=int(a), end=int(b))
                for a, b in card.get("read_spans", [])
            ],
            emphasis_spans=[
                TextSpan(start=int(a), end=int(b))
                for a, b in card.get("emphasis_spans", [])
            ],
            side=str(card.get("side") or "unknown"),
            evidence_type=str(card.get("evidence_type") or "unknown"),
            topic_tags=list(card.get("topic_tags", [])),
            indexable=is_indexable(card),
            flags=list(card.get("flags", [])),
            explanation=str(card.get("explanation") or ""),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CaseFileError(
            ErrorCode.INGESTION_CARD_INVALID,
            "An extracted evidence card failed deterministic validation.",
            stage="ingestion.validate_cards",
            agent="evidence_librarian",
            safe_details={"card_position": position},
            cause=exc,
        ) from exc
