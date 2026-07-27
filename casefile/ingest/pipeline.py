"""End-to-end, confirmation-gated DOCX ingestion pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from casefile.config import Settings, get_settings
from casefile.llm import AnthropicJSONClient
from casefile.models import CardBoundary, CardRecord, ParagraphRecord

from .boundary_pass import BARE_URL, run_boundary_pass
from .field_pass import label_card, parse_citation
from .serialize_index import detect_convention, paragraph_records
from .slice_spans import (
    card_convention,
    join_paragraphs,
    span_text,
    spans_for_paragraphs,
)


HTML_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);")
CORRUPTION_FLAGS = {"pdf_paste_fragmented", "text_corrupt", "html_entity"}
NON_INDEXABLE_FLAGS = CORRUPTION_FLAGS | {"do_not_ingest", "paraphrase_no_source"}


@dataclass
class IngestionPreview:
    token: str
    source_file: str
    source_sha256: str
    resolution: str
    resolution_confidence: str
    marking_convention: str
    marking_votes: dict[str, int]
    boundary_method: str
    field_method: str
    validation: dict[str, Any]
    cards: list[dict[str, Any]]

    @property
    def counts(self) -> dict[str, int]:
        counts = {"ok": 0, "flagged": 0, "incomplete": 0, "not_indexable": 0}
        for card in self.cards:
            counts[card["ingest_status"]] += 1
            if not is_indexable(card):
                counts["not_indexable"] += 1
        return counts

    def to_dict(self, include_cards: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["counts"] = self.counts
        if not include_cards:
            value.pop("cards", None)
        return value

    def summary(self) -> str:
        counts = self.counts
        lines = [
            f"Parsed {len(self.cards)} units from {Path(self.source_file).name}.",
            f"  {counts['ok']} ok, {counts['flagged']} flagged, "
            f"{counts['incomplete']} incomplete; {counts['not_indexable']} excluded from search",
            f"  marking convention: {self.marking_convention} votes={self.marking_votes}",
            f"  boundary pass: {self.boundary_method}; field pass: {self.field_method}",
        ]
        flagged = [
            f"{card['header'] or card['author'] or card['id'][:8]} ({', '.join(card['flags'])})"
            for card in self.cards
            if card["flags"]
        ]
        if flagged:
            lines.append("  review: " + "; ".join(flagged))
        lines.append(
            f"Resolution: {self.resolution} ({self.resolution_confidence} confidence)."
        )
        if not self.validation.get("valid"):
            lines.append("  BLOCKED: boundary validation failed; confirmation is disabled.")
        else:
            lines.append(f"Confirm with token: {self.token}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestionPreview":
        allowed = {
            "token", "source_file", "source_sha256", "resolution",
            "resolution_confidence", "marking_convention", "marking_votes",
            "boundary_method", "field_method", "validation", "cards",
        }
        return cls(**{key: value[key] for key in allowed})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card_id(header: str, body: str) -> str:
    normalized = re.sub(r"\s+", " ", body).strip()
    return hashlib.sha256(f"{header.strip()}\n{normalized}".encode("utf-8")).hexdigest()


def _resolution(path: Path, resolution: str | None) -> tuple[str, str]:
    if resolution and resolution.strip():
        return resolution.strip(), "high"
    inferred = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").upper()
    return f"INFERRED-{inferred or 'UNKNOWN'}", "low"


def _boundary_flags(
    boundary: CardBoundary,
    body: str,
    cite: str,
    read_spans: list[list[int]],
) -> set[str]:
    flags = set(boundary.flags)
    if boundary.header is None:
        flags.add("no_header")
    if not boundary.body or not body.strip():
        flags.add("no_body")
    if BARE_URL.fullmatch(cite.strip()):
        flags.add("cite_is_bare_url")
    if not re.search(r"https?://|\b(?:19|20)\d{2}\b", cite, re.I) and len(cite) < 180:
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
        flags.add("pdf_paste_fragmented")
        flags.update({"text_corrupt", "do_not_ingest"})
    if flags & CORRUPTION_FLAGS:
        flags.add("do_not_ingest")
    return flags


def _status(body: str, flags: set[str]) -> str:
    if not body.strip() or "no_body" in flags:
        return "incomplete"
    return "flagged" if flags else "ok"


def _source_key(card: CardRecord) -> str:
    if card.url:
        return re.sub(r"[/?#]+$", "", card.url.lower())
    return re.sub(r"\s+", " ", card.body.lower()).strip()


def is_indexable(card: dict[str, Any]) -> bool:
    return (
        card.get("ingest_status") != "incomplete"
        and bool(card.get("body"))
        and not (set(card.get("flags", [])) & NON_INDEXABLE_FLAGS)
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: AnthropicJSONClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self.llm = llm or AnthropicJSONClient(
            self.settings.anthropic_api_key, self.settings.model
        )

    def preview(
        self,
        docx: str | Path,
        *,
        resolution: str | None,
        default_side: str | None = None,
        use_model: bool = True,
        stage: bool = True,
    ) -> IngestionPreview:
        path = Path(docx).expanduser().resolve()
        records = paragraph_records(path)
        by_id: dict[int, ParagraphRecord] = {record.i: record for record in records}
        convention, votes = detect_convention(records)
        boundary_result, validation, boundary_method = run_boundary_pass(
            records, client=self.llm, use_model=use_model
        )
        resolved, confidence = _resolution(path, resolution)
        cards: list[CardRecord] = []
        field_methods: set[str] = set()

        for raw_boundary in boundary_result.get("cards", []):
            boundary = CardBoundary.from_dict(raw_boundary)
            header = by_id[boundary.header].text if boundary.header in by_id else ""
            tag = join_paragraphs(boundary.tag, by_id)
            cite = join_paragraphs(boundary.cite, by_id)
            card_marking = card_convention(boundary.body, by_id, convention)
            body, read_spans, emphasis_spans = spans_for_paragraphs(
                boundary.body, by_id, card_marking
            )
            flags = _boundary_flags(boundary, body, cite, read_spans)
            labels, field_method = label_card(
                header=header,
                tag=tag,
                cite=cite,
                body=body,
                source_file=path.name,
                client=self.llm if use_model else None,
            )
            field_methods.add(field_method)
            flags.update(labels.get("flags", []))
            if labels.get("evidence_type") == "paraphrased" and not labels.get(
                "source_text_present"
            ):
                flags.add("paraphrase_no_source")
            if flags & CORRUPTION_FLAGS:
                flags.add("do_not_ingest")
            citation = parse_citation(cite, header)
            read_text = span_text(body, read_spans) if read_spans else body
            emphasis_text = span_text(body, emphasis_spans)
            embedding_parts = [header, tag, read_text, emphasis_text]
            embedding_text = "\n".join(part for part in embedding_parts if part)
            returned_document = "\n".join(
                part for part in (cite, header, tag, body) if part
            )
            side = default_side or labels.get("side", "unknown")
            if side not in {"pro", "con", "unknown"}:
                raise ValueError("default_side must be pro, con, or unknown")
            source_paragraphs = sorted(
                set(
                    ([] if boundary.header is None else [boundary.header])
                    + boundary.tag
                    + boundary.cite
                    + boundary.body
                )
            )
            cards.append(
                CardRecord(
                    id=_card_id(header, body),
                    header=header,
                    tag=tag,
                    cite_full=cite,
                    body=body,
                    read_spans=read_spans,
                    emphasis_spans=emphasis_spans,
                    marking_convention=card_marking,
                    evidence_type=labels.get("evidence_type", "unknown"),
                    source_text_present=bool(labels.get("source_text_present")),
                    resolution=resolved,
                    resolution_confidence=confidence,
                    side=side,
                    topic_tags=list(labels.get("topic_tags", [])),
                    ingest_status=_status(body, flags),
                    flags=sorted(flags),
                    source_file=path.name,
                    source_paragraphs=source_paragraphs,
                    embedding_text=embedding_text,
                    returned_document=returned_document,
                    **citation,
                )
            )

        seen_sources: dict[str, CardRecord] = {}
        for card in cards:
            key = _source_key(card)
            if key and key in seen_sources and seen_sources[key].id != card.id:
                card.flags = sorted(set(card.flags) | {"duplicate_source"})
                card.ingest_status = "flagged"
            elif key:
                seen_sources[key] = card

        token = uuid.uuid4().hex
        preview = IngestionPreview(
            token=token,
            source_file=str(path),
            source_sha256=_sha256_file(path),
            resolution=resolved,
            resolution_confidence=confidence,
            marking_convention=convention,
            marking_votes=votes,
            boundary_method=boundary_method,
            field_method="+".join(sorted(field_methods)) or "none",
            validation=validation,
            cards=[card.to_dict() for card in cards],
        )
        if stage:
            _atomic_json(self.settings.pending_dir / f"{token}.json", preview.to_dict())
        return preview

    def confirm(self, token: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("Invalid confirmation token")
        pending = self.settings.pending_dir / f"{token}.json"
        if not pending.exists():
            raise FileNotFoundError("Confirmation token was not found or was already used")
        preview = IngestionPreview.from_dict(json.loads(pending.read_text(encoding="utf-8")))
        if not preview.validation.get("valid"):
            raise ValueError("Cannot confirm an ingest with failed boundary validation")
        source = Path(preview.source_file)
        if not source.exists() or _sha256_file(source) != preview.source_sha256:
            raise ValueError("Source DOCX changed after preview; create a new preview")

        current: list[dict[str, Any]] = []
        if self.settings.cards_path.exists():
            loaded = json.loads(self.settings.cards_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                current = loaded
        by_id = {card["id"]: card for card in current}
        for card in preview.cards:
            by_id[card["id"]] = card
        saved = sorted(by_id.values(), key=lambda card: (card["resolution"], card["id"]))
        _atomic_json(self.settings.cards_path, saved)
        pending.unlink()

        from casefile.retrieval import CaseFileIndex

        index = CaseFileIndex(self.settings)
        index.rebuild_cards(saved)
        return {
            "written": len(preview.cards),
            "searchable": sum(1 for card in preview.cards if is_indexable(card)),
            "total_records": len(saved),
            "cards_path": str(self.settings.cards_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", nargs="?")
    parser.add_argument("--resolution")
    parser.add_argument("--side", choices=["pro", "con", "unknown"])
    parser.add_argument("--confirm", metavar="TOKEN")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    pipeline = IngestionPipeline()
    if args.confirm:
        result = pipeline.confirm(args.confirm)
        print(json.dumps(result, indent=2) if args.as_json else result)
        return
    if not args.docx:
        parser.error("docx is required unless --confirm is used")
    preview = pipeline.preview(
        args.docx,
        resolution=args.resolution,
        default_side=args.side,
        use_model=not args.no_model,
    )
    print(json.dumps(preview.to_dict(), ensure_ascii=False, indent=2) if args.as_json else preview.summary())


if __name__ == "__main__":
    main()
