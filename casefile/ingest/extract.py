"""Safe DOCX inspection, OOXML extraction, and document screening."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.security.audit import SecurityAuditor
from casefile.security.prompt_guard import GuardDecision, inspect_text

from .contracts import ParagraphRecord
from .ooxml import DocxFormatError, detect_convention, paragraph_records


MAX_DOCX_FILES = 5_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_PARAGRAPHS = 5_000


@dataclass(frozen=True)
class ExtractedDocument:
    source_path: Path
    source_filename: str
    source_sha256: str
    records: list[ParagraphRecord]
    marking_convention: str
    marking_votes: dict[str, int]
    screening: GuardDecision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_docx(path: str | Path, *, max_bytes: int) -> Path:
    source = Path(path).expanduser().resolve()
    try:
        if not source.is_file() or source.suffix.lower() != ".docx":
            raise ValueError("source is not a DOCX file")
        if source.stat().st_size <= 0 or source.stat().st_size > max_bytes:
            raise ValueError("DOCX size is outside the ingestion limit")
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_FILES:
                raise ValueError("DOCX contains too many package members")
            if (
                sum(member.file_size for member in members)
                > MAX_DOCX_UNCOMPRESSED_BYTES
            ):
                raise ValueError("DOCX expands beyond the ingestion limit")
            if "word/document.xml" not in {member.filename for member in members}:
                raise ValueError("DOCX does not contain word/document.xml")
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise CaseFileError(
            ErrorCode.DOCUMENT_PARSE_FAILED,
            "The DOCX could not be inspected safely.",
            stage="ingestion.inspect_docx",
            agent="evidence_librarian",
            safe_details={"source_filename": source.name},
            cause=exc,
        ) from exc
    return source


def extract_docx_paragraphs(source: Path) -> list[ParagraphRecord]:
    try:
        records = paragraph_records(source)
    except (OSError, DocxFormatError, ValueError) as exc:
        raise CaseFileError(
            ErrorCode.DOCUMENT_PARSE_FAILED,
            "The DOCX paragraphs could not be extracted.",
            stage="ingestion.extract_docx_paragraphs",
            agent="evidence_librarian",
            safe_details={"source_filename": source.name},
            cause=exc,
        ) from exc
    if not records or len(records) > MAX_PARAGRAPHS:
        raise CaseFileError(
            ErrorCode.DOCUMENT_PARSE_FAILED,
            "The DOCX contains no usable paragraphs or exceeds the ingestion limit.",
            stage="ingestion.extract_docx_paragraphs",
            agent="evidence_librarian",
            safe_details={
                "source_filename": source.name,
                "paragraph_count": len(records),
            },
        )
    return records


def screen_document(
    records: list[ParagraphRecord],
    *,
    source_filename: str,
    audit: SecurityAuditor,
) -> GuardDecision:
    text = "\n".join(record.text for record in records)
    decision = inspect_text(text, trust="untrusted_document")
    if not decision.safe_for_model:
        audit.record(
            "document_ingestion_blocked",
            decision=decision,
            raw_text=text,
            details={"source_filename": source_filename},
        )
        raise CaseFileError(
            ErrorCode.DOCUMENT_UNSAFE,
            "The DOCX contains unsafe instruction-like content and was not processed.",
            stage="ingestion.screen_document",
            agent="evidence_librarian",
            safe_details={"signals": decision.signals},
        )
    return decision


def extract_document(
    path: str | Path,
    *,
    max_bytes: int,
    audit: SecurityAuditor,
) -> ExtractedDocument:
    source = inspect_docx(path, max_bytes=max_bytes)
    records = extract_docx_paragraphs(source)
    decision = screen_document(
        records,
        source_filename=source.name,
        audit=audit,
    )
    convention, votes = detect_convention(records)
    try:
        digest = sha256_file(source)
    except OSError as exc:
        raise CaseFileError(
            ErrorCode.DOCUMENT_PARSE_FAILED,
            "The DOCX source hash could not be calculated.",
            stage="ingestion.inspect_docx",
            agent="evidence_librarian",
            safe_details={"source_filename": source.name},
            cause=exc,
        ) from exc
    return ExtractedDocument(
        source_path=source,
        source_filename=source.name,
        source_sha256=digest,
        records=records,
        marking_convention=convention,
        marking_votes=votes,
        screening=decision,
    )
