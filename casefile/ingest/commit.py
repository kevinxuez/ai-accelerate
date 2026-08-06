"""Staging, atomic ledger commit, reindexing, and quarantine approval."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from casefile.agents.contracts import IngestionCommitResult
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.config import Settings
from casefile.retrieval import COLLECTION_SCHEMA_VERSION, CaseFileIndex
from casefile.security.audit import SecurityAuditor

from .extract import sha256_file
from .preview import StagedIngestion
from .validate import is_indexable


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
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
    except OSError as exc:
        raise CaseFileError(
            ErrorCode.STORAGE_WRITE_FAILED,
            "CaseFile could not write ingestion storage atomically.",
            stage="ingestion.storage.write",
            agent="evidence_librarian",
            cause=exc,
        ) from exc


def stage_ingestion(settings: Settings, staged: StagedIngestion) -> None:
    atomic_json(
        settings.pending_dir / f"{staged.confirmation_token}.json",
        staged.to_dict(),
    )


def load_staged_ingestion(settings: Settings, token: str) -> StagedIngestion:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise CaseFileError(
            ErrorCode.CONFIRMATION_INVALID,
            "The ingestion confirmation token is invalid.",
            stage="ingestion.confirmation",
            agent="evidence_librarian",
        )
    path = settings.pending_dir / f"{token}.json"
    if not path.exists():
        raise CaseFileError(
            ErrorCode.CONFIRMATION_INVALID,
            "The ingestion confirmation token is unknown, expired, or already used.",
            stage="ingestion.confirmation",
            agent="evidence_librarian",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("staged ingestion must be an object")
        return StagedIngestion.from_dict(value)
    except OSError as exc:
        raise CaseFileError(
            ErrorCode.STORAGE_READ_FAILED,
            "The staged ingestion preview could not be read.",
            stage="ingestion.storage.read",
            agent="evidence_librarian",
            cause=exc,
        ) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CaseFileError(
            ErrorCode.STORAGE_READ_FAILED,
            "The staged ingestion preview is malformed.",
            stage="ingestion.storage.read",
            agent="evidence_librarian",
            cause=exc,
        ) from exc


def commit_ingestion(
    settings: Settings,
    token: str,
) -> IngestionCommitResult:
    staged = load_staged_ingestion(settings, token)
    source = Path(staged.source_path)
    try:
        unchanged = source.is_file() and sha256_file(source) == staged.source_sha256
    except OSError as exc:
        raise CaseFileError(
            ErrorCode.INGESTION_SOURCE_CHANGED,
            "The source DOCX could not be verified; create a new preview.",
            stage="ingestion.verify_source",
            agent="evidence_librarian",
            cause=exc,
        ) from exc
    if not unchanged:
        raise CaseFileError(
            ErrorCode.INGESTION_SOURCE_CHANGED,
            "The source DOCX changed after preview; create a new preview.",
            stage="ingestion.verify_source",
            agent="evidence_librarian",
        )

    current: list[dict[str, Any]] = []
    if settings.cards_path.exists():
        try:
            loaded = json.loads(settings.cards_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The evidence ledger could not be read.",
                stage="ingestion.ledger.read",
                agent="evidence_librarian",
                cause=exc,
            ) from exc
        if not isinstance(loaded, list) or not all(
            isinstance(card, dict) for card in loaded
        ):
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The evidence ledger is malformed.",
                stage="ingestion.ledger.read",
                agent="evidence_librarian",
            )
        current = loaded
    by_id = {str(card.get("id")): card for card in current}
    by_id.update({str(card["id"]): card for card in staged.cards})
    saved = sorted(
        by_id.values(),
        key=lambda card: (str(card.get("resolution", "")), str(card.get("id", ""))),
    )
    atomic_json(settings.cards_path, saved)

    try:
        CaseFileIndex(settings).rebuild_cards(saved)
    except Exception as exc:
        try:
            atomic_json(
                settings.data_dir / "index_inconsistency.json",
                {
                    "job_id": staged.job_id,
                    "source_filename": staged.source_filename,
                    "ledger_committed": True,
                    "index_rebuilt": False,
                    "cause_type": type(exc).__name__,
                },
            )
        except CaseFileError:
            pass
        raise CaseFileError(
            ErrorCode.INDEX_REBUILD_FAILED,
            "The evidence ledger was committed but its Chroma index rebuild failed.",
            stage="ingestion.rebuild_index",
            agent="evidence_librarian",
            retryable=False,
            safe_details={"ledger_committed": True, "job_id": staged.job_id},
            cause=exc,
        ) from exc

    try:
        (settings.pending_dir / f"{token}.json").unlink()
    except OSError as exc:
        raise CaseFileError(
            ErrorCode.STORAGE_WRITE_FAILED,
            "The used ingestion confirmation token could not be retired.",
            stage="ingestion.confirmation.retire",
            agent="evidence_librarian",
            cause=exc,
        ) from exc
    return IngestionCommitResult(
        job_id=staged.job_id,
        source_filename=staged.source_filename,
        written_cards=len(staged.cards),
        searchable_cards=sum(is_indexable(card) for card in staged.cards),
        ledger_schema_version=COLLECTION_SCHEMA_VERSION,
        index_rebuilt=True,
    )


def approve_quarantined_card(
    settings: Settings,
    audit: SecurityAuditor,
    card_id: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", card_id):
        raise ValueError("Invalid card id")
    if not settings.cards_path.exists():
        raise FileNotFoundError("No ingested cards are available")
    loaded = json.loads(settings.cards_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("Card ledger is malformed")
    found = False
    for card in loaded:
        if card.get("id") == card_id:
            if card.get("injection_risk") != "high":
                raise ValueError("Only high-risk quarantined cards require approval")
            card["injection_approved"] = True
            found = True
            break
    if not found:
        raise FileNotFoundError("Card id was not found")
    atomic_json(settings.cards_path, loaded)
    searchable = CaseFileIndex(settings).rebuild_cards(loaded)
    audit.record(
        "quarantined_card_approved",
        details={"card_id": card_id, "searchable_records": searchable},
    )
    return {
        "card_id": card_id,
        "injection_approved": True,
        "searchable_records": searchable,
    }
