"""Token-gated ingestion preview, commit, and quarantine tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from casefile.agents.contracts import StrictContract
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.ingest.pipeline import IngestionPipeline

from .contracts import ApproveCardArgs
from .context import ToolContext, ToolRuntime


class StageIngestionArgs(StrictContract):
    file_path: str = Field(min_length=1, max_length=2000)
    resolution: str | None = Field(default=None, max_length=500)
    side: str | None = Field(default=None, pattern=r"^(pro|con)$")


class CommitIngestionArgs(StrictContract):
    confirmation_token: str = Field(pattern=r"^[0-9a-f]{32}$")
    idempotency_key: str | None = Field(default=None, max_length=200)


class IngestionTools:
    def __init__(
        self,
        runtime: ToolRuntime,
        pipeline: IngestionPipeline | None = None,
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline or IngestionPipeline(runtime.settings)

    def stage_ingestion_preview(
        self,
        context: ToolContext,
        *,
        file_path: str,
        resolution: str | None = None,
        side: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.authorize(
            context,
            tool="stage_ingestion_preview",
            action="stage an evidence ingestion preview",
            roles=frozenset({"student", "coach"}),
            agents=frozenset({"evidence_librarian"}),
        )
        try:
            args = StageIngestionArgs(
                file_path=file_path,
                resolution=resolution,
                side=side,
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "stage_ingestion_preview", exc)
        safe_path = self._resolve_ingest_path(args.file_path, context=context)
        preview = self.pipeline.preview(
            safe_path,
            resolution=args.resolution or context.resolution,
            default_side=args.side or "",
        )
        result = preview.model_dump(mode="json")
        self.runtime.audit(
            context,
            "stage_ingestion_preview",
            {
                "file_path": str(safe_path),
                "resolution": args.resolution or context.resolution,
                "side": args.side,
            },
            result,
        )
        return result

    def commit_ingestion(
        self,
        context: ToolContext,
        *,
        confirmation_token: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.authorize(
            context,
            tool="commit_ingestion",
            action="commit a staged evidence ingestion",
            roles=frozenset({"student", "coach"}),
            agents=frozenset({"evidence_librarian"}),
        )
        try:
            args = CommitIngestionArgs(
                confirmation_token=confirmation_token,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "commit_ingestion", exc)
        prior = self.runtime.idempotent_result("commit_ingestion", args.idempotency_key)
        if prior is not None:
            return prior
        token = args.confirmation_token
        source_path = self._validate_pending_ingest_path(token, context=context)
        committed = self.pipeline.confirm(token)
        result = committed.model_dump(mode="json")
        self._remove_staged_upload(source_path, context=context)
        self.runtime.save_idempotent_result(
            "commit_ingestion", args.idempotency_key, result
        )
        self.runtime.audit(
            context,
            "commit_ingestion",
            {"confirmation_token": token},
            result,
        )
        return result

    def approve_quarantined_card(
        self,
        context: ToolContext,
        *,
        card_id: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        self.runtime.authorize(
            context,
            tool="approve_quarantined_card",
            action="approve quarantined evidence",
            roles=frozenset({"coach"}),
            agents=frozenset({"evidence_librarian"}),
        )
        try:
            args = ApproveCardArgs(
                card_id=card_id,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "approve_quarantined_card", exc)
        prior = self.runtime.idempotent_result(
            "approve_quarantined_card", args.idempotency_key
        )
        if prior is not None:
            return prior
        try:
            result = self.pipeline.approve_quarantined_card(args.card_id)
        except CaseFileError:
            raise
        except FileNotFoundError as exc:
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "The requested quarantined card was not found.",
                tool="approve_quarantined_card",
                cause=exc,
            )
        except json.JSONDecodeError as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_READ_FAILED,
                "The evidence ledger could not be read.",
                tool="approve_quarantined_card",
                cause=exc,
            )
        except ValueError as exc:
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "The requested card cannot be approved.",
                tool="approve_quarantined_card",
                cause=exc,
            )
        except OSError as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_WRITE_FAILED,
                "The quarantined-card approval could not be stored.",
                tool="approve_quarantined_card",
                cause=exc,
            )
        self.runtime.save_idempotent_result(
            "approve_quarantined_card", args.idempotency_key, result
        )
        self.runtime.audit(
            context,
            "approve_quarantined_card",
            {"card_id": args.card_id},
            result,
        )
        return result

    def _resolve_ingest_path(self, value: str, *, context: ToolContext) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "The ingestion file was not found.",
                tool="stage_ingestion_preview",
                safe_details={"source_filename": path.name},
            )
        if path.suffix.lower() != ".docx":
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "Only DOCX ingestion is supported.",
                tool="stage_ingestion_preview",
            )
        roots = (
            *self.runtime.settings.allowed_ingest_roots,
            self.runtime.settings.uploads_dir.resolve(),
        )
        if not any(path.is_relative_to(root) for root in roots):
            self.runtime.fail(
                context,
                ErrorCode.AUTHORIZATION_DENIED,
                "The ingestion file is outside the configured upload roots.",
                tool="stage_ingestion_preview",
                safe_details={"allowed_roots": len(roots)},
            )
        return path

    def _validate_pending_ingest_path(
        self, token: str, *, context: ToolContext
    ) -> Path:
        pending = self.runtime.settings.pending_dir / f"{token}.json"
        if not pending.exists():
            self.runtime.fail(
                context,
                ErrorCode.CONFIRMATION_INVALID,
                "The ingestion confirmation token is unknown, expired, or already used.",
                tool="commit_ingestion",
            )
        try:
            value = json.loads(pending.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_READ_FAILED,
                "The staged ingestion preview could not be read.",
                tool="commit_ingestion",
                cause=exc,
            )
        if not isinstance(value, dict) or not value.get("source_path"):
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_READ_FAILED,
                "The staged ingestion preview is malformed.",
                tool="commit_ingestion",
            )
        source = Path(str(value["source_path"])).expanduser().resolve()
        if not source.exists():
            return source
        return self._resolve_ingest_path(str(source), context=context)

    def _remove_staged_upload(self, path: Path, *, context: ToolContext) -> None:
        uploads = self.runtime.settings.uploads_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(uploads):
            return
        try:
            resolved.unlink(missing_ok=True)
        except OSError as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_WRITE_FAILED,
                "The staged upload could not be removed after ingestion.",
                tool="commit_ingestion",
                cause=exc,
            )
        if resolved.parent != uploads:
            try:
                resolved.parent.rmdir()
            except OSError:
                pass
