"""Role-enforced tool implementations and audit logging."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from casefile.config import Settings, get_settings
from casefile.ingest.pipeline import IngestionPipeline
from casefile.models import ProgressRecord
from casefile.retrieval import CaseFileIndex
from casefile.security.prompt_guard import (
    BLOCKED_RESPONSE,
    inspect_text,
    redact_secrets,
    summarize_untrusted_text,
)
from casefile.security.schemas import (
    ApproveCardArgs,
    AssessmentArgs,
    DrillArgs,
    IngestArgs,
    ProgressArgs,
    ScheduleArgs,
    SearchCardsArgs,
    SearchRulesArgs,
)
from pydantic import ValidationError

from .roles import available_tools, denial


_write_lock = threading.RLock()
CALENDAR_SCOPE = ["https://www.googleapis.com/auth/calendar.events"]


@dataclass(frozen=True)
class ToolContext:
    role: str
    user_id: str
    resolution: str


class CaseFileTools:
    def __init__(
        self,
        settings: Settings | None = None,
        index: CaseFileIndex | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self.index = index or CaseFileIndex(self.settings)
        self.ingestion = IngestionPipeline(self.settings)

    def names_for_role(self, role: str) -> set[str]:
        return available_tools(role)

    def search_cards(
        self,
        context: ToolContext,
        query: str,
        side: str,
        resolution: str | None = None,
        n: int = 5,
    ) -> list[dict[str, Any]] | str:
        if "search_cards" not in available_tools(context.role):
            return self._denied(context, "search evidence", "search_cards")
        if side not in {"pro", "con"}:
            return "[INVALID] side must be 'pro' or 'con'."
        active_resolution = resolution or context.resolution
        if not active_resolution:
            return "[INVALID] resolution is required for evidence retrieval."
        try:
            args = SearchCardsArgs(
                query=query, side=side, resolution=active_resolution, n=n
            )
        except ValidationError as exc:
            return self._invalid(exc)
        decision = inspect_text(args.query, trust="untrusted_user")
        if decision.action == "block":
            self._audit(context, "search_cards", {"query": query}, BLOCKED_RESPONSE)
            return BLOCKED_RESPONSE
        result = self.index.search_cards(
            args.query, resolution=args.resolution, side=args.side, n=args.n
        )
        self._audit(
            context,
            "search_cards",
            {"query": query, "side": side, "resolution": active_resolution, "n": n},
            result,
        )
        return result

    def search_rules(
        self, context: ToolContext, question: str, n: int = 3
    ) -> list[dict[str, Any]] | str:
        if "search_rules" not in available_tools(context.role):
            return self._denied(context, "search rules", "search_rules")
        try:
            args = SearchRulesArgs(question=question, n=n)
        except ValidationError as exc:
            return self._invalid(exc)
        decision = inspect_text(args.question, trust="untrusted_user")
        if decision.action == "block":
            self._audit(context, "search_rules", {"question": question}, BLOCKED_RESPONSE)
            return BLOCKED_RESPONSE
        result = self.index.search_rules(args.question, n=args.n)
        self._audit(context, "search_rules", {"question": question, "n": n}, result)
        return result

    def generate_drill(
        self,
        context: ToolContext,
        student_id: str,
        speech_position: str,
        resolution: str | None,
        side: str,
    ) -> dict[str, Any] | str:
        if "generate_drill" not in available_tools(context.role):
            return self._denied(context, "generate drills", "generate_drill")
        if context.role == "student" and student_id != context.user_id:
            return self._denied(
                context, "generate a drill for another student", "generate_drill"
            )
        if not speech_position.strip():
            return "[INVALID] speech_position is required."
        if side not in {"pro", "con"}:
            return "[INVALID] side must be 'pro' or 'con'."
        active_resolution = resolution or context.resolution
        try:
            args = DrillArgs(
                student_id=student_id,
                speech_position=speech_position,
                resolution=active_resolution,
                side=side,
            )
        except ValidationError as exc:
            return self._invalid(exc)
        records = self._read_progress()
        recent = next(
            (record for record in reversed(records) if record.get("student_id") == student_id),
            None,
        )
        weakness_tags = list((recent or {}).get("weakness_tags", []))
        query = " ".join(weakness_tags) or f"{args.speech_position} evidence"
        cards = self.index.search_cards(
            query, resolution=args.resolution, side=args.side, n=3
        )
        card_refs = [
            {
                "id": card["id"],
                "header": card["header"],
                "cite_full": card["cite_full"],
            }
            for card in cards
        ]
        drill = {
            "student_id": student_id,
            "speech_position": args.speech_position,
            "resolution": args.resolution,
            "side": args.side,
            "weakness_tags": weakness_tags,
            "instructions": self._drill_instructions(speech_position, weakness_tags, card_refs),
            "card_refs": card_refs,
        }
        self._audit(
            context,
            "generate_drill",
            {
                "student_id": student_id,
                "speech_position": speech_position,
                "resolution": active_resolution,
                "side": side,
            },
            drill,
        )
        return drill

    def log_assessment(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        resolution: str,
        weakness_tags: list[str],
        assessment_text: str,
        date: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any] | str:
        if context.role != "coach" or "log_assessment" not in available_tools(context.role):
            return self._denied(context, "log assessment records", "log_assessment")
        try:
            args = AssessmentArgs(
                student_id=student_id,
                speech_position=speech_position,
                resolution=resolution,
                weakness_tags=weakness_tags,
                assessment_text=assessment_text,
                date=date,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            return self._invalid(exc)
        decision = inspect_text(args.assessment_text, trust="untrusted_user")
        if not decision.safe_for_write_tools:
            self._audit(context, "log_assessment", {"assessment_text": assessment_text}, BLOCKED_RESPONSE)
            return BLOCKED_RESPONSE
        prior = self._idempotent_result("log_assessment", args.idempotency_key)
        if prior is not None:
            return prior
        record = ProgressRecord(
            student_id=args.student_id,
            date=args.date or datetime.now(timezone.utc).date().isoformat(),
            speech_position=args.speech_position,
            resolution=args.resolution,
            weakness_tags=sorted({tag.lower() for tag in args.weakness_tags}),
            assessment_text=args.assessment_text,
            author_role="coach",
            author_id=context.user_id,
        ).to_dict()
        with _write_lock:
            records = self._read_progress()
            records.append(record)
            self._atomic_json(self.settings.progress_path, records)
            self._save_idempotent_result(
                "log_assessment", args.idempotency_key, record
            )
        self._audit(context, "log_assessment", record, {"written": True})
        return record

    def get_progress(
        self, context: ToolContext, student_id: str
    ) -> list[dict[str, Any]] | str:
        if "get_progress" not in available_tools(context.role):
            return self._denied(context, "read progress records", "get_progress")
        if context.role == "student" and student_id != context.user_id:
            return self._denied(
                context,
                "read progress records for another student",
                "get_progress",
            )
        try:
            args = ProgressArgs(student_id=student_id)
        except ValidationError as exc:
            return self._invalid(exc)
        result = [
            record
            for record in self._read_progress()
            if record.get("student_id") == args.student_id
        ]
        self._audit(context, "get_progress", {"student_id": student_id}, result)
        return result

    def ingest_cards(
        self,
        context: ToolContext,
        *,
        file_path: str | None = None,
        resolution: str | None = None,
        side: str | None = None,
        dry_run: bool = True,
        confirmation_token: str | None = None,
        use_model: bool = True,
        idempotency_key: str | None = None,
    ) -> dict[str, Any] | str:
        if "ingest_cards" not in available_tools(context.role):
            return self._denied(context, "ingest evidence cards", "ingest_cards")
        try:
            args = IngestArgs(
                file_path=file_path,
                resolution=resolution,
                side=side,
                dry_run=dry_run,
                confirmation_token=confirmation_token,
                use_model=use_model,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            return self._invalid(exc)
        if args.confirmation_token:
            prior = self._idempotent_result("ingest_confirm", args.idempotency_key)
            if prior is not None:
                return prior
            source_path = self._validate_pending_ingest_path(args.confirmation_token)
            result = self.ingestion.confirm(args.confirmation_token)
            self._remove_staged_upload(source_path)
            self._save_idempotent_result(
                "ingest_confirm", args.idempotency_key, result
            )
            self._audit(
                context,
                "ingest_cards",
                {"confirmation_token": confirmation_token, "dry_run": False},
                result,
            )
            return result
        if not args.file_path:
            return "[INVALID] file_path or confirmation_token is required."
        if not args.dry_run:
            return "[INVALID] a preview confirmation_token is required before writing."
        safe_path = self._resolve_ingest_path(args.file_path)
        preview = self.ingestion.preview(
            safe_path,
            resolution=args.resolution or context.resolution,
            default_side=args.side,
            use_model=args.use_model,
        )
        result = preview.to_dict(include_cards=False)
        result["quarantined_cards"] = [
            {
                "id": card["id"],
                "injection_risk": card.get("injection_risk", "low"),
                "injection_signals": card.get("injection_signals", []),
            }
            for card in preview.cards
            if card.get("injection_risk") == "high"
            and not card.get("injection_approved")
        ]
        result["summary"] = preview.summary()
        self._audit(
            context,
            "ingest_cards",
            {
                "file_path": str(Path(file_path).resolve()),
                "resolution": resolution or context.resolution,
                "side": side,
                "dry_run": True,
            },
            result,
        )
        return result

    def approve_quarantined_card(
        self,
        context: ToolContext,
        *,
        card_id: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any] | str:
        if (
            context.role != "coach"
            or "approve_quarantined_card" not in available_tools(context.role)
        ):
            return self._denied(
                context, "approve quarantined evidence", "approve_quarantined_card"
            )
        try:
            args = ApproveCardArgs(
                card_id=card_id, idempotency_key=idempotency_key
            )
        except ValidationError as exc:
            return self._invalid(exc)
        prior = self._idempotent_result(
            "approve_quarantined_card", args.idempotency_key
        )
        if prior is not None:
            return prior
        result = self.ingestion.approve_quarantined_card(args.card_id)
        self._save_idempotent_result(
            "approve_quarantined_card", args.idempotency_key, result
        )
        self._audit(
            context, "approve_quarantined_card", {"card_id": args.card_id}, result
        )
        return result

    def schedule_session(
        self,
        context: ToolContext,
        *,
        student_id: str,
        start: str,
        duration_minutes: int = 45,
        attendee_email: str | None = None,
        timezone_name: str = "America/Chicago",
        confirmation_token: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any] | str:
        if "schedule_session" not in available_tools(context.role):
            return self._denied(context, "schedule sessions", "schedule_session")
        if context.role == "student" and student_id != context.user_id:
            return self._denied(
                context, "schedule a session for another student", "schedule_session"
            )
        try:
            args = ScheduleArgs(
                student_id=student_id,
                start=start,
                duration_minutes=duration_minutes,
                attendee_email=attendee_email,
                timezone_name=timezone_name,
                confirmation_token=confirmation_token,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            return self._invalid(exc)
        decision = inspect_text(
            "\n".join(
                value
                for value in (
                    args.student_id,
                    args.attendee_email or "",
                )
                if value
            ),
            trust="untrusted_user",
        )
        if not decision.safe_for_write_tools:
            self._audit(
                context,
                "schedule_session",
                {"student_id": args.student_id},
                BLOCKED_RESPONSE,
            )
            return BLOCKED_RESPONSE
        prior = self._idempotent_result("schedule_session", args.idempotency_key)
        if prior is not None:
            return prior
        if not self.settings.mock_calendar and args.confirmation_token:
            pending = self.settings.calendar_pending_dir / f"{args.confirmation_token}.json"
            if not pending.exists():
                return "[INVALID] Calendar confirmation token was not found or was already used."
            staged = json.loads(pending.read_text(encoding="utf-8"))
            if staged.get("caller_id") != context.user_id:
                return self._denied(
                    context, "confirm another caller's calendar event", "schedule_session"
                )
            result = self._google_calendar_event(staged["event"])
            pending.unlink()
            self._save_idempotent_result(
                "schedule_session", args.idempotency_key, result
            )
            self._audit(
                context,
                "schedule_session",
                {"confirmation_token": args.confirmation_token},
                result,
            )
            return result
        try:
            local_start = datetime.fromisoformat(args.start)
            zone = ZoneInfo(args.timezone_name)
            if local_start.tzinfo is None:
                local_start = local_start.replace(tzinfo=zone)
        except (ValueError, TypeError) as exc:
            return f"[INVALID] start/timezone is not valid: {exc}"
        from datetime import timedelta

        local_end = local_start + timedelta(minutes=args.duration_minutes)
        event_body = {
            "summary": f"CaseFile coaching session — {args.student_id}",
            "description": f"Public Forum coaching for resolution {context.resolution}",
            "start": {"dateTime": local_start.isoformat(), "timeZone": args.timezone_name},
            "end": {"dateTime": local_end.isoformat(), "timeZone": args.timezone_name},
        }
        if args.attendee_email:
            event_body["attendees"] = [{"email": args.attendee_email}]
        if self.settings.mock_calendar:
            result = self._mock_calendar_event(event_body)
        else:
            token = uuid.uuid4().hex
            staged = {
                "caller_id": context.user_id,
                "event": event_body,
            }
            self._atomic_json(self.settings.calendar_pending_dir / f"{token}.json", staged)
            result = {
                "confirmation_required": True,
                "confirmation_token": token,
                "event": event_body,
                "summary": (
                    "A real Google Calendar write requires confirmation. "
                    f"Confirm with token: {token}"
                ),
            }
        if not result.get("confirmation_required"):
            self._save_idempotent_result(
                "schedule_session", args.idempotency_key, result
            )
        self._audit(
            context,
            "schedule_session",
            {
                "student_id": args.student_id,
                "start": args.start,
                "duration_minutes": args.duration_minutes,
                "timezone_name": args.timezone_name,
            },
            result,
        )
        return result

    @staticmethod
    def _invalid(exc: ValidationError) -> str:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first.get("loc", ())) or "arguments"
        return f"[INVALID] {location}: {first.get('msg', 'invalid value')}."

    def _resolve_ingest_path(self, value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Ingest file was not found: {path}")
        if path.suffix.lower() != ".docx":
            raise ValueError("Only .docx ingestion is supported")
        roots = (*self.settings.allowed_ingest_roots, self.settings.uploads_dir.resolve())
        if not any(path.is_relative_to(root) for root in roots):
            allowed = ", ".join(str(root) for root in roots)
            raise ValueError(f"Ingest path is outside configured roots: {allowed}")
        return path

    def _validate_pending_ingest_path(self, token: str) -> Path:
        pending = self.settings.pending_dir / f"{token}.json"
        if not pending.exists():
            raise FileNotFoundError("Confirmation token was not found or was already used")
        value = json.loads(pending.read_text(encoding="utf-8"))
        return self._resolve_ingest_path(str(value.get("source_file", "")))

    def _remove_staged_upload(self, path: Path) -> None:
        uploads = self.settings.uploads_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(uploads):
            return
        resolved.unlink(missing_ok=True)
        if resolved.parent != uploads:
            try:
                resolved.parent.rmdir()
            except OSError:
                pass

    def _read_idempotency(self) -> dict[str, Any]:
        if not self.settings.idempotency_path.exists():
            return {}
        value = json.loads(self.settings.idempotency_path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}

    def _idempotent_result(self, operation: str, key: str | None) -> Any | None:
        if not key:
            return None
        return self._read_idempotency().get(self._idempotency_slot(operation, key))

    def _save_idempotent_result(
        self, operation: str, key: str | None, result: Any
    ) -> None:
        if not key:
            return
        with _write_lock:
            values = self._read_idempotency()
            values[self._idempotency_slot(operation, key)] = result
            self._atomic_json(self.settings.idempotency_path, values)

    @staticmethod
    def _idempotency_slot(operation: str, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{operation}:{digest}"

    def _denied(self, context: ToolContext, action: str, tool: str) -> str:
        result = denial(context.role, action)
        self._audit(context, tool, {}, result)
        return result

    def _read_progress(self) -> list[dict[str, Any]]:
        if not self.settings.progress_path.exists():
            return []
        value = json.loads(self.settings.progress_path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    def _audit(
        self,
        context: ToolContext,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        retrieved = []
        if isinstance(result, list):
            retrieved = [
                item.get("_chunk_id")
                for item in result
                if isinstance(item, dict) and item.get("_chunk_id")
            ]
        elif isinstance(result, dict):
            retrieved = [
                item.get("id")
                for item in result.get("card_refs", [])
                if isinstance(item, dict) and item.get("id")
            ]
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "arguments": self._safe_audit_arguments(arguments),
            "caller_role": context.role,
            "caller_id": context.user_id,
            "active_resolution": context.resolution,
            "retrieved_chunk_ids": retrieved,
            "outcome": (
                "denied"
                if isinstance(result, str) and result.startswith("[DENIED]")
                else "blocked"
                if isinstance(result, str)
                and result.startswith("[BLOCKED_PROMPT_INJECTION]")
                else "ok"
            ),
        }
        with _write_lock:
            self.settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_audit_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in {"query", "question", "assessment_text"} and isinstance(value, str):
                safe[key] = summarize_untrusted_text(value)
            elif key == "file_path" and isinstance(value, str):
                safe[key] = str(Path(value).name)
            else:
                safe[key] = redact_secrets(value, key=key)
        return safe

    @staticmethod
    def _drill_instructions(
        speech_position: str,
        weakness_tags: list[str],
        card_refs: list[dict[str, Any]],
    ) -> list[str]:
        focus = ", ".join(weakness_tags) if weakness_tags else "clear claim-evidence-warrant links"
        evidence = ", ".join(card["header"] or card["id"][:8] for card in card_refs)
        evidence_line = (
            f"Evidence set: {evidence}."
            if evidence
            else "No matching card is on file; add evidence before running this drill."
        )
        if speech_position == "general":
            return [
                "Complete a general claim-evidence-warrant drill using only the cited cards listed below.",
                "For each card, state the claim in 15 seconds, explain its warrant in 30 seconds, and give one comparison to the other side.",
                f"Focus on {focus}.",
                evidence_line,
            ]
        return [
            f"Prepare a timed {speech_position} using only the cited cards listed below.",
            f"Focus on {focus}.",
            "After delivery, identify the claim, cited evidence, and warrant for each extension.",
            evidence_line,
        ]

    def _mock_calendar_event(self, event: dict[str, Any]) -> dict[str, Any]:
        path = self.settings.data_dir / "calendar_events.json"
        with _write_lock:
            events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            event_id = f"mock-{len(events) + 1}"
            saved = {"id": event_id, "status": "confirmed", "mock": True, **event}
            events.append(saved)
            self._atomic_json(path, events)
        return saved

    def _google_calendar_event(self, event: dict[str, Any]) -> dict[str, Any]:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError("Install CaseFile with the 'calendar' extra") from exc
        credentials = None
        if self.settings.google_token.exists():
            credentials = Credentials.from_authorized_user_file(
                str(self.settings.google_token), CALENDAR_SCOPE
            )
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.settings.google_credentials), CALENDAR_SCOPE
                )
                credentials = flow.run_local_server(port=0)
            self.settings.google_token.write_text(credentials.to_json(), encoding="utf-8")
        service = build("calendar", "v3", credentials=credentials)
        return service.events().insert(calendarId="primary", body=event).execute()

    @staticmethod
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
