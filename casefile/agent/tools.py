"""Role-enforced tool implementations and audit logging."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from casefile.config import Settings, get_settings
from casefile.ingest.pipeline import IngestionPipeline
from casefile.models import ProgressRecord
from casefile.retrieval import CaseFileIndex

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
        result = self.index.search_cards(
            query, resolution=active_resolution, side=side, n=max(1, min(n, 10))
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
        result = self.index.search_rules(question, n=max(1, min(n, 8)))
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
        records = self._read_progress()
        recent = next(
            (record for record in reversed(records) if record.get("student_id") == student_id),
            None,
        )
        weakness_tags = list((recent or {}).get("weakness_tags", []))
        query = " ".join(weakness_tags) or f"{speech_position} evidence"
        cards = self.index.search_cards(
            query, resolution=active_resolution, side=side, n=3
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
            "speech_position": speech_position,
            "resolution": active_resolution,
            "side": side,
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
    ) -> dict[str, Any] | str:
        if context.role != "coach" or "log_assessment" not in available_tools(context.role):
            return self._denied(context, "log assessment records", "log_assessment")
        if not student_id or not assessment_text.strip():
            return "[INVALID] student_id and assessment_text are required."
        record = ProgressRecord(
            student_id=student_id,
            date=date or datetime.now(timezone.utc).date().isoformat(),
            speech_position=speech_position,
            resolution=resolution,
            weakness_tags=sorted({tag.strip().lower() for tag in weakness_tags if tag.strip()}),
            assessment_text=assessment_text.strip(),
            author_role="coach",
            author_id=context.user_id,
        ).to_dict()
        with _write_lock:
            records = self._read_progress()
            records.append(record)
            self._atomic_json(self.settings.progress_path, records)
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
        result = [
            record
            for record in self._read_progress()
            if record.get("student_id") == student_id
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
    ) -> dict[str, Any] | str:
        if context.role != "coach" or "ingest_cards" not in available_tools(context.role):
            return self._denied(context, "ingest evidence cards", "ingest_cards")
        if confirmation_token:
            result = self.ingestion.confirm(confirmation_token)
            self._audit(
                context,
                "ingest_cards",
                {"confirmation_token": confirmation_token, "dry_run": False},
                result,
            )
            return result
        if not file_path:
            return "[INVALID] file_path or confirmation_token is required."
        if not dry_run:
            return "[INVALID] a preview confirmation_token is required before writing."
        preview = self.ingestion.preview(
            file_path,
            resolution=resolution or context.resolution,
            default_side=side,
            use_model=use_model,
        )
        result = preview.to_dict(include_cards=False)
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

    def schedule_session(
        self,
        context: ToolContext,
        *,
        student_id: str,
        start: str,
        duration_minutes: int = 45,
        attendee_email: str | None = None,
        timezone_name: str = "America/Chicago",
    ) -> dict[str, Any] | str:
        if "schedule_session" not in available_tools(context.role):
            return self._denied(context, "schedule sessions", "schedule_session")
        if context.role == "student" and student_id != context.user_id:
            return self._denied(
                context, "schedule a session for another student", "schedule_session"
            )
        if not 15 <= duration_minutes <= 180:
            return "[INVALID] duration_minutes must be between 15 and 180."
        try:
            local_start = datetime.fromisoformat(start)
            zone = ZoneInfo(timezone_name)
            if local_start.tzinfo is None:
                local_start = local_start.replace(tzinfo=zone)
        except (ValueError, TypeError) as exc:
            return f"[INVALID] start/timezone is not valid: {exc}"
        from datetime import timedelta

        local_end = local_start + timedelta(minutes=duration_minutes)
        event_body = {
            "summary": f"CaseFile coaching session — {student_id}",
            "description": f"Public Forum coaching for resolution {context.resolution}",
            "start": {"dateTime": local_start.isoformat(), "timeZone": timezone_name},
            "end": {"dateTime": local_end.isoformat(), "timeZone": timezone_name},
        }
        if attendee_email:
            event_body["attendees"] = [{"email": attendee_email}]
        result = (
            self._mock_calendar_event(event_body)
            if self.settings.mock_calendar
            else self._google_calendar_event(event_body)
        )
        self._audit(
            context,
            "schedule_session",
            {
                "student_id": student_id,
                "start": start,
                "duration_minutes": duration_minutes,
                "timezone_name": timezone_name,
            },
            result,
        )
        return result

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
            "arguments": arguments,
            "caller_role": context.role,
            "caller_id": context.user_id,
            "active_resolution": context.resolution,
            "retrieved_chunk_ids": retrieved,
            "outcome": "denied" if isinstance(result, str) and result.startswith("[DENIED]") else "ok",
        }
        with _write_lock:
            self.settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.settings.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _drill_instructions(
        speech_position: str,
        weakness_tags: list[str],
        card_refs: list[dict[str, Any]],
    ) -> list[str]:
        focus = ", ".join(weakness_tags) if weakness_tags else "clear claim-evidence-warrant links"
        evidence = ", ".join(card["header"] or card["id"][:8] for card in card_refs)
        return [
            f"Prepare a timed {speech_position} using only the cited cards listed below.",
            f"Focus on {focus}.",
            "After delivery, identify the claim, cited evidence, and warrant for each extension.",
            f"Evidence set: {evidence}." if evidence else "No matching card is on file; ask a coach to add evidence before running this drill.",
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
