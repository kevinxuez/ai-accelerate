"""Confirmation-gated coaching-session scheduling tool."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.security.prompt_guard import inspect_text

from .contracts import ScheduleArgs
from .context import WRITE_LOCK, ToolContext, ToolRuntime


CALENDAR_SCOPE = ["https://www.googleapis.com/auth/calendar.events"]


class CalendarTools:
    def __init__(self, runtime: ToolRuntime) -> None:
        self.runtime = runtime

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
    ) -> dict[str, Any]:
        self.runtime.authorize(
            context,
            tool="schedule_session",
            action="schedule sessions",
            roles=frozenset({"student", "coach"}),
            agents=frozenset({"supervisor"}),
        )
        if self.runtime.settings.calendar_provider == "disabled":
            self.runtime.fail(
                context,
                ErrorCode.CAPABILITY_DISABLED,
                "The calendar provider capability is disabled.",
                tool="schedule_session",
            )
        if context.role == "student" and student_id != context.user_id:
            self.runtime.denied(
                context,
                "schedule a session for another student",
                "schedule_session",
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
            self.runtime.invalid(context, "schedule_session", exc)
        decision = inspect_text(
            "\n".join(
                value for value in (args.student_id, args.attendee_email or "") if value
            ),
            trust="untrusted_user",
        )
        if not decision.safe_for_write_tools:
            self.runtime.blocked(
                context,
                "schedule_session",
                {"student_id": args.student_id},
                signals=decision.signals,
            )
        prior = self.runtime.idempotent_result("schedule_session", args.idempotency_key)
        if prior is not None:
            return prior
        if (
            self.runtime.settings.calendar_provider == "google"
            and args.confirmation_token
        ):
            return self._confirm(context, args)
        try:
            local_start = datetime.fromisoformat(args.start)
            zone = ZoneInfo(args.timezone_name)
            if local_start.tzinfo is None:
                local_start = local_start.replace(tzinfo=zone)
        except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
            self.runtime.fail(
                context,
                ErrorCode.REQUEST_INVALID,
                "start or timezone is not valid.",
                tool="schedule_session",
                cause=exc,
            )
        local_end = local_start + timedelta(minutes=args.duration_minutes)
        event_body = {
            "summary": f"CaseFile coaching session — {args.student_id}",
            "description": (
                f"Public Forum coaching for resolution {context.resolution}"
            ),
            "start": {
                "dateTime": local_start.isoformat(),
                "timeZone": args.timezone_name,
            },
            "end": {
                "dateTime": local_end.isoformat(),
                "timeZone": args.timezone_name,
            },
        }
        if args.attendee_email:
            event_body["attendees"] = [{"email": args.attendee_email}]
        if self.runtime.settings.calendar_provider == "fixture":
            result = self._fixture_calendar_event(event_body)
        else:
            token = uuid.uuid4().hex
            staged = {"caller_id": context.user_id, "event": event_body}
            self.runtime.atomic_json(
                self.runtime.settings.calendar_pending_dir / f"{token}.json",
                staged,
            )
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
            self.runtime.save_idempotent_result(
                "schedule_session", args.idempotency_key, result
            )
        self.runtime.audit(
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

    def _confirm(self, context: ToolContext, args: ScheduleArgs) -> dict[str, Any]:
        token = args.confirmation_token or ""
        pending = self.runtime.settings.calendar_pending_dir / f"{token}.json"
        if not pending.exists():
            self.runtime.fail(
                context,
                ErrorCode.CONFIRMATION_INVALID,
                "The calendar confirmation token is unknown, expired, or already used.",
                tool="schedule_session",
            )
        try:
            staged = json.loads(pending.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_READ_FAILED,
                "The staged calendar event could not be read.",
                tool="schedule_session",
                cause=exc,
            )
        if not isinstance(staged, dict) or "event" not in staged:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_READ_FAILED,
                "The staged calendar event is malformed.",
                tool="schedule_session",
            )
        if staged.get("caller_id") != context.user_id:
            self.runtime.denied(
                context,
                "confirm another caller's calendar event",
                "schedule_session",
            )
        try:
            result = self._google_calendar_event(staged["event"])
        except RuntimeError as exc:
            self.runtime.fail(
                context,
                ErrorCode.CALENDAR_NOT_CONFIGURED,
                "Google Calendar is not configured for this deployment.",
                tool="schedule_session",
                cause=exc,
            )
        except CaseFileError:
            raise
        except Exception as exc:
            self.runtime.fail(
                context,
                ErrorCode.CALENDAR_UPSTREAM_ERROR,
                "Google Calendar could not create the requested event.",
                tool="schedule_session",
                cause=exc,
                retryable=True,
            )
        try:
            pending.unlink()
        except OSError as exc:
            self.runtime.fail(
                context,
                ErrorCode.STORAGE_WRITE_FAILED,
                "The used calendar confirmation could not be invalidated.",
                tool="schedule_session",
                cause=exc,
            )
        self.runtime.save_idempotent_result(
            "schedule_session", args.idempotency_key, result
        )
        self.runtime.audit(
            context,
            "schedule_session",
            {"confirmation_token": token},
            result,
        )
        return result

    def _fixture_calendar_event(self, event: dict[str, Any]) -> dict[str, Any]:
        path = self.runtime.settings.data_dir / "calendar_events.json"
        with WRITE_LOCK:
            try:
                events = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else []
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CaseFileError(
                    ErrorCode.STORAGE_READ_FAILED,
                    "The fixture calendar store could not be read.",
                    stage="storage.calendar_fixture.read",
                    tool="schedule_session",
                    cause=exc,
                ) from exc
            if not isinstance(events, list):
                raise CaseFileError(
                    ErrorCode.STORAGE_READ_FAILED,
                    "The fixture calendar store is malformed.",
                    stage="storage.calendar_fixture.read",
                    tool="schedule_session",
                )
            event_id = f"fixture-{len(events) + 1}"
            saved = {
                "id": event_id,
                "status": "confirmed",
                "synthetic": True,
                **event,
            }
            events.append(saved)
            self.runtime.atomic_json(path, events)
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
        settings = self.runtime.settings
        if settings.google_token.exists():
            credentials = Credentials.from_authorized_user_file(
                str(settings.google_token), CALENDAR_SCOPE
            )
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(settings.google_credentials), CALENDAR_SCOPE
                )
                credentials = flow.run_local_server(port=0)
            settings.google_token.write_text(credentials.to_json(), encoding="utf-8")
        service = build("calendar", "v3", credentials=credentials)
        return (
            service.events()
            .insert(
                calendarId="primary",
                body=event,
            )
            .execute()
        )
