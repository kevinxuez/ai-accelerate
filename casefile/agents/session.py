"""Versioned, atomic persistence for complete four-agent conversation state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from casefile.config import Settings

from .errors import CaseFileError, ErrorCode
from .state import CASEFILE_STATE_SCHEMA_VERSION, CaseFileState


SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{15,127}$")
_SESSION_LOCK = threading.RLock()


class CaseFileSessionStore:
    """Persist state without deleting or repairing unreadable sessions."""

    def __init__(self, settings: Settings) -> None:
        self.directory = settings.sessions_dir
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_WRITE_FAILED,
                "The session directory could not be created.",
                stage="session.initialize",
                cause=exc,
            ) from exc

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "session_id must be 16 to 128 URL-safe characters.",
                stage="session.validate_id",
                safe_details={"field": "session_id"},
            )
        return session_id

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def load(
        self,
        session_id: str,
        *,
        role: str | None = None,
        user_id: str | None = None,
        resolution: str | None = None,
        required: bool = False,
    ) -> CaseFileState | None:
        path = self._path(session_id)
        with _SESSION_LOCK:
            if not path.exists():
                if required:
                    raise CaseFileError(
                        ErrorCode.SESSION_NOT_FOUND,
                        "The requested session does not exist.",
                        stage="session.load",
                    )
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CaseFileError(
                    ErrorCode.SESSION_CORRUPT,
                    "The saved session could not be read.",
                    stage="session.load",
                    cause=exc,
                ) from exc

        if not isinstance(value, dict):
            self._corrupt("The saved session envelope is malformed.")
        version = value.get("schema_version")
        if version != CASEFILE_STATE_SCHEMA_VERSION:
            raise CaseFileError(
                ErrorCode.SESSION_VERSION_UNSUPPORTED,
                "The saved session uses an unsupported schema version.",
                stage="session.load",
                safe_details={
                    "expected_version": CASEFILE_STATE_SCHEMA_VERSION,
                    "found_version": version,
                },
            )
        raw_state = value.get("state")
        if not isinstance(raw_state, dict):
            self._corrupt("The saved session state is malformed.")
        nested_version = raw_state.get("schema_version")
        if nested_version != CASEFILE_STATE_SCHEMA_VERSION:
            raise CaseFileError(
                ErrorCode.SESSION_VERSION_UNSUPPORTED,
                "The saved state uses an unsupported schema version.",
                stage="session.load",
                safe_details={
                    "expected_version": CASEFILE_STATE_SCHEMA_VERSION,
                    "found_version": nested_version,
                },
            )
        try:
            state = CaseFileState.model_validate_json(json.dumps(raw_state))
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.SESSION_CORRUPT,
                "The saved session does not match the required state schema.",
                stage="session.load",
                safe_details={"schema": CaseFileState.__name__},
                cause=exc,
            ) from exc
        if state.request.session_id != session_id:
            self._corrupt("The saved session identity does not match its storage key.")
        if role is not None and state.request.role != role:
            self._context_denied("role")
        if user_id is not None and state.request.user_id != user_id:
            self._context_denied("user_id")
        if resolution is not None and state.request.active_resolution != resolution:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "The session resolution does not match the active resolution.",
                stage="session.context",
                safe_details={"field": "resolution"},
            )
        return state

    def save(self, state: CaseFileState) -> None:
        state = CaseFileState.model_validate(state.model_dump(mode="python"))
        path = self._path(state.request.session_id)
        payload = {
            "schema_version": CASEFILE_STATE_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state.model_dump(mode="json"),
        }
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=self.directory,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = None
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            with _SESSION_LOCK:
                os.replace(temporary, path)
            temporary = None
        except (OSError, TypeError, ValueError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_WRITE_FAILED,
                "The conversation session could not be saved.",
                stage="session.save",
                request_id=state.request.request_id,
                cause=exc,
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def _path(self, session_id: str) -> Path:
        validated = self.validate_session_id(session_id)
        digest = hashlib.sha256(validated.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    @staticmethod
    def _corrupt(message: str) -> None:
        raise CaseFileError(
            ErrorCode.SESSION_CORRUPT,
            message,
            stage="session.load",
        )

    @staticmethod
    def _context_denied(field: str) -> None:
        raise CaseFileError(
            ErrorCode.AUTHORIZATION_DENIED,
            "The session is owned by a different caller context.",
            stage="session.context",
            safe_details={"field": field},
        )
