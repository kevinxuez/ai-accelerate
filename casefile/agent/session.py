"""Small durable store for bounded clarification sessions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from casefile.config import Settings


SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{15,127}$")
SESSION_TTL = timedelta(hours=24)
_session_lock = threading.RLock()


@dataclass(frozen=True)
class PendingClarification:
    session_id: str
    role: str
    user_id: str
    resolution: str
    message: str
    intent: str
    parameters: dict[str, Any]
    question: str
    turns: int
    updated_at: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PendingClarification":
        return cls(
            session_id=str(value["session_id"]),
            role=str(value["role"]),
            user_id=str(value["user_id"]),
            resolution=str(value["resolution"]),
            message=str(value["message"]),
            intent=str(value["intent"]),
            parameters=dict(value.get("parameters", {})),
            question=str(value.get("question", "")),
            turns=int(value.get("turns", 1)),
            updated_at=str(value["updated_at"]),
        )


class ClarificationSessionStore:
    def __init__(self, settings: Settings) -> None:
        self.directory = settings.sessions_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError(
                "session_id must be 16 to 128 URL-safe characters"
            )
        return session_id

    def _path(self, session_id: str) -> Path:
        self.validate_session_id(session_id)
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.json"

    def load(
        self,
        session_id: str,
        *,
        role: str,
        user_id: str,
        resolution: str,
    ) -> PendingClarification | None:
        path = self._path(session_id)
        with _session_lock:
            if not path.exists():
                return None
            try:
                pending = PendingClarification.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                updated = datetime.fromisoformat(pending.updated_at)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                return None
            now = datetime.now(timezone.utc)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if now - updated > SESSION_TTL:
                path.unlink(missing_ok=True)
                return None
            if (
                pending.role != role
                or pending.user_id != user_id
                or pending.resolution != resolution
            ):
                raise ValueError(
                    "session context does not match the original role, user, and resolution"
                )
            return pending

    def save(self, pending: PendingClarification) -> None:
        path = self._path(pending.session_id)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(pending), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            with _session_lock:
                os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def clear(self, session_id: str) -> None:
        with _session_lock:
            self._path(session_id).unlink(missing_ok=True)


def pending_clarification(
    *,
    session_id: str,
    role: str,
    user_id: str,
    resolution: str,
    message: str,
    intent: str,
    parameters: dict[str, Any],
    question: str,
    turns: int,
) -> PendingClarification:
    return PendingClarification(
        session_id=session_id,
        role=role,
        user_id=user_id,
        resolution=resolution,
        message=message,
        intent=intent,
        parameters=parameters,
        question=question,
        turns=turns,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
