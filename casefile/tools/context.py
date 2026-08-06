"""Shared context, authorization, storage, idempotency, and audit boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from casefile.agents.contracts import AgentName, Role
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.config import Settings
from casefile.security.prompt_guard import (
    redact_secrets,
    summarize_untrusted_text,
)


WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class ToolContext:
    role: Role
    user_id: str
    resolution: str
    request_id: str | None = None
    agent: AgentName | None = None


class ToolRuntime:
    """Infrastructure shared by focused tool modules."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_runtime_dirs()

    def authorize(
        self,
        context: ToolContext,
        *,
        tool: str,
        action: str,
        roles: frozenset[Role],
        agents: frozenset[AgentName],
    ) -> None:
        if context.role not in roles:
            self.denied(context, action, tool)
        if context.agent is not None and context.agent not in agents:
            self.agent_denied(context, action, tool)

    def invalid(
        self, context: ToolContext, tool: str, exc: ValidationError
    ) -> NoReturn:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first.get("loc", ())) or "arguments"
        self.fail(
            context,
            ErrorCode.REQUEST_INVALID,
            f"Invalid {location}: {first.get('msg', 'invalid value')}.",
            tool=tool,
            cause=exc,
            safe_details={
                "field": location,
                "validation_type": first.get("type", "validation_error"),
            },
        )

    def denied(self, context: ToolContext, action: str, tool: str) -> NoReturn:
        self.fail(
            context,
            ErrorCode.AUTHORIZATION_DENIED,
            f"Role '{context.role}' cannot {action}.",
            tool=tool,
        )

    def agent_denied(self, context: ToolContext, action: str, tool: str) -> NoReturn:
        self.fail(
            context,
            ErrorCode.AUTHORIZATION_DENIED,
            f"Agent '{context.agent}' cannot {action}.",
            tool=tool,
        )

    def blocked(
        self,
        context: ToolContext,
        tool: str,
        arguments: dict[str, Any],
        *,
        signals: list[str],
    ) -> NoReturn:
        self.fail(
            context,
            ErrorCode.AUTHORIZATION_DENIED,
            "Security policy blocked unsafe content from tool execution.",
            tool=tool,
            audit_arguments=arguments,
            safe_details={"signals": list(signals)},
        )

    def fail(
        self,
        context: ToolContext,
        code: ErrorCode,
        message: str,
        *,
        tool: str,
        cause: BaseException | None = None,
        retryable: bool | None = None,
        safe_details: dict[str, Any] | None = None,
        audit_arguments: dict[str, Any] | None = None,
    ) -> NoReturn:
        error = CaseFileError(
            code,
            message,
            stage=f"tools.{tool}",
            agent=context.agent,
            tool=tool,
            retryable=retryable,
            request_id=context.request_id,
            safe_details=safe_details,
            cause=cause,
        )
        self.audit(context, tool, audit_arguments or {}, error)
        raise error

    def read_progress(self) -> list[dict[str, Any]]:
        if not self.settings.progress_path.exists():
            return []
        try:
            value = json.loads(self.settings.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The progress store could not be read.",
                stage="storage.progress.read",
                cause=exc,
            ) from exc
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The progress store is malformed.",
                stage="storage.progress.read",
            )
        return value

    def idempotent_result(self, operation: str, key: str | None) -> Any | None:
        if not key:
            return None
        return self._read_idempotency().get(self._idempotency_slot(operation, key))

    def save_idempotent_result(
        self, operation: str, key: str | None, result: Any
    ) -> None:
        if not key:
            return
        with WRITE_LOCK:
            values = self._read_idempotency()
            values[self._idempotency_slot(operation, key)] = result
            self.atomic_json(self.settings.idempotency_path, values)

    def _read_idempotency(self) -> dict[str, Any]:
        if not self.settings.idempotency_path.exists():
            return {}
        try:
            value = json.loads(
                self.settings.idempotency_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The idempotency store could not be read.",
                stage="storage.idempotency.read",
                cause=exc,
            ) from exc
        if not isinstance(value, dict):
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                "The idempotency store is malformed.",
                stage="storage.idempotency.read",
            )
        return value

    @staticmethod
    def _idempotency_slot(operation: str, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{operation}:{digest}"

    def audit(
        self,
        context: ToolContext,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        retrieved: list[str] = []
        if isinstance(result, list):
            retrieved = [
                str(item["_chunk_id"])
                for item in result
                if isinstance(item, dict) and item.get("_chunk_id")
            ]
        elif isinstance(result, dict):
            retrieved = [
                str(item["id"])
                for item in result.get("card_refs", [])
                if isinstance(item, dict) and item.get("id")
            ]
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "arguments": self._safe_audit_arguments(arguments),
            "caller_role": context.role,
            "caller_id": context.user_id,
            "agent": context.agent,
            "active_resolution": context.resolution,
            "retrieved_chunk_ids": retrieved,
            "outcome": (
                "denied"
                if isinstance(result, CaseFileError)
                and result.code == ErrorCode.AUTHORIZATION_DENIED
                else "failed"
                if isinstance(result, CaseFileError)
                else "ok"
            ),
        }
        try:
            with WRITE_LOCK:
                self.settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.settings.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_WRITE_FAILED,
                "The tool audit record could not be written.",
                stage="storage.tool_audit.write",
                tool=tool,
                request_id=context.request_id,
                cause=exc,
            ) from exc

    @staticmethod
    def _safe_audit_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in {"query", "question", "assessment_text"} and isinstance(
                value, str
            ):
                safe[key] = summarize_untrusted_text(value)
            elif key == "file_path" and isinstance(value, str):
                safe[key] = Path(value).name
            else:
                safe[key] = redact_secrets(value, key=key)
        return safe

    @staticmethod
    def atomic_json(path: Path, value: Any) -> None:
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_WRITE_FAILED,
                "A CaseFile storage write failed.",
                stage="storage.atomic_write",
                cause=exc,
            ) from exc
        finally:
            if temporary is not None and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError as exc:
                    raise CaseFileError(
                        ErrorCode.STORAGE_WRITE_FAILED,
                        "A temporary CaseFile storage file could not be removed.",
                        stage="storage.atomic_write.cleanup",
                        cause=exc,
                    ) from exc
