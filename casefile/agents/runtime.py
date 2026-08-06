"""Construction and session-aware invocation of the four-agent graph."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from casefile.config import Settings, get_settings
from casefile.llm import build_anthropic_client
from casefile.security.audit import RateLimiter, SecurityAuditor
from casefile.tools import CaseFileTools

from .argument_strategist import ArgumentStrategist
from .contracts import (
    MAX_MESSAGES,
    MAX_GRAPH_STEPS,
    AttachmentHandle,
    ConversationMessage,
    RequestContext,
)
from .errors import CaseFileError, ErrorCode
from .evidence_librarian import EvidenceLibrarian
from .graph import FourAgentGraphNodes, compile_four_agent_graph
from .session import CaseFileSessionStore
from .skills_coach import SkillsCoach
from .state import CaseFileState
from .supervisor import Supervisor


class CaseFileRuntime:
    """The only new-runtime entrypoint; every allowed request reaches Supervisor."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: Any | None = None,
        tools: CaseFileTools | None = None,
        supervisor: Supervisor | None = None,
        evidence_librarian: EvidenceLibrarian | None = None,
        argument_strategist: ArgumentStrategist | None = None,
        skills_coach: SkillsCoach | None = None,
        session_store: CaseFileSessionStore | None = None,
        graph_factory: Callable[[FourAgentGraphNodes], Any] | None = None,
        max_steps: int = MAX_GRAPH_STEPS,
        attachment_resolver: Callable[[AttachmentHandle], str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        try:
            self.settings.validate_configuration()
        except ValueError as exc:
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                "CaseFile provider configuration is invalid.",
                stage="runtime.configuration",
                cause=exc,
            ) from exc
        self.model = model or build_anthropic_client(self.settings)
        validate_model = getattr(self.model, "validate_configuration", None)
        if callable(validate_model):
            validate_model()
        self.tools = tools or CaseFileTools(self.settings, model=self.model)
        self.supervisor = supervisor or Supervisor(self.model)
        self.evidence_librarian = evidence_librarian or EvidenceLibrarian(
            self.tools,
            self.model,
        )
        self.argument_strategist = argument_strategist or ArgumentStrategist(self.model)
        self.skills_coach = skills_coach or SkillsCoach(self.tools, self.model)
        self.sessions = session_store or CaseFileSessionStore(self.settings)
        self.nodes = FourAgentGraphNodes(
            supervisor=self.supervisor,
            evidence_librarian=self.evidence_librarian,
            argument_strategist=self.argument_strategist,
            skills_coach=self.skills_coach,
            tools=self.tools,
            model=self.model,
            security_auditor=SecurityAuditor(self.settings.security_audit_path),
            rate_limiter=RateLimiter(max(1, self.settings.requests_per_minute)),
            max_steps=max_steps,
            attachment_resolver=attachment_resolver,
        )
        self._compiled = (graph_factory or compile_four_agent_graph)(self.nodes)

    @property
    def backend(self) -> str:
        return "langgraph"

    def ask(
        self,
        message: str,
        *,
        role: str,
        user_id: str,
        resolution: str,
        request_id: str | None = None,
        session_id: str | None = None,
        attachments: list[AttachmentHandle | dict[str, Any]] | None = None,
    ) -> CaseFileState:
        active_request_id = request_id or uuid.uuid4().hex
        active_session_id = self.sessions.validate_session_id(
            session_id or uuid.uuid4().hex
        )
        try:
            parsed_attachments = [
                AttachmentHandle.model_validate(attachment)
                for attachment in (attachments or [])
            ]
            request = RequestContext(
                request_id=active_request_id,
                session_id=active_session_id,
                role=role,
                user_id=user_id,
                active_resolution=resolution,
                attachments=parsed_attachments,
            )
            user_message = ConversationMessage(role="user", content=message)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            field = ".".join(str(item) for item in first.get("loc", ())) or "request"
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                f"Invalid {field}: {first.get('msg', 'invalid value')}.",
                stage="runtime.request",
                request_id=active_request_id,
                safe_details={"field": field},
                cause=exc,
            ) from exc

        saved = self.sessions.load(
            active_session_id,
            role=role,
            user_id=user_id,
            resolution=resolution,
        )
        if saved is None:
            state = CaseFileState(request=request, messages=[user_message])
        else:
            if len(saved.messages) >= MAX_MESSAGES:
                raise CaseFileError(
                    ErrorCode.STATE_LIMIT_EXCEEDED,
                    "The conversation message limit was exceeded.",
                    stage="runtime.state.messages",
                    request_id=active_request_id,
                    safe_details={"limit": MAX_MESSAGES},
                )
            retained_attachments = parsed_attachments or saved.request.attachments
            request = request.model_copy(update={"attachments": retained_attachments})
            waiting = saved.status in {"needs_input", "needs_confirmation"}
            state = saved.model_copy(
                update={
                    "request": request,
                    "messages": [*saved.messages, user_message],
                    "status": saved.status if waiting else "running",
                    "active_agent": "supervisor",
                    "supervisor_decision": None,
                    "pending_question": saved.pending_question if waiting else None,
                    "pending_confirmation": (
                        saved.pending_confirmation if waiting else None
                    ),
                    "error": None,
                }
            )
            try:
                state = CaseFileState.model_validate(state.model_dump(mode="python"))
            except ValidationError as exc:
                raise CaseFileError(
                    ErrorCode.SESSION_CORRUPT,
                    "The saved session could not accept the new conversation turn.",
                    stage="runtime.session_resume",
                    request_id=active_request_id,
                    cause=exc,
                ) from exc

        result = self._compiled.invoke(
            state,
            config={"recursion_limit": self.nodes.max_steps + 2},
        )
        try:
            completed = CaseFileState.model_validate(result)
        except ValidationError as exc:
            raise CaseFileError(
                ErrorCode.INTERNAL_ERROR,
                "The agent graph returned invalid persisted state.",
                stage="runtime.graph_output",
                request_id=active_request_id,
                safe_details={"schema": CaseFileState.__name__},
                cause=exc,
            ) from exc
        self.sessions.save(completed)
        return completed


__all__ = ["CaseFileRuntime"]
