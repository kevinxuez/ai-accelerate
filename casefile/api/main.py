"""FastAPI boundary and local four-agent demonstration console."""

from __future__ import annotations

import hashlib
import io
import re
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import ValidationError

from casefile.agents.contracts import AttachmentHandle
from casefile.agents.errors import CaseFileError, ErrorCode, HTTP_STATUS_BY_CODE
from casefile.agents.runtime import CaseFileRuntime
from casefile.agents.state import CaseFileState
from casefile.api.contracts import (
    CalendarConfirmationRequest,
    ChatRequest,
    ChatSuccessResponse,
    ErrorResponse,
    IngestionConfirmRequest,
    QuarantineApprovalRequest,
)
from casefile.api.errors import install_error_handlers, request_id_for
from casefile.api.nsda import provider_router as nsda_provider_router
from casefile.config import Settings, get_settings
from casefile.security.audit import RateLimiter
from casefile.tools import ToolContext


app = FastAPI(title="CaseFile", version="0.1.0")
install_error_handlers(app)
app.include_router(nsda_provider_router)
DEMO_HTML = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
DEMO_JS = Path(__file__).with_name("demo.js").read_text(encoding="utf-8")
MAX_DOCX_FILES = 5_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def _attachment_path(settings: Settings, handle: AttachmentHandle) -> str:
    path = (settings.uploads_dir / handle.attachment_id).resolve()
    if not path.is_relative_to(settings.uploads_dir.resolve()):
        raise CaseFileError(
            ErrorCode.AUTHORIZATION_DENIED,
            "The attachment handle is outside the configured upload directory.",
            stage="api.attachment.resolve",
        )
    return str(path)


@lru_cache(maxsize=1)
def get_runtime() -> CaseFileRuntime:
    settings = get_settings()
    return CaseFileRuntime(
        settings,
        attachment_resolver=lambda handle: _attachment_path(settings, handle),
    )


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(max(1, get_settings().requests_per_minute))


def _rate_limit(route: str, user_id: str) -> None:
    if not get_rate_limiter().allow(f"{route}:{user_id}"):
        raise CaseFileError(
            ErrorCode.RATE_LIMITED,
            "The request rate limit was exceeded.",
            stage="api.rate_limit",
        )


def _safe_docx_name(filename: str | None) -> str:
    original = Path(filename or "").name
    if Path(original).suffix.lower() != ".docx":
        raise HTTPException(
            status_code=400,
            detail="Only .docx attachments are supported",
        )
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original).stem).strip(".-_")
    return f"{(stem[:120] or 'evidence')}.docx"


def _validate_docx_payload(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (
                len(members) > MAX_DOCX_FILES
                or sum(member.file_size for member in members)
                > MAX_DOCX_UNCOMPRESSED_BYTES
            ):
                raise HTTPException(
                    status_code=413,
                    detail="The DOCX expands beyond the safe parsing limit",
                )
            if "word/document.xml" not in {member.filename for member in members}:
                raise HTTPException(
                    status_code=400,
                    detail="The attachment is not a valid Word DOCX document",
                )
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="The attachment is not a valid Word DOCX document",
        ) from exc


def _remove_upload(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _latest_assistant_response(state: CaseFileState) -> str:
    for message in reversed(state.messages):
        if message.role == "assistant":
            return message.content
    raise CaseFileError(
        ErrorCode.INTERNAL_ERROR,
        "The completed agent state did not contain a response.",
        stage="api.response_mapping",
        request_id=state.request.request_id,
    )


def _state_response(state: CaseFileState) -> JSONResponse:
    headers = {"X-Request-ID": state.request.request_id}
    if state.status == "failed":
        if state.error is None:
            raise CaseFileError(
                ErrorCode.INTERNAL_ERROR,
                "The failed agent state did not contain an error.",
                stage="api.response_mapping",
                request_id=state.request.request_id,
            )
        envelope = ErrorResponse(
            request_id=state.request.request_id,
            session_id=state.request.session_id,
            error=state.error,
            agent_trace=state.agent_trace,
            tool_trace=state.tool_trace,
            model_trace=state.model_trace,
        )
        return JSONResponse(
            status_code=HTTP_STATUS_BY_CODE[state.error.code],
            content=envelope.model_dump(mode="json"),
            headers=headers,
        )
    success = ChatSuccessResponse(
        status=state.status,
        response=_latest_assistant_response(state),
        request_id=state.request.request_id,
        session_id=state.request.session_id,
        active_agent=state.active_agent,
        active_goal=state.active_goal,
        awaiting_input=state.status == "needs_input",
        awaiting_confirmation=state.status == "needs_confirmation",
        artifacts=state.artifacts,
        agent_trace=state.agent_trace,
        tool_trace=state.tool_trace,
        model_trace=state.model_trace,
    )
    return JSONResponse(
        status_code=200,
        content=success.model_dump(mode="json"),
        headers=headers,
    )


def _bind_request_context(
    request: Request,
    *,
    session_id: str | None,
) -> str:
    active_request_id = request_id_for(request)
    request.state.request_id = active_request_id
    request.state.session_id = session_id
    return active_request_id


@app.get("/health/live")
def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    runtime = get_runtime()
    settings = runtime.settings
    runtime.tools.index.validate_ready()
    return {
        "status": "ready",
        "graph": runtime.backend,
        "model": settings.model,
        "retrieval": runtime.tools.index.backend,
        "retrieval_collections": "cards,rules",
        "embedding_model": runtime.tools.index.embedding_model,
        "storage": "ready",
        "calendar": settings.calendar_provider,
        "nsda": settings.nsda_provider,
    }


@app.post("/chat")
def chat(request: ChatRequest, http_request: Request) -> JSONResponse:
    request_id = _bind_request_context(
        http_request,
        session_id=request.session_id,
    )
    state = get_runtime().ask(
        request.message,
        role=request.role,
        user_id=request.user_id,
        resolution=request.resolution,
        request_id=request_id,
        session_id=request.session_id,
    )
    http_request.state.session_id = state.request.session_id
    return _state_response(state)


@app.post("/chat/with-attachment")
async def chat_with_attachment(
    http_request: Request,
    message: str = Form(...),
    role: Literal["student", "coach"] = Form(...),
    user_id: str = Form(...),
    resolution: str = Form(...),
    attachment: UploadFile = File(...),
    session_id: str | None = Form(None),
) -> JSONResponse:
    """Validate transport safety, then hand the attachment to the Supervisor."""

    try:
        request = ChatRequest(
            message=message,
            role=role,
            user_id=user_id,
            resolution=resolution,
            session_id=session_id,
        )
    except ValidationError as exc:
        raise CaseFileError(
            ErrorCode.REQUEST_INVALID,
            "The multipart request did not match the required schema.",
            stage="api.request_validation",
            safe_details={"schema": ChatRequest.__name__},
            cause=exc,
            http_status=422,
        ) from exc
    request_id = _bind_request_context(
        http_request,
        session_id=request.session_id,
    )
    runtime = get_runtime()
    safe_name = _safe_docx_name(attachment.filename)
    try:
        payload = await attachment.read(runtime.settings.max_upload_bytes + 1)
    finally:
        await attachment.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The attached DOCX is empty")
    if len(payload) > runtime.settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Attachment exceeds the {runtime.settings.max_upload_bytes}-byte limit"
            ),
        )
    _validate_docx_payload(payload)

    upload_id = uuid.uuid4().hex
    upload_directory = runtime.settings.uploads_dir / upload_id
    upload_directory.mkdir(parents=True, exist_ok=False)
    upload_path = upload_directory / safe_name
    upload_path.write_bytes(payload)
    handle = AttachmentHandle(
        attachment_id=f"{upload_id}/{safe_name}",
        filename=safe_name,
        media_type=(
            attachment.content_type
            or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    retain_upload = False
    try:
        state = runtime.ask(
            request.message,
            role=request.role,
            user_id=request.user_id,
            resolution=request.resolution,
            request_id=request_id,
            session_id=request.session_id,
            attachments=[handle],
        )
        http_request.state.session_id = state.request.session_id
        retain_upload = state.status in {"needs_input", "needs_confirmation"} or any(
            artifact.artifact_type in {"ingestion_preview", "ingestion_commit_result"}
            for artifact in state.artifacts
        )
        return _state_response(state)
    finally:
        if not retain_upload:
            _remove_upload(upload_path)


@app.post("/ingestion/confirm")
def ingestion_confirm(
    request: IngestionConfirmRequest,
    http_request: Request,
) -> JSONResponse:
    _rate_limit("ingestion-confirm", request.user_id)
    request_id = _bind_request_context(http_request, session_id=None)
    runtime = get_runtime()
    context = ToolContext(
        role=request.role,
        user_id=request.user_id,
        resolution=request.resolution,
        request_id=request_id,
        agent="evidence_librarian",
    )
    result = runtime.evidence_librarian.commit_ingestion(
        context,
        confirmation_token=request.confirmation_token,
        idempotency_key=request.idempotency_key,
    )
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


@app.post("/ingestion/quarantine/approve")
def approve_quarantined(
    request: QuarantineApprovalRequest,
    http_request: Request,
) -> JSONResponse:
    _rate_limit("ingestion-quarantine-approve", request.user_id)
    request_id = _bind_request_context(http_request, session_id=None)
    context = ToolContext(
        role=request.role,
        user_id=request.user_id,
        resolution=request.resolution,
        request_id=request_id,
        agent="evidence_librarian",
    )
    result = get_runtime().tools.ingestion_tools.approve_quarantined_card(
        context,
        card_id=request.card_id,
        idempotency_key=request.idempotency_key,
    )
    return JSONResponse(
        status_code=200,
        content=result,
        headers={"X-Request-ID": request_id},
    )


@app.post("/calendar/session/confirm")
def calendar_confirm(
    request: CalendarConfirmationRequest,
    http_request: Request,
) -> JSONResponse:
    _rate_limit("calendar-confirm", request.user_id)
    request_id = _bind_request_context(http_request, session_id=None)
    runtime = get_runtime()
    context = ToolContext(
        role=request.role,
        user_id=request.user_id,
        resolution=request.resolution,
        request_id=request_id,
        agent="supervisor",
    )
    result = runtime.tools.registry.invoke(
        "supervisor",
        "schedule_session",
        context,
        {
            "student_id": request.user_id,
            "start": "1970-01-01T00:00:00+00:00",
            "duration_minutes": 45,
            "attendee_email": None,
            "timezone_name": "UTC",
            "confirmation_token": request.confirmation_token,
            "idempotency_key": request.idempotency_key,
        },
    )
    return JSONResponse(
        status_code=200,
        content=result,
        headers={"X-Request-ID": request_id},
    )


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return DEMO_HTML


@app.get("/demo.js", response_class=Response)
def demo_javascript() -> Response:
    return Response(DEMO_JS, media_type="application/javascript")
