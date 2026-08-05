"""FastAPI backend and responsive local demo console."""

from __future__ import annotations

import io
import re
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from casefile.agent.graph import CaseFileAgent
from casefile.agent.tools import ToolContext
from casefile.api.nsda import provider_router as nsda_provider_router
from casefile.api.nsda import router as nsda_mock_router
from casefile.config import get_settings
from casefile.security.audit import RateLimiter


app = FastAPI(title="CaseFile", version="0.1.0")
app.include_router(nsda_mock_router)
app.include_router(nsda_provider_router)
DEMO_HTML = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
MAX_DOCX_FILES = 5_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


class StrictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ChatRequest(StrictRequest):
    message: str = Field(min_length=1, max_length=20_000)
    role: Literal["student", "coach"]
    user_id: str = Field(min_length=1, max_length=100)
    resolution: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{15,127}$",
    )


class IngestPreviewRequest(StrictRequest):
    file_path: str = Field(min_length=1, max_length=2000)
    resolution: str = Field(min_length=1, max_length=200)
    side: Literal["pro", "con", "unknown"] | None = None
    role: Literal["student", "coach"]
    user_id: str = Field(min_length=1, max_length=100)
    use_model: bool = True


class IngestConfirmRequest(StrictRequest):
    confirmation_token: str = Field(pattern=r"^[0-9a-f]{32}$")
    role: Literal["student", "coach"]
    user_id: str = Field(min_length=1, max_length=100)
    resolution: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class ApproveCardRequest(StrictRequest):
    card_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["student", "coach"]
    user_id: str = Field(min_length=1, max_length=100)
    resolution: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class ScheduleRequest(StrictRequest):
    student_id: str = Field(min_length=1, max_length=100)
    start: str = Field(default="", max_length=100)
    duration_minutes: int = Field(default=45, ge=15, le=180)
    attendee_email: str | None = Field(default=None, max_length=320)
    timezone_name: str = Field(default="America/Chicago", min_length=1, max_length=100)
    confirmation_token: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    role: Literal["student", "coach"]
    user_id: str = Field(min_length=1, max_length=100)
    resolution: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


@lru_cache(maxsize=1)
def get_agent() -> CaseFileAgent:
    return CaseFileAgent()


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(max(1, get_settings().requests_per_minute))


def _rate_limit(route: str, user_id: str) -> None:
    if not get_rate_limiter().allow(f"{route}:{user_id}"):
        raise HTTPException(status_code=429, detail="Request rate limit exceeded")


def _safe_docx_name(filename: str | None) -> str:
    original = Path(filename or "").name
    if Path(original).suffix.lower() != ".docx":
        raise HTTPException(status_code=400, detail="Only .docx attachments are supported")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original).stem).strip(".-_")
    return f"{(stem[:120] or 'evidence')}.docx"


def _validate_docx_payload(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_FILES or sum(
                member.file_size for member in members
            ) > MAX_DOCX_UNCOMPRESSED_BYTES:
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


@app.get("/health")
def health() -> dict[str, str]:
    agent = get_agent()
    settings = get_settings()
    return {
        "status": "ok",
        "agent_backend": agent.backend,
        "retrieval_backend": agent.tools.index.backend,
        "model_status": "configured" if settings.anthropic_api_key else "offline",
        "calendar_backend": "mock" if settings.mock_calendar else "google",
        "nsda_backend": "http" if getattr(settings, "nsda_base_url", None) else "mock",
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    _rate_limit("chat", request.user_id)
    try:
        return get_agent().ask(
            request.message,
            role=request.role,
            user_id=request.user_id,
            resolution=request.resolution,
            request_id=request.idempotency_key,
            session_id=request.session_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat/with-attachment")
async def chat_with_attachment(
    message: str = Form(...),
    role: Literal["student", "coach"] = Form(...),
    user_id: str = Form(...),
    resolution: str = Form(...),
    attachment: UploadFile = File(...),
    side: Literal["pro", "con", "unknown"] = Form("unknown"),
    use_model: bool = Form(True),
    idempotency_key: str | None = Form(None),
    session_id: str | None = Form(None),
) -> dict:
    """Screen a request, then stage and parse its attached DOCX for confirmation."""

    try:
        request = ChatRequest(
            message=message,
            role=role,
            user_id=user_id,
            resolution=resolution,
            idempotency_key=idempotency_key,
            session_id=session_id,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        ) from exc
    _rate_limit("chat-attachment", request.user_id)
    agent = get_agent()
    routed = agent.ask(
        request.message,
        role=request.role,
        user_id=request.user_id,
        resolution=request.resolution,
        request_id=request.idempotency_key,
        session_id=request.session_id,
    )
    if routed.get("security_decision", {}).get("action") == "block":
        return routed
    if routed.get("intent") == "integrity_refusal":
        return routed
    if routed.get("intent") not in {"ingest_cards", "unknown"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "The prompt does not request evidence import. Remove the attachment or "
                "ask CaseFile to import or parse it."
            ),
        )

    safe_name = _safe_docx_name(attachment.filename)
    settings = agent.settings
    try:
        payload = await attachment.read(settings.max_upload_bytes + 1)
    finally:
        await attachment.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The attached DOCX is empty")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Attachment exceeds the {settings.max_upload_bytes}-byte limit",
        )
    _validate_docx_payload(payload)

    upload_directory = settings.uploads_dir / uuid.uuid4().hex
    upload_directory.mkdir(parents=True, exist_ok=False)
    upload_path = upload_directory / safe_name
    upload_path.write_bytes(payload)
    try:
        result = agent.tools.ingest_cards(
            ToolContext(request.role, request.user_id, request.resolution),
            file_path=str(upload_path),
            resolution=request.resolution,
            side=side if side in {"pro", "con"} else None,
            dry_run=True,
            use_model=use_model,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        _remove_upload(upload_path)
        detail = str(exc).replace(str(upload_path), safe_name)
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception:
        _remove_upload(upload_path)
        raise
    if isinstance(result, str):
        _remove_upload(upload_path)
        agent.clear_session(routed["session_id"])
        return {
            **routed,
            "intent": "ingest_cards",
            "response": result,
            "awaiting_clarification": False,
        }

    public_result = {**result, "source_file": safe_name}
    if not public_result.get("validation", {}).get("valid"):
        (settings.pending_dir / f"{public_result['token']}.json").unlink(
            missing_ok=True
        )
        _remove_upload(upload_path)

    tool_trace = [
        *routed.get("tool_trace", []),
        {
            "tool": "ingest_cards",
            "arguments": {
                "attachment": safe_name,
                "size_bytes": len(payload),
                "resolution": request.resolution,
                "side": side,
                "dry_run": True,
            },
            "result_type": "dict",
            "status": "success",
            "attempts": 1,
            "depends_on": [],
        },
    ]
    agent.clear_session(routed["session_id"])
    return {
        **routed,
        "intent": "ingest_cards",
        "response": public_result["summary"],
        "awaiting_clarification": False,
        "iterations": 1,
        "tool_trace": tool_trace,
        "task_trace": [
            {
                "id": "ingest",
                "action": "ingest_cards",
                "status": "success",
                "attempts": 1,
                "depends_on": [],
            }
        ],
        "attachment": {
            "name": safe_name,
            "size_bytes": len(payload),
        },
        "ingest_preview": public_result,
    }


@app.post("/ingest/preview")
def ingest_preview(request: IngestPreviewRequest) -> dict | str:
    _rate_limit("ingest-preview", request.user_id)
    context = ToolContext(request.role, request.user_id, request.resolution)
    try:
        return get_agent().tools.ingest_cards(
            context,
            file_path=request.file_path,
            resolution=request.resolution,
            side=request.side,
            dry_run=True,
            use_model=request.use_model,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/confirm")
def ingest_confirm(request: IngestConfirmRequest) -> dict | str:
    _rate_limit("ingest-confirm", request.user_id)
    context = ToolContext(request.role, request.user_id, request.resolution)
    try:
        return get_agent().tools.ingest_cards(
            context,
            confirmation_token=request.confirmation_token,
            dry_run=False,
            idempotency_key=request.idempotency_key,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/approve-quarantined")
def approve_quarantined(request: ApproveCardRequest) -> dict | str:
    _rate_limit("ingest-approve", request.user_id)
    context = ToolContext(request.role, request.user_id, request.resolution)
    try:
        return get_agent().tools.approve_quarantined_card(
            context,
            card_id=request.card_id,
            idempotency_key=request.idempotency_key,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/calendar/session")
def schedule_session(request: ScheduleRequest) -> dict | str:
    _rate_limit("calendar", request.user_id)
    context = ToolContext(request.role, request.user_id, request.resolution)
    try:
        return get_agent().tools.schedule_session(
            context,
            student_id=request.student_id,
            start=request.start,
            duration_minutes=request.duration_minutes,
            attendee_email=request.attendee_email,
            timezone_name=request.timezone_name,
            confirmation_token=request.confirmation_token,
            idempotency_key=request.idempotency_key,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return DEMO_HTML
