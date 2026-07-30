"""FastAPI backend and a deliberately small demo chat UI."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from casefile.agent.graph import CaseFileAgent
from casefile.agent.tools import ToolContext
from casefile.config import get_settings
from casefile.security.audit import RateLimiter


app = FastAPI(title="CaseFile", version="0.1.0")


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


@app.get("/health")
def health() -> dict[str, str]:
    agent = get_agent()
    return {
        "status": "ok",
        "agent_backend": agent.backend,
        "retrieval_backend": agent.tools.index.backend,
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
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CaseFile</title><style>
body{font:16px system-ui;max-width:850px;margin:2rem auto;padding:0 1rem;color:#18212b}
.row{display:flex;gap:.75rem;flex-wrap:wrap}input,select,textarea,button{font:inherit;padding:.65rem}
input{flex:1}textarea{width:100%;box-sizing:border-box;margin-top:1rem}button{margin-top:.5rem;background:#173f5f;color:white;border:0;border-radius:5px}
pre{white-space:pre-wrap;background:#f4f6f8;padding:1rem;border-radius:6px;min-height:5rem}
</style></head><body><h1>CaseFile</h1><p>Citation-preserving PF evidence and coaching.</p>
<div class="row"><select id="role"><option>student</option><option>coach</option></select>
<input id="uid" value="student-1" aria-label="User id"><input id="resolution" value="2026-09-CRYPTO" aria-label="Resolution"></div>
<textarea id="message" rows="4" placeholder="Ask for Pro evidence, a drill, a rule, or progress..."></textarea>
<button id="send">Send</button><pre id="answer"></pre><script>
document.getElementById('send').onclick=async()=>{const answer=document.getElementById('answer');answer.textContent='Working…';
const body={message:document.getElementById('message').value,role:document.getElementById('role').value,user_id:document.getElementById('uid').value,resolution:document.getElementById('resolution').value};
const response=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await response.json();answer.textContent=data.response||JSON.stringify(data,null,2)};
</script></body></html>"""
