"""FastAPI exception mapping for CaseFile's typed error envelope."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from casefile.agents.errors import CaseFileError, ErrorCode


LOGGER = logging.getLogger(__name__)
REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def request_id_for(request: Request) -> str:
    current = getattr(request.state, "request_id", None)
    if isinstance(current, str) and REQUEST_ID.fullmatch(current):
        return current
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    return request_id


def _response(request: Request, error: CaseFileError) -> JSONResponse:
    request_id = request_id_for(request)
    payload = error.public_envelope(
        request_id=request_id,
        session_id=getattr(request.state, "session_id", None),
    )
    return JSONResponse(
        status_code=error.http_status,
        content=payload,
        headers={"X-Request-ID": request_id},
    )


async def casefile_error_handler(
    request: Request, error: CaseFileError
) -> JSONResponse:
    LOGGER.warning(
        "CaseFile request failed request_id=%s code=%s stage=%s agent=%s tool=%s cause_type=%s",
        request_id_for(request),
        error.code,
        error.stage,
        error.agent,
        error.tool,
        error.cause_type,
    )
    return _response(request, error)


async def validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": ".".join(str(part) for part in item.get("loc", ())),
            "message": item.get("msg", "Invalid value"),
            "type": item.get("type", "validation_error"),
        }
        for item in error.errors()
    ]
    typed = CaseFileError(
        ErrorCode.REQUEST_INVALID,
        "The request did not match the required schema.",
        stage="api.request_validation",
        request_id=request_id_for(request),
        safe_details={"errors": details},
        http_status=422,
    )
    return _response(request, typed)


async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
    code = {
        403: ErrorCode.AUTHORIZATION_DENIED,
        429: ErrorCode.RATE_LIMITED,
        503: ErrorCode.CAPABILITY_DISABLED,
    }.get(error.status_code, ErrorCode.REQUEST_INVALID)
    message = error.detail if isinstance(error.detail, str) else "The request failed."
    safe_details: dict[str, Any] = {}
    if not isinstance(error.detail, str):
        safe_details["errors"] = error.detail
    typed = CaseFileError(
        code,
        message,
        stage="api.request",
        request_id=request_id_for(request),
        safe_details=safe_details,
        http_status=error.status_code,
    )
    return _response(request, typed)


async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    request_id = request_id_for(request)
    LOGGER.exception(
        "Unhandled CaseFile request failure request_id=%s cause_type=%s",
        request_id,
        type(error).__name__,
    )
    typed = CaseFileError(
        ErrorCode.INTERNAL_ERROR,
        "CaseFile could not complete the request.",
        stage="api.unhandled",
        request_id=request_id,
        cause=error,
    )
    return _response(request, typed)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(CaseFileError, casefile_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
