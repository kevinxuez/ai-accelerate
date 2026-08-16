from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

import casefile.api.main as api_main
from casefile.agents.contracts import (
    ActiveGoal,
    AgentTraceEntry,
    AttachmentHandle,
    ClarificationRequest,
    ConversationMessage,
    RequestContext,
    TopicPacket,
)
from casefile.agents.errors import ErrorCode, ErrorDetail
from casefile.agents.state import CaseFileState


class StubSessionStore:
    """No prior session on disk: every turn behaves like a fresh session."""

    def load(self, session_id: str, **kwargs: Any) -> None:
        return None


class StubRuntime:
    def __init__(self, settings: Any, response_factory: Any) -> None:
        self.settings = settings
        self.response_factory = response_factory
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sessions = StubSessionStore()

    def ask(self, message: str, **kwargs: Any) -> CaseFileState:
        self.calls.append((message, kwargs))
        return self.response_factory(message, kwargs)


def _request(kwargs: dict[str, Any]) -> RequestContext:
    return RequestContext(
        request_id=kwargs["request_id"],
        session_id=kwargs.get("session_id") or "generated-session-id-0001",
        role=kwargs["role"],
        user_id=kwargs["user_id"],
        active_resolution=kwargs["resolution"],
        attachments=kwargs.get("attachments", []),
    )


def _completed_state(_: str, kwargs: dict[str, Any]) -> CaseFileState:
    topic = TopicPacket(
        event="Public Forum",
        resolution="Resolved: A demo topic.",
        provider="Test provider",
        backend="fixture",
        synthetic=True,
        source_ref="fixture://topic",
    )
    return CaseFileState(
        request=_request(kwargs),
        status="completed",
        active_agent="evidence_librarian",
        active_goal=ActiveGoal(
            summary="Retrieve the current topic.",
            completion_criteria=["Return a TopicPacket."],
        ),
        messages=[
            ConversationMessage(role="user", content="Find the topic."),
            ConversationMessage(role="assistant", content="The topic is ready."),
        ],
        artifacts=[topic],
        agent_trace=[
            AgentTraceEntry(
                sequence=1,
                agent="supervisor",
                event="handoff",
                from_agent="supervisor",
                to_agent="evidence_librarian",
                reason_code="topic_lookup",
                summary="Delegated topic lookup.",
            )
        ],
    )


def _failed_state(_: str, kwargs: dict[str, Any]) -> CaseFileState:
    return CaseFileState(
        request=_request(kwargs),
        status="failed",
        active_agent="skills_coach",
        messages=[ConversationMessage(role="user", content="Read another student.")],
        error=ErrorDetail(
            code=ErrorCode.AUTHORIZATION_DENIED,
            message="Students may read only their own progress.",
            stage="tools.get_progress",
            agent="skills_coach",
            tool="get_progress",
            retryable=False,
            details={},
        ),
        agent_trace=[
            AgentTraceEntry(
                sequence=1,
                agent="skills_coach",
                event="activated",
                summary="Attempted an authorized progress lookup.",
            )
        ],
    )


def _needs_input_state(_: str, kwargs: dict[str, Any]) -> CaseFileState:
    return CaseFileState(
        request=_request(kwargs),
        status="needs_input",
        active_agent="supervisor",
        messages=[
            ConversationMessage(role="user", content="Import this exact request."),
            ConversationMessage(
                role="assistant",
                content="Which side should the Librarian assign to this document?",
            ),
        ],
        pending_question=ClarificationRequest(
            question="Which side should the Librarian assign to this document?",
            missing_fields=["side"],
            reason_code="missing_ingestion_side",
        ),
    )


def test_demo_console_serves_phase_nine_typed_ui() -> None:
    client = TestClient(api_main.app)

    html = client.get("/")
    javascript = client.get("/demo.js")

    assert html.status_code == 200
    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith("application/javascript")
    assert "CaseFile Demo Console" in html.text
    for test_id in ("submitted-prompt", "model-details"):
        assert f'data-testid="{test_id}"' in html.text
    assert "Quick scenarios" not in html.text
    assert "data-scenario=" not in html.text
    assert 'id="trace-agent"' in html.text
    assert 'id="trace-handoffs"' in html.text
    assert 'id="trace-tools"' in html.text
    assert 'id="trace-models"' in html.text
    assert 'byId("prompt-entered").textContent = message' in javascript.text
    assert "switch (artifact.artifact_type)" in javascript.text
    for artifact_type in (
        "evidence_packet",
        "rule_packet",
        "topic_packet",
        "ingestion_preview",
        "ingestion_commit_result",
        "argument_draft",
        "drill_plan",
        "coach_turn",
        "progress_summary",
        "assessment_proposal",
        "calendar_event",
    ):
        assert f'case "{artifact_type}"' in javascript.text
    assert "data." + "intent" not in javascript.text
    assert 'startsWith("[' not in javascript.text
    assert 'fetch("/ingestion/confirm"' in javascript.text
    assert 'fetch("/health/ready"' in javascript.text


def test_health_exposes_required_runtime_dependencies(monkeypatch) -> None:
    ready_calls: list[bool] = []
    index = SimpleNamespace(
        backend="in_memory",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        validate_ready=lambda: ready_calls.append(True),
    )
    runtime = SimpleNamespace(
        backend="langgraph",
        tools=SimpleNamespace(index=index),
        settings=SimpleNamespace(
            model="claude-test",
            calendar_provider="fixture",
            nsda_provider="fixture",
        ),
    )
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    client = TestClient(api_main.app)
    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.json() == {"status": "live"}
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "graph": "langgraph",
        "model": "claude-test",
        "retrieval": "in_memory",
        "retrieval_sources": "cards,rules",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "storage": "ready",
        "calendar": "fixture",
        "nsda": "fixture",
    }
    assert ready_calls == [True]


def test_chat_maps_new_runtime_state_to_typed_success_envelope(
    monkeypatch,
    isolated_settings,
) -> None:
    runtime = StubRuntime(isolated_settings, _completed_state)
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    response = TestClient(api_main.app).post(
        "/chat",
        json={
            "message": "Find the topic.",
            "role": "student",
            "user_id": "student-1",
            "resolution": "R1",
            "session_id": "phase-nine-session-0001",
        },
        headers={"X-Request-ID": "phase-nine-request-1"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "phase-nine-request-1"
    body = response.json()
    assert body["status"] == "completed"
    assert body["response"] == "The topic is ready."
    assert body["request_id"] == "phase-nine-request-1"
    assert body["session_id"] == "phase-nine-session-0001"
    assert body["active_agent"] == "evidence_librarian"
    assert body["active_goal"]["summary"] == "Retrieve the current topic."
    assert body["awaiting_input"] is False
    assert body["awaiting_confirmation"] is False
    assert body["artifacts"][0]["artifact_type"] == "topic_packet"
    assert body["agent_trace"][0]["event"] == "handoff"
    assert runtime.calls[0][0] == "Find the topic."


def test_chat_omits_prior_turns_artifacts_already_seen_by_the_caller(
    monkeypatch,
    isolated_settings,
) -> None:
    """A turn that produces no new artifact must not resurface an old one.

    state.artifacts accumulates for the life of a session so later turns can
    reference earlier work, but the HTTP response for a given turn should
    only report what that turn actually produced -- otherwise an unrelated
    reply (e.g. an authorization denial) appears to carry stale evidence.
    """

    session_id = "phase-nine-session-carryover"
    topic = TopicPacket(
        event="Public Forum",
        resolution="Resolved: A demo topic.",
        provider="Test provider",
        backend="fixture",
        synthetic=True,
        source_ref="fixture://topic",
    )

    class StubSessionStoreWithPriorArtifact:
        def load(self, loaded_session_id: str, **kwargs: Any) -> CaseFileState | None:
            if loaded_session_id != session_id:
                return None
            return CaseFileState(
                request=_request(
                    {
                        "request_id": "prior-turn-request",
                        "session_id": session_id,
                        "role": "student",
                        "user_id": "student-1",
                        "resolution": "R1",
                    }
                ),
                status="completed",
                artifacts=[topic],
            )

    def _denied_turn_carrying_old_artifact(
        _: str, kwargs: dict[str, Any]
    ) -> CaseFileState:
        return CaseFileState(
            request=_request(kwargs),
            status="completed",
            active_agent="skills_coach",
            messages=[
                ConversationMessage(role="user", content="Show another student."),
                ConversationMessage(role="assistant", content="Access denied."),
            ],
            artifacts=[topic],
        )

    runtime = StubRuntime(isolated_settings, _denied_turn_carrying_old_artifact)
    runtime.sessions = StubSessionStoreWithPriorArtifact()
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    response = TestClient(api_main.app).post(
        "/chat",
        json={
            "message": "Show another student.",
            "role": "student",
            "user_id": "student-1",
            "resolution": "R1",
            "session_id": session_id,
        },
        headers={"X-Request-ID": "phase-nine-request-carryover"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Access denied."
    assert body["artifacts"] == []


def test_chat_maps_failed_state_to_non_200_typed_error_with_trace(
    monkeypatch,
    isolated_settings,
) -> None:
    runtime = StubRuntime(isolated_settings, _failed_state)
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    response = TestClient(api_main.app).post(
        "/chat",
        json={
            "message": "Read another student.",
            "role": "student",
            "user_id": "student-1",
            "resolution": "R1",
            "session_id": "phase-nine-session-0002",
        },
        headers={"X-Request-ID": "phase-nine-request-2"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "status": "failed",
        "request_id": "phase-nine-request-2",
        "session_id": "phase-nine-session-0002",
        "error": {
            "code": "AUTHORIZATION_DENIED",
            "message": "Students may read only their own progress.",
            "stage": "tools.get_progress",
            "agent": "skills_coach",
            "tool": "get_progress",
            "retryable": False,
            "details": {},
        },
        "agent_trace": [
            {
                "sequence": 1,
                "agent": "skills_coach",
                "event": "activated",
                "from_agent": None,
                "to_agent": None,
                "reason_code": None,
                "summary": "Attempted an authorized progress lookup.",
            }
        ],
        "tool_trace": [],
        "model_trace": [],
    }


def test_request_validation_error_preserves_session_context() -> None:
    response = TestClient(api_main.app).post(
        "/chat",
        json={
            "message": "",
            "role": "student",
            "user_id": "student-1",
            "resolution": "R1",
            "session_id": "phase-nine-session-0003",
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "REQUEST_INVALID"
    # Body validation occurs before the endpoint binds its validated session.
    assert body["session_id"] is None


def test_attachment_transport_hands_opaque_handle_and_exact_prompt_to_runtime(
    monkeypatch,
    sample_docx,
    isolated_settings,
) -> None:
    runtime = StubRuntime(isolated_settings, _needs_input_state)
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    with sample_docx.open("rb") as stream:
        response = TestClient(api_main.app).post(
            "/chat/with-attachment",
            data={
                "message": "Import this exact request.",
                "role": "student",
                "user_id": "student-1",
                "resolution": "R1",
                "session_id": "phase-nine-session-0004",
            },
            files={
                "attachment": (
                    "uploaded cards.docx",
                    stream,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_input"
    assert body["awaiting_input"] is True
    assert body["awaiting_confirmation"] is False
    assert body["active_agent"] == "supervisor"
    message, call = runtime.calls[0]
    assert message == "Import this exact request."
    assert len(call["attachments"]) == 1
    handle = call["attachments"][0]
    assert isinstance(handle, AttachmentHandle)
    assert handle.filename == "uploaded-cards.docx"
    assert not handle.attachment_id.startswith("/")
    assert (isolated_settings.uploads_dir / handle.attachment_id).is_file()


def test_attachment_upload_rejects_invalid_documents(
    monkeypatch,
    isolated_settings,
) -> None:
    runtime = StubRuntime(isolated_settings, _completed_state)
    monkeypatch.setattr(api_main, "get_runtime", lambda: runtime)

    response = TestClient(api_main.app).post(
        "/chat/with-attachment",
        data={
            "message": "Parse the attached evidence file.",
            "role": "student",
            "user_id": "student-1",
            "resolution": "R1",
        },
        files={"attachment": ("cards.docx", b"not a docx")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "REQUEST_INVALID",
        "message": "The attachment is not a valid Word DOCX document",
        "stage": "api.request",
        "agent": None,
        "tool": None,
        "retryable": False,
        "details": {},
    }
    assert runtime.calls == []
    assert list(isolated_settings.uploads_dir.rglob("*.docx")) == []


def test_only_integration_required_explicit_mutation_routes_remain() -> None:
    routes = {
        (method, route.path)
        for route in api_main.app.routes
        for method in getattr(route, "methods", set()) or set()
    }

    assert ("POST", "/calendar/session/confirm") in routes
    assert ("POST", "/ingestion/confirm") in routes
    assert ("POST", "/ingestion/quarantine/approve") in routes
    assert ("POST", "/ingest/confirm") not in routes
    assert ("POST", "/ingest/approve-quarantined") not in routes
    assert ("POST", "/calendar/session") not in routes
    assert ("GET", "/health") not in routes
