from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import casefile.api.main as api_main
from casefile.agent.graph import CaseFileAgent


def test_demo_console_is_served_with_safe_testing_controls() -> None:
    response = TestClient(api_main.app).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "CaseFile Demo Console" in response.text
    assert 'data-testid="scenario-evidence"' in response.text
    assert 'data-testid="scenario-topic"' in response.text
    assert 'data-testid="scenario-coaching"' in response.text
    assert 'data-testid="role-select"' not in response.text
    assert "Coach access" not in response.text
    assert 'data-testid="send-button"' in response.text
    assert 'data-testid="attachment-input"' in response.text
    assert 'data-testid="submitted-prompt"' in response.text
    assert 'data-testid="prompt-entered"' in response.text
    assert 'byId("prompt-entered").textContent = message' in response.text
    assert "/chat/with-attachment" in response.text
    assert "Confirm evidence import" in response.text
    assert "Agent trace" in response.text
    assert "New session" in response.text
    assert 'id="trace-session"' in response.text
    assert 'payload.append("session_id", sessionId)' in response.text
    assert "Developer status" in response.text
    assert "renderGroundingCard" in response.text
    assert "source-mark emphasis" not in response.text
    assert ".source-mark.emphasis" in response.text
    assert "AI-generated · grounded in cards below" in response.text
    assert "Demo console" not in response.text
    assert 'id="status-calendar" data-testid="status-calendar" hidden' in response.text
    assert "ANTHROPIC_API_KEY=" not in response.text


def test_health_exposes_safe_demo_backend_status(monkeypatch) -> None:
    fake_agent = SimpleNamespace(
        backend="langgraph",
        tools=SimpleNamespace(index=SimpleNamespace(backend="chroma")),
    )
    fake_settings = SimpleNamespace(anthropic_api_key="configured", mock_calendar=True)
    monkeypatch.setattr(api_main, "get_agent", lambda: fake_agent)
    monkeypatch.setattr(api_main, "get_settings", lambda: fake_settings)

    response = TestClient(api_main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "agent_backend": "langgraph",
        "retrieval_backend": "chroma",
        "model_status": "configured",
        "calendar_backend": "mock",
        "nsda_backend": "mock",
    }


def test_student_can_upload_preview_and_confirm_attached_docx(
    monkeypatch,
    sample_docx,
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    monkeypatch.setattr(api_main, "get_agent", lambda: agent)
    monkeypatch.setattr(api_main, "_rate_limit", lambda route, user_id: None)
    client = TestClient(api_main.app)

    with sample_docx.open("rb") as stream:
        response = client.post(
            "/chat/with-attachment",
            data={
                "message": "Import and parse the attached DOCX evidence file.",
                "role": "student",
                "user_id": "student-1",
                "resolution": "2026-09-CRYPTO",
                "side": "pro",
                "use_model": "false",
            },
            files={
                "attachment": (
                    "uploaded-cards.docx",
                    stream,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert response.status_code == 200
    preview = response.json()
    assert preview["intent"] == "ingest_cards"
    assert preview["attachment"]["name"] == "uploaded-cards.docx"
    assert preview["ingest_preview"]["source_file"] == "uploaded-cards.docx"
    assert preview["ingest_preview"]["counts"]["ok"] >= 1
    assert preview["tool_trace"][-1]["tool"] == "ingest_cards"
    assert preview["awaiting_clarification"] is False
    assert preview["session_id"]
    staged = list(isolated_settings.uploads_dir.rglob("*.docx"))
    assert len(staged) == 1

    confirmed = client.post(
        "/ingest/confirm",
        json={
            "confirmation_token": preview["ingest_preview"]["token"],
            "role": "student",
            "user_id": "student-1",
            "resolution": "2026-09-CRYPTO",
            "idempotency_key": "upload-confirm-1",
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["written"] >= 1
    assert list(isolated_settings.uploads_dir.rglob("*.docx")) == []


def test_attachment_upload_rejects_invalid_documents(
    monkeypatch,
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    monkeypatch.setattr(api_main, "get_agent", lambda: agent)
    monkeypatch.setattr(api_main, "_rate_limit", lambda route, user_id: None)
    client = TestClient(api_main.app)
    common = {
        "message": "Parse the attached evidence file.",
        "user_id": "student-1",
        "resolution": "R1",
        "side": "pro",
        "use_model": "false",
    }

    invalid = client.post(
        "/chat/with-attachment",
        data={**common, "role": "student"},
        files={"attachment": ("cards.docx", b"not a docx")},
    )
    assert invalid.status_code == 400
    assert "not a valid Word DOCX" in invalid.json()["detail"]
    assert list(isolated_settings.uploads_dir.rglob("*.docx")) == []


def test_chat_api_resumes_a_pending_clarification(
    monkeypatch,
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    monkeypatch.setattr(api_main, "get_agent", lambda: agent)
    monkeypatch.setattr(api_main, "_rate_limit", lambda route, user_id: None)
    client = TestClient(api_main.app)
    context = {
        "role": "student",
        "user_id": "student-1",
        "resolution": "R1",
        "session_id": "api-session-clarify-01",
    }

    first = client.post("/chat", json={**context, "message": "Build a drill."})
    assert first.status_code == 200
    assert first.json()["awaiting_clarification"] is True

    completed = client.post(
        "/chat",
        json={**context, "message": "Pro"},
    )
    assert completed.status_code == 200
    assert completed.json()["intent"] == "generate_drill"
    assert completed.json()["resumed_from_clarification"] is True
    assert completed.json()["awaiting_clarification"] is False
