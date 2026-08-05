from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

import casefile.api.nsda as nsda_api
from casefile.agent.graph import CaseFileAgent
from casefile.api.main import app
from casefile.providers.nsda import (
    HTTPNSDAProvider,
    MockNSDAProvider,
    NSDANotFound,
    build_nsda_provider,
)


def test_mock_nsda_provider_is_explicitly_synthetic_and_filterable() -> None:
    provider = MockNSDAProvider()

    metadata = provider.metadata()
    assert metadata["mock"] is True
    assert metadata["synthetic"] is True
    assert "not an official NSDA API" in metadata["disclaimer"]
    assert metadata["counts"] == {
        "topics": 3,
        "rules": 3,
        "tournaments": 3,
        "members": 3,
    }

    topic = provider.current_topic("pf", as_of=date(2026, 8, 4))
    assert topic["id"] == "pf-2026-07-demo"
    assert topic["synthetic"] is True

    rules = provider.search_rules("evidence source", event="public_forum")
    assert [rule["id"] for rule in rules] == ["pf-mock-2.1"]
    assert rules[0]["score"] > 0

    tournaments = provider.list_tournaments(state="ca", event="PF")
    assert [item["id"] for item in tournaments] == ["nsda-mock-ca-001"]

    assert provider.get_member("student-1")["status"] == "active"
    with pytest.raises(NSDANotFound):
        provider.get_member("missing-member")


def test_provider_builder_honors_configured_mock_data(
    tmp_path, isolated_settings
) -> None:
    payload = MockNSDAProvider().dataset.model_dump(mode="json")
    payload["dataset_version"] = "custom-fixture-test"
    custom_fixture = tmp_path / "custom-nsda.json"
    custom_fixture.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    settings = replace(
        isolated_settings,
        nsda_base_url=None,
        nsda_mock_data=custom_fixture,
    )

    provider = build_nsda_provider(settings)

    assert isinstance(provider, MockNSDAProvider)
    assert provider.metadata()["dataset_version"] == "custom-fixture-test"


def test_mock_nsda_api_returns_versioned_envelopes_and_404s() -> None:
    client = TestClient(app)

    metadata = client.get("/mock/nsda/v1/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["mock"] is True
    assert metadata.json()["synthetic"] is True
    assert metadata.json()["dataset_version"] == "2026.08-demo.1"

    topic = client.get(
        "/mock/nsda/v1/topics/current",
        params={"event": "pf", "as_of": "2026-08-04"},
    )
    assert topic.status_code == 200
    assert topic.json()["data"]["id"] == "pf-2026-07-demo"

    rules = client.get(
        "/mock/nsda/v1/rules/search",
        params={"q": "evidence source", "event": "Public Forum"},
    )
    assert rules.status_code == 200
    assert rules.json()["data"][0]["id"] == "pf-mock-2.1"

    tournaments = client.get(
        "/mock/nsda/v1/tournaments",
        params={"state": "CA", "event": "pf"},
    )
    assert tournaments.status_code == 200
    assert tournaments.json()["data"][0]["state"] == "CA"

    missing = client.get("/mock/nsda/v1/members/does-not-exist")
    assert missing.status_code == 404
    assert "Synthetic NSDA member was not found" in missing.json()["detail"]


def test_configured_nsda_facade_uses_bundled_mock_by_default() -> None:
    response = TestClient(app).get(
        "/nsda/v1/topics/current",
        params={"event": "pf", "as_of": "2026-08-04"},
    )

    assert response.status_code == 200
    assert response.json()["backend"] == "mock"
    assert response.json()["data"]["id"] == "pf-2026-07-demo"


def test_chat_agent_uses_configured_nsda_provider_for_current_topic(
    isolated_settings,
) -> None:
    settings = replace(isolated_settings, nsda_base_url=None)
    agent = CaseFileAgent(settings)

    result = agent.ask(
        "What is the current Public Forum topic for 2026-08-04?",
        role="student",
        user_id="student-1",
        resolution="2026-09-CRYPTO",
    )

    assert result["intent"] == "current_topic"
    assert result["tool_trace"][0]["tool"] == "current_topic"
    assert result["task_trace"][0]["status"] == "success"
    assert "four-day instructional week" in result["response"]
    assert "Synthetic NSDA-compatible fixture" in result["response"]
    assert "not an official NSDA" in result["response"]


def test_configured_nsda_facade_calls_selected_provider(monkeypatch) -> None:
    class FakeHTTPProvider:
        backend = "http"

        def metadata(self):
            return {
                "provider": "National Speech & Debate Association",
                "provider_code": "nsda",
                "backend": "mock",
                "dataset_version": "remote-test",
                "generated_at": "2026-08-04T00:00:00Z",
                "mock": True,
                "synthetic": True,
                "disclaimer": "Synthetic remote fixture.",
                "counts": {
                    "topics": 1,
                    "rules": 0,
                    "tournaments": 0,
                    "members": 0,
                },
            }

        def current_topic(self, event="Public Forum", *, as_of=None):
            return {
                "id": "remote-topic",
                "event": event,
                "as_of": as_of.isoformat() if as_of else None,
                "synthetic": True,
            }

    monkeypatch.setattr(
        nsda_api,
        "get_configured_nsda_provider",
        lambda: FakeHTTPProvider(),
    )

    response = TestClient(app).get(
        "/nsda/v1/topics/current",
        params={"event": "pf", "as_of": "2026-08-04"},
    )

    assert response.status_code == 200
    assert response.json()["backend"] == "http"
    assert response.json()["dataset_version"] == "remote-test"
    assert response.json()["data"] == {
        "id": "remote-topic",
        "event": "pf",
        "as_of": "2026-08-04",
        "synthetic": True,
    }


def test_http_nsda_adapter_accepts_https_and_unwraps_mock_envelope() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "provider": "National Speech & Debate Association",
                "provider_code": "nsda",
                "mock": True,
                "synthetic": True,
                "dataset_version": "test",
                "disclaimer": "synthetic",
                "data": {
                    "id": "pf-test",
                    "event": "Public Forum",
                    "division": "Open",
                    "season": "2026-2027",
                    "release_window": "July/August 2026",
                    "resolution": "Resolved: Synthetic integration topic.",
                    "effective_from": "2026-07-01",
                    "effective_to": "2026-08-31",
                    "current": True,
                    "source_ref": "mock://nsda/topics/pf-test",
                    "synthetic": True,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HTTPNSDAProvider(
        "https://nsda-mock.example/v1",
        api_key="test-token",
        client=client,
    )

    result = provider.current_topic("pf", as_of=date(2026, 8, 4))

    assert result["id"] == "pf-test"
    assert result["synthetic"] is True
    assert observed[0].url.path == "/v1/topics/current"
    assert observed[0].url.params["event"] == "pf"
    assert observed[0].headers["authorization"] == "Bearer test-token"


@pytest.mark.parametrize(
    "url",
    [
        "http://nsda-mock.example/v1",
        "ftp://nsda-mock.example/v1",
        "https://user:password@nsda-mock.example/v1",
        "https://nsda-mock.example/v1?token=secret",
    ],
)
def test_http_nsda_adapter_rejects_unsafe_base_urls(url: str) -> None:
    with pytest.raises(ValueError):
        HTTPNSDAProvider(url)


def test_http_nsda_adapter_allows_local_http_for_development() -> None:
    provider = HTTPNSDAProvider(
        "http://127.0.0.1:8000/mock/nsda/v1",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )
    assert provider.base_url.startswith("http://127.0.0.1")
