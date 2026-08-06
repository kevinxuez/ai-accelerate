from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

import casefile.api.nsda as nsda_api
from casefile.api.main import app
from casefile.providers.nsda import (
    FixtureNSDAProvider,
    HTTPNSDAProvider,
    NSDANotFound,
    NSDAProviderDisabled,
    build_nsda_provider,
)


def test_fixture_nsda_provider_is_explicitly_synthetic_and_filterable() -> None:
    provider = FixtureNSDAProvider()

    metadata = provider.metadata()
    assert metadata["fixture"] is True
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
    assert [rule["id"] for rule in rules] == ["pf-fixture-2.1"]
    assert rules[0]["score"] > 0

    tournaments = provider.list_tournaments(state="ca", event="PF")
    assert [item["id"] for item in tournaments] == ["nsda-fixture-ca-001"]

    assert provider.get_member("student-1")["status"] == "active"
    with pytest.raises(NSDANotFound):
        provider.get_member("missing-member")


def test_provider_builder_honors_explicit_fixture_configuration(
    tmp_path, isolated_settings
) -> None:
    payload = FixtureNSDAProvider().dataset.model_dump(mode="json")
    payload["dataset_version"] = "custom-fixture-test"
    custom_fixture = tmp_path / "custom-nsda.json"
    custom_fixture.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    settings = replace(
        isolated_settings,
        nsda_provider="fixture",
        nsda_base_url=None,
        nsda_fixture_path=custom_fixture,
    )

    provider = build_nsda_provider(settings)

    assert isinstance(provider, FixtureNSDAProvider)
    assert provider.metadata()["dataset_version"] == "custom-fixture-test"


def test_provider_builder_does_not_select_a_fixture_when_disabled(
    isolated_settings,
) -> None:
    settings = replace(
        isolated_settings,
        nsda_provider="disabled",
        nsda_base_url=None,
    )

    with pytest.raises(NSDAProviderDisabled):
        build_nsda_provider(settings)


def test_configured_fixture_api_returns_versioned_envelopes_and_404s(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        nsda_api,
        "get_configured_nsda_provider",
        lambda: FixtureNSDAProvider(),
    )
    client = TestClient(app)

    metadata = client.get("/nsda/v1/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["fixture"] is True
    assert metadata.json()["synthetic"] is True
    assert metadata.json()["dataset_version"] == "2026.08-demo.1"

    topic = client.get(
        "/nsda/v1/topics/current",
        params={"event": "pf", "as_of": "2026-08-04"},
    )
    assert topic.status_code == 200
    assert topic.json()["data"]["id"] == "pf-2026-07-demo"

    rules = client.get(
        "/nsda/v1/rules/search",
        params={"q": "evidence source", "event": "Public Forum"},
    )
    assert rules.status_code == 200
    assert rules.json()["data"][0]["id"] == "pf-fixture-2.1"

    tournaments = client.get(
        "/nsda/v1/tournaments",
        params={"state": "CA", "event": "pf"},
    )
    assert tournaments.status_code == 200
    assert tournaments.json()["data"][0]["state"] == "CA"

    missing = client.get("/nsda/v1/members/does-not-exist")
    assert missing.status_code == 404
    assert "Synthetic NSDA member was not found" in missing.json()["error"]["message"]


def test_configured_nsda_facade_calls_selected_provider(monkeypatch) -> None:
    class FakeHTTPProvider:
        backend = "http"

        def metadata(self):
            return {
                "provider": "National Speech & Debate Association",
                "provider_code": "nsda",
                "backend": "http",
                "dataset_version": "remote-test",
                "generated_at": "2026-08-04T00:00:00Z",
                "fixture": True,
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


def test_http_nsda_adapter_accepts_https_and_unwraps_fixture_envelope() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "provider": "National Speech & Debate Association",
                "provider_code": "nsda",
                "fixture": True,
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
                    "source_ref": "fixture://nsda/topics/pf-test",
                    "synthetic": True,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HTTPNSDAProvider(
        "https://nsda-fixture.example/v1",
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
        "http://nsda-fixture.example/v1",
        "ftp://nsda-fixture.example/v1",
        "https://user:password@nsda-fixture.example/v1",
        "https://nsda-fixture.example/v1?token=secret",
    ],
)
def test_http_nsda_adapter_rejects_unsafe_base_urls(url: str) -> None:
    with pytest.raises(ValueError):
        HTTPNSDAProvider(url)


def test_http_nsda_adapter_allows_local_http_for_development() -> None:
    provider = HTTPNSDAProvider(
        "http://127.0.0.1:8000/nsda/v1",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )
    assert provider.base_url.startswith("http://127.0.0.1")
