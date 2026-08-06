"""Explicit synthetic-fixture and HTTP NSDA-compatible providers.

The bundled records are deliberately fictional. They exercise provider boundaries without
claiming that the National Speech & Debate Association publishes these endpoints or data.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from casefile.config import Settings, get_settings


DEFAULT_DATA_PATH = Path(__file__).with_name("nsda_fixture.json")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]{1,}")
EVENT_ALIASES = {
    "pf": "Public Forum",
    "public forum": "Public Forum",
    "public_forum": "Public Forum",
    "public-forum": "Public Forum",
    "ld": "Lincoln-Douglas",
    "lincoln douglas": "Lincoln-Douglas",
    "lincoln-douglas": "Lincoln-Douglas",
}


class NSDAProviderError(RuntimeError):
    """The configured NSDA-compatible provider failed or returned malformed data."""


class NSDANotFound(NSDAProviderError):
    """A requested synthetic provider record does not exist."""


class NSDAProviderDisabled(NSDAProviderError):
    """The optional NSDA capability is explicitly disabled."""


class NSDAModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NSDATopic(NSDAModel):
    id: str = Field(min_length=1, max_length=100)
    event: str = Field(min_length=1, max_length=100)
    division: str = Field(min_length=1, max_length=100)
    season: str = Field(min_length=1, max_length=20)
    release_window: str = Field(min_length=1, max_length=100)
    resolution: str = Field(min_length=1, max_length=500)
    effective_from: date
    effective_to: date
    current: bool = False
    source_ref: str = Field(min_length=1, max_length=300)
    synthetic: bool


class NSDARule(NSDAModel):
    id: str = Field(min_length=1, max_length=100)
    event: str = Field(min_length=1, max_length=100)
    section_number: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=10_000)
    edition: str = Field(min_length=1, max_length=100)
    effective_date: date
    source_ref: str = Field(min_length=1, max_length=300)
    synthetic: bool


class NSDARuleResult(NSDARule):
    score: float = Field(ge=0, le=1)


class NSDATournament(NSDAModel):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    start_date: date
    end_date: date
    timezone: str = Field(min_length=1, max_length=100)
    city: str = Field(min_length=1, max_length=100)
    state: str = Field(pattern=r"^[A-Z]{2}$")
    events: list[str] = Field(min_length=1, max_length=30)
    status: str = Field(min_length=1, max_length=50)
    registration_ref: str = Field(min_length=1, max_length=300)
    synthetic: bool


class NSDAMember(NSDAModel):
    member_id: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)
    roles: list[str] = Field(min_length=1, max_length=10)
    school: str = Field(min_length=1, max_length=200)
    state: str = Field(pattern=r"^[A-Z]{2}$")
    eligible_events: list[str] = Field(max_length=30)
    status: str = Field(min_length=1, max_length=50)
    synthetic: bool


class NSDADataset(NSDAModel):
    provider: str = Field(min_length=1, max_length=200)
    provider_code: str = Field(pattern=r"^nsda$")
    dataset_version: str = Field(min_length=1, max_length=50)
    generated_at: str = Field(min_length=1, max_length=50)
    fixture: Literal[True]
    synthetic: Literal[True]
    disclaimer: str = Field(min_length=1, max_length=1000)
    topics: list[NSDATopic] = Field(max_length=100)
    rules: list[NSDARule] = Field(max_length=1000)
    tournaments: list[NSDATournament] = Field(max_length=1000)
    members: list[NSDAMember] = Field(max_length=1000)


class NSDACounts(NSDAModel):
    topics: int = Field(ge=0)
    rules: int = Field(ge=0)
    tournaments: int = Field(ge=0)
    members: int = Field(ge=0)


class NSDAMetadata(NSDAModel):
    provider: str = Field(min_length=1, max_length=200)
    provider_code: str = Field(pattern=r"^nsda$")
    backend: str = Field(min_length=1, max_length=50)
    dataset_version: str = Field(min_length=1, max_length=50)
    generated_at: str = Field(min_length=1, max_length=50)
    fixture: bool
    synthetic: bool
    disclaimer: str = Field(min_length=1, max_length=1000)
    counts: NSDACounts


class NSDAEnvelope(NSDAModel):
    provider: str = Field(min_length=1, max_length=200)
    provider_code: str = Field(pattern=r"^nsda$")
    backend: str | None = Field(default=None, min_length=1, max_length=50)
    fixture: bool
    synthetic: bool
    dataset_version: str = Field(min_length=1, max_length=50)
    disclaimer: str = Field(min_length=1, max_length=1000)
    data: Any


class NSDAProvider(Protocol):
    backend: str

    def metadata(self) -> dict[str, Any]: ...

    def current_topic(
        self, event: str = "Public Forum", *, as_of: date | None = None
    ) -> dict[str, Any]: ...

    def search_rules(
        self, query: str, *, event: str = "Public Forum", limit: int = 10
    ) -> list[dict[str, Any]]: ...

    def list_tournaments(
        self,
        *,
        state: str | None = None,
        event: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]: ...

    def get_member(self, member_id: str) -> dict[str, Any]: ...


def normalize_event(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip()).lower()
    return EVENT_ALIASES.get(cleaned, value.strip())


def _terms(value: str) -> set[str]:
    return {term.lower() for term in TOKEN.findall(value)}


class FixtureNSDAProvider:
    """Read-only provider backed by a validated, bundled synthetic fixture."""

    backend = "fixture"

    def __init__(self, data_path: str | Path | None = None) -> None:
        path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        try:
            self.dataset = NSDADataset.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise NSDAProviderError(f"NSDA fixture dataset is invalid: {exc}") from exc
        if not self.dataset.fixture or not self.dataset.synthetic:
            raise NSDAProviderError(
                "NSDA fixture dataset must be marked fixture=true and synthetic=true"
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.dataset.provider,
            "provider_code": self.dataset.provider_code,
            "backend": self.backend,
            "dataset_version": self.dataset.dataset_version,
            "generated_at": self.dataset.generated_at,
            "fixture": True,
            "synthetic": True,
            "disclaimer": self.dataset.disclaimer,
            "counts": {
                "topics": len(self.dataset.topics),
                "rules": len(self.dataset.rules),
                "tournaments": len(self.dataset.tournaments),
                "members": len(self.dataset.members),
            },
        }

    def current_topic(
        self, event: str = "Public Forum", *, as_of: date | None = None
    ) -> dict[str, Any]:
        event_name = normalize_event(event)
        matching = [topic for topic in self.dataset.topics if topic.event == event_name]
        if as_of is not None:
            matching = [
                topic
                for topic in matching
                if topic.effective_from <= as_of <= topic.effective_to
            ]
        else:
            current = [topic for topic in matching if topic.current]
            matching = current or matching
        if not matching:
            suffix = f" for {as_of.isoformat()}" if as_of else ""
            raise NSDANotFound(f"No synthetic {event_name} topic exists{suffix}")
        topic = sorted(matching, key=lambda item: item.effective_from, reverse=True)[0]
        return topic.model_dump(mode="json")

    def search_rules(
        self, query: str, *, event: str = "Public Forum", limit: int = 10
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        event_name = normalize_event(event)
        query_terms = _terms(query)
        candidates: list[tuple[float, NSDARule]] = []
        for rule in self.dataset.rules:
            if rule.event != event_name:
                continue
            searchable = _terms(f"{rule.section_number} {rule.title} {rule.text}")
            if query_terms:
                overlap = len(query_terms & searchable)
                if not overlap:
                    continue
                score = overlap / len(query_terms)
            else:
                score = 1.0
            candidates.append((score, rule))
        candidates.sort(key=lambda item: (-item[0], item[1].section_number))
        return [
            {**rule.model_dump(mode="json"), "score": round(score, 6)}
            for score, rule in candidates[:limit]
        ]

    def list_tournaments(
        self,
        *,
        state: str | None = None,
        event: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        state_code = state.strip().upper() if state else None
        if state_code and not re.fullmatch(r"[A-Z]{2}", state_code):
            raise ValueError("state must be a two-letter code")
        event_name = normalize_event(event) if event else None
        values = [
            tournament
            for tournament in self.dataset.tournaments
            if (state_code is None or tournament.state == state_code)
            and (event_name is None or event_name in tournament.events)
            and (start_date is None or tournament.end_date >= start_date)
            and (end_date is None or tournament.start_date <= end_date)
        ]
        values.sort(key=lambda item: (item.start_date, item.name))
        return [value.model_dump(mode="json") for value in values[:limit]]

    def get_member(self, member_id: str) -> dict[str, Any]:
        cleaned = member_id.strip()
        member = next(
            (record for record in self.dataset.members if record.member_id == cleaned),
            None,
        )
        if member is None:
            raise NSDANotFound(f"Synthetic NSDA member was not found: {cleaned}")
        return member.model_dump(mode="json")


class HTTPNSDAProvider:
    """Client for an explicitly configured NSDA-compatible HTTP API."""

    backend = "http"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        if parsed.scheme != "https" and not local_http:
            raise ValueError(
                "NSDA_BASE_URL must use HTTPS except for localhost testing"
            )
        if not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("NSDA_BASE_URL is malformed or contains credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("NSDA_BASE_URL must not contain a query or fragment")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def metadata(self) -> dict[str, Any]:
        return self._validated(NSDAMetadata, self._get("/metadata"), "metadata")

    def current_topic(
        self, event: str = "Public Forum", *, as_of: date | None = None
    ) -> dict[str, Any]:
        params = {"event": event}
        if as_of:
            params["as_of"] = as_of.isoformat()
        return self._validated(
            NSDATopic,
            self._get("/topics/current", params=params),
            "topic",
        )

    def search_rules(
        self, query: str, *, event: str = "Public Forum", limit: int = 10
    ) -> list[dict[str, Any]]:
        payload = self._get(
            "/rules/search",
            params={"q": query, "event": event, "limit": limit},
        )
        if not isinstance(payload, list):
            raise NSDAProviderError("NSDA provider rules data must be a list")
        return [self._validated(NSDARuleResult, value, "rule") for value in payload]

    def list_tournaments(
        self,
        *,
        state: str | None = None,
        event: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        for key, value in {
            "state": state,
            "event": event,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }.items():
            if value is not None:
                params[key] = value
        payload = self._get("/tournaments", params=params)
        if not isinstance(payload, list):
            raise NSDAProviderError("NSDA provider tournament data must be a list")
        return [
            self._validated(NSDATournament, value, "tournament") for value in payload
        ]

    def get_member(self, member_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", member_id):
            raise ValueError("member_id contains unsupported characters")
        return self._validated(
            NSDAMember,
            self._get(f"/members/{member_id}"),
            "member",
        )

    @staticmethod
    def _validated(model: type[BaseModel], value: Any, label: str) -> dict[str, Any]:
        try:
            return model.model_validate(value).model_dump(mode="json")
        except ValidationError as exc:
            raise NSDAProviderError(
                f"NSDA provider returned invalid {label} data"
            ) from exc

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self.client.get(
                f"{self.base_url}{path}", params=params, headers=headers
            )
            if response.status_code == 404:
                raise NSDANotFound(response.text[:500] or "NSDA record not found")
            response.raise_for_status()
            payload = response.json()
        except NSDANotFound:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise NSDAProviderError(f"NSDA provider request failed: {exc}") from exc
        try:
            envelope = NSDAEnvelope.model_validate(payload)
        except ValidationError as exc:
            raise NSDAProviderError(
                "NSDA provider returned a malformed response envelope"
            ) from exc
        return envelope.data


def build_nsda_provider(settings: Settings | None = None) -> NSDAProvider:
    active = settings or get_settings()
    if active.nsda_provider == "http":
        return HTTPNSDAProvider(
            active.nsda_base_url or "",
            api_key=active.nsda_api_key,
            timeout_seconds=active.nsda_timeout_seconds,
        )
    if active.nsda_provider == "fixture":
        return FixtureNSDAProvider(active.nsda_fixture_path)
    raise NSDAProviderDisabled("The NSDA provider capability is disabled")
