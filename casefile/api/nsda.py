"""Read-only HTTP surfaces for local and configured NSDA-compatible providers."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Path, Query

from casefile.providers.nsda import (
    NSDANotFound,
    NSDAProvider,
    NSDAProviderError,
    build_nsda_provider,
)


provider_router = APIRouter(prefix="/nsda/v1", tags=["NSDA provider facade"])


@lru_cache(maxsize=1)
def get_configured_nsda_provider() -> NSDAProvider:
    try:
        return build_nsda_provider()
    except NSDAProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _envelope(
    provider: NSDAProvider,
    data: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider_metadata = metadata or provider.metadata()
    return {
        "provider": provider_metadata["provider"],
        "provider_code": provider_metadata["provider_code"],
        "backend": provider.backend,
        "fixture": bool(provider_metadata.get("fixture")),
        "synthetic": bool(provider_metadata.get("synthetic")),
        "dataset_version": provider_metadata["dataset_version"],
        "disclaimer": provider_metadata["disclaimer"],
        "data": data,
    }


def _call(provider: NSDAProvider, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return _envelope(provider, operation())
    except NSDANotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NSDAProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@provider_router.get("/metadata")
def configured_metadata() -> dict[str, Any]:
    """Describe the explicitly selected NSDA-compatible provider."""

    provider = get_configured_nsda_provider()
    provider_metadata = provider.metadata()
    return _envelope(provider, provider_metadata, metadata=provider_metadata)


@provider_router.get("/topics/current")
def configured_current_topic(
    event: str = Query(default="Public Forum", min_length=1, max_length=100),
    as_of: date | None = Query(default=None),
) -> dict[str, Any]:
    """Return the configured provider's current topic for an event or date."""

    provider = get_configured_nsda_provider()
    return _call(provider, lambda: provider.current_topic(event, as_of=as_of))


@provider_router.get("/rules/search")
def configured_search_rules(
    q: str = Query(default="", max_length=20_000),
    event: str = Query(default="Public Forum", min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Search configured synthetic rule fixtures without changing search_rules."""

    provider = get_configured_nsda_provider()
    return _call(
        provider,
        lambda: provider.search_rules(q, event=event, limit=limit),
    )


@provider_router.get("/tournaments")
def configured_tournaments(
    state: str | None = Query(default=None, min_length=2, max_length=2),
    event: str | None = Query(default=None, min_length=1, max_length=100),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    """List tournaments from the configured NSDA-compatible provider."""

    provider = get_configured_nsda_provider()
    return _call(
        provider,
        lambda: provider.list_tournaments(
            state=state,
            event=event,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        ),
    )


@provider_router.get("/members/{member_id}")
def configured_member(
    member_id: str = Path(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
) -> dict[str, Any]:
    """Return a non-sensitive synthetic member record from the configured provider."""

    provider = get_configured_nsda_provider()
    return _call(provider, lambda: provider.get_member(member_id))
