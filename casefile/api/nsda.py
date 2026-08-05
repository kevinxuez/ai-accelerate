"""Read-only HTTP surface for the bundled synthetic NSDA provider."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Path, Query

from casefile.providers.nsda import MockNSDAProvider, NSDANotFound, NSDAProviderError


router = APIRouter(prefix="/mock/nsda/v1", tags=["NSDA mock provider"])


@lru_cache(maxsize=1)
def get_mock_nsda_provider() -> MockNSDAProvider:
    return MockNSDAProvider()


def _envelope(data: Any) -> dict[str, Any]:
    metadata = get_mock_nsda_provider().metadata()
    return {
        "provider": metadata["provider"],
        "provider_code": metadata["provider_code"],
        "mock": True,
        "synthetic": True,
        "dataset_version": metadata["dataset_version"],
        "disclaimer": metadata["disclaimer"],
        "data": data,
    }


def _call(operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return _envelope(operation())
    except NSDANotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NSDAProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/metadata")
def metadata() -> dict[str, Any]:
    """Describe the synthetic fixture and its record counts."""

    return _envelope(get_mock_nsda_provider().metadata())


@router.get("/topics/current")
def current_topic(
    event: str = Query(default="Public Forum", min_length=1, max_length=100),
    as_of: date | None = Query(default=None),
) -> dict[str, Any]:
    """Return a synthetic current topic for an event or explicit date."""

    return _call(
        lambda: get_mock_nsda_provider().current_topic(event, as_of=as_of)
    )


@router.get("/rules/search")
def search_rules(
    q: str = Query(default="", max_length=20_000),
    event: str = Query(default="Public Forum", min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Search fictional rule fixtures; results are never authoritative rule answers."""

    return _call(
        lambda: get_mock_nsda_provider().search_rules(q, event=event, limit=limit)
    )


@router.get("/tournaments")
def tournaments(
    state: str | None = Query(default=None, min_length=2, max_length=2),
    event: str | None = Query(default=None, min_length=1, max_length=100),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    """List synthetic tournaments using optional state, event, and date filters."""

    return _call(
        lambda: get_mock_nsda_provider().list_tournaments(
            state=state,
            event=event,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    )


@router.get("/members/{member_id}")
def member(
    member_id: str = Path(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
) -> dict[str, Any]:
    """Return one non-sensitive synthetic member and eligibility record."""

    return _call(lambda: get_mock_nsda_provider().get_member(member_id))
