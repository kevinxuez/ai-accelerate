"""Confirmed-file listing and filtered evidence retrieval tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, ValidationError

from casefile.agents.contracts import StrictContract
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.retrieval import CaseFileIndex
from casefile.security.prompt_guard import inspect_text

from .contracts import SearchCardsArgs
from .context import ToolContext, ToolRuntime


LIBRARIAN = frozenset({"evidence_librarian"})
READ_ROLES = frozenset({"student", "coach"})


class ListConfirmedFilesArgs(StrictContract):
    resolution: str = Field(min_length=1, max_length=500)
    side: Literal["pro", "con"] | None = None


class EvidenceTools:
    def __init__(self, runtime: ToolRuntime, index: CaseFileIndex) -> None:
        self.runtime = runtime
        self.index = index

    def list_confirmed_files(
        self,
        context: ToolContext,
        *,
        resolution: str | None = None,
        side: str | None = None,
    ) -> list[str]:
        """List committed source files that contain at least one indexable card."""

        self.runtime.authorize(
            context,
            tool="list_confirmed_files",
            action="list confirmed evidence files",
            roles=READ_ROLES,
            agents=LIBRARIAN,
        )
        try:
            args = ListConfirmedFilesArgs(
                resolution=resolution or context.resolution,
                side=side,
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "list_confirmed_files", exc)
        result = self.index.available_card_files(
            resolution=args.resolution,
            side=args.side,
        )
        self.runtime.audit(
            context,
            "list_confirmed_files",
            {"resolution": args.resolution, "side": args.side},
            result,
        )
        return result

    def search_cards(
        self,
        context: ToolContext,
        query: str,
        side: str,
        resolution: str | None = None,
        n: int = 5,
        source_files: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.runtime.authorize(
            context,
            tool="search_cards",
            action="search evidence",
            roles=READ_ROLES,
            agents=LIBRARIAN,
        )
        active_resolution = resolution or context.resolution
        try:
            args = SearchCardsArgs(
                query=query,
                side=side,
                resolution=active_resolution,
                n=n,
                source_files=source_files or [],
            )
        except ValidationError as exc:
            self.runtime.invalid(context, "search_cards", exc)
        decision = inspect_text(args.query, trust="untrusted_user")
        if decision.action == "block":
            self.runtime.blocked(
                context,
                "search_cards",
                {"query": query},
                signals=decision.signals,
            )
        try:
            result = self.index.search_cards(
                args.query,
                resolution=args.resolution,
                side=args.side,
                source_files=args.source_files,
                n=args.n,
            )
        except CaseFileError:
            raise
        except Exception as exc:
            self.runtime.fail(
                context,
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The evidence retrieval backend is unavailable.",
                tool="search_cards",
                cause=exc,
                retryable=True,
            )
        self.runtime.audit(
            context,
            "search_cards",
            {
                "query": query,
                "side": side,
                "resolution": active_resolution,
                "n": n,
                "source_files": args.source_files,
            },
            result,
        )
        return result
