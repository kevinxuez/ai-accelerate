"""Authoritative rules retrieval tool."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.retrieval import CaseFileIndex
from casefile.security.prompt_guard import inspect_text

from .contracts import SearchRulesArgs
from .context import ToolContext, ToolRuntime


class RuleTools:
    def __init__(self, runtime: ToolRuntime, index: CaseFileIndex) -> None:
        self.runtime = runtime
        self.index = index

    def search_rules(
        self, context: ToolContext, question: str, n: int = 3
    ) -> list[dict[str, Any]]:
        self.runtime.authorize(
            context,
            tool="search_rules",
            action="search rules",
            roles=frozenset({"student", "coach"}),
            agents=frozenset({"evidence_librarian"}),
        )
        try:
            args = SearchRulesArgs(question=question, n=n)
        except ValidationError as exc:
            self.runtime.invalid(context, "search_rules", exc)
        decision = inspect_text(args.question, trust="untrusted_user")
        if decision.action == "block":
            self.runtime.blocked(
                context,
                "search_rules",
                {"question": question},
                signals=decision.signals,
            )
        try:
            result = self.index.search_rules(args.question, n=args.n)
        except CaseFileError:
            raise
        except Exception as exc:
            self.runtime.fail(
                context,
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The rules retrieval backend is unavailable.",
                tool="search_rules",
                cause=exc,
                retryable=True,
            )
        self.runtime.audit(
            context,
            "search_rules",
            {"question": question, "n": n},
            result,
        )
        return result
