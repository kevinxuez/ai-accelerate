"""Typed, agent-scoped registration for CaseFile tools."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ValidationError

from casefile.agents.contracts import AgentName, Role
from casefile.agents.errors import CaseFileError
from .contracts import (
    AssessmentArgs,
    CurrentTopicArgs,
    ProgressArgs,
    ScheduleArgs,
    SearchCardsArgs,
    SearchRulesArgs,
)
from .context import ToolContext, ToolRuntime
from .evidence import ListConfirmedFilesArgs
from .ingestion import CommitIngestionArgs, StageIngestionArgs


ToolHandler = Callable[..., Any]
_TRACE_SINK: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "casefile_tool_trace_sink",
    default=None,
)


@contextmanager
def capture_tool_invocations(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Capture sanitized per-invocation events for the active graph call."""

    token = _TRACE_SINK.set(sink)
    try:
        yield
    finally:
        _TRACE_SINK.reset(token)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_type: Any
    handler: ToolHandler
    agents: frozenset[AgentName]
    allowed_roles: frozenset[Role]


class ToolRegistry:
    def __init__(
        self,
        runtime: ToolRuntime,
        definitions: list[ToolDefinition],
    ) -> None:
        self.runtime = runtime
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("tool definition names must be unique")

    def for_agent(self, agent: AgentName) -> tuple[ToolDefinition, ...]:
        return tuple(
            definition
            for definition in self._definitions.values()
            if agent in definition.agents
        )

    def names_for_agent(self, agent: AgentName) -> set[str]:
        return {definition.name for definition in self.for_agent(agent)}

    def invoke(
        self,
        agent: AgentName,
        name: str,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> Any:
        definition = self._definitions.get(name)
        if definition is None or agent not in definition.agents:
            self.runtime.agent_denied(
                replace(context, agent=agent),
                f"invoke tool '{name}'",
                name,
            )
        try:
            args = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            self.runtime.invalid(context, name, exc)
        active_context = (
            context if context.agent == agent else replace(context, agent=agent)
        )
        arguments = args.model_dump(mode="python")
        sink = _TRACE_SINK.get()
        if sink is not None:
            sink.append(
                {
                    "agent": agent,
                    "tool": name,
                    "stage": f"tools.{name}",
                    "status": "started",
                    "arguments": self.runtime._safe_audit_arguments(arguments),
                    "result_summary": None,
                    "error_code": None,
                }
            )
        try:
            result = definition.handler(active_context, **arguments)
        except CaseFileError as error:
            if sink is not None:
                sink.append(
                    {
                        "agent": agent,
                        "tool": name,
                        "stage": error.stage,
                        "status": "failed",
                        "arguments": {},
                        "result_summary": None,
                        "error_code": error.code.value,
                    }
                )
            raise
        if sink is not None:
            sink.append(
                {
                    "agent": agent,
                    "tool": name,
                    "stage": f"tools.{name}",
                    "status": "completed",
                    "arguments": {},
                    "result_summary": self._result_summary(result),
                    "error_code": None,
                }
            )
        return result

    @staticmethod
    def _result_summary(result: Any) -> str:
        if isinstance(result, list):
            return f"Returned {len(result)} items."
        if isinstance(result, dict):
            artifact = result.get("artifact_type")
            if artifact:
                return f"Returned {artifact}."
            return "Returned a structured result."
        return f"Returned {type(result).__name__}."


def build_tool_registry(tools: Any, runtime: ToolRuntime) -> ToolRegistry:
    librarian = frozenset({"evidence_librarian"})
    coach = frozenset({"skills_coach"})
    supervisor = frozenset({"supervisor"})
    both = frozenset({"student", "coach"})
    definitions = [
        ToolDefinition(
            "list_confirmed_files",
            "List committed evidence files containing indexable cards.",
            ListConfirmedFilesArgs,
            list[str],
            tools.evidence.list_confirmed_files,
            librarian,
            both,
        ),
        ToolDefinition(
            "search_cards",
            "Search confirmed cards with resolution, side, and source filters.",
            SearchCardsArgs,
            list[dict[str, Any]],
            tools.evidence.search_cards,
            librarian,
            both,
        ),
        ToolDefinition(
            "search_rules",
            "Search indexed authoritative rules.",
            SearchRulesArgs,
            list[dict[str, Any]],
            tools.rules.search_rules,
            librarian,
            both,
        ),
        ToolDefinition(
            "get_current_topic",
            "Read the configured provider's current topic.",
            CurrentTopicArgs,
            dict[str, Any] | str,
            tools.topic.get_current_topic,
            librarian,
            both,
        ),
        ToolDefinition(
            "stage_ingestion_preview",
            "Build and stage a non-committed DOCX ingestion preview.",
            StageIngestionArgs,
            dict[str, Any],
            tools.ingestion_tools.stage_ingestion_preview,
            librarian,
            both,
        ),
        ToolDefinition(
            "commit_ingestion",
            "Commit a staged ingestion using its confirmation token.",
            CommitIngestionArgs,
            dict[str, Any],
            tools.ingestion_tools.commit_ingestion,
            librarian,
            both,
        ),
        ToolDefinition(
            "get_progress",
            "Read progress records subject to ownership policy.",
            ProgressArgs,
            list[dict[str, Any]],
            tools.progress.get_progress,
            coach,
            both,
        ),
        ToolDefinition(
            "log_assessment",
            "Write a coach-authorized assessment record.",
            AssessmentArgs,
            dict[str, Any],
            tools.progress.log_assessment,
            coach,
            frozenset({"coach"}),
        ),
        ToolDefinition(
            "schedule_session",
            "Stage or create a coaching calendar event.",
            ScheduleArgs,
            dict[str, Any],
            tools.calendar.schedule_session,
            supervisor,
            both,
        ),
    ]
    return ToolRegistry(runtime, definitions)
