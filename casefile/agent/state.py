"""LangGraph state contract."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:
    def add_messages(left: list[Any], right: list[Any]) -> list[Any]:
        return [*left, *right]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    iterations: int
    intent: str
    role: str
    user_id: str
    resolution: str
    parameters: dict[str, Any]
    clarification_needed: bool
    clarification_question: str
    next_action: str
    tool_result: Any
    tool_trace: list[dict[str, Any]]
    response: str

