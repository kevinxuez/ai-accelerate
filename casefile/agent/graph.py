"""LangGraph orchestration with a dependency-free sequential fallback."""

from __future__ import annotations

from typing import Any

from casefile.config import Settings, get_settings
from casefile.llm import AnthropicJSONClient

from .nodes import AgentNodes
from .state import AgentState
from .tools import CaseFileTools


class CaseFileAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        tools: CaseFileTools | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tools = tools or CaseFileTools(self.settings)
        self.nodes = AgentNodes(
            self.tools,
            AnthropicJSONClient(self.settings.anthropic_api_key, self.settings.model),
        )
        self._compiled: Any | None = self._build_langgraph()

    @property
    def backend(self) -> str:
        return "langgraph" if self._compiled is not None else "sequential"

    def _build_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        graph = StateGraph(AgentState)
        graph.add_node("receive_request", self.nodes.receive_request)
        graph.add_node("classify_intent", self.nodes.classify_intent)
        graph.add_node("ask_clarification", self.nodes.ask_clarification)
        graph.add_node("route_on_intent", self.nodes.route_on_intent)
        graph.add_node("execute_tools", self.nodes.execute_tools)
        graph.add_node("call_model", self.nodes.call_model)
        graph.add_node("should_continue", self.nodes.should_continue)
        graph.add_edge(START, "receive_request")
        graph.add_edge("receive_request", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            lambda state: "clarify" if state.get("clarification_needed") else "route",
            {"clarify": "ask_clarification", "route": "route_on_intent"},
        )
        graph.add_edge("ask_clarification", END)
        graph.add_edge("route_on_intent", "execute_tools")
        graph.add_edge("execute_tools", "call_model")
        graph.add_edge("call_model", "should_continue")
        graph.add_edge("should_continue", END)
        return graph.compile()

    def ask(
        self,
        message: str,
        *,
        role: str,
        user_id: str,
        resolution: str,
    ) -> dict[str, Any]:
        if role not in {"student", "coach"}:
            raise ValueError("role must be 'student' or 'coach'")
        if not user_id.strip():
            raise ValueError("user_id is required")
        initial: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "iterations": 0,
            "intent": "",
            "role": role,
            "user_id": user_id,
            "resolution": resolution,
            "tool_trace": [],
        }
        if self._compiled is not None:
            result = self._compiled.invoke(initial)
        else:
            result: AgentState = dict(initial)
            for node in (self.nodes.receive_request, self.nodes.classify_intent):
                result.update(node(result))
            if result.get("clarification_needed"):
                result.update(self.nodes.ask_clarification(result))
            else:
                for node in (
                    self.nodes.route_on_intent,
                    self.nodes.execute_tools,
                    self.nodes.call_model,
                    self.nodes.should_continue,
                ):
                    result.update(node(result))
        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", "unknown"),
            "iterations": result.get("iterations", 0),
            "tool_trace": result.get("tool_trace", []),
            "backend": self.backend,
        }
