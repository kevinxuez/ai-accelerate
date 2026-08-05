"""LangGraph orchestration with a dependency-free sequential fallback."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

from casefile.config import Settings, get_settings
from casefile.llm import build_anthropic_client
from casefile.security.audit import RateLimiter, SecurityAuditor

from .nodes import AgentNodes, deterministic_classification
from .session import ClarificationSessionStore, pending_clarification
from .state import AgentState
from .tools import CaseFileTools


END_COACHING = re.compile(
    r"^(?:end|stop|exit|finish|quit)(?:\s+(?:the\s+)?coach(?:ing)?(?:\s+simulation|\s+session)?)?[.!]?$",
    re.I,
)


class CaseFileAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        tools: CaseFileTools | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tools = tools or CaseFileTools(self.settings)
        self.sessions = ClarificationSessionStore(self.settings)
        self.nodes = AgentNodes(
            self.tools,
            build_anthropic_client(self.settings),
            SecurityAuditor(self.settings.security_audit_path),
            RateLimiter(max(1, self.settings.requests_per_minute)),
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
        graph.add_node("screen_request", self.nodes.screen_request)
        graph.add_node("block_prompt_injection", self.nodes.block_prompt_injection)
        graph.add_node("classify_intent", self.nodes.classify_intent)
        graph.add_node("ask_clarification", self.nodes.ask_clarification)
        graph.add_node("route_on_intent", self.nodes.route_on_intent)
        graph.add_node("execute_tools", self.nodes.execute_tools)
        graph.add_node("observe_results", self.nodes.observe_results)
        graph.add_node("call_model", self.nodes.call_model)
        graph.add_node("should_continue", self.nodes.should_continue)
        graph.add_edge(START, "receive_request")
        graph.add_edge("receive_request", "screen_request")
        graph.add_conditional_edges(
            "screen_request",
            lambda state: (
                "block"
                if state.get("security_decision", {}).get("action") == "block"
                else "classify"
            ),
            {"block": "block_prompt_injection", "classify": "classify_intent"},
        )
        graph.add_edge("block_prompt_injection", END)
        graph.add_conditional_edges(
            "classify_intent",
            lambda state: "clarify" if state.get("clarification_needed") else "route",
            {"clarify": "ask_clarification", "route": "route_on_intent"},
        )
        graph.add_edge("ask_clarification", END)
        graph.add_edge("route_on_intent", "execute_tools")
        graph.add_edge("execute_tools", "observe_results")
        graph.add_conditional_edges(
            "observe_results",
            lambda state: state.get("next_step", "finish"),
            {"replan": "execute_tools", "finish": "call_model"},
        )
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
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip() or len(message) > 20_000:
            raise ValueError("message must be 1 to 20000 characters")
        if role not in {"student", "coach"}:
            raise ValueError("role must be 'student' or 'coach'")
        if not user_id.strip() or len(user_id) > 100:
            raise ValueError("user_id must be 1 to 100 characters")
        if len(resolution) > 200:
            raise ValueError("resolution must be at most 200 characters")
        if request_id is not None and (not request_id.strip() or len(request_id) > 200):
            raise ValueError("request_id must be 1 to 200 characters")
        active_session_id = self.sessions.validate_session_id(
            session_id or uuid.uuid4().hex
        )
        active_request_id = (
            hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            if request_id
            else uuid.uuid4().hex
        )
        pending = self.sessions.load(
            active_session_id,
            role=role,
            user_id=user_id,
            resolution=resolution,
        )
        if (
            pending is not None
            and pending.intent == "coach_simulation"
            and END_COACHING.fullmatch(message.strip())
        ):
            self.sessions.clear(active_session_id)
            return {
                "response": "Coach simulation ended.",
                "intent": "coach_simulation",
                "iterations": 0,
                "tool_trace": [],
                "task_trace": [],
                "backend": self.backend,
                "request_id": active_request_id,
                "security_decision": {
                    "risk": "low",
                    "action": "allow",
                    "signals": [],
                    "safe_for_model": True,
                    "safe_for_write_tools": True,
                    "trust": "untrusted_user",
                },
                "session_id": active_session_id,
                "awaiting_clarification": False,
                "resumed_from_clarification": False,
                "coach_simulation_active": False,
                "plan_cycles": 0,
                "observations": [],
            }
        effective_message = message
        resumed_from_clarification = False
        if pending is not None:
            reply_intent = deterministic_classification(
                message,
                {
                    "role": role,
                    "user_id": user_id,
                    "resolution": resolution,
                },
            )["intent"]
            changed_task = (
                pending.intent != "unknown"
                and reply_intent not in {"unknown", pending.intent}
            )
            if changed_task:
                self.sessions.clear(active_session_id)
                pending = None
            else:
                effective_message = (
                    f"{pending.message}\nUser clarification: {message.strip()}"
                )
                resumed_from_clarification = True
        initial: AgentState = {
            "messages": [{"role": "user", "content": effective_message}],
            "iterations": 0,
            "intent": "",
            "role": role,
            "user_id": user_id,
            "resolution": resolution,
            "tool_trace": [],
            "request_id": active_request_id,
            "security_events": [],
        }
        if self._compiled is not None:
            result = self._compiled.invoke(initial)
        else:
            result: AgentState = dict(initial)
            for node in (self.nodes.receive_request, self.nodes.screen_request):
                result.update(node(result))
            if result.get("security_decision", {}).get("action") == "block":
                result.update(self.nodes.block_prompt_injection(result))
            else:
                result.update(self.nodes.classify_intent(result))
            if result.get("response"):
                pass
            elif result.get("clarification_needed"):
                result.update(self.nodes.ask_clarification(result))
            else:
                result.update(self.nodes.route_on_intent(result))
                while True:
                    result.update(self.nodes.execute_tools(result))
                    result.update(self.nodes.observe_results(result))
                    if result.get("next_step") != "replan":
                        break
                result.update(self.nodes.call_model(result))
                result.update(self.nodes.should_continue(result))
        awaiting_clarification = bool(result.get("clarification_needed")) and bool(
            result.get("response")
        )
        clarification_turns = (
            pending.turns + 1 if resumed_from_clarification and pending else 1
        )
        coach_simulation_active = (
            result.get("intent") == "coach_simulation"
            and not awaiting_clarification
            and any(
                item.get("action") == "coach_simulation"
                and item.get("status") == "success"
                for item in result.get("task_results", [])
            )
        )
        if awaiting_clarification and clarification_turns <= 6:
            self.sessions.save(
                pending_clarification(
                    session_id=active_session_id,
                    role=role,
                    user_id=user_id,
                    resolution=resolution,
                    message=effective_message,
                    intent=str(result.get("intent", "unknown")),
                    parameters=dict(result.get("parameters", {})),
                    question=str(result.get("response", "")),
                    turns=clarification_turns,
                )
            )
        elif coach_simulation_active and clarification_turns <= 12:
            self.sessions.save(
                pending_clarification(
                    session_id=active_session_id,
                    role=role,
                    user_id=user_id,
                    resolution=resolution,
                    message=effective_message,
                    intent="coach_simulation",
                    parameters=dict(result.get("parameters", {})),
                    question=str(result.get("response", "")),
                    turns=clarification_turns,
                )
            )
        else:
            self.sessions.clear(active_session_id)
            if awaiting_clarification:
                result["response"] = (
                    "I couldn't resolve the request after six clarification turns. "
                    "Please start a new session with the action, side, resolution, and "
                    "any required time or coaching focus."
                )
                awaiting_clarification = False
            elif coach_simulation_active:
                result["response"] = (
                    str(result.get("response", ""))
                    + "\n\nCoach simulation ended after twelve turns."
                )
                coach_simulation_active = False
        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", "unknown"),
            "iterations": result.get("iterations", 0),
            "tool_trace": result.get("tool_trace", []),
            "task_trace": [
                {
                    "id": item.get("id"),
                    "action": item.get("action"),
                    "status": item.get("status"),
                    "attempts": item.get("attempts"),
                    "depends_on": item.get("depends_on", []),
                }
                for item in result.get("task_results", [])
            ],
            "backend": self.backend,
            "request_id": result.get("request_id", ""),
            "security_decision": result.get("security_decision", {}),
            "session_id": active_session_id,
            "awaiting_clarification": awaiting_clarification,
            "resumed_from_clarification": resumed_from_clarification,
            "coach_simulation_active": coach_simulation_active,
            "plan_cycles": result.get("plan_cycles", 0),
            "observations": result.get("observations", []),
            "grounding_cards": result.get("grounding_cards", []),
            "evidence_argument": result.get("evidence_argument"),
            "coach_turn": result.get("coach_turn"),
        }

    def clear_session(self, session_id: str) -> None:
        """Forget any pending clarification for a completed external workflow."""

        self.sessions.clear(session_id)
