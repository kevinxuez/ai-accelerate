"""Named graph nodes for classification, routing, tools, and grounded responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from casefile.llm import AnthropicJSONClient, LLMResponseError, LLMUnavailable

from .prompts import CLASSIFIER_SYSTEM
from .state import AgentState
from .tools import CaseFileTools, ToolContext


MAX_ITERATIONS = 5
INTENTS = {
    "retrieve_evidence",
    "explain_rule",
    "generate_drill",
    "progress",
    "ingest_cards",
    "schedule_session",
    "integrity_refusal",
    "unknown",
}
SPEECH_POSITIONS = (
    "first constructive",
    "second constructive",
    "summary",
    "final focus",
    "crossfire",
    "grand crossfire",
    "rebuttal",
)


def _message(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", last))


def _side(text: str) -> str:
    lowered = text.lower()
    pro = bool(re.search(r"(?:^|\W)(?:pro|affirmative|aff)(?:\W|$)", lowered))
    con = bool(re.search(r"(?:^|\W)(?:con|negative|neg)(?:\W|$)", lowered))
    return "pro" if pro and not con else "con" if con and not pro else "unknown"


def _student_id(text: str) -> str | None:
    patterns = (
        r"(?:progress|record|assessment)\s+(?:for|of)\s+([A-Za-z0-9_.-]+)",
        r"(?:drill|session)\s+(?:for|of)\s+(?!(?:the|pro|con)\b)([A-Za-z0-9_.-]+)",
        r"student\s+([A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).rstrip(".,;:")
    return None


def _speech_position(text: str) -> str | None:
    lowered = text.lower()
    return next((position for position in SPEECH_POSITIONS if position in lowered), None)


def _file_path(text: str) -> str | None:
    match = re.search(r'(?:"([^\"]+\.docx)"|\'([^\']+\.docx)\'|(\S+\.docx))', text, re.I)
    return next((value for value in match.groups() if value), None) if match else None


def _start(text: str) -> str | None:
    match = re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b", text)
    return match.group(0) if match else None


def _confirmation_token(text: str) -> str | None:
    match = re.search(r"\b[0-9a-f]{32}\b", text, re.I)
    return match.group(0).lower() if match else None


def deterministic_classification(text: str, state: AgentState) -> dict[str, Any]:
    lowered = text.lower()
    if _confirmation_token(text) and re.search(r"\bconfirm\b", lowered):
        intent = "ingest_cards"
    elif re.search(
        r"\b(?:ingest|import|upload)\b|"
        r"\badd\b.*\b(?:document|cards?|evidence file)\b|"
        r"\.docx.*\b(?:ingest|import|add|upload)\b",
        lowered,
    ):
        intent = "ingest_cards"
    elif re.search(
        r"\b(?:write|draft|generate|make up|invent|fabricate|repair|correct)\b.*"
        r"\b(?:speech|case|card|evidence|citation)\b",
        lowered,
    ):
        intent = "integrity_refusal"
    elif "drill" in lowered or "practice" in lowered:
        intent = "generate_drill"
    elif re.search(r"\b(?:progress|assessment|weakness|feedback|record)\b", lowered):
        intent = "progress"
    elif re.search(r"\b(?:rule|legal|allowed|evidence violation|cite|citation format)\b", lowered):
        intent = "explain_rule"
    elif re.search(r"\b(?:schedule|book|calendar|meeting|session)\b", lowered):
        intent = "schedule_session"
    elif re.search(r"\b(?:evidence|card|source|author|what do we have)\b", lowered) or _side(text) != "unknown":
        intent = "retrieve_evidence"
    else:
        intent = "unknown"

    parameters = {
        "side": _side(text),
        "student_id": _student_id(text),
        "speech_position": _speech_position(text),
        "file_path": _file_path(text),
        "confirmation_token": _confirmation_token(text),
        "start": _start(text),
    }
    missing: list[str] = []
    if intent == "retrieve_evidence":
        if parameters["side"] == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "generate_drill":
        if not parameters["speech_position"]:
            missing.append("speech position")
        if parameters["side"] == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "ingest_cards":
        if not parameters["file_path"] and not parameters["confirmation_token"]:
            missing.append("DOCX file path")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "schedule_session" and not parameters["start"]:
        missing.append("start time in ISO format")
    elif intent == "unknown":
        missing.append("whether you want evidence, a rule, a drill, or progress")
    return {
        "intent": intent,
        **parameters,
        "clarification_needed": bool(missing),
        "clarification_question": (
            "Please provide " + ", ".join(missing) + "." if missing else ""
        ),
    }


@dataclass
class AgentNodes:
    tools: CaseFileTools
    llm: AnthropicJSONClient

    def receive_request(self, state: AgentState) -> AgentState:
        role = state.get("role", "")
        if role not in {"student", "coach"}:
            return {
                "intent": "unknown",
                "clarification_needed": True,
                "clarification_question": "Role must be 'student' or 'coach'.",
            }
        return {
            "iterations": int(state.get("iterations", 0)),
            "tool_trace": list(state.get("tool_trace", [])),
        }

    def classify_intent(self, state: AgentState) -> AgentState:
        text = _message(state)
        classified = deterministic_classification(text, state)
        if self.llm.available:
            try:
                value = self.llm.complete_json(
                    system=CLASSIFIER_SYSTEM.format(
                        tools=", ".join(sorted(self.tools.names_for_role(state["role"])))
                    ),
                    user=text,
                    max_tokens=500,
                )
                if value.get("intent") in INTENTS:
                    classified["intent"] = value["intent"]
                    for key in (
                        "side", "student_id", "speech_position", "file_path",
                        "confirmation_token", "start",
                    ):
                        if key in value and classified.get(key) in {None, "", "unknown"}:
                            classified[key] = value[key]
                    for key in ("clarification_needed", "clarification_question"):
                        if key in value:
                            classified[key] = value[key]
            except (LLMUnavailable, LLMResponseError):
                pass
        intent = classified.pop("intent")
        clarification_needed = bool(classified.pop("clarification_needed", False))
        question = str(classified.pop("clarification_question", "") or "")
        # Re-apply deterministic required-field checks after model classification.
        checked = deterministic_classification(text, {**state, "intent": intent})
        if checked["intent"] == intent and checked["clarification_needed"]:
            clarification_needed = True
            question = checked["clarification_question"]
        return {
            "intent": intent,
            "parameters": classified,
            "clarification_needed": clarification_needed,
            "clarification_question": question,
        }

    def ask_clarification(self, state: AgentState) -> AgentState:
        return {
            "response": state.get("clarification_question")
            or "What would you like CaseFile to do?",
        }

    def route_on_intent(self, state: AgentState) -> AgentState:
        mapping = {
            "retrieve_evidence": "search_cards",
            "explain_rule": "search_rules",
            "generate_drill": "generate_drill",
            "progress": "progress",
            "ingest_cards": "ingest_cards",
            "schedule_session": "schedule_session",
            "integrity_refusal": "integrity_refusal",
        }
        return {"next_action": mapping.get(state.get("intent", "unknown"), "unknown")}

    def execute_tools(self, state: AgentState) -> AgentState:
        if int(state.get("iterations", 0)) >= MAX_ITERATIONS:
            return {"tool_result": "[STOPPED] maximum agent iterations reached."}
        context = ToolContext(
            role=state["role"],
            user_id=state["user_id"],
            resolution=state.get("resolution", ""),
        )
        params = state.get("parameters", {})
        message = _message(state)
        action = state.get("next_action")
        arguments: dict[str, Any]
        trace_entries: list[dict[str, Any]] = []
        if action == "search_cards":
            arguments = {"query": message, "side": params["side"], "n": 3}
            result = self.tools.search_cards(context, **arguments)
        elif action == "search_rules":
            arguments = {"question": message, "n": 3}
            result = self.tools.search_rules(context, **arguments)
        elif action == "generate_drill":
            arguments = {
                "student_id": params.get("student_id") or state["user_id"],
                "speech_position": params.get("speech_position") or "",
                "resolution": state.get("resolution"),
                "side": params.get("side", "unknown"),
            }
            result = self.tools.generate_drill(context, **arguments)
        elif action == "progress":
            student_id = params.get("student_id") or state["user_id"]
            if state["role"] == "coach" and re.search(r"\b(?:log|record|add)\b", message, re.I):
                assessment = message.split(":", 1)[-1].strip()
                weaknesses = re.search(r"weakness(?:es)?\s*[:=]\s*([^.;]+)", message, re.I)
                arguments = {
                    "student_id": student_id,
                    "speech_position": params.get("speech_position") or "unspecified",
                    "resolution": state.get("resolution", ""),
                    "weakness_tags": (
                        [part.strip() for part in weaknesses.group(1).split(",")]
                        if weaknesses
                        else []
                    ),
                    "assessment_text": assessment,
                }
                result = self.tools.log_assessment(context, **arguments)
                if params.get("start") and re.search(
                    r"\b(?:schedule|book|calendar|session)\b", message, re.I
                ):
                    assessment_result = result
                    calendar_arguments = {
                        "student_id": student_id,
                        "start": params["start"],
                        "duration_minutes": 45,
                    }
                    calendar_result = self.tools.schedule_session(
                        context, **calendar_arguments
                    )
                    result = {
                        "assessment": assessment_result,
                        "calendar": calendar_result,
                    }
                    trace_entries.extend(
                        [
                            {
                                "tool": "log_assessment",
                                "arguments": arguments,
                                "result_type": type(assessment_result).__name__,
                            },
                            {
                                "tool": "schedule_session",
                                "arguments": calendar_arguments,
                                "result_type": type(calendar_result).__name__,
                            },
                        ]
                    )
            else:
                arguments = {"student_id": student_id}
                result = self.tools.get_progress(context, **arguments)
        elif action == "ingest_cards":
            arguments = {
                "file_path": params.get("file_path"),
                "resolution": state.get("resolution"),
                "side": params.get("side") if params.get("side") in {"pro", "con"} else None,
                "dry_run": not bool(params.get("confirmation_token")),
                "confirmation_token": params.get("confirmation_token"),
            }
            result = self.tools.ingest_cards(context, **arguments)
        elif action == "schedule_session":
            duration = re.search(r"\b(\d{2,3})\s*minutes?\b", message, re.I)
            email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", message)
            arguments = {
                "student_id": params.get("student_id") or state["user_id"],
                "start": params.get("start") or "",
                "duration_minutes": int(duration.group(1)) if duration else 45,
                "attendee_email": email.group(0) if email else None,
            }
            result = self.tools.schedule_session(context, **arguments)
        elif action == "integrity_refusal":
            arguments = {}
            result = {
                "refusal": "CaseFile cannot fabricate citations, repair or alter evidence text, or write a competition speech.",
                "alternative": "I can retrieve intact cited evidence, explain an on-file rule, or create a drill that requires you to do the speaking.",
            }
        else:
            arguments = {}
            result = "I couldn't determine an action."
        if not trace_entries:
            trace_entries.append(
                {"tool": action, "arguments": arguments, "result_type": type(result).__name__}
            )
        trace = [*state.get("tool_trace", []), *trace_entries]
        return {"tool_result": result, "tool_trace": trace}

    def call_model(self, state: AgentState) -> AgentState:
        """Render only grounded tool output; do not answer from model memory."""
        action = state.get("next_action")
        result = state.get("tool_result")
        if isinstance(result, str):
            return {"response": result}
        if action == "search_cards":
            if not result:
                params = state.get("parameters", {})
                return {
                    "response": (
                        f"No card on file for the {params.get('side', 'requested')} side of "
                        f"{state.get('resolution', 'the active resolution')}. I won't substitute "
                        "a low-similarity card from another topic."
                    )
                }
            blocks = []
            for card in result:
                blocks.append(
                    f"{card['cite_full']}\n\n{card['header']}\n{card['tag']}\n{card['body']}".strip()
                )
            return {"response": "\n\n---\n\n".join(blocks)}
        if action == "search_rules":
            if not result:
                return {
                    "response": "No authoritative rule is on file. I won't answer this from model memory; ask a coach to load the current NSDA rule documents."
                }
            return {
                "response": "\n\n".join(
                    f"Per section {chunk['section_number']} ({chunk['section_title']}) in "
                    f"{chunk['document']}:\n{chunk['text']}"
                    for chunk in result
                )
            }
        if action == "generate_drill":
            lines = [*result["instructions"]]
            for card in result["card_refs"]:
                lines.append(f"- {card['header']}: {card['cite_full']}")
            return {"response": "\n".join(lines)}
        if action == "progress":
            if isinstance(result, list):
                return {
                    "response": "No progress records found."
                    if not result
                    else json.dumps(result, ensure_ascii=False, indent=2)
                }
            if isinstance(result, dict) and "assessment" in result and "calendar" in result:
                calendar = result["calendar"]
                if isinstance(calendar, str):
                    return {"response": f"Assessment logged. {calendar}"}
                mode = "mock calendar" if calendar.get("mock") else "Google Calendar"
                return {
                    "response": (
                        f"Assessment logged. Session scheduled in {mode}: "
                        f"{calendar['start']['dateTime']} (event {calendar['id']})."
                    )
                }
            return {"response": json.dumps(result, ensure_ascii=False, indent=2)}
        if action == "ingest_cards":
            return {"response": result.get("summary", json.dumps(result, indent=2))}
        if action == "schedule_session":
            mode = "mock calendar" if result.get("mock") else "Google Calendar"
            return {
                "response": f"Session scheduled in {mode}: {result['start']['dateTime']} (event {result['id']})."
            }
        if action == "integrity_refusal":
            return {"response": f"{result['refusal']} {result['alternative']}"}
        return {"response": json.dumps(result, ensure_ascii=False, indent=2)}

    def should_continue(self, state: AgentState) -> AgentState:
        iterations = int(state.get("iterations", 0)) + 1
        if iterations > MAX_ITERATIONS:
            return {
                "iterations": iterations,
                "response": "I stopped after the maximum number of tool iterations.",
            }
        return {"iterations": iterations}
