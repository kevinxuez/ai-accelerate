"""Named graph nodes for classification, routing, tools, and grounded responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from casefile.llm import AnthropicJSONClient, LLMResponseError, LLMUnavailable
from casefile.security.audit import RateLimiter, SecurityAuditor
from casefile.security.prompt_guard import (
    BLOCKED_RESPONSE,
    GuardDecision,
    inspect_text,
    summarize_untrusted_text,
)
from casefile.security.schemas import (
    ClassifierOutput,
    CoachTurnOutput,
    EvidenceArgumentOutput,
)

from .planning import TaskPlan, TaskSpec, execute_task_plan, make_task
from .prompts import (
    CLASSIFIER_SYSTEM,
    COACH_SIMULATOR_SYSTEM,
    EVIDENCE_ARGUMENT_SYSTEM,
)
from .state import AgentState
from .tools import CaseFileTools, ToolContext


MAX_ITERATIONS = 5
MAX_PLAN_CYCLES = 2
SPEECH_POSITIONS = (
    "first constructive",
    "second constructive",
    "summary",
    "final focus",
    "crossfire",
    "grand crossfire",
    "rebuttal",
)

EVIDENCE_NOUN = re.compile(
    r"\b(?:evidence|cards?|documents?|files?|sources?|authors?)\b",
    re.I,
)
INGEST_REQUEST = re.compile(
    r"\b(?:ingest|import|upload)\b|"
    r"\b(?:add|load|store|save|put|parse|process|attach(?:ed)?)\b[^.;]{0,60}"
    r"\b(?:evidence|cards?|documents?|files?|sources?)\b|"
    r"\b(?:evidence|cards?|documents?|files?|sources?)\b[^.;]{0,45}"
    r"\b(?:ingest|import|upload|add|load|parse|process|attach)\b",
    re.I,
)
RETRIEVE_REQUEST = re.compile(
    r"\b(?:find|search|show|retrieve|get|locate|pull)\b[^.;]{0,90}"
    r"\b(?:evidence|cards?|sources?|authors?)\b|"
    r"\blook\s+up\b[^.;]{0,90}\b(?:evidence|cards?|sources?|authors?)\b|"
    r"\b(?:what|which|whose|who)\b[^.;]{0,90}"
    r"\b(?:evidence|cards?|sources?|authors?)\b|"
    r"\b(?:evidence|cards?|sources?)\b[^.;]{0,45}"
    r"\b(?:about|on|for|against|supporting)\b|"
    r"\b(?:what\s+do\s+we\s+have|do\s+we\s+have)\b[^.;]{0,90}"
    r"\b(?:evidence|cards?|sources?)\b",
    re.I,
)
INTEGRITY_REQUEST = re.compile(
    r"\b(?:write|draft|make up|invent|fabricate|repair|correct)\b[^.;]{0,100}"
    r"\b(?:speech|case|cards?|evidence|citation)\b|"
    r"\bgenerate\b(?:(?!\b(?:and|then)\b|[.;]).){0,100}"
    r"\b(?:speech|case|cards?|evidence|citation)\b|"
    r"\b(?:fake|fabricated|invented|made-up)\b[^.;]{0,60}"
    r"\b(?:cards?|evidence|citation)\b",
    re.I,
)
COACH_SIMULATION_REQUEST = re.compile(
    r"\b(?:coach\s+me|coach\s+simulation|simulat(?:e|ed)\s+(?:a\s+)?coach|"
    r"act\s+as\s+(?:my\s+|a\s+)?(?:debate\s+)?coach|"
    r"practice\s+with\s+(?:my\s+|a\s+|the\s+)?coach|"
    r"start\s+(?:a\s+)?coach(?:ing)?\s+session|mock\s+coach|"
    r"give\s+me\s+debate\s+feedback|drill\s+me\s+like\s+a\s+coach)\b",
    re.I,
)
CURRENT_TOPIC_REQUEST = re.compile(
    r"\b(?:current|latest|new|official|this\s+month(?:'s)?)\b.{0,80}"
    r"\b(?:topic|resolution)\b|"
    r"\b(?:topic|resolution)\b.{0,80}"
    r"\b(?:current|latest|right\s+now|this\s+month)\b|"
    r"\bwhat(?:'s|\s+is)\b.{0,60}"
    r"\b(?:public\s+forum|pf|lincoln[-\s]douglas|ld|nsda)?\s*"
    r"(?:topic|resolution)\b|"
    r"\b(?:public\s+forum|pf|lincoln[-\s]douglas|ld|nsda)\b.{0,40}"
    r"\b(?:topic|resolution)\b",
    re.I | re.S,
)

QUERY_STOPWORDS = {
    "about", "against", "and", "any", "are", "card", "cards", "clarification",
    "con", "do", "does", "evidence", "find", "for", "from", "get", "have",
    "help", "locate", "look", "me", "on", "please", "pro", "pull", "request",
    "retrieve", "search", "show", "side", "some", "source", "sources",
    "supporting", "the", "user", "want", "what", "which", "whose", "with",
}


def _normalize_intent_text(text: str) -> str:
    """Repair a narrow class of joined determiners without rewriting user content."""

    return re.sub(
        r"\b(some|any|the)(?=(?:evidence|cards?|sources?|documents?|files?)\b)",
        r"\1 ",
        text,
        flags=re.I,
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


def _topic_event(text: str) -> str:
    if re.search(r"\b(?:lincoln[-\s]douglas|ld)\b", text, re.I):
        return "Lincoln-Douglas"
    return "Public Forum"


def _topic_as_of(text: str) -> str | None:
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return match.group(0) if match else None


def _confirmation_token(text: str) -> str | None:
    match = re.search(r"\b[0-9a-f]{32}\b", text, re.I)
    return match.group(0).lower() if match else None


def _refine_evidence_query(text: str) -> str:
    """Remove request boilerplate while preserving topic-bearing search terms."""

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text.lower())
    meaningful: list[str] = []
    for word in words:
        if word in QUERY_STOPWORDS or len(word) < 3 or word in meaningful:
            continue
        meaningful.append(word)
    return " ".join(meaningful[:16])


def _matches_refined_topic(query: str, card: dict[str, Any]) -> bool:
    query_terms = set(_refine_evidence_query(query).split())
    card_text = " ".join(
        str(card.get(field, ""))
        for field in ("header", "tag", "body", "cite_full")
    ).lower()
    card_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", card_text))
    return bool(query_terms & card_terms)


def _bounded_span_text(
    text: str,
    spans: list[list[int]],
    *,
    limit: int = 1_200,
) -> str:
    fragments: list[str] = []
    length = 0
    for raw in spans:
        if len(raw) != 2:
            continue
        start = max(0, min(int(raw[0]), len(text)))
        end = max(start, min(int(raw[1]), len(text)))
        fragment = text[start:end].strip()
        if not fragment:
            continue
        remaining = limit - length
        if remaining <= 0:
            break
        fragments.append(fragment[:remaining])
        length += len(fragments[-1]) + 1
    return " ".join(fragments).strip()


def _grounding_card(card: dict[str, Any]) -> dict[str, Any]:
    body = str(card.get("body", ""))
    header = str(card.get("header", "")).strip()
    citation = str(card.get("cite_full", "")).strip()
    citation_label = citation.split(",", 1)[0].strip()[:200]
    read_spans = [list(span) for span in card.get("read_spans", [])]
    emphasis_spans = [list(span) for span in card.get("emphasis_spans", [])]
    return {
        "header": header,
        "source_label": (
            header or citation_label or str(card.get("id", "Evidence card"))[:200]
        ),
        "cite_full": citation,
        "tag": str(card.get("tag", "")),
        "body": body,
        "read_spans": read_spans,
        "emphasis_spans": emphasis_spans,
        "read_excerpt": _bounded_span_text(body, read_spans),
        "emphasis_excerpt": _bounded_span_text(body, emphasis_spans),
    }


def _compound_actions(text: str, state: AgentState) -> list[str]:
    """Return explicit, independently requested actions in their textual order."""

    if not re.search(r"(?:;|,|\band\b|\bthen\b)", text, re.I):
        return []
    if deterministic_classification(text, state)["intent"] in {
        "integrity_refusal",
        "ingest_cards",
    }:
        return []

    candidates: list[tuple[int, str]] = []

    def add(action: str, match: re.Match[str] | None) -> None:
        if match is not None:
            candidates.append((match.start(), action))

    evidence = re.search(
        r"\b(?:find|search|show|retrieve|get|what)\b[^.;]{0,80}\b(?:evidence|cards?)\b"
        r"|\b(?:evidence|cards?)\b[^.;]{0,40}\b(?:find|search|show|retrieve|get)\b",
        text,
        re.I,
    )
    rule = re.search(
        r"\b(?:explain|find|show|what|which)\b[^.;]{0,80}"
        r"\b(?:rule|legal|allowed|citation format)\b",
        text,
        re.I,
    )
    drill = re.search(r"\b(?:drill|practice)\b", text, re.I)
    assessment_write = re.search(
        r"\b(?:log|record|add)\b[^.;]{0,40}\b(?:assessment|feedback|progress record)\b",
        text,
        re.I,
    )
    progress_read = re.search(
        r"\b(?:show|get|read|what)\b[^.;]{0,50}"
        r"\b(?:progress|assessment|weakness|feedback|record)\b",
        text,
        re.I,
    )
    schedule = re.search(
        r"\b(?:schedule|book)\b[^.;]{0,60}\b(?:session|meeting|calendar)?",
        text,
        re.I,
    )
    current_topic = CURRENT_TOPIC_REQUEST.search(text)

    add("search_cards", evidence)
    add("search_rules", rule)
    add("current_topic", current_topic)
    add("generate_drill", drill)
    if assessment_write is not None and state.get("role") == "coach":
        add("log_assessment", assessment_write)
    else:
        add("progress", progress_read)
    add("schedule_session", schedule)

    ordered: list[str] = []
    for _, action in sorted(candidates):
        if action not in ordered:
            ordered.append(action)
    return ordered if len(ordered) >= 2 else []


def _compound_missing(
    actions: list[str],
    parameters: dict[str, Any],
    state: AgentState,
) -> list[str]:
    missing: list[str] = []
    if any(action in {"search_cards", "generate_drill"} for action in actions):
        if parameters.get("side") == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    if (
        "schedule_session" in actions
        and not parameters.get("start")
        and not parameters.get("confirmation_token")
    ):
        missing.append("start time in ISO format")
    return list(dict.fromkeys(missing))


def _required_clarification(
    intent: str,
    parameters: dict[str, Any],
    state: AgentState,
) -> str:
    """Return a concise question for parameters required by a resolved intent."""

    missing: list[str] = []
    if intent == "retrieve_evidence":
        if parameters.get("side") == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "generate_drill":
        if parameters.get("side") == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "coach_simulation":
        if not parameters.get("speech_position"):
            missing.append("speech position")
        if parameters.get("side") == "unknown":
            missing.append("side (Pro or Con)")
        if not state.get("resolution"):
            missing.append("resolution")
    elif intent == "ingest_cards":
        needs_file = not parameters.get("file_path") and not parameters.get(
            "confirmation_token"
        )
        if needs_file:
            question = "Please attach or provide the DOCX file to preview."
            if not state.get("resolution"):
                question += " Also provide the active resolution."
            return question
        if not state.get("resolution"):
            missing.append("resolution")
    elif (
        intent == "schedule_session"
        and not parameters.get("start")
        and not parameters.get("confirmation_token")
    ):
        missing.append("start time in ISO format")
    return "Please provide " + ", ".join(missing) + "." if missing else ""


def _ambiguous_question(
    kind: str,
    parameters: dict[str, Any],
    state: AgentState,
) -> str:
    if kind == "evidence_action":
        return (
            "Do you want to search existing evidence or import a DOCX file? "
            "For a search, include Pro or Con. Importing requires an attached "
            "DOCX file."
        )
    if kind == "docx_action":
        return "Do you want to preview this DOCX file for evidence ingestion?"
    if kind == "side_action":
        side = str(parameters.get("side", "")).title()
        return (
            f"What would you like to do for the {side} side: search evidence "
            "or generate a drill?"
        )
    if kind == "add_object":
        return (
            "What would you like to add: a DOCX evidence file or a coach "
            "assessment?"
        )
    return (
        "What would you like to do: search evidence, import a DOCX file, check "
        "a rule, look up the current topic, generate a drill, view progress, or "
        "schedule a session?"
    )


def deterministic_classification(text: str, state: AgentState) -> dict[str, Any]:
    classification_text = _normalize_intent_text(text)
    lowered = classification_text.lower()
    parameters = {
        "side": _side(classification_text),
        "student_id": _student_id(classification_text),
        "speech_position": _speech_position(classification_text),
        "file_path": _file_path(classification_text),
        "confirmation_token": _confirmation_token(classification_text),
        "start": _start(classification_text),
    }
    ingest_request = bool(INGEST_REQUEST.search(classification_text))
    retrieve_request = bool(RETRIEVE_REQUEST.search(classification_text))
    ambiguity = ""
    if (
        parameters["confirmation_token"]
        and re.search(r"\bconfirm\b", lowered)
        and re.search(r"\b(?:calendar|schedule|session|meeting)\b", lowered)
    ):
        intent = "schedule_session"
    elif parameters["confirmation_token"] and re.search(r"\bconfirm\b", lowered):
        intent = "ingest_cards"
    elif INTEGRITY_REQUEST.search(classification_text):
        intent = "integrity_refusal"
    elif ingest_request and not retrieve_request:
        intent = "ingest_cards"
    elif COACH_SIMULATION_REQUEST.search(classification_text):
        intent = "coach_simulation"
    elif CURRENT_TOPIC_REQUEST.search(classification_text) and not retrieve_request:
        intent = "current_topic"
    elif "drill" in lowered or "practice" in lowered:
        intent = "generate_drill"
    elif re.search(r"\b(?:progress|assessment|weakness|feedback|record)\b", lowered):
        intent = "progress"
    elif re.search(r"\b(?:rule|legal|allowed|evidence violation|cite|citation format)\b", lowered):
        intent = "explain_rule"
    elif re.search(r"\b(?:schedule|book|calendar|meeting|session)\b", lowered):
        intent = "schedule_session"
    elif retrieve_request and not ingest_request:
        intent = "retrieve_evidence"
    else:
        intent = "unknown"

    if intent == "unknown":
        if ingest_request and retrieve_request:
            ambiguity = "evidence_action"
        elif EVIDENCE_NOUN.search(classification_text):
            ambiguity = "evidence_action"
        elif parameters["file_path"]:
            ambiguity = "docx_action"
        elif parameters["side"] != "unknown":
            ambiguity = "side_action"
        elif re.search(r"\b(?:add|create|make)\b", lowered):
            ambiguity = "add_object"
        else:
            ambiguity = "general"

    question = (
        _ambiguous_question(ambiguity, parameters, state)
        if ambiguity
        else _required_clarification(intent, parameters, state)
    )
    return {
        "intent": intent,
        **parameters,
        "clarification_needed": bool(question),
        "clarification_question": question,
        "_ambiguity": ambiguity or None,
    }


@dataclass
class AgentNodes:
    tools: CaseFileTools
    llm: AnthropicJSONClient
    security_audit: SecurityAuditor
    rate_limiter: RateLimiter

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
            "security_events": list(state.get("security_events", [])),
        }

    def screen_request(self, state: AgentState) -> AgentState:
        text = _message(state)
        decision = inspect_text(text, trust="untrusted_user")
        if not self.rate_limiter.allow(
            f"agent:{state.get('user_id', '')}:{state.get('role', '')}"
        ):
            decision = GuardDecision(
                risk="high",
                action="block",
                signals=["request_rate_limit"],
                safe_for_model=False,
                safe_for_write_tools=False,
                trust="untrusted_user",
            )
        event = {
            "event": "request_screened",
            "risk": decision.risk,
            "action": decision.action,
            "signals": decision.signals,
        }
        self.security_audit.record(
            "request_screened",
            decision=decision,
            request_id=state.get("request_id"),
            user_id=state.get("user_id"),
            raw_text=text,
            details={"role": state.get("role"), "resolution": state.get("resolution")},
        )
        return {
            "security_decision": decision.to_dict(),
            "security_events": [*state.get("security_events", []), event],
        }

    def block_prompt_injection(self, state: AgentState) -> AgentState:
        signals = state.get("security_decision", {}).get("signals", [])
        response = (
            "[RATE_LIMITED] Request rate limit exceeded."
            if "request_rate_limit" in signals
            else BLOCKED_RESPONSE
        )
        return {
            "intent": "unknown",
            "response": response,
            "tool_trace": list(state.get("tool_trace", [])),
        }

    def classify_intent(self, state: AgentState) -> AgentState:
        text = _message(state)
        classified = deterministic_classification(text, state)
        deterministic_intent = classified["intent"]
        ambiguity = classified.pop("_ambiguity", None)
        decision = state.get("security_decision", {})
        if (
            self.llm.available
            and deterministic_intent == "unknown"
            and ambiguity == "general"
            and decision.get("safe_for_model", True)
        ):
            try:
                value = self.llm.complete_json(
                    system=CLASSIFIER_SYSTEM.format(
                        tools=", ".join(sorted(self.tools.names_for_role(state["role"])))
                    ),
                    user=json.dumps(
                        {
                            "trust": "untrusted_user",
                            "length": len(text),
                            "content": text,
                        },
                        ensure_ascii=False,
                    ),
                    max_tokens=500,
                    schema=ClassifierOutput,
                )
                value = ClassifierOutput.model_validate(value).model_dump(mode="json")
                # A model may resolve an unknown read-only request, but cannot create
                # write authority or override a deterministic policy/safety route.
                if (
                    deterministic_intent == "unknown"
                    and value.get("intent")
                    in {
                        "retrieve_evidence",
                        "explain_rule",
                        "generate_drill",
                        "coach_simulation",
                        "progress",
                        "current_topic",
                        "integrity_refusal",
                        "unknown",
                    }
                ):
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
            except (LLMUnavailable, LLMResponseError, ValidationError):
                pass
        intent = classified.pop("intent")
        clarification_needed = bool(classified.pop("clarification_needed", False))
        question = str(classified.pop("clarification_question", "") or "")
        # A model may resolve generic language, but required fields remain
        # deterministic and cannot be waived by model output.
        required_question = _required_clarification(intent, classified, state)
        if required_question:
            clarification_needed = True
            question = required_question
        elif intent != "unknown":
            clarification_needed = False
            question = ""
        elif not question:
            clarification_needed = True
            question = _ambiguous_question("general", classified, state)
        compound_missing = _compound_missing(
            _compound_actions(text, state), classified, state
        )
        if compound_missing:
            clarification_needed = True
            question = "Please provide " + ", ".join(compound_missing) + "."
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
            "current_topic": "current_topic",
            "generate_drill": "generate_drill",
            "coach_simulation": "coach_simulation",
            "progress": "progress",
            "ingest_cards": "ingest_cards",
            "schedule_session": "schedule_session",
            "integrity_refusal": "integrity_refusal",
        }
        action = mapping.get(state.get("intent", "unknown"), "unknown")
        decision = state.get("security_decision", {})
        message = _message(state)
        actions = _compound_actions(message, state) or [action]
        if (
            actions == ["progress"]
            and state.get("role") == "coach"
            and re.search(r"\b(?:log|record|add)\b", message, re.I)
        ):
            actions = ["log_assessment"]
        if "log_assessment" in actions and "schedule_session" in actions:
            actions = [
                "log_assessment",
                *[
                    candidate
                    for candidate in actions
                    if candidate not in {"log_assessment", "schedule_session"}
                ],
                "schedule_session",
            ]
        write_request = any(
            candidate in {"log_assessment", "ingest_cards", "schedule_session"}
            for candidate in actions
        )
        if write_request and not decision.get("safe_for_write_tools", True):
            actions = ["security_block"]
            action = "security_block"

        task_ids = {
            "search_cards": "evidence",
            "search_rules": "rules",
            "current_topic": "current_topic",
            "generate_drill": "drill",
            "coach_simulation": "coach_simulation",
            "progress": "progress",
            "log_assessment": "assessment",
            "ingest_cards": "ingest",
            "schedule_session": "scheduling",
            "integrity_refusal": "integrity",
            "security_block": "security",
            "unknown": "unknown",
        }
        tasks = []
        for candidate in actions:
            dependencies = (
                ["assessment"]
                if candidate == "schedule_session" and "log_assessment" in actions
                else []
            )
            tasks.append(
                make_task(
                    task_id=task_ids[candidate],
                    action=candidate,  # type: ignore[arg-type]
                    arguments=self._arguments_for_action(candidate, state),
                    depends_on=dependencies,
                    confirmation_required=(
                        candidate == "ingest_cards"
                        or (
                            candidate == "schedule_session"
                            and not self.tools.settings.mock_calendar
                        )
                    ),
                )
            )
        plan = TaskPlan(tasks=tasks)
        return {
            "next_action": action,
            "task_plan": plan.model_dump(mode="python"),
        }

    def _arguments_for_action(
        self, action: str, state: AgentState
    ) -> dict[str, Any]:
        params = state.get("parameters", {})
        message = _message(state)
        student_id = params.get("student_id") or state["user_id"]
        if action == "search_cards":
            return {"query": message, "side": params["side"], "n": 3}
        if action == "search_rules":
            return {"question": message, "n": 3}
        if action == "current_topic":
            return {
                "event": _topic_event(message),
                "as_of": _topic_as_of(message),
            }
        if action == "generate_drill":
            return {
                "student_id": student_id,
                "speech_position": params.get("speech_position") or "general",
                "resolution": state.get("resolution"),
                "side": params.get("side", "unknown"),
            }
        if action == "coach_simulation":
            return {
                "student_id": student_id,
                "speech_position": params.get("speech_position") or "",
                "side": params.get("side", "unknown"),
                "message": message,
                "use_model": state.get("security_decision", {}).get(
                    "safe_for_model", True
                ),
            }
        if action == "progress":
            return {"student_id": student_id}
        if action == "log_assessment":
            assessment = message.split(":", 1)[-1].strip()
            assessment = re.split(
                r"\s*(?:;|\band\b|\bthen\b)\s*(?=(?:schedule|book)\b)",
                assessment,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
            weaknesses = re.search(
                r"weakness(?:es)?\s*[:=]\s*([^.;]+)", message, re.I
            )
            return {
                "student_id": student_id,
                "speech_position": params.get("speech_position") or "unspecified",
                "resolution": state.get("resolution", ""),
                "weakness_tags": (
                    [part.strip() for part in weaknesses.group(1).split(",")]
                    if weaknesses
                    else []
                ),
                "assessment_text": assessment,
                "idempotency_key": state.get("request_id"),
            }
        if action == "ingest_cards":
            return {
                "file_path": params.get("file_path"),
                "resolution": state.get("resolution"),
                "side": (
                    params.get("side")
                    if params.get("side") in {"pro", "con"}
                    else None
                ),
                "dry_run": not bool(params.get("confirmation_token")),
                "confirmation_token": params.get("confirmation_token"),
                "idempotency_key": state.get("request_id"),
            }
        if action == "schedule_session":
            duration = re.search(r"\b(\d{2,3})\s*minutes?\b", message, re.I)
            email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", message)
            return {
                "student_id": student_id,
                "start": params.get("start") or "",
                "duration_minutes": int(duration.group(1)) if duration else 45,
                "attendee_email": email.group(0) if email else None,
                "confirmation_token": params.get("confirmation_token"),
                "idempotency_key": state.get("request_id"),
            }
        return {}

    def execute_tools(self, state: AgentState) -> AgentState:
        if int(state.get("iterations", 0)) >= MAX_ITERATIONS:
            return {"tool_result": "[STOPPED] maximum agent iterations reached."}
        context = ToolContext(
            role=state["role"],
            user_id=state["user_id"],
            resolution=state.get("resolution", ""),
        )
        plan = TaskPlan.model_validate(state.get("task_plan"))
        outcomes = execute_task_plan(
            plan,
            lambda task: self._execute_task(task, context),
        )
        tasks = {task.id: task for task in plan.tasks}
        trace_entries = [
            {
                "tool": outcome.action,
                "arguments": self._trace_arguments(
                    tasks[outcome.id].arguments
                ),
                "result_type": type(outcome.result).__name__,
                "status": outcome.status,
                "attempts": outcome.attempts,
                "depends_on": outcome.depends_on,
            }
            for outcome in outcomes
        ]
        trace = [*state.get("tool_trace", []), *trace_entries]
        task_results = [outcome.to_dict() for outcome in outcomes]
        result: Any = (
            outcomes[0].result
            if len(outcomes) == 1
            else {outcome.id: outcome.result for outcome in outcomes}
        )
        return {
            "tool_result": result,
            "tool_trace": trace,
            "task_results": task_results,
        }

    def observe_results(self, state: AgentState) -> AgentState:
        """Inspect tool outcomes and make one bounded, read-only recovery attempt."""

        cycle = int(state.get("plan_cycles", 0)) + 1
        observations = list(state.get("observations", []))
        results = state.get("task_results", [])
        plan = TaskPlan.model_validate(state.get("task_plan"))
        task = plan.tasks[0]
        filtered_results: list[dict[str, Any]] | None = None
        if (
            task.id == "evidence_refined"
            and len(results) == 1
            and results[0].get("action") == "search_cards"
            and results[0].get("status") == "success"
            and isinstance(results[0].get("result"), list)
        ):
            candidates = results[0]["result"]
            query = str(task.arguments.get("query", ""))
            grounded = [
                card
                for card in candidates
                if isinstance(card, dict) and _matches_refined_topic(query, card)
            ]
            if len(grounded) != len(candidates):
                updated = {**results[0], "result": grounded}
                filtered_results = [updated]
                results = filtered_results
                observations.append(
                    {
                        "cycle": cycle,
                        "outcome": "off_topic_retry_filtered",
                        "action": "finish",
                        "detail": (
                            "Discarded refined-query candidates without explicit "
                            "topic overlap."
                        ),
                    }
                )
        if (
            cycle < MAX_PLAN_CYCLES
            and len(results) == 1
            and results[0].get("action") == "search_cards"
            and results[0].get("status") == "success"
            and results[0].get("result") == []
        ):
            original_query = str(task.arguments.get("query", ""))
            refined_query = _refine_evidence_query(original_query)
            if refined_query and refined_query.casefold() != original_query.casefold():
                revised = make_task(
                    task_id="evidence_refined",
                    action="search_cards",
                    arguments={**task.arguments, "query": refined_query},
                )
                observations.append(
                    {
                        "cycle": cycle,
                        "outcome": "empty_retrieval",
                        "action": "replan",
                        "detail": (
                            "Removed request boilerplate and retried the same "
                            "resolution and side."
                        ),
                    }
                )
                return {
                    "task_plan": TaskPlan(tasks=[revised]).model_dump(mode="python"),
                    "plan_cycles": cycle,
                    "observations": observations,
                    "next_step": "replan",
                }
        finished: AgentState = {
            "plan_cycles": cycle,
            "observations": observations,
            "next_step": "finish",
        }
        if filtered_results is not None:
            finished.update(
                {
                    "task_results": filtered_results,
                    "tool_result": filtered_results[0]["result"],
                }
            )
        return finished

    def _execute_task(self, task: TaskSpec, context: ToolContext) -> Any:
        arguments = task.arguments
        if task.action == "search_cards":
            return self.tools.search_cards(context, **arguments)
        if task.action == "search_rules":
            return self.tools.search_rules(context, **arguments)
        if task.action == "current_topic":
            return self.tools.get_current_topic(context, **arguments)
        if task.action == "generate_drill":
            return self.tools.generate_drill(context, **arguments)
        if task.action == "coach_simulation":
            return self._simulate_coach(context, **arguments)
        if task.action == "progress":
            return self.tools.get_progress(context, **arguments)
        if task.action == "log_assessment":
            return self.tools.log_assessment(context, **arguments)
        if task.action == "ingest_cards":
            return self.tools.ingest_cards(context, **arguments)
        if task.action == "schedule_session":
            return self.tools.schedule_session(context, **arguments)
        if task.action == "security_block":
            return BLOCKED_RESPONSE
        if task.action == "integrity_refusal":
            return {
                "refusal": "CaseFile cannot fabricate citations, repair or alter evidence text, or write a competition speech.",
                "alternative": "I can retrieve intact cited evidence, explain an on-file rule, or create a drill that requires you to do the speaking.",
            }
        return "I couldn't determine an action."

    def _simulate_coach(
        self,
        context: ToolContext,
        *,
        student_id: str,
        speech_position: str,
        side: str,
        message: str,
        use_model: bool,
    ) -> dict[str, Any]:
        progress = self.tools.get_progress(context, student_id=student_id)
        records = progress if isinstance(progress, list) else []
        latest = records[-1] if records else {}
        weakness_tags = [
            str(tag) for tag in latest.get("weakness_tags", []) if str(tag).strip()
        ][:4]
        focus = weakness_tags[0] if weakness_tags else speech_position
        cards = self.tools.search_cards(
            context,
            query=" ".join(weakness_tags) or f"{speech_position} evidence",
            side=side,
            n=2,
        )
        card_list = cards if isinstance(cards, list) else []
        card_refs = [_grounding_card(card) for card in card_list]
        latest_reply = message.rsplit("User clarification:", 1)[-1].strip()
        turn: dict[str, Any] | None = None
        if self.llm.available and use_model:
            try:
                turn = self.llm.complete_json(
                    system=COACH_SIMULATOR_SYSTEM,
                    user=json.dumps(
                        {
                            "student_message": latest_reply,
                            "speech_position": speech_position,
                            "side": side,
                            "weakness_tags": weakness_tags,
                            "grounded_cards": [
                                {
                                    "header": card["header"],
                                    "source_label": card["source_label"],
                                    "citation": card["cite_full"],
                                    "tag": card["tag"],
                                    "read_excerpt": card["read_excerpt"],
                                    "emphasis_excerpt": card["emphasis_excerpt"],
                                    "body_excerpt": card["body"][:900],
                                }
                                for card in card_refs
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    max_tokens=500,
                    schema=CoachTurnOutput,
                )
            except (LLMUnavailable, LLMResponseError, ValidationError):
                turn = None
        if turn is None:
            continuing = "User clarification:" in message
            turn = {
                "focus": focus,
                "feedback": (
                    f"Keep this turn focused on {focus}; make the reasoning explicit "
                    "instead of relying on an asserted conclusion."
                    if continuing
                    else (
                        f"We’ll practice your {speech_position} speech. I’ll ask "
                        "questions, and you do the debating."
                        if focus == speech_position
                        else f"We’ll practice {focus} in your {speech_position} speech. "
                        "I’ll ask questions, and you do the debating."
                    )
                ),
                "question": (
                    "Pressure-test your last answer: what is its weakest warrant, and "
                    "how would you defend it against the other side?"
                    if continuing
                    else "Give me your claim, cited evidence, and warrant in 30 seconds. "
                    "Why should the judge prioritize it?"
                ),
            }
        return {
            "label": "Simulated coach",
            **turn,
            "student_id": student_id,
            "speech_position": speech_position,
            "side": side,
            "card_refs": card_refs,
            "continue_session": True,
        }

    def _format_evidence_argument(
        self,
        cards: list[dict[str, Any]],
        state: AgentState,
    ) -> dict[str, Any]:
        grounded = [_grounding_card(card) for card in cards]
        source_labels = {card["source_label"] for card in grounded}
        decision = state.get("security_decision", {})
        if self.llm.available and decision.get("safe_for_model", True):
            try:
                value = self.llm.complete_json(
                    system=EVIDENCE_ARGUMENT_SYSTEM,
                    user=json.dumps(
                        {
                            "request": _message(state),
                            "side": state.get("parameters", {}).get("side"),
                            "resolution": state.get("resolution"),
                            "grounded_cards": [
                                {
                                    "header": card["source_label"],
                                    "citation": card["cite_full"],
                                    "tag": card["tag"],
                                    "read_excerpt": card["read_excerpt"],
                                    "emphasis_excerpt": card["emphasis_excerpt"],
                                    "body_excerpt": card["body"][:900],
                                }
                                for card in grounded
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    max_tokens=700,
                    schema=EvidenceArgumentOutput,
                )
                if set(value["citations_used"]).issubset(source_labels):
                    return {**value, "generated_by": "model"}
            except (LLMUnavailable, LLMResponseError, ValidationError, KeyError):
                pass

        first = grounded[0]
        excerpt = (
            first["read_excerpt"]
            or first["emphasis_excerpt"]
            or first["body"][:500]
        )
        return {
            "claim": first["tag"]
            or "The retrieved evidence supports the requested position.",
            "warrant": "Retrieved marked excerpt: " + excerpt,
            "impact": (
                "Use this evidence to make the comparison explicit; the retrieved "
                "text does not by itself decide how the judge should prioritize it."
            ),
            "citations_used": [first["source_label"]],
            "generated_by": "deterministic",
        }

    @staticmethod
    def _render_evidence_argument(argument: dict[str, Any]) -> str:
        label = (
            "AI-generated grounded argument"
            if argument.get("generated_by") == "model"
            else "Grounded argument outline"
        )
        return "\n".join(
            [
                f"[{label}]",
                f"Claim: {argument['claim']}",
                f"Warrant: {argument['warrant']}",
                f"Impact: {argument['impact']}",
                "Cards used: " + ", ".join(argument["citations_used"]),
            ]
        )

    def call_model(self, state: AgentState) -> AgentState:
        """Render only grounded tool output; do not answer from model memory."""
        task_results = state.get("task_results", [])
        if len(task_results) == 1 and task_results[0].get("status") == "success":
            action = task_results[0].get("action")
            result = task_results[0].get("result")
            if action == "search_cards" and isinstance(result, list) and result:
                argument = self._format_evidence_argument(result, state)
                return {
                    "response": (
                        self._render_evidence_argument(argument)
                        + "\n\nRetrieved evidence:\n\n"
                        + self._render_action(action, result, state)
                    ),
                    "evidence_argument": argument,
                    "grounding_cards": [_grounding_card(card) for card in result],
                }
            if action == "coach_simulation" and isinstance(result, dict):
                return {
                    "response": self._render_action(action, result, state),
                    "coach_turn": {
                        key: result[key]
                        for key in ("label", "focus", "feedback", "question")
                    },
                    "grounding_cards": list(result.get("card_refs", [])),
                }
        if len(task_results) > 1:
            labels = {
                "search_cards": "Evidence",
                "search_rules": "Rules",
                "current_topic": "Current topic",
                "generate_drill": "Drill",
                "coach_simulation": "Coach simulation",
                "progress": "Progress",
                "log_assessment": "Assessment",
                "ingest_cards": "Ingestion",
                "schedule_session": "Scheduling",
            }
            sections = []
            for item in task_results:
                rendered = (
                    str(item["result"])
                    if item["status"] != "success"
                    else self._render_action(item["action"], item["result"], state)
                )
                sections.append(
                    f"{labels.get(item['action'], item['id'].replace('_', ' ').title())}:\n"
                    f"{rendered}"
                )
            return {"response": "\n\n".join(sections)}

        action = (
            task_results[0]["action"]
            if task_results
            else state.get("next_action")
        )
        result = (
            task_results[0]["result"]
            if task_results
            else state.get("tool_result")
        )
        return {"response": self._render_action(action, result, state)}

    def _render_action(
        self, action: str | None, result: Any, state: AgentState
    ) -> str:
        if isinstance(result, str):
            return result
        if action == "search_cards":
            if not result:
                params = state.get("parameters", {})
                return (
                    f"No card on file for the {params.get('side', 'requested')} side of "
                    f"{state.get('resolution', 'the active resolution')}. I won't substitute "
                    "a low-similarity card from another topic."
                )
            blocks = []
            for card in result:
                blocks.append(
                    f"{card['cite_full']}\n\n{card['header']}\n{card['tag']}\n{card['body']}".strip()
                )
            return "\n\n---\n\n".join(blocks)
        if action == "search_rules":
            if not result:
                return "No authoritative rule is on file. I won't answer this from model memory; ask a coach to load the current NSDA rule documents."
            return "\n\n".join(
                f"Per section {chunk['section_number']} ({chunk['section_title']}) in "
                f"{chunk['document']}:\n{chunk['text']}"
                for chunk in result
            )
        if action == "current_topic":
            topic = result["topic"]
            provenance = (
                "Synthetic NSDA-compatible fixture — not an official NSDA publication"
                if result.get("synthetic")
                else str(result.get("provider", "NSDA-compatible provider"))
            )
            lines = [
                f"[{provenance}]",
                str(topic["resolution"]),
                f"Event: {topic['event']}",
                f"Effective: {topic['effective_from']} through {topic['effective_to']}",
            ]
            if topic.get("source_ref"):
                lines.append(f"Source: {topic['source_ref']}")
            if result.get("disclaimer"):
                lines.append(f"Note: {result['disclaimer']}")
            return "\n".join(lines)
        if action == "generate_drill":
            lines = [*result["instructions"]]
            for card in result["card_refs"]:
                lines.append(f"- {card['header']}: {card['cite_full']}")
            return "\n".join(lines)
        if action == "coach_simulation":
            lines = [
                f"[{result['label']} — {result['focus']}]",
                "",
                result["feedback"],
                "",
                result["question"],
            ]
            if result.get("card_refs"):
                lines.extend(["", "Practice evidence:"])
                for index, card in enumerate(result["card_refs"], start=1):
                    lines.extend(
                        [
                            "",
                            f"Evidence {index}: {card['cite_full']}",
                            card["header"] or card["source_label"],
                            card["tag"],
                            card["body"],
                        ]
                    )
            lines.extend(["", "Reply to continue, or type ‘end coaching’."])
            return "\n".join(lines)
        if action == "progress":
            if isinstance(result, list):
                return (
                    "No progress records found."
                    if not result
                    else json.dumps(result, ensure_ascii=False, indent=2)
                )
            return json.dumps(result, ensure_ascii=False, indent=2)
        if action == "log_assessment":
            return f"Assessment logged for {result['student_id']}."
        if action == "ingest_cards":
            return result.get("summary", json.dumps(result, indent=2))
        if action == "schedule_session":
            if result.get("confirmation_required"):
                return result["summary"]
            mode = "mock calendar" if result.get("mock") else "Google Calendar"
            return f"Session scheduled in {mode}: {result['start']['dateTime']} (event {result['id']})."
        if action == "integrity_refusal":
            return f"{result['refusal']} {result['alternative']}"
        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def _trace_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in {"query", "question", "assessment_text"} and isinstance(value, str):
                safe[key] = summarize_untrusted_text(value)
            elif "token" in key.lower() or "idempotency" in key.lower():
                safe[key] = "[REDACTED]" if value else value
            else:
                safe[key] = value
        return safe

    def should_continue(self, state: AgentState) -> AgentState:
        iterations = int(state.get("iterations", 0)) + 1
        if iterations > MAX_ITERATIONS:
            return {
                "iterations": iterations,
                "response": "I stopped after the maximum number of tool iterations.",
            }
        return {"iterations": iterations}
