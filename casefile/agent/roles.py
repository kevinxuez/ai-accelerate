"""Role policy. Tool functions still perform their own authorization checks."""

from __future__ import annotations


TOOLS_BY_ROLE = {
    "student": {
        "search_cards",
        "generate_argument",
        "search_rules",
        "get_current_topic",
        "generate_drill",
        "coach_simulation",
        "get_progress",
        "ingest_cards",
        "schedule_session",
    },
    "coach": {
        "search_cards",
        "generate_argument",
        "search_rules",
        "get_current_topic",
        "generate_drill",
        "coach_simulation",
        "log_assessment",
        "get_progress",
        "ingest_cards",
        "approve_quarantined_card",
        "schedule_session",
    },
}


def available_tools(role: str) -> set[str]:
    return set(TOOLS_BY_ROLE.get(role, set()))


def denial(role: str, action: str) -> str:
    return f"[DENIED] role '{role}' cannot {action}."
