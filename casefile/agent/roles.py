"""Role policy. Tool functions still perform their own authorization checks."""

from __future__ import annotations


TOOLS_BY_ROLE = {
    "student": {
        "search_cards",
        "search_rules",
        "generate_drill",
        "get_progress",
        "schedule_session",
    },
    "coach": {
        "search_cards",
        "search_rules",
        "generate_drill",
        "log_assessment",
        "get_progress",
        "ingest_cards",
        "schedule_session",
    },
}


def available_tools(role: str) -> set[str]:
    return set(TOOLS_BY_ROLE.get(role, set()))


def denial(role: str, action: str) -> str:
    return f"[DENIED] role '{role}' cannot {action}."

