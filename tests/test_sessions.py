from __future__ import annotations

import pytest

from casefile.agent.graph import CaseFileAgent


def test_clarification_session_resumes_original_task(isolated_settings) -> None:
    agent = CaseFileAgent(isolated_settings)
    session_id = "session-clarify-0001"

    first = agent.ask(
        "Build a drill.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert first["awaiting_clarification"] is True
    assert "side (Pro or Con)" in first["response"]
    assert list(isolated_settings.sessions_dir.glob("*.json"))

    completed = agent.ask(
        "Pro",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert completed["intent"] == "generate_drill"
    assert completed["resumed_from_clarification"] is True
    assert completed["awaiting_clarification"] is False
    assert completed["tool_trace"][0]["tool"] == "generate_drill"
    assert list(isolated_settings.sessions_dir.glob("*.json")) == []


def test_session_is_bound_to_role_user_and_resolution(isolated_settings) -> None:
    agent = CaseFileAgent(isolated_settings)
    session_id = "session-context-00001"
    agent.ask(
        "Build a drill.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )

    with pytest.raises(ValueError, match="session context"):
        agent.ask(
            "summary Pro",
            role="coach",
            user_id="coach-1",
            resolution="R1",
            session_id=session_id,
        )


def test_explicit_new_intent_replaces_pending_clarification(isolated_settings) -> None:
    agent = CaseFileAgent(isolated_settings)
    session_id = "session-replace-00001"
    agent.ask(
        "Build a drill.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )

    result = agent.ask(
        "Show my progress.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert result["intent"] == "progress"
    assert result["resumed_from_clarification"] is False
    assert result["awaiting_clarification"] is False


def test_simulated_coach_continues_in_student_session_and_can_end(
    isolated_settings,
) -> None:
    agent = CaseFileAgent(isolated_settings)
    session_id = "coach-simulation-0001"

    first = agent.ask(
        "Coach me through a Pro summary speech.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert first["intent"] == "coach_simulation"
    assert first["coach_simulation_active"] is True
    assert "[Simulated coach" in first["response"]
    assert "you do the debating" in first["response"]

    follow_up = agent.ask(
        "My claim is that regulation protects consumers.",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert follow_up["intent"] == "coach_simulation"
    assert follow_up["resumed_from_clarification"] is True
    assert follow_up["coach_simulation_active"] is True
    assert "weakest warrant" in follow_up["response"]

    ended = agent.ask(
        "end coaching",
        role="student",
        user_id="student-1",
        resolution="R1",
        session_id=session_id,
    )
    assert ended["response"] == "Coach simulation ended."
    assert ended["coach_simulation_active"] is False
    assert list(isolated_settings.sessions_dir.glob("*.json")) == []
