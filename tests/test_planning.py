from __future__ import annotations

import json
import threading

import pytest
from pydantic import ValidationError

from casefile.agent.graph import CaseFileAgent
from casefile.agent.planning import (
    TaskPlan,
    TransientTaskError,
    execute_task_plan,
    make_task,
)


def test_task_plan_rejects_cycles_and_parallelizes_independent_reads():
    with pytest.raises(ValidationError, match="cycle"):
        TaskPlan(
            tasks=[
                make_task(
                    task_id="one",
                    action="progress",
                    arguments={},
                    depends_on=["two"],
                ),
                make_task(
                    task_id="two",
                    action="search_rules",
                    arguments={},
                    depends_on=["one"],
                ),
            ]
        )

    barrier = threading.Barrier(2)
    thread_names: set[str] = set()
    plan = TaskPlan(
        tasks=[
            make_task(task_id="one", action="progress", arguments={}),
            make_task(task_id="two", action="search_rules", arguments={}),
        ]
    )

    def runner(task):
        thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=2)
        return {"task": task.id}

    outcomes = execute_task_plan(plan, runner)
    assert [outcome.status for outcome in outcomes] == ["success", "success"]
    assert len(thread_names) == 2


def test_read_retry_is_bounded_and_failed_dependency_skips_write():
    attempts = 0

    def transient_runner(task):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientTaskError("temporary")
        return {"ok": True}

    retry_plan = TaskPlan(
        tasks=[make_task(task_id="read", action="search_rules", arguments={})]
    )
    retried = execute_task_plan(retry_plan, transient_runner)
    assert retried[0].status == "success"
    assert retried[0].attempts == 2

    calls: list[str] = []
    dependency_plan = TaskPlan(
        tasks=[
            make_task(
                task_id="assessment",
                action="log_assessment",
                arguments={},
            ),
            make_task(
                task_id="scheduling",
                action="schedule_session",
                arguments={},
                depends_on=["assessment"],
            ),
        ]
    )

    def failing_runner(task):
        calls.append(task.id)
        return "[INVALID] assessment_text is too long."

    outcomes = execute_task_plan(dependency_plan, failing_runner)
    assert calls == ["assessment"]
    assert [outcome.status for outcome in outcomes] == ["failed", "skipped"]
    assert outcomes[1].attempts == 0


def test_compound_assessment_failure_does_not_schedule(isolated_settings):
    agent = CaseFileAgent(isolated_settings)
    oversized_tag = "comparison framing prioritization weighing extension sequencing"
    result = agent.ask(
        "Log assessment for student-a summary: Needs work. "
        f"weaknesses: {oversized_tag}; and schedule a session at "
        "2026-08-04T16:00.",
        role="coach",
        user_id="coach-1",
        resolution="R1",
    )

    assert [item["action"] for item in result["task_trace"]] == [
        "log_assessment",
        "schedule_session",
    ]
    assert [item["status"] for item in result["task_trace"]] == [
        "failed",
        "skipped",
    ]
    assert not (isolated_settings.data_dir / "calendar_events.json").exists()
    assert json.loads(isolated_settings.progress_path.read_text(encoding="utf-8")) == []
    assert "[SKIPPED]" in result["response"]


def test_compound_assessment_success_and_independent_read_plan(isolated_settings):
    agent = CaseFileAgent(isolated_settings)
    written = agent.ask(
        "Log assessment for student-a summary: Needs cleaner collapse. "
        "weaknesses: collapse; and schedule a session at 2026-08-04T16:00.",
        role="coach",
        user_id="coach-1",
        resolution="R1",
    )
    assert [item["status"] for item in written["task_trace"]] == [
        "success",
        "success",
    ]
    assert json.loads(
        (isolated_settings.data_dir / "calendar_events.json").read_text(
            encoding="utf-8"
        )
    )

    reads = agent.ask(
        "Show me my progress and give me a summary drill for the Pro side.",
        role="student",
        user_id="student-a",
        resolution="R1",
    )
    assert {item["action"] for item in reads["task_trace"]} == {
        "progress",
        "generate_drill",
    }
    assert {item["status"] for item in reads["task_trace"]} == {"success"}
    assert "Progress:" in reads["response"]
    assert "Drill:" in reads["response"]


def test_empty_evidence_result_triggers_one_bounded_refined_search(
    isolated_settings,
):
    agent = CaseFileAgent(isolated_settings)
    queries: list[str] = []

    def search_cards(context, query, side, resolution=None, n=5):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [
            {
                "cite_full": "Example Author. Example Source. 2026.",
                "header": "Refined result",
                "tag": "Consumer payment protections reduce losses.",
                "body": "Grounded card text about consumer protections.",
            }
        ]

    agent.tools.search_cards = search_cards
    result = agent.ask(
        "Find Pro evidence about consumer payment protections.",
        role="student",
        user_id="student-1",
        resolution="R1",
    )

    assert len(queries) == 2
    assert queries[1] == "consumer payment protections"
    assert result["plan_cycles"] == 2
    assert result["observations"][0]["action"] == "replan"
    assert [item["tool"] for item in result["tool_trace"]] == [
        "search_cards",
        "search_cards",
    ]
    assert "Refined result" in result["response"]
