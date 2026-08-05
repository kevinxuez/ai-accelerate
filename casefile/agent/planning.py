"""Strict task plans and a small dependency-aware execution engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskAction = Literal[
    "search_cards",
    "search_rules",
    "generate_drill",
    "coach_simulation",
    "progress",
    "log_assessment",
    "ingest_cards",
    "schedule_session",
    "integrity_refusal",
    "security_block",
    "unknown",
]
TaskMode = Literal["read", "write", "policy"]
TaskStatus = Literal["success", "failed", "skipped"]

ACTION_MODES: dict[str, TaskMode] = {
    "search_cards": "read",
    "search_rules": "read",
    "generate_drill": "read",
    "coach_simulation": "read",
    "progress": "read",
    "log_assessment": "write",
    "ingest_cards": "write",
    "schedule_session": "write",
    "integrity_refusal": "policy",
    "security_block": "policy",
    "unknown": "policy",
}

FAILURE_PREFIXES = (
    "[BLOCKED",
    "[DENIED]",
    "[INVALID]",
    "[RATE_LIMITED]",
    "[STOPPED]",
)


class StrictPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TaskSpec(StrictPlanModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    action: TaskAction
    mode: TaskMode
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    max_attempts: int = Field(default=1, ge=1, le=3)
    confirmation_required: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> "TaskSpec":
        expected = ACTION_MODES[self.action]
        if self.mode != expected:
            raise ValueError(
                f"task action {self.action!r} must use mode {expected!r}"
            )
        if self.mode != "read" and self.max_attempts != 1:
            raise ValueError("write and policy tasks cannot be retried automatically")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("task dependencies must be unique")
        return self


class TaskPlan(StrictPlanModel):
    tasks: list[TaskSpec] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskPlan":
        by_id = {task.id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("task ids must be unique")
        for task in self.tasks:
            unknown = set(task.depends_on) - set(by_id)
            if unknown:
                raise ValueError(
                    f"task {task.id!r} has unknown dependencies: {sorted(unknown)}"
                )
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id!r} cannot depend on itself")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task dependencies contain a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in by_id[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in by_id:
            visit(task_id)
        return self


@dataclass(frozen=True)
class TaskOutcome:
    id: str
    action: str
    status: TaskStatus
    attempts: int
    depends_on: list[str]
    result: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "status": self.status,
            "attempts": self.attempts,
            "depends_on": list(self.depends_on),
            "result": self.result,
        }


class TransientTaskError(RuntimeError):
    """Signal that a read-only task may be retried within its configured bound."""


def result_failed(result: Any) -> bool:
    return isinstance(result, str) and result.startswith(FAILURE_PREFIXES)


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, (TransientTaskError, TimeoutError, ConnectionError))


def _run_one(
    task: TaskSpec,
    runner: Callable[[TaskSpec], Any],
) -> TaskOutcome:
    for attempt in range(1, task.max_attempts + 1):
        try:
            result = runner(task)
        except Exception as exc:  # The outcome is rendered without leaking exception text.
            if task.mode == "read" and attempt < task.max_attempts and _is_transient(exc):
                continue
            return TaskOutcome(
                id=task.id,
                action=task.action,
                status="failed",
                attempts=attempt,
                depends_on=task.depends_on,
                result=f"[TASK_FAILED] {task.action} failed ({type(exc).__name__}).",
            )
        return TaskOutcome(
            id=task.id,
            action=task.action,
            status="failed" if result_failed(result) else "success",
            attempts=attempt,
            depends_on=task.depends_on,
            result=result,
        )
    raise AssertionError("task attempt loop did not return")


def execute_task_plan(
    plan: TaskPlan,
    runner: Callable[[TaskSpec], Any],
    *,
    max_parallel_reads: int = 4,
) -> list[TaskOutcome]:
    """Execute a validated DAG.

    Independent read-only tasks in the same dependency wave run concurrently. Writes and
    policy decisions are always serial, and every task with an unsuccessful dependency is
    skipped without calling ``runner``.
    """

    if max_parallel_reads < 1:
        raise ValueError("max_parallel_reads must be positive")
    pending = {task.id: task for task in plan.tasks}
    outcomes: dict[str, TaskOutcome] = {}

    while pending:
        skipped = [
            task
            for task in pending.values()
            if any(
                dependency in outcomes
                and outcomes[dependency].status != "success"
                for dependency in task.depends_on
            )
        ]
        for task in skipped:
            failed_dependencies = [
                dependency
                for dependency in task.depends_on
                if outcomes[dependency].status != "success"
            ]
            outcomes[task.id] = TaskOutcome(
                id=task.id,
                action=task.action,
                status="skipped",
                attempts=0,
                depends_on=task.depends_on,
                result=(
                    "[SKIPPED] prerequisite task(s) did not succeed: "
                    + ", ".join(failed_dependencies)
                    + "."
                ),
            )
            pending.pop(task.id)

        ready = [
            task
            for task in pending.values()
            if all(dependency in outcomes for dependency in task.depends_on)
        ]
        if not ready:
            if pending:
                raise ValueError("task plan could not make progress")
            break

        read_tasks = [task for task in ready if task.mode == "read"]
        serial_tasks = [task for task in ready if task.mode != "read"]

        if len(read_tasks) == 1:
            task = read_tasks[0]
            outcomes[task.id] = _run_one(task, runner)
            pending.pop(task.id)
        elif read_tasks:
            workers = min(max_parallel_reads, len(read_tasks))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="casefile-read"
            ) as executor:
                futures = {
                    executor.submit(_run_one, task, runner): task for task in read_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    outcomes[task.id] = future.result()
                    pending.pop(task.id)

        for task in serial_tasks:
            outcomes[task.id] = _run_one(task, runner)
            pending.pop(task.id)

    return [outcomes[task.id] for task in plan.tasks]


def make_task(
    *,
    task_id: str,
    action: TaskAction,
    arguments: dict[str, Any],
    depends_on: list[str] | None = None,
    confirmation_required: bool = False,
) -> TaskSpec:
    mode = ACTION_MODES[action]
    return TaskSpec(
        id=task_id,
        action=action,
        mode=mode,
        arguments=arguments,
        depends_on=depends_on or [],
        max_attempts=2 if mode == "read" else 1,
        confirmation_required=confirmation_required,
    )
