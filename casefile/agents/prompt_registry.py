"""Versioned prompt assets used by the four-agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from typing import Literal

from .contracts import AgentName
from .errors import CaseFileError, ErrorCode


@dataclass(frozen=True)
class PromptAsset:
    agent: AgentName
    filename: str
    version: str
    content: str

    @property
    def template_name(self) -> str:
        return f"agents/prompts/{self.filename}"


PromptName = Literal[
    "supervisor",
    "evidence_librarian",
    "evidence_librarian_boundaries",
    "evidence_librarian_labels",
    "argument_strategist",
    "skills_coach",
]

PROMPTS: dict[PromptName, tuple[AgentName, str, str]] = {
    "supervisor": (
        "supervisor",
        "supervisor.md",
        "1",
    ),
    "evidence_librarian": (
        "evidence_librarian",
        "evidence_librarian.md",
        "1",
    ),
    "evidence_librarian_boundaries": (
        "evidence_librarian",
        "evidence_librarian_boundaries.md",
        "1",
    ),
    "evidence_librarian_labels": (
        "evidence_librarian",
        "evidence_librarian_labels.md",
        "1",
    ),
    "argument_strategist": (
        "argument_strategist",
        "argument_strategist.md",
        "1",
    ),
    "skills_coach": (
        "skills_coach",
        "skills_coach.md",
        "1",
    ),
}


@lru_cache(maxsize=None)
def load_prompt(name: PromptName) -> PromptAsset:
    """Load a required packaged prompt or raise a configuration error."""

    definition = PROMPTS.get(name)
    if definition is None:
        raise CaseFileError(
            ErrorCode.CONFIGURATION_ERROR,
            f"No prompt is registered with name '{name}'.",
            stage="prompt_registry.load",
        )
    agent, filename, version = definition
    try:
        content = (
            resources.files("casefile.agents")
            .joinpath("prompts", filename)
            .read_text(encoding="utf-8")
            .strip()
        )
    except (FileNotFoundError, OSError) as exc:
        raise CaseFileError(
            ErrorCode.CONFIGURATION_ERROR,
            f"The required prompt asset for '{agent}' is unavailable.",
            stage="prompt_registry.load",
            agent=agent,
            cause=exc,
        ) from exc
    if not content:
        raise CaseFileError(
            ErrorCode.CONFIGURATION_ERROR,
            f"The required prompt asset for '{agent}' is empty.",
            stage="prompt_registry.load",
            agent=agent,
        )
    return PromptAsset(agent, filename, version, content)
