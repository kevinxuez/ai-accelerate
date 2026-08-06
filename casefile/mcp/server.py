"""Expose validated, read-oriented CaseFile tools over a stdio MCP server."""

from __future__ import annotations

from typing import Literal

from casefile.agents.contracts import EvidencePacket
from casefile.agents.evidence_librarian import EvidenceLibrarian
from casefile.agents.skills_coach import SkillsCoach
from casefile.config import get_settings
from casefile.llm import build_anthropic_client
from casefile.security.audit import RateLimiter
from casefile.tools import CaseFileTools, ToolContext

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Install CaseFile with the 'mcp' extra to run this server"
    ) from exc


mcp = FastMCP("CaseFile")
settings = get_settings()
model = build_anthropic_client(settings)
tools = CaseFileTools(settings, model=model)
rate_limiter = RateLimiter(max(1, settings.requests_per_minute))


def _limited(user_id: str) -> bool:
    return not rate_limiter.allow(f"mcp:{user_id}")


def _enforce_rate_limit(user_id: str) -> None:
    if _limited(user_id):
        from casefile.agents.errors import CaseFileError, ErrorCode

        raise CaseFileError(
            ErrorCode.RATE_LIMITED,
            "The MCP request rate limit was exceeded.",
            stage="mcp.rate_limit",
        )


@mcp.tool()
def search_cards(
    query: str,
    side: Literal["pro", "con"],
    resolution: str,
    role: Literal["student", "coach"],
    user_id: str,
    n: int = 5,
):
    """Use this to retrieve intact cited PF evidence for one side and resolution."""
    _enforce_rate_limit(user_id)
    return tools.registry.invoke(
        "evidence_librarian",
        "search_cards",
        ToolContext(role, user_id, resolution, agent="evidence_librarian"),
        {
            "query": query,
            "side": side,
            "resolution": resolution,
            "n": n,
            "source_files": [],
        },
    )


@mcp.tool()
def generate_drill(
    student_id: str,
    speech_position: str,
    resolution: str,
    side: Literal["pro", "con"],
    role: Literal["student", "coach"],
    user_id: str,
):
    """Build a model-generated drill through Coach-to-Librarian handoffs."""
    _enforce_rate_limit(user_id)
    coach_context = ToolContext(
        role,
        user_id,
        resolution,
        agent="skills_coach",
    )
    coach = SkillsCoach(tools, model)
    progress = coach.summarize_progress(coach_context, student_id=student_id)
    request = coach.request_evidence(
        coach_context,
        student_id=student_id,
        speech_position=speech_position,
        side=side,
        focus=f"{speech_position} practice",
        intended_use="drill",
        progress_summary=progress,
    )
    packet = EvidenceLibrarian(tools, model).retrieve_evidence(
        ToolContext(
            role,
            user_id,
            resolution,
            agent="evidence_librarian",
        ),
        request=request.request_summary,
        requested_side=side,
    )
    if not isinstance(packet, EvidencePacket):
        return packet.model_dump(mode="json")
    return coach.generate_drill(
        coach_context,
        student_id=student_id,
        speech_position=speech_position,
        side=side,
        focus=f"{speech_position} practice",
        progress_summary=progress,
        evidence_packet=packet,
    ).model_dump(mode="json")


@mcp.tool()
def search_rules(
    question: str,
    resolution: str,
    role: Literal["student", "coach"],
    user_id: str,
    n: int = 3,
):
    """Retrieve only indexed, non-quarantined Public Forum rule text."""
    _enforce_rate_limit(user_id)
    return tools.registry.invoke(
        "evidence_librarian",
        "search_rules",
        ToolContext(role, user_id, resolution, agent="evidence_librarian"),
        {"question": question, "n": n},
    )


@mcp.tool()
def get_progress(
    student_id: str,
    resolution: str,
    role: Literal["student", "coach"],
    user_id: str,
):
    """Read progress subject to the same ownership checks as the API."""
    _enforce_rate_limit(user_id)
    return tools.registry.invoke(
        "skills_coach",
        "get_progress",
        ToolContext(role, user_id, resolution, agent="skills_coach"),
        {"student_id": student_id},
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
