"""Expose validated, read-oriented CaseFile tools over a stdio MCP server."""

from __future__ import annotations

from typing import Literal

from casefile.agent.tools import CaseFileTools, ToolContext
from casefile.config import get_settings
from casefile.security.audit import RateLimiter

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install CaseFile with the 'mcp' extra to run this server") from exc


mcp = FastMCP("CaseFile")
tools = CaseFileTools()
rate_limiter = RateLimiter(max(1, get_settings().requests_per_minute))


def _limited(user_id: str) -> bool:
    return not rate_limiter.allow(f"mcp:{user_id}")


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
    if _limited(user_id):
        return "[RATE_LIMITED] MCP request rate limit exceeded."
    return tools.search_cards(
        ToolContext(role, user_id, resolution), query, side, resolution, n
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
    """Use this to build a grounded practice drill from progress and indexed cards."""
    if _limited(user_id):
        return "[RATE_LIMITED] MCP request rate limit exceeded."
    return tools.generate_drill(
        ToolContext(role, user_id, resolution),
        student_id,
        speech_position,
        resolution,
        side,
    )


@mcp.tool()
def search_rules(
    question: str,
    resolution: str,
    role: Literal["student", "coach"],
    user_id: str,
    n: int = 3,
):
    """Retrieve only indexed, non-quarantined Public Forum rule text."""
    if _limited(user_id):
        return "[RATE_LIMITED] MCP request rate limit exceeded."
    return tools.search_rules(
        ToolContext(role, user_id, resolution),
        question,
        n,
    )


@mcp.tool()
def get_progress(
    student_id: str,
    resolution: str,
    role: Literal["student", "coach"],
    user_id: str,
):
    """Read progress subject to the same ownership checks as the API."""
    if _limited(user_id):
        return "[RATE_LIMITED] MCP request rate limit exceeded."
    return tools.get_progress(
        ToolContext(role, user_id, resolution),
        student_id,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
