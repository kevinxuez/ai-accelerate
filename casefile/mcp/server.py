"""Expose card search and drill generation over a stdio MCP server."""

from __future__ import annotations

from casefile.agent.tools import CaseFileTools, ToolContext

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install CaseFile with the 'mcp' extra to run this server") from exc


mcp = FastMCP("CaseFile")
tools = CaseFileTools()


@mcp.tool()
def search_cards(
    query: str,
    side: str,
    resolution: str,
    role: str,
    user_id: str,
    n: int = 5,
):
    """Use this to retrieve intact cited PF evidence for one side and resolution."""
    return tools.search_cards(
        ToolContext(role, user_id, resolution), query, side, resolution, n
    )


@mcp.tool()
def generate_drill(
    student_id: str,
    speech_position: str,
    resolution: str,
    side: str,
    role: str,
    user_id: str,
):
    """Use this to build a grounded practice drill from progress and indexed cards."""
    return tools.generate_drill(
        ToolContext(role, user_id, resolution),
        student_id,
        speech_position,
        resolution,
        side,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

