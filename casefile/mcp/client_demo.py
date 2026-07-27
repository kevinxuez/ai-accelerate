"""Minimal stdio MCP client demonstration."""

from __future__ import annotations

import asyncio
import json
import sys


async def run() -> None:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise SystemExit("Install CaseFile with the 'mcp' extra") from exc
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "casefile.mcp.server"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_cards",
                {
                    "query": "regulatory certainty",
                    "side": "pro",
                    "resolution": "2026-09-CRYPTO",
                    "role": "student",
                    "user_id": "student-1",
                    "n": 3,
                },
            )
            print(json.dumps([item.model_dump() for item in result.content], indent=2))


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
