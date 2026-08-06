"""Command-line client for the four-agent CaseFile runtime."""

from __future__ import annotations

import argparse
import json

from casefile.agents.runtime import CaseFileRuntime
from casefile.agents.state import CaseFileState


def _print_state(state: CaseFileState) -> None:
    if state.status == "failed":
        print(
            json.dumps(
                {
                    "status": state.status,
                    "request_id": state.request.request_id,
                    "session_id": state.request.session_id,
                    "error": state.error.model_dump(mode="json")
                    if state.error
                    else None,
                    "agent_trace": [
                        entry.model_dump(mode="json") for entry in state.agent_trace
                    ],
                    "tool_trace": [
                        entry.model_dump(mode="json") for entry in state.tool_trace
                    ],
                    "model_trace": [
                        entry.model_dump(mode="json") for entry in state.model_trace
                    ],
                },
                indent=2,
            )
        )
        return
    response = next(
        message.content
        for message in reversed(state.messages)
        if message.role == "assistant"
    )
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["student", "coach"], default="student")
    parser.add_argument("--user-id", default="student-1")
    parser.add_argument("--resolution", required=True)
    parser.add_argument("message", nargs="*")
    args = parser.parse_args()
    runtime = CaseFileRuntime()
    if args.message:
        result = runtime.ask(
            " ".join(args.message),
            role=args.role,
            user_id=args.user_id,
            resolution=args.resolution,
        )
        _print_state(result)
        return
    print(f"CaseFile ({args.role}, {args.user_id}, {args.resolution}); Ctrl-D to exit")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            break
        if not message:
            continue
        result = runtime.ask(
            message,
            role=args.role,
            user_id=args.user_id,
            resolution=args.resolution,
        )
        _print_state(result)


if __name__ == "__main__":
    main()
