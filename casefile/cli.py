"""Command-line chat client."""

from __future__ import annotations

import argparse

from casefile.agent.graph import CaseFileAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["student", "coach"], default="student")
    parser.add_argument("--user-id", default="student-1")
    parser.add_argument("--resolution", required=True)
    parser.add_argument("message", nargs="*")
    args = parser.parse_args()
    agent = CaseFileAgent()
    if args.message:
        result = agent.ask(
            " ".join(args.message),
            role=args.role,
            user_id=args.user_id,
            resolution=args.resolution,
        )
        print(result["response"])
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
        result = agent.ask(
            message,
            role=args.role,
            user_id=args.user_id,
            resolution=args.resolution,
        )
        print(result["response"])


if __name__ == "__main__":
    main()

