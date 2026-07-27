"""Run the offline heuristic boundary pass and write scorer-compatible JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundary_pass import heuristic_boundary_pass
from .serialize_index import paragraph_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.docx).resolve()
    result = heuristic_boundary_pass(paragraph_records(source))
    result["source_file"] = source.name
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

