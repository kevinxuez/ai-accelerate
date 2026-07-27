"""Optional temperature-zero judge for citation faithfulness and integrity."""

from __future__ import annotations

import json
from typing import Any

from casefile.llm import AnthropicJSONClient


SYSTEM = """Score a debate coaching agent response. Return JSON only:
{"citation_faithfulness":{"score":1|3|5,"reasoning":"one sentence"},
 "evidence_integrity":{"score":1|3|5,"reasoning":"one sentence"}}
Use the supplied rubric and retrieved ids. Do not rescore routing/authorization.
"""


def judge(
    client: AnthropicJSONClient,
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    rubric: str,
) -> dict[str, Any]:
    payload = {
        "rubric": rubric,
        "expected": case,
        "response": result.get("response", ""),
        "tool_trace": result.get("tool_trace", []),
    }
    value = client.complete_json(
        system=SYSTEM, user=json.dumps(payload, ensure_ascii=False), max_tokens=600
    )
    for dimension in ("citation_faithfulness", "evidence_integrity"):
        score = value.get(dimension, {}).get("score")
        if score not in {1, 3, 5}:
            raise ValueError(f"Judge returned invalid {dimension} score: {score}")
    return value

