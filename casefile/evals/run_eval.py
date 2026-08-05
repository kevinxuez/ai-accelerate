"""Run the 20-item golden set once and record per-dimension scores."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from casefile.agent.graph import CaseFileAgent
from casefile.config import Settings, get_settings
from casefile.llm import build_anthropic_client
from casefile.retrieval import CaseFileIndex

from .judge import judge


HERE = Path(__file__).resolve().parent


def _fixture_settings(root: Path) -> Settings:
    data = root / "data"
    rules = root / "rules"
    chroma = root / "chroma"
    data.mkdir(parents=True)
    rules.mkdir(parents=True)
    card = {
        "id": "eval-card-1",
        "header": "Evaluation '26",
        "tag": "Regulatory certainty protects consumers.",
        "cite_full": "Evaluation Organization, 2026, Synthetic Evaluation Source.",
        "author": "Evaluation Organization",
        "author_type": "organization",
        "year": 2026,
        "date_raw": "",
        "source": "Synthetic Evaluation Source",
        "url": "https://example.invalid/eval",
        "date_accessed": "",
        "cutter": "",
        "body": "Synthetic fixture evidence about regulatory certainty and consumer protection.",
        "read_spans": [[0, 78]],
        "emphasis_spans": [],
        "marking_convention": "bold",
        "evidence_type": "quoted",
        "source_text_present": True,
        "resolution": "EVAL-RES",
        "resolution_confidence": "high",
        "side": "pro",
        "topic_tags": ["regulatory", "certainty", "consumer", "protection"],
        "ingest_status": "ok",
        "flags": [],
        "source_file": "synthetic-eval.docx",
        "source_paragraphs": [1, 2, 3],
        "embedding_text": "Evaluation '26\nRegulatory certainty consumer protection",
        "returned_document": "Evaluation Organization, 2026, Synthetic Evaluation Source.\nEvaluation '26\nSynthetic fixture evidence about regulatory certainty and consumer protection."
    }
    (data / "cards_labeled.json").write_text(json.dumps([card], indent=2) + "\n", encoding="utf-8")
    (data / "progress.json").write_text(json.dumps([{
        "student_id":"student-a","date":"2026-07-20","speech_position":"summary",
        "resolution":"EVAL-RES","weakness_tags":["collapse"],
        "assessment_text":"Practice collapse and comparison.","author_role":"coach","author_id":"coach-1"
    }], indent=2) + "\n", encoding="utf-8")
    (rules / "eval_rules.md").write_text(
        "# Synthetic evaluation rules\n\n## 7.2 Evidence Integrity\n"
        "For this synthetic fixture, evidence must remain attached to its citation.\n",
        encoding="utf-8",
    )
    base = get_settings()
    return replace(
        base,
        data_dir=data,
        chroma_dir=chroma,
        rules_dir=rules,
        anthropic_api_key=None,
        mock_calendar=True,
    )


def _deterministic_scores(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    trace = result.get("tool_trace", [])
    actual_tool = trace[0]["tool"] if trace else None
    route_ok = (
        result.get("intent") == case["expected_intent"]
        and actual_tool == case.get("expected_tool")
    )
    content_ok = case.get("response_contains", "").lower() in result.get("response", "").lower()
    is_retrieval = case["expected_intent"] in {"retrieve_evidence", "explain_rule"}
    citation_score = 5 if (not is_retrieval or content_ok) else 1
    routing_score = 5 if route_ok and content_ok else 1
    integrity_score = 5 if content_ok else 1
    return {
        "citation_faithfulness": {"score": citation_score, "reasoning": "Grounded/empty-result behavior matched." if citation_score == 5 else "Expected grounded content was absent."},
        "routing_authorization": {"score": routing_score, "reasoning": "Intent, tool, and response matched." if routing_score == 5 else "Intent, tool, or authorization response differed."},
        "evidence_integrity": {"score": integrity_score, "reasoning": "Integrity expectation matched." if integrity_score == 5 else "Expected integrity behavior was absent."},
    }


def run(*, use_llm_judge: bool = False, judge_repeats: int = 3) -> dict[str, Any]:
    cases = json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))
    rubric = (HERE / "rubric.md").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="casefile-eval-") as directory:
        settings = _fixture_settings(Path(directory))
        index = CaseFileIndex(settings)
        index.rebuild_cards()
        index.rebuild_rules()
        agent = CaseFileAgent(settings)
        llm = build_anthropic_client(get_settings())
        outputs = []
        for case in cases:
            result = agent.ask(
                case["query"],
                role=case["role"],
                user_id=case["user_id"],
                resolution="EVAL-RES",
            )
            scores = _deterministic_scores(case, result)
            judge_method = "deterministic"
            if use_llm_judge and llm.available:
                if judge_repeats < 1:
                    raise ValueError("judge_repeats must be positive")
                judgments = [
                    judge(llm, case=case, result=result, rubric=rubric)
                    for _ in range(judge_repeats)
                ]
                score_pairs = {
                    (
                        item["citation_faithfulness"]["score"],
                        item["evidence_integrity"]["score"],
                    )
                    for item in judgments
                }
                if len(score_pairs) != 1:
                    raise ValueError(
                        f"LLM judge was inconsistent for {case['id']}: {sorted(score_pairs)}"
                    )
                judged = judgments[0]
                scores["citation_faithfulness"] = judged["citation_faithfulness"]
                scores["evidence_integrity"] = judged["evidence_integrity"]
                judge_method = f"llm-{judge_repeats}x-consistent+deterministic-routing"
            outputs.append({
                "id": case["id"], "category": case["category"],
                "response": result["response"], "intent": result["intent"],
                "tool_trace": result["tool_trace"], "scores": scores,
                "judge_method": judge_method,
            })
    dimensions = ("citation_faithfulness", "routing_authorization", "evidence_integrity")
    averages = {
        dimension: round(sum(item["scores"][dimension]["score"] for item in outputs) / len(outputs), 3)
        for dimension in dimensions
    }
    return {"cases": len(outputs), "averages": averages, "results": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--output", default=str(HERE / "eval_results.json"))
    args = parser.parse_args()
    result = run(use_llm_judge=args.llm_judge, judge_repeats=args.judge_repeats)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": result["cases"], "averages": result["averages"]}, indent=2))


if __name__ == "__main__":
    main()
