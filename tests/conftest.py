from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from casefile.config import (
    PINNED_EMBEDDING_DIMENSIONS,
    PINNED_EMBEDDING_MODEL,
    get_settings,
)
from casefile.agents.contracts import (
    AssessmentProposal,
    BoundaryOutput,
    CardLabelOutput,
    CoachTurn,
    DrillPlan,
    EvidenceQueryPlan,
    EvidenceRequest,
    ProgressSummary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]{1,}")


def _test_embedding(text: str) -> list[float]:
    words = [word.lower() for word in TOKEN.findall(text)]
    terms = words + [f"{left}_{right}" for left, right in zip(words, words[1:])]
    vector = [0.0] * PINNED_EMBEDDING_DIMENSIONS
    for term in terms:
        value = int.from_bytes(
            hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        vector[value % len(vector)] += -1.0 if value & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class _TestEmbedder:
    name = PINNED_EMBEDDING_MODEL
    dimensions = PINNED_EMBEDDING_DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_test_embedding(text) for text in texts]


class _TestModel:
    model = "test-model"

    def validate_configuration(self) -> None:
        return None

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        schema = kwargs.get("schema")
        payload = json.loads(kwargs["user"])
        if schema is ProgressSummary:
            weakness_tags = sorted(
                {
                    tag
                    for record in payload["records"]
                    for tag in record.get("weakness_tags", [])
                }
            )
            return {
                "artifact_type": "progress_summary",
                "student_id": payload["student_id"],
                "records": payload["records"],
                "summary": (
                    "No recorded progress history is available."
                    if not payload["records"]
                    else (
                        f"Progress for {payload['student_id']}: "
                        f"focus on {', '.join(weakness_tags)}."
                    )
                ),
            }
        if schema is EvidenceRequest:
            return {
                "request_summary": f"Find {payload['required_side']} evidence for {payload['focus']}.",
                "resolution": payload["active_resolution"],
                "side": payload["required_side"],
                "subject": payload["focus"],
                "entities": [],
                "source_files": payload["allowed_source_files"],
                "intended_use": payload["intended_use"],
            }
        if schema is DrillPlan:
            packet = payload.get("evidence_packet") or {}
            return {
                "artifact_type": "drill_plan",
                "student_id": payload["student_id"],
                "speech_position": payload["speech_position"],
                "resolution": payload["active_resolution"],
                "side": payload["required_side"],
                "title": "Claim-evidence-warrant practice",
                "focus": [payload["focus"]],
                "instructions": [
                    (
                        "Complete a general claim-evidence-warrant drill; you do the speaking."
                        if payload["speech_position"] == "general"
                        else f"Prepare a timed {payload['speech_position']} drill; you do the speaking."
                    )
                ],
                "duration_minutes": 10,
                "evidence_card_ids": [
                    card["card_id"] for card in packet.get("cards", [])
                ],
                "personalization_summary": payload["progress_summary"]["summary"],
            }
        if schema is CoachTurn:
            continuing = bool(payload.get("prior_turns"))
            packet = payload.get("evidence_packet") or {}
            return {
                "artifact_type": "coach_turn",
                "label": "simulated_coach",
                "student_id": payload["student_id"],
                "speech_position": payload["speech_position"],
                "side": payload["required_side"],
                "focus": str(payload.get("focus") or "argument clarity"),
                "feedback": (
                    "I will coach the structure; you do the debating."
                    if not continuing
                    else "Your claim is clear; now test its internal link."
                ),
                "question": (
                    "What is your first warrant?"
                    if not continuing
                    else "What is the weakest warrant in that chain?"
                ),
                "evidence_card_ids": [
                    card["card_id"] for card in packet.get("cards", [])
                ],
                "continue_session": True,
            }
        if schema is AssessmentProposal:
            return {
                "artifact_type": "assessment_proposal",
                "student_id": payload["student_id"],
                "speech_position": payload["speech_position"],
                "resolution": payload["active_resolution"],
                "weakness_tags": ["warrant clarity"],
                "assessment_text": "The student should make the internal link explicit.",
                "confirmation_required": True,
            }
        if schema is EvidenceQueryPlan:
            request = str(payload["request"])
            side = payload.get("requested_side")
            if side is None:
                return {
                    "resolution": payload["active_resolution"],
                    "side": None,
                    "subject": "",
                    "entities": [],
                    "source_files": [],
                    "queries": [],
                    "result_limit": 10,
                    "clarification_needed": True,
                    "clarification_question": "Which side should I search?",
                }
            confirmed = payload.get("confirmed_available_files", [])
            return {
                "resolution": payload["active_resolution"],
                "side": side,
                "subject": request[:1000],
                "entities": [],
                "source_files": [
                    filename
                    for filename in confirmed
                    if filename.lower() in request.lower()
                ],
                "queries": [request],
                "result_limit": 10,
                "clarification_needed": False,
                "clarification_question": None,
            }
        if schema is BoundaryOutput:
            ids = {
                int(value)
                for value in re.findall(
                    r"^\[\s*(\d+)\]", payload["paragraph_index"], re.MULTILINE
                )
            }
            expected = {
                1,
                2,
                3,
                4,
                5,
                7,
                9,
                10,
                11,
                12,
                14,
                15,
                16,
                18,
                20,
                21,
                22,
                24,
                25,
                26,
                28,
                29,
                30,
                32,
                33,
                34,
            }
            if ids != expected:
                raise AssertionError(f"No boundary fixture for paragraph ids {ids}")
            return {
                "junk": [1],
                "cards": [
                    {"header": 2, "tag": [], "cite": [3], "body": [4], "flags": []},
                    {
                        "header": None,
                        "tag": [],
                        "cite": [5],
                        "body": [7],
                        "flags": ["no_header"],
                    },
                    {
                        "header": 9,
                        "tag": [],
                        "cite": [10, 11],
                        "body": [12],
                        "flags": [],
                    },
                    {
                        "header": 14,
                        "tag": [],
                        "cite": [15, 16],
                        "body": [18],
                        "flags": [],
                    },
                    {"header": 20, "tag": [], "cite": [21], "body": [22], "flags": []},
                    {
                        "header": 24,
                        "tag": [],
                        "cite": [25, 26],
                        "body": [26],
                        "flags": ["cite_body_same_paragraph"],
                    },
                    {"header": 28, "tag": [], "cite": [29], "body": [30], "flags": []},
                    {
                        "header": 32,
                        "tag": [],
                        "cite": [33],
                        "body": [34],
                        "flags": ["cite_is_bare_headline"],
                    },
                ],
            }
        if schema is CardLabelOutput:
            flags = list(payload.get("deterministic_validation_flags", []))
            paraphrased = str(payload.get("body", "")).startswith(
                "Throughout the opinion piece"
            )
            return {
                "evidence_type": "paraphrased" if paraphrased else "quoted",
                "source_text_present": not paraphrased,
                "side": payload["default_side"],
                "topic_tags": ["crypto"],
                "flags": flags,
                "explanation": (
                    "This paraphrase lacks source text and is excluded."
                    if paraphrased
                    else "Deterministic validation flags require review."
                    if flags
                    else ""
                ),
            }
        raise AssertionError(f"No test model response for {schema}")


@pytest.fixture(autouse=True)
def explicit_runtime_test_doubles(monkeypatch):
    monkeypatch.setattr(
        "casefile.retrieval.build_embedder", lambda settings: _TestEmbedder()
    )
    monkeypatch.setattr(
        "casefile.ingest.pipeline.build_anthropic_client",
        lambda settings: _TestModel(),
    )


@pytest.fixture
def sample_docx() -> Path:
    return REPO_ROOT / "background" / "Copy of Pro Cards - Crypto.docx"


@pytest.fixture
def isolated_settings(tmp_path):
    settings = replace(
        get_settings(),
        data_dir=tmp_path / "data",
        rules_dir=tmp_path / "rules",
        anthropic_api_key=None,
        calendar_provider="fixture",
        nsda_provider="fixture",
        nsda_base_url=None,
    )
    settings.ensure_runtime_dirs()
    settings.progress_path.write_text("[]\n", encoding="utf-8")
    settings.cards_path.write_text("[]\n", encoding="utf-8")
    settings.rules_dir.mkdir(parents=True, exist_ok=True)
    return settings
