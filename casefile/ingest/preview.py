"""Typed ingestion preview construction and staged-storage representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from casefile.agents.contracts import (
    IngestionPreview,
    IngestionProvenance,
)

from .validate import preview_card


STAGED_INGESTION_SCHEMA_VERSION = 1


@dataclass
class StagedIngestion:
    schema_version: int
    job_id: str
    confirmation_token: str
    source_path: str
    source_filename: str
    source_sha256: str
    resolution: str
    side: str
    marking_convention: str
    marking_votes: dict[str, int]
    cards: list[dict[str, Any]]
    warnings: list[str]
    model: str
    boundary_prompt: str
    labeling_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StagedIngestion":
        if value.get("schema_version") != STAGED_INGESTION_SCHEMA_VERSION:
            raise ValueError("staged ingestion schema is unsupported")
        return cls(**value)

    def artifact(self) -> IngestionPreview:
        return IngestionPreview(
            job_id=self.job_id,
            confirmation_token=self.confirmation_token,
            source_filename=self.source_filename,
            source_sha256=self.source_sha256,
            resolution=self.resolution,
            side=self.side,
            cards=[
                preview_card(card, position=position)
                for position, card in enumerate(self.cards, start=1)
            ],
            warnings=self.warnings,
            provenance=IngestionProvenance(
                model=self.model,
                boundary_prompt=self.boundary_prompt,
                labeling_prompt=self.labeling_prompt,
            ),
        )


def preview_summary(preview: IngestionPreview) -> str:
    indexable = sum(card.indexable for card in preview.cards)
    flagged = sum(bool(card.flags) for card in preview.cards)
    return (
        f"Parsed {len(preview.cards)} evidence cards from {preview.source_filename}; "
        f"{indexable} are indexable and {flagged} require review. "
        "Confirm the staged preview before anything is committed."
    )
