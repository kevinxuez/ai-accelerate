"""Shared, JSON-serializable data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Role = Literal["student", "coach"]
Side = Literal["pro", "con", "unknown"]
IngestStatus = Literal["ok", "flagged", "incomplete"]


@dataclass
class RunRecord:
    text: str
    start: int
    end: int
    bold: bool = False
    underline: bool = False
    highlight: str | None = None


@dataclass
class ParagraphRecord:
    i: int
    style: str
    text: str
    length: int
    bold_fraction: float
    underline_fraction: float
    highlights: dict[str, float]
    runs: list[RunRecord] = field(default_factory=list)
    link: bool = False

    def compact(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "style": self.style,
            "text": self.text,
            "len": self.length,
            "b": self.bold_fraction,
            "u": self.underline_fraction,
            "hl": self.highlights,
            "runs": len(self.runs),
            "link": self.link,
        }


@dataclass
class CardBoundary:
    header: int | None
    tag: list[int] = field(default_factory=list)
    cite: list[int] = field(default_factory=list)
    body: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CardBoundary":
        return cls(
            header=value.get("header"),
            tag=list(value.get("tag") or []),
            cite=list(value.get("cite") or []),
            body=list(value.get("body") or []),
            flags=list(value.get("flags") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CardRecord:
    id: str
    header: str
    tag: str
    cite_full: str
    author: str
    author_type: Literal["person", "organization", "unknown"]
    year: int | None
    date_raw: str
    source: str
    url: str
    date_accessed: str
    cutter: str
    body: str
    read_spans: list[list[int]]
    emphasis_spans: list[list[int]]
    marking_convention: str
    evidence_type: Literal["quoted", "paraphrased", "unknown"]
    source_text_present: bool
    resolution: str
    resolution_confidence: Literal["high", "low"]
    side: Side
    topic_tags: list[str]
    ingest_status: IngestStatus
    flags: list[str]
    source_file: str
    source_paragraphs: list[int]
    embedding_text: str
    returned_document: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleChunk:
    id: str
    section_number: str
    section_title: str
    text: str
    document: str
    event: str = "Public Forum"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressRecord:
    student_id: str
    date: str
    speech_position: str
    resolution: str
    weakness_tags: list[str]
    assessment_text: str
    author_role: Role
    author_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

