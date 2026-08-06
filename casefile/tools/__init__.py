"""Typed, agent-scoped CaseFile tool registry and focused implementations."""

from __future__ import annotations

from typing import Any

from casefile.config import Settings, get_settings
from casefile.ingest.pipeline import IngestionPipeline
from casefile.retrieval import CaseFileIndex

from .calendar import CalendarTools
from .context import ToolContext, ToolRuntime
from .evidence import EvidenceTools
from .ingestion import IngestionTools
from .progress import ProgressTools
from .registry import ToolDefinition, ToolRegistry, build_tool_registry
from .rules import RuleTools
from .topic import TopicTools


class CaseFileTools:
    """Construct the focused tools and their single typed registry."""

    def __init__(
        self,
        settings: Settings | None = None,
        index: CaseFileIndex | None = None,
        ingestion: IngestionPipeline | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime = ToolRuntime(self.settings)
        self.index = index or CaseFileIndex(self.settings)
        self.evidence = EvidenceTools(self.runtime, self.index)
        self.rules = RuleTools(self.runtime, self.index)
        self.topic = TopicTools(self.runtime)
        self.progress = ProgressTools(self.runtime)
        pipeline = ingestion or IngestionPipeline(self.settings, llm=model)
        self.ingestion_tools = IngestionTools(self.runtime, pipeline)
        self.calendar = CalendarTools(self.runtime)
        self.registry = build_tool_registry(self, self.runtime)


__all__ = [
    "CaseFileTools",
    "ToolContext",
    "ToolDefinition",
    "ToolRegistry",
]
