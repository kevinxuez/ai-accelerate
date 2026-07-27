"""Deterministic-first DOCX ingestion."""

__all__ = ["IngestionPipeline", "IngestionPreview"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import IngestionPipeline, IngestionPreview

        return {"IngestionPipeline": IngestionPipeline, "IngestionPreview": IngestionPreview}[name]
    raise AttributeError(name)
