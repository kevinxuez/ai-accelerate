"""In-memory semantic retrieval over confirmed evidence and rules ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.config import (
    PINNED_EMBEDDING_DIMENSIONS,
    PINNED_EMBEDDING_MODEL,
    Settings,
    get_settings,
)
from casefile.ingest.validate import is_indexable
from casefile.security.prompt_guard import inspect_text


LEDGER_SCHEMA_VERSION = 1
_UNSET = object()


@dataclass
class RuleChunk:
    id: str
    section_number: str
    section_title: str
    text: str
    document: str
    event: str = "Public Forum"
    content_trust: Literal["untrusted_document"] = "untrusted_document"
    injection_risk: Literal["low", "medium", "high"] = "low"
    injection_signals: list[str] = field(default_factory=list)
    injection_approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    name = PINNED_EMBEDDING_MODEL
    dimensions = PINNED_EMBEDDING_DIMENSIONS

    def __init__(self, settings: Settings) -> None:
        path = settings.embedding_model_path
        if not path.is_dir():
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                "The pinned embedding model assets are not provisioned locally.",
                stage="retrieval.embedding.startup",
                safe_details={"model": self.name},
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                "sentence-transformers is required but is not installed.",
                stage="retrieval.embedding.startup",
                safe_details={"model": self.name},
                cause=exc,
            ) from exc
        try:
            self._model = SentenceTransformer(str(path), local_files_only=True)
            get_dimensions = getattr(self._model, "get_embedding_dimension", None)
            actual = (
                get_dimensions()
                if get_dimensions is not None
                else self._model.get_sentence_embedding_dimension()
            )
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                "The pinned embedding model assets could not be loaded.",
                stage="retrieval.embedding.startup",
                safe_details={"model": self.name},
                cause=exc,
            ) from exc
        if actual != self.dimensions:
            raise CaseFileError(
                ErrorCode.CONFIGURATION_ERROR,
                "The embedding model dimensions do not match the pinned configuration.",
                stage="retrieval.embedding.startup",
                safe_details={
                    "model": self.name,
                    "expected_dimensions": self.dimensions,
                    "actual_dimensions": actual,
                },
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            values = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [list(map(float, vector)) for vector in values]
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The pinned embedding model could not encode the retrieval request.",
                stage="retrieval.embedding.encode",
                retryable=True,
                cause=exc,
            ) from exc


def build_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings)


def chunk_rules(path: str | Path) -> list[RuleChunk]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    heading = re.compile(r"^(#{1,6})\s+(?:(\d+(?:\.\d+)*[A-Za-z]?)\s*[:.-]?\s*)?(.+)$")
    chunks: list[RuleChunk] = []
    current_number = ""
    current_title = source.stem.replace("_", " ").title()
    body: list[str] = []

    def finish() -> None:
        content = "\n".join(body).strip()
        if not content:
            return
        identity = f"{source.name}:{current_number}:{current_title}"
        decision = inspect_text(content, trust="untrusted_document")
        chunks.append(
            RuleChunk(
                id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                section_number=current_number or "unsectioned",
                section_title=current_title,
                text=content,
                document=source.name,
                injection_risk=decision.risk,
                injection_signals=decision.signals,
            )
        )

    for line in text.splitlines():
        match = heading.match(line)
        if match:
            finish()
            current_number = match.group(2) or ""
            current_title = match.group(3).strip()
            body = []
        else:
            body.append(line)
    finish()
    return chunks


class CaseFileIndex:
    """Rank the small committed ledgers directly without a persistent vector DB."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | object = _UNSET,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self._lock = threading.RLock()
        self._embedder = (
            build_embedder(self.settings) if embedder is _UNSET else embedder
        )
        if not hasattr(self._embedder, "embed"):
            raise TypeError("embedder test double must implement embed(texts)")

    @property
    def backend(self) -> str:
        return "in_memory"

    @property
    def embedding_model(self) -> str:
        return str(self._embedder.name)

    def validate_ready(self) -> None:
        self.rebuild_cards()
        chunks = self._load_rule_chunks()
        searchable = [
            chunk
            for chunk in chunks
            if chunk.get("injection_risk", "low") != "high"
            or bool(chunk.get("injection_approved"))
        ]
        self._validate_embeddings(
            [f"{chunk['section_title']}\n{chunk['text']}" for chunk in searchable]
        )

    def rebuild_cards(self, cards: list[dict[str, Any]] | None = None) -> int:
        cards = self._load_cards() if cards is None else cards
        searchable = [card for card in cards if is_indexable(card)]
        self._validate_embeddings(
            [str(card["embedding_text"]) for card in searchable]
        )
        return len(searchable)

    def rebuild_rules(self, chunks: list[RuleChunk] | None = None) -> int:
        try:
            if chunks is None:
                chunks = [
                    chunk
                    for path in sorted(self.settings.rules_dir.glob("*.md"))
                    if path.name.lower() != "readme.md"
                    for chunk in chunk_rules(path)
                ]
            payload = [chunk.to_dict() for chunk in chunks]
            searchable = [
                chunk
                for chunk in chunks
                if chunk.injection_risk != "high" or chunk.injection_approved
            ]
            rules_json = self.settings.data_dir / "rules_chunks.json"
            rules_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._validate_embeddings(
                [f"{chunk.section_title}\n{chunk.text}" for chunk in searchable]
            )
            return len(searchable)
        except CaseFileError:
            raise
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.INDEX_REBUILD_FAILED,
                "The in-memory rules index could not be validated.",
                stage="retrieval.rules.rebuild",
                cause=exc,
            ) from exc

    def search_cards(
        self,
        query: str,
        *,
        resolution: str,
        side: str | None = None,
        source_files: list[str] | None = None,
        n: int = 5,
        min_relevance: float | None = None,
    ) -> list[dict[str, Any]]:
        if not resolution:
            raise CaseFileError(
                ErrorCode.REQUEST_INVALID,
                "resolution is required for card retrieval.",
                stage="retrieval.cards.query",
                tool="search_cards",
            )
        allowed_sources = set(source_files or [])
        cards = [
            card
            for card in self._load_cards()
            if is_indexable(card)
            and card.get("resolution") == resolution
            and (side is None or card.get("side") == side)
            and (
                not allowed_sources
                or str(card.get("source_file", "")) in allowed_sources
            )
        ]
        return self._rank(
            cards,
            [str(card["embedding_text"]) for card in cards],
            query=query,
            n=n,
            threshold=(
                self.settings.min_relevance
                if min_relevance is None
                else min_relevance
            ),
            kind="cards",
        )

    def available_card_files(
        self,
        *,
        resolution: str,
        side: str | None = None,
    ) -> list[str]:
        if not resolution:
            return []
        return sorted(
            {
                str(card.get("source_file", "")).strip()
                for card in self._load_cards()
                if is_indexable(card)
                and card.get("resolution") == resolution
                and (side is None or card.get("side") == side)
                and str(card.get("source_file", "")).strip()
            }
        )

    def search_rules(self, question: str, n: int = 3) -> list[dict[str, Any]]:
        chunks = [
            chunk
            for chunk in self._load_rule_chunks()
            if chunk.get("injection_risk", "low") != "high"
            or bool(chunk.get("injection_approved"))
        ]
        return self._rank(
            chunks,
            [f"{chunk['section_title']}\n{chunk['text']}" for chunk in chunks],
            query=question,
            n=n,
            threshold=self.settings.min_relevance,
            kind="rules",
        )

    def _rank(
        self,
        records: list[dict[str, Any]],
        texts: list[str],
        *,
        query: str,
        n: int,
        threshold: float,
        kind: str,
    ) -> list[dict[str, Any]]:
        if not records or n <= 0:
            return []
        vectors = self._embed([query, *texts], stage=f"retrieval.{kind}.query")
        query_vector = vectors[0]
        scored = [
            (self._cosine(query_vector, vector), str(record["id"]), record)
            for record, vector in zip(records, vectors[1:])
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                **record,
                "content_trust": record.get("content_trust", "untrusted_document"),
                "retrieval_trust": "untrusted_retrieval",
                "injection_risk": record.get("injection_risk", "low"),
                "injection_signals": record.get("injection_signals", []),
                "score": round(score, 6),
                "_chunk_id": item_id,
            }
            for score, item_id, record in scored[:n]
            if score >= threshold
        ]

    def _validate_embeddings(self, texts: list[str]) -> None:
        if texts:
            self._embed(texts, stage="retrieval.embedding.validate")

    def _embed(self, texts: list[str], *, stage: str) -> list[list[float]]:
        try:
            with self._lock:
                vectors = self._embedder.embed(texts)
        except CaseFileError:
            raise
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The embedding model could not rank the retrieval request.",
                stage=stage,
                retryable=True,
                cause=exc,
            ) from exc
        dimensions = int(self._embedder.dimensions)
        if len(vectors) != len(texts) or any(
            len(vector) != dimensions for vector in vectors
        ):
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The embedding model returned malformed vectors.",
                stage=stage,
                retryable=True,
            )
        return vectors

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if not denominator:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / denominator

    def _load_cards(self) -> list[dict[str, Any]]:
        return self._load_json_list(self.settings.cards_path, "evidence ledger")

    def _load_rule_chunks(self) -> list[dict[str, Any]]:
        path = self.settings.data_dir / "rules_chunks.json"
        if not path.exists():
            self.rebuild_rules()
        return self._load_json_list(path, "rules ledger")

    @staticmethod
    def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                f"The {label} could not be read.",
                stage="retrieval.ledger.read",
                cause=exc,
            ) from exc
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise CaseFileError(
                ErrorCode.STORAGE_READ_FAILED,
                f"The {label} is malformed.",
                stage="retrieval.ledger.read",
            )
        return value
