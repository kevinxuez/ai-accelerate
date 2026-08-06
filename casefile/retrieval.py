"""Required Chroma retrieval over the confirmed evidence and rules ledgers."""

from __future__ import annotations

import hashlib
import json
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


COLLECTION_SCHEMA_VERSION = 1
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
            self._model = SentenceTransformer(
                str(path),
                local_files_only=True,
            )
            actual = self._model.get_sentence_embedding_dimension()
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


def build_chroma_client(settings: Settings) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise CaseFileError(
            ErrorCode.CONFIGURATION_ERROR,
            "Chroma is required but is not installed.",
            stage="retrieval.chroma.startup",
            cause=exc,
        ) from exc
    try:
        return chromadb.PersistentClient(path=str(settings.chroma_dir))
    except Exception as exc:
        raise CaseFileError(
            ErrorCode.RETRIEVAL_UNAVAILABLE,
            "The configured Chroma store is unavailable.",
            stage="retrieval.chroma.startup",
            retryable=True,
            cause=exc,
        ) from exc


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
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any = _UNSET,
        embedder: Embedder | object = _UNSET,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self._lock = threading.RLock()
        self._embedder = (
            build_embedder(self.settings) if embedder is _UNSET else embedder
        )
        self._client = (
            build_chroma_client(self.settings) if client is _UNSET else client
        )
        if not hasattr(self._embedder, "embed"):
            raise TypeError("embedder test double must implement embed(texts)")
        self._ensure_collection("cards")
        self._ensure_collection("rules")

    @property
    def backend(self) -> str:
        return "chroma"

    @property
    def embedding_model(self) -> str:
        return str(self._embedder.name)

    def validate_ready(self) -> None:
        self._ensure_collection("cards")
        self._ensure_collection("rules")

    def rebuild_cards(self, cards: list[dict[str, Any]] | None = None) -> int:
        with self._lock:
            cards = self._load_cards() if cards is None else cards
            searchable = [card for card in cards if is_indexable(card)]
            collection = self._reset_collection("cards")
            if searchable:
                try:
                    collection.upsert(
                        ids=[str(card["id"]) for card in searchable],
                        documents=[
                            str(card["returned_document"]) for card in searchable
                        ],
                        embeddings=self._embedder.embed(
                            [str(card["embedding_text"]) for card in searchable]
                        ),
                        metadatas=[
                            {
                                "resolution": str(card["resolution"]),
                                "side": str(card["side"]),
                                "source_file": str(card["source_file"]),
                                "card_id": str(card["id"]),
                                "indexable": True,
                            }
                            for card in searchable
                        ],
                    )
                except CaseFileError:
                    raise
                except Exception as exc:
                    raise CaseFileError(
                        ErrorCode.INDEX_REBUILD_FAILED,
                        "The Chroma evidence collection could not be rebuilt.",
                        stage="retrieval.cards.rebuild",
                        cause=exc,
                    ) from exc
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
            collection = self._reset_collection("rules")
            if searchable:
                collection.upsert(
                    ids=[chunk.id for chunk in searchable],
                    documents=[chunk.text for chunk in searchable],
                    embeddings=self._embedder.embed(
                        [f"{chunk.section_title}\n{chunk.text}" for chunk in searchable]
                    ),
                    metadatas=[
                        {
                            "section_number": chunk.section_number,
                            "section_title": chunk.section_title,
                            "document": chunk.document,
                            "event": chunk.event,
                        }
                        for chunk in searchable
                    ],
                )
            return len(searchable)
        except CaseFileError:
            raise
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.INDEX_REBUILD_FAILED,
                "The Chroma rules collection could not be rebuilt.",
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
        threshold = (
            self.settings.min_relevance if min_relevance is None else min_relevance
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
        if not cards:
            return []
        by_id = {str(card["id"]): card for card in cards}
        clauses: list[dict[str, Any]] = [{"resolution": {"$eq": resolution}}]
        if side is not None:
            clauses.append({"side": {"$eq": side}})
        if allowed_sources:
            sources = sorted(allowed_sources)
            clauses.append(
                {
                    "source_file": (
                        {"$eq": sources[0]} if len(sources) == 1 else {"$in": sources}
                    )
                }
            )
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        result = self._query(
            "cards",
            query=query,
            n=min(n, len(cards)),
            where=where,
        )
        return self._resolve_hits(result, by_id, threshold, kind="card")

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
        if not chunks:
            return []
        by_id = {str(chunk["id"]): chunk for chunk in chunks}
        result = self._query("rules", query=question, n=min(n, len(chunks)))
        return self._resolve_hits(
            result,
            by_id,
            self.settings.min_relevance,
            kind="rule",
        )

    def _query(
        self,
        collection_name: str,
        *,
        query: str,
        n: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            collection = self._client.get_collection(
                collection_name,
                embedding_function=None,
            )
            arguments: dict[str, Any] = {
                "query_embeddings": self._embedder.embed([query]),
                "n_results": n,
                "include": ["distances"],
            }
            if where is not None:
                arguments["where"] = where
            result = collection.query(**arguments)
        except CaseFileError:
            raise
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The configured Chroma collection could not be queried.",
                stage=f"retrieval.{collection_name}.query",
                retryable=True,
                cause=exc,
            ) from exc
        if not isinstance(result, dict):
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The configured Chroma collection returned an invalid query result.",
                stage=f"retrieval.{collection_name}.query",
            )
        return result

    def _resolve_hits(
        self,
        result: dict[str, Any],
        by_id: dict[str, dict[str, Any]],
        threshold: float,
        *,
        kind: str,
    ) -> list[dict[str, Any]]:
        ids = result.get("ids", [[]])
        distances = result.get("distances", [[]])
        hit_ids = ids[0] if isinstance(ids, list) and ids else []
        hit_distances = (
            distances[0] if isinstance(distances, list) and distances else []
        )
        if not isinstance(hit_ids, list) or not isinstance(hit_distances, list):
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The configured Chroma collection returned malformed hits.",
                stage=f"retrieval.{kind}s.resolve",
            )
        unknown = [str(item) for item in hit_ids if str(item) not in by_id]
        if unknown:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_INDEX_MISMATCH,
                "Chroma referenced an id that is missing from the committed ledger.",
                stage=f"retrieval.{kind}s.resolve",
                safe_details={"unknown_id_count": len(unknown)},
            )
        resolved: list[dict[str, Any]] = []
        for item, distance in zip(hit_ids, hit_distances):
            score = 1.0 - float(distance)
            if score < threshold:
                continue
            record = by_id[str(item)]
            resolved.append(
                {
                    **record,
                    "content_trust": record.get("content_trust", "untrusted_document"),
                    "retrieval_trust": "untrusted_retrieval",
                    "injection_risk": record.get("injection_risk", "low"),
                    "injection_signals": record.get("injection_signals", []),
                    "score": round(score, 6),
                    "_chunk_id": str(item),
                }
            )
        return resolved

    def _expected_metadata(self) -> dict[str, Any]:
        return {
            "casefile_schema_version": COLLECTION_SCHEMA_VERSION,
            "embedding_model": self._embedder.name,
            "embedding_dimensions": self._embedder.dimensions,
            "hnsw:space": "cosine",
        }

    def _ensure_collection(self, name: str) -> Any:
        expected = self._expected_metadata()
        try:
            collection = self._client.get_or_create_collection(
                name,
                metadata=expected,
                embedding_function=None,
            )
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_UNAVAILABLE,
                "The configured Chroma collection is unavailable.",
                stage=f"retrieval.{name}.startup",
                retryable=True,
                cause=exc,
            ) from exc
        metadata = getattr(collection, "metadata", None) or {}
        mismatch = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise CaseFileError(
                ErrorCode.RETRIEVAL_INDEX_MISMATCH,
                "The Chroma collection schema does not match the configured embedding.",
                stage=f"retrieval.{name}.startup",
                safe_details={"mismatched_fields": sorted(mismatch)},
            )
        return collection

    def _reset_collection(self, name: str) -> Any:
        try:
            self._client.delete_collection(name)
            return self._client.get_or_create_collection(
                name,
                metadata=self._expected_metadata(),
                embedding_function=None,
            )
        except Exception as exc:
            raise CaseFileError(
                ErrorCode.INDEX_REBUILD_FAILED,
                "The configured Chroma collection could not be reset.",
                stage=f"retrieval.{name}.rebuild",
                cause=exc,
            ) from exc

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
