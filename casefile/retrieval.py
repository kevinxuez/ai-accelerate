"""Filtered card/rule retrieval with optional Chroma and an offline JSON fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

from casefile.config import Settings, get_settings
from casefile.ingest.pipeline import is_indexable
from casefile.models import RuleChunk
from casefile.security.prompt_guard import inspect_text


DIMENSIONS = 384
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]{1,}")


def _terms(text: str) -> list[str]:
    words = [word.lower() for word in TOKEN.findall(text)]
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


def hash_embedding(text: str, dimensions: int = DIMENSIONS) -> list[float]:
    """Stable local feature hashing; no model download or network call is required."""
    vector = [0.0] * dimensions
    for term in _terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        vector[index] += -1.0 if value & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


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
        enable_chroma: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_dirs()
        self._lock = threading.RLock()
        self._client: Any | None = None
        if enable_chroma:
            try:
                import chromadb  # type: ignore

                self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            except (ImportError, RuntimeError, ValueError):
                self._client = None

    @property
    def backend(self) -> str:
        return "chroma" if self._client is not None else "json"

    def rebuild_cards(self, cards: list[dict[str, Any]] | None = None) -> int:
        with self._lock:
            if cards is None:
                cards = self._load_cards()
            searchable = [card for card in cards if is_indexable(card)]
            if self._client is not None:
                try:
                    self._client.delete_collection("cards")
                except Exception:
                    pass
                collection = self._client.get_or_create_collection(
                    "cards", metadata={"hnsw:space": "cosine"}
                )
                if searchable:
                    collection.upsert(
                        ids=[card["id"] for card in searchable],
                        documents=[card["returned_document"] for card in searchable],
                        embeddings=[hash_embedding(card["embedding_text"]) for card in searchable],
                        metadatas=[
                            {
                                "resolution": card["resolution"],
                                "side": card["side"],
                                "header": card["header"],
                                "source_file": card["source_file"],
                            }
                            for card in searchable
                        ],
                    )
            return len(searchable)

    def rebuild_rules(self) -> int:
        chunks: list[RuleChunk] = []
        for path in sorted(self.settings.rules_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            chunks.extend(chunk_rules(path))
        payload = [chunk.to_dict() for chunk in chunks]
        searchable = [
            chunk
            for chunk in chunks
            if chunk.injection_risk != "high" or chunk.injection_approved
        ]
        rules_json = self.settings.data_dir / "rules_chunks.json"
        rules_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if self._client is not None:
            try:
                self._client.delete_collection("rules")
            except Exception:
                pass
            collection = self._client.get_or_create_collection(
                "rules", metadata={"hnsw:space": "cosine"}
            )
            if searchable:
                collection.upsert(
                    ids=[chunk.id for chunk in searchable],
                    documents=[chunk.text for chunk in searchable],
                    embeddings=[
                        hash_embedding(f"{chunk.section_title}\n{chunk.text}")
                        for chunk in searchable
                    ],
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
            raise ValueError("resolution is required for card retrieval")
        threshold = self.settings.min_relevance if min_relevance is None else min_relevance
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
        by_id = {card["id"]: card for card in cards}
        if self._client is not None:
            try:
                clauses: list[dict[str, Any]] = [{"resolution": {"$eq": resolution}}]
                if side is not None:
                    clauses.append({"side": {"$eq": side}})
                if allowed_sources:
                    sources = sorted(allowed_sources)
                    clauses.append(
                        {
                            "source_file": (
                                {"$eq": sources[0]}
                                if len(sources) == 1
                                else {"$in": sources}
                            )
                        }
                    )
                where: dict[str, Any] = clauses[0] if len(clauses) == 1 else {"$and": clauses}
                result = self._client.get_collection("cards").query(
                    query_embeddings=[hash_embedding(query)],
                    n_results=min(n, len(cards)),
                    where=where,
                    include=["distances"],
                )
                ids = result.get("ids", [[]])[0]
                distances = result.get("distances", [[]])[0]
                ranked = []
                for card_id, distance in zip(ids, distances):
                    score = 1.0 - float(distance)
                    if score >= threshold and card_id in by_id:
                        ranked.append(self._with_score(by_id[card_id], score))
                return ranked
            except Exception:
                # A committed JSON ledger is the reliability path for a live demo.
                pass
        query_vector = hash_embedding(query)
        ranked = sorted(
            (
                (
                    cosine(query_vector, hash_embedding(card["embedding_text"])),
                    card,
                )
                for card in cards
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            self._with_score(card, score)
            for score, card in ranked[:n]
            if score >= threshold
        ]

    def available_card_files(
        self,
        *,
        resolution: str,
        side: str | None = None,
    ) -> list[str]:
        """List committed, searchable evidence files for the active context."""

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
        query_vector = hash_embedding(question)
        ranked = sorted(
            (
                (
                    cosine(
                        query_vector,
                        hash_embedding(f"{chunk['section_title']}\n{chunk['text']}"),
                    ),
                    chunk,
                )
                for chunk in chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            {
                **chunk,
                "content_trust": chunk.get("content_trust", "untrusted_document"),
                "retrieval_trust": "untrusted_retrieval",
                "score": round(score, 6),
                "_chunk_id": chunk["id"],
            }
            for score, chunk in ranked[:n]
            if score >= self.settings.min_relevance
        ]

    def _load_cards(self) -> list[dict[str, Any]]:
        if not self.settings.cards_path.exists():
            return []
        value = json.loads(self.settings.cards_path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    def _load_rule_chunks(self) -> list[dict[str, Any]]:
        path = self.settings.data_dir / "rules_chunks.json"
        if not path.exists():
            self.rebuild_rules()
        if not path.exists():
            return []
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    @staticmethod
    def _with_score(card: dict[str, Any], score: float) -> dict[str, Any]:
        # The returned document always starts with the intact citation.
        return {
            **card,
            "content_trust": card.get("content_trust", "untrusted_document"),
            "retrieval_trust": "untrusted_retrieval",
            "injection_risk": card.get("injection_risk", "low"),
            "injection_signals": card.get("injection_signals", []),
            "score": round(score, 6),
            "_chunk_id": card["id"],
        }
