"""Evaluate ranked card/rule retrieval against curated relevant chunk IDs."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from casefile.config import Settings, get_settings
from casefile.retrieval import CaseFileIndex


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "retrieval_dataset.json"
CardTextMode = Literal["stored", "header-tag", "full-card"]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class StrictEvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class EvalCard(StrictEvalModel):
    id: Identifier
    header: str = Field(max_length=300)
    tag: str = Field(max_length=2_000)
    cite_full: str = Field(max_length=4_000)
    body: str = Field(min_length=1, max_length=50_000)
    resolution: str = Field(min_length=1, max_length=200)
    side: Literal["pro", "con"]
    topic_tags: list[str] = Field(default_factory=list, max_length=20)
    embedding_text: str = Field(min_length=1, max_length=50_000)


class EvalRule(StrictEvalModel):
    id: Identifier
    section_number: str = Field(min_length=1, max_length=100)
    section_title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=50_000)
    document: str = Field(min_length=1, max_length=500)


class RetrievalQuery(StrictEvalModel):
    id: Identifier
    corpus: Literal["cards", "rules"]
    query: str = Field(min_length=1, max_length=20_000)
    expected_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    resolution: str | None = Field(default=None, max_length=200)
    side: Literal["pro", "con"] | None = None

    @model_validator(mode="after")
    def validate_filters(self) -> "RetrievalQuery":
        if self.corpus == "cards" and (not self.resolution or not self.side):
            raise ValueError("card queries require resolution and side")
        if self.corpus == "rules" and (
            self.resolution is not None or self.side is not None
        ):
            raise ValueError("rule queries cannot specify card filters")
        if len(set(self.expected_ids)) != len(self.expected_ids):
            raise ValueError("expected_ids must be unique")
        return self


class RetrievalDataset(StrictEvalModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=2_000)
    cards: list[EvalCard] = Field(min_length=1, max_length=5_000)
    rules: list[EvalRule] = Field(min_length=1, max_length=5_000)
    queries: list[RetrievalQuery] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_references(self) -> "RetrievalDataset":
        card_ids = [card.id for card in self.cards]
        rule_ids = [rule.id for rule in self.rules]
        query_ids = [query.id for query in self.queries]
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("card ids must be unique")
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("rule ids must be unique")
        if set(card_ids) & set(rule_ids):
            raise ValueError("card and rule ids must not overlap")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query ids must be unique")
        available = {"cards": set(card_ids), "rules": set(rule_ids)}
        for query in self.queries:
            unknown = set(query.expected_ids) - available[query.corpus]
            if unknown:
                raise ValueError(
                    f"query {query.id!r} references unknown ids: {sorted(unknown)}"
                )
        return self


@dataclass(frozen=True)
class RetrievalStrategy:
    name: str
    card_text_mode: CardTextMode
    k: int = 3
    min_relevance: float = 0.08

    def __post_init__(self) -> None:
        if self.k < 1 or self.k > 10:
            raise ValueError("k must be between 1 and 10")
        if not -1.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between -1 and 1")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return round(ordered[position], 3)


def _embedding_text(card: EvalCard, mode: CardTextMode) -> str:
    if mode == "stored":
        return card.embedding_text
    if mode == "header-tag":
        return "\n".join(part for part in (card.header, card.tag) if part)
    return "\n".join(
        part for part in (card.header, card.tag, card.body) if part
    )


def _card_payload(card: EvalCard, mode: CardTextMode) -> dict:
    value = card.model_dump(mode="json")
    value.update(
        {
            "embedding_text": _embedding_text(card, mode),
            "returned_document": "\n".join(
                part
                for part in (card.cite_full, card.header, card.tag, card.body)
                if part
            ),
            "ingest_status": "ok",
            "flags": [],
            "injection_risk": "low",
            "injection_approved": False,
        }
    )
    return value


def _rule_payload(rule: EvalRule) -> dict:
    return {
        **rule.model_dump(mode="json"),
        "event": "Public Forum",
        "content_trust": "untrusted_document",
        "injection_risk": "low",
        "injection_signals": [],
        "injection_approved": False,
    }


def _settings_for_strategy(
    root: Path,
    base: Settings,
    strategy: RetrievalStrategy,
) -> Settings:
    data_dir = root / "data"
    rules_dir = root / "rules"
    chroma_dir = root / "chroma"
    data_dir.mkdir(parents=True)
    rules_dir.mkdir(parents=True)
    chroma_dir.mkdir(parents=True)
    return replace(
        base,
        data_dir=data_dir,
        rules_dir=rules_dir,
        chroma_dir=chroma_dir,
        min_relevance=strategy.min_relevance,
        anthropic_api_key=None,
    )


def _evaluate_strategy(
    dataset: RetrievalDataset,
    strategy: RetrievalStrategy,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="casefile-retrieval-eval-") as directory:
        settings = _settings_for_strategy(Path(directory), get_settings(), strategy)
        settings.cards_path.write_text(
            json.dumps(
                [_card_payload(card, strategy.card_text_mode) for card in dataset.cards],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (settings.data_dir / "rules_chunks.json").write_text(
            json.dumps(
                [_rule_payload(rule) for rule in dataset.rules],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        settings.progress_path.write_text("[]\n", encoding="utf-8")
        index = CaseFileIndex(settings, enable_chroma=False)

        results = []
        latencies: list[float] = []
        positive_recalls: list[float] = []
        positive_precisions: list[float] = []
        reciprocal_ranks: list[float] = []
        positive_hits: list[float] = []
        empty_results: list[float] = []
        query_success: list[float] = []
        filter_leakage = 0

        for query in dataset.queries:
            started = time.perf_counter()
            if query.corpus == "cards":
                hits = index.search_cards(
                    query.query,
                    resolution=query.resolution or "",
                    side=query.side,
                    n=strategy.k,
                    min_relevance=strategy.min_relevance,
                )
            else:
                hits = index.search_rules(query.query, n=strategy.k)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            retrieved = [str(hit["_chunk_id"]) for hit in hits]
            relevant = set(query.expected_ids)
            matched = [chunk_id for chunk_id in retrieved if chunk_id in relevant]

            if relevant:
                recall = len(set(matched)) / len(relevant)
                precision = len(matched) / strategy.k
                first_rank = next(
                    (
                        rank
                        for rank, chunk_id in enumerate(retrieved, start=1)
                        if chunk_id in relevant
                    ),
                    None,
                )
                reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
                hit = 1.0 if matched else 0.0
                positive_recalls.append(recall)
                positive_precisions.append(precision)
                reciprocal_ranks.append(reciprocal_rank)
                positive_hits.append(hit)
                success = hit
            else:
                empty_correct = 1.0 if not retrieved else 0.0
                empty_results.append(empty_correct)
                recall = precision = reciprocal_rank = None
                success = empty_correct
            query_success.append(success)

            if query.corpus == "cards":
                filter_leakage += sum(
                    1
                    for hit in hits
                    if hit.get("resolution") != query.resolution
                    or hit.get("side") != query.side
                )
            results.append(
                {
                    "id": query.id,
                    "corpus": query.corpus,
                    "query": query.query,
                    "expected_ids": query.expected_ids,
                    "retrieved_ids": retrieved,
                    "scores": [hit.get("score") for hit in hits],
                    "recall_at_k": (
                        None if recall is None else round(recall, 6)
                    ),
                    "precision_at_k": (
                        None if precision is None else round(precision, 6)
                    ),
                    "reciprocal_rank": (
                        None
                        if reciprocal_rank is None
                        else round(reciprocal_rank, 6)
                    ),
                    "success": bool(success),
                    "latency_ms": round(elapsed, 3),
                }
            )

    metrics = {
        "queries": len(results),
        "positive_queries": len(positive_recalls),
        "empty_queries": len(empty_results),
        "recall_at_k": round(statistics.fmean(positive_recalls), 6),
        "precision_at_k": round(statistics.fmean(positive_precisions), 6),
        "mrr": round(statistics.fmean(reciprocal_ranks), 6),
        "hit_rate": round(statistics.fmean(positive_hits), 6),
        "no_result_accuracy": (
            round(statistics.fmean(empty_results), 6) if empty_results else None
        ),
        "query_success_rate": round(statistics.fmean(query_success), 6),
        "filter_leakage": filter_leakage,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return {
        "strategy": {
            "name": strategy.name,
            "card_text_mode": strategy.card_text_mode,
            "k": strategy.k,
            "min_relevance": strategy.min_relevance,
            "backend": "json",
        },
        "metrics": metrics,
        "results": results,
    }


def load_dataset(path: str | Path = DEFAULT_DATASET) -> RetrievalDataset:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalDataset.model_validate(value)


def run(
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    card_text_modes: list[CardTextMode] | None = None,
    k: int = 3,
    min_relevances: list[float] | None = None,
) -> dict:
    dataset = load_dataset(dataset_path)
    modes = card_text_modes or ["stored", "header-tag", "full-card"]
    thresholds = min_relevances or [0.08, 0.15]
    strategies = [
        RetrievalStrategy(
            name=f"{mode}-k{k}-t{threshold:g}",
            card_text_mode=mode,
            k=k,
            min_relevance=threshold,
        )
        for mode in modes
        for threshold in thresholds
    ]
    evaluations = [
        _evaluate_strategy(dataset, strategy) for strategy in strategies
    ]
    best = max(
        evaluations,
        key=lambda item: (
            item["metrics"]["query_success_rate"],
            item["metrics"]["mrr"],
            item["metrics"]["recall_at_k"],
            item["metrics"]["precision_at_k"],
        ),
    )
    return {
        "dataset": dataset.name,
        "description": dataset.description,
        "cases": len(dataset.queries),
        "strategies": evaluations,
        "best_strategy": best["strategy"]["name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--card-text",
        action="append",
        choices=("stored", "header-tag", "full-card"),
        dest="card_text_modes",
        help="Card text strategy to evaluate; repeat to compare multiple modes.",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--min-relevance",
        action="append",
        type=float,
        dest="min_relevances",
        help="Relevance threshold to evaluate; repeat to compare multiple thresholds.",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "retrieval_eval_results.json"),
    )
    args = parser.parse_args()
    result = run(
        dataset_path=args.dataset,
        card_text_modes=args.card_text_modes,
        k=args.k,
        min_relevances=args.min_relevances,
    )
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "cases": result["cases"],
                "best_strategy": result["best_strategy"],
                "metrics": {
                    item["strategy"]["name"]: item["metrics"]
                    for item in result["strategies"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
