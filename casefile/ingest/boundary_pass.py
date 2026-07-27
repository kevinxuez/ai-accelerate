"""Boundary judgment over compact paragraph indices, with structural validation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from casefile.llm import AnthropicJSONClient
from casefile.models import CardBoundary, ParagraphRecord

from .serialize_index import render_index


WINDOW = 18
OVERLAP = 4

SYSTEM = """You segment competitive Public Forum debate evidence files into cards.

Each input line is a compact paragraph index:
[id] style len=N b=<bold fraction> u=<underline fraction> hl=<highlight> URL? "60-char preview"

A card may have a shorthand header, a claim tag, a citation, and a body. Only the citation
is required. Styles are hints. Cards can lack headers. Citation and body may share a
paragraph. Section titles and page furniture are junk.

Return paragraph ids only. Never return or reconstruct card text. Every id must occur in
exactly one card or junk, except that one id may be in cite and body for the same card.
Return JSON only:
{"cards":[{"header":4,"tag":[],"cite":[5],"body":[6],"flags":[]}],"junk":[0]}
Use null for a missing header. Allowed boundary flags: no_header, no_body,
cite_is_bare_url, cite_is_bare_headline, tag_merged_into_cite,
cite_body_same_paragraph, pdf_paste_fragmented, text_corrupt, duplicate_source.
"""

HEADER = re.compile(
    r"^(?:.{0,45}?)(?:[‘’']\s?\d{2}|\b(?:19|20)\d{2}\b|"
    r"\b\d{2}\b(?:\s*\([^)]*\))?)(?:\s*[-–—:].*)?$"
)
BARE_URL = re.compile(r"^\s*(?:https?://|www\.)\S+[\s./]*$", re.I)


def windows(
    records: list[ParagraphRecord], size: int = WINDOW, overlap: int = OVERLAP
) -> Iterable[list[ParagraphRecord]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("window size must be positive and overlap smaller than size")
    step = size - overlap
    for start in range(0, len(records), step):
        chunk = records[start : start + size]
        if chunk:
            yield chunk
        if start + size >= len(records):
            break


def citation_signal(record: ParagraphRecord) -> int:
    text = record.text
    score = 0
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        score += 1
    if record.link:
        score += 1
    if any(mark in text for mark in ('"', "“", "”")):
        score += 1
    if re.search(
        r"\b(?:Accessed|DOA|Press|Times|Reuters|Journal|Review|Institute|"
        r"University|News|Commission|School|Report|BitAML)\b",
        text,
        re.I,
    ):
        score += 1
    if re.match(r"^\s*[A-Z][\wÀ-ɏ.-]+,\s+[A-Z]", text):
        score += 1
    return score


def _header_like(record: ParagraphRecord) -> bool:
    text = record.text.strip()
    if record.length > 90 or record.link:
        return False
    return bool(HEADER.match(text)) and citation_signal(record) < 2


def heuristic_boundary_pass(records: list[ParagraphRecord]) -> dict[str, Any]:
    """Offline fallback and measured baseline for structurally conventional files.

    It deliberately does not pretend to solve the judgment cases assigned to the model;
    validators and the confirmation gate make its uncertainty visible.
    """
    cards: list[CardBoundary] = []
    junk: list[int] = []
    current: CardBoundary | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        if current.header is None and "no_header" not in current.flags:
            current.flags.append("no_header")
        if not current.body and "no_body" not in current.flags:
            current.flags.append("no_body")
        cards.append(current)
        current = None

    for record in records:
        text = record.text.strip()
        if _header_like(record):
            finish()
            current = CardBoundary(header=record.i)
            continue

        strong_cite = citation_signal(record) >= 2
        if current is not None and current.body and strong_cite:
            finish()
            current = CardBoundary(header=None, cite=[record.i], flags=["no_header"])
            continue

        if current is None:
            if strong_cite:
                current = CardBoundary(header=None, cite=[record.i], flags=["no_header"])
            else:
                junk.append(record.i)
            continue

        if not current.cite:
            current.cite.append(record.i)
            if BARE_URL.match(text):
                current.flags.append("cite_is_bare_url")
            elif citation_signal(record) == 0 and record.length < 180:
                current.flags.append("cite_is_bare_headline")
            continue

        if not current.body:
            previous = next((r for r in records if r.i == current.cite[-1]), None)
            if (
                record.link
                and previous is not None
                and not previous.link
                and previous.length < 80
                and record.length >= 220
            ):
                # Common paste shape: a one-line author followed by a paragraph that
                # contains both the rest of the citation and the evidence body.
                current.cite.append(record.i)
                current.body.append(record.i)
                current.flags.append("cite_body_same_paragraph")
                continue
            if BARE_URL.match(text) or (
                record.link
                and previous is not None
                and not previous.link
                and record.length < 220
            ):
                current.cite.append(record.i)
            else:
                current.body.append(record.i)
            continue

        # Consecutive body fragments stay with the card. The next header/cite closes it.
        current.body.append(record.i)

    finish()
    return {"cards": [card.to_dict() for card in cards], "junk": sorted(junk)}


def model_boundary_pass(
    records: list[ParagraphRecord], client: AnthropicJSONClient
) -> dict[str, Any]:
    return client.complete_json(system=SYSTEM, user=render_index(records))


def merge(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile duplicate cards from overlapping windows by citation id set."""
    merged: dict[tuple[int, ...], CardBoundary] = {}
    junk: set[int] = set()
    for result in results:
        junk.update(int(value) for value in result.get("junk", []))
        for raw in result.get("cards", []):
            card = CardBoundary.from_dict(raw)
            key = tuple(sorted(set(card.cite)))
            if not key:
                continue
            if key not in merged:
                merged[key] = card
                continue
            prior = merged[key]
            prior.tag = sorted(set(prior.tag) | set(card.tag))
            prior.body = sorted(set(prior.body) | set(card.body))
            prior.flags = sorted(set(prior.flags) | set(card.flags))
            if prior.header is None:
                prior.header = card.header

    cards = sorted(merged.values(), key=lambda card: min(card.cite))
    assigned = {
        value
        for card in cards
        for value in (
            ([] if card.header is None else [card.header])
            + card.tag
            + card.cite
            + card.body
        )
    }
    return {
        "cards": [card.to_dict() for card in cards],
        "junk": sorted(junk - assigned),
    }


def validate(result: dict[str, Any], records: list[ParagraphRecord]) -> dict[str, Any]:
    ids = {record.i for record in records}
    owners: dict[int, set[tuple[int, str]]] = {}
    invalid_ids: set[int] = set()
    malformed: list[int] = []
    cards = [CardBoundary.from_dict(card) for card in result.get("cards", [])]
    for number, card in enumerate(cards, start=1):
        if not card.cite:
            malformed.append(number)
        fields = {
            "header": [] if card.header is None else [card.header],
            "tag": card.tag,
            "cite": card.cite,
            "body": card.body,
        }
        for field, values in fields.items():
            for value in values:
                if value not in ids:
                    invalid_ids.add(value)
                owners.setdefault(value, set()).add((number, field))
    for value in result.get("junk", []):
        if value not in ids:
            invalid_ids.add(value)
        owners.setdefault(value, set()).add((0, "junk"))

    duplicates: list[int] = []
    for paragraph_id, assignments in owners.items():
        card_numbers = {number for number, _ in assignments}
        same_card_shared = (
            len(card_numbers) == 1
            and 0 not in card_numbers
            and {field for _, field in assignments} <= {"cite", "body"}
        )
        if len(assignments) > 1 and not same_card_shared:
            duplicates.append(paragraph_id)

    cite_like = sum(1 for record in records if citation_signal(record) >= 2)
    return {
        "unassigned": sorted(ids - set(owners)),
        "duplicate_assignment": sorted(duplicates),
        "invalid_paragraph_ids": sorted(invalid_ids),
        "cards_without_citation": malformed,
        "cards_found": len(cards),
        "citation_signal_count": cite_like,
        "count_divergence": cite_like - len(cards),
        "valid": not (
            ids - set(owners) or duplicates or invalid_ids or malformed
        ),
    }


def run_boundary_pass(
    records: list[ParagraphRecord],
    *,
    client: AnthropicJSONClient | None = None,
    use_model: bool = True,
    caller: Callable[[list[ParagraphRecord]], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if caller is None:
        if use_model and client is not None and client.available:
            caller = lambda chunk: model_boundary_pass(chunk, client)  # noqa: E731
            method = "llm"
        else:
            # Windowing a deterministic state machine creates artificial partial cards.
            result = heuristic_boundary_pass(records)
            return result, validate(result, records), "heuristic"
    else:
        method = "custom"
    result = merge([caller(chunk) for chunk in windows(records)])
    return result, validate(result, records), method


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from casefile.config import get_settings
    from casefile.llm import AnthropicJSONClient

    from .serialize_index import detect_convention, paragraph_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    path = Path(args.docx).resolve()
    records = paragraph_records(path)
    settings = get_settings()
    client = AnthropicJSONClient(settings.anthropic_api_key, settings.model)
    result, validation, method = run_boundary_pass(
        records, client=client, use_model=not args.no_model
    )
    convention, votes = detect_convention(records)
    print(
        json.dumps(
            {
                "source_file": path.name,
                "result": result,
                "validation": validation,
                "boundary_method": method,
                "marking_convention": convention,
                "marking_votes": votes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
