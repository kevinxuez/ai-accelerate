"""Model-only card labeling and deterministic citation parsing."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from casefile.agents.contracts import CardLabelOutput
from casefile.agents.errors import CaseFileError, ErrorCode
from casefile.agents.prompt_registry import load_prompt
from casefile.llm import AnthropicJSONClient


URL = re.compile(r"https?://[^\s<>()]+", re.I)
YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
DATE = re.compile(
    r"\b(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}|"
    r"\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.I,
)
ACCESS = re.compile(r"(?:Accessed|DOA)\s*[:.]?\s*([^,;\n]+(?:\d{4})?)", re.I)
BARE_URL = re.compile(r"\s*(?:https?://|www\.)\S+\s*", re.I)
ORGANIZATION_WORDS = {
    "commission",
    "committee",
    "department",
    "federal",
    "institute",
    "organization",
    "university",
    "bank",
    "council",
    "association",
    "agency",
    "office",
    "ministry",
}


def _clean_url(value: str) -> str:
    return value.rstrip('.,;)]}"')


def parse_citation(cite: str, header: str) -> dict[str, Any]:
    urls = URL.findall(cite)
    url = _clean_url(urls[0]) if urls else ""
    years = YEAR.findall(cite)
    year = int(years[0]) if years else None
    if year is None:
        shorthand = re.search(r"[‘’']\s?(\d{2})\b", header)
        if shorthand:
            short = int(shorthand.group(1))
            year = 2000 + short if short <= 50 else 1900 + short
    date_match = DATE.search(cite)
    access_match = ACCESS.search(cite)

    author = ""
    prefix = re.split(r"[,.\n]", cite, maxsplit=1)[0].strip()
    if prefix and not BARE_URL.fullmatch(prefix):
        author = prefix
    elif header:
        author = re.sub(r"\s*[‘’']\s?\d{2}.*$", "", header).strip()
    lowered = author.lower()
    author_type = (
        "organization"
        if any(word in lowered.split() for word in ORGANIZATION_WORDS)
        or len(author.split()) > 4
        else "person"
        if author
        else "unknown"
    )

    source = urlparse(url).netloc.removeprefix("www.") if url else ""
    quoted = re.findall(r"[\"“]([^\"”]{4,})[\"”]", cite)
    if quoted:
        tail = cite[cite.find(quoted[-1]) + len(quoted[-1]) :]
        for candidate in [part.strip(" ,.") for part in tail.split(",")]:
            if candidate and not YEAR.search(candidate) and not URL.search(candidate):
                source = candidate
                break
    return {
        "author": author,
        "author_type": author_type,
        "year": year,
        "date_raw": date_match.group(0) if date_match else "",
        "source": source,
        "url": url,
        "date_accessed": access_match.group(1).strip() if access_match else "",
        "cutter": "",
    }


def label_card(
    *,
    header: str,
    tag: str,
    citation: str,
    body: str,
    source_filename: str,
    default_side: str,
    validation_flags: list[str],
    client: AnthropicJSONClient,
) -> CardLabelOutput:
    prompt = load_prompt("evidence_librarian_labels")
    labels = client.complete_json(
        system=prompt.content,
        user=json.dumps(
            {
                "header": header,
                "tag": tag,
                "citation": citation,
                "body": body,
                "source_filename": source_filename,
                "default_side": default_side,
                "deterministic_validation_flags": validation_flags,
            },
            ensure_ascii=False,
        ),
        max_tokens=1_000,
        schema=CardLabelOutput,
        agent="evidence_librarian",
        prompt_template=prompt.template_name,
        prompt_version=prompt.version,
    )
    try:
        return CardLabelOutput.model_validate(labels)
    except ValidationError as exc:
        raise CaseFileError(
            ErrorCode.MODEL_OUTPUT_INVALID,
            "The Evidence Librarian returned card labels that did not match CardLabelOutput.",
            stage="ingestion.label_cards",
            agent="evidence_librarian",
            safe_details={"schema": "CardLabelOutput"},
            cause=exc,
        ) from exc
