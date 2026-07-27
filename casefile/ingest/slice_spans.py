"""Slice original paragraph/run records into card text and marking spans."""

from __future__ import annotations

from casefile.models import ParagraphRecord


def _merge_spans(spans: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def join_paragraphs(ids: list[int], by_id: dict[int, ParagraphRecord]) -> str:
    return "\n".join(by_id[index].text for index in ids if index in by_id)


def spans_for_paragraphs(
    ids: list[int],
    by_id: dict[int, ParagraphRecord],
    convention: str,
) -> tuple[str, list[list[int]], list[list[int]]]:
    text_parts: list[str] = []
    read: list[list[int]] = []
    emphasis: list[list[int]] = []
    offset = 0
    for position, paragraph_id in enumerate(ids):
        record = by_id[paragraph_id]
        text_parts.append(record.text)
        for run in record.runs:
            if convention == "bold":
                marked = run.bold
            elif convention == "underline":
                marked = run.underline
            else:
                marked = run.highlight == convention
            if marked:
                read.append([offset + run.start, offset + run.end])
            if run.highlight == "YELLOW":
                emphasis.append([offset + run.start, offset + run.end])
        offset += len(record.text)
        if position < len(ids) - 1:
            offset += 1
    return "\n".join(text_parts), _merge_spans(read), _merge_spans(emphasis)


def span_text(text: str, spans: list[list[int]]) -> str:
    return " ".join(text[start:end] for start, end in spans).strip()


def card_convention(
    body_ids: list[int], by_id: dict[int, ParagraphRecord], document_default: str
) -> str:
    """Choose a per-card override only when its body supplies clear evidence."""
    scores: dict[str, int] = {"bold": 0, "underline": 0}
    for paragraph_id in body_ids:
        record = by_id[paragraph_id]
        if record.bold_fraction > 0:
            scores["bold"] += round(record.bold_fraction * record.length)
        if record.underline_fraction > 0:
            scores["underline"] += round(record.underline_fraction * record.length)
        for color, fraction in record.highlights.items():
            scores[color] = scores.get(color, 0) + round(fraction * record.length)
    winner, amount = max(scores.items(), key=lambda item: (item[1], item[0]))
    return winner if amount > 0 else document_default

