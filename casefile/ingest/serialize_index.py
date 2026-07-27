"""Serialize DOCX paragraphs without normalizing or reproducing evidence text.

Only the standard library is used. Paragraph indices match the original ``w:p`` order,
including gaps caused by empty paragraphs, which keeps model boundaries auditable.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from casefile.models import ParagraphRecord, RunRecord


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TAG = lambda name: f"{{{W}}}{name}"  # noqa: E731
NOISE_HIGHLIGHT = {"WHITE", "AUTO", "NONE"}


class DocxFormatError(ValueError):
    pass


def _on(element: ET.Element | None, property_name: str) -> bool:
    if element is None:
        return False
    node = element.find(TAG(property_name))
    if node is None:
        return False
    return node.get(TAG("val"), "true").lower() not in {"0", "false", "off", "none"}


def _run_text(run: ET.Element) -> str:
    parts: list[str] = []
    for node in run.iter():
        if node.tag in {TAG("t"), TAG("delText")}:
            parts.append(node.text or "")
        elif node.tag == TAG("tab"):
            parts.append("\t")
        elif node.tag in {TAG("br"), TAG("cr")}:
            parts.append("\n")
        elif node.tag == TAG("noBreakHyphen"):
            parts.append("‑")
        elif node.tag == TAG("softHyphen"):
            parts.append("\u00ad")
    return "".join(parts)


def _styles(archive: zipfile.ZipFile) -> dict[str, str]:
    try:
        root = ET.fromstring(archive.read("word/styles.xml"))
    except KeyError:
        return {}
    names: dict[str, str] = {}
    for style in root.findall(TAG("style")):
        style_id = style.get(TAG("styleId"), "")
        name = style.find(TAG("name"))
        if style_id:
            names[style_id] = (
                name.get(TAG("val"), style_id) if name is not None else style_id
            ).lower()
    return names


def _paragraph_style(paragraph: ET.Element, styles: dict[str, str]) -> str:
    ppr = paragraph.find(TAG("pPr"))
    node = ppr.find(TAG("pStyle")) if ppr is not None else None
    style_id = node.get(TAG("val"), "Normal") if node is not None else "Normal"
    return styles.get(style_id, style_id).lower()


def _visible_runs(paragraph: ET.Element) -> list[ET.Element]:
    # Runs in deleted revisions are intentionally excluded; ingesting deleted evidence
    # would surface text that is not visible in the source document.
    deleted = {id(node) for deletion in paragraph.findall(f".//{TAG('del')}") for node in deletion.iter(TAG("r"))}
    return [run for run in paragraph.iter(TAG("r")) if id(run) not in deleted]


def paragraph_records(path: str | Path) -> list[ParagraphRecord]:
    """Return one record per non-empty visible paragraph in a DOCX."""
    source = Path(path)
    if source.suffix.lower() != ".docx":
        raise DocxFormatError(f"Expected a .docx file: {source}")
    try:
        with zipfile.ZipFile(source) as archive:
            styles = _styles(archive)
            root = ET.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise DocxFormatError(f"Invalid DOCX package: {source}") from exc

    output: list[ParagraphRecord] = []
    for index, paragraph in enumerate(root.iter(TAG("p"))):
        raw_runs = _visible_runs(paragraph)
        text_parts: list[str] = []
        runs: list[RunRecord] = []
        offset = 0
        for run in raw_runs:
            text = _run_text(run)
            if not text:
                continue
            properties = run.find(TAG("rPr"))
            highlight_node = properties.find(TAG("highlight")) if properties is not None else None
            highlight = (
                highlight_node.get(TAG("val"), "").upper()
                if highlight_node is not None
                else None
            )
            if highlight in NOISE_HIGHLIGHT:
                highlight = None
            record = RunRecord(
                text=text,
                start=offset,
                end=offset + len(text),
                bold=_on(properties, "b"),
                underline=_on(properties, "u"),
                highlight=highlight,
            )
            runs.append(record)
            text_parts.append(text)
            offset = record.end
        text = "".join(text_parts)
        if not text.strip():
            continue
        total = sum(len(run.text) for run in runs) or 1
        highlights: dict[str, int] = {}
        for run in runs:
            if run.highlight:
                highlights[run.highlight] = highlights.get(run.highlight, 0) + len(run.text)
        output.append(
            ParagraphRecord(
                i=index,
                style=_paragraph_style(paragraph, styles),
                text=text,
                length=len(text),
                bold_fraction=round(
                    sum(len(run.text) for run in runs if run.bold) / total, 2
                ),
                underline_fraction=round(
                    sum(len(run.text) for run in runs if run.underline) / total, 2
                ),
                highlights={
                    color: round(length / total, 2)
                    for color, length in sorted(highlights.items())
                },
                runs=runs,
                link=bool(re.search(r"https?://|doi\.org", text, re.I)),
            )
        )
    return output


def detect_convention(records: list[ParagraphRecord]) -> tuple[str, dict[str, int]]:
    """Vote for the read-marking attribute using body-like paragraphs only."""
    bodies = [record for record in records if record.length > 200]
    scores = {
        "bold": sum(1 for record in bodies if 0.05 < record.bold_fraction < 0.95),
        "underline": sum(
            1 for record in bodies if 0.05 < record.underline_fraction < 0.95
        ),
    }
    for record in bodies:
        for color, fraction in record.highlights.items():
            if 0.05 < fraction < 0.95:
                scores[color] = scores.get(color, 0) + 1
    # A deterministic tie break avoids silently changing convention across runtimes.
    rank = {"bold": 3, "underline": 2, "YELLOW": 1}
    winner = max(scores, key=lambda key: (scores[key], rank.get(key, 0), key))
    return winner, scores


def render_index(records: list[ParagraphRecord], preview: int = 60) -> str:
    """Return the compact, non-authoritative view sent to the boundary model."""
    lines = []
    for record in records:
        excerpt = record.text[:preview].replace("\n", " ")
        highlights = ",".join(
            f"{color}={fraction}" for color, fraction in record.highlights.items()
        ) or "-"
        lines.append(
            f"[{record.i:3}] {record.style[:8]:<8} len={record.length:<5} "
            f"b={record.bold_fraction:.2f} u={record.underline_fraction:.2f} "
            f"hl={highlights:<10} {'URL' if record.link else '   '} \"{excerpt}\""
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx")
    args = parser.parse_args()
    records = paragraph_records(args.docx)
    convention, votes = detect_convention(records)
    print(f"# paragraphs: {len(records)}")
    print(f"# read-marking convention: {convention} votes={votes}")
    print(render_index(records))


if __name__ == "__main__":
    main()

