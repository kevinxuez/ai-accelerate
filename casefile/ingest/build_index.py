"""Validate card and rule retrieval from committed JSON/Markdown sources."""

from __future__ import annotations

from casefile.retrieval import CaseFileIndex


def main() -> None:
    index = CaseFileIndex()
    cards = index.rebuild_cards()
    rules = index.rebuild_rules()
    print(f"backend={index.backend} cards={cards} rule_chunks={rules}")


if __name__ == "__main__":
    main()
