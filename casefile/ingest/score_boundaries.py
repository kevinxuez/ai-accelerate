"""Score exact cite/body boundaries and reject mismatched source benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def key(card: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sorted(card.get("cite") or [])), tuple(sorted(card.get("body") or []))


def score(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    predicted_source = Path(prediction.get("source_file", "")).name
    truth_source = Path(truth.get("file", truth.get("source_file", ""))).name
    if predicted_source and truth_source and predicted_source != truth_source:
        raise ValueError(
            f"Benchmark source mismatch: prediction={predicted_source!r}, truth={truth_source!r}"
        )
    predicted = prediction.get("result", prediction).get("cards", [])
    expected = truth.get("cards", [])
    predicted_keys = {key(card): card for card in predicted}
    expected_keys = {key(card): card for card in expected}
    exact_keys = set(predicted_keys) & set(expected_keys)
    cite_pred = {tuple(sorted(card.get("cite") or [])): card for card in predicted}
    cite_truth = {tuple(sorted(card.get("cite") or [])): card for card in expected}
    header_correct = sum(
        1
        for boundary in exact_keys
        if (predicted_keys[boundary].get("header") or None)
        == (expected_keys[boundary].get("header") or None)
    )
    return {
        "ground_truth_cards": len(expected),
        "predicted_cards": len(predicted),
        "exact": len(exact_keys),
        "exact_rate": len(exact_keys) / len(expected) if expected else 0.0,
        "citation_matched": len(set(cite_pred) & set(cite_truth)),
        "header_correct_on_exact": header_correct,
        "missed": [
            expected_keys[value] for value in set(expected_keys) - set(predicted_keys)
        ],
        "spurious": [
            predicted_keys[value] for value in set(predicted_keys) - set(expected_keys)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction")
    parser.add_argument("truth")
    args = parser.parse_args()
    prediction = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    result = score(prediction, truth)
    print(f"ground truth cards : {result['ground_truth_cards']}")
    print(f"predicted cards    : {result['predicted_cards']}")
    print(f"exact boundary     : {result['exact']} ({result['exact_rate']:.0%})")
    print(f"citation matched   : {result['citation_matched']}")
    print(
        f"header correct     : {result['header_correct_on_exact']}/{result['exact']} of exact"
    )
    print(f"missed             : {len(result['missed'])}")
    print(f"spurious           : {len(result['spurious'])}")


if __name__ == "__main__":
    main()
