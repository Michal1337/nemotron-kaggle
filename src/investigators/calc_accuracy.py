"""One-off script to calculate accuracy by category.

Usage:
    uv run python3 investigators/calc_accuracy.py v20.csv v26.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

from build_generation_index import verify

csv.field_size_limit(sys.maxsize)


def load_categories(path: str) -> dict[str, str]:
    """Load id -> category mapping from problems.jsonl."""
    mapping = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            mapping[obj["id"]] = obj["category"]
    return mapping


def calc_accuracy(csv_path: str, categories: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Return {category: (correct, total)} from a CSV file."""
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["id"]
            cat = categories.get(rid)
            if cat is None:
                continue
            answer = row["answer"]
            predicted = row["predicted"]
            correct = verify(str(answer), str(predicted))
            stats[cat][1] += 1
            if correct:
                stats[cat][0] += 1
    return {k: tuple(v) for k, v in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="CSV files to evaluate (e.g. v20.csv v26.csv)")
    args = parser.parse_args()

    categories = load_categories("problems.jsonl")

    category_order = [
        "numeral",
        "unit_conversion",
        "gravity",
        "cipher",
        "bit_manipulation",
        "equation_numeric_deduce",
        "equation_numeric_guess",
        "cryptarithm_deduce",
        "cryptarithm_guess",
    ]

    for path in args.csvs:
        version = path.removesuffix(".csv")
        stats = calc_accuracy(path, categories)
        total_correct = sum(v[0] for v in stats.values())
        total_count = sum(v[1] for v in stats.values())
        print(f"\n{'=' * 60}")
        print(f"  {version}  ({path})")
        print(f"{'=' * 60}")
        print(f"{'Category':<30} {'Correct':>7} {'Total':>7} {'Accuracy':>8}")
        print(f"{'-' * 60}")
        for cat in category_order:
            if cat in stats:
                c, t = stats[cat]
                pct = c / t * 100 if t else 0
                print(f"{cat:<30} {c:>7} {t:>7} {pct:>7.1f}%")
        print(f"{'-' * 60}")
        pct = total_correct / total_count * 100 if total_count else 0
        print(f"{'TOTAL':<30} {total_correct:>7} {total_count:>7} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
