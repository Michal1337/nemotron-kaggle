"""Pick example problems from a training run by sorting mode.

Usage:
    uv run investigators/get_examples.py sort --min
    uv run investigators/get_examples.py sort --last
    uv run investigators/get_examples.py sort --min --logpath 04-08-02-34
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAINING_DIR = Path(__file__).parent.parent / "training" / "sft"
DEFAULT_COUNT = 10
RELAXED_CATS = {"cryptarithm_guess", "equation_numeric_guess"}
INCLUSION_THRESHOLD = -0.69


def latest_logpath() -> str:
    """Return the most recent training run directory name."""
    dirs = sorted(TRAINING_DIR.iterdir(), reverse=True)
    return dirs[0].name if dirs else ""


def load_problems(index_path: Path) -> dict[str, dict]:
    """Read training logprobs index - keep latest step per problem, skip duplicates."""
    problems: dict[str, dict] = {}
    with open(index_path) as f:
        for line in f:
            e = json.loads(line)
            pid = e["problem_id"]
            if "-" in pid:
                continue
            if pid not in problems or e["step"] > problems[pid]["step"]:
                problems[pid] = e
    return problems


def sort_min(
    problems: dict[str, dict],
    counts: dict[str, int],
    default_count: int,
) -> list[str]:
    """Sort by min logprob (lowest first)."""
    by_cat: dict[str, list[tuple[str, float]]] = {}
    for pid, info in problems.items():
        cat = info["category"]
        by_cat.setdefault(cat, []).append((pid, info["min_logprob"]))

    all_ids: list[str] = []
    print("problem_set = {")
    for cat in sorted(by_cat):
        n = counts.get(cat, default_count)
        items = sorted(by_cat[cat], key=lambda x: x[1])[:n]
        all_ids.extend(x[0] for x in items)
        print(f"    # {cat}")
        for i in range(0, len(items), 8):
            chunk = items[i : i + 8]
            ids_str = ", ".join(f'"{x[0]}"' for x in chunk)
            print(f"    {ids_str},")
    print("}")
    return all_ids


def sort_last(
    problems: dict[str, dict],
    counts: dict[str, int],
    default_count: int,
) -> list[str]:
    """Sort by latest step, filtered for min_logprob < INCLUSION_THRESHOLD."""
    by_cat: dict[str, list[tuple[str, float, int]]] = {}
    by_cat_relaxed: dict[str, list[tuple[str, float, int]]] = {}
    for pid, info in problems.items():
        cat = info["category"]
        entry = (pid, info["min_logprob"], info["step"])
        if info["min_logprob"] < INCLUSION_THRESHOLD:
            by_cat.setdefault(cat, []).append(entry)
        elif cat in RELAXED_CATS:
            by_cat_relaxed.setdefault(cat, []).append(entry)

    all_cats = sorted(set(by_cat) | set(by_cat_relaxed))
    all_ids: list[str] = []
    print("problem_set = {")
    for cat in all_cats:
        n = counts.get(cat, default_count)
        # Sort by step descending (latest first)
        items = sorted(by_cat.get(cat, []), key=lambda x: x[2], reverse=True)[:n]
        if len(items) < n and cat in RELAXED_CATS:
            seen = {x[0] for x in items}
            extra = sorted(
                by_cat_relaxed.get(cat, []), key=lambda x: x[2], reverse=True
            )
            items.extend(x for x in extra if x[0] not in seen)
            items = items[:n]
        all_ids.extend(x[0] for x in items)
        print(f"    # {cat}")
        for i in range(0, len(items), 8):
            chunk = items[i : i + 8]
            ids_str = ", ".join(f'"{x[0]}"' for x in chunk)
            print(f"    {ids_str},")
    print("}")
    return all_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sort_parser = sub.add_parser("sort")
    sort_parser.add_argument(
        "--logpath", default=None, help="Training run directory name"
    )
    sort_group = sort_parser.add_mutually_exclusive_group(required=True)
    sort_group.add_argument("--min", action="store_true", help="Sort by min logprob")
    sort_group.add_argument(
        "--last",
        action="store_true",
        help="Sort by latest step, filtered for logprob < INCLUSION_THRESHOLD",
    )
    args = parser.parse_args()

    if args.command != "sort":
        parser.print_help()
        return

    logpath = args.logpath or latest_logpath()
    index_path = TRAINING_DIR / logpath / "logprobs" / "index.jsonl"

    counts: dict[str, int] = {
        "bit_manipulation": DEFAULT_COUNT * 5,
        "equation_numeric_deduce": DEFAULT_COUNT * 3 // 2,
    }
    default_count = DEFAULT_COUNT

    problems = load_problems(index_path)
    print(f"# logpath: {logpath}")

    if getattr(args, "min"):
        sort_min(problems, counts, default_count)
    elif args.last:
        sort_last(problems, counts, default_count)


if __name__ == "__main__":
    main()
