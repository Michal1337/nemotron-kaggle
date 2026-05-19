"""Browse the Nemotron training data — pick problems by filter, see full details.

Reads from huikang's repo snapshot. Default paths assume the cluster layout
(/mnt/evafs/groups/re-com/mgromadzki/nemotron-master); override with --repo-root.

Usage examples
--------------
    # List all unsolved cryptarithm_deduce problems
    python src/inspect_problem.py --category cryptarithm_deduce --status rule_unknown

    # Show 10 random hard bit_manipulation problems
    python src/inspect_problem.py --category bit_manipulation \\
        --status rule_unknown --random 10

    # Full detail of one problem (prompt + reasoning + investigation)
    python src/inspect_problem.py --id 02664ad5

    # Per-category counts of problems missing from 04-08-16-14 corpus
    python src/inspect_problem.py --summary

    # Cycle through unsolved cryptarithm problems one at a time, press Enter for next
    python src/inspect_problem.py --status rule_unknown \\
        --category cryptarithm_deduce --browse
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path


DEFAULT_REPO_ROOT = "/mnt/evafs/groups/re-com/mgromadzki/nemotron-master"
DEFAULT_BASE_INDEX = "training/sft/04-08-16-14/logprobs/index.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=DEFAULT_REPO_ROOT,
                   help="huikang repo snapshot root (contains problems.jsonl, train.csv, reasoning/, etc.)")
    p.add_argument("--base-index", default=None,
                   help=f"Path to logprobs/index.jsonl to define 'in-corpus' (default: <repo-root>/{DEFAULT_BASE_INDEX})")

    # Selection
    p.add_argument("--id", help="Show one specific problem id")
    p.add_argument("--category", help="Filter by category (bit_manipulation, cipher, cryptarithm_deduce, ...)")
    p.add_argument("--status", choices=["rule_found", "rule_unknown", "hypothesis_formed"],
                   help="Filter by status in problems.jsonl")
    p.add_argument("--in-corpus", action="store_true", help="Only problems IN 04-08-16-14 corpus")
    p.add_argument("--not-in-corpus", action="store_true", help="Only problems NOT in 04-08-16-14 corpus")

    # Output mode
    p.add_argument("--summary", action="store_true",
                   help="Print per-category counts of matching problems (no details)")
    p.add_argument("--list", action="store_true",
                   help="Print one-line per matching problem (no full details)")
    p.add_argument("--random", type=int, default=0,
                   help="Sample N random matching problems, show full details for each")
    p.add_argument("--limit", type=int, default=20, help="Max problems to show in --list mode")
    p.add_argument("--browse", action="store_true",
                   help="Show one at a time; press Enter to advance, q to quit")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --random")
    p.add_argument("--no-reasoning", action="store_true", help="Suppress reasoning/investigation text")

    return p.parse_args()


def load_problems(repo: Path) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in (repo / "problems.jsonl").open()}


def load_train_csv(repo: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (repo / "train.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {"prompt": row["prompt"], "answer": row["answer"]}
    return out


def load_corpus_pids(index_path: Path) -> set[str]:
    s: set[str] = set()
    if not index_path.is_file():
        return s
    with index_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("epoch", 0) == 0:
                s.add(r["problem_id"])
    return s


def filter_problems(problems: dict, train: dict, in_corpus: set, args) -> list[dict]:
    out = []
    for pid, p in problems.items():
        if args.category and p.get("category") != args.category:
            continue
        if args.status and p.get("status") != args.status:
            continue
        if args.in_corpus and pid not in in_corpus:
            continue
        if args.not_in_corpus and pid in in_corpus:
            continue
        if pid not in train:
            continue
        out.append({**p, **train[pid], "in_corpus": pid in in_corpus})
    return sorted(out, key=lambda x: x["id"])


def hr(c="─", n=100) -> str:
    return c * n


def show_one(p: dict, repo: Path, show_reasoning: bool) -> None:
    print()
    print(hr("="))
    print(f"  ID: {p['id']}   category: {p['category']}   status: {p['status']}   "
          f"in_corpus: {p['in_corpus']}")
    if p.get("submission"):
        print(f"  huikang's submission: {p['submission']!r}")
    print(hr("="))
    print("\n[PROMPT]")
    print(p["prompt"])
    print(f"\n[GOLD ANSWER]  {p['answer']!r}")

    if show_reasoning:
        reasoning_path = repo / "reasoning" / f"{p['id']}.txt"
        if reasoning_path.is_file():
            text = reasoning_path.read_text()
            print(f"\n[REASONING] ({reasoning_path.name}, {len(text)} chars)")
            print(hr("─"))
            lines = text.splitlines()
            head = lines[:40]
            tail = lines[-10:] if len(lines) > 50 else []
            print("\n".join(head))
            if tail:
                print(f"... ({len(lines) - 50} lines elided) ...")
                print("\n".join(tail))

        inv_path = repo / "investigations" / f"{p['id']}.txt"
        if inv_path.is_file():
            text = inv_path.read_text()
            print(f"\n[INVESTIGATION] ({inv_path.name}, {len(text)} chars)")
            print(hr("─"))
            lines = text.splitlines()
            print("\n".join(lines[:40]))
            if len(lines) > 40:
                print(f"... ({len(lines) - 40} more lines) ...")

        prob_detail_path = repo / "problems" / f"{p['id']}.jsonl"
        if prob_detail_path.is_file():
            try:
                detail = json.loads(prob_detail_path.read_text().splitlines()[0])
                if detail.get("question"):
                    print(f"\n[QUESTION (test input)]  {detail['question']!r}")
                if "examples" in detail:
                    print(f"\n[EXAMPLES] ({len(detail['examples'])})")
                    for i, ex in enumerate(detail["examples"][:8]):
                        inp = ex.get("input_value", ex.get("input"))
                        out = ex.get("output_value", ex.get("output"))
                        print(f"  {i}: {inp!r}  ->  {out!r}")
                    if len(detail["examples"]) > 8:
                        print(f"  ... ({len(detail['examples']) - 8} more)")
            except (json.JSONDecodeError, IndexError):
                pass
    print()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root)
    if not repo.is_dir():
        sys.exit(f"--repo-root not found: {repo}")
    base_index = Path(args.base_index) if args.base_index else (repo / DEFAULT_BASE_INDEX)

    problems = load_problems(repo)
    train = load_train_csv(repo)
    in_corpus = load_corpus_pids(base_index)

    # --id mode: show single problem (ignores most filters)
    if args.id:
        if args.id not in problems:
            sys.exit(f"id {args.id!r} not in problems.jsonl")
        if args.id not in train:
            sys.exit(f"id {args.id!r} not in train.csv")
        rec = {**problems[args.id], **train[args.id], "in_corpus": args.id in in_corpus}
        show_one(rec, repo, not args.no_reasoning)
        return

    matching = filter_problems(problems, train, in_corpus, args)

    if args.summary:
        print(f"\nMatching: {len(matching):,} problems\n")
        print(f"  filters: category={args.category}  status={args.status}  "
              f"in_corpus={args.in_corpus}  not_in_corpus={args.not_in_corpus}\n")
        cat = Counter(p["category"] for p in matching)
        status = Counter(p["status"] for p in matching)
        ic = Counter(p["in_corpus"] for p in matching)
        print("by category:")
        for c, n in cat.most_common():
            print(f"  {c:>24s}  {n:>5d}")
        print("\nby status:")
        for s, n in status.most_common():
            print(f"  {s:>24s}  {n:>5d}")
        print("\nby in_corpus:")
        for k, n in ic.most_common():
            print(f"  {str(k):>24s}  {n:>5d}")
        return

    if not matching:
        print("No problems matched filters.")
        return

    if args.random:
        rng = random.Random(args.seed)
        matching = rng.sample(matching, min(args.random, len(matching)))

    if args.list:
        for p in matching[: args.limit]:
            print(f"  {p['id']}  {p['category']:>24s}  {p['status']:>20s}  "
                  f"in_corpus={p['in_corpus']!s:>5s}  answer={p['answer']!r}")
        if len(matching) > args.limit:
            print(f"\n... ({len(matching) - args.limit} more — increase --limit)")
        print(f"\nTotal: {len(matching):,}")
        return

    if args.browse:
        for i, p in enumerate(matching):
            show_one(p, repo, not args.no_reasoning)
            print(f"[{i + 1}/{len(matching)}] press Enter for next, 'q' then Enter to quit")
            if input().strip().lower() == "q":
                return
        return

    # Default: show all matching (capped at --limit)
    for p in matching[: args.limit]:
        show_one(p, repo, not args.no_reasoning)
    if len(matching) > args.limit:
        print(f"\n... ({len(matching) - args.limit} more — increase --limit, "
              f"use --list for one-line view, or --browse to step through)")


if __name__ == "__main__":
    main()
