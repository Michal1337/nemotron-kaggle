"""Rewrite logprobs/index.jsonl so categories are mixed across batches.

build_extended_corpus.py preserves the base 04-08-16-14 order and *appends*
the new rule_found problems at the end. That means the final batches are pure
cryptarithm / equation_numeric — bad for SFT. This script reshuffles so each
batch is a balanced mix across categories.

Usage:
    python src/restratify_index.py <corpus_root> [--batch 32] [--seed 0]

Writes <corpus_root>/logprobs/index.jsonl in-place (after backing up the
original to index.jsonl.bak the first time).
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_path = args.corpus_root / "logprobs" / "index.jsonl"
    if not index_path.is_file():
        sys.exit(f"no index at {index_path}")
    backup = index_path.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy(index_path, backup)
        print(f"backed up original -> {backup.name}")

    recs = []
    with index_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("epoch", 0) != 0:
                continue
            recs.append(r)
    n = len(recs)
    print(f"epoch-0 records: {n}")

    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        by_cat[r.get("category", "unknown")].append(i)
    print("by category:")
    for c, idxs in sorted(by_cat.items()):
        print(f"  {c}: {len(idxs)}")

    n_batches = (n + args.batch - 1) // args.batch
    rng = random.Random(args.seed)
    for idxs in by_cat.values():
        rng.shuffle(idxs)
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    batches: list[list[int]] = [[] for _ in range(n_batches)]
    assigned = 0
    # Round-robin each category across batches in the shuffled order.
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    new_order: list[int] = []
    for b in batches:
        new_order.extend(b)

    print(f"\nstratification check (first 5 batches):")
    for bi in range(min(5, n_batches)):
        cats = Counter(recs[i]["category"] for i in batches[bi])
        print(f"  batch {bi}: {dict(cats)}")

    with index_path.open("w") as f:
        for i in new_order:
            f.write(json.dumps(recs[i]) + "\n")
    print(f"\nwrote {index_path} ({len(new_order)} records, {n_batches} batches)")


if __name__ == "__main__":
    main()
