"""Flip problems.jsonl status to rule_found for pids with a reasoning file.

After running ``investigations_to_reasoning.py`` to narrate newly-solved
problems, those problems still show as ``hypothesis_formed`` or
``rule_unknown`` in problems.jsonl. ``build_extended_corpus.py`` only picks
up pids whose status is ``rule_found``, so we need a small bridge.

This script flips status -> rule_found for problems in the chosen categories
that satisfy BOTH:
  - reasoning/<pid>.txt exists, AND
  - investigations/<category>/correct/<pid>.txt exists (gold-verified).

A timestamped backup is written next to the original.

Usage:
    python src/flip_status_to_rule_found.py \
        --repo-root $NEMO \
        --categories bit_manipulation \
        --dry-run         # remove --dry-run to actually rewrite problems.jsonl
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, type=Path,
                    help="nemotron-master root (contains problems.jsonl, reasoning/, investigations/).")
    ap.add_argument("--categories", nargs="+", required=True,
                    help="Categories to consider (e.g. bit_manipulation).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    problems_path = args.repo_root / "problems.jsonl"
    reasoning_dir = args.repo_root / "reasoning"
    inv_root = args.repo_root / "investigations"

    if not problems_path.is_file():
        sys.exit(f"no problems.jsonl at {problems_path}")

    recs = []
    flipped: list[tuple[str, str, str]] = []  # (pid, category, old_status)
    skipped_no_reasoning = 0
    skipped_no_inv = 0

    with problems_path.open() as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                recs.append(line.rstrip("\n"))
                continue
            cat = o.get("category")
            if cat in args.categories and o.get("status") != "rule_found":
                pid = o["id"]
                has_reasoning = (reasoning_dir / f"{pid}.txt").is_file()
                has_inv = (inv_root / cat / "correct" / f"{pid}.txt").is_file()
                if has_reasoning and has_inv:
                    flipped.append((pid, cat, o.get("status", "unknown")))
                    o["status"] = "rule_found"
                elif not has_reasoning:
                    skipped_no_reasoning += 1
                else:
                    skipped_no_inv += 1
            recs.append(json.dumps(o) if isinstance(o, dict) else o)

    print(f"would flip {len(flipped)} pids to rule_found:")
    by_cat: dict[str, int] = {}
    by_old_status: dict[str, int] = {}
    for _, c, s in flipped:
        by_cat[c] = by_cat.get(c, 0) + 1
        by_old_status[s] = by_old_status.get(s, 0) + 1
    for c, n in sorted(by_cat.items()):
        print(f"  {c}: {n}")
    print(f"  from old statuses: {by_old_status}")
    print(f"skipped (no reasoning file): {skipped_no_reasoning}")
    print(f"skipped (no investigation):  {skipped_no_inv}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    if not flipped:
        print("nothing to flip; exiting")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = problems_path.with_suffix(f".jsonl.bak-{ts}")
    shutil.copy(problems_path, backup)
    print(f"\nbacked up -> {backup.name}")

    with problems_path.open("w") as f:
        for r in recs:
            f.write(r + "\n")
    print(f"wrote {problems_path} with {len(flipped)} statuses flipped")


if __name__ == "__main__":
    main()
