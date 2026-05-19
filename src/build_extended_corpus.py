"""Extend the 04-08-16-14 corpus with newly-solved hard-category problems.

huikang's 04-08-16-14 corpus excludes ~2,163 ``rule_found`` problems
(deliberately downsampled numeral/gravity/unit_conversion). Between then
and his latest snapshot he improved the per-category reasoners so problems
that were ``rule_unknown`` in 04-08 are now ``rule_found`` with valid
``reasoning/<pid>.txt`` files.

This script writes a NEW corpus that:
  * Keeps every example from 04-08-16-14 unchanged (symlinks tokens, preserves
    its index.jsonl entries).
  * Adds ``rule_found`` problems in the selected hard categories that were
    NOT in 04-08-16-14. By default: cryptarithm_deduce, cryptarithm_guess,
    equation_numeric_guess, equation_numeric_deduce, bit_manipulation —
    categories where huikang's reasoners improved over time.
  * Does NOT add gravity / numeral / unit_conversion (huikang deliberately
    excluded those — keeping his downsampling).
  * Does NOT add any synthetic-only categories (matching/concatenation/etc.) —
    those torpedoed the 04-10-04-33 fresh-init run.

Output layout matches what train_huikang_style.py expects:
    <output>/tokens/<pid>/synthetic.json   (with {tokens: [...], mask: [...]})
    <output>/logprobs/index.jsonl

Example
-------
    python src/build_extended_corpus.py \\
        --base-tokens /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/04-08-16-14/tokens \\
        --base-index  /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl \\
        --reasoning-dir /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/reasoning \\
        --train-csv /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/train.csv \\
        --problems-jsonl /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/problems.jsonl \\
        --tokenizer /mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16 \\
        --output data/custom-corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# Categories where huikang's reasoners improved between 04-08 and the current
# snapshot. Adding these to the 04-08 base is a "free" corpus extension that
# doesn't dilute the harder-category gradient.
DEFAULT_ADD_CATEGORIES = (
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_guess",
    "equation_numeric_deduce",
    "bit_manipulation",
)

# Matches metric_reference.py / corpus.py — appended to every prompt before
# tokenization so the prompt format matches what the scorer feeds.
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

TOKEN_LIMIT = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-tokens", required=True,
                   help="Existing tokens dir to inherit from (e.g. 04-08-16-14/tokens).")
    p.add_argument("--base-index", required=True,
                   help="Existing index.jsonl to inherit from.")
    p.add_argument("--reasoning-dir", required=True,
                   help="Directory of reasoning/{pid}.txt files (huikang's current snapshot).")
    p.add_argument("--train-csv", required=True,
                   help="Competition train.csv with id,prompt,answer.")
    p.add_argument("--problems-jsonl", required=True,
                   help="problems.jsonl with id/category/status.")
    p.add_argument("--tokenizer", required=True,
                   help="Path to base model (loaded via AutoTokenizer.from_pretrained).")
    p.add_argument("--output", required=True,
                   help="Output dir; writes <output>/tokens/{pid}/synthetic.json + <output>/logprobs/index.jsonl.")
    p.add_argument("--add-categories", default=",".join(DEFAULT_ADD_CATEGORIES),
                   help="Comma-separated categories to add. Default is the 5 hard categories.")
    p.add_argument("--copy-instead-of-symlink", action="store_true",
                   help="Copy existing tokens instead of symlinking. Slower, uses disk.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be added without writing anything.")
    return p.parse_args()


def load_problems(path: str) -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def load_train_csv(path: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = (row["prompt"], row["answer"])
    return out


def load_base_index(path: str) -> tuple[list[dict], set[str]]:
    """Return (epoch-0 records in original order, set of pids in epoch 0)."""
    recs: list[dict] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("epoch", 0) != 0:
                continue
            pid = r["problem_id"]
            if pid in seen:
                continue
            seen.add(pid)
            recs.append(r)
    return recs, seen


def tokenize_problem(pid: str, prompts_answers: dict, reasoning_dir: str,
                     chat_tok, problems: dict) -> dict | None:
    """Return {tokens, mask} ready for training, or None if pid can't be tokenized."""
    if pid not in prompts_answers:
        return None
    reasoning_path = Path(reasoning_dir) / f"{pid}.txt"
    if not reasoning_path.is_file():
        return None
    prompt_text, gold_answer = prompts_answers[pid]
    reasoning_text = reasoning_path.read_text().rstrip("\n")

    boxed = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
    reasoning_answer = boxed[-1] if boxed else gold_answer
    completion_text = (
        f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
    )

    messages = [{"role": "user", "content": prompt_text + PROMPT_SUFFIX}]
    prompt_ids = chat_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    completion_ids = chat_tok.encode(completion_text, add_special_tokens=False)

    all_tokens = list(prompt_ids) + list(completion_ids)
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if len(all_tokens) > TOKEN_LIMIT:
        all_tokens = all_tokens[:TOKEN_LIMIT]
        mask = mask[:TOKEN_LIMIT]
    if not any(mask):
        return None
    return {"tokens": all_tokens, "mask": mask}


def main() -> None:
    args = parse_args()

    problems = load_problems(args.problems_jsonl)
    prompts_answers = load_train_csv(args.train_csv)
    base_recs, base_pids = load_base_index(args.base_index)
    add_cats = set(args.add_categories.split(","))

    candidates = sorted(
        pid for pid, p in problems.items()
        if p.get("status") == "rule_found"
        and p.get("category") in add_cats
        and pid not in base_pids
        and pid in prompts_answers
        and (Path(args.reasoning_dir) / f"{pid}.txt").is_file()
    )

    print(f"Base corpus (from {Path(args.base_index).parent.name}): {len(base_pids):,} pids")
    print(f"Adding categories: {sorted(add_cats)}")
    print(f"Candidates to add (rule_found, not in base, has reasoning): {len(candidates):,}")
    print(f"  by category:")
    cat_counts = Counter(problems[p]["category"] for p in candidates)
    for c, n in cat_counts.most_common():
        print(f"    {c}: {n}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokens_dir = output / "tokens"
    logprobs_dir = output / "logprobs"
    tokens_dir.mkdir(exist_ok=True)
    logprobs_dir.mkdir(exist_ok=True)

    # 1. Mirror existing tokens (symlink unless --copy)
    print(f"\nMirroring {len(base_pids):,} existing token dirs...")
    base_tokens = Path(args.base_tokens).resolve()
    if not base_tokens.is_dir():
        sys.exit(f"--base-tokens not found: {base_tokens}")
    n_mirrored = 0
    for pid in base_pids:
        src = base_tokens / pid
        if not src.is_dir():
            continue
        dst = tokens_dir / pid
        if dst.exists() or dst.is_symlink():
            continue
        if args.copy_instead_of_symlink:
            import shutil
            shutil.copytree(src, dst)
        else:
            os.symlink(src, dst, target_is_directory=True)
        n_mirrored += 1
    print(f"  mirrored {n_mirrored} dirs")

    # 2. Tokenize and write the new problems
    print(f"\nTokenizing {len(candidates):,} new problems...")
    from transformers import AutoTokenizer
    chat_tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    new_recs: list[dict] = []
    skipped = 0
    for i, pid in enumerate(candidates):
        result = tokenize_problem(pid, prompts_answers, args.reasoning_dir,
                                  chat_tok, problems)
        if result is None:
            skipped += 1
            continue
        out_dir = tokens_dir / pid
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "synthetic.json", "w") as f:
            json.dump(result, f)
        new_recs.append({
            "epoch": 0,
            "step": -1,  # placeholder; train_huikang_style.py doesn't read this
            "problem_id": pid,
            "segment": "synthetic.jsonl",
            "category": problems[pid]["category"],
            "num_loss_tokens": sum(result["mask"]),
            "total_loss": 0.0,
            "min_logprob": 0.0,
        })
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(candidates)}")
    print(f"  wrote {len(new_recs)} new token files; skipped {skipped}")

    # 3. Write merged index.jsonl preserving original order, appending new entries
    index_path = logprobs_dir / "index.jsonl"
    with open(index_path, "w") as f:
        for r in base_recs:
            f.write(json.dumps(r) + "\n")
        for r in new_recs:
            f.write(json.dumps(r) + "\n")
    total = len(base_recs) + len(new_recs)
    print(f"\nWrote {index_path}")
    print(f"  total examples: {total:,} (base {len(base_recs):,} + new {len(new_recs):,})")
    print(f"  at batch=32, that's {total // 32} steps in 1 epoch")

    # 4. Per-category summary
    print(f"\nFinal per-category counts:")
    final_cats: Counter = Counter()
    for r in base_recs + new_recs:
        final_cats[r["category"]] += 1
    for c, n in sorted(final_cats.items()):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
