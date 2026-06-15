"""Evaluate a PEFT adapter on a stratified problem sample, via vLLM.

Default mode is **in-sample**: pick problems that were in the training corpus's
index.jsonl. If a trained adapter can't get its own training problems right,
the pipeline is broken. Not a generalization test — a smoke test.

Reproduces the competition scorer:
  * vLLM greedy decoding (``temperature=0``, ``max_tokens=7680``)
  * Chat template with ``enable_thinking=True``
  * Last non-empty ``\\boxed{...}`` group extracted
  * ``math.isclose(rel_tol=1e-2, abs_tol=1e-5)`` for numerics, lowercase
    string fallback, exact match for binary strings.

Requires vLLM 0.12+ (matches NVIDIA's Nemotron-3 model card recommendation):
    pip install "vllm>=0.12.0"

Example
-------
    python src/eval_adapter.py \\
        --base-model /mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16 \\
        --adapter runs/huikang-repro-04-08-16-14 \\
        --corpus-index /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl \\
        --train-csv /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/train.csv \\
        --problems-jsonl /mnt/evafs/groups/re-com/mgromadzki/nemotron-master/problems.jsonl \\
        --output runs/huikang-repro-04-08-16-14/eval-in-sample.json \\
        --sample-per-category 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", default=None,
                   help="PEFT adapter dir. Omit (with --no-adapter) to evaluate base model only.")
    p.add_argument("--no-adapter", action="store_true",
                   help="Evaluate BASE model only. Disables LoRA loading.")
    p.add_argument("--corpus-index", default=None,
                   help="Path to logprobs/index.jsonl. Defines what's IN the training corpus. "
                        "Not needed with --val-jsonl.")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--problems-jsonl", default=None)
    p.add_argument("--val-jsonl", default=None,
                   help="Evaluate records directly from a val.jsonl (each line: id, category, "
                        "source, prompt, answer). Bypasses pool/sampling and reports accuracy "
                        "per (category, source) so the real-val and synth-val splits are scored "
                        "separately.")
    p.add_argument("--output", required=True)

    # Sampling
    p.add_argument("--pool-mode", choices=["in-corpus", "out-of-corpus", "all"],
                   default="in-corpus",
                   help="in-corpus = problems IN --corpus-index (in-sample smoke test). "
                        "out-of-corpus = problems in train.csv but NOT in corpus index (held-out). "
                        "all = every problem in train.csv.")
    p.add_argument("--status", choices=["rule_found", "rule_unknown", "hypothesis_formed"],
                   default=None,
                   help="Optional filter by status in problems.jsonl. "
                        "Pair --pool-mode out-of-corpus with --status rule_found to eval on "
                        "huikang's deliberately-excluded easy problems (~2,152), or with "
                        "--status rule_unknown for the hard tail (~1,166).")
    p.add_argument("--sample-per-category", type=int, default=10,
                   help="Stratified sample per category. 0 = use entire pool (slow).")
    p.add_argument("--seed", type=int, default=42)

    # Generation
    p.add_argument("--max-new-tokens", type=int, default=7680)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                   help="vLLM KV cache headroom; lower if you OOM (default 0.92).")
    p.add_argument("--max-num-seqs", type=int, default=64,
                   help="vLLM max concurrent sequences (default 64).")
    return p.parse_args()


def extract_answer(text: str) -> str:
    # The answer is the content of the FINAL \boxed{...}. The cryptarithm/equation
    # symbol alphabet includes '{' and '}' (~20% of crypt golds, e.g. '}{+{', '24}'),
    # which collide with the LaTeX delimiter. A non-greedy [^}]* stops at the first
    # inner '}' and truncates/empties the answer -> false zero at scoring AND at
    # submission. So take the last '\boxed{' (preferring the post-</think> answer
    # line) and capture through to the LAST '}' in that region.
    tail = text.rsplit("</think>", 1)[-1]
    src = tail if "\\boxed{" in tail else text
    i = src.rfind("\\boxed{")
    if i >= 0:
        inner = src[i + len("\\boxed{"):]
        j = inner.rfind("}")
        ans = (inner[:j] if j >= 0 else inner).strip()
        if ans:
            return ans
    # fallback: legacy non-empty boxed groups (handles the no-final-brace case)
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def verify(stored: str, predicted: str) -> bool:
    s, p = stored.strip(), predicted.strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def load_corpus_pids(corpus_index: str) -> set[str]:
    s: set[str] = set()
    with open(corpus_index) as f:
        for line in f:
            r = json.loads(line)
            if r.get("epoch", 0) == 0:
                s.add(r["problem_id"])
    return s


def build_pool(corpus_index: str, problems: dict, train: dict,
               pool_mode: str, status: str | None) -> list[dict]:
    in_corpus = load_corpus_pids(corpus_index)
    out: list[dict] = []
    for pid, p in problems.items():
        if pid not in train:
            continue
        if pool_mode == "in-corpus" and pid not in in_corpus:
            continue
        if pool_mode == "out-of-corpus" and pid in in_corpus:
            continue
        # pool_mode == "all" includes everything in train.csv
        if status and p.get("status") != status:
            continue
        out.append({**p, **train[pid]})
    return out


def stratified_sample(pool: list[dict], n_per_cat: int, seed: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pool:
        by_cat[p["category"]].append(p)
    rng = random.Random(seed)
    out: list[dict] = []
    for cat in sorted(by_cat):
        items = list(by_cat[cat])
        rng.shuffle(items)
        out.extend(items if n_per_cat <= 0 else items[:n_per_cat])
    return out


def main() -> None:
    args = parse_args()
    if not args.no_adapter and not args.adapter:
        raise SystemExit("--adapter required (or pass --no-adapter for baseline eval)")

    if args.val_jsonl:
        sample = [json.loads(l) for l in open(args.val_jsonl)]
        for s in sample:
            s.setdefault("source", "real")
        cat_counts = Counter((s["category"], s["source"]) for s in sample)
        print(f"Val-jsonl: {len(sample)} records from {args.val_jsonl}")
        print(f"  by (category, source): {dict(cat_counts)}")
    else:
        if not (args.corpus_index and args.train_csv and args.problems_jsonl):
            raise SystemExit("--corpus-index, --train-csv, --problems-jsonl required (or use --val-jsonl)")
        problems = {json.loads(l)["id"]: json.loads(l) for l in open(args.problems_jsonl)}
        train: dict[str, dict] = {}
        with open(args.train_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train[row["id"]] = {"prompt": row["prompt"], "answer": row["answer"]}

        pool = build_pool(args.corpus_index, problems, train, args.pool_mode, args.status)
        pool_cat = Counter(p["category"] for p in pool)
        print(f"Pool ({args.pool_mode}"
              + (f", status={args.status}" if args.status else "")
              + f"): {len(pool):,} problems")
        print(f"  by category: {dict(pool_cat)}")
        if not pool:
            raise SystemExit("Pool is empty — check --pool-mode / --status combination.")
        sample = stratified_sample(pool, args.sample_per_category, args.seed)
        for s in sample:
            s.setdefault("source", "real")
        cat_counts = Counter(s["category"] for s in sample)
        print(f"Sample: {len(sample)} ({dict(cat_counts)})")

    # Late import so the script can be inspected without vLLM installed
    print("\n=== Initializing vLLM ===")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    enable_lora = not args.no_adapter
    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_lora=enable_lora,
        max_lora_rank=32 if enable_lora else None,
        max_loras=1 if enable_lora else None,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"Base: {args.base_model}")
    if enable_lora:
        print(f"Adapter: {args.adapter}")
    else:
        print("(no adapter — base-model baseline)")

    # Build prompts with the chat template (matches what the scorer feeds).
    prompts: list[str] = []
    for ex in sample:
        messages = [{"role": "user", "content": ex["prompt"] + PROMPT_SUFFIX}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        ))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        # Stop at the chat template's end-of-turn so we don't generate past <|im_end|>
        stop=["<|im_end|>", "<|eot_id|>"],
    )

    print(f"\n=== Generating {len(prompts)} completions ===")
    t0 = time.time()
    lora_req = LoRARequest("eval_adapter", 1, args.adapter) if enable_lora else None
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    total_elapsed = time.time() - t0
    print(f"Generated {len(outputs)} in {total_elapsed:.1f}s "
          f"({total_elapsed / len(outputs):.2f}s/example avg)")

    # Score
    results: list[dict] = []
    for ex, prompt, out in zip(sample, prompts, outputs):
        gen_text = out.outputs[0].text
        predicted = extract_answer(gen_text)
        correct = verify(ex["answer"], predicted)
        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "source": ex.get("source", "real"),
            "gold": ex["answer"],
            "predicted": predicted,
            "correct": correct,
            "gen_chars": len(gen_text),
            "gen_tokens": len(out.outputs[0].token_ids),
            "prompt": prompt,
            "generation": gen_text,
        })
        flag = "OK " if correct else "!! "
        print(f"  {flag} {ex['id']} {ex['category']:>24s}  "
              f"gold={ex['answer']!r}  pred={predicted!r}  "
              f"gen={len(out.outputs[0].token_ids)}tok")

    # Aggregate
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_cat_src: dict[tuple, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
        by_cat_src[(r["category"], r["source"])].append(r["correct"])
    total_correct = sum(r["correct"] for r in results)

    print(f"\n=== Per-category accuracy ===")
    print(f"  {'category':>24s}  {'n':>4s}  {'correct':>8s}  {'acc':>6s}")
    for cat in sorted(by_cat):
        n = len(by_cat[cat])
        c = sum(by_cat[cat])
        print(f"  {cat:>24s}  {n:>4d}  {c:>8d}  {100 * c / n:>5.1f}%")
    print(f"  {'TOTAL':>24s}  {len(results):>4d}  {total_correct:>8d}  "
          f"{100 * total_correct / len(results):>5.1f}%")

    # Per (category, source) — real-val vs synth-val scored separately
    sources = sorted({s for _, s in by_cat_src})
    if len(sources) > 1:
        print(f"\n=== Per (category, source) accuracy ===")
        print(f"  {'category':>24s}  {'source':>6s}  {'n':>4s}  {'correct':>8s}  {'acc':>6s}")
        for (cat, src) in sorted(by_cat_src):
            v = by_cat_src[(cat, src)]
            print(f"  {cat:>24s}  {src:>6s}  {len(v):>4d}  {sum(v):>8d}  {100 * sum(v) / len(v):>5.1f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "adapter": str(args.adapter) if enable_lora else None,
        "base_model": str(args.base_model),
        "corpus_index": str(args.corpus_index),
        "pool_mode": args.pool_mode,
        "status_filter": args.status,
        "sample_per_category": args.sample_per_category,
        "n_total": len(results),
        "n_correct": total_correct,
        "accuracy": total_correct / len(results),
        "by_category": {cat: {"n": len(by_cat[cat]), "correct": sum(by_cat[cat]),
                              "accuracy": sum(by_cat[cat]) / len(by_cat[cat])}
                        for cat in sorted(by_cat)},
        "by_category_source": {f"{cat}/{src}": {"n": len(by_cat_src[(cat, src)]),
                                                "correct": sum(by_cat_src[(cat, src)]),
                                                "accuracy": sum(by_cat_src[(cat, src)]) / len(by_cat_src[(cat, src)])}
                               for (cat, src) in sorted(by_cat_src)},
        "total_elapsed_s": total_elapsed,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
