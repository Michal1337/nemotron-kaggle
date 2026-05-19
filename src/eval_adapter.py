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
    p.add_argument("--corpus-index", required=True,
                   help="Path to logprobs/index.jsonl defining the pool of problem_ids. "
                        "Use the 04-08-16-14 index for in-sample eval of huikang's corpus.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--problems-jsonl", required=True)
    p.add_argument("--output", required=True)

    # Sampling
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


def load_pool(corpus_index: str, problems: dict, train: dict) -> list[dict]:
    pids: list[str] = []
    seen: set[str] = set()
    with open(corpus_index) as f:
        for line in f:
            r = json.loads(line)
            if r.get("epoch", 0) != 0:
                continue
            pid = r["problem_id"]
            if pid in seen or pid not in problems or pid not in train:
                continue
            seen.add(pid)
            pids.append(pid)
    return [{**problems[pid], **train[pid]} for pid in pids]


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

    problems = {json.loads(l)["id"]: json.loads(l) for l in open(args.problems_jsonl)}
    train: dict[str, dict] = {}
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train[row["id"]] = {"prompt": row["prompt"], "answer": row["answer"]}

    pool = load_pool(args.corpus_index, problems, train)
    print(f"Pool: {len(pool):,} problems")
    sample = stratified_sample(pool, args.sample_per_category, args.seed)
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
    for ex, out in zip(sample, outputs):
        gen_text = out.outputs[0].text
        predicted = extract_answer(gen_text)
        correct = verify(ex["answer"], predicted)
        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "gold": ex["answer"],
            "predicted": predicted,
            "correct": correct,
            "gen_chars": len(gen_text),
            "gen_tokens": len(out.outputs[0].token_ids),
        })
        flag = "OK " if correct else "!! "
        print(f"  {flag} {ex['id']} {ex['category']:>24s}  "
              f"gold={ex['answer']!r}  pred={predicted!r}  "
              f"gen={len(out.outputs[0].token_ids)}tok")

    # Aggregate
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
    total_correct = sum(r["correct"] for r in results)

    print(f"\n=== Per-category accuracy ===")
    print(f"  {'category':>24s}  {'n':>4s}  {'correct':>8s}  {'acc':>6s}")
    for cat in sorted(by_cat):
        n = len(by_cat[cat])
        c = sum(by_cat[cat])
        print(f"  {cat:>24s}  {n:>4d}  {c:>8d}  {100 * c / n:>5.1f}%")
    print(f"  {'TOTAL':>24s}  {len(results):>4d}  {total_correct:>8d}  "
          f"{100 * total_correct / len(results):>5.1f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "adapter": str(args.adapter) if enable_lora else None,
        "base_model": str(args.base_model),
        "corpus_index": str(args.corpus_index),
        "sample_per_category": args.sample_per_category,
        "n_total": len(results),
        "n_correct": total_correct,
        "accuracy": total_correct / len(results),
        "by_category": {cat: {"n": len(by_cat[cat]), "correct": sum(by_cat[cat]),
                              "accuracy": sum(by_cat[cat]) / len(by_cat[cat])}
                        for cat in sorted(by_cat)},
        "total_elapsed_s": total_elapsed,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
