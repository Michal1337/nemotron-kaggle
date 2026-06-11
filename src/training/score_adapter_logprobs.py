"""Per-token logprobs of training rationales under a TRAINED adapter.

The min-logprob metric: for each trace, the minimum per-token logprob over
loss (completion) tokens. Tokens with loss > ln 2 (logprob < -0.693, p < 0.5)
are more likely than not to NOT reproduce at sampling time - the real SFT
objective is to drive the worst token up, not the mean down.

For each pid this writes one jsonl row:
  {pid, num_loss_tokens, total_loss, mean_loss, min_logprob,
   n_below_ln2, frac_below_ln2,
   worst: [{i, tok, lp, ctx}]}            # K worst tokens with decoded context

Usage:
    python score_adapter_logprobs.py --base-model <BASE> --adapter <ADAPTER> \
        --tokens-root <DIR/tokens> --output <out.jsonl> [--pids-file F] [--worst-k 12]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LN2 = 0.6931471805599453


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--tokens-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pids-file", default=None,
                   help="Newline-delimited pids; default = every dir in tokens-root.")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--worst-k", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokens_root = Path(args.tokens_root)
    if args.pids_file:
        pids = [ln.strip() for ln in open(args.pids_file) if ln.strip()]
    else:
        pids = sorted(d.name for d in tokens_root.iterdir() if (d / "synthetic.json").exists())
    print(f"Scoring {len(pids)} pids under adapter {args.adapter}")

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    FastLanguageModel.for_inference(model)

    device = "cuda"
    t0 = time.time()
    out_f = open(args.output, "w")
    for k, pid in enumerate(pids):
        tok_file = tokens_root / pid / "synthetic.json"
        td = json.loads(tok_file.read_text())
        tokens, mask = td["tokens"], td["mask"]
        if len(tokens) > args.max_seq_len:
            print(f"[{k+1}/{len(pids)}] {pid}: too long, skip")
            continue
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        with torch.inference_mode():
            logits = model(ids).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            targets = ids[0, 1:]
            lp = log_probs[:-1].gather(1, targets[:, None]).squeeze(1).cpu().tolist()
        mask_shift = mask[1:]
        scored = [(lp[i], i) for i in range(len(lp)) if mask_shift[i] == 1]
        if not scored:
            continue
        lps = [s for s, _ in scored]
        worst = sorted(scored)[:args.worst_k]
        worst_rows = []
        for w_lp, i in worst:
            tok_id = tokens[i + 1]
            ctx = tokenizer.decode(tokens[max(0, i - 24):i + 2])
            worst_rows.append({"i": i + 1, "tok": tokenizer.decode([tok_id]),
                               "lp": round(w_lp, 4), "ctx": ctx[-120:]})
        n_below = sum(1 for s in lps if s < -LN2)
        row = {"pid": pid,
               "num_loss_tokens": len(lps),
               "total_loss": round(-sum(lps), 4),
               "mean_loss": round(-sum(lps) / len(lps), 6),
               "min_logprob": round(min(lps), 4),
               "n_below_ln2": n_below,
               "frac_below_ln2": round(n_below / len(lps), 6),
               "worst": worst_rows}
        out_f.write(json.dumps(row) + "\n")
        if (k + 1) % 25 == 0:
            out_f.flush()
            rate = (k + 1) / max(time.time() - t0, 1e-3)
            print(f"[{k+1}/{len(pids)}] {pid} min={row['min_logprob']:.3f} "
                  f"below-ln2={n_below}  ({rate:.2f}/s, ETA {(len(pids)-k-1)/max(rate,1e-6)/60:.0f}min)",
                  flush=True)
        del ids, logits, log_probs
        if k % 32 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    out_f.close()
    print("DONE", args.output)
    print("SCORE_LOGPROBS_DONE")


if __name__ == "__main__":
    main()
