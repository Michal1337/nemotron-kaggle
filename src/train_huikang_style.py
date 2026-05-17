"""Reproduce huikang's 0.85+ training pipeline on a single 100GB+ GPU cluster node.

Adapted from suryamilenial/end-to-end-finetuning-for-lb-0-83-6e2fa5 — keeps the
Unsloth FastLanguageModel path, the manual lm_head LoRA, Cut Cross-Entropy on
the LM head, Mamba CUDA fast path, MoE expert weight tying, fp32 LoRA + fp32
router, and the `base_model.model.lm_head.` -> `base_model.model.backbone.lm_head.`
key rename at save time.

Differences from the Kaggle notebook:
  * No Kaggle/Modal branching — assumes a plain Linux GPU node.
  * Paths are CLI args, not magic ``/kaggle/input`` lookups.
  * The training corpus is huikang's pre-tokenized snapshot, downloaded once
    via ``kagglehub`` (see download.sh).
  * Saves to a writable directory; you decide whether to zip + submit.

Example
-------
    python train_huikang_style.py \\
        --model-path ./models/nemotron-3-nano-30b-a3b-bf16 \\
        --corpus-path ./data/huikang-corpus/tokens \\
        --train-order ./data/huikang-corpus/logprobs/index.jsonl \\
        --output-dir ./runs/huikang-repro \\
        --num-steps 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path


# Reduce CUDA fragmentation. Must be set before torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Project baseline — matches huikang's published config exactly.
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj",
    "in_proj", "out_proj",
    "lm_head",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="Local folder with the Nemotron-3-Nano-30B-A3B-BF16 base model.")
    p.add_argument("--corpus-path", required=True,
                   help="Path to the pre-tokenized corpus dir (one subdir per problem_id, each with synthetic.json).")
    p.add_argument("--train-order", required=True,
                   help="logprobs/index.jsonl from huikang's repo snapshot (replays training order).")
    p.add_argument("--train-csv", default=None,
                   help="Optional path to competition train.csv (only used with --original-problems-only).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the adapter (creates adapter_config.json + adapter_model.safetensors).")
    p.add_argument("--pretrained-adapter", default=None,
                   help="Optional path to an already-trained adapter to warm-start from. "
                        "If omitted, starts from fresh LoRA init (RESET_WEIGHTS=True).")
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-moe-tie", action="store_true",
                   help="Disable MoE expert weight tying (default: tied, Tinker-style).")
    p.add_argument("--in-proj-only", action="store_true",
                   help="Freeze every LoRA param except in_proj (ablation lever).")
    p.add_argument("--original-problems-only", action="store_true",
                   help="Filter corpus to only problem_ids present in --train-csv.")
    p.add_argument("--shuffle-dataset", action="store_true",
                   help="Shuffle examples instead of replaying the original training order.")
    p.add_argument("--zip-submission", action="store_true",
                   help="Also write <output-dir>/submission.zip ready for upload.")
    return p.parse_args()


def kernel_sanity_check() -> None:
    """Confirm mamba_ssm + causal_conv1d CUDA kernels actually run on this GPU.

    The fast Mamba path silently falls back to a pure-PyTorch reference path
    if the kernels can't launch — much slower and uses more VRAM. Failing here
    early beats finding out 4 hours into training.
    """
    import causal_conv1d
    import mamba_ssm
    import torch

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(1, 512, 32, device="cuda", dtype=torch.bfloat16) + 4e-3
    w = torch.randn(512, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(x, w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")


def load_corpus(args: argparse.Namespace) -> list[dict]:
    import csv

    ordered_ids: list[str] = []
    seen: set[str] = set()
    with open(args.train_order) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("epoch", 0) != 0:
                continue
            pid = rec["problem_id"]
            if pid in seen:
                continue
            seen.add(pid)
            ordered_ids.append(pid)
    print(f"Loaded {len(ordered_ids)} problem_ids in training order from {args.train_order}")

    examples: list[dict] = []
    for sid in ordered_ids:
        seg_path = os.path.join(args.corpus_path, sid, "synthetic.json")
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(
                f"problem_id {sid} from training order missing in corpus: {seg_path}"
            )
        with open(seg_path) as f:
            rec = json.load(f)
        tokens = rec["tokens"]
        mask = rec["mask"]
        if not tokens:
            continue
        if len(tokens) > args.max_seq_len:
            tokens = tokens[: args.max_seq_len]
            mask = mask[: args.max_seq_len]
        if not any(mask):
            continue
        examples.append({
            "problem_id": sid,
            "tokens": tokens[:-1],
            "targets": tokens[1:],
            "weights": [float(m) for m in mask[1:]],
        })

    if args.original_problems_only:
        if not args.train_csv:
            raise SystemExit("--original-problems-only requires --train-csv")
        with open(args.train_csv) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        print(f"ORIGINAL_PROBLEMS_ONLY: filtered {before} -> {len(examples)} examples "
              f"using {len(original_ids)} ids from {args.train_csv}")

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(f"Loaded {len(examples)} examples, {total_tokens:,} tokens "
          f"(unmasked={total_unmasked:,.0f})")
    return examples


def patch_nemotron_fast_path() -> None:
    """Force the Mamba CUDA fast path to be considered available.

    The model's modeling file computes ``is_fast_path_available`` at import
    time as ``all(...)``. If any one symbol failed to import it becomes False
    permanently. Forcing True is safe iff all kernels are actually present —
    which kernel_sanity_check() above confirms.
    """
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            print(f"  was: is_fast_path_available={mod.is_fast_path_available}")
            mod.is_fast_path_available = True
            print(f"  now: is_fast_path_available=True")
            return
    raise RuntimeError("Could not find modeling_nemotron_h module to patch")


def add_lm_head_lora(model) -> None:
    """Unsloth drops the lm_head LoRA for MoE models. Re-add it manually."""
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear

    causal_lm = model
    while hasattr(causal_lm, "model"):
        causal_lm = causal_lm.model
    lm_head = causal_lm.lm_head
    if isinstance(lm_head, LoraLinear):
        print("  lm_head already has LoRA")
        return
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(
        cfg, "default", target=lm_head, target_name="lm_head", parent=causal_lm,
    )
    print("  added LoRA to lm_head")


def cast_lora_to_fp32(model) -> None:
    """LoRA params -> fp32, MoE router stays fp32, everything else stays bf16."""
    import torch
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)
    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, \
                f"LoRA param {name} expected fp32, got {param.dtype}"
            continue
        is_router = ".mixer.gate." in name
        if is_router:
            assert param.dtype == torch.float32, \
                f"router param {name} expected fp32, got {param.dtype}"
            continue
        assert param.dtype == torch.bfloat16, \
            f"param {name} expected bf16, got {param.dtype}"
    print("  verified: LoRA params fp32, base params bf16 (MoE router fp32)")


def patch_cce_forward(model):
    """Patch the causal LM forward to use Cut Cross-Entropy on the LM head.

    Avoids materializing the full [batch, seq, vocab=131072] logits tensor.
    Returns the patched ``_base`` module so callers can keep the reference.
    """
    import torch
    from cut_cross_entropy import linear_cross_entropy

    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = _base.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items()
               if k in ("position_ids", "past_key_values", "use_cache")},
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states, lm_weight, labels, reduction="none"
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce
        return loss

    _base.forward = _patched_forward
    print("  patched CausalLM.forward with CCE (no logits materialization)")
    return _base


def load_pretrained_adapter(model, adapter_path: str) -> None:
    from peft import load_peft_weights
    print(f"Loading pretrained adapter from {adapter_path}...")
    adapter_weights = load_peft_weights(adapter_path)
    model_sd = model.state_dict()
    new_sd: dict = {}
    loaded = 0
    for ak, av in adapter_weights.items():
        if ak in model_sd:
            new_sd[ak] = av
            loaded += 1
            continue
        ak_with_default = ak.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        if ak_with_default in model_sd:
            new_sd[ak_with_default] = av
            loaded += 1
            continue
        ak_lm = ak.replace(".backbone.lm_head.", ".lm_head.")
        ak_lm_default = ak_lm.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        if ak_lm_default in model_sd:
            new_sd[ak_lm_default] = av
            loaded += 1
            continue
    model.load_state_dict(new_sd, strict=False)
    assert loaded == len(adapter_weights), \
        f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
    print(f"  loaded {loaded}/{len(adapter_weights)} weights into model")


def identify_moe_tied_params(model) -> list:
    """Tinker convention:
      gate_up_proj / up_proj / gate_proj / w1 -> tie A (input/hidden side)
      down_proj / w2                          -> tie B (output/hidden side)
    Unsloth stores experts as a batched [num_experts, ...] tensor; "tying"
    means all 128 expert slices stay identical. Saved adapter naturally emits
    128 per-expert copies for PEFT compatibility.
    """
    w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
    w2_proj_names = ("down_proj", ".w2.")
    tied: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".experts." not in name or ".lora_" not in name:
            continue
        is_w1 = any(p in name for p in w1_proj_names)
        is_w2 = any(p in name for p in w2_proj_names)
        is_A = ".lora_A." in name
        is_B = ".lora_B." in name
        should_tie = (is_w1 and is_A) or (is_w2 and is_B)
        if not should_tie:
            continue
        if param.dim() < 2 or param.shape[0] <= 1:
            continue
        tied.append(param)
    return tied


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {args.model_path}")
    print(f"Corpus:     {args.corpus_path}")
    print(f"Order:      {args.train_order}")
    print(f"Output:     {output_dir}")

    print("\n=== Kernel sanity check ===")
    kernel_sanity_check()

    print("\n=== Load corpus ===")
    examples = load_corpus(args)

    print("\n=== Load base model (Unsloth) ===")
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )

    print("\n=== Wrap in LoRA ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    FastLanguageModel.for_training(model)

    print("\n=== Patch Mamba fast path ===")
    patch_nemotron_fast_path()

    print("\n=== Add lm_head LoRA (Unsloth drops it for MoE) ===")
    add_lm_head_lora(model)

    print("\n=== Cast LoRA to fp32 ===")
    cast_lora_to_fp32(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model: {trainable:,} trainable / {total:,} total parameters")

    print("\n=== Patch forward with Cut Cross-Entropy ===")
    patch_cce_forward(model)

    print("\n=== Adapter init ===")
    if args.pretrained_adapter:
        load_pretrained_adapter(model, args.pretrained_adapter)
    else:
        print("  No --pretrained-adapter; using fresh LoRA init.")

    if args.in_proj_only:
        print("\n=== Freezing all LoRA params except in_proj ===")
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    moe_tied_params: list = []
    if not args.no_moe_tie:
        print("\n=== MoE tied-weight setup ===")
        moe_tied_params = identify_moe_tied_params(model)
        print(f"  {len(moe_tied_params)} params identified for tying")
        if moe_tied_params:
            print(f"  example shapes: {[tuple(p.shape) for p in moe_tied_params[:3]]}")
        with torch.no_grad():
            for p in moe_tied_params:
                mean = p.data.mean(dim=0, keepdim=True)
                p.data.copy_(mean.expand_as(p.data))

    def _tie_grads() -> None:
        # Sum grads across the expert dim (not mean): if W is the shared LoRA
        # factor and each expert uses a copy W_i = W, chain rule gives
        # dL/dW = sum_i dL/dW_i. Inactive experts contribute 0; router weights
        # are baked into active g_i so no double-counting. Mean would shift the
        # effective LR by 1/128 and not be equivalent under AdamW's eps/wd.
        if not moe_tied_params:
            return
        with torch.no_grad():
            for p in moe_tied_params:
                if p.grad is None:
                    continue
                grad_sum = p.grad.sum(dim=0, keepdim=True)
                p.grad.copy_(grad_sum.expand_as(p.grad))

    print("\n=== Training ===")
    gc.collect()
    torch.cuda.empty_cache()
    device = next(model.parameters()).device
    optimizer: torch.optim.AdamW | None = None

    indices = list(range(len(examples)))
    if args.shuffle_dataset:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        print(f"  shuffled {len(indices)} examples (seed={args.seed})")
    else:
        print(f"  keeping corpus order ({len(indices)} examples)")

    max_steps = len(examples) // args.batch_size
    num_steps = min(args.num_steps, max_steps)
    if num_steps < args.num_steps:
        print(f"  WARNING: clamped num_steps to {num_steps} (data ran out)")
    print(f"  steps={num_steps}, batch={args.batch_size}, micro_batch={args.micro_batch_size}, lr={args.learning_rate}")

    step = 0
    for batch_start in range(0, len(indices), args.batch_size):
        if step >= num_steps:
            break
        batch_indices = indices[batch_start : batch_start + args.batch_size]
        batch = [examples[i] for i in batch_indices]
        batch_tokens = [e["tokens"] for e in batch]
        batch_targets = [e["targets"] for e in batch]
        batch_weights = [e["weights"] for e in batch]

        n = len(batch)
        n_accum = math.ceil(n / args.micro_batch_size)
        total_loss_sum = 0.0
        total_weight_sum = 0.0

        for mb_start in range(0, n, args.micro_batch_size):
            mb_end = min(mb_start + args.micro_batch_size, n)
            mb_toks = batch_tokens[mb_start:mb_end]
            mb_tgts = batch_targets[mb_start:mb_end]
            mb_wts = batch_weights[mb_start:mb_end]

            n_micro = len(mb_toks)
            max_len = max(len(t) for t in mb_toks)
            total_len = sum(len(t) for t in mb_toks)

            padded_input = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            padded_targets = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            padded_weights = torch.zeros(n_micro, max_len, dtype=torch.float32, device=device)
            attention_mask = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            for i in range(n_micro):
                seq_len = len(mb_toks[i])
                padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
                padded_weights[i, :seq_len] = torch.tensor(mb_wts[i], dtype=torch.float32)
                attention_mask[i, :seq_len] = 1

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids=padded_input,
                    attention_mask=attention_mask,
                    labels=padded_targets,
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce
                weighted_loss = per_token_ce * padded_weights
                weight_sum_t = padded_weights.sum()
                loss_sum_t = weighted_loss.sum()
                loss = loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0
            (loss / n_accum).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del loss, per_token_ce, weighted_loss

            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"    mb {mb_start // args.micro_batch_size}: {n_micro}x{max_len} "
                  f"total={total_len} wall={time.time() - t0:.1f}s "
                  f"peak={peak_gb:.1f}GB mem={mem_gb:.1f}GB")

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
            )
        lr = args.learning_rate * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1e9
        )
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        print(f"  step {step}/{num_steps}: loss={loss_mean:.6f} grad_norm={grad_norm:.4f} lr={lr:.2e}")

    print(f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    print(f"\n=== Save adapter to {output_dir} ===")
    from safetensors.torch import load_file, save_file

    for f in os.listdir(output_dir):
        if f.startswith("adapter"):
            os.remove(output_dir / f)
    model.save_pretrained(str(output_dir))
    st_path = output_dir / "adapter_model.safetensors"
    tensors = load_file(str(st_path))
    renamed = {
        k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
        for k, v in tensors.items()
    }
    save_file(renamed, str(st_path))
    print(f"  wrote adapter_config.json + adapter_model.safetensors")

    if args.zip_submission:
        zip_path = output_dir / "submission.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for fname in ("adapter_config.json", "adapter_model.safetensors"):
                zf.write(output_dir / fname, fname)
        print(f"  wrote {zip_path}")


if __name__ == "__main__":
    main()
