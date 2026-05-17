"""Baseline LoRA SFT trainer for the NVIDIA Nemotron Model Reasoning Challenge.

Adapts the public ``kienngx/nvidia-nemotron-training-copy-run-instantly``
recipe to a self-contained script. Trains a rank-32 LoRA on
Nemotron-3-Nano-30B-A3B against the raw competition ``train.csv`` and writes
``submission.zip`` (containing ``adapter_config.json`` + ``adapter_model.safetensors``)
to the output directory.

The LoRA config is fixed to the project baseline (rank 32, alpha 32, dropout 0,
target_modules covering attention + MLP + Mamba in/out_proj + lm_head, with an
explicit ``rank_pattern``/``alpha_pattern`` entry for ``in_proj``).

Example
-------
    python train_baseline.py \
        --model-path ./models/nemotron-3-nano-30b-a3b-bf16 \
        --train-csv ./data/train.csv \
        --output-dir ./runs/baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Reduce CUDA fragmentation for the 30B model. Must be set before torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import polars as pl  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


# Project baseline LoRA shape.
TARGET_MODULES = [
    "down_proj",
    "in_proj",
    "k_proj",
    "lm_head",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="Local folder with config.json + tokenizer + safetensors shards.")
    p.add_argument("--train-csv", required=True,
                   help="Path to competition train.csv (columns: id, prompt, answer).")
    p.add_argument("--output-dir", default="./runs/baseline",
                   help="Where to write the LoRA adapter + submission.zip.")
    p.add_argument("--subsample", type=int, default=1200,
                   help="Random subsample of train rows (0 = use everything).")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    return p.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
        # Kept explicit so the saved adapter_config.json matches the project
        # baseline schema even if the global rank/alpha is later swept.
        rank_pattern={"in_proj": args.lora_rank},
        alpha_pattern={"in_proj": args.lora_alpha},
        init_lora_weights=True,
        fan_in_fan_out=False,
    )


def disable_nemotron_fast_path() -> None:
    # The Mamba fast path needs causal_conv1d + mamba_ssm CUDA kernels. On
    # hardware/OS where those aren't available the slow PyTorch path is used
    # automatically; we force False so behavior is identical regardless.
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")


def load_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Dataset:
    df = pl.read_csv(args.train_csv)
    if args.subsample and 0 < args.subsample < len(df):
        df = df.sample(n=args.subsample, seed=args.seed)
    print(f"Loaded {len(df)} training rows from {args.train_csv}")

    ds = Dataset.from_pandas(df.to_pandas())

    def build_text(example: dict) -> dict:
        # Baseline mirrors kienngx: trains on raw `answer` with no synthetic
        # CoT. Swap in a CoT-augmented dataset (e.g. the kienngx
        # nemotron-30b-competition-trainingdata-cot-labels dataset) for a
        # measurable LB jump.
        user_msg = example["prompt"] + "\nPut your final answer inside \\boxed{}."
        assistant_msg = str(example["answer"])
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    return ds.map(build_text, remove_columns=ds.column_names)


def package_submission(output_dir: Path) -> Path:
    zip_path = output_dir / "submission.zip"
    required = {"adapter_config.json", "adapter_model.safetensors"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            if fname.startswith("adapter") and (output_dir / fname).is_file():
                zf.write(output_dir / fname, fname)
    with zipfile.ZipFile(zip_path) as zf:
        contents = set(zf.namelist())
    print(f"Wrote {zip_path} with {sorted(contents)}")
    missing = required - contents
    if missing:
        raise RuntimeError(f"submission.zip missing: {sorted(missing)} — Kaggle scoring will fail.")
    return zip_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {args.model_path}")
    print(f"Output:  {output_dir}")
    print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args, tokenizer)

    print("Loading base model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    disable_nemotron_fast_path()

    print("Wrapping with LoRA...")
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        bf16=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    package_submission(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
