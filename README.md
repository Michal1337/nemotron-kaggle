# nemotron-kaggle

NVIDIA Nemotron Model Reasoning Challenge — adapter packaging + retraining pipeline for a Linux GPU cluster.

See [competition-notes.md](competition-notes.md) for the full landscape: leaderboard structure, the huikang/Tinker chain that produces the 0.86 public baseline, and what's been ruled out.

## What this repo does

Three paths:

| Mode | Time | GPU? | Output |
|---|---|---|---|
| **A** — lock the 0.86 public baseline | ~5 min | no | repackaged [`kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1`](https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter) |
| **B** — convert a newer huikang raw adapter (v27 latest) | ~30 min | yes | PEFT-converted candidate that may break the 0.86 ceiling |
| **C** — full retrain on huikang's pre-tokenized corpus | hours | yes | trained-from-scratch adapter (suryamilenial-style) |

Mode A is the floor every other run should beat. Mode B is the cheapest probe past 0.86. Mode C is where the real work is.

## Quick start

```bash
git clone <this-repo> && cd nemotron-kaggle
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools packaging
pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
pip install \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install -r requirements.txt
# For Mode C only:
pip install --no-deps "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
pip install --no-deps "unsloth[base] @ git+https://github.com/unslothai/unsloth"

# Downloads
mkdir -p adapters data models runs
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data
python -m zipfile -e data/nvidia-nemotron-model-reasoning-challenge.zip data/

kaggle models instances versions download kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1 -p adapters/kien-tinker --untar
kaggle models instances versions download huikang/nemotron-adapter/Transformers/default/20 -p adapters/huikang-v20 --untar
kaggle models instances versions download huikang/nemotron-adapter/Transformers/default/27 -p adapters/huikang-v27 --untar

# Mode A — package kien's adapter at root, ready to submit (no GPU)
mkdir -p runs/lock-086
cp adapters/kien-tinker/adapter_config.json adapters/kien-tinker/adapter_model.safetensors runs/lock-086/
(cd runs/lock-086 && python -m zipfile -c submission.zip adapter_config.json adapter_model.safetensors)
python src/verify_adapter.py --adapter runs/lock-086

# Mode B — convert huikang v27 (uses base model at the cluster path)
python src/convert_tinker_adapter.py \
  --base-model /mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16 \
  --adapter-path adapters/huikang-v27 \
  --output-dir runs/huikang-v27-peft
python src/verify_adapter.py --adapter runs/huikang-v27-peft --reference adapters/kien-tinker

# Mode C — full retrain (needs huikang corpus dataset; check current slug with `kaggle datasets list --user huikang`)
python src/train_huikang_style.py \
  --model-path /mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16 \
  --corpus-path data/huikang-corpus/nemotron-master/training/sft/04-08-16-14/tokens \
  --train-order data/huikang-corpus/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl \
  --train-csv data/train.csv \
  --output-dir runs/huikang-repro \
  --num-steps 1000 --batch-size 32 --micro-batch-size 4 --learning-rate 2e-4 \
  --zip-submission
```

## Files

| Path | Purpose |
|---|---|
| [requirements.txt](requirements.txt) | Pinned to torch 2.10 / cu128 / Python 3.12 (only combo with prebuilt CUDA-kernel wheels) |
| [src/convert_tinker_adapter.py](src/convert_tinker_adapter.py) | huikang raw → PEFT format via `tinker-cookbook` + asalhi's rank-32 SVD patch |
| [src/verify_adapter.py](src/verify_adapter.py) | Header-only structural audit — checks any adapter against the kien 0.86 fingerprint without loading the model |
| [src/train_huikang_style.py](src/train_huikang_style.py) | Full retrain pipeline: Unsloth + Cut Cross-Entropy + Mamba fast path + MoE expert weight tying + fp32 LoRA/router |
| [src/train_baseline.py](src/train_baseline.py) | Older naive baseline (raw answer column, no CoT). Caps ~0.67. Kept for reference. |
| [competition-notes.md](competition-notes.md) | Full competition notes — start here for context |

`data/`, `models/`, `adapters/`, `runs/` are gitignored and populated by the download commands below.

## Submitting

```bash
kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
  -f runs/<run>/submission.zip -m "<description>"
```

Daily submission limit applies — always run `verify_adapter.py` against the candidate first; structural breakage is the most common reason for a wasted slot.

## Prereqs

- Linux x86_64 with a >= 60 GB VRAM GPU for Mode B/C; Mode A needs no GPU.
- Python 3.12 (the prebuilt wheels are cp312-only).
- `nvcc` is **not** required — the prebuilt mamba_ssm / causal_conv1d wheels avoid the local CUDA compile entirely.
- `~/.kaggle/kaggle.json` with comp rules accepted on kaggle.com.

## Reference chain

The 0.86 public baseline came together as:

```
huikang's investigator/augmenter corpus
        │  (perfect rationales for previously-unsolved problems)
        ▼
Thinking Machines Tinker SFT → huikang/nemotron-adapter (raw, ~1.5 GB)
        │
        │  tinker-cookbook.weights.build_lora_adapter + SVD rank-32 patch
        │  (un-fuses Q/K/V, merges Mamba gate_proj+x_proj → in_proj,
        │   unties MoE experts 1× → 128×)
        ▼
PEFT-format adapter (~3.5 GB) — kien republished as `tinker-adapter`
```

Mode A submits kien's repackaged result. Mode B runs the conversion ourselves on a (potentially newer) huikang checkpoint. Mode C bypasses huikang and trains the SFT step ourselves on his published corpus.
