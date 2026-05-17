# nemotron-kaggle

NVIDIA Nemotron Model Reasoning Challenge — adapter packaging + retraining pipeline for a Linux GPU cluster.

See [competition-notes.md](competition-notes.md) for the full landscape: leaderboard structure, the huikang/Tinker chain that produces the 0.86 public baseline, and what's been ruled out.

## What this repo does

Three paths, all driven by [setup.sh](setup.sh):

| Mode | Time | GPU? | Output |
|---|---|---|---|
| **A** — lock the 0.86 public baseline | ~5 min | no | repackaged [`kienngx/nemotron-nano-30b-trained/triton/tinker-adapter/1`](https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter) |
| **B** — convert a newer huikang raw adapter (v27 latest) | ~30 min | yes | PEFT-converted candidate that may break the 0.86 ceiling |
| **C** — full retrain on huikang's pre-tokenized corpus | hours | yes | trained-from-scratch adapter (suryamilenial-style) |

Mode A is the floor every other run should beat. Mode B is the cheapest probe past 0.86. Mode C is where the real work is.

## Quick start

```bash
git clone <this-repo> && cd nemotron-kaggle
source setup.sh
setup_env             # creates .venv with torch 2.10 + cu128 + prebuilt mamba_ssm/causal_conv1d wheels
setup_kaggle_token    # checks ~/.kaggle/kaggle.json exists and is 0600

# Mode A — 0.86 in 5 min
download_kien_adapter
mode_a_lock_086

# Mode B — try huikang v27
download_base_model            # ~60 GB
download_huikang_adapter 27    # ~1.5 GB
mode_b_convert_huikang 27

# Mode C — full retrain
download_training_corpus
mode_c_train
```

Each command is a shell function defined in [setup.sh](setup.sh); read it for the exact `kaggle`/`python` call it wraps.

## Files

| Path | Purpose |
|---|---|
| [setup.sh](setup.sh) | Env setup + downloads + Mode A/B/C orchestration |
| [requirements.txt](requirements.txt) | Pinned to torch 2.10 / cu128 / Python 3.12 (only combo with prebuilt CUDA-kernel wheels) |
| [convert_tinker_adapter.py](convert_tinker_adapter.py) | huikang raw → PEFT format via `tinker-cookbook` + asalhi's rank-32 SVD patch |
| [verify_adapter.py](verify_adapter.py) | Header-only structural audit — checks any adapter against the kien 0.86 fingerprint without loading the model |
| [train_huikang_style.py](train_huikang_style.py) | Full retrain pipeline: Unsloth + Cut Cross-Entropy + Mamba fast path + MoE expert weight tying + fp32 LoRA/router |
| [train_baseline.py](train_baseline.py) | Older naive baseline (raw answer column, no CoT). Caps ~0.67. Kept for reference. |
| [adapter_config.json](adapter_config.json) | Reference LoRA config — the canonical 0.86 shape (rank 32, alpha 32, `all-linear`, dropout 0) |
| [competition-notes.md](competition-notes.md) | Full competition notes — start here for context |

`data/`, `models/`, `adapters/`, `runs/`, `src/` are gitignored and populated by the download functions in [setup.sh](setup.sh).

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
