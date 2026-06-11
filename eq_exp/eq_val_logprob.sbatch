#!/bin/bash
# Held-out generalization logprob: build val gold-CoT tokens then score the eq
# 6ep adapter on them (134 UNSEEN problems' correct reasoning).
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eq-val-logprob
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/eq-val-logprob-%j.out
set -uo pipefail
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
source .venv/bin/activate
export TRITON_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
B=/mnt/evafs/groups/re-com/mgromadzki
MODEL=$B/llms/nemotron-3-nano-30b-a3b-bf16
DS=$B/nemotron-master/training/sft/eq-deduce-guess-8020
ADAPTER=$DS/adapter-6ep-b8
python eq_exp/build_eq_val_logprob.py
python src/training/score_adapter_logprobs.py \
  --base-model "$MODEL" --adapter "$ADAPTER" \
  --tokens-root "$DS/val_gold_tokens" --pids-file "$DS/_val_gold_pids.txt" \
  --output "$ADAPTER/val_gold_logprobs.jsonl" --worst-k 12
echo EQ_VAL_LOGPROB_DONE
