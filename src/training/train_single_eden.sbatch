#!/bin/bash
# Single-GPU SFT on EDEN (proven path: train_huikang_style_clipped.py, grad-clip).
# Fallback while the multi-GPU DDP trainer's rank-desync is unresolved.
#   DS_PATH=<dir> STEPS=<n> TAG=<tag> EPOCHS=<e> BATCH=<b> \
#     sbatch src/training/train_single_eden.sbatch
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=nemo-train-1gpu
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/train-1gpu-%j.out
set -uo pipefail
REPO=/home2/faculty/mgromadzki/src/nemotron-kaggle
cd "$REPO"
source .venv/bin/activate
MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS_PATH="${DS_PATH:?set DS_PATH}"
STEPS="${STEPS:?set STEPS}"
TAG="${TAG:-1gpu}"
EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-16}"
OUT="$DS_PATH/adapter-${TAG}"
SHUF=""; [ "$EPOCHS" -ge 2 ] && SHUF="--shuffle-dataset"
echo "1GPU: DS_PATH=$DS_PATH STEPS=$STEPS BATCH=$BATCH OUT=$OUT EPOCHS=$EPOCHS"
python -c 'import mamba_ssm, os; p=os.path.join(os.path.dirname(mamba_ssm.__file__),"ops","triton","ssd_chunk_scan.py"); s=os.path.getsize(p); assert s>100000, f"mamba_ssm corrupt: {s}"; print("mamba ok", s)'
python src/training/train_huikang_style_clipped.py \
  --model-path "$MODEL" --corpus-path "$DS_PATH/tokens" --train-order "$DS_PATH/logprobs/index.jsonl" \
  --output-dir "$OUT" --num-steps "$STEPS" --batch-size "$BATCH" $SHUF
RC=$?
echo "TRAIN_RC=$RC"
if [ "$RC" = "0" ] && [ -f "$DS_PATH/val.jsonl" ]; then
  echo "=== eval on $DS_PATH/val.jsonl ==="
  python src/training/eval_adapter.py \
    --base-model "$MODEL" --adapter "$OUT" --val-jsonl "$DS_PATH/val.jsonl" --output "$OUT/eval-val.json"
fi
echo "TRAIN_EVAL_DONE"
