#!/bin/bash
# Multi-GPU (DDP) SFT on EDEN — single-node data parallel via torchrun.
# Ported from helios train_gh200_ddp.sh (2026-06-11); same trainer
# (src/training/train_huikang_style_ddp.py, grad-clip 1.0 built in).
#
# Submit (NGPU must equal the --gres GPU count):
#   NGPU=2 DS_PATH=<dir> STEPS=<n> TAG=<tag> EPOCHS=<e> \
#     sbatch --gres=gpu:h100:2 src/training/train_ddp_eden.sbatch
# eden hopper has 8x H100, so NGPU up to 8.
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=nemo-train-ddp
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/train-ddp-%j.out
set -uo pipefail
REPO=/home2/faculty/mgromadzki/src/nemotron-kaggle
cd "$REPO"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# eden H100 = 80GB (vs helios GH200 96GB); expandable_segments reclaims the
# fragmented reserve that caused a marginal OOM at step 2.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compile caches on the re-com group space (16T), NOT the small /home2 quota;
# DDP appends rank<N>/ subdirs so it would double the home2 usage
export TRITON_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/inductor
mkdir -p "$REPO/runs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# per-job work dir; the DDP trainer chdir's into rank<N>/ subfolders under cwd
WORK=/mnt/evafs/groups/re-com/mgromadzki/runwork/$SLURM_JOB_ID
mkdir -p "$WORK"; cd "$WORK"

MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS_PATH="${DS_PATH:?set DS_PATH=<dataset dir with tokens/ + logprobs/index.jsonl>}"
TAG="${TAG:-ddp}"
EPOCHS="${EPOCHS:-1}"
NGPU="${NGPU:-2}"
BATCH="${BATCH:-32}"
MICRO="${MICRO:-2}"
# STEPS auto-computed for EPOCHS over the GLOBAL batch when unset/auto
if [ -z "${STEPS:-}" ] || [ "${STEPS:-}" = "auto" ]; then
  N=$(wc -l < "$DS_PATH/logprobs/index.jsonl")
  SPE=$(( (N + BATCH - 1) / BATCH )); STEPS=$(( EPOCHS * SPE ))
  echo "auto STEPS: N=$N batch=$BATCH epochs=$EPOCHS -> $STEPS"
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT="$DS_PATH/adapter-${TAG}"
SHUF=""; [ "$EPOCHS" -ge 2 ] && SHUF="--shuffle-dataset"

echo "DDP: NGPU=$NGPU DS_PATH=$DS_PATH STEPS=$STEPS BATCH=$BATCH OUT=$OUT EPOCHS=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv
# mamba_ssm corruption guard (recurring on this cluster)
python -c 'import mamba_ssm, os; p=os.path.join(os.path.dirname(mamba_ssm.__file__),"ops","triton","ssd_chunk_scan.py"); s=os.path.getsize(p); assert s>100000, f"mamba_ssm corrupt: {s}"; print("mamba ok", s)'

torchrun --standalone --nproc_per_node="$NGPU" \
  "$REPO/src/training/train_huikang_style_ddp.py" \
  --model-path  "$MODEL" \
  --corpus-path "$DS_PATH/tokens" \
  --train-order "$DS_PATH/logprobs/index.jsonl" \
  --output-dir  "$OUT" \
  --num-steps   "$STEPS" \
  --batch-size "$BATCH" --micro-batch-size "$MICRO" --learning-rate 2e-4 \
  $SHUF $EXTRA_ARGS
RC=$?
echo "TRAIN_RC=$RC"
# final-adapter eval on the dataset's held-out val (single GPU; DDP has torn down)
if [ "$RC" = "0" ] && [ -f "$DS_PATH/val.jsonl" ]; then
  cd "$REPO"
  echo "=== eval final adapter on $DS_PATH/val.jsonl ==="
  python src/training/eval_adapter.py \
    --base-model "$MODEL" --adapter "$OUT" \
    --val-jsonl "$DS_PATH/val.jsonl" --output "$OUT/eval-val.json"
fi
echo "DDP_TRAIN_EVAL_DONE"
