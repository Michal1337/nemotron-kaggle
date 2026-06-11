#!/bin/bash
# Min-logprob scoring on EDEN (migrated off plgrid/helios 2026-06-11, permanent).
# Per-token logprobs of a dataset's training rationales under a trained adapter,
# with the worst-K tokens per trace. Override DS/ADAPTER/OUT via --export.
#   sbatch --export=ALL,DS=<dataset_dir>,ADAPTER=<adapter_dir> score_logprobs_eden.sbatch
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=score-logprobs
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/score-logprobs-%j.out
set -uo pipefail
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
source .venv/bin/activate
# compile caches on re-com (16T), not the small /home2 quota
export TRITON_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS=${DS:-/mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/bit-zyx-v2-realsynth}
ADAPTER=${ADAPTER:-$DS/adapter-3ep-b8-gh200}
OUT=${OUT:-$ADAPTER/train_logprobs.jsonl}
# mamba_ssm corruption guard (recurring on this cluster)
python -c 'import mamba_ssm, os; p=os.path.join(os.path.dirname(mamba_ssm.__file__),"ops","triton","ssd_chunk_scan.py"); s=os.path.getsize(p); assert s>100000, f"mamba_ssm corrupt: {s}"; print("mamba ok", s)'
python src/training/score_adapter_logprobs.py \
  --base-model "$MODEL" --adapter "$ADAPTER" \
  --tokens-root "$DS/tokens" --output "$OUT" --worst-k 12
echo SCORE_LOGPROBS_DONE
