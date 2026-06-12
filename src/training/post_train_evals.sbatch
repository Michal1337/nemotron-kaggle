#!/bin/bash
# Post-train: per-epoch + final accuracy eval on val.jsonl, then held-out
# gold-CoT min-logprob probe on the final adapter. Override via --export=ALL,
# DS=<dataset_dir>,FINAL=<final_adapter_dir>
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=post-eval
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/post-eval-%j.out
set -uo pipefail
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
source .venv/bin/activate
export TRITON_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/evafs/groups/re-com/mgromadzki/.cache/inductor
export NEMOTRON_TOKENIZER_JSON=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16/tokenizer.json
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS="${DS:?set DS}"; FINAL="${FINAL:?set FINAL}"
# 1) accuracy eval: final + any per-epoch checkpoints
for A in $(ls -d ${FINAL}* 2>/dev/null); do
  [ -f "$A/adapter_model.safetensors" ] || continue
  [ -f "$A/eval-val.json" ] && { echo "skip evaluated $A"; continue; }
  echo "=== eval $A ==="
  python src/training/eval_adapter.py --base-model "$MODEL" --adapter "$A" \
    --val-jsonl "$DS/val.jsonl" --output "$A/eval-val.json" || echo "eval failed $A"
done
# 2) held-out gold-CoT min-logprob probe on the final adapter
echo "=== build val gold tokens ==="
python src/training/build_val_gold_tokens.py --ds "$DS"
echo "=== score val gold logprob (generalization) ==="
python src/training/score_adapter_logprobs.py --base-model "$MODEL" --adapter "$FINAL" \
  --tokens-root "$DS/val_gold_tokens" --pids-file "$DS/_val_gold_pids.txt" \
  --output "$FINAL/val_gold_logprobs.jsonl" --worst-k 12
echo POST_EVAL_DONE
