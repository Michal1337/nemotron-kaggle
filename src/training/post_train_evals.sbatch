#!/bin/bash
# Post-train: per-EPOCH + final accuracy eval on val.jsonl, then held-out
# gold-CoT min-logprob probe. SUBMIT WITH afterANY (not afterok): long-CoT
# micro-1 runs may TIMEOUT before all epochs finish, but per-epoch dirs
# <OUT>-epochN are saved as training progresses, so we still eval what landed.
# Override via --export=ALL,DS=<dataset_dir>,FINAL=<final_adapter_dir>
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
# 1) accuracy eval. Each adapter = a full 30B reload+recompile (~slow), so by
#    DEFAULT eval ONLY the last-saved checkpoint. Set EVAL_EPOCHS=all for the
#    full per-epoch curve. sort -V => bare $FINAL first then epoch1..N, so the
#    last item is the highest epoch saved (also the timeout fallback).
ALLCK=$(ls -d ${FINAL} ${FINAL}-epoch* 2>/dev/null | sort -V)
LAST=""
for A in $ALLCK; do [ -f "$A/adapter_model.safetensors" ] && LAST="$A"; done
if [ "${EVAL_EPOCHS:-last}" = "all" ]; then EVALSET="$ALLCK"; else EVALSET="$LAST"; fi
for A in $EVALSET; do
  [ -f "$A/adapter_model.safetensors" ] || continue
  if [ -f "$A/eval-val.json" ]; then echo "skip evaluated $A"; continue; fi
  echo "=== eval $A ==="
  python src/training/eval_adapter.py --base-model "$MODEL" --adapter "$A" \
    --val-jsonl "$DS/val.jsonl" --output "$A/eval-val.json" || echo "eval failed $A"
done
# 2) held-out gold-CoT min-logprob probe. Prefer the final adapter; if the run
#    timed out before writing $FINAL, fall back to the last saved epoch.
PROBE="$FINAL"; [ -f "$FINAL/adapter_model.safetensors" ] || PROBE="$LAST"
if [ -n "$PROBE" ] && [ -f "$PROBE/adapter_model.safetensors" ]; then
  # gold tokens are adapter-independent: build once into the dataset, reuse every eval
  if [ ! -s "$DS/_val_gold_pids.txt" ]; then
    echo "=== build val gold tokens (held-out synth + frozen-real self-checked + insample) ==="
    python src/training/build_val_gold_tokens.py --ds "$DS"
  else
    echo "=== reuse stored val gold tokens ($(wc -l < "$DS/_val_gold_pids.txt") pids) ==="
  fi
  echo "=== score val gold logprob on $PROBE ==="
  python src/training/score_adapter_logprobs.py --base-model "$MODEL" --adapter "$PROBE" \
    --tokens-root "$DS/val_gold_tokens" --pids-file "$DS/_val_gold_pids.txt" \
    --output "$PROBE/val_gold_logprobs.jsonl" --worst-k 12
  echo "=== group logprobs by tier/category (held-out = generalization) ==="
  python src/training/report_val_logprobs.py \
    --logprobs "$PROBE/val_gold_logprobs.jsonl" \
    --manifest "$DS/_val_gold_manifest.jsonl" | tee "$PROBE/val_gold_report.txt"
else
  echo "no adapter checkpoint found for logprob probe"
fi
echo POST_EVAL_DONE
