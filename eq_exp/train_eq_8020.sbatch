#!/bin/bash
# 6-epoch SFT on eq-deduce-guess-8020. num_steps computed from the train index
# so LR decays to 0 over exactly 6 epochs: steps = 6 * ceil(N_train / batch).
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eq-8020-6ep
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/eq-8020-6ep-%j.out
set -uo pipefail
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
source .venv/bin/activate
MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS=/mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/eq-deduce-guess-8020
OUT=$DS/adapter-6ep-b8
BATCH=8
EPOCHS=6
N=$(wc -l < "$DS/logprobs/index.jsonl")
SPE=$(( (N + BATCH - 1) / BATCH ))
STEPS=$(( EPOCHS * SPE ))
echo "N_train=$N batch=$BATCH steps_per_epoch=$SPE -> num_steps=$STEPS (=$EPOCHS epochs)"
python -c 'import mamba_ssm, os; p=os.path.join(os.path.dirname(mamba_ssm.__file__),"ops","triton","ssd_chunk_scan.py"); s=os.path.getsize(p); assert s>100000, f"mamba_ssm corrupt: {s}"; print("mamba ok", s)'
python src/training/train_huikang_style_clipped.py \
  --model-path "$MODEL" --corpus-path "$DS/tokens" --train-order "$DS/logprobs/index.jsonl" \
  --output-dir "$OUT" --num-steps "$STEPS" --batch-size "$BATCH" --shuffle-dataset
echo '=== eval on held-out 20% real (source-separable) ==='
python src/training/eval_adapter.py \
  --base-model "$MODEL" --adapter "$OUT" --val-jsonl "$DS/val.jsonl" --output "$OUT/eval-val.json"
echo EQ_8020_6EP_DONE
