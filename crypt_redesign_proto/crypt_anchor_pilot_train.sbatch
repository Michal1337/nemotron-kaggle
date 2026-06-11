#!/bin/bash
# STAGED ONLY - submit manually (or with --dependency=afterok:<build job>) when
# training is approved. Mirrors crypt_pilot2_train.sbatch minus the v2-format
# dashboard step (it parses v2 grammar; v4 CoTs need their own dashboard).
#SBATCH --account=re-com
#SBATCH --partition=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=crypt-anchor-pilot
#SBATCH --output=/home2/faculty/mgromadzki/src/nemotron-kaggle/runs/crypt-anchor-pilot-%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
source .venv/bin/activate
MODEL=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16
DS=/mnt/evafs/groups/re-com/mgromadzki/nemotron-master/training/sft/crypt-anchor-pilot
OUT=$DS/adapter-pilot
python -c 'import mamba_ssm, os; p=os.path.join(os.path.dirname(mamba_ssm.__file__),"ops","triton","ssd_chunk_scan.py"); s=os.path.getsize(p); assert s>100000, f"mamba_ssm corrupt: {s}"; print("mamba ok", s)'
python src/training/train_huikang_style_clipped.py \
  --model-path "$MODEL" --corpus-path "$DS/tokens" --train-order "$DS/logprobs/index.jsonl" \
  --output-dir "$OUT" --num-steps 1800 --batch-size 2 --shuffle-dataset
echo '=== eval on held-out real ==='
python src/training/eval_adapter.py \
  --base-model "$MODEL" --adapter "$OUT" --val-jsonl "$DS/val.jsonl" --output "$OUT/eval-val.json"
echo CRYPT_ANCHOR_PILOT_DONE
