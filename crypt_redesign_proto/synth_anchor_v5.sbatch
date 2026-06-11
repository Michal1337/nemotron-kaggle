#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=synth-anchor-v5
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/synth_anchor_v5_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
export NEMOTRON_TOKENIZER_JSON=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16/tokenizer.json
.venv/bin/python src/synth/synth_crypt_anchor.py \
  /mnt/evafs/groups/re-com/mgromadzki/synth_crypt_anchor_v5.jsonl \
  "std=1500,swap=900,easy=400,concat=200" \
  --patterns crypt_redesign_proto/real_anchor_patterns.json --per-pattern 12
