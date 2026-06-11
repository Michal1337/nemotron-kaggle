#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=v5-sweep-measure
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/v5_sweep_measure_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
# exact-tokenizer governor (no runs/baseline on the cluster; without this the
# narrator silently falls back to a chars/1.40 estimate and over-abstains)
export NEMOTRON_TOKENIZER_JSON=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16/tokenizer.json
.venv/bin/python crypt_redesign_proto/v5_sweep_measure.py
