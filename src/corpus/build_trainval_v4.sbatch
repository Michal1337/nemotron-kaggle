#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=build-tv4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/build_tv4_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
export NEMOTRON_TOKENIZER_JSON=/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16/tokenizer.json
.venv/bin/python src/corpus/build_trainval_v4.py
