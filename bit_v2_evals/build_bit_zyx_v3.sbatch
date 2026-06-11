#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=build-bit-v3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/build_bit_zyx_v3_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
.venv/bin/python /home2/faculty/mgromadzki/build_bit_zyx_v3.py
