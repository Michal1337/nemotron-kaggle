#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=build-anchor-pilot
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/build_anchor_pilot_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
.venv/bin/python /home2/faculty/mgromadzki/build_crypt_anchor_pilot.py
