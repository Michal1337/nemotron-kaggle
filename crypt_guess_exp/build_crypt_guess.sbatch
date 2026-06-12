#!/bin/bash
#SBATCH --account=re-com
#SBATCH --partition=short
#SBATCH --job-name=build-cguess
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/faculty/mgromadzki/build_cguess_%j.out
cd /home2/faculty/mgromadzki/src/nemotron-kaggle
.venv/bin/python crypt_guess_exp/build_crypt_guess_standalone.py
