#!/bin/bash -l
# Min-logprob metric for the bit-zyx-v2-realsynth run: per-token logprobs of
# all 2330 training rationales under the final (3ep) adapter.
#SBATCH --account=plgmgarcagi-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=119G
#SBATCH --time=24:00:00
#SBATCH --job-name=score-bitv2-logprobs
#SBATCH --output=/net/scratch/hscra/plgrid/plgmajkel1337/nemotron-kaggle/runs/score-bitv2-%j.out
set -uo pipefail
module purge
module load CUDA/12.9.1 Python/3.11.5
source /net/scratch/hscra/plgrid/plgmajkel1337/venvs/nemo311/bin/activate
export HF_HOME=/net/scratch/hscra/plgrid/plgmajkel1337/hf-home
export XDG_CACHE_HOME=/net/scratch/hscra/plgrid/plgmajkel1337/.cache
export TRITON_CACHE_DIR=/net/scratch/hscra/plgrid/plgmajkel1337/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/net/scratch/hscra/plgrid/plgmajkel1337/.cache/inductor
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPO=/net/scratch/hscra/plgrid/plgmajkel1337/nemotron-kaggle
MODEL=/net/scratch/hscra/plgrid/plgmajkel1337/llms/nemotron-3-nano-30b-a3b-bf16
DS=/net/scratch/hscra/plgrid/plgmajkel1337/nemotron-master/training/sft/bit-zyx-v2-realsynth
cd "$REPO"
python src/training/score_adapter_logprobs.py \
  --base-model "$MODEL" \
  --adapter "$DS/adapter-3ep-b8-gh200" \
  --tokens-root "$DS/tokens" \
  --output "$DS/adapter-3ep-b8-gh200/train_logprobs.jsonl" \
  --worst-k 12
