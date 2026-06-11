#!/bin/bash
#SBATCH -A re-com
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J eqsyn
#SBATCH -o /mnt/evafs/groups/re-com/mgromadzki/eq_synth_%j.out
cd ~/src/nemotron-kaggle
# 800/800 so v4 can split 600 train + 200 val (id-disjoint) per sub-category
.venv/bin/python src/synth/synth_eq_full.py \
  --out /mnt/evafs/groups/re-com/mgromadzki/eq_synth_v4.jsonl \
  --n-deduce ${NDED:-800} --n-guess ${NGUESS:-800} --seed ${SEED:-0}
wc -l /mnt/evafs/groups/re-com/mgromadzki/eq_synth_v4.jsonl
echo EQ_SYNTH_DONE
