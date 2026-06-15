#!/bin/bash
#SBATCH -A re-com
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J eqsyn
#SBATCH -o /mnt/evafs/groups/re-com/mgromadzki/eq_synth_%j.out
cd ~/src/nemotron-kaggle
# Faithful generator: matches the REAL solved (568 deduce + 85 guess) distribution in
# every measured aspect (rule mix, sign incl. literal-minus, orientation, canonical
# +,-,* ops, operand value floor, q_cnt, rule co-occurrence). Loads eq_real_dist.json
# (mined from real by _mine_eq_real_dist.py; ship it with the repo, 10.8KB).
# Writes a NEW file (eq_synth_v5.jsonl) so production eq_synth_v4.jsonl is not clobbered
# until the new corpus is reviewed/A-B'd. Re-mine on the cluster only if eq_real_all.jsonl
# is present (it is local-only); otherwise the shipped eq_real_dist.json is used.
.venv/bin/python src/synth/synth_eq_full.py \
  --out /mnt/evafs/groups/re-com/mgromadzki/eq_synth_v5.jsonl \
  --dist eq_real_dist.json \
  --n-deduce ${NDED:-1100} --n-guess ${NGUESS:-1100} --seed ${SEED:-0}
wc -l /mnt/evafs/groups/re-com/mgromadzki/eq_synth_v5.jsonl
echo EQ_SYNTH_DONE
