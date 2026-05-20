# Eden cluster cheat sheet

## Login

```bash
ssh eden.mini.pw.edu.pl     # uses ~/.ssh/config ProxyJump
```

Project repo lives in `~/src/<repo>`. Python is at `~/.pyenv/shims/python`.

## Account + resource budget

All jobs **must use `-A re-com`**. Cap at **4 GPUs / 600 GB RAM concurrent across all your jobs** — leave headroom for other users.

## Picking a node (most → least capable)

| node | partition | GPUs | when to use |
| --- | --- | --- | --- |
| **hopper-2** | `hopper-2` | 4× H200 + 4× H200_4g (71GB MIG) + 4× H200_3g | Big batch / fast jobs. **DDP on H200 SIGSEGVs** — use 1-GPU only. |
| **hopper** | `hopper` | 4× H200 | Same constraint. |
| **stud-2, stud-3** | `short` | 2× RTX6000 each | Usually IDLE — best for opportunistic jobs. Stable DDP. |
| **dgx-1/2/3/4** | `short`/`long` | A100s | Stable DDP. Often busy. |

Check live state: `sinfo -o '%P %N %T %c %G' | head -20` and `scontrol show node <name> | grep AllocTRES`.

## Partition time caps

| partition | max walltime |
| --- | --- |
| `short` | 24h |
| `long` | 5d |
| `experimental` | 5d |
| `hopper` | 5d |
| `hopper-2` | 24h |

If you need >24h, use `long` or `experimental` (not `short` despite its name).

## Submitting jobs

Use `sbatch` with `--wrap` for one-liners, or a script file for multi-line. **Always log to `$HOME/<name>.log`** so output survives node teardown.

### Single-GPU on a specific node (preferred for most training jobs)

```bash
sbatch -A re-com -p short -t 24:00:00 \
  --nodelist=stud-2 --gres=gpu:rtx6000:1 \
  --cpus-per-task=16 --mem=200G \
  --output=$HOME/my_job.log \
  --wrap='cd ~/src/<repo> && PYTHONUNBUFFERED=1 ~/.pyenv/shims/python -m <module>'
```

### Multi-GPU DDP (only on stud / dgx — NOT H200)

```bash
sbatch -A re-com -p short -t 24:00:00 \
  --gres=gpu:2 --cpus-per-task=16 --mem=300G \
  --output=$HOME/my_job.log \
  --wrap='cd ~/src/<repo> && PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
    ~/.pyenv/shims/torchrun --standalone --nproc-per-node=2 -m <module> [args]'
```

### Hopper-2 single-GPU (when you need H200 speed)

```bash
sbatch -A re-com -p hopper-2 -t 24:00:00 \
  --gres=gpu:h200:1            # or gpu:h200_4g.71gb:1 (MIG slice, ~1/4 perf)
  --cpus-per-task=16 --mem=200G ...
```

### Chained jobs (dependencies)

```bash
sbatch ... --dependency=afterok:$PARENT_JOB_ID --wrap='...'
# Variants: afterany (regardless of exit), afternotok (only on failure)
```

The dependent job stays PENDING with reason `Dependency` until parent state matches, then auto-starts.

### Extending a job's walltime

Running jobs can't be extended by users (`scontrol update Job=ID TimeLimit=...` returns "Access/permission denied"). Plan walltime upfront — submitting with 24h is fine even if you expect <6h.

## Always-needed flags

- `PYTHONUNBUFFERED=1` — flush logs immediately so you see progress live
- `NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo` — DDP only, prevents NCCL hangs
- `--cpus-per-task=16` — slurm defaults to 1 CPU → DataLoader workers starve

## Monitoring

```bash
squeue -u $USER -o '%i %T %P %j %M %N'        # my jobs
sacct  -j JOBID -o JobID,State,Elapsed,ExitCode  # finished job info
scontrol show job JOBID                        # full details (StartTime, Reason, etc)
tail -f ~/my_job.log                           # live log
```

## Common pitfalls

1. **`--wrap='...for X in ...; do ...; done'`** runs in `/bin/sh`, NOT bash → `do` syntax errors. Write multi-line scripts to a file (`cat > script.sh`), `chmod +x`, and submit the file path.
2. **Output-dir collision**: training that writes to `<output>/{cfg.name}/...` will OVERWRITE an existing checkpoint if you reuse the same name with different overrides. **Always rename when changing the recipe.**
3. **Job array `%1` limit**: other users' big arrays (e.g., `1682886_[4-40%1]`) hold their nodes' GPU slots even when only 1 element runs. Their priority can push your jobs back.
4. **H200 + 2-GPU DDP = SIGSEGV** after ~2 min. Use 1-GPU or switch to stud/dgx.
5. **Don't delete cluster files** other than your own logs.
6. **Don't run training locally** — only on cluster. Login node is for `sbatch` + monitoring only.

## When no resources are free

The fast path is usually to **resubmit on a different GRES**: if `gpu:h200:1` waits long, try `gpu:h200_4g.71gb:1` (smaller MIG slice, often free); if hopper-2 is full, try `-p short --nodelist=stud-2`. Stud nodes are usually IDLE.

If everything's busy, dep-chain new jobs behind your own running ones (`--dependency=afterany:<your_running_jid>`) so they auto-start as GPUs free.
