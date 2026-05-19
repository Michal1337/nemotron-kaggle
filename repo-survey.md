# huikang/nemotron repo survey

Deep dive into the [huikang/huikang-nemotron-repository-snapshot](https://www.kaggle.com/datasets/huikang/huikang-nemotron-repository-snapshot) (mirror of [tonghuikang/nemotron](https://github.com/tonghuikang/nemotron)). This is the Progress Prize winner's full pipeline — far richer than what was published on Kaggle as the 0.85 notebook.

> **TL;DR**: The public Kaggle notebook (`huikang/end-to-end-finetuning-for-lb-0-85`) is a simplified Unsloth port doing plain SFT. The real pipeline is **multi-epoch SFT + RL fine-tuning** (PPO / CISPO / DRO / importance sampling) layered on top, plus a deterministic corpus generator backed by **per-category programmatic solvers** and **5 augmenters** that produce ~half the training data. None of this is in any public notebook.

## The pipeline

```
problems.jsonl + train.csv         (9,500 original problems)
        │
        ▼  reasoning.py
           Per-category reasoner (bit_manipulation, cipher, gravity, …)
           writes deterministic CoT to reasoning/{pid}.txt for solved ones.
        │
        ▼  augmentation.py
           Calls 5 augmenters → synthetic problems in augmentations/{pid}.txt:
             matching (4515), concatenation (1500), splitting (1500),
             spelling (648), lstrip (300)
        │
        ▼  corpus.py
           Tokenizes prompt + reasoning + \boxed{answer} into segments,
           masks prompt tokens. TOKEN_LIMIT=8192. Writes corpus/{pid}/synthetic.jsonl
           and corpus.jsonl index.
        │
        ▼  train_sft.py (Tinker or Modal backend)
           Epoch 0:  cross_entropy SFT, saves per-token logprobs
           Epoch 1+: switch to importance_sampling / ppo / cispo / dro using
                     epoch-0 logprobs as reference (KL/clip anchor)
        │
        ▼  upload_adapter.py
           Downloads Tinker checkpoint, uploads to Kaggle Models hub
           at huikang/nemotron-adapter/Transformers/default
```

## File inventory

### Top-level Python entry points

| File | Size | Purpose |
|---|---:|---|
| [reasoning.py](nemotron-master/reasoning.py) | 8.7K | Run per-category reasoners over problems.jsonl, write CoT to reasoning/{pid}.txt, update status flags |
| [augmentation.py](nemotron-master/augmentation.py) | 1.4K | Driver that calls every augmenter and writes augmentations/{pid}.txt |
| [corpus.py](nemotron-master/corpus.py) | 9.7K | Tokenize (prompt + reasoning + boxed answer) → corpus/{pid}/synthetic.jsonl + index |
| [train_sft.py](nemotron-master/train_sft.py) | 20K | **Real training script** — multi-epoch SFT + RL via Tinker/Modal |
| [train_common.py](nemotron-master/train_common.py) | 3.6K | TrainingExample loader, build_datum helper |
| [loss_config.py](nemotron-master/loss_config.py) | 13K | **The RL machinery** — 5 loss functions: cross_entropy, ce-weighted, importance_sampling, PPO, CISPO, DRO |
| [lr_schedule.py](nemotron-master/lr_schedule.py) | 1.0K | LinearDecay + StepLinearDecay schedules |
| [generate_csv.py](nemotron-master/generate_csv.py) | 4.4K | One-off dataset.csv builder (decodes raw token logs for the HTML dashboards) |
| [upload_adapter.py](nemotron-master/upload_adapter.py) | 6.6K | Pulls Tinker checkpoint → uploads to `huikang/nemotron-adapter/Transformers/default` |
| [notebook_tinker.py](nemotron-master/notebook_tinker.py) | 55K | Alternative Python-notebook entry; also defines the `verify()` function used by the eval |

### Per-category solvers (`reasoners/`)

Called from `reasoning.py` to produce the training CoT. Each takes a `Problem` and returns a multi-line trace ending with `\boxed{answer}`.

| File | Size | Category | Strategy |
|---|---:|---|---|
| [bit_manipulation.py](nemotron-master/reasoners/bit_manipulation.py) | 36K | bit_manipulation | Per-bit rule selector over I/NOT/Const/AND/OR/XOR families with stride-consistency preference |
| [cipher.py](nemotron-master/reasoners/cipher.py) | 15K | cipher | Substitution-cipher decoder backed by `wonderland.txt` (77-word lexicon!) |
| [cryptarithm.py](nemotron-master/reasoners/cryptarithm.py) | 5.1K | cryptarithm_{deduce,guess} | **Concat-only solver** — fails on non-concat ops, returns None |
| [equation_numeric.py](nemotron-master/reasoners/equation_numeric.py) | 23K | equation_numeric_{deduce,guess} | Enumerates ~30 candidate ops × 4 reverse-combos with prefix/suffix `-` detection |
| [gravity.py](nemotron-master/reasoners/gravity.py) | 3.1K | gravity | `d = k * t²`, median-k via long division, long-multiply for question |
| [numeral.py](nemotron-master/reasoners/numeral.py) | 1.4K | numeral | Arabic → Roman greedy place-value table |
| [unit_conversion.py](nemotron-master/reasoners/unit_conversion.py) | 2.7K | unit_conversion | Same median-factor pattern as gravity |
| [store_types.py](nemotron-master/reasoners/store_types.py) | 8.6K | (shared) | `Problem`/`Example` dataclasses + long_division_lines / long_multiplication_lines (verbose trace helpers) |
| [wonderland.txt](nemotron-master/reasoners/wonderland.txt) | 532B | (data) | 77-word Alice's-Wonderland vocabulary — *the only candidate plaintexts for cipher* |
| [dictionary.txt](nemotron-master/reasoners/dictionary.txt) | 15K | (unused) | 1596-word frequency-sorted list, no `.py` reads it — staged for future cipher fallback |

### Synthetic data generators (`augmenters/`)

All called from `augmentation.py`. Return `list[{id, prompt, completion, category}]`. All wrap prompts in the same "In Alice's Wonderland, secret processing rules…" frame.

| File | Size | Category | Output | What it teaches |
|---|---:|---|---:|---|
| [matching.py](nemotron-master/augmenters/matching.py) | 8.4K | matching | **4,515** | Parses existing `reasoning/*.txt` bit-manipulation traces and turns each per-op section (Matching/Left/Right/Best) into a focused subtask. *Downstream of bit-manipulation reasoner.* |
| [concatenation.py](nemotron-master/augmenters/concatenation.py) | 2.6K | concatenation | 1,500 | `【]】【}】【@】 → 【]}@】` — merge per-char bracketed symbols. 28-char symbol pool, seed 99 |
| [splitting.py](nemotron-master/augmenters/splitting.py) | 2.6K | splitting | 1,500 | Inverse of concatenation. seed 77 |
| [spelling.py](nemotron-master/augmenters/spelling.py) | 3.9K | spelling | 648 | `dog cat bee → –d–o–g–c–a–t–b–e–e–`. Loads tokens 2-8 chars from `tokenizer.json`, both bare + space-prefixed (Ġ). seed 42 |
| [lstrip.py](nemotron-master/augmenters/lstrip.py) | 2.4K | lstrip | 300 | `【   $%^】 → 【$%^】` — strip leading spaces. seed 91 |
| `__init__.py` | 0B | — | — | Empty marker |

**Total synthetic: 8,463 problems** across 5 categories — none exist in `problems.jsonl`/`train.csv`. These ARE the "secret weapon."

### Offline investigation tools (`investigators/`)

These are NOT in the training loop — they're standalone scripts huikang ran to populate `reasoning/` and `investigations/` directories, and to analyze runs.

| File | Size | Purpose |
|---|---:|---|
| [bit_manipulation.py](nemotron-master/investigators/bit_manipulation.py) | 11K | Brute-force solver: tries 44 single transforms (rotate/shift/NOT/identity) then 1-/2-/3-element combinations under XOR/AND/OR — writes `investigations/{pid}.txt` |
| [cryptarithm_deduce.py](nemotron-master/investigators/cryptarithm_deduce.py) | 15K | Backtracking DFS solver for symbol→digit + operator→{add, abs_diff, mul, concat, rev_concat} mappings. **Uses SIGALRM — Unix-only** |
| [bit_manipulation_analysis.py](nemotron-master/investigators/bit_manipulation_analysis.py) | 6.0K | Diagnostic — bins bit problems by "stride-consistent sections" in the rule vector |
| [augment_data.py](nemotron-master/investigators/augment_data.py) | 5.1K | Earlier/standalone spelling generator, superseded by `augmenters/spelling.py` |
| [calc_accuracy.py](nemotron-master/investigators/calc_accuracy.py) | 2.6K | **Eval reference** — scores `id,answer,predicted` CSV per category using a `verify()` function (matches the comp scorer exactly) |
| [get_examples.py](nemotron-master/investigators/get_examples.py) | 4.8K | Picks problem_ids from a run's index.jsonl sorted by min_logprob or step (for follow-up runs) |

### Backend abstraction (`trainer/`)

| File | Size | Purpose |
|---|---:|---|
| [client.py](nemotron-master/trainer/client.py) | 5.7K | `ServiceClient(backend="tinker"|"modal")` — same interface, switches between hosted Tinker and a Modal-deployed `trainer-gpu` GPU class. This is how huikang ran identical code on both Tinker (paid) and his own Modal infra (cheaper). |

### Data / output dirs

| Dir | Contents |
|---|---|
| `problems/` | One JSONL per problem (~9,500 files) with full details (question + examples + answer) |
| `problems.jsonl` | Index with `{id, category, status, submission}` per problem |
| `reasoning/` | Per-problem CoT text from reasoners — input to corpus.py |
| `investigations/` | Per-problem investigation notes from offline investigators — fallback for "hypothesis_formed" status |
| `augmentations/` | Per-problem synthetic problem files from augmenters |
| `corpus/{pid}/synthetic.jsonl` | Tokenized segments (output of corpus.py, input to train_sft.py) |
| `corpus.jsonl` | Index of corpus entries with token counts + `included` flag |
| `training/sft/{date}/` | 8 dated training runs with tokens/, logprobs/index.jsonl, metrics.jsonl, loss.jsonl, config.json |
| `raw/` | Raw per-generation token logs (input to generate_csv.py) |
| `*.html` | Dashboard (`./serve.sh` to view) — surfaces metrics from training runs |

## The big finding: the RL machinery

[loss_config.py](nemotron-master/loss_config.py) defines **5 loss functions**:

| Class | name | When used |
|---|---|---|
| `CrossEntropyLossConfig` | `cross_entropy` | Default SFT — epoch 0 |
| `CrossEntropyWithWeightingLossConfig` | `cross_entropy` | + per-token `branch_weight = min(1, |lp|/branch_logprob)` (downweight easy tokens) + `first_cutoff_weight` (e.g. 0.5 epoch 0) |
| `ImportanceSamplingLossConfig` | `importance_sampling` | Off-policy correction via stored ref logprobs |
| `PPOLossConfig` | `ppo` | Clipped surrogate, `clip_low/high = 0.2 / 0.2` |
| `CISPOLossConfig` | `cispo` | Asymmetric clip `0.8 / 1.2` (favors trajectories where new policy assigns higher prob) |
| `DROLossConfig` | `dro` | KL penalty with `beta`, with epoch-0 passthrough (beta=0) |

[train_sft.py:346-502](nemotron-master/train_sft.py#L346-L502) per-epoch loop:

```python
for epoch in range(cfg.num_epochs):
    loss_fn_config = cfg.loss_config.config(epoch)   # epoch-aware config
    ...
    for batch_indices in batches:
        for example in batch_examples:
            tokens, advantages = example.load_tokens()
            if epoch == 0:
                ref_logprobs = None
                prev_logprobs = None
            else:
                ref_logprobs = all_ref_logprobs[key][...]   # collected during epoch 0
                prev_logprobs = all_prev_logprobs[key][...] # last epoch's
            datum = build_datum(tokens, advantages, ref_logprobs, prev_logprobs, epoch, cfg.loss_config)
            ...
        # forward_backward with chosen loss_fn name
        # optim step
        # save per-example logprobs:
        if epoch == 0:
            all_ref_logprobs[key] = list(lp_data)
        all_prev_logprobs[key] = list(lp_data)
```

**This is the gap between huikang's adapter (0.86 after our conversion) and every public Unsloth notebook (0.83-0.85).** The public notebooks all do epoch 0 only. huikang does epoch 0 → save logprobs → epoch 1+ with PPO/CISPO/DRO using saved logprobs as the policy-anchor for KL/clip.

## How the data is shaped (and what it teaches)

### Original-problem CoT (from `reasoners/`)
Verbose, deterministic, step-by-step trace mimicking student work. E.g. for `gravity` (`d = k * t²`): compute k for each example via long division, take median k, long-multiply k * t² for the question. **Every intermediate digit is shown.** The model learns to emit this verbose procedure.

For `cipher`: enumerates *every* word in `wonderland.txt` (77 words!) as candidate plaintext, prints match/unmatchable/contradiction per position. **The wonderland.txt = 77-word lexicon is the entire cipher answer space.** No notebook reproduces this.

### Synthetic data (from `augmenters/`)
Pure prompt-completion pairs, all framed as "secret processing rules" in Alice's Wonderland. These don't appear in `problems.jsonl` and are pure augmentation:

- **matching (4515)** — Bit-manipulation sub-task isolation. Each example is one operation-section (Matching output / Left chain / Right chain / Best) extracted from existing reasoning traces. Teaches the model to do the *inner* steps of bit_manipulation reasoning standalone. Sampled with deterministic SHA-256 hash mod (1/100 for `all_absent`, 1/10 for `both_none`, 1/5 for `<4 matches`).
- **concatenation/splitting (1500 each)** — Per-char bracket manipulation, 28-symbol pool.
- **spelling (648)** — Char-by-char spelling with bare + space-prefixed tokens, 100 lines per problem.
- **lstrip (300)** — Leading-space stripping in bracket strings.

**Augmentations have a different completion format** ([corpus.py:240](nemotron-master/corpus.py#L240)) — no `\boxed{...}` suffix, just `{completion}\n</think><|im_end|>`. The original problems use `</think>\n\\boxed{{{answer}}}<|im_end|>`. That's a notable structural difference.

### The filter chain
[corpus.py:170-227](nemotron-master/corpus.py#L170-L227):
```python
problem_ids = sorted(
    pid for pid in problem_cats
    if (REASONING_DIR / f"{pid}.txt").exists()    # only included if reasoner succeeded
    and pid in prompts                            # only if in train.csv
)
```
- Reasoners only write to `reasoning/{pid}.txt` if `compare_answer(stored, predicted)` passes (rel_tol=1e-2 for nums, lowercase for strings, exact for binary)
- That's why `status: rule_found` problems can still be excluded (the reasoner returned `None` for hard instances)
- 8,333 rule_found in `problems.jsonl` → 7,830 in `04-08-16-14` corpus (after `cryptarithm.py` returning None on non-concat ops + `equation_numeric.py` returning None on novel-op cases + truncation)

## Tinker vs Modal backend

[trainer/client.py](nemotron-master/trainer/client.py) — `ServiceClient(backend="tinker" | "modal")`. Same interface:
```python
training_client = await service_client.create_lora_training_client_async(
    base_model=cfg.model_name, rank=cfg.lora_rank, ...
)
await training_client.forward_backward_async(data, loss_fn=..., loss_fn_config=...)
await training_client.optim_step_async(adam_params)
```

For `backend="modal"`, it calls `modal.Cls.from_name("trainer-gpu", "Trainer")` — meaning huikang has a self-hosted Modal-deployed trainer that mirrors Tinker's interface. **We could in principle plug Unsloth into the same interface** instead of Tinker — that's the path to running this code without paying Tinker.

## The eval — `calc_accuracy.py`

The simplest possible scorer. Loads `problems.jsonl` for `id → category`, reads a CSV with `id, answer, predicted`, calls `verify(answer, predicted)` per row, accumulates per-category accuracy.

`verify()` is **not in calc_accuracy.py itself** — it's imported from a missing `build_generation_index` module, but the same function is defined in [notebook_tinker.py:484](nemotron-master/notebook_tinker.py#L484):
```python
def verify(stored: str, predicted: str) -> bool:
    s, p = stored.strip(), predicted.strip()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return s.lower() == p.lower()
```

This matches the competition scorer exactly. Output is a per-category table in fixed order: numeral, unit_conversion, gravity, cipher, bit_manipulation, equation_numeric_{deduce,guess}, cryptarithm_{deduce,guess}.

## What's actionable for our repo

### Tier 1 — works as-is on our cluster, high value

1. **Use the latest corpus dir for SFT**. We've been using `04-08-16-14` (7,830). The bigger one `04-10-04-33` has 15,679 examples including the synthetic augmenter categories. Use it if you want signal the public notebooks don't have:
   ```bash
   python src/train_huikang_style.py \
     --corpus-path .../training/sft/04-10-04-33/tokens \
     --train-order .../training/sft/04-10-04-33/logprobs/index.jsonl \
     ...
   ```

2. **Port `calc_accuracy.py` to local eval**. Generate adapter predictions with vLLM on a held-out slice, score with the same `verify()` logic. Saves daily submission slots. ~50 lines.

3. **Re-run augmenters to regenerate / extend augmentation data**. The augmenter rng seeds are hard-coded (42, 77, 91, 99) — outputs are deterministic. Change seeds + re-run `augmentation.py` → get fresh synthetic problems (different concrete instances, same distribution).

### Tier 2 — meaningful work but tractable

4. **Port the multi-epoch + RL loop to Unsloth**. `loss_config.py` is 13KB of pure-Python loss math, no Tinker dependency. The hard part is:
   - Save per-token logprobs at end of epoch 0
   - Pass them as `ref_logprobs` to the loss function in epoch 1+
   - Implement PPO clip / CISPO clip / DRO in PyTorch (small)
   - The `tinker.Datum` abstraction needs replacing with a HF-style batch
   Estimated effort: 1-2 days for a clean port, longer to tune.

5. **Use the Modal-trainer pattern** as inspiration for a single-GPU trainer with the same loss-config interface. If we can run on our cluster's 100GB GPU with the same loss-config flexibility, we don't need either Tinker or Modal.

### Tier 3 — bigger investments

6. **Extend `reasoners/cryptarithm.py`** beyond concat ops. Currently returns None for any non-concat operator, which is why cryptarithm coverage is so low (19% in `04-10-04-33`). Adding `add` / `mul` / `abs_diff` rule discovery + verbose trace would directly raise the included-corpus count for the hardest category.

7. **Extend `reasoners/equation_numeric.py`** — already enumerates ~30 ops but falls back to "most common + force abs diff" when the question op is novel. Better fallback handling could rescue some currently-None outputs.

8. **Write a new augmenter** — e.g. `augmenters/concatenation.py` and `splitting.py` are 80-line templates. A targeted `augmenters/cipher_substitution.py` that generates cipher problems with broader vocabulary than the 77-word `wonderland.txt` could fill an obvious gap.

### What we can't easily do

- **`investigators/cryptarithm_deduce.py` uses SIGALRM** (Unix-only timeout). If you want to re-run it on your cluster (Linux, so it works), you can extend the corpus with newly-solved cryptarithm problems. On Windows it won't run.
- **Re-train via Tinker** — paid service, not free. Modal backend is an alternative if you have Modal credits.
- **Reproduce the dashboard** — `./serve.sh` runs a static HTTP server for `*.html`. We don't need it for training but it's useful for browsing.

## Honest assessment of the public-LB gap

| Gap | Where it lives | Tractable to close? |
|---|---|---|
| **0.83 → 0.85** | Faithful single-epoch SFT on `04-08-16-14` (or `04-10-04-33`) corpus. Mostly execution noise + maybe corpus choice. | Yes — already happening with our `train_huikang_style.py` |
| **0.85 → 0.86** | Multi-epoch RL training (PPO/CISPO/DRO using epoch-0 logprobs). | Yes — port `loss_config.py` to Unsloth. Medium effort. |
| **0.86 → 0.87** | huikang's actual published adapters got 0.86 publicly via the kien-tinker chain. The 0.87 cluster is 4 teams; we don't know what they did. Plausibly: extended corpus (custom augmenters, more cryptarithm coverage), or RL on top of huikang's adapter. | Speculative — corpus extension is the obvious starting point |

The path from "where we are" (0.86 from our v20 self-conversion) past 0.86 is **either** corpus extension (write new augmenters / improve reasoners) **or** RL on top of huikang's adapter (port loss_config + run a short PPO continuation). Both are weeks of work, not days. Neither has been done publicly.
