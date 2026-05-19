# Training-script survey — NVIDIA Nemotron Reasoning Challenge

Cataloged 17 public notebooks (plus suryamilenial pulled earlier) that *actually train* a LoRA adapter on Nemotron-3-Nano-30B-A3B. Sorted by claimed public LB. Pure packaging/inference notebooks are listed separately at the bottom.

> Every notebook that scores ≥ 0.83 forks one of two recipes: **huikang's manual-loop pipeline** (Unsloth + Cut Cross-Entropy + MoE expert weight tying + manual lm_head LoRA, replays huikang's pre-tokenized corpus) or **dgxchen's TRL StratifiedSFTTrainer pipeline** (Tong's CoT CSV + per-type-balanced batches). Most other notebooks are variations on data or one hyperparameter.

## 0.85 LB tier — reference recipes

### huikang/end-to-end-finetuning-for-lb-0-85 — **0.85** (canonical)
- URL: <https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85>
- **Data**: `huikang/huikang-nemotron-repository-snapshot` (pre-tokenized `tokens/{problem_id}/synthetic.json` from huikang's own repo, training order replayed via `logprobs/index.jsonl`)
- **LoRA**: r=32 α=32 d=0, targets q/k/v/o + up/down + in_proj/out_proj + **lm_head**
- **Framework**: Unsloth FastLanguageModel + manual training loop (no SFTTrainer), Cut Cross-Entropy patched into forward
- **Optimizer**: AdamW (β=0.9/0.95), lr=2e-4 linear over 1000 steps, max_grad_norm=1e9, bf16
- **Key tricks**:
  - Manual lm_head LoRA add (Unsloth drops it for MoE)
  - LoRA params fp32, base bf16, MoE router fp32
  - `cut_cross_entropy.linear_cross_entropy` (no logits materialization)
  - MoE expert weight tying: all 128 experts share one LoRA factor on hidden-side; grads summed across experts
  - Patches `modeling_nemotron_h.is_fast_path_available = True`
  - Per-token mask training (prompt tokens weight=0)
  - Renames `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*` at save time
- **Why it works**: this is the only recipe that does all of CCE + MoE tying + lm_head LoRA + replay-the-good-corpus simultaneously. Every "0.85 reproduction" below is a partial restatement.

### panweiw/huikang-end-to-end-public-repro — **0.85** (byte-identical fork)
- URL: <https://www.kaggle.com/code/panweiw/huikang-end-to-end-public-repro>
- Identical file (109,526 bytes) to huikang's. Republished under a different account, no code changes. Listed only for completeness.

### dgxchen/training-with-unsloth-to-achieve-0-85-lb — **0.85** (TRL alternative)
- URL: <https://www.kaggle.com/code/dgxchen/training-with-unsloth-to-achieve-0-85-lb>
- **Data**: `dgxchen/nemotron-cot-tong/problem_ids_matched.csv` (huikang's CoT, re-matched to problem_ids in CSV form)
- **LoRA**: r=32 α=32 d=0, targets q/k/v/o + in_proj/out_proj + up/down — **no lm_head**
- **Framework**: Unsloth FastLanguageModel + TRL SFTTrainer with custom **StratifiedSFTTrainer** (per-type-balanced batches)
- **Optimizer**: AdamW (β=0.9/0.95), lr=2e-4 linear, max_grad_norm=1e9, bf16; mb=1, grad_accum=32, 1 epoch
- **Key tricks**:
  - StratifiedSFTTrainer distributes problem types evenly across effective batches
  - Drops lm_head LoRA + drops Muon vs Tong, otherwise Tong's hparams
  - Author notes mb=2 saves 3h but costs 0.1 LB — sticks with mb=1
- **Why care**: cleaner code than huikang's, drops the fastest-to-trip tricks (CCE, MoE tying, lm_head, fp32-LoRA cast). Matches LB despite that — strong evidence that **data + stratified batching is most of the signal**.

### pkuszboi/0-85-lb-training-with-muon — **~0.85** (Muon optimizer)
- URL: <https://www.kaggle.com/code/pkuszboi/0-85-lb-training-with-muon>
- **Data**: same as dgxchen
- **Framework**: forks dgxchen recipe; overrides `create_optimizer` with a custom **Muon** (Newton-Schulz zeropower iteration, torch.compile-d)
- **Optimizer**: Muon lr=2e-3, momentum=0.95, nesterov, backend_steps=5, applied to LoRA params only; 1 epoch, mb=2 grad_accum=8
- **Only Muon-based training notebook in the survey.** Author reports "around 0.85" — ties dgxchen but takes ~5h.

## 0.83 LB tier — variations

### suryamilenial/end-to-end-finetuning-for-lb-0-83-6e2fa5 — **0.83**
- URL: <https://www.kaggle.com/code/suryamilenial/end-to-end-finetuning-for-lb-0-83-6e2fa5>
- Verbatim huikang recipe with Modal-on-RTX-PRO-6000 deployment. This is the notebook our [src/train_huikang_style.py](src/train_huikang_style.py) is adapted from.

### konbu17/nemotron-tong-style-cot-sft-updated-v2 — **0.834 local CV**
- URL: <https://www.kaggle.com/code/konbu17/nemotron-tong-style-cot-sft-updated-v2>
- **Data**: `konbu17/exp024-tong-style-cot-data` (per-category CSVs reproducing Tong's deterministic CoT for GC/NC/UC/TE/BM/ET-Numeric/Cryptarithm)
- **LoRA**: r=32 α=32 d=0, full target set including lm_head
- **Framework**: HF transformers + TRL SFTTrainer
- **Optimizer**: AdamW (β=0.9/0.95), lr=2e-4 linear, wd=0
- **Key tricks**:
  - Deterministic per-category CoT generators (1-100 Roman table, 77-word Alice's Wonderland dictionary, 354-candidate per-bit testing, 32 op × 4 pairings for ET)
  - **Priority duplication**: rows with min logprob < -0.69 are duplicated for 2× weight
  - max_length forced 4096 (OOM at 8192 on Kaggle GPU)
- As shipped, default mode is `USE_PRETRAINED=1` (packages locally pre-trained adapter). The in-notebook training path runs at 4096 ctx and is admittedly weaker.

### tahaalam2009/nemotron-sft-final-0-83-lb — **0.83**
- URL: <https://www.kaggle.com/code/tahaalam2009/nemotron-sft-final-0-83-lb>
- **Data**: dgxchen's Tong CoT **merged** with `tahaalam2009/nemotron-0-90/dataset_generated.csv` for ONLY cryptarithm_deduce/guess and equation_numeric_guess (3× oversampled)
- **LoRA / framework / optimizer**: identical to dgxchen (StratifiedSFTTrainer)
- **Key tricks**:
  - Surgical category-level CoT merge: Tong's CoT for easy categories; author's regenerated CoT for hard ones (where author claims 89.8%/85.4%/92.6% generation accuracy vs Tong's 8.2%/6.7%/15.4%)
  - 3× oversample for the 3 hard categories
  - In-notebook vLLM validation on 950 rows with category breakdown
- **The most data-focused intervention in the set.** Keeps proven 0.85 hparams, only swaps in better hard-category CoT — but **LB drops to 0.83**. Suggests local-eval gain doesn't translate; possible test-set overlap with Tong's CoT or quality regression on the swapped categories.

## 0.82 LB tier

### tahaalam2009/end-to-end-finetuning-for-lb-0-82-csv-custom — **0.82**
- URL: <https://www.kaggle.com/code/tahaalam2009/end-to-end-finetuning-for-lb-0-82-csv-custom>
- **Data**: own CSV (`tahaalam2009/nemotron-0-90/problem_ids_matched.csv`), re-tokenized to huikang's `(tokens, targets, weights)` format on the fly using `<|im_start|>assistant\n` substring offset for assistant-mask
- **Framework / tricks**: huikang's manual loop (CCE, MoE tying, lm_head LoRA, fp32 LoRA, Mamba fast-path patch, lm_head key rename) — fully preserved
- Markdown brags "95.8% train.csv accuracy vs huikang's 87.7%" using a custom 0-90 CoT dataset. **LB 0.82 undercuts the train claim** → probable train/test gap on the regenerated CoT.

## 0.72 LB tier

### yashm917/nemotronmodel-reasoning-0-72-lb-score-unsloth — **0.72**
- URL: <https://www.kaggle.com/code/yashm917/nemotronmodel-reasoning-0-72-lb-score-unsloth>
- **Data**: competition `train.csv` (raw) + author's rule-based `build_think` CoT generator (rotate/XOR brute force for bitmanip, char-map for symbol_transform, etc.)
- **LoRA**: r=32 **α=16 d=0.05** (only entry with α<rank and non-zero dropout), `target_modules="all-linear"`
- **Framework**: Unsloth + TRL SFTTrainer + `train_on_responses_only` (Unsloth helper masks user/system tokens automatically)
- **Optimizer**: **adamw_8bit**, lr=1e-4 cosine, warmup_ratio=0.03, batch=4 grad_accum=3, 2 epochs, **max_seq_len=2048** (only entry this low)
- The only notebook here using `target_modules="all-linear"`, `adamw_8bit`, and a 2048 seq cap. The low seq cap probably truncates many CoTs — likely explains the LB gap.

## 0.67 LB tier

### kienngx/nvidia-nemotron-training-copy-run-instantly — **0.67** (in this repo as a local .ipynb)
- URL: <https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-copy-run-instantly>
- **Data**: raw `train.csv` only (no CoT — trains on the bare `answer` string)
- **LoRA**: r=32 α=32 d=0.05, `target_modules="all-linear"`
- **Framework**: TRL SFTTrainer (no Unsloth)
- **Optimizer**: lr=1e-4 cosine, 2 epochs on a 1,200-row random subsample
- **What it proves**: with no CoT and a small random subsample, the ceiling is ~0.67. Every CoT-trained notebook beats this by ≥ 0.15. **The "training notebook → ≥ 0.83" requires a real CoT corpus, not just the comp's raw `answer` column.**

## Unscored but interesting

### konbu17/nemotron-sft-lora-with-cot — upstream CoT-data source
- URL: <https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot>
- **Data**: `konbu17/nemotron-sft-lora-cot-selection/train_split_with_cot.csv` (LLM-generated CoT, rule-verified-correct; 2,907 sampled from 6,558)
- **LoRA**: r=32 α=32 d=0.05, regex `.*(in_proj|out_proj|up_proj|down_proj)$`
- **Framework**: HF transformers + TRL SFTTrainer (no Unsloth)
- **Key tricks**:
  - **Type-stratified sampling**: NC 300, GC 400, UC 700, TE 700, BM 607, ET 200 (uses all BM/ET because they're hard)
  - 77-word Alice's Wonderland dictionary embedded in cipher prompts to lift TE pass rate
  - Per-bit boolean function CoT generator for bit_manipulation
  - 7600-token CoT length cap with shorten-and-retry
- **Why care**: this is the upstream dataset consumed by amanatar and used as fallback by tahaalam2009. The recipe itself is conservative HF+TRL — the contribution is the **data + the cipher dictionary trick**.

### amanatar/nemotron-ultimate-sft-grpo-v3 — scaffolds GRPO but doesn't run it
- URL: <https://www.kaggle.com/code/amanatar/nemotron-ultimate-sft-grpo-v3>
- **Data**: `konbu17/nemotron-sft-lora-cot-selection` (all 6,558 verified CoT) + own deterministic solvers as fallback
- **LoRA**: r=32 α=32 d=0, regex targets
- **Framework**: TRL SFTTrainer (+ GRPOTrainer scaffolded but `USE_GRPO=False`)
- **Optimizer**: AdamW (TRL default), lr=1e-4 cosine, warmup 0.05, **NEFTune α=5.0**
- **Key tricks**:
  - NEFTune noise α=5.0 during SFT (only notebook to do this)
  - SFT_MAX_LEN=7680 (matches eval `max_tokens`, no CoT truncation)
  - Uses **all** 6,558 verified konbu17 CoT samples (vs konbu17's downsample to 2,907)
  - GRPO machinery present (cosine + format + reasoning rewards) but `USE_GRPO=False` — only SFT actually runs
- **Only notebook in survey scaffolded for RL**; reward functions are written but disabled.

### konstantinboyko/01-04-training-with-unsloth-batch-2-epoch-2 — dgxchen scaled up
- URL: <https://www.kaggle.com/code/konstantinboyko/01-04-training-with-unsloth-batch-2-epoch-2>
- Forks dgxchen: same data, same StratifiedSFTTrainer. Doubles training (mb=2 grad_accum=32, 2 epochs, effective batch 64) and **re-adds lm_head LoRA** (which dgxchen had removed). No score claim.

### leegongman/sft-my-data-balace — huikang loop + custom corpus
- URL: <https://www.kaggle.com/code/leegongman/sft-my-data-balace>
- **Data**: `leegongman/merged-sft-data-balance/merged_sft_dataset` — author's own re-balanced corpus packaged in huikang's `tokens/{pid}/synthetic.json` layout
- Bit-for-bit copy of huikang's manual training loop. Differentiator is the custom dataset.

### llkh0a/nemotron-unsloth-sft-training-3-30-2 — most engineered single file
- URL: <https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2>
- **Data**: `llkh0a/nvidia-nemotron-distiled-dataset/splited.csv` + author's per-category trace CSVs (bit_manipulation_traces_v4, numeric_equation_traces_new); per-category sample caps
- **LoRA**: r=32 **α=16 d=0.1** (atypical), targets include `embed_tokens` + `gate_proj` (only notebook to LoRA `embed_tokens`)
- **Framework**: Unsloth + TRL SFTTrainer + warm-start from prior PEFT adapter
- **Optimizer**: **adamw_8bit**, lr=1e-4 cosine, warmup 0.03, 2 epochs, batch=2 grad_accum=1, max_seq_len=3500
- **Key tricks**:
  - **Custom per-token weighted loss**: tokens after the LAST `\boxed{` are upweighted (`BOXED_LOSS_WEIGHT=5.0`)
  - LoRA on `embed_tokens` (rare)
  - Per-category sample budgets (bit_manipulation 1,400 / text_decryption 1,300 / numeric_equation 600 / others ~200)
  - Optional warm-start from previous run's adapter
  - Built-in vLLM offline CV harness with EarlyStopping
  - α<rank + dropout 0.1 — counter to consensus
- **The most heavily engineered single notebook** (~187 KB). Hyperparameters diverge sharply from the consensus.

### warmtea/lb-training — "Think Twice" verifier suffixes
- URL: <https://www.kaggle.com/code/warmtea/lb-training>
- Forks dgxchen recipe + data. Augments every CoT by:
  - `strip_boxed_commands` removes intermediate `\boxed{...}` from CoT (keeps only final)
  - `build_type_aware_verifier` appends a category-specific 3-line "Independent check" block before `</think>\boxed{}`
  - "MISMATCH_POLICY": when extracted answer from CoT ≠ ground truth, emit a short stub assistant response instead of dropping
- Markdown verbatim-copies dgxchen claiming 0.84–0.85.

### tahaalam2009/nemotron-batched-logprob-filter-train — "Stage 2"
- URL: <https://www.kaggle.com/code/tahaalam2009/nemotron-batched-logprob-filter-train>
- **Data**: `tahaalam2009/nemotron-logprob/results/filtered_merged_dataset.csv` (own dataset filtered by per-sample min-logprob)
- Uses huikang's manual loop. Distinctive: **lr=5e-5** ("to protect base knowledge") vs everyone else's 2e-4, and 3× oversample for cryptarithm_deduce/guess and equation_numeric_guess.

## Packaging / inference notebooks (not training)

| Notebook | What it is | LB |
|---|---|---|
| [hammadfarooq470/think-twice-self-correcting-reasoning](https://www.kaggle.com/code/hammadfarooq470/think-twice-self-correcting-reasoning) | Pure packaging — patched `tinker-cookbook` rank-32 SVD merge on huikang v20. No training despite the title. | ? |
| [jiazhuang/nemotron-local-cv](https://www.kaggle.com/code/jiazhuang/nemotron-local-cv) | Inference-only CV harness: vLLM + LoRARequest against `jiazhuang/nemotron-val-set-950`. **Reproduces the comp scoring code with `rel_tol=1e-2`** — useful for offline eval of any adapter. | n/a |

## Datasets-used summary

The single most-used training data is **huikang's pre-tokenized corpus**:
- `huikang/huikang-nemotron-repository-snapshot` — used by huikang, panweiw, suryamilenial (with the `tokens/{pid}/synthetic.json` layout)
- `dgxchen/nemotron-cot-tong/problem_ids_matched.csv` — Tong's CoT re-matched to problem_ids in CSV form; used by dgxchen, pkuszboi, konstantinboyko, warmtea, tahaalam2009 (final-0-83 in part)

Second tier — author-generated CoT corpora used as variations:
- `konbu17/nemotron-sft-lora-cot-selection` — LLM-generated rule-verified CoT (used by konbu17, amanatar)
- `konbu17/exp024-tong-style-cot-data` — per-category deterministic CoT reproducing Tong's (konbu17 v2)
- `tahaalam2009/nemotron-0-90` — regenerated CoT with explicit focus on cryptarithm/equation_numeric (used by tahaalam2009 variants)
- `tahaalam2009/nemotron-logprob/results/filtered_merged_dataset.csv` — logprob-filtered CoT (tahaalam2009 stage 2)
- `leegongman/merged-sft-data-balance` — custom rebalanced corpus
- `llkh0a/nvidia-nemotron-distiled-dataset` — author's distilled traces (llkh0a only)

Raw `train.csv` only (no CoT): yashm917 (0.72), kienngx (0.67) — both demonstrate the no-CoT ceiling.

## Cross-cutting observations

1. **Two reference recipes account for every ≥ 0.83 notebook.** Either huikang's manual-loop + tied MoE + CCE, or dgxchen's StratifiedSFTTrainer. The rest is variation on data (corpus choice, CoT generator, category oversampling).
2. **No public RL.** amanatar scaffolds GRPO but disables it. No PPO/DPO loops in any sampled notebook. The 0.87 cluster on the leaderboard isn't being approached publicly via RL.
3. **One notebook tries Muon** (pkuszboi). Result: ties at 0.85, no jump. Optimizer is not the bottleneck.
4. **The data > hparams pattern is consistent**: dgxchen drops lm_head LoRA + CCE + MoE tying from huikang's recipe but matches 0.85 because the corpus is the same. tahaalam2009 keeps the recipe but swaps the corpus → drops to 0.82–0.83.
5. **Hard categories matter disproportionately.** Multiple authors single out `cryptarithm_deduce/guess` and `equation_numeric_guess` as the categories Tong's CoT solves poorly (~7–15% accuracy in tahaalam2009's measurement). All targeted-CoT efforts so far have *lost* LB while raising local CV — suggesting the public LB is dominated by other categories or that the regenerated CoT for hard categories overfits.
6. **Sequence length is load-bearing.** 8,192 ctx (huikang/dgxchen) works at LB 0.85; 4,096 (konbu17 v2 in training mode) gives 0.834 local; 2,048 (yashm917) gives 0.72. Truncating CoTs hurts.
7. **Local CV ≠ LB.** tahaalam2009 reports 95.8% train accuracy → 0.82 LB. konbu17 v2 reports 0.834 local → unclear LB. The jiazhuang harness is the right tool to validate before spending a submission slot.

## What this implies for our pipeline

- [src/train_huikang_style.py](src/train_huikang_style.py) already implements the canonical huikang recipe. That's the ceiling-of-public-knowledge starting point.
- The cheapest experimental knob is **the corpus**: swap in dgxchen's CSV form, konbu17's verified-CoT, or generate hard-category CoT ourselves. None of those would change the training loop.
- The cheapest *recipe* knob worth trying is **NEFTune α=5.0** (only amanatar tries it; no LB number).
- Before any LB submission, run [src/verify_adapter.py](src/verify_adapter.py) for structure, then ideally run jiazhuang's CV harness offline on a held-out slice.
