# NVIDIA Nemotron Model Reasoning Challenge — Notes

Compiled from the Kaggle competition page, the official starter notebook, public datasets/notebooks, the winning team's GitHub repo, and NVIDIA's posts.

> Sources cited inline. Where the live Kaggle page returned only metadata (JS-rendered),
> facts are pulled from the official starter notebook (`ryanholbrook/nvidia-nemotron-submission-demo`),
> the Progress Prize-winning repo (`tonghuikang/nemotron`), and public Kaggle datasets.

---

## 1. The competition at a glance

| Field | Value |
|---|---|
| Slug | `nvidia-nemotron-model-reasoning-challenge` |
| Host / sponsor | NVIDIA AI; powered by Google Cloud |
| Prize pool | **$106,388** total. 1st place: **$25,000 + 5× DGX Spark** (~$4,699 each). |
| Start date | ~March 16, 2026 |
| Progress Prize cutoff | ~April 9, 2026 (winner already declared — see §6) |
| Entry / final deadline | June 8, 2026 (Kaggle's LinkedIn) — June 15, 2026 (other source). Verify on the Kaggle page. |
| Compute | Google Cloud **G4 VMs**, **NVIDIA RTX PRO 6000 Blackwell** (sm_120, ~96 GB VRAM) |
| Internet | **Disabled** during scoring runs (per starter notebook) |
| Tagline | "Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark" |

What makes this competition unusual: **you do not submit predictions.** You submit a **LoRA adapter** that Kaggle then runs against a held-out reasoning benchmark.

---

## 2. The task

Given a base model (Nemotron-3-Nano-30B-A3B), produce a LoRA adapter (max **rank 32**) that improves its accuracy on a private NVIDIA reasoning benchmark.

### Problem format
Each problem shows a few input → output examples that share a hidden transformation rule, then asks for the output of a new input. From the public `train.csv` columns and `tonghuikang/nemotron`'s breakdown:

| Column | Meaning |
|---|---|
| `id` | 8-char hex id |
| `prompt` | full natural-language statement (with examples + final query) |
| `answer` | the gold answer string |

The model is expected to put the answer in `\boxed{...}`. Per the winning repo:
> *"Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`"* — appended to every prompt.

### Problem categories (counts from `kishanvavdara/nemotron-reasoning-traj`, 9,500 problems)

| Category | Count | Example |
|---|---:|---|
| `bit_manipulation` | 1,602 | 8-bit binary input → output (XOR/rotate/NOT compositions) |
| `gravity` | 1,597 | physics-style numeric reasoning |
| `unit_conversion` | 1,594 | unit/measure transformations |
| `cipher` | 1,576 | letter shifts, substitutions on text |
| `numeral` | 1,576 | base / numeral-system conversions (e.g. → Roman numerals) |
| `equation_symbolic` | 823 | algebraic manipulation |
| `equation_numeric` | 732 | numeric equation solving |

Smaller / synthetic categories surfaced in the winning repo: `cryptarithm_deduce`, `matching`, `spelling`, `concatenation`, `splitting`, `lstrip`.

### Baseline difficulty
Unfine-tuned Nemotron-3-Nano-30B-A3B with chain-of-thought scores roughly:
- `true` (all runs correct): 4,423 / 9,500 (~46.6 %)
- `false` (all runs wrong): 4,738 / 9,500 (~49.9 %)
- `partial`: 339 / 9,500 (~3.6 %)

So the headroom is the ~50 % the base model can't solve.

### Evaluation metric
Accuracy on the hidden test set (likely pass@1 over a Nemotron run with the submitted adapter). NVIDIA's parallel public materials describe **pass@1 (maj@64)** evaluation patterns; the exact protocol is on the leaderboard tab.

---

## 3. Submission format

A single **`submission.zip`** containing two files at the root:

```
submission.zip
├── adapter_config.json
└── adapter_model.safetensors
```

From [`ryanholbrook/nvidia-nemotron-submission-demo`](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo):

```python
LORA_RANK = 32  # max
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Note the regex includes Mamba's `in_proj`/`out_proj` — the model is a **hybrid Mamba-Transformer MoE**, not a pure transformer.

The reference `huikang/nvidia-nemotron-all-linear` notebook shows the simpler `target_modules="all-linear"` form, with `LORA_RANK = 32`, `lora_alpha = 16`, `lora_dropout = 0.05`.

### Kaggle inference container expectations
- Base model is loaded from `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` on the Kaggle Models hub.
- The loader applies `peft.PeftModel.from_pretrained(...)` to the unzipped adapter.
- The submission notebook runs on a Kaggle VM with `nvidiaRtxPro6000` accelerator and **`isInternetEnabled: false`**.
- Required Python packages on the runtime image (id `31287`): `transformers`, `peft`, `mamba_ssm`, `causal_conv1d`, `torch`, etc.

### Adapter key-name gotcha
For Nemotron-H, the model wraps the transformer in `model.backbone`. PEFT's default LoRA save names tensors `base_model.model.model.<...>` but the loader expects `base_model.model.backbone.<...>`. The winning notebook explicitly renames:

```python
key.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head.")
key.replace("base_model.model.model", "base_model.model.backbone")
```

---

## 4. The base model

[`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), mirrored on Kaggle as `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`.

- **Architecture**: hybrid Mamba-Transformer MoE — Mamba layers for fast sequence processing, transformer attention layers, and 128-expert MoE.
- **Total / active params**: 30 B / ~3 B active per token.
- **Pretraining**: 25T tokens (3T new vs Nemotron 2), code/math/reasoning-heavy mix; WSD LR schedule, peak 1e-3, min 1e-5.
- **Native context**: 1 M tokens.
- **Reasoning capability**: produces a `<think>` trace then answers; the chat template auto-injects the opening `<think>\n` after the user message.

Quirks that matter for fine-tuning:
- The MoE **router weights** are stored in fp32 (`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`); training must preserve that.
- 16-bit LoRA needs ≈ 60 GB VRAM (Unsloth docs). Comfortable on the RTX PRO 6000 (96 GB) but not on free Colab.
- Unsloth advice: **don't fine-tune the router layer**, and keep ≥ 75 % reasoning examples / ≤ 25 % non-reasoning in the dataset to preserve thinking behavior.

---

## 5. Allowed approaches (per NVIDIA)

> "Prompting, data filtering, synthetic data generation, reinforcement learning, and lightweight fine-tuning."

The hard constraint is the rank-32 LoRA delivery format — so any improvement must be encoded in those LoRA weights.

---

## 6. Top public approach: `tonghuikang` (Progress Prize winner, public LB ≈ 0.85)

Resources:
- Repo: <https://github.com/tonghuikang/nemotron>
- Notebook: [`huikang/end-to-end-finetuning-for-lb-0-85`](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)
- Writeup: [discussion/689915](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915) (paywalled to JS)
- Live dashboard: <https://nemotron.huikang.dev>

### Pipeline
1. **Run the baseline.** Dump per-problem trajectories (prompt, completion, extracted answer, correctness) — published as `kishanvavdara/nemotron-reasoning-traj`.
2. **Investigate.** For each unsolved problem, run a category-specific *programmatic* solver to discover the rule and produce a perfect rationale. The repo's `investigators/` contains:
   - `bit_manipulation.py` — brute-forces single transforms (identity, NOT, ROT(k), SHL/SHR(k), and NOT-of-each), then pairs combined with XOR/AND/OR, then triples — all 8-bit.
   - `cryptarithm_deduce.py`, `bit_manipulation_analysis.py`, etc.
3. **Augment.** Add synthetic problems via `augmenters/` (`spelling`, `concatenation`, `splitting`, `matching`, `lstrip`) for categories that are otherwise underrepresented.
4. **Build the corpus.** `corpus.py` tokenizes each `(prompt, reasoning, answer)` with the chat template, masking the prompt so loss is only computed on the rationale + boxed answer. Hard cap **`TOKEN_LIMIT = 8192`**.
5. **SFT the LoRA.** Train via Thinking Machines' Tinker, then re-train (and ship) inside the Kaggle notebook with Unsloth + Cut Cross-Entropy.

### Training hyperparameters (the published config)

```python
LORA_RANK       = 32
LORA_ALPHA      = 32        # alpha = rank
LORA_DROPOUT    = 0.0
MAX_SEQ_LEN     = 8192
NUM_STEPS       = 1000
BATCH_SIZE      = 32
MICRO_BATCH_SIZE= 4         # gradient accumulation
LEARNING_RATE   = 2e-4      # linear decay to 0
OPTIMIZER       = AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
TARGET_MODULES  = ["q_proj","k_proj","v_proj","o_proj",
                   "up_proj","down_proj",
                   "in_proj","out_proj",   # Mamba
                   "lm_head"]              # incl. lm_head
DTYPE           = bf16 base, fp32 LoRA, fp32 MoE router
RESET_WEIGHTS   = True      # train from fresh LoRA init
```

### Implementation tricks (most have measurable effect)

- **Unsloth `FastLanguageModel`** with `unsloth_force_compile=True`, `attn_implementation="eager"`, gradient checkpointing.
- **Cut Cross-Entropy** (`cut_cross_entropy.linear_cross_entropy`) for the LM head — avoids materializing full logits, big VRAM win.
- **Manual `lm_head` LoRA**: Unsloth drops the LM head adapter for MoE models; the notebook adds it back via `model.base_model._create_and_replace(...)` and includes it in saving.
- **Mamba CUDA fast path**: monkey-patch `modeling_nemotron_h.is_fast_path_available = True`.
- **MoE weight tying during training**: tie one LoRA factor across all 128 experts (A side for w1/`up_proj`/`gate_proj`/`gate_up_proj`, B side for w2/`down_proj`); sum gradients across experts each step. Saved adapter is **untied** (128 per-expert copies) so it works with the standard PEFT loader.
- **LoRA params kept in fp32** while base stays bf16; router stays fp32.
- **Token masking**: prompt tokens have weight 0; loss is computed only on `<think>...</think>\boxed{answer}<|im_end|>`.
- **Custom backward hooks** in `loss_config.py` allow advantage weighting / cutoff weighting (used by their later RL-flavored experiments).

### What the dashboard reveals
Per-category solve status is tracked in `problems.jsonl` as `rule_found` / `hypothesis_formed` / `rule_unknown` — the corpus only includes problems where a rule was found (`included: true`), so SFT trains the model on perfect rationales rather than noisy baseline outputs.

---

## 7. Other notable public artefacts

> **Note on enumerating the public leaderboard.** The competition page's
> `code?sortBy=scoreDescending` listing is JS-rendered, and the Kaggle API
> `kernels/list` endpoint returns `401 Unauthenticated` from this environment
> even though `kernels/pull` and `models/list` work. The user reports only
> ~3 entries at **0.87** and many at **0.86**; the highest-scoring public
> notebook I can fully verify here is huikang's at **LB ≈ 0.85**. The 0.86
> notebooks are likely lightly-tuned forks of either huikang's pipeline or
> kienngx's training notebook — the user can confirm by sorting the live
> Code page in their browser.

### Notebooks (code)

| Name | Author | What it gives you |
|---|---|---|
| [`huikang/end-to-end-finetuning-for-lb-0-85`](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85) | huikang | **Strongest public notebook (LB ≈ 0.85)**. Winning end-to-end pipeline — Unsloth + Cut Cross-Entropy, MoE weight tying, Mamba fast-path patch. Documented in §6. |
| [`huikang/tinker-submission-notebook`](https://www.kaggle.com/code/huikang/tinker-submission-notebook) | huikang | Companion adapter-packaging notebook. Loads a pre-trained adapter (`huikang/nemotron-adapter` v26) and rewrites the safetensors keys (`base_model.model.model` → `…backbone`, expert unfusing, gate_proj+x_proj → `in_proj` via SVD) before zipping. Useful template if you train via Tinker / Modal off-Kaggle. |
| [`huikang/nvidia-nemotron-all-linear`](https://www.kaggle.com/code/huikang/nvidia-nemotron-all-linear) | huikang | Reference adapter shape: rank 32, alpha 16, `target_modules="all-linear"`. Used as the canonical "what should the adapter_config look like" check in the tinker notebook. |
| [`kienngx/nvidia-nemotron-training-copy-run-instantly`](https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-copy-run-instantly) | kienngx | **Popular self-contained training starter.** TRL `SFTTrainer`, LoRA r=32 / α=32 / dropout 0.05 / `target_modules="all-linear"`, max_seq_len 2048, 2 epochs over a 1,200-row random subsample of `train.csv`, lr 1e-4 cosine, bs 1 × grad_accum 4. Patches Triton/PTXAS for Blackwell + replaces `rmsnorm_fn` with a pure-PyTorch fallback. **Caveat:** the `assistant_msg` is just the raw `answer` column, so it does not actually train on chain-of-thought — combining this notebook with the kienngx CoT dataset (below) is the obvious next step. |
| [`ryanholbrook/nvidia-nemotron-submission-demo`](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo) | Kaggle staff | Official minimal starter — model load, LoRA init, save, zip. |

### Datasets

| Name | Author | What it gives you |
|---|---|---|
| [`kienngx/nemotron-30b-competition-trainingdata-cot-labels`](https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels) | kienngx (1,147 dl, 46 votes) | 9,500 prompts + answer + Gemini-2.0-flash CoT + category label. Most-used SFT dataset on Kaggle. |
| [`kishanvavdara/nemotron-reasoning-traj`](https://www.kaggle.com/datasets/kishanvavdara/nemotron-reasoning-traj) | kishanvavdara (276 dl, 25 votes) | Baseline 30B trajectories with correctness aggregation — extracted from `tonghuikang/nemotron`. |
| `ritwikakancharla/nemotron-math-v2-filtered-high` | — | 3.7 GB filtered math reasoning corpus (Nemotron-Math v2 subset). |
| `mayukh18/nemotron-packages` (referenced by huikang) | mayukh18 | Pre-built `mamba_ssm` + `causal_conv1d` wheels for offline pip install. Critical because Kaggle's notebook has internet disabled at submission time. |
| `nvidia-nemotron-offline-packages/offline_packages` (referenced by kienngx) | — | Equivalent offline-pip directory used by the kienngx training notebook. |

### Models (Kaggle Models hub)

| Name | What it is |
|---|---|
| [`metric/nemotron-3-nano-30b-a3b-bf16`](https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16) | Official base model used by the scorer. |
| `huikang/nemotron-adapter` | Winning team's published LoRA adapter (multiple versions; the tinker notebook references **v26**, the end-to-end notebook references **v20**). |
| `kienngx/nemotron-nano-30b-trained` | "Nemotron-Nano-30B variances" — multiple parameter sweeps trained from the kienngx notebook above. |
| `ashok205/nemotron-30b-nvfp4` | NVFP4-quantized variant useful for inference. |
| `charancherrychowdary/nemotron-lora-adapter-v1` | Third-party LoRA adapter. |

---

## 8. Practical checklist for a new entry

1. Pull the base model: `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")`.
2. Build (or download) a training corpus of `(prompt, reasoning, answer)`.
3. Mask prompt tokens → only train on rationale + `\boxed{answer}<|im_end|>`.
4. Wrap with LoRA, **rank ≤ 32**, `target_modules` covering attention (`q/k/v/o_proj`), MLP (`up/down_proj`), **Mamba (`in_proj`, `out_proj`)**, and ideally `lm_head`. Use `lora_alpha = rank` (Tinker convention) or 16 (starter).
5. Keep LoRA params in fp32, base in bf16, MoE router in fp32. Don't touch the router itself.
6. Save the adapter, **rename `base_model.model.model` → `base_model.model.backbone`** in the safetensors keys, then `zip submission.zip adapter_config.json adapter_model.safetensors`.
7. Submit. The Kaggle scorer loads it, generates with thinking enabled, and grades the `\boxed{...}` content.

### Things to verify on the live Kaggle page before relying on them
- Final submission deadline (sources say either **June 8** or **June 15, 2026**).
- Exact evaluation metric / pass@k formulation.
- Daily submission limit and team-merge deadline.
- License of the released benchmark / training data.

---

## 9. Reference links

- Competition: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge>
- Official starter: <https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo>
- Winning repo: <https://github.com/tonghuikang/nemotron>
- Winning notebook: <https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85>
- Winning writeup: <https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915>
- Base model card: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>
- Nemotron 3 tech report: <https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf>
- Unsloth Nemotron-3 guide: <https://unsloth.ai/docs/models/nemotron-3>
- NVIDIA blog (Nemotron 3 architecture): <https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/>
