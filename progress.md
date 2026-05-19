# Per-category progress dashboard

Snapshot of what's solved, what's left, and what we're working on per category. Update the "Last action / next action" rows as we go.

## Global

- **Current LB floor**: 0.86 (from our v20→PEFT conversion via tinker-cookbook)
- **Our best train-from-scratch**: 0.84 (huikang corpus, 244 steps SFT)
- **Public ceiling we know about**: 0.87 (4 teams, methods not public)
- **In-sample eval accuracy on training corpus**: 97.5% (mostly fine; bit_manip drops to 86%)
- **OOD eval accuracy on held-out**: 47.1% — driven by 0% on the 4 hard categories

## Categories

### 🟢 SATURATED — no headroom from corpus extension

| Category | LB share | rule_found in corpus | OOD acc | Status |
|---|---:|---:|---:|---|
| **numeral** | 16.6% | 650 (need 1,576) | **100%** | Saturated. huikang downsampled but model generalizes perfectly to the 926 excluded ones. |
| **gravity** | 16.8% | 975 (need 1,597) | **99.5%** | Saturated. Same story. |
| **unit_conversion** | 16.8% | 990 (need 1,594) | **99%** | Saturated. |
| **cipher** | 16.6% | 1,576 (all) | n/a (no held-out) | In-sample 99% — assume saturated. 77-word `wonderland.txt` vocabulary is the full answer space. |

**Total LB locked**: ~67% of test set scoring at 99-100% (assuming test distribution mirrors train.csv).

- **Last action**: OOD eval on all 4 confirmed >99% accuracy.
- **Next action**: none — don't spend time here.

---

### 🟡 BIT_MANIPULATION — has headroom, needs reasoner extension

| Field | Value |
|---|---|
| LB share | 16.9% (largest single category) |
| Total in train.csv | 1,602 |
| rule_found in 04-08 corpus | 1,354 |
| hypothesis_formed (correct inv file, not in corpus) | 121 |
| rule_unknown (no solver works) | 117 |
| In-sample eval accuracy | 86% (low — only category not >98%) |
| OOD eval accuracy on held-out | **12.5%** ← the problem |

**Diagnosis** — investigated 2026-05-19:
- Initial hypothesis: 8,192-token truncation of long bit_manip rationales. **Rejected** — avg `gen_tokens` 6,674 (under 7,680 limit), only 1.1% empty predictions, failures are 1-4 bits off the gold (not garbage).
- Real diagnosis: rules outside huikang's reasoner search space. The 248 missing problems need 3-input gates (MAJ, CHO, PAR3, AO/OA/AX/OX/XA/XO) and 4-input gates (AOA, OAO, PAR4, XX, AXA) per the forum post. huikang's reasoner only tries XOR/AND/OR over 1-2-3 elements.

**Available levers**:
1. Extend [`reasoners/bit_manipulation.py`](nemotron-master/reasoners/bit_manipulation.py) to enumerate MAJ/CHO/PAR3/4-input gates. Could unlock 80-150 of the 248 missing.
2. Convert the 121 existing-correct-investigation bit_manip files into reasoning files. Format is terse — needs a narrator step.

- **Last action**: confirmed truncation hypothesis is wrong; logged exact failure mode (model produces close-but-not-equal binary strings).
- **Next action**: deferred. Coming back after the cryptarithm work resolves.

---

### 🟢 CRYPTARITHM_DEDUCE — v3 unlocked 130 of 559

| Field | Value (pre-v3) | Value (post-v3) |
|---|---|---|
| LB share | 6.9% | 6.9% |
| Total in train.csv | 659 | 659 |
| rule_found in 04-08 corpus | 54 | 54 |
| hypothesis_formed (correct inv file, not in corpus) | 38 | 38 + **130 new** = 168 |
| rule_unknown | 559 | **429** |
| Solve rate | 14% | **40%** |

**v3 run results (2026-05-19)**:
- Newly solved: **130 of 559 = 23.3% unlock rate**
- Wrong answer: ~200 (estimated, false positives from broader search)
- No answer: ~200 (ops outside the 32-op pool)
- Timeout: a few %
- All 72 transform-using solutions used `rev_ops=True, rev_res=True` exclusively — clean signal that cryptarithm's "reversal twist" is always full-reversal, never partial.

**Operators that unlocked problems**: multiplication (88), addition (81), subtraction (53), multiply+1/-1 (45), max_mod_min (23), add+1/-1 (43), rev_subtraction (17), absolute_diff (11), modulo (3), digit_add_mod10 (1).

- **Last action**: ran cryptarithm_v3.py on 687 rule_unknown, 142 newly solved across both subcategories
- **Next action**: write narrator to convert the 130 + 38 + 12 = 180 terse cryptarithm investigations into reasoning files

---

### 🟢 CRYPTARITHM_GUESS — v3 unlocked 12 of 128

| Field | Value (pre-v3) | Value (post-v3) |
|---|---|---|
| LB share | 1.7% | 1.7% |
| Total in train.csv | 164 | 164 |
| rule_found in 04-08 corpus | 11 | 11 |
| hypothesis_formed (correct inv file, not in corpus) | 17 | 17 + **12 new** = 29 |
| rule_unknown | 128 | **116** |
| Solve rate | 17% | **24%** |

**v3 run results**: 12 of 128 newly solved (9.4% — lower than deduce because guess requires inferring the query op when it's unseen in examples).

- **Last action**: v3 run included guess via `solve_problem_v3()`
- **Next action**: convert 17 + 12 = 29 cryptarithm_guess correct investigations (17 narrated + 12 terse from v3) to reasoning files

---

### 🟡 EQUATION_NUMERIC_GUESS — small pool, hard

| Field | Value |
|---|---|
| LB share | 1.4% |
| Total in train.csv | 136 |
| rule_found in 04-08 corpus | 21 |
| hypothesis_formed | 35 (from LLM, not from any .py investigator) |
| rule_unknown | 80 |
| In-sample eval | 100% |
| OOD eval | **0%** |

**Diagnosis**: `_guess` variant — query op is unseen in examples. The existing 30-op equation_numeric reasoner already handles 561/732 of equation_numeric_deduce (90.6%) but only 21/136 of guess (15%). The guess variants need inferential extrapolation that the reasoner doesn't do.

**Available levers**:
1. Convert 35 LLM-narrated investigations to reasoning files. Format-compatible (already narrated).
2. Add elimination-style inference to the equation_numeric reasoner (when query op is unseen, try each op and see which produces a result with no symbol contradictions).

- **Last action**: classified investigations (40 correct, 8 incorrect)
- **Next action**: low priority; small absolute pool. Visit after cryptarithm work pays off.

---

### 🟡 EQUATION_NUMERIC_DEDUCE — almost saturated, marginal upside

| Field | Value |
|---|---|
| LB share | 6.3% |
| Total in train.csv | 596 |
| rule_found in 04-08 corpus | 540 |
| hypothesis_formed | 22 |
| rule_unknown | 34 |
| In-sample eval | 98% |
| OOD eval (rule_found subset) | **96.4%** |

**Diagnosis**: Reasoner already covers 90.6%. The remaining 56 problems use ops outside the existing 30-op enumeration. Hard to predict which ops without inspecting each one.

**Available levers**:
1. Convert 22 LLM-narrated investigations to reasoning files (small win).
2. Inspect 34 rule_unknown problems for common patterns and extend reasoner.

- **Last action**: noted in OOD eval that the rule_found subset transfers well (3.6% looks bad but pool is just the 56 missing — 2 correct out of 56)
- **Next action**: low priority. Visit after cryptarithm + bit_manipulation.

---

## Conversion candidates summary (across all categories)

If we wrote a converter to turn `investigations/<category>/correct/` files into reasoning files for problems not in corpus:

| Category | Convertible (correct inv + not in corpus) | Format |
|---|---:|---|
| bit_manipulation | 128 | terse, needs narrator |
| cryptarithm_deduce | 38 | terse, needs narrator |
| cryptarithm_guess | 17 | narrated, direct convert |
| equation_numeric_deduce | 12 | narrated, direct convert |
| equation_numeric_guess | 28 | narrated, direct convert |
| **Total** | **223** | mixed |

Direct conversion (no narrator needed): **57 problems**. With narrator for bit_manip + cryptarithm_deduce: **+166** (total 223).

## Priority order

| Priority | Track | Expected yield | Effort |
|---|---|---:|---|
| 1 | Run cryptarithm_v3 + convert resulting investigations to reasoning | +50 to +300 | hours (run) + days (converter) |
| 2 | Extend bit_manipulation reasoner with 3-input/4-input gates | +50 to +200 | days |
| 3 | Direct conversion of 57 narrated investigations (cryptarithm_guess + equation_numeric_*) | +57 | hours |
| 4 | Equation_numeric_guess inference extension | +30 to +80 | days |

## Action log (append-only)

- **2026-05-19** — found cryptarithm reframe in forum post; abandoned v2 generic ops; wrote v3 with equation_numeric op pool
- **2026-05-19** — reorganized investigations/ into per-category + correct/incorrect subdirs
- **2026-05-19** — confirmed bit_manipulation OOD failure is rule-discovery, not truncation
- **2026-05-19** — measured OOD eval: 99-100% on easy 4 categories, 0% on 4 hard ones, 12.5% on bit_manip
- **2026-05-19** — measured in-sample eval: 97.5% global, 86% on bit_manipulation
- **2026-05-19** — **ran cryptarithm_v3 on 687 rule_unknown: 142 newly solved (130 deduce + 12 guess), 33min on 16 cores. Cryptarithm coverage 14.1% → 31.4%.**
- **2026-05-20** — discovered `lkevincc0/kaggle-nemotron-equation-symbolic` solver achieving 97.2% on cryptarithm (gold-conditioned). Their `solver_results.parquet` ships verified-correct mapping/ops/mode for 800 of 823 problems. Wrote narrator that uses the parquet directly + proper scorer extraction (handles `}` in gold). **800 / 800 narratable, ready to add ~735 new reasoning files to the corpus.**
