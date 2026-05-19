# Findings — cryptarithm corpus extension

Running log of what we've established about cryptarithm. Append, don't rewrite.

## Where we are right now

We've isolated the cryptarithm category as the focus of corpus-extension work. Goal: increase the number of `rule_found` cryptarithm problems in the training corpus so the model sees more verified-correct rationales for this category.

Baseline LB: **0.86** (from converting huikang's v20 raw adapter).
Our pipeline retrains from scratch on huikang's `04-08-16-14` corpus at **0.84**.
Public ceiling: **0.87** (4 teams; method not public).

## Cryptarithm — current inventory

### Total problems in train.csv
| Subcategory | Count | LB weight |
|---|---:|---:|
| cryptarithm_deduce | 659 | 6.9% |
| cryptarithm_guess | 164 | 1.7% |
| **Total** | **823** | **8.6%** |

### Status breakdown (from `problems.jsonl`)
| | rule_found | hypothesis_formed | rule_unknown | Total |
|---|---:|---:|---:|---:|
| cryptarithm_deduce | 54 | 46 | 559 | 659 |
| cryptarithm_guess | 11 | 25 | 128 | 164 |
| **Total** | **65** | **71** | **687** | **823** |

- `rule_found` = the runtime reasoner ([reasoners/cryptarithm.py](nemotron-master/reasoners/cryptarithm.py)) solved it correctly → in `reasoning/{pid}.txt` → in 04-08-16-14 corpus
- `hypothesis_formed` = the investigator solved it (file in `investigations/{pid}.txt`) but the reasoner couldn't replicate → **NOT in 04-08-16-14 corpus**
- `rule_unknown` = neither solved it → no usable rationale exists today

### What's in `investigations/` for cryptarithm
- **129 investigation files total** (100 deduce + 29 guess)
- All 100 deduce investigations: 54 rule_found + 46 hypothesis_formed
- All 29 guess investigations: 4 rule_found + 25 hypothesis_formed
- Cross-checked against gold via `compare_answer`: **109 investigations predict the correct answer** (88 deduce + 21 guess). The other 20 have wrong predictions that the investigator wrote anyway (bug or stale code path).

### Correct-investigation problems NOT in 04-08-16-14 corpus
This is the **immediate conversion pool** — problems where the investigator nailed the answer and we just need to write a `reasoning/{pid}.txt` to flip status to `rule_found`.

| Subcategory | Correct AND not-in-corpus | Notes |
|---|---:|---|
| cryptarithm_deduce | **38** | format is **terse** — needs a narrator |
| cryptarithm_guess | **17** | format is **narrated** — direct conversion works |
| **Total** | **55** | |

### Investigation format — two distinct styles

**TERSE** (cryptarithm_deduce most files, e.g. [dc240ebd.txt](nemotron-master/investigations/dc240ebd.txt)):
```
problem id: dc240ebd
category: cryptarithm_deduce

symbol-to-digit mapping:
  '"' = 6
  ')' = 7
  ...

operator-to-operation mapping:
  '-' = mul

examples:
  >@-`` = ::``  =>  51 * 44 = 2244
  ...

query: >|-}|  =>  53 * 93 = 4929
predicted answer: `}:}
```

Machine-readable, no prose. Needs a narrator step to become CoT-style training data.

**NARRATED** (cryptarithm_guess most files, e.g. `25ee72c3.txt`):
```
examples:
1. ...

inferred rule:
Treat each non-operator symbol as a decimal digit...

The examples pin down this partial digit map:
- `>` = 0
- `&` = 1
...

step-by-step check on the examples:
1. `:>+'>`
   `:>` is 80 and `'>` is 30.
   80 + 30 = 110.
   ...

step-by-step application to the query:
1. The left chunk is `#"` and the right chunk is `<<`.
2. Apply `*` as concatenation...

predicted answer:
`#"<<`

confidence note:
Medium. ...
```

Already CoT-shaped. Just needs final formatting (`</think>\n\boxed{ANSWER}<|im_end|>` suffix).

## The bigger pool (`rule_unknown`) is closed

Re-ran [investigators/cryptarithm_deduce.py](nemotron-master/investigators/cryptarithm_deduce.py) `solve_problem` on every cryptarithm with `status = rule_unknown`:
- **cryptarithm_deduce**: 0 / 559 correct (97 wrong answers, 462 returned None). Investigator is saturated.
- **cryptarithm_guess**: ~3 / 128 correct (partial run, 2/100 confirmed). Investigator was never run on this subcategory via `main()`, but extending it doesn't unlock much.

**687 cryptarithm problems are beyond what the existing investigator can solve.** To unlock more we'd need new operator families or a fundamentally different solver — multi-day work for unclear yield.

## What we can actually do

| Option | Effort | Problems added to corpus | Notes |
|---|---|---:|---|
| **A** — Convert narrated guess investigations only | small (~2h) | **17** | safest, validates the conversion pipeline |
| **B** — A + narrator for terse cryptarithm_deduce | medium (~1d) | **55** | full hypothesis_formed pool |
| C — A + B + re-run investigator on guess rule_unknown | medium (~1d) | ~58 | marginal +3 from guess re-run |
| ~~D — extend investigator with new operator families~~ | ~~large~~ | ~~unknown, probably small~~ | ~~payoff per op family is small; deferred~~ |

## Decision pending

Pick **A** (17 problems, fast, validates pipeline) or **B** (55 problems, ~1 day, fuller corpus extension).

## Action log

- 2026-05-19 — confirmed 129 cryptarithm investigations exist, 109 with correct predicted answers, 55 not in corpus
- 2026-05-19 — re-ran investigator on 559 cryptarithm_deduce rule_unknown: 0 new correct
- 2026-05-19 — re-ran investigator on cryptarithm_guess rule_unknown (partial): 2/100 newly correct
- 2026-05-19 — confirmed two distinct investigation formats: terse (deduce) vs narrated (guess)
- 2026-05-19 — investigations regrouped: `investigations/<category>/<correct|incorrect>/`
- 2026-05-19 — wrote `investigators/cryptarithm_v2.py` with 30 generic operators + multiprocessing — **superseded** by the methodology below
- 2026-05-19 — **MAJOR REFRAME** from forum post: cryptarithm = equation_numeric + bijective cipher

## Reframe: cryptarithm IS encrypted equation_numeric

A Kaggle discussion post (~200h of reverse engineering, 100% claim across categories) revealed the actual structure:

- The 5-char cryptarithm input `AB op CD` is literally a symbol-digit problem `AB ⊕ CD`, encrypted with a **fresh bijective cipher per problem** that maps each digit and each operator-character to a random symbol
- The operator is always at index 2 of the input
- The underlying problem space is **4 pairings × 14 operations × 14 formats** ≈ 784 combos, of which ~47 cover 99% of the competition distribution
  - 4 pairings: `AB_CD`, `CD_AB`, `BA_DC`, `DC_BA` (operand order + reversal)
  - ~14 ops include: `add`, `mul`, `sub`, `cat`, `add1`, `addm1`, `mulsub1`, `muladd1`, etc. (same op pool as equation_numeric)
  - ~14 formats applied to the raw result: `raw`, `rev` (digit reverse), `abs`, `dsum` (digit sum), `zpad2`, `<op-prefixed>` variants, etc.

The author's pipeline:
1. **DETECT** — operator is `input[2]`
2. **CRACK** — build symbol→digit mapping from examples (Sudoku-style constraint solving — same as v1 investigator's `Solver` class)
3. **SCAN** — frequency-weighted brute force over the ~47 most-common (pairing, op, format) combos, decoded into bare digits
4. **LOCK** — when a combo fits BOTH EX1 and EX2, commit (single-example match is rejected to avoid coincidences)
5. **APPLY** — decode target, run the locked combo
6. **ENCODE** — re-encrypt the digit answer back into cipher symbols using the inverse mapping

This explains every observation:
- The existing v1 `Solver` already does the CRACK step well — that's why ~136 problems are solved (the easy concat/add/abs_diff/mul cases)
- The 687 rule_unknown problems use ops in the broader 14-op pool that v1 doesn't enumerate (e.g. `add1`, `mulsub1`, formats other than raw)
- Generic operator extensions (xor, gcd, etc.) in our cryptarithm_v2.py probably won't help because the actual operator distribution is the equation_numeric distribution, not arbitrary

## Implications and revised plan

**Strategic pivot**: stop extending `cryptarithm_v2.py` with generic ops. Instead:

1. **Reuse the equation_numeric reasoner** (`reasoners/equation_numeric.py`, 604 lines, ~30 ops enumerated, 90% solve rate on equation_numeric_deduce). It already knows the right op pool.
2. **New cryptarithm pipeline**: CRACK (using v1's `Solver` for symbol→digit) → decode examples + query into bare-digit equation_numeric form → call equation_numeric reasoner → re-encode answer with the inverse mapping
3. **Verify against BOTH examples**, not just one (the post's key false-positive-rejection rule)

Expected: most of the 687 rule_unknown cryptarithm problems should become solvable because they're already solvable as equation_numeric — we've just been searching the wrong operator pool.

## Other useful tidbits from the post (for later)

| Category | The actionable nugget |
|---|---|
| bit_manipulation | 8 independent 1-bit problems. Search: const → identity → NOT → 2-input gates → 3-input (MAJ, CHO, PAR3, AO/OA/etc.) → 4-input (AOA, OAO, PAR4, XX, AXA). Verify on target, not just examples. The reasoner's current 1-2-3 element XOR/AND/OR misses 3-input MAJ/CHO and all 4-input gates → matches our finding that 248 bit_manip problems are unsolved. |
| cipher | Bijective derangement (no fixed points). Fixed ~90-word vocabulary in predictable templates. VOCAB fill handles incomplete tables. |
| gravity / unit_conversion | Rate-first decomposition + cross-rate consistency check (`|rate1 - rate2| < tol`). Same approach already in `reasoners/gravity.py` and `unit_conversion.py`. |
| numeral (Roman) | Bidirectional. Incremental concatenation + round-trip verification. |
| equation_numeric | Same 4×14×14 search as cryptarithm minus the cipher layer. 47 high-frequency combos cover 99%. |

The post also has very specific GRPO reward shaping for an RL training run — relevant if/when we move to RL after SFT plateaus. Per-step partial credit, contamination markers, thrash markers, "champagne" bonus for full correctness. Not relevant for our SFT corpus-extension work, but worth keeping for later.
