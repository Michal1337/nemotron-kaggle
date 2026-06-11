"""trainval-v1 STEP 4: build the dataset. Tokenizes UNIFORMLY (one chat-template
path) so real + synth share format:
  - real-train (9 cats): re-narrate via the category narrator; gold match is
    isclose(1%) for gravity/unit/numeral, exact otherwise; box the narrator's answer.
  - synth (6 cats): use the record's 'cot' if present (crypt), else narrate; box gold.
Drops are reported with pid+reason (the user requires no unexplained appear/disappear).
Enforces loss-tokens <= 7680. Writes tokens/, logprobs/index.jsonl, val_pids.txt.
"""
import json
import math
import os
import csv
import sys
import random
import multiprocessing as mp
from collections import Counter, defaultdict

sys.path.insert(0, "src")
import reasoners.cipher as CIP
# bit narrator switched to the zyx-with-merges production module (audit 2026-06-10):
# strict superset of the old port narrator (1454 vs 1364 of 1602, PORT-ONLY=0),
# answer-blind (decoy byte-identical 1602/1602). No INCLUDE_3INPUT flag exists.
# The gold_ok filter below is LOAD-BEARING: the narrator never abstains, so the
# filter is the only barrier keeping wrong-boxed CoTs out of training.
import reasoners.bit_manipulation_zyx as BIT
import reasoners.equation_numeric as EQ
import reasoners.gravity as GR
import reasoners.unit_conversion as UC
import reasoners.numeral as NU
import reasoners.cryptarithm as CR
from reasoners.store_types import Problem, Example
from narrators.investigations_to_reasoning import extract_final_answer

# Crypt narrator config (audit 2026-06-11):
#   "vfirst" (default) - deduce -> reasoners.crypt_vfirst (verification-first
#       P0-P8 narrator, 209/659 real coverage incl its own concat gate, honest);
#       guess -> the patched concat narrator (abstains on arithmetic; the
#       guess-fam table narrator stays env-gated until audited).
#   "anchor" - deduce -> reasoners.crypt_anchor (v4 anchor-DFS narrator,
#       160/659 real coverage incl its own concat gate, exact-tokenizer
#       governed); guess -> the patched concat narrator (same as vfirst).
#   "combo"  - legacy: charset->ops lookup deduce + family-table guess
#       (lookup is gold-conditioned per the audit; do not use for new corpora).
#   "concat" - legacy v1/v2 behavior (concat-only; arithmetic now abstains).
TV_CRYPT_NARR = os.environ.get("TV_CRYPT_NARR", "vfirst")
LC = CGF = VF = AN = None
if TV_CRYPT_NARR == "combo":
    import reasoners.crypt_lookup_combo as LC
    import reasoners.crypt_guess_fam as CGF
elif TV_CRYPT_NARR == "vfirst":
    import reasoners.crypt_vfirst as VF
elif TV_CRYPT_NARR == "anchor":
    os.environ.setdefault(
        "NEMOTRON_TOKENIZER_JSON",
        "/mnt/evafs/groups/re-com/mgromadzki/llms/nemotron-3-nano-30b-a3b-bf16/tokenizer.json")
    import reasoners.crypt_anchor as AN

G = "/mnt/evafs/groups/re-com/mgromadzki"
PB = G + "/nemotron-master/problems"
# Paths overridable via env so the SAME pipeline builds trainval-v2 (defaults = v1).
SPLIT = os.environ.get("TV_SPLIT", G + "/trainval_v1/split")
SYNTH = os.environ.get("TV_SYNTH", G + "/trainval_v1/synth")
OUT = os.environ.get("TV_OUT", G + "/nemotron-master/training/sft/trainval-v1")
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`")
CATS = ["cipher", "bit_manipulation", "equation_numeric_deduce", "equation_numeric_guess",
        "cryptarithm_deduce", "cryptarithm_guess", "gravity", "unit_conversion", "numeral"]
NUMERIC = {"gravity", "unit_conversion", "numeral"}
SYNTH_FILES = {"cipher": ["cipher"], "eq": ["equation_numeric_deduce", "equation_numeric_guess"],
               "crypt": ["cryptarithm_deduce", "cryptarithm_guess"], "bit": ["bit_manipulation"]}

trcsv = {}
with open(G + "/nemotron-master/train.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        trcsv[r["id"]] = (r["prompt"], str(r["answer"]))

_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(G + "/llms/nemotron-3-nano-30b-a3b-bf16", trust_remote_code=True)


def narrate(prob, cat):
    if cat == "cipher":
        return CIP.reasoning_cipher(prob)
    if cat == "bit_manipulation":
        return BIT.reasoning_bit_manipulation(prob)
    if cat.startswith("equation_numeric"):
        return EQ.reasoning_equation_numeric(prob, None, None)
    if cat == "gravity":
        return GR.reasoning_gravity(prob)
    if cat == "unit_conversion":
        return UC.reasoning_unit_conversion(prob)
    if cat == "numeral":
        return NU.reasoning_numeral(prob)
    if cat.startswith("cryptarithm"):
        if TV_CRYPT_NARR == "combo":
            return (LC.reasoning_cryptarithm_lookup(prob) if cat == "cryptarithm_deduce"
                    else CGF.reasoning_cryptarithm_guess_fam(prob))
        if TV_CRYPT_NARR == "vfirst" and cat == "cryptarithm_deduce":
            return VF.reasoning_cryptarithm_vfirst(prob)
        if TV_CRYPT_NARR == "anchor" and cat == "cryptarithm_deduce":
            return AN.reasoning_cryptarithm_anchor(prob)
        return CR.reasoning_cryptarithm(prob)
    return None


# v3: skip real-train narration for categories copied verbatim from another dataset
TV_SKIP_REAL_CATS = set(c for c in os.environ.get("TV_SKIP_REAL_CATS", "").split(",") if c)


def gold_ok(cat, gold, ra):
    if cat in NUMERIC:
        try:
            return math.isclose(float(gold), float(ra), rel_tol=1e-2, abs_tol=1e-5)
        except (ValueError, TypeError):
            return str(gold) == str(ra)
    return str(gold) == str(ra)


def tokenize(prompt, cot, ans):
    completion = f"{cot}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    rendered = _tok.apply_chat_template([{"role": "user", "content": prompt + PROMPT_SUFFIX}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    return pids + cids, [0] * len(pids) + [1] * len(cids), len(cids)


def work(task):
    kind, payload, cat = task
    try:
        if kind == "real":
            pid = payload
            d = json.load(open(f"{PB}/{pid}.jsonl"))
            gold = trcsv.get(pid, (None, str(d.get("answer"))))[1]
            prompt = trcsv[pid][0]
            prob = Problem(id=pid, category=cat,
                           examples=[Example(e["input_value"], e["output_value"]) for e in d["examples"]],
                           question=d["question"], answer=gold)
            cot = narrate(prob, cat)
            out_pid = pid
        else:  # synth
            rec = payload
            gold = str(rec["answer"])
            prompt = rec["prompt"]
            cot = rec.get("cot")
            if cot is None:
                prob = Problem(id=rec["id"], category=cat,
                               examples=[Example(e["input_value"], e["output_value"]) for e in rec["examples"]],
                               question=rec["question"], answer=gold)
                cot = narrate(prob, cat)
            out_pid = "syn_" + rec["id"]
        if cot is None:
            return (out_pid, cat, kind, None, None, "narrate_none")
        ra = extract_final_answer(cot)
        if ra == "NOT_FOUND":
            ra = gold
        if not gold_ok(cat, gold, ra):
            return (out_pid, cat, kind, None, None, "wrong_box")
        toks, mask, nloss = tokenize(prompt, cot, ra)
        if nloss > 7680:
            return (out_pid, cat, kind, None, None, "over_7680")
        return (out_pid, cat, kind, toks, mask, "ok")
    except Exception as e:
        return (payload if kind == "synth" else payload, cat, kind, None, None, "err:" + type(e).__name__)


def main():
    import shutil
    # wipe prior build outputs so a rebuild can't leave orphan token dirs (index must == dirs)
    if os.path.isdir(OUT + "/tokens"):
        shutil.rmtree(OUT + "/tokens")
    os.makedirs(OUT + "/tokens", exist_ok=True)
    os.makedirs(OUT + "/logprobs", exist_ok=True)
    tasks = []
    expect = defaultdict(lambda: {"real": 0, "synth": 0})
    # real-train tasks
    for cat in CATS:
        if cat in TV_SKIP_REAL_CATS:
            continue
        pids = [p.strip() for p in open(f"{SPLIT}/{cat}.train.txt") if p.strip()]
        for pid in pids:
            tasks.append(("real", pid, cat))
        expect[cat]["real"] = len(pids)
    # synth tasks (cap per-category to the exact target; eq_guess generated 858
    # for per-rule firing -> cap to 850)
    SYNTH_CAP = {"cipher": 750, "equation_numeric_deduce": 1704, "equation_numeric_guess": 850,
                 "cryptarithm_deduce": 540, "cryptarithm_guess": 110, "bit_manipulation": 1500}
    if os.environ.get("TV_SYNTH_CAP_JSON"):   # v2 overrides (e.g. bigger bit/crypt caps)
        SYNTH_CAP.update(json.loads(os.environ["TV_SYNTH_CAP_JSON"]))
    syn_count = Counter()
    for fname, fcats in SYNTH_FILES.items():
        fp = f"{SYNTH}/{fname}.jsonl"
        if not os.path.exists(fp):
            continue
        for l in open(fp):
            rec = json.loads(l)
            cat = rec["category"]
            if cat in SYNTH_CAP and syn_count[cat] >= SYNTH_CAP[cat]:
                continue
            syn_count[cat] += 1
            tasks.append(("synth", rec, cat))
            expect[cat]["synth"] += 1

    recs = []
    drops = []
    got = defaultdict(lambda: {"real": 0, "synth": 0})
    seen_pids = set()
    maxloss = 0
    with mp.Pool(16, initializer=_init) as pool:
        for out_pid, cat, kind, toks, mask, status in pool.imap_unordered(work, tasks, chunksize=8):
            if status != "ok":
                drops.append((str(out_pid)[:40], cat, kind, status))
                continue
            if out_pid in seen_pids:
                drops.append((out_pid, cat, kind, "pid_collision"))
                continue
            seen_pids.add(out_pid)
            od = OUT + "/tokens/" + out_pid
            os.makedirs(od, exist_ok=True)
            json.dump({"tokens": toks, "mask": mask}, open(od + "/synthetic.json", "w"))
            recs.append({"epoch": 0, "step": -1, "problem_id": out_pid, "segment": "synthetic.jsonl",
                         "category": cat, "num_loss_tokens": sum(mask), "total_loss": 0.0, "min_logprob": 0.0})
            got[cat][kind] += 1
            maxloss = max(maxloss, sum(mask))

    # stratified shuffle (mix categories across batches)
    rng = random.Random(0)
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["category"]].append(i)
    for v in by.values():
        rng.shuffle(v)
    nb = max(1, (len(recs) + 31) // 32)
    batches = [[] for _ in range(nb)]
    a = 0
    for cat in sorted(by):
        for idx in by[cat]:
            batches[a % nb].append(idx)
            a += 1
    order = [recs[i] for b in batches for i in b]
    with open(OUT + "/logprobs/index.jsonl", "w") as f:
        for r in order:
            f.write(json.dumps(r) + "\n")

    # val_pids (real-val per cat with category)
    val = []
    for cat in CATS:
        for p in (x.strip() for x in open(f"{SPLIT}/{cat}.val.txt") if x.strip()):
            val.append((p, cat))
    with open(OUT + "/val_pids.txt", "w") as f:
        for p, cat in val:
            f.write(f"{p}\t{cat}\n")

    # report
    print("=== STEP 4 BUILD: per-category counts (got real+synth / expected) ===")
    allok = True
    for cat in CATS:
        gr, gs = got[cat]["real"], got[cat]["synth"]
        er, es = expect[cat]["real"], expect[cat]["synth"]
        ok = (gr == er and gs == es)
        allok &= ok
        print(f"  {cat:26s} real {gr}/{er}  synth {gs}/{es}  train_total={gr+gs}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n  TRAIN total={len(recs)}  VAL total={len(val)}  max_loss_tokens={maxloss}")
    print(f"  drops: {len(drops)}")
    dc = Counter((c, k, s) for _, c, k, s in drops)
    for (c, k, s), n in dc.most_common(20):
        print(f"     {c} {k} {s}: {n}")
    print(f"  ALL COUNTS OK: {allok}  MAX<=7680: {maxloss <= 7680}")
    print("TV_STEP4_DONE")


if __name__ == "__main__":
    main()
