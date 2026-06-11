"""Synthetic cryptarithm_deduce generator for the v4 anchor narrator
(crypt_anchor.py, commit d2db9ca).

Deterministic (hash-seeded per index, no clock/global RNG). Every candidate
problem is rendered by the REAL narrator and kept only if the narrator's boxed
answer equals the generated gold (the same gold filter training uses), then
bucketed by narration shape. Default quotas (total 2200):

  std     1150  solved in the standard-mode candidate list (no swap section)
  swap     600  standard visibly fails -> swap section solves
  easy     300  any arithmetic tail with real tok < 1200 (short procedure)
  concat   150  concat-gate problems

v4-specific generation constraints:
  - every problem carries a guaranteed ANCHOR fact: a mul-family op char with
    a 4-digit product (rl=4), so the narrator anchors the same way it does on
    real pids;
  - anchor operands are repeat-biased half the time (drawn from a 2-3 glyph
    subset) - repeat-heavy patterns keep the candidate list under LIST_CAP,
    which is the design wall on real data;
  - mode 50:50 by index parity (procedure, not prior);
  - op families include mod_max (the narrator's extended rl 1-2 pool);
  - >=50% of problems contain an add/mul +-1 variant (the measured op-vs-+-1
    disambiguation skill).

Output: jsonl rows {id, category, prompt, answer, examples, question, _meta}
compatible with tv_step4_build synth ingestion.
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, "src")
import reasoners.crypt_vfirst_solver as S  # noqa: E402
from reasoners.crypt_anchor import (  # noqa: E402
    reasoning_cryptarithm_anchor, op_val, tok_count)
from reasoners.store_types import Problem, Example  # noqa: E402

GLYPHS = sorted(set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") - set("=,_;~"))
PROMPT = ("In Alice's Wonderland, a secret set of transformation rules is applied "
          "to equations. Below are a few examples:\n{exs}\nNow, determine the result for: {q}")

FAMS = [["add", "add_p1", "add_m1"], ["mul", "mul_p1", "mul_m1"],
        ["sub_signed"], ["rsub_signed"], ["absdiff"], ["neg_absdiff"], ["mod_max"]]
MUL_FAM = ["mul", "mul_p1", "mul_m1"]
SIGNED = {"sub_signed", "rsub_signed", "neg_absdiff"}

N_EX_W = [(3, 180), (4, 241), (5, 238)]
N_OPC_W = [(1, 13), (2, 272), (3, 374)]

QUOTAS = {"std": 1150, "swap": 600, "easy": 300, "concat": 150}


def wchoice(rng, pairs):
    tot = sum(w for _, w in pairs)
    x = rng.randrange(tot)
    for v, w in pairs:
        x -= w
        if x < 0:
            return v
    return pairs[-1][0]


def synth_op_value(op, a, b):
    if op == "mod_max":
        return None if min(a, b) == 0 else max(a, b) % min(a, b)
    return S.op_value(op, a, b)


def encode_output(v, op, oc, mode, enc):
    """Output string for value v under the solver's exact semantics."""
    sign = op in SIGNED and v < 0
    s = str(abs(v))
    if mode == "rev_both":
        s = s[::-1]
    body = "".join(enc[int(c)] for c in s)
    return (oc + body) if sign else body


def gen_candidate(i):
    rng = random.Random(int(hashlib.md5(f"cryptsyna-{i}".encode()).hexdigest(), 16))
    mode = "standard" if i % 2 == 0 else "rev_both"
    n_opc = wchoice(rng, N_OPC_W)
    n_ex = max(wchoice(rng, N_EX_W), n_opc)
    glyphs = rng.sample(GLYPHS, 10 + n_opc)
    dig_glyphs, op_glyphs = glyphs[:10], glyphs[10:]
    mapping = {g: d for d, g in enumerate(dig_glyphs)}
    enc = {d: g for g, d in mapping.items()}

    concat_slice = (i % 14 == 13)  # ~7% of candidates carry a concat op
    ops = {}
    want_pm1 = rng.random() < 0.5
    # op char 0 is always the anchor: a mul-family op
    ops[op_glyphs[0]] = rng.choice(MUL_FAM[1:]) if (want_pm1 and rng.random() < 0.5) \
        else rng.choice(MUL_FAM)
    other_fams = [f for f in FAMS if f != MUL_FAM]
    fams = rng.sample(other_fams, max(0, n_opc - 1))
    for j, oc in enumerate(op_glyphs[1:]):
        if concat_slice and j == n_opc - 2:
            ops[oc] = rng.choice(["concat_fwd", "concat_rev"])
        elif want_pm1 and len(fams[j]) == 3:
            ops[oc] = rng.choice(fams[j][1:])
        else:
            ops[oc] = rng.choice(fams[j])

    def make_example(oc, force_anchor=False):
        op = ops[oc]
        for _ in range(80):
            if force_anchor and rng.random() < 0.5:
                # repeat-bias: operand glyphs from a small subset
                pool = rng.sample(dig_glyphs, rng.choice([2, 3]))
                a1, a2, b1, b2 = (rng.choice(pool) for _ in range(4))
            else:
                a1, a2, b1, b2 = (rng.choice(dig_glyphs) for _ in range(4))
            inp = a1 + a2 + oc + b1 + b2
            if inp in seen_inputs:
                continue
            L = mapping[a1] * 10 + mapping[a2]
            R = mapping[b1] * 10 + mapping[b2]
            if mode == "rev_both":
                L, R = S.rev2(L), S.rev2(R)
            if op in S.CONCAT:
                out = "".join(inp[k] for k in S.CONCAT[op])
            else:
                v = synth_op_value(op, L, R)
                if v is None or (v < 0 and op not in SIGNED):
                    continue
                if force_anchor and v < 1000:
                    continue  # the anchor fact must be rl=4
                if op == "neg_absdiff" and v == 0 and (a1 != b1 or a2 != b2):
                    continue
                out = encode_output(v, op, oc, mode, enc)
            seen_inputs.add(inp)
            return (inp, out)
        return None

    seen_inputs = set()
    exs = []
    anchor_ex = make_example(op_glyphs[0], force_anchor=True)
    if anchor_ex is None:
        return None
    exs.append(anchor_ex)
    order = list(op_glyphs[1:]) + [rng.choice(op_glyphs) for _ in range(n_ex - n_opc)]
    rng.shuffle(order)
    for oc in order:
        ex = make_example(oc)
        if ex is None:
            return None
        exs.append(ex)
    rng.shuffle(exs)

    # query: demonstrated op char, fresh operands, encodable non-empty answer
    for _ in range(40):
        oc = rng.choice(op_glyphs)
        op = ops[oc]
        a1, a2, b1, b2 = (rng.choice(dig_glyphs) for _ in range(4))
        q = a1 + a2 + oc + b1 + b2
        if q in seen_inputs:
            continue
        L = mapping[a1] * 10 + mapping[a2]
        R = mapping[b1] * 10 + mapping[b2]
        if mode == "rev_both":
            L, R = S.rev2(L), S.rev2(R)
        if op in S.CONCAT:
            gold = "".join(q[k] for k in S.CONCAT[op])
        else:
            v = synth_op_value(op, L, R)
            if v is None or (v < 0 and op not in SIGNED):
                continue
            gold = encode_output(v, op, oc, mode, enc)
        if not gold:
            continue
        examples = [{"input_value": i_, "output_value": o_} for i_, o_ in exs]
        prompt = PROMPT.format(exs="\n".join(f"{i_} = {o_}" for i_, o_ in exs), q=q)
        pid = "syna_" + hashlib.md5((str(exs) + q).encode()).hexdigest()[:10]
        return {"id": pid, "category": "cryptarithm_deduce", "prompt": prompt,
                "answer": gold, "examples": examples, "question": q,
                "_meta": {"mode": mode, "ops": ops, "n_ex": n_ex}}
    return None


def classify(prob):
    """Render with the real narrator; return (tail, tok, cot) or None."""
    p = Problem(id=prob["id"], category="cryptarithm_deduce",
                examples=[Example(e["input_value"], e["output_value"]) for e in prob["examples"]],
                question=prob["question"], answer=prob["answer"])
    cot = reasoning_cryptarithm_anchor(p)
    if cot is None:
        return None
    i = cot.rfind("boxed{")
    j = cot.rfind("}")
    if not (0 <= i < j) or cot[i + 6:j] != prob["answer"]:
        return None
    tok = tok_count(cot)
    if "concat check" in cot:
        tail = "concat"
    elif "try swap" in cot:
        tail = "swap"
    else:
        tail = "std"
    return tail, tok, cot


def bucket_of(tail, tok):
    if tail == "concat":
        return "concat"
    if tok < 1200:
        return "easy"
    return tail


def worker(i):
    prob = gen_candidate(i)
    if prob is None:
        return None
    r = classify(prob)
    if r is None:
        return None
    tail, tok, cot = r
    prob["_meta"]["tail"] = tail
    prob["_meta"]["tok"] = tok
    return bucket_of(tail, tok), prob


def _init_worker():
    sys.path.insert(0, "src")


def main():
    import multiprocessing as mp
    out_path = sys.argv[1] if len(sys.argv) > 1 else "synth_crypt_anchor.jsonl"
    quotas = dict(QUOTAS)
    if len(sys.argv) > 2:
        quotas = {}
        for kv in sys.argv[2].split(","):
            k, v = kv.split("=")
            quotas[k] = int(v)
    globals()["QUOTAS"].clear()
    globals()["QUOTAS"].update(quotas)
    filled = {k: [] for k in QUOTAS}
    seen = set()
    idx = 0
    BATCH = 400
    with mp.Pool(10, initializer=_init_worker) as pool:
        while any(len(filled[k]) < QUOTAS[k] for k in QUOTAS) and idx < 120000:
            batch = list(range(idx, idx + BATCH))
            idx += BATCH
            for r in pool.map(worker, batch, chunksize=10):
                if r is None:
                    continue
                bucket, prob = r
                if prob["id"] in seen or bucket not in filled:
                    continue
                if len(filled[bucket]) >= QUOTAS[bucket]:
                    continue
                seen.add(prob["id"])
                filled[bucket].append(prob)
            print(f"idx={idx} " + " ".join(f"{k}={len(v)}/{QUOTAS[k]}"
                                           for k, v in filled.items()), flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k in filled:
            for prob in filled[k]:
                f.write(json.dumps(prob) + "\n")
    total = sum(len(v) for v in filled.values())
    print(f"wrote {total} -> {out_path}")


if __name__ == "__main__":
    main()
