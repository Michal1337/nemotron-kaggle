"""Synthetic cryptarithm_deduce generator for the crypt_vfirst narrator
(audit wf_b996f4a7-4a9, build step 6).

Deterministic (hash-seeded per index, no clock/global RNG). Every candidate
problem is rendered by the REAL narrator and kept only if the narrator's
boxed answer equals the generated gold (self-verification), then bucketed by
its narration tail. Quotas (total 2400):

  det     1250  DETERMINED tail (single agreeing reading)
  agree    700  agree-open tail (resolved readings agree, some hypotheses open)
  easy     300  det/agree with est tok < 1800 (short-procedure anchors)
  concat   150  concat-gate problems (incl. mixed example ops)

Distribution knobs mirror the real set: n_ex 3/4/5 at 180/241/238, op chars
1/2/3 at 13/272/374, mode strictly 50:50 by index parity (op/mode identity
must be learned as procedure, not prior), >=40% of problems contain an add/mul
+-1 variant (the measured op-vs-+-1 disambiguation failure skill), glyphs drawn
from the real punctuation universe. Sign-prefix encoding follows the solver
semantics exactly (own op char marks a negative result; rev_both reverses
operand digits and the result string). zpad generation skipped (0.7% real rate).

Output: jsonl rows {id, category, prompt, answer, examples, question, _meta}
compatible with tv_step4_build synth ingestion and with
src/corpus/crypt_vfirst_gates.py --generate.
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, "src")
import reasoners.crypt_vfirst_solver as S  # noqa: E402
from reasoners.crypt_vfirst import reasoning_cryptarithm_vfirst, CHARS_PER_TOK, TOK_OVERHEAD  # noqa: E402
from reasoners.store_types import Problem, Example  # noqa: E402

GLYPHS = sorted(set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") - set("=,_;~"))
PROMPT = ("In Alice's Wonderland, a secret set of transformation rules is applied "
          "to equations. Below are a few examples:\n{exs}\nNow, determine the result for: {q}")

ARITH_OPS = ["add", "add_p1", "add_m1", "mul", "mul_p1", "mul_m1",
             "sub_signed", "rsub_signed", "absdiff", "neg_absdiff"]
FAMS = [["add", "add_p1", "add_m1"], ["mul", "mul_p1", "mul_m1"],
        ["sub_signed"], ["rsub_signed"], ["absdiff"], ["neg_absdiff"]]
SIGNED = {"sub_signed", "rsub_signed", "neg_absdiff"}

N_EX_W = [(3, 180), (4, 241), (5, 238)]
N_OPC_W = [(1, 13), (2, 272), (3, 374)]
if os.environ.get("SYNV_3OPC"):
    # survivor-heavy mode: favor 3 op chars (more stage-1 combos -> more
    # survivors; eval pids run survivor-count median ~12 vs synth default ~3)
    N_OPC_W = [(2, 100), (3, 550)]

QUOTAS = {"det": 1250, "agree": 700, "easy": 300, "concat": 150}


def wchoice(rng, pairs):
    tot = sum(w for _, w in pairs)
    x = rng.randrange(tot)
    for v, w in pairs:
        x -= w
        if x < 0:
            return v
    return pairs[-1][0]


def encode_output(v, op, oc, mode, enc):
    """Output string for value v under the solver's exact semantics."""
    sign = op in SIGNED and v < 0
    s = str(abs(v))
    if mode == "rev_both":
        s = s[::-1]
    body = "".join(enc[int(c)] for c in s)
    return (oc + body) if sign else body


def gen_candidate(i):
    rng = random.Random(int(hashlib.md5(f"cryptsynv-{i}".encode()).hexdigest(), 16))
    mode = "standard" if i % 2 == 0 else "rev_both"
    n_opc = wchoice(rng, N_OPC_W)
    n_ex = max(wchoice(rng, N_EX_W), n_opc)
    glyphs = rng.sample(GLYPHS, 10 + n_opc)
    dig_glyphs, op_glyphs = glyphs[:10], glyphs[10:]
    mapping = {g: d for d, g in enumerate(dig_glyphs)}
    enc = {d: g for g, d in mapping.items()}

    concat_slice = (i % 20 == 19)  # ~5% of candidates carry a concat op
    ops = {}
    want_pm1 = rng.random() < 0.55
    fams = rng.sample(FAMS, min(n_opc, len(FAMS)))
    for j, oc in enumerate(op_glyphs):
        fam = fams[j]
        if concat_slice and j == n_opc - 1 and n_opc > 1:
            ops[oc] = rng.choice(["concat_fwd", "concat_rev"])
        elif want_pm1 and len(fam) == 3 and j == 0:
            ops[oc] = rng.choice(fam[1:])  # force a +-1 variant
        else:
            ops[oc] = rng.choice(fam)

    # examples: every op char demonstrated at least once
    order = list(op_glyphs) + [rng.choice(op_glyphs) for _ in range(n_ex - n_opc)]
    rng.shuffle(order)
    exs = []
    seen_inputs = set()
    for oc in order:
        op = ops[oc]
        for _ in range(40):
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
                v = S.op_value(op, L, R)
                if v is None or (v < 0 and op not in SIGNED):
                    continue
                if op == "neg_absdiff" and v == 0 and (a1 != b1 or a2 != b2):
                    continue
                out = encode_output(v, op, oc, mode, enc)
            full = {g: d for g, d in mapping.items()}
            if not S.check_example_full(inp, out, op, mode, full):
                continue
            seen_inputs.add(inp)
            exs.append((inp, out))
            break
        else:
            return None
    if len(exs) != n_ex:
        return None

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
            v = S.op_value(op, L, R)
            if v is None or (v < 0 and op not in SIGNED):
                continue
            gold = encode_output(v, op, oc, mode, enc)
        if not gold:
            continue
        examples = [{"input_value": i_, "output_value": o_} for i_, o_ in exs]
        prompt = PROMPT.format(exs="\n".join(f"{i_} = {o_}" for i_, o_ in exs), q=q)
        pid = "synv_" + hashlib.md5((str(exs) + q).encode()).hexdigest()[:10]
        return {"id": pid, "category": "cryptarithm_deduce", "prompt": prompt,
                "answer": gold, "examples": examples, "question": q,
                "_meta": {"mode": mode, "ops": ops, "n_ex": n_ex}}
    return None


def classify(prob):
    """Render with the real narrator; return (tail, est_tok, cot) or None."""
    p = Problem(id=prob["id"], category="cryptarithm_deduce",
                examples=[Example(e["input_value"], e["output_value"]) for e in prob["examples"]],
                question=prob["question"], answer=prob["answer"])
    cot = reasoning_cryptarithm_vfirst(p)
    if cot is None:
        return None
    i = cot.rfind("boxed{")
    j = cot.rfind("}")
    if not (0 <= i < j) or cot[i + 6:j] != prob["answer"]:
        return None
    est = TOK_OVERHEAD + len(cot) / CHARS_PER_TOK
    if "no digit code needed" in cot:
        tail = "concat"
    elif "UNDERDETERMINED:" in cot:
        tail = "under"
    elif "DETERMINED:" in cot:
        tail = "det"
    elif "resolved readings agree" in cot:
        tail = "agree"
    else:
        return None
    return tail, est, cot


def bucket_of(tail, est):
    if tail == "concat":
        return "concat"
    if est < 1800:
        return "easy"
    return tail


def worker(i):
    prob = gen_candidate(i)
    if prob is None:
        return None
    r = classify(prob)
    if r is None:
        return None
    tail, est, cot = r
    prob["_meta"]["tail"] = tail
    prob["_meta"]["est_tok"] = round(est)
    prob["_meta"]["n_survivors"] = cot.count(": survives")
    return bucket_of(tail, est), prob


def _init_worker():
    sys.path.insert(0, "src")


def main():
    import multiprocessing as mp
    out_path = sys.argv[1] if len(sys.argv) > 1 else "synth_crypt_vfirst.jsonl"
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
        while any(len(filled[k]) < QUOTAS[k] for k in QUOTAS) and idx < 60000:
            batch = list(range(idx, idx + BATCH))
            idx += BATCH
            for r in pool.map(worker, batch, chunksize=10):
                if r is None:
                    continue
                bucket, prob = r
                if prob["id"] in seen or len(filled[bucket]) >= QUOTAS[bucket]:
                    continue
                seen.add(prob["id"])
                filled[bucket].append(prob)
            done = {k: len(v) for k, v in filled.items()}
            print(f"idx={idx} filled={done}", flush=True)
    rows = [p for k in sorted(filled) for p in filled[k]]
    with open(out_path, "w", encoding="utf-8") as f:
        for p in rows:
            f.write(json.dumps(p) + "\n")
    modes = sum(1 for p in rows if p["_meta"]["mode"] == "standard")
    print(f"WROTE {len(rows)} rows -> {out_path}  (standard {modes} / rev {len(rows)-modes})")
    print("SYNTH_DONE")


if __name__ == "__main__":
    main()
