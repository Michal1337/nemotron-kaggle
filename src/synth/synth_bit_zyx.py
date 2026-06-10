"""Synthetic bit_manipulation generator for the zyx narrator (shape-quota'd).

Motivation (3ep eval on bit-zyx-solved-8020): base shape executes at 95% and
climbing, the override shapes were FLAT (reselect 23%, witness 0%) at ~6% of
the corpus. This generator samples problems from the WORD-PROGRAM class
(random <=3-term ROT/SHL/SHR programs, optionally negated leaves, AND/OR/XOR
combiners - exactly bitword_solver's hypothesis class), narrates them with the
UNIFIED narrator (override = explicit program render), keeps narrator-correct
CoTs and fills shape quotas:

    override  500   (stride pick disagrees, program render fires)
    base      700   (stride agrees; word-class problems still in-distribution)

Deterministic hash-seeded, no clock/global RNG. Token safety: CoTs capped at
CHAR_CAP=8800 chars (~6800 real tokens; the budget is tight - real corpus max
was 7651/7680) and the build's real-tokenizer filter remains the final gate.

Output: jsonl {id, category, prompt, answer, examples, question, _meta}
compatible with tv_step4_build synth ingestion.
"""
import hashlib
import json
import random
import sys

sys.path.insert(0, "src")
from reasoners.bitword_solver import eval_program  # noqa: E402
import reasoners.bit_manipulation_zyx as Z  # noqa: E402
from reasoners.store_types import Problem, Example  # noqa: E402

PROMPT = ("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
          "binary numbers. The transformation involves operations like bit shifts, "
          "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.\n\n"
          "Here are some examples of input -> output:\n{exs}\n\n"
          "Now, determine the output for: {q}")

KINDS = ["ROT", "SHL", "SHR"]
OPS = ["AND", "OR", "XOR"]
N_EX = 8
CHAR_CAP = 8800
QUOTAS = {"override": 500, "base": 700}


def gen_program(rng):
    n_terms = rng.choice([1, 2, 2, 3, 3, 3])
    prog = []
    for t in range(n_terms):
        kind = rng.choice(KINDS)
        k = rng.randrange(8) if kind == "ROT" else rng.randrange(1, 8)
        neg = rng.random() < 0.3
        if t > 0:
            prog.append(rng.choice(OPS))
        prog.append((kind, k, neg))
    return tuple(prog)


def gen_candidate(i):
    rng = random.Random(int(hashlib.md5(f"bitsynz-{i}".encode()).hexdigest(), 16))
    prog = gen_program(rng)
    words = set()
    guard = 0
    while len(words) < N_EX + 1 and guard < 200:
        words.add(format(rng.randrange(256), "08b"))
        guard += 1
    if len(words) < N_EX + 1:
        return None
    words = sorted(words, key=lambda w: rng.random())
    q = words[-1]
    exs = []
    try:
        for w in words[:N_EX]:
            _, res = eval_program(w, prog)
            exs.append((w, res))
        _, gold = eval_program(q, prog)
    except Exception:
        return None
    if not gold or len(gold) != 8:
        return None
    examples = [{"input_value": a, "output_value": b} for a, b in exs]
    prompt = PROMPT.format(exs="\n".join(f"{a} -> {b}" for a, b in exs), q=q)
    pid = "synz_" + hashlib.md5((str(exs) + q).encode()).hexdigest()[:10]
    return {"id": pid, "category": "bit_manipulation", "prompt": prompt,
            "answer": gold, "examples": examples, "question": q,
            "_meta": {"prog_terms": len(prog[0::2])}}


def classify(prob):
    p = Problem(id=prob["id"], category="bit_manipulation",
                examples=[Example(e["input_value"], e["output_value"]) for e in prob["examples"]],
                question=prob["question"], answer=prob["answer"])
    try:
        cot = Z.reasoning_bit_manipulation(p)
    except Exception:
        return None
    if not cot or len(cot) > CHAR_CAP:
        return None
    boxed = cot.rsplit("boxed{", 1)[-1].split("}")[0]
    if boxed != prob["answer"]:
        return None
    return "override" if "Cross-check (whole-word)" in cot else "base"


def worker(i):
    prob = gen_candidate(i)
    if prob is None:
        return None
    shape = classify(prob)
    if shape is None:
        return None
    prob["_meta"]["shape"] = shape
    return shape, prob


def _init_worker():
    sys.path.insert(0, "src")


def main():
    import multiprocessing as mp
    out_path = sys.argv[1] if len(sys.argv) > 1 else "synth_bit_zyx.jsonl"
    filled = {k: [] for k in QUOTAS}
    seen = set()
    idx = 0
    BATCH = 400
    with mp.Pool(12, initializer=_init_worker) as pool:
        while any(len(filled[k]) < QUOTAS[k] for k in QUOTAS) and idx < 40000:
            batch = list(range(idx, idx + BATCH))
            idx += BATCH
            for r in pool.map(worker, batch, chunksize=10):
                if r is None:
                    continue
                shape, prob = r
                if prob["id"] in seen or len(filled[shape]) >= QUOTAS[shape]:
                    continue
                seen.add(prob["id"])
                filled[shape].append(prob)
            print(f"idx={idx} " + str({k: len(v) for k, v in filled.items()}), flush=True)
    rows = [p for k in sorted(filled) for p in filled[k]]
    with open(out_path, "w", encoding="utf-8") as f:
        for p in rows:
            f.write(json.dumps(p) + "\n")
    print(f"WROTE {len(rows)} rows -> {out_path}")
    print("SYNTH_BIT_DONE")


if __name__ == "__main__":
    main()
