"""Synthetic cryptarithm_GUESS generator — the crypt analog of eq_guess_v2.py.

Mirrors eq_guess_v2: it walks the narrator's guess table cell-by-cell (R1 family
signatures, R2 canonical +/-/*, R3 default), generates problems whose examples
realize that cell, and VALIDATES each by running the narrator
(reasoning_cryptarithm_guess_fam) — keeping only samples whose boxed answer ==
gold, that fire the intended tier, and that fit the token budget.

The crypt-specific layer is the CIPHER: a problem fixes one global digit->symbol
bijection; every operand and result digit is rendered through it, under the
chosen orientation (standard | rev_both, the only two that occur). The query
operator char is UNSEEN in the examples (that is what makes it a guess); its
underlying op is the cell's target so the table's prediction is correct by
construction.

Run (SLURM): NPROC unused (single proc, fast). Output JSONL of problem dicts
matching the master schema (category cryptarithm_guess).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ on path

from reasoners.store_types import Example, Problem
from reasoners.crypt_joint_csp import OP_FNS, OP_FNS_SIGNED, SIGNED_OPS, _MODE_FLAGS
import reasoners.crypt_guess_fam as NARR


# Digit-cipher symbol vocab (printable, distinct from the operator vocab).
SYM_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
# Operator-char vocab (same family eq uses; must be disjoint from chosen symbols).
OP_VOCAB = ['-', '*', '+', '[', '<', '|', '!', '@', ']', '"', '^', '/', '%',
            '(', "'", '}', '`', '>', '?', '#', ')', '\\', ':', '{', '&', '$']

# Concrete ops per family (what an example operator of that family can be).
FAM_OPS = {
    "ADD":  ["add", "add_p1", "add_m1"],
    "MUL":  ["mul", "mul_p1", "mul_m1"],
    "SUB":  ["sub", "sub_signed", "sub_p1", "sub_m1"],
    "RSUB": ["rsub", "rsub_signed", "rsub_p1", "rsub_m1"],
    "ABS":  ["absdiff", "neg_absdiff", "absdiff_p1", "absdiff_m1"],
    "MOD":  ["mod"], "RMOD": ["rmod"], "GCD": ["gcd"],
}
CANON_CHARS = ['+', '-', '*']


def _compute(op, da1, da2, db1, db2, mode):
    """Render one example under cipher-free digit semantics: returns
    (result_magnitude_digit_string_post_orientation, neg_flag) or None."""
    L = da1 * 10 + da2
    R = db1 * 10 + db2
    swap, rev = _MODE_FLAGS[mode]
    if swap:
        L = (L % 10) * 10 + L // 10
        R = (R % 10) * 10 + R // 10
    signed = op in SIGNED_OPS
    fn = OP_FNS_SIGNED.get(op) if signed else OP_FNS.get(op)
    if fn is None:
        return None
    try:
        v = fn(L, R)
    except Exception:
        return None
    if v is None:
        return None
    if signed:
        neg = v < 0
        va = abs(v)
    else:
        if v < 0:
            return None
        neg = False
        va = v
    s = str(va)
    if rev:
        s = s[::-1]
    return s, neg


def _gen_one(target_famsig, target_qop, qop_char_pool, rng, max_tries=60):
    """Generate a single (examples, query, gold, mode) realizing the cell, or None."""
    fams = target_famsig.split(",")
    for _ in range(max_tries):
        mode = rng.choice(["standard", "rev_both"])
        # digit -> symbol bijection
        syms = rng.sample(SYM_VOCAB, 10)
        sym = lambda d: syms[d]
        # operator chars disjoint from the digit symbols
        avail_ops = [c for c in OP_VOCAB if c not in syms]
        if len(avail_ops) < len(fams) + 1:
            continue
        op_chars = rng.sample(avail_ops, len(fams) + 1)
        q_op = (rng.choice([c for c in qop_char_pool if c not in syms])
                if qop_char_pool else op_chars[0])
        if q_op in op_chars[1:]:
            continue
        ex_op_chars = op_chars[1:]
        # concrete op per family
        ex_ops = [rng.choice(FAM_OPS[f]) for f in fams]
        n_ex = rng.randint(max(4, len(fams)), 5)
        # distribute examples across the family operators
        assign = [i % len(fams) for i in range(n_ex)]
        rng.shuffle(assign)
        examples = []
        ok = True
        for idx in assign:
            op_char = ex_op_chars[idx]
            op = ex_ops[idx]
            made = None
            for _t in range(40):
                da1, da2 = rng.randint(0, 9), rng.randint(0, 9)
                db1, db2 = rng.randint(0, 9), rng.randint(0, 9)
                r = _compute(op, da1, da2, db1, db2, mode)
                if r is None:
                    continue
                s, neg = r
                if len(s) > 4:
                    continue
                inp = sym(da1) + sym(da2) + op_char + sym(db1) + sym(db2)
                out = (op_char if neg else "") + "".join(sym(int(c)) for c in s)
                made = (inp, out)
                break
            if made is None:
                ok = False
                break
            examples.append(made)
        if not ok:
            continue
        # query under the GUESSED op (target_qop)
        made_q = None
        for _t in range(60):
            da1, da2 = rng.randint(0, 9), rng.randint(0, 9)
            db1, db2 = rng.randint(0, 9), rng.randint(0, 9)
            r = _compute(target_qop, da1, da2, db1, db2, mode)
            if r is None:
                continue
            s, neg = r
            if len(s) > 4:
                continue
            ques = sym(da1) + sym(da2) + q_op + sym(db1) + sym(db2)
            gold = (q_op if neg else "") + "".join(sym(int(c)) for c in s)
            made_q = (ques, gold)
            break
        if made_q is None:
            continue
        ques, gold = made_q
        return examples, ques, gold, mode, ex_ops
    return None


def _build_prob_dict(pid, examples, question, gold, meta):
    prompt = (
        "In Alice's Wonderland, each cipher symbol stands for a digit and a secret "
        "rule maps the operator. Below are a few examples:\n"
        + "\n".join(f"{inp} = {out}" for inp, out in examples)
        + f"\nNow, determine the result for: {question}"
    )
    return {
        "id": pid,
        "category": "cryptarithm_guess",
        "prompt": prompt,
        "answer": gold,
        "examples": [{"input_value": inp, "output_value": out} for inp, out in examples],
        "question": question,
        "_meta": meta,
    }


def _validate(pid, examples, question, gold, tier_str, tok, max_tokens):
    prob = Problem(
        id=pid, category="cryptarithm_guess",
        examples=[Example(i, o) for i, o in examples],
        question=question, answer=gold,
    )
    try:
        cot = NARR.reasoning_cryptarithm_guess_fam(prob)
    except Exception:
        return None
    if cot is None:
        return None
    if tier_str and tier_str not in cot:
        return None
    if tok is not None and len(tok.encode(cot).ids) > max_tokens:
        return None
    return cot


def _gen_cell(target_famsig, target_qop, qop_char_pool, tier_str, n_target,
              rng, tok, max_tokens, max_attempts):
    out = []
    seen = set()
    fails = Counter()
    attempts = 0
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        g = _gen_one(target_famsig, target_qop, qop_char_pool, rng)
        if g is None:
            fails["gen"] += 1
            continue
        examples, question, gold, mode, ex_ops = g
        if not gold or len(gold) > 5:
            fails["gold"] += 1
            continue
        pid = hashlib.md5(
            f"{target_famsig}_{target_qop}_{attempts}".encode()
        ).hexdigest()[:8]
        if pid in seen:
            fails["dup"] += 1
            continue
        cot = _validate(pid, examples, question, gold, tier_str, tok, max_tokens)
        if cot is None:
            fails["validate"] += 1
            continue
        seen.add(pid)
        out.append(_build_prob_dict(pid, examples, question, gold, {
            "famsig": target_famsig, "query_op": target_qop, "mode": mode,
            "example_ops": ex_ops, "tier": tier_str or "R3",
        }))
    return out, fails


_TOK = None
_MAXTOK = 7600


def _init_tok(tokenizer_path, max_tokens):
    global _TOK, _MAXTOK
    _MAXTOK = max_tokens
    if tokenizer_path:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(tokenizer_path)


def _cell_worker(task):
    # one independent family cell -> its own deterministic rng (cells were a
    # sequential single-process loop; they are independent so parallelize)
    tier, label, famsig, qop, char_pool, tier_str, per_cell, max_attempts, seed = task
    rng = random.Random(int(hashlib.md5(
        f"{seed}|{tier}|{label}|{qop}".encode()).hexdigest(), 16))
    samples, fails = _gen_cell(famsig, qop, char_pool, tier_str, per_cell,
                               rng, _TOK, _MAXTOK, max_attempts)
    return (tier, label, qop, samples, dict(fails))


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=7600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts-factor", type=int, default=90)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.json (skip token check if unset)")
    args = ap.parse_args()

    fam_r1 = NARR._FAM_R1
    r2 = NARR._R2_CANON
    r3 = NARR._R3_DEFAULT
    # For R2/R3 fallthrough we need an example signature that (a) is NOT an R1
    # key and (b) is cipher-recoverable. A MUL operator anchors the cipher
    # (its length-4 results strongly pin digits), so prefer a MUL-containing
    # multi-family sig that misses R1.
    # cap-friendly: ADD (length-3, ~9 candidates) bounds the recovery search and
    # anchors the cipher; MOD/GCD complete a 2-family sig that misses R1.
    non_r1_sig = next(
        (s for s in ["ADD,MOD", "ADD,GCD", "ADD,RMOD", "ABS,MOD"]
         if s not in fam_r1),
        "ADD,MOD",
    )

    maxa = args.per_cell * args.max_attempts_factor
    tasks = []
    # ---- R1 cells (skip concat: positional, not CSP-recoverable) ----
    for sig in sorted(fam_r1):
        if any(fm not in FAM_OPS for fm in sig.split(",")):
            print(f"  [R1] ({sig}) -> SKIP (non-arith family)", flush=True)
            continue
        tasks.append(("R1", sig, sig, fam_r1[sig], None, "R1 fires",
                      args.per_cell, maxa, args.seed))
    # ---- R2 cells (canonical +/-/*) ----
    for ch in CANON_CHARS:
        tasks.append(("R2", ch, non_r1_sig, r2[ch], [ch], "R2",
                      args.per_cell, maxa, args.seed))
    # ---- R3 default (query char not canonical) ----
    r3_pool = [c for c in OP_VOCAB if c not in CANON_CHARS]
    tasks.append(("R3", "default", non_r1_sig, r3, r3_pool, "R3 default",
                  args.per_cell, maxa, args.seed))

    total = 0
    short = []
    with open(args.out, "w") as f:
        with mp.Pool(args.workers, initializer=_init_tok,
                     initargs=(args.tokenizer, args.max_tokens)) as pool:
            for tier, label, qop, samples, fails in pool.imap_unordered(_cell_worker, tasks):
                for s in samples:
                    f.write(json.dumps(s) + "\n")
                total += len(samples)
                if len(samples) < args.per_cell:
                    short.append(((tier, label, qop), len(samples), fails))
                print(f"  [{tier} {label}] -> {qop}  n={len(samples)}", flush=True)

    print(f"\nTotal synth: {total}")
    print(f"Cells under target: {len(short)}")
    for cell, n, fails in short[:20]:
        print(f"  SHORT {cell} -> {n}/{args.per_cell}  fails={fails}")


if __name__ == "__main__":
    main()
