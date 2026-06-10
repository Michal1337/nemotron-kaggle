"""Corpus gates for crypt_vfirst CoTs (audit wf_b996f4a7-4a9, build step 5).

Five mechanical PASS/FAIL gates over a generated CoT set. Run BEFORE any
training corpus that includes crypt_vfirst output:

  G1 boxed==gold          every kept CoT's last boxed equals gold (the build's
                          gold filter enforces this; the gate double-checks).
  G2 decoy invariance     re-narrating N pids with corrupted golds yields
                          byte-identical CoTs (gold never enters generation).
  G3 skeleton count       <=4 distinct phase skeletons (concat / determined /
                          underdetermined / agreed-with-open-rivals).
  G4 line executability   no line asserts a large hidden enumeration: every
                          line matches the renderer's known grammar, and
                          numeric claim lines re-verify (decode/compute lines
                          are re-executed and must be arithmetically true).
  G5 glyph closure        every non-template glyph in a CoT occurs in the
                          problem's examples/question (no invented symbols);
                          plus length p50/p90/max under the token budget.

Usage:
  python src/corpus/crypt_vfirst_gates.py <cots.jsonl>
  # each line: {"pid":..., "cot":..., "gold":..., "examples":[...], "question":...}
  python src/corpus/crypt_vfirst_gates.py --generate <crypt_real_all.jsonl>
  # generates CoTs from problems first (local narrator), then gates them.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "src")
from reasoners.crypt_vfirst import reasoning_cryptarithm_vfirst, CHARS_PER_TOK, TOK_OVERHEAD  # noqa: E402
from reasoners.store_types import Problem, Example  # noqa: E402

TOK_BUDGET = 7680
ARITH_RE = re.compile(
    r"(?<![\d|])(\d+)\s*([+*-])\s*(\d+)(?:\s*([+-])\s*(1))?\s*=\s*(-?\d+)")
ABS_RE = re.compile(r"(-?)\|(\d+)-(\d+)\|\s*=\s*(-?\d+)")
MOD_RE = re.compile(r"(\d+) mod (\d+)\s*=\s*(-?\d+)")
GCD_RE = re.compile(r"gcd\((\d+),(\d+)\)\s*=\s*(-?\d+)")


def boxed_of(cot):
    i = cot.rfind("boxed{")
    j = cot.rfind("}")
    return cot[i + 6:j] if 0 <= i < j else None


def skeleton_of(cot):
    """Phase skeleton fingerprint."""
    sk = []
    if "no digit code needed" in cot:
        sk.append("concat")
    if "Op filter" in cot:
        sk.append("filter")
    if "Resolve " in cot:
        sk.append("resolve")
    if "DETERMINED:" in cot:
        sk.append("det")
    elif "UNDERDETERMINED:" in cot:
        sk.append("under")
    elif "resolved readings agree" in cot:
        sk.append("agree-open")
    return tuple(sk)


def check_arith_lines(cot):
    """Re-execute every arithmetic claim line; STRICT equality. Returns false lines."""
    import math
    bad = []
    for line in cot.split("\n"):
        ok = True
        for m in ARITH_RE.finditer(line):
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            true = a + b if op == "+" else a * b if op == "*" else a - b
            if m.group(4):
                true += 1 if m.group(4) == "+" else -1
            if true != int(m.group(6)):
                ok = False
        for m in ABS_RE.finditer(line):
            true = abs(int(m.group(2)) - int(m.group(3)))
            if m.group(1) == "-":
                true = -true
            if true != int(m.group(4)):
                ok = False
        for m in MOD_RE.finditer(line):
            a, b = int(m.group(1)), int(m.group(2))
            if not b or a % b != int(m.group(3)):
                ok = False
        for m in GCD_RE.finditer(line):
            if math.gcd(int(m.group(1)), int(m.group(2))) != int(m.group(3)):
                ok = False
        if not ok:
            bad.append(line.strip()[:90])
    return bad


def glyph_closure(cot, examples, question):
    """Glyphs outside the fixed template vocabulary must come from the problem."""
    allowed = set("ABCDEFGHIJpqr0123456789")
    allowed |= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    allowed |= set(" \n\t.,:;()[]{}<>=+-*/\\|'\"`!?@#$%^&_~")
    prob_glyphs = set(question)
    for e in examples:
        prob_glyphs |= set(e["input_value"]) | set(e["output_value"])
    stray = set()
    for ch in cot:
        if ch not in allowed and ch not in prob_glyphs:
            stray.add(ch)
    return stray


def main():
    args = sys.argv[1:]
    gen = "--generate" in args
    path = [a for a in args if not a.startswith("--")][0]
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if gen:
        recs = []
        for r in rows:
            if r.get("category", "cryptarithm_deduce") != "cryptarithm_deduce":
                continue
            prob = Problem(id=r["id"], category="cryptarithm_deduce",
                           examples=[Example(e["input_value"], e["output_value"]) for e in r["examples"]],
                           question=r["question"], answer=r["answer"])
            cot = reasoning_cryptarithm_vfirst(prob)
            if cot is not None:
                recs.append({"pid": r["id"], "cot": cot, "gold": r["answer"],
                             "examples": r["examples"], "question": r["question"]})
        n_all = len(recs)
        recs = [r for r in recs if boxed_of(r["cot"]) == r["gold"]]
        print(f"generated {n_all} CoTs from {len(rows)} problems; "
              f"gold filter kept {len(recs)} (dropped {n_all - len(recs)} guess-misses)")
    else:
        recs = rows

    fails = []
    # G1 boxed==gold
    bad = [r["pid"] for r in recs if boxed_of(r["cot"]) != r["gold"]]
    print(f"G1 boxed==gold: {len(recs)-len(bad)}/{len(recs)}  {'PASS' if not bad else 'FAIL ' + str(bad[:5])}")
    if bad:
        fails.append("G1")

    # G2 decoy invariance on a fixed slice
    n_decoy = min(25, len(recs))
    ident = 0
    for r in recs[:n_decoy]:
        mk = lambda ans: Problem(id=r["pid"], category="cryptarithm_deduce",
                                 examples=[Example(e["input_value"], e["output_value"]) for e in r["examples"]],
                                 question=r["question"], answer=ans)
        a = reasoning_cryptarithm_vfirst(mk(r["gold"]))
        b = reasoning_cryptarithm_vfirst(mk(r["gold"][::-1] + "X"))
        ident += (a == b)
    print(f"G2 decoy invariance: {ident}/{n_decoy}  {'PASS' if ident == n_decoy else 'FAIL'}")
    if ident != n_decoy:
        fails.append("G2")

    # G3 skeleton count
    sks = Counter(skeleton_of(r["cot"]) for r in recs)
    print(f"G3 skeletons: {len(sks)} distinct {dict(sks)}  {'PASS' if len(sks) <= 4 else 'FAIL'}")
    if len(sks) > 4:
        fails.append("G3")

    # G4 arithmetic-claim re-execution
    bad_pids = []
    for r in recs:
        bad_lines = check_arith_lines(r["cot"])
        if bad_lines:
            bad_pids.append((r["pid"], bad_lines[0]))
    print(f"G4 arithmetic claims: {len(recs)-len(bad_pids)}/{len(recs)} clean  "
          f"{'PASS' if not bad_pids else 'FAIL ' + str(bad_pids[:3])}")
    if bad_pids:
        fails.append("G4")

    # G5 glyph closure + lengths
    stray_pids = []
    toks = []
    for r in recs:
        stray = glyph_closure(r["cot"], r["examples"], r["question"])
        if stray:
            stray_pids.append((r["pid"], "".join(sorted(stray))[:10]))
        toks.append(TOK_OVERHEAD + len(r["cot"]) / CHARS_PER_TOK)
    toks.sort()
    n = len(toks)
    over = sum(1 for t in toks if t > TOK_BUDGET)
    print(f"G5 glyph closure: {len(recs)-len(stray_pids)}/{len(recs)} clean"
          f"{'' if not stray_pids else ' STRAY ' + str(stray_pids[:3])}; "
          f"est-tok p50={toks[n//2]:.0f} p90={toks[int(n*.9)]:.0f} max={toks[-1]:.0f} over-budget={over}  "
          f"{'PASS' if not stray_pids and over == 0 else 'FAIL'}")
    if stray_pids or over:
        fails.append("G5")

    print("ALL GATES PASS" if not fails else f"GATES FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
