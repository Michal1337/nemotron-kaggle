"""Forensics for the 24 ep3 misses: locate first divergence vs gold narrator CoT,
verify per-bit rules / whole-word programs against the prompt examples, classify cause."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reasoners.store_types import Problem, Example  # noqa: E402
from reasoners.bit_manipulation_zyx import reasoning_bit_manipulation as zyx  # noqa: E402
from reasoners.bitword_solver import apply_term, combine_words  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX_RE = re.compile(r"^([01]{8}) -> ([01]{8})$", re.M)
Q_RE = re.compile(r"determine the output for:\s*([01]{8})")

ep3 = json.load(open(os.path.join(HERE, "eval-ep3.json")))
old = json.load(open(os.path.join(HERE, "old-eval-ep3.json")))
r3 = {r["id"]: r for r in ep3["results"]}
ro = {r["id"]: r for r in old["results"]}
miss = sorted([i for i, r in r3.items() if not r["correct"]])


def parse_prompt(prompt):
    pairs = EX_RE.findall(prompt)
    q = Q_RE.search(prompt)
    return pairs, q.group(1)


# ---- per-bit rule eval ----
def eval_expr(expr, bits):
    """Evaluate Selected expr ('I6','NOT3','C0','C1','AND25','XOR-NOT14','default 1') on word."""
    if expr.startswith("default"):
        return "1"
    if expr == "C0":
        return "0"
    if expr == "C1":
        return "1"
    m = re.match(r"^(AND-NOT|OR-NOT|XOR-NOT|AND|OR|XOR|NOT|I)(\d)(\d)?$", expr)
    if not m:
        return None
    fam, p, s = m.group(1), int(m.group(2)), m.group(3)
    a = bits[p]
    if fam == "I":
        return a
    if fam == "NOT":
        return "1" if a == "0" else "0"
    b = bits[int(s)]
    if "-NOT" in fam:
        b = "1" if b == "0" else "0"
    base = fam.split("-")[0]
    if base == "AND":
        return "1" if a == "1" and b == "1" else "0"
    if base == "OR":
        return "1" if a == "1" or b == "1" else "0"
    return "1" if a != b else "0"


def rule_fits_examples(expr, pairs, out_bit):
    ok = all(eval_expr(expr, i) == o[out_bit] for i, o in pairs)
    return ok


# ---- whole-word program parsing ('(ROT5) XOR (NOT SHL1) OR (SHR6)') ----
TERM_RE = re.compile(r"\((NOT )?(ROT|SHL|SHR)(\d)\)")


def parse_prog(name):
    terms = [(m.group(2), int(m.group(3)), bool(m.group(1))) for m in TERM_RE.finditer(name)]
    ops = re.findall(r"\)\s+(AND|OR|XOR)\s+\(", name)
    if len(terms) != len(ops) + 1:
        return None
    return terms, ops


def eval_prog(word, terms, ops):
    words = [apply_term(word, t) for t in terms]
    acc = words[0]
    for op, w in zip(ops, words[1:]):
        acc = combine_words(acc, w, op)
    return words, acc


def first_diff(gold_cot, gen):
    gl = gold_cot.split("\n")
    nl = gen.split("\n")
    for k in range(min(len(gl), len(nl))):
        a, b = gl[k].strip(), nl[k].strip()
        if a != b:
            # tolerate the en-dash mojibake line
            if "boxed" in a and "boxed" in b and re.sub(r"[^\x00-\x7f]", "?", a) == re.sub(r"[^\x00-\x7f]", "?", b):
                continue
            return k, gl[k], nl[k]
    if len(gl) != len(nl):
        return min(len(gl), len(nl)), "<end len %d>" % len(gl), "<end len %d>" % len(nl)
    return None, None, None


report = {}
for pid in miss:
    r = r3[pid]
    pairs, qbits = parse_prompt(r["prompt"])
    gold = r["gold"]
    gen = r["generation"]
    prob = Problem(id=pid, category="bit_manipulation",
                   examples=[Example(i, o) for i, o in pairs],
                   question=qbits, answer=gold, prompt=r["prompt"])
    gold_cot = zyx(prob)
    k, gline, nline = first_diff(gold_cot, gen)

    info = {
        "pid": pid, "gold": gold, "pred": r["predicted"],
        "old_correct": ro[pid]["correct"], "gen_tokens": r["gen_tokens"],
        "first_diff_line": k,
        "first_diff_frac": None if k is None else round(k / max(1, len(gold_cot.split("\n"))), 3),
        "gold_line": gline, "gen_line": nline,
        "n_gold_lines": len(gold_cot.split("\n")),
        "truncated": "boxed" not in gen[-200:],
        "near_miss_gold_in_gen": gold in gen,
        # where does gold appear (if anywhere)?
        "gold_locations": [],
    }
    if gold in gen:
        for m in re.finditer(re.escape(gold), gen):
            s = max(0, m.start() - 60)
            ctx = gen[s:m.end() + 20].split("\n")[-1][:90]
            info["gold_locations"].append(ctx)
        info["gold_locations"] = info["gold_locations"][:4]

    # transcription check: restated example tables in gen
    trans_errs = []
    for j, (inp, out) in enumerate(pairs):
        mo = re.search(r"Output %d: ([01]{8})" % j, gen)
        mi = re.search(r"Input %d: ([01]{8})" % j, gen)
        if mo and mo.group(1) != out:
            trans_errs.append(f"Output{j} restated {mo.group(1)} != {out}")
        if mi and mi.group(1) != inp:
            trans_errs.append(f"Input{j} restated {mi.group(1)} != {inp}")
    # query transcription in 'Applying to'
    ma = re.search(r"Applying to ([01]{8})", gen)
    if ma and ma.group(1) != qbits:
        trans_errs.append(f"Applying-to {ma.group(1)} != query {qbits}")
    info["transcription_errors"] = trans_errs

    # path taken by the MODEL
    has_xchk = "Cross-check (whole-word)" in gen
    info["gen_path"] = "crosscheck" if has_xchk else "per_bit"
    info["gold_path"] = "crosscheck" if "Cross-check (whole-word)" in gold_cot else "per_bit"

    if has_xchk:
        mf = re.search(r"Found (.+)", gen)
        prog_name = mf.group(1).strip() if mf else None
        info["gen_program"] = prog_name
        mg = re.search(r"Found (.+)", gold_cot)
        info["gold_program"] = mg.group(1).strip() if mg else None
        pp = parse_prog(prog_name) if prog_name else None
        if pp:
            terms, ops = pp
            fits = [eval_prog(i, terms, ops)[1] == o for i, o in pairs]
            info["gen_program_fits_examples"] = f"{sum(fits)}/{len(fits)}"
            words, res = eval_prog(qbits, terms, ops)
            info["gen_program_on_query"] = res
            info["gen_program_query_eq_gold"] = res == gold
            # model's claimed Combined
            mc = re.search(r"Combined = ([01]{8})", gen)
            info["gen_claimed_combined"] = mc.group(1) if mc else None
            # model's claimed term words on query (after 'Applying to')
            tail = gen[gen.rfind("Applying to "):]
            claimed = re.findall(r"= ([01]{8})", tail)
            true_words = [apply_term(qbits, t) for t in terms]
            info["term_word_errors"] = [
                f"term{ix}({t[0]}{t[1]}{'N' if t[2] else ''}) claimed {c} true {w}"
                for ix, (t, c, w) in enumerate(zip(terms, claimed, true_words)) if c != w
            ]
            # check the model's per-example 'yes' lines for confabulation
            confab = 0
            for inp, out in pairs:
                mline = re.search(re.escape(inp) + r" .*-> ([01]{8}) vs ([01]{8}) (yes|no)", gen)
                if mline:
                    _, true_res = eval_prog(inp, terms, ops)
                    if mline.group(3) == "yes" and (true_res != out or mline.group(1) != true_res):
                        confab += 1
            info["confabulated_yes_lines"] = confab
    else:
        # parse Selected block (the LAST one)
        msel = None
        for msel in re.finditer(r"\nSelected\n((?:\d .+\n){8})", gen):
            pass
        if msel:
            sel = [ln.split(" ", 1)[1].strip() for ln in msel.group(1).strip().split("\n")]
            info["gen_selected"] = sel
            # gold narrator Selected
            mgs = None
            for mgs in re.finditer(r"\nSelected\n((?:\d .+\n){8})", gold_cot):
                pass
            info["gold_selected"] = [ln.split(" ", 1)[1].strip() for ln in mgs.group(1).strip().split("\n")] if mgs else None
            bad_bits = []
            for b in range(8):
                expr = sel[b]
                fit = rule_fits_examples(expr, pairs, b)
                pred_bit = eval_expr(expr, qbits)
                gold_bit = gold[b]
                final_bit = r["predicted"][b] if r["predicted"] and len(r["predicted"]) == 8 else None
                if final_bit != gold_bit or pred_bit != gold_bit or not fit:
                    # model's claimed applied bit
                    mapp = re.search(r"\n%d %s = .*?(\d)\s*$" % (b, re.escape(expr)), gen[gen.rfind("Applying to "):], re.M)
                    claimed = mapp.group(1) if mapp else None
                    bad_bits.append({
                        "bit": b, "expr": expr, "fits_examples": fit,
                        "rule_on_query": pred_bit, "claimed": claimed,
                        "gold_bit": gold_bit, "final_bit": final_bit,
                    })
            info["bad_bits"] = bad_bits
        else:
            info["gen_selected"] = None
    report[pid] = info

json.dump(report, open(os.path.join(HERE, "_miss_forensics.json"), "w"), indent=1)
for pid, info in report.items():
    print("=" * 80)
    for k, v in info.items():
        if k in ("gold_line", "gen_line") and v and len(str(v)) > 100:
            v = str(v)[:100] + "..."
        print(f"  {k}: {v}")
