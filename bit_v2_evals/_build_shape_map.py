"""Build per-pid shape map for the 304 bit_v2 val pids.

For each pid: run the production zyx narrator (src/reasoners/bit_manipulation_zyx.py)
on the parsed problem; is_override = the CoT takes the 'Cross-check (whole-word)'
path (whole-word solver disagrees with per-bit stride selection). Also derive
whole-word program attrs (solve_program), per-bit Selected family mix, PORT
(huikang) solvability, ops-per-line proxy from the apply block, and eval
correctness per epoch.
"""
import json
import re
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from reasoners.store_types import Problem, Example  # noqa: E402
from reasoners.bit_manipulation_zyx import reasoning_bit_manipulation as zyx_narrate  # noqa: E402
from reasoners.bit_manipulation import reasoning_bit_manipulation as port_narrate  # noqa: E402
from reasoners.bitword_solver import solve_program, program_name  # noqa: E402

EX_RE = re.compile(r"^([01]{8}) -> ([01]{8})$", re.M)
Q_RE = re.compile(r"determine the output for:\s*([01]{8})")
BOX_RE = re.compile(r"\\boxed\{([01]{8})\}")


def parse_prompt(prompt):
    pairs = EX_RE.findall(prompt)
    q = Q_RE.search(prompt)
    assert pairs and q, prompt[:200]
    return pairs, q.group(1)


def boxed(text):
    if not text:
        return None
    m = BOX_RE.findall(text)
    return m[-1] if m else None


def family_of_expr(expr):
    if expr.startswith("default"):
        return "DEFAULT"
    if expr.startswith("C0") or expr.startswith("C1"):
        return "CONST"
    for fam in ("AND-NOT", "OR-NOT", "XOR-NOT", "AND", "OR", "XOR", "NOT", "I"):
        if expr.startswith(fam):
            return fam
    return "?"


def parse_selected(cot):
    """Family per bit from the Selected section (non-override CoTs only)."""
    m = re.search(r"\nSelected\n((?:\d .+\n){8})", cot)
    if not m:
        return None
    fams = []
    for line in m.group(1).strip().split("\n"):
        expr = line.split(" ", 1)[1]
        fams.append(family_of_expr(expr))
    return fams


def apply_block_eq_stats(cot):
    """Ops-per-line proxy: '=' count per line in the Applying block."""
    idx = cot.rfind("Applying to ")
    if idx < 0:
        return None, None
    block = cot[idx:]
    counts = [ln.count("=") for ln in block.split("\n") if "=" in ln and "boxed" not in ln]
    if not counts:
        return None, None
    return round(sum(counts) / len(counts), 3), max(counts)


def prog_attrs(prog):
    """prog = [term, op, term, op, term...]; term = (kind, k, neg) per bitword_solver."""
    terms = prog[0::2]
    ops = list(prog[1::2])
    kinds = sorted({t[0] for t in terms})
    has_not = any(t[2] for t in terms)
    if kinds == ["ROT"]:
        base = "rot"
    elif set(kinds) <= {"SHL", "SHR"}:
        base = "shift"
    else:
        base = "shift_rot_mix"
    if len(terms) == 1:
        kind = f"unary_{base}" + ("_not" if has_not else "")
    else:
        kind = f"{'_'.join(sorted(set(ops)))}_{base}".lower() + ("_not" if has_not else "")
    return {
        "ww_program": program_name(prog),
        "ww_n_terms": len(terms),
        "ww_term_kinds": kinds,
        "ww_combine_ops": sorted(set(ops)),
        "ww_has_not": has_not,
        "program_kind": kind,
    }


def classify_one(row):
    pid, gold = row["id"], row["answer"]
    pairs, qbits = parse_prompt(row["prompt"])
    # narrator is answer-blind (decoy-verified 1602/1602); answer field unused
    prob = Problem(id=pid, category="bit_manipulation",
                   examples=[Example(i, o) for i, o in pairs],
                   question=qbits, answer=gold, prompt=row["prompt"])

    cot = zyx_narrate(prob)
    is_override = bool(cot) and "Cross-check (whole-word)" in cot
    narr_ans = boxed(cot)

    port_prob = Problem(id=pid, category="bit_manipulation",
                        examples=[Example(i, o) for i, o in pairs],
                        question=qbits, answer=gold, prompt=row["prompt"])
    try:
        port_ans = boxed(port_narrate(port_prob))
    except Exception:
        port_ans = None

    # solve_program has the same global-agreement semantics as solve():
    # ww_ans is not None only when ALL example-consistent <=3-term programs agree.
    ww_ans, prog = solve_program(list(pairs), qbits)

    eq_mean, eq_max = apply_block_eq_stats(cot) if cot else (None, None)
    sel = parse_selected(cot) if cot else None

    attrs = {
        "is_override": is_override,
        "n_examples": len(pairs),
        "word_width": 8,
        "query_bits": qbits,
        "gold": gold,
        "narrator_answer": narr_ans,
        "narrator_solves": narr_ans == gold,
        "port_answer": port_ans,
        "port_solves": port_ans == gold,
        "huikang_solvable": port_ans == gold,
        "ww_answer": ww_ans,
        "ww_program": None,
        "ww_n_terms": None,
        "ww_term_kinds": None,
        "ww_combine_ops": None,
        "ww_has_not": None,
        "program_kind": "per_bit_only",
        "selected_families": sorted(set(sel)) if sel else None,
        "n_default_bits": sel.count("DEFAULT") if sel else None,
        "apply_eq_per_line_mean": eq_mean,
        "apply_eq_per_line_max": eq_max,
        "cot_chars": len(cot) if cot else None,
        "cot_lines": cot.count("\n") + 1 if cot else None,
    }
    if prog is not None and ww_ans is not None:
        attrs.update(prog_attrs(prog))
    return pid, attrs


def main():
    evdir = os.path.join(ROOT, "bit_v2_evals")
    rows = [json.loads(l) for l in open(os.path.join(evdir, "val.jsonl"), encoding="utf-8")]

    evals = {}
    for tag, fn in [("ep1", "eval-ep1.json"), ("ep2", "eval-ep2.json"),
                    ("ep3", "eval-ep3.json"), ("old_ep3", "old-eval-ep3.json")]:
        d = json.load(open(os.path.join(evdir, fn), encoding="utf-8"))
        evals[tag] = {r["id"]: r for r in d["results"]}

    shape = {}
    stats = {"override": 0, "narr_solves": 0, "port_solves": 0, "ww_some": 0}
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as pool:
        for pid, attrs in pool.map(classify_one, rows, chunksize=4):
            attrs["correct_ep1"] = evals["ep1"][pid]["correct"]
            attrs["correct_ep2"] = evals["ep2"][pid]["correct"]
            attrs["correct_ep3"] = evals["ep3"][pid]["correct"]
            attrs["correct_old_ep3"] = evals["old_ep3"][pid]["correct"]
            attrs["gen_has_crosscheck_ep3"] = "Cross-check" in evals["ep3"][pid]["generation"]
            shape[pid] = attrs
            stats["override"] += attrs["is_override"]
            stats["narr_solves"] += attrs["narrator_solves"]
            stats["port_solves"] += attrs["port_solves"]
            stats["ww_some"] += attrs["ww_answer"] is not None

    out = os.path.join(evdir, "shape_map.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(shape, f, indent=1)
    print("wrote", out, "n=", len(shape))
    print(stats)

    # summaries
    from collections import Counter
    pk = Counter(v["program_kind"] for v in shape.values())
    print("program_kind:", dict(pk.most_common()))
    ov = [v for v in shape.values() if v["is_override"]]
    nov = [v for v in shape.values() if not v["is_override"]]
    for tag in ("ep1", "ep2", "ep3", "old_ep3"):
        k = f"correct_{tag}"
        a = sum(v[k] for v in ov) / max(len(ov), 1)
        b = sum(v[k] for v in nov) / max(len(nov), 1)
        print(f"{tag}: override acc {a:.3f} (n={len(ov)})  non-override acc {b:.3f} (n={len(nov)})")
    # cross-validation: model generation contains Cross-check vs my label
    agree = sum(1 for v in shape.values() if v["gen_has_crosscheck_ep3"] == v["is_override"])
    print(f"ep3 generation Cross-check marker agrees with is_override: {agree}/304")
    nex = Counter(v["n_examples"] for v in shape.values())
    print("n_examples:", dict(sorted(nex.items())))
    sf = Counter(tuple(v["selected_families"]) if v["selected_families"] else ("<override/none>",) for v in shape.values())
    print("selected_families top:", sf.most_common(12))
    print("huikang_solvable:", sum(v["huikang_solvable"] for v in shape.values()))
    hk_no = [v for v in shape.values() if not v["huikang_solvable"]]
    for tag in ("ep3", "old_ep3"):
        k = f"correct_{tag}"
        print(f"{tag} acc on NON-huikang-solvable (n={len(hk_no)}):",
              round(sum(v[k] for v in hk_no) / max(len(hk_no), 1), 3))


if __name__ == "__main__":
    main()
