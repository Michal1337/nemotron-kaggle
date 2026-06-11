"""Anchor-DFS solver core (measurement-only): the mul-anchored design from the
shared writeup (27.8% reasoner-side), generalized:
  - anchor = the most-constraining fact: mul r.len=4 > mul r.len=3 > add r.len=3
    (leading digit forced to 1) > none (problem skipped for now)
  - anchor candidate list = all (a,b,op) consistent with the fact's repeat
    pattern under bijection, capped at LIST_CAP
  - DFS over remaining facts ordered by fewest new symbols; per fact per
    surviving op, solve the <=2 unknowns (small visible algebra later)
  - modes {none(std), swap(rev_both)}; std first
  - query: surviving op(s) of its char; unique answer or priority pick
Outputs per-pid: solved?, anchor type, list size, dfs nodes, narration-cost
proxy. Compare vs writeup 27.8% and our v2 (193/659).
"""
import json
import sys
from collections import defaultdict
from itertools import product

sys.path.insert(0, 'src')
import reasoners.crypt_vfirst_solver as S

LIST_CAP = 24
DFS_CAP = 20000
OP_PRIOR = ["add", "mul", "concat_fwd", "concat_rev", "concat_fwd_ro", "concat_rev_ro",
            "absdiff", "add_p1", "add_m1",
            "mul_p1", "mul_m1", "sub_signed", "rsub_signed", "neg_absdiff", "mod_max"]


def op_val(op, a, b):
    if op == "mod_max":
        big, small = max(a, b), min(a, b)
        return big % small if small else None
    return S.op_value(op, a, b)


def fact_ops(inp, out, opchars):
    """r.len/sign-based op candidates for one fact (per writeup table +
    our stage1 rules; concat checked structurally)."""
    sign = len(out) > 1 and out[0] == inp[2]
    body = out[1:] if sign else out
    if any(c in opchars for c in body):
        return []
    rl = len(body)
    cands = []
    for op, idx in S.CONCAT.items():
        if len(out) == 4 and all(out[j] == inp[i] for j, i in enumerate(idx)):
            cands.append(op)
    if sign:
        if rl in (1, 2):
            cands += ["sub_signed", "rsub_signed", "neg_absdiff"]
    else:
        if rl in (2, 3):
            cands += ["add", "add_p1", "add_m1"]
        if rl in (1, 2):
            cands += ["sub_signed", "rsub_signed", "absdiff", "mod_max"]
        if rl in (2, 3, 4):
            # leading-zero operands make 2-digit mul results real (05x17=85)
            cands += ["mul", "mul_p1", "mul_m1"]
        if rl == 1:
            cands += ["neg_absdiff"]  # zero case
    return cands


def decode(sym_pair, mode, asg):
    v = asg[sym_pair[0]] * 10 + asg[sym_pair[1]]
    return S.rev2(v) if mode == "rev_both" else v


def fact_check(inp, out, op, mode, asg):
    return S.check_example_full(inp, out, op, mode, asg) if op not in ("mod_max",) \
        else _check_modmax(inp, out, mode, asg)


def _check_modmax(inp, out, mode, asg):
    a1, a2, oc, b1, b2 = inp
    L = decode((a1, a2), mode, asg)
    R = decode((b1, b2), mode, asg)
    if min(L, R) == 0:
        return False
    v = max(L, R) % min(L, R)
    sign = len(out) > 1 and out[0] == oc
    if sign:
        return False
    body = out
    s = str(v).zfill(len(body))
    if len(s) != len(body):
        return False
    if mode == "rev_both":
        s = s[::-1]
    return all(asg.get(c) == int(d) for c, d in zip(body, s))


def fact_syms(inp, out):
    sign = len(out) > 1 and out[0] == inp[2]
    body = out[1:] if sign else out
    return [c for c in (inp[0], inp[1], inp[3], inp[4]) ] + list(body)


def anchor_candidates(inp, out, ops_for_char, mode, opchars):
    """All assignments of this fact's symbols consistent with some mul/add
    anchor op. Returns (op_used_set, [assignments]) or None if over cap."""
    syms = []
    for c in fact_syms(inp, out):
        if c not in syms:
            syms.append(c)
    res = []
    anchor_ops = [o for o in ops_for_char if o in
                  ("mul", "mul_p1", "mul_m1", "add", "add_p1", "add_m1")]
    if not anchor_ops:
        return None
    for av in range(100):
        for bv in range(100):
            for op in anchor_ops:
                v = op_val(op, av, bv)
                if v is None or v < 0:
                    continue
                # build assignment from the fact pattern
                a_s = f"{av:02d}"
                b_s = f"{bv:02d}"
                if mode == "rev_both":
                    a_s, b_s = a_s[::-1], b_s[::-1]
                sign = len(out) > 1 and out[0] == inp[2]
                body = out[1:] if sign else out
                if sign:
                    continue  # anchors here are non-negative ops
                r_s = str(v).zfill(len(body))
                if len(r_s) != len(body):
                    continue
                if mode == "rev_both":
                    r_s = r_s[::-1]
                asg = {}
                ok = True
                for c, d in zip((inp[0], inp[1], inp[3], inp[4]), a_s + b_s):
                    d = int(d)
                    if asg.get(c, d) != d:
                        ok = False
                        break
                    asg[c] = d
                if ok:
                    for c, d in zip(body, r_s):
                        d = int(d)
                        if asg.get(c, d) != d:
                            ok = False
                            break
                        asg[c] = d
                if not ok:
                    continue
                if len(set(asg.values())) != len(asg):
                    continue
                res.append((op, dict(asg)))
                if len(res) > LIST_CAP:
                    return "OVER"
    return res


def solve(prob, mode):
    exs, q, opchars, syms = S.parse(prob)
    by_char = defaultdict(list)
    for inp, out in exs:
        by_char[inp[2]].append((inp, out))
    if q[2] not in by_char:
        return "R:q-undemo"
    # per-char op candidates (intersect across the char's facts)
    char_ops = {}
    for c, facts in by_char.items():
        cand = None
        for inp, out in facts:
            fo = set(fact_ops(inp, out, opchars))
            cand = fo if cand is None else cand & fo
        if not cand:
            return "R:char-ops-empty"
        char_ops[c] = cand
    # concat gate (vfirst P1 semantics): if some rearrangement reproduces
    # EVERY fact of the query char, answer by rearrangement - checked FIRST
    qfacts = by_char[q[2]]
    for op, idx in S.CONCAT.items():
        if all(len(out) == 4 and all(out[j] == inp[i] for j, i in enumerate(idx))
               for inp, out in qfacts):
            return {"answer": "".join(q[i] for i in idx),
                    "anchor": "concat", "list": 0, "nodes": 0}
    # anchor selection: mul rlen4 > mul rlen3 > add rlen3
    best = None
    for k, (inp, out) in enumerate(exs):
        c = inp[2]
        ops = char_ops[c]
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        rl = len(body)
        score = None
        if any(o.startswith("mul") for o in ops) and rl == 4:
            score = 3
        elif any(o.startswith("mul") for o in ops) and rl == 3:
            score = 2
        elif any(o.startswith("add") for o in ops) and rl == 3:
            score = 1
        if score is not None and (best is None or score > best[0]):
            best = (score, k)
    if best is None:
        return "R:no-anchor"
    ak = best[1]
    ainp, aout = exs[ak]
    cands = anchor_candidates(ainp, aout, char_ops[ainp[2]], mode, opchars)
    if cands == "OVER":
        return "R:list-over"
    if cands is None or not cands:
        return "R:list-empty"
    # restrict anchor char's ops per candidate later; order remaining facts
    rest = [i for i in range(len(exs)) if i != ak]
    known = set(fact_syms(ainp, aout))
    order = sorted(rest, key=lambda i: len([s for s in fact_syms(*exs[i]) if s not in known]))
    nodes = [0]
    answers = set()

    def dfs(idx, asg, ops_pinned):
        if nodes[0] > DFS_CAP:
            return
        if idx == len(order):
            # all facts consistent -> answer the query
            qc = q[2]
            for op in sorted(ops_pinned.get(qc, char_ops[qc]), key=OP_PRIOR.index):
                if op in S.CONCAT:
                    continue
                qa = (q[0], q[1])
                qb = (q[3], q[4])
                if any(s not in asg for s in qa + qb):
                    continue
                L = decode(qa, mode, asg)
                R = decode(qb, mode, asg)
                v = op_val(op, L, R)
                if v is None:
                    continue
                s = str(abs(v))
                if mode == "rev_both":
                    s = s[::-1]
                inv = {d: c for c, d in asg.items()}
                if any(int(ch) not in inv for ch in s):
                    continue
                ans = (qc if v < 0 else "") + "".join(inv[int(ch)] for ch in s)
                answers.add(ans)
                return  # priority op picked
            return
        i = order[idx]
        inp, out = exs[i]
        c = inp[2]
        ops = ops_pinned.get(c, char_ops[c])
        unk = [s for s in dict.fromkeys(fact_syms(inp, out)) if s not in asg]
        free = [d for d in range(10) if d not in asg.values()]
        for op in sorted(ops, key=OP_PRIOR.index):
            if op in S.CONCAT:
                idxs = S.CONCAT[op]
                if len(out) == 4 and all(out[j] == inp[ii] for j, ii in enumerate(idxs)):
                    np_ = dict(ops_pinned)
                    np_[c] = {op}
                    dfs(idx + 1, asg, np_)
                continue
            for combo in product(free, repeat=len(unk)):
                if len(set(combo)) != len(combo):
                    continue
                nodes[0] += 1
                if nodes[0] > DFS_CAP:
                    return
                a2 = dict(asg)
                a2.update(zip(unk, combo))
                if fact_check(inp, out, op, mode, a2):
                    np_ = dict(ops_pinned)
                    np_[c] = {op}
                    dfs(idx + 1, a2, np_)

    for aop, asg in cands:
        dfs(0, dict(asg), {ainp[2]: {aop}})
    if not answers:
        return "R:dfs-noanswer"
    if len(answers) == 1:
        return {"answer": next(iter(answers)), "anchor": f"score{best[0]}",
                "list": len(cands), "nodes": nodes[0]}
    if answers:
        # priority over answers: pick the one from the highest-priority path —
        # approximation: lexicographic smallest for determinism here
        return {"answer": sorted(answers)[0], "anchor": f"score{best[0]}-multi",
                "list": len(cands), "nodes": nodes[0]}
    return None


def work(r):
    prob = {"id": r["id"], "examples": r["examples"], "question": r["question"]}
    err = None
    reasons = []
    for mode in ("standard", "rev_both"):
        try:
            res = solve(prob, mode)
        except Exception as e:
            res = None
            err = type(e).__name__
        if isinstance(res, str):
            reasons.append(res)
            res = None
        if res:
            res["mode"] = mode
            res["pid"] = r["id"]
            res["correct"] = res["answer"] == r["answer"]
            return res
    tag = ("ERR:" + err) if err else (reasons[-1] if reasons else "none")
    return {"pid": r["id"], "answer": None, "correct": False, "anchor": tag}


def _init():
    sys.path.insert(0, 'src')


if __name__ == "__main__":
    import multiprocessing as mp
    rows = [json.loads(l) for l in open("crypt_real_all.jsonl")
            if json.loads(l)["category"] == "cryptarithm_deduce"]
    out = []
    with mp.Pool(10, initializer=_init) as pool:
        for res in pool.imap_unordered(work, rows, chunksize=8):
            out.append(res)
            if len(out) % 100 == 0:
                print("progress", len(out), flush=True)
    json.dump(out, open("crypt_redesign_proto/anchor_results.json", "w"))
    from collections import Counter
    solved = [r for r in out if r.get("correct")]
    anc = Counter(r["anchor"] for r in out)
    anc_ok = Counter(r["anchor"] for r in solved)
    print(f"SOLVED {len(solved)}/659 = {len(solved)/659:.1%}  (writeup mul-only: 27.8%; our v2: 29.3%)")
    print("by anchor (solved/total):", {k: f"{anc_ok.get(k,0)}/{v}" for k, v in anc.items()})
    wrong = [r for r in out if r.get("answer") and not r["correct"]]
    print(f"wrong-emits {len(wrong)} (gold-filtered in training)")
    ls = sorted(r["list"] for r in solved if r.get("list"))
    nd = sorted(r["nodes"] for r in solved if r.get("nodes") is not None)
    if ls:
        print(f"anchor list size p50={ls[len(ls)//2]} p90={ls[int(len(ls)*.9)]}")
    if nd:
        print(f"dfs nodes p50={nd[len(nd)//2]} p90={nd[int(len(nd)*.9)]}")
