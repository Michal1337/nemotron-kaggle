"""Solver core for the verification-first cryptarithm_deduce narrator.

Vendored from the measured prototype (crypt_redesign_proto/_dim4_propagator.py,
audit workflow wf_b996f4a7-4a9): stage-1 op filtering by length/sign signatures,
no-branch units/tens/carry propagation, alldiff, bounded enumeration. Honest:
everything derives from THIS problem's examples+question; gold never read.
Used by reasoners.crypt_vfirst (the narrator). Do not edit casually - the
narrator's emitted lines must stay faithful to what these functions compute.
"""
import json, sys, time
from collections import defaultdict, Counter
from itertools import product

ADD_FAM = {"add": 0, "add_p1": 1, "add_m1": -1}
MUL_FAM = {"mul": 0, "mul_p1": 1, "mul_m1": -1}
SUB_FAM = {"sub_signed", "rsub_signed", "absdiff", "neg_absdiff"}
CONCAT = {"concat_fwd": (0, 1, 3, 4), "concat_rev": (3, 4, 0, 1),
          "concat_fwd_ro": (1, 0, 4, 3), "concat_rev_ro": (4, 3, 1, 0)}
POOL = list(ADD_FAM) + list(MUL_FAM) + sorted(SUB_FAM) + list(CONCAT)

def op_value(op, L, R):
    if op in ADD_FAM: return L + R + ADD_FAM[op]
    if op in MUL_FAM: return L * R + MUL_FAM[op]
    if op == "sub_signed": return L - R
    if op == "rsub_signed": return R - L
    if op == "absdiff": return abs(L - R)
    if op == "neg_absdiff": return -abs(L - R)
    return None

def rev2(x): return (x % 10) * 10 + x // 10

# ---------------- parsing ----------------
def parse(prob):
    exs = [(e["input_value"], e["output_value"]) for e in prob["examples"]]
    q = prob["question"]
    opchars = {i[2] for i, _ in exs} | {q[2]}
    syms = set()
    for inp, out in exs:
        syms |= {inp[0], inp[1], inp[3], inp[4]}
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        syms |= set(body)
    syms |= {q[0], q[1], q[3], q[4]}
    syms -= opchars
    return exs, q, opchars, sorted(syms)

# ---------------- stage 1 ----------------
def stage1_char(opchar, exlist, opchars):
    """exlist: [(inp,out)] for this op char. Returns surviving candidate ops."""
    cands = []
    for op in POOL:
        ok = True
        for inp, out in exlist:
            if op in CONCAT:
                idx = CONCAT[op]
                if len(out) != 4 or any(out[k] != inp[i] for k, i in enumerate(idx)):
                    ok = False; break
                continue
            sign = len(out) > 1 and out[0] == inp[2]
            body = out[1:] if sign else out
            if any(c in opchars for c in body): ok = False; break
            rl = len(body)
            if sign and op not in ("sub_signed", "rsub_signed", "neg_absdiff"):
                ok = False; break
            if op == "neg_absdiff" and not sign:
                # only v==0 -> single digit 0
                if rl != 1: ok = False; break
            if op in ADD_FAM and rl not in (2, 3): ok = False; break
            if op in ("mul", "mul_p1") and rl not in (3, 4): ok = False; break
            if op == "mul_m1" and rl not in (2, 3, 4): ok = False; break
            if op in ("sub_signed", "rsub_signed", "absdiff") and rl not in (1, 2):
                ok = False; break
        if ok: cands.append(op)
    return cands

# ---------------- no-branch propagator ----------------
class Dead(Exception): pass

def project_column(domains, pinned, col_syms, rel, cin_set):
    """col_syms: list of (role) symbols participating; rel(digit_tuple, cin)->
    (ok, cout). Enumerate domain product with alldiff; returns per-sym supports,
    cout set. Counts as ONE column constraint application."""
    uniq = []
    for s in col_syms:
        if s not in uniq: uniq.append(s)
    doms = [sorted(domains[s]) for s in uniq]
    sup = {s: set() for s in uniq}
    couts = set()
    found = False
    for tup in product(*doms):
        # bijection: distinct symbols -> distinct digits
        if len(set(tup)) != len(tup): continue
        assign = dict(zip(uniq, tup))
        ds = tuple(assign[s] for s in col_syms)
        for cin in cin_set:
            r = rel(ds, cin)
            if r is None: continue
            found = True
            couts.add(r)
            for s in uniq: sup[s].add(assign[s])
    if not found: raise Dead
    changed = False
    for s in uniq:
        nd = domains[s] & sup[s]
        if not nd: raise Dead
        if nd != domains[s]: domains[s] = nd; changed = True
    return changed, couts

def apply_alldiff(domains, steps):
    changed = True; any_change = False
    syms = sorted(domains)
    full = len(syms) >= 10
    while changed:
        changed = False
        pinned = {s: next(iter(domains[s])) for s in syms if len(domains[s]) == 1}
        used = {}
        for s, d in pinned.items():
            if d in used and used[d] != s: raise Dead
            used[d] = s
        for s in syms:
            if s in pinned: continue
            nd = domains[s] - set(used)
            if not nd: raise Dead
            if nd != domains[s]:
                domains[s] = nd; changed = True; any_change = True; steps[0] += 1
        if full:
            # digit-side: a digit possible for exactly one symbol -> pin
            where = defaultdict(list)
            for s in syms:
                for d in domains[s]: where[d].append(s)
            for d, ss in where.items():
                if len(ss) == 1 and len(domains[ss[0]]) > 1:
                    domains[ss[0]] = {d}; changed = True; any_change = True; steps[0] += 1
            if len(syms) == 10:
                for d in range(10):
                    if d not in where: raise Dead
    return any_change

def example_units_tens(inp, out, op, mode, domains, evars, steps, forced=None):
    """Apply single-column constraints for one numeric example. evars: persistent
    dict for this example's hidden vars ('c1' carry/borrow set, 'dir' set).
    forced: optional dict overriding hidden var sets (for the one-branch split)."""
    a1, a2, oc, b1, b2 = inp
    sign = len(out) > 1 and out[0] == oc
    body = out[1:] if sign else out
    rl = len(body)
    # natural-order result symbols (most significant first)
    nat = body[::-1] if mode == "rev_both" else body
    # used operand digit symbols (tens, units)
    if mode == "rev_both": A = (a2, a1); B = (b2, b1)
    else: A = (a1, a2); B = (b1, b2)
    ch = False
    if op in ADD_FAM:
        d = ADD_FAM[op]
        c1s = forced.get("c1") if forced else evars.setdefault("c1", {-1, 0, 1})
        c2_allowed = {0, 1}
        if rl == 2: c2_allowed = {0}
        elif rl >= 3:
            c2_allowed = set(domains[nat[-3]]) & {0, 1}
            if not c2_allowed: raise Dead
            for k in range(rl - 3):  # zpad zeros above
                if 0 not in domains[nat[k]]: raise Dead
                if domains[nat[k]] != {0}:
                    domains[nat[k]] = {0}; ch = True; steps[0] += 1
        # units column
        def rel_u(ds, cin):
            u = ds[0] + ds[1] + d
            if u % 10 != ds[2]: return None
            return u // 10
        c, c1out = project_column(domains, None, [A[1], B[1], nat[-1]], rel_u, {0})
        ch |= c; steps[0] += c
        c1s2 = (c1s & c1out) if isinstance(c1s, set) else c1out
        if not c1s2: raise Dead
        # tens column
        def rel_t(ds, cin):
            t = ds[0] + ds[1] + cin
            if t % 10 != ds[2]: return None
            co = t // 10
            return co if co in c2_allowed else None
        c, c2out = project_column(domains, None, [A[0], B[0], nat[-2]], rel_t, c1s2)
        ch |= c; steps[0] += c
        if rl >= 3:
            nd = domains[nat[-3]] & c2out
            if not nd: raise Dead
            if nd != domains[nat[-3]]: domains[nat[-3]] = nd; ch = True; steps[0] += 1
        if not forced: evars["c1"] = c1s2
    elif op in SUB_FAM:
        # resolve direction: X - Y = |v|
        if op == "sub_signed": dirs = {1} if sign else {0}      # 0: L-R>=0, 1: R-L
        elif op == "rsub_signed": dirs = {0} if sign else {1}
        else:
            dirs = forced.get("dir") if forced and "dir" in forced else evars.setdefault("dir", {0, 1})
            if op == "neg_absdiff" and not sign:
                # v == 0 -> operands identical digit-wise: needs same symbols
                if A != B: raise Dead
                if rl != 1 or 0 not in domains[nat[0]]: raise Dead
                if domains[nat[0]] != {0}: domains[nat[0]] = {0}; ch = True; steps[0] += 1
                return ch
        b1s = forced.get("c1") if forced and "c1" in forced else evars.setdefault("c1", {0, 1})
        for k in range(rl - 2):  # zpad zeros above 2 digits
            if 0 not in domains[nat[k]]: raise Dead
            if domains[nat[k]] != {0}: domains[nat[k]] = {0}; ch = True; steps[0] += 1
        sup_dirs = set()
        agg_b1 = set()
        # apply per direction, union supports (single constraint w/ hidden dir)
        units_syms = lambda dr: ([A[1], B[1]] if dr == 0 else [B[1], A[1]])
        tens_syms = lambda dr: ([A[0], B[0]] if dr == 0 else [B[0], A[0]])
        # units: x0 - y0 = r_u - 10*b ; tens: x1 - y1 - b = r_t (>=0, ==0 if rl==1)
        usup = defaultdict(set); tsup = defaultdict(set)
        u_syms_all = sorted({A[1], B[1], nat[-1]})
        t_par = [A[0], B[0]] + ([nat[-2]] if rl >= 2 else [])
        t_syms_all = sorted(set(t_par))
        for dr in sorted(dirs):
            xs, ys = units_syms(dr)
            ru = nat[-1]
            uniq = []
            for s in [xs, ys, ru]:
                if s not in uniq: uniq.append(s)
            ok_dir = False
            loc_b1 = set()
            for tup in product(*[sorted(domains[s]) for s in uniq]):
                if len(set(tup)) != len(tup): continue
                asg = dict(zip(uniq, tup))
                u = asg[xs] - asg[ys]
                b = 1 if u < 0 else 0
                if (u + 10 * b) != asg[ru]: continue
                if b not in b1s: continue
                # tens feasibility check happens in tens pass; record
                ok_dir = True; loc_b1.add(b)
                for s in uniq: usup[s].add(asg[s])
            if not ok_dir: continue
            xt, yt = tens_syms(dr)
            rt = nat[-2] if rl >= 2 else None
            uniq2 = []
            for s in ([xt, yt] + ([rt] if rt else [])):
                if s not in uniq2: uniq2.append(s)
            ok2 = False
            for tup in product(*[sorted(domains[s]) for s in uniq2]):
                if len(set(tup)) != len(tup): continue
                asg = dict(zip(uniq2, tup))
                for b in sorted(loc_b1):
                    t = asg[xt] - asg[yt] - b
                    if t < 0: continue
                    if rl == 1:
                        if t != 0: continue
                    else:
                        if t != asg[rt]: continue
                    ok2 = True; agg_b1.add(b)
                    for s in uniq2: tsup[s].add(asg[s])
            if ok2: sup_dirs.add(dr)
        if not sup_dirs: raise Dead
        for s, sp in list(usup.items()) + list(tsup.items()):
            nd = domains[s] & sp
            if not nd: raise Dead
            if nd != domains[s]: domains[s] = nd; ch = True; steps[0] += 1
        if not forced:
            if op in ("absdiff", "neg_absdiff"): evars["dir"] = sup_dirs
            evars["c1"] = agg_b1 if agg_b1 else b1s
    elif op in MUL_FAM:
        d = MUL_FAM[op]
        # units anchor: (a0*b0 + d) mod 10 == units digit
        def rel_u(ds, cin):
            return 0 if (ds[0] * ds[1] + d) % 10 == ds[2] else None
        c, _ = project_column(domains, None, [A[1], B[1], nat[-1]], rel_u, {0})
        ch |= c; steps[0] += c
        # magnitude/leading bounds: v in [lo, hi)
        lead_dom = domains[nat[0]]
        lo = 10 ** (rl - 1) if 0 not in lead_dom else 0
        hi = 10 ** rl
        Amin = 10 * min(domains[A[0]]) + min(domains[A[1]])
        Amax = 10 * max(domains[A[0]]) + max(domains[A[1]])
        Bmin = 10 * min(domains[B[0]]) + min(domains[B[1]])
        Bmax = 10 * max(domains[B[0]]) + max(domains[B[1]])
        # prune tens digits pairwise by interval arithmetic
        for (P, Q, Qmin, Qmax) in ((A, B, Bmin, Bmax), (B, A, Amin, Amax)):
            keep = set()
            for dp in sorted(domains[P[0]]):
                pmin = 10 * dp + min(domains[P[1]])
                pmax = 10 * dp + max(domains[P[1]])
                if pmax * Qmax + d >= lo and pmin * Qmin + d < hi:
                    keep.add(dp)
            nd = domains[P[0]] & keep
            if not nd: raise Dead
            if nd != domains[P[0]]: domains[P[0]] = nd; ch = True; steps[0] += 1
    return ch

def propagate(exs, opchars, syms, ops, mode, max_passes=8, init=None, forced_map=None):
    """Returns (domains, evars, steps) or raises Dead."""
    domains = {s: set(init[s]) for s in syms} if init else {s: set(range(10)) for s in syms}
    evars = [dict() for _ in exs]
    steps = [0]
    for p in range(max_passes):
        ch = False
        try:
            ch |= apply_alldiff(domains, steps)
        except Dead: raise
        for k, (inp, out) in enumerate(exs):
            op = ops[inp[2]]
            if op in CONCAT:
                idx = CONCAT[op]
                if len(out) != 4 or any(out[j] != inp[i] for j, i in enumerate(idx)):
                    raise Dead
                continue
            forced = forced_map.get(k) if forced_map else None
            ch |= example_units_tens(inp, out, op, mode, domains, evars[k], steps, forced)
        ch |= apply_alldiff(domains, steps)
        if not ch: break
    return domains, evars, steps[0]

# ---------------- answer encoding ----------------
def try_answer(q, ops, mode, domains, syms):
    """Return answer string if pinned, else None."""
    a1, a2, oc, b1, b2 = q
    op = ops[oc]
    if op in CONCAT:
        idx = CONCAT[op]
        return "".join(q[i] for i in idx)
    for s in (a1, a2, b1, b2):
        if len(domains[s]) != 1: return None
    L = next(iter(domains[a1])) * 10 + next(iter(domains[a2]))
    R = next(iter(domains[b1])) * 10 + next(iter(domains[b2]))
    if mode == "rev_both": L, R = rev2(L), rev2(R)
    v = op_value(op, L, R)
    if v is None: return None
    s = str(abs(v))
    if mode == "rev_both": s = s[::-1]
    inv = {}
    for sym in syms:
        if len(domains[sym]) == 1: inv[next(iter(domains[sym]))] = sym
    out = []
    for c in s:
        if int(c) not in inv: return None
        out.append(inv[int(c)])
    return (oc if v < 0 else "") + "".join(out)

def full_pin(domains):
    return all(len(d) == 1 for d in domains.values())

# ---------------- full enumeration (complete semantics) ----------------
def check_example_full(inp, out, op, mode, mp):
    a1, a2, oc, b1, b2 = inp
    L = mp[a1] * 10 + mp[a2]; R = mp[b1] * 10 + mp[b2]
    if mode == "rev_both": L, R = rev2(L), rev2(R)
    if op in CONCAT:
        idx = CONCAT[op]
        return len(out) == 4 and all(out[j] == inp[i] for j, i in enumerate(idx))
    v = op_value(op, L, R)
    if v is None: return False
    signed = op in ("sub_signed", "rsub_signed", "neg_absdiff")
    sign = signed and len(out) > 1 and out[0] == oc
    if signed and (v < 0) != sign: return False
    if not signed and v < 0: return False
    body = out[1:] if sign else out
    s = str(abs(v)).zfill(len(body))
    if len(s) != len(body): return False
    if mode == "rev_both": s = s[::-1]
    return all(mp[c] == int(d) for c, d in zip(body, s))

def enumerate_answers(exs, q, ops, mode, domains, syms, cap_nodes=250000, cap_ans=64):
    """Backtracking over syms within propagated domains; full example check;
    returns (set of answers incl 'NOSYM', capped_flag)."""
    order = sorted(syms, key=lambda s: (len(domains[s]), s))
    exneed = []
    for inp, out in exs:
        need = {inp[0], inp[1], inp[3], inp[4]}
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        need |= set(body)
        need &= set(syms)
        exneed.append(need)
    pos_of = {s: i for i, s in enumerate(order)}
    ex_ready = [max(pos_of[s] for s in need) if need else -1 for need in exneed]
    answers = set(); nodes = [0]; capped = [False]
    mp = {}
    def bt(i, used):
        if nodes[0] > cap_nodes or len(answers) >= cap_ans:
            capped[0] = True; return
        if i == len(order):
            ans = answer_of(q, ops, mode, mp)
            answers.add(ans); return
        s = order[i]
        for d in sorted(domains[s]):
            if d in used: continue
            nodes[0] += 1
            mp[s] = d
            ok = True
            for k, (inp, out) in enumerate(exs):
                if ex_ready[k] == i:
                    if not check_example_full(inp, out, ops[inp[2]], mode, mp):
                        ok = False; break
            if ok: bt(i + 1, used | {d})
            del mp[s]
    def answer_of(q, ops, mode, mp):
        a1, a2, oc, b1, b2 = q
        op = ops[oc]
        if op in CONCAT:
            return "".join(q[i] for i in CONCAT[op])
        L = mp[a1] * 10 + mp[a2]; R = mp[b1] * 10 + mp[b2]
        if mode == "rev_both": L, R = rev2(L), rev2(R)
        v = op_value(op, L, R)
        s = str(abs(v))
        if mode == "rev_both": s = s[::-1]
        inv = {d: c for c, d in mp.items()}
        if any(int(c) not in inv for c in s): return "NOSYM"
        return (oc if v < 0 else "") + "".join(inv[int(c)] for c in s)
    bt(0, set())
    return answers, capped[0]

# ---------------- per-problem classification ----------------
def classify(prob):
    exs, q, opchars, syms = parse(prob)
    res = {"pid": prob["id"], "n_ex": len(exs), "n_sym": len(syms),
           "n_opchar": len(opchars)}
    by_char = defaultdict(list)
    for inp, out in exs: by_char[inp[2]].append((inp, out))
    if q[2] not in by_char:
        res["class"] = "query-op-undemonstrated"; return res
    cands = {}
    s1_steps = 0
    for c, exl in sorted(by_char.items()):
        cl = stage1_char(c, exl, opchars)
        s1_steps += len(exl) + len(POOL)
        if not cl:
            res["class"] = "outside-pool"; res["s1_dead_char"] = c; return res
        cands[c] = cl
    chars = sorted(cands)
    combos = []
    seen = set()
    for tup in product(*[cands[c] for c in chars]):
        for mode in ("standard", "rev_both"):
            ops = dict(zip(chars, tup))
            key = (tup, "NA" if all(o in CONCAT for o in tup) else mode)
            if key in seen: continue
            seen.add(key)
            combos.append((ops, mode))
    res["n_combos_s1"] = len(combos)
    res["s1_steps"] = s1_steps
    if len(combos) > 6000:
        res["class"] = "combo-explosion"; return res
    survivors = []
    total_steps = 0
    for ops, mode in combos:
        try:
            domains, evars, st = propagate(exs, opchars, syms, ops, mode)
            total_steps += st
            survivors.append((ops, mode, domains, evars, st))
        except Dead:
            total_steps += 1
            continue
    res["n_survivors"] = len(survivors)
    res["s2_steps_total"] = total_steps
    if not survivors:
        res["class"] = "outside-pool"; return res
    # zero-branch?
    answers = []
    pin_full = 0
    for ops, mode, domains, evars, st in survivors:
        a = try_answer(q, ops, mode, domains, syms)
        answers.append(a)
        if full_pin(domains): pin_full += 1
    res["n_pinned_survivors"] = sum(a is not None for a in answers)
    res["n_fullpin_survivors"] = pin_full
    res["gold_steps"] = max(s[4] for s in survivors)
    if all(a is not None for a in answers) and len(set(answers)) == 1:
        res["class"] = "zero-branch"; res["pred"] = answers[0]
        res["fullpin"] = pin_full == len(survivors)
        return res
    # one-branch?
    ob_answers = []
    ob_ok = True
    for (ops, mode, domains, evars, st), a in zip(survivors, answers):
        if a is not None:
            ob_answers.append(a); continue
        fixed = None
        split_candidates = []
        for k, ev in enumerate(evars):
            for var, vals in sorted(ev.items()):
                if len(vals) == 2: split_candidates.append((k, var, sorted(vals)))
        for k, var, vals in split_candidates:
            branch_ans = set(); ok = True
            for val in vals:
                try:
                    init = {s: set(d) for s, d in domains.items()}
                    d2, e2, st2 = propagate(exs, opchars, syms, ops, mode,
                                            init=init, forced_map={k: {var: {val}}})
                    a2 = try_answer(q, ops, mode, d2, syms)
                    if a2 is None: ok = False; break
                    branch_ans.add(a2)
                except Dead:
                    continue
            if ok and len(branch_ans) == 1:
                fixed = next(iter(branch_ans)); break
        if fixed is None: ob_ok = False; break
        ob_answers.append(fixed)
    if ob_ok and ob_answers and len(set(ob_answers)) == 1:
        res["class"] = "one-branch"; res["pred"] = ob_answers[0]
        return res
    # deeper: full enumeration
    all_ans = set(); capped = False
    for ops, mode, domains, evars, st in survivors:
        ans, cp = enumerate_answers(exs, q, ops, mode, domains, syms)
        all_ans |= ans; capped |= cp
        if len(all_ans) > 8: break
    all_ans.discard(None)
    res["n_distinct_answers"] = len(all_ans)
    res["capped"] = capped
    if capped and len(all_ans) <= 1:
        res["class"] = "capped-unknown"
    elif len(all_ans) == 0:
        res["class"] = "outside-pool"
    elif len(all_ans) == 1:
        res["class"] = "deeper-search"; res["pred"] = next(iter(all_ans))
    else:
        res["class"] = "under-determined"
        res["answer_set_sample"] = sorted(all_ans)[:6]
    return res
