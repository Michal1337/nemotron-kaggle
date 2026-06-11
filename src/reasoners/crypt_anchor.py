"""Anchor-DFS narrator for cryptarithm_deduce (v5 render of the v4 design).

v5 render changes (pilot forensics 2026-06-11: ep1-6 = 0/52 arithmetic,
9-12/25 concat): single glyph table + one line per fact (old pair lines were
57% of miss causes via one-atom slips); concat gate ALWAYS rendered with a
failing witness on arithmetic problems (11 hallucinated-gate escapes); concat
apply spelled per position (9 byte-perfect prefixes lost a one-shot
permutation). Candidate-list derivability is the remaining open design issue
(0/620 gold rows reproduced at pilot scale).

Skeleton: remap (per-position, Cyrillic digit atoms - 1 BPE token each on the
Nemotron tokenizer) -> per-op-char narrowing by result length/sign -> concat
gate -> MUL ANCHOR (the most-constraining fact; its candidate (a,b,r) list is
a finite memorizable lookup, capped at LIST_CAP rows, each row self-verifiable
arithmetic) -> labeled DFS over remaining facts ordered by fewest new symbols,
each step a tiny visible algebra block (<=2 unknowns: substitute knowns, solve
linearly or case-split over free digits) -> query with printed priority pick ->
inverse remap -> boxed.

Measured basis (2026-06-11): mul-rlen4 anchors solve 84/87 when the list fits;
lists p50=10 p90=22; concat 59/59; total 168/659 with ~95% arithmetic
precision. Gold never read; the external filter is the only consumer.
Modes ({none,swap} = standard/rev_both) run sequentially: the standard list is
narrated and searched first; the swap list only appears if standard yields
nothing - so a resolving standard reading never pays for swap narration.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Optional

from reasoners.store_types import Problem
import reasoners.crypt_vfirst_solver as S

LIST_CAP = 24
TOK_CAP = 7300
# Measured on 180 real v4 CoTs (anchor_tokcal): chars/tok 1.44-2.07, p50 1.89.
CHARS_PER_TOK = 1.40       # fallback estimate when no tokenizer is available
RUNAWAY_CHARS_PER_TOK = 2.1  # in-flight bound: above any observed ratio, so it
                             # only kills CoTs that cannot fit at ANY ratio
TOK_OVERHEAD = 100

DIGIT_ATOMS = "БГДЖИЛМПТФ"          # 10 digit symbols, 1 token each
PATTERN_ATOMS = "УХЦЧШЩЭЮ"          # canonical pattern letters for the lookup
BRANCH_LABELS = "RKMNOPQVW"          # per-DFS-depth state labels

OP_PRIOR = ["add", "mul", "concat_fwd", "concat_rev", "concat_fwd_ro", "concat_rev_ro",
            "absdiff", "add_p1", "add_m1", "mul_p1", "mul_m1",
            "sub_signed", "rsub_signed", "neg_absdiff", "mod_max"]
OP_SHOW = {"add": "a+b", "add_p1": "a+b+1", "add_m1": "a+b-1",
           "mul": "a*b", "mul_p1": "a*b+1", "mul_m1": "a*b-1",
           "sub_signed": "a-b", "rsub_signed": "b-a", "absdiff": "|a-b|",
           "neg_absdiff": "-|a-b|", "mod_max": "max mod min"}


def op_val(op, a, b):
    if op == "mod_max":
        big, small = max(a, b), min(a, b)
        return big % small if small else None
    return S.op_value(op, a, b)


def fact_ops(inp, out, opchars):
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
            cands += ["mul", "mul_p1", "mul_m1"]
        if rl == 1:
            cands += ["neg_absdiff"]
    return cands


def fact_syms(inp, out):
    sign = len(out) > 1 and out[0] == inp[2]
    body = out[1:] if sign else out
    return list(inp[0] + inp[1] + inp[3] + inp[4]) + list(body)


def fact_check(inp, out, op, mode, asg):
    if op == "mod_max":
        a1, a2, oc, b1, b2 = inp
        L = asg[a1] * 10 + asg[a2]
        R = asg[b1] * 10 + asg[b2]
        if mode == "rev_both":
            L, R = S.rev2(L), S.rev2(R)
        if min(L, R) == 0:
            return False
        v = max(L, R) % min(L, R)
        if len(out) > 1 and out[0] == oc:
            return False
        s = str(v).zfill(len(out))
        if len(s) != len(out):
            return False
        if mode == "rev_both":
            s = s[::-1]
        return all(asg.get(c) == int(d) for c, d in zip(out, s))
    return S.check_example_full(inp, out, op, mode, asg)


def anchor_candidates(inp, out, anchor_ops, opchars, mode):
    """(op, assignment, a, b, v) rows for ONE mode; 'OVER' past LIST_CAP.
    Results are zero-padded to the fact's length (leading-zero result digits
    are legal, same as the leading-zero operands already admitted)."""
    sign = len(out) > 1 and out[0] == inp[2]
    if sign:
        return []
    body = out
    res = []
    for av in range(100):
        for bv in range(100):
            for op in anchor_ops:
                v = op_val(op, av, bv)
                if v is None or v < 0:
                    continue
                a_s, b_s = f"{av:02d}", f"{bv:02d}"
                if mode == "rev_both":
                    a_s, b_s = a_s[::-1], b_s[::-1]
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
                if not ok or len(set(asg.values())) != len(asg):
                    continue
                res.append((op, dict(asg), av, bv, v))
                if len(res) > LIST_CAP:
                    return "OVER"
    return res


def op_sym(op):
    return "*" if op.startswith("mul") else "+"


def op_suffix(op):
    return "+1" if op.endswith("_p1") else "-1" if op.endswith("_m1") else ""


_TOKENIZER = None


def tok_count(text):
    """Exact Nemotron token count; falls back to chars/1.40 if no tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import os
        try:
            from tokenizers import Tokenizer
            p = os.environ.get(
                "NEMOTRON_TOKENIZER_JSON",
                os.path.join(os.path.dirname(__file__), "..", "..",
                             "runs", "baseline", "tokenizer.json"))
            _TOKENIZER = Tokenizer.from_file(p)
        except Exception:
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER.encode(text).ids)
    return int(len(text) / CHARS_PER_TOK)


class Emitter:
    def __init__(self):
        self.lines = []
        self.chars = 0

    def emit(self, line=""):
        self.lines.append(line)
        self.chars += len(line) + 1

    def over(self):
        # in-flight runaway bound only; the exact check happens at return
        return TOK_OVERHEAD + self.chars / RUNAWAY_CHARS_PER_TOK > TOK_CAP

    def over_exact(self):
        return TOK_OVERHEAD + tok_count("\n".join(self.lines)) > TOK_CAP


class Abort(Exception):
    pass


LAST = {"reason": None}  # debug: why the most recent call abstained


def _why(tag):
    LAST["reason"] = tag


def _asg_str(asg, ren):
    return " ".join(f"{ren[c]}{d}" for c, d in sorted(asg.items(), key=lambda kv: kv[1]))


def reasoning_cryptarithm_anchor(problem: Problem) -> Optional[str]:
    prob = {"id": problem.id,
            "examples": [{"input_value": str(e.input_value),
                          "output_value": str(e.output_value)} for e in problem.examples],
            "question": str(problem.question)}
    try:
        return _narrate(prob)
    except (Abort, S.Dead, KeyError, IndexError, ValueError):
        return None


def _narrate(prob):
    LAST.clear()
    LAST["reason"] = None
    exs = [(e["input_value"], e["output_value"]) for e in prob["examples"]]
    q = prob["question"]
    if len(q) != 5 or any(len(i) != 5 for i, _ in exs):
        _why("shape")
        return None
    opchars = {i[2] for i, _ in exs} | {q[2]}
    em = Emitter()
    em.emit("We infer the transformation rule from the facts by bounded rule-based trial.")
    em.emit("I will put my final answer inside \\boxed{}.")
    em.emit()

    # ---- remap (per-position pairs; Cyrillic atoms, 1 token each) ----
    ren = {}

    def see(c):
        if c in opchars or c in ren:
            return
        if len(ren) >= 10:
            _why("syms>10")
            raise Abort
        ren[c] = DIGIT_ATOMS[len(ren)]

    # v5: single glyph table + one line per fact. The old 3-line-per-fact
    # per-position pair render was the dominant failure surface in the pilot
    # (57% of misses = one atom pair swapped in those lines, carried
    # consistently). Fewer glyph restatements = fewer slip opportunities.
    by_char = defaultdict(list)
    lexs = []
    for inp, out in exs:
        sign = len(out) > 1 and out[0] == inp[2]
        body = out[1:] if sign else out
        for c in (inp[0], inp[1], inp[3], inp[4]) + tuple(body):
            see(c)
    for c in (q[0], q[1], q[3], q[4]):
        see(c)
    em.emit("remap(pos2 is op): " + " ".join(f"【{c}】={a}" for c, a in ren.items()))
    for k, (inp, out) in enumerate(exs):
        sign = len(out) > 1 and out[0] == inp[2]
        body = out[1:] if sign else out
        lin = "".join(ren.get(c, c) for c in (inp[0], inp[1])) + f"【{inp[2]}】" + \
              "".join(ren.get(c, c) for c in (inp[3], inp[4]))
        lout = ("-" if sign else "") + "".join(ren[c] for c in body)
        syms = sorted(set(body) | {inp[0], inp[1], inp[3], inp[4]}, key=lambda c: ren[c])
        em.emit(f"fact{k}【{inp}】=【{out}】 -> {lin} = {lout}  "
                f"r.len={len(body)}{' neg' if sign else ''} sym=[{','.join(ren[c] for c in syms)}]")
        by_char[inp[2]].append((k, inp, out))
        lexs.append((inp, out))
    em.emit(f"Query【{q}】 -> " + "".join(ren.get(c, c) for c in (q[0], q[1]))
            + f"【{q[2]}】" + "".join(ren.get(c, c) for c in (q[3], q[4])))
    em.emit()
    if q[2] not in by_char:
        _why("q-undemo")
        return None

    # ---- concat gate (FIRST; v5: ALWAYS rendered, with a failing witness on
    # the NO path - the pilot showed 11 std/swap generations hallucinating a
    # concat gate because the NO case was silent/undemonstrated) ----
    qfacts = [(inp, out) for _, inp, out in by_char[q[2]]]
    hit = None
    for op, idx in S.CONCAT.items():
        if all(len(out) == 4 and all(out[j] == inp[i] for j, i in enumerate(idx))
               for inp, out in qfacts):
            hit = (op, idx)
            break
    if hit is not None:
        op, idx = hit
        em.emit(f"concat check【{q[2]}】: every fact's result is its operand symbols "
                f"rearranged ({op}) - no digit code needed:")
        for inp, out in qfacts:
            em.emit(f"  【{inp}】=【{out}】 ok")
        # per-position apply scaffold (the pilot lost 9 byte-perfect prefixes
        # on this step as a one-shot permutation; spell it out per position)
        em.emit(f"apply to Query【{q}】 ({op}):")
        for j, i in enumerate(idx):
            em.emit(f"  out{j} = q{i} = {q[i]}")
        ans = "".join(q[i] for i in idx)
        em.emit(f"answer: {ans}")
        em.emit("I will now return the answer in \\boxed{}")
        em.emit(f"The answer in \\boxed{{–}} is \\boxed{{{ans}}}")
        if em.over_exact():
            _why("governor-final")
            raise Abort
        return "\n".join(em.lines)
    # NO witness: a short result kills all 4 rearrangement ops at once;
    # otherwise name the first mismatch per op
    short = next(((inp, out) for inp, out in qfacts if len(out) != 4), None)
    if short is not None:
        inp, out = short
        em.emit(f"concat check【{q[2]}】: fact result【{out}】 has {len(out)} symbols, "
                f"not a rearrangement of the 4 operand symbols - no")
    else:
        for op, idx in S.CONCAT.items():
            inp, out = next((inp, out) for inp, out in qfacts
                            if any(out[j] != inp[i] for j, i in enumerate(idx)))
            j = next(j for j, i in enumerate(idx) if out[j] != inp[i])
            em.emit(f"concat check【{q[2]}】({op}): 【{inp}】=【{out}】 pos{j} "
                    f"{out[j]}≠{inp[idx[j]]} - no")

    # ---- narrowing (per op char) ----
    em.emit("narrowing:")
    char_ops = {}
    for c in sorted(by_char):
        facts = [(inp, out) for _, inp, out in by_char[c]]
        cand = None
        for inp, out in facts:
            fo = set(fact_ops(inp, out, opchars))
            cand = fo if cand is None else cand & fo
        if not cand:
            _why("char-ops-empty")
            return None
        char_ops[c] = cand
        rls = ",".join(str(len(out[1:] if (len(out) > 1 and out[0] == inp[2]) else out))
                       for inp, out in facts)
        shown = sorted((o for o in cand if o not in S.CONCAT), key=OP_PRIOR.index)
        em.emit(f"【{c}】 facts r.len={rls} -> 《{','.join(OP_SHOW.get(o, o) for o in shown)}》")
    em.emit("pre: 《standard,swap》 - standard first, swap only if standard yields nothing")
    em.emit()

    # ---- anchor selection ----
    best = None
    for k, (inp, out) in enumerate(lexs):
        ops = char_ops[inp[2]]
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        rl = len(body)
        mul_ok = any(o.startswith("mul") for o in ops)
        score = 3 if (mul_ok and rl == 4) else 2 if (mul_ok and rl == 3) else None
        if score is not None and (best is None or score > best[0]):
            best = (score, k)
    if best is None:
        _why("no-anchor")
        return None
    ak = best[1]
    ainp, aout = lexs[ak]
    anchor_ops = sorted((o for o in char_ops[ainp[2]]
                         if o.startswith("mul") or o.startswith("add")), key=OP_PRIOR.index)

    # canonical pattern (the memorizable lookup key)
    pat = {}
    for c in fact_syms(ainp, aout):
        if c not in opchars and c not in pat:
            pat[c] = PATTERN_ATOMS[len(pat)] if len(pat) < len(PATTERN_ATOMS) else None
    if any(v is None for v in pat.values()):
        _why("pattern-atoms")
        return None
    lpat = "".join(pat.get(c, c) for c in (ainp[0], ainp[1])) + "?" + \
           "".join(pat.get(c, c) for c in (ainp[3], ainp[4])) + "=" + \
           "".join(pat[c] for c in aout)
    em.emit(f"anchor: fact{ak} (most constrained, ops 《{','.join(OP_SHOW[o] for o in anchor_ops)}》)")

    # ---- DFS over remaining facts ----
    rest = [i for i in range(len(lexs)) if i != ak]
    order = list(rest)
    answers = []

    def alg_block(prefix, inp, out, op, mode, asg, unk, free):
        """Visible check/solve for one fact under one op. Returns list of
        (new_asg) accepted, emitting the algebra."""
        ops_show = OP_SHOW.get(op, op)
        sign = len(out) > 1 and out[0] == inp[2]
        body = out[1:] if sign else out
        acc = []
        if not unk:
            ok = fact_check(inp, out, op, mode, asg)
            L = asg[inp[0]] * 10 + asg[inp[1]]
            R = asg[inp[3]] * 10 + asg[inp[4]]
            if mode == "rev_both":
                L, R = S.rev2(L), S.rev2(R)
            v = op_val(op, L, R)
            em.emit(f"{prefix}{ops_show}: {L}?{R} -> {v} {'OK' if ok else 'NG'}")
            if ok:
                acc.append(dict(asg))
            return acc
        # 1-3 unknowns: enumerate free digits, print the fits (each row is
        # self-verifiable arithmetic); a no-fit op gets one NG line
        from itertools import permutations
        tried = 0
        for combo in permutations(free, len(unk)):
            tried += 1
            a2 = dict(asg)
            a2.update(zip(unk, combo))
            if fact_check(inp, out, op, mode, a2):
                em.emit(f"{prefix}{ops_show} " + ",".join(f"{ren[u]}={d}"
                                                          for u, d in zip(unk, combo))
                        + ": fits OK")
                acc.append(a2)
        if not acc:
            em.emit(f"{prefix}{ops_show}: no fit in {tried} tries NG")
        return acc

    def dfs(remaining, asg, mode, ops_pinned, label):
        if em.over():
            _why("governor")
            raise Abort
        if not remaining:
            qc = q[2]
            for op in sorted(ops_pinned.get(qc, char_ops[qc]), key=OP_PRIOR.index):
                if op in S.CONCAT:
                    continue
                if any(s not in asg for s in (q[0], q[1], q[3], q[4])):
                    continue
                L = asg[q[0]] * 10 + asg[q[1]]
                R = asg[q[3]] * 10 + asg[q[4]]
                if mode == "rev_both":
                    L, R = S.rev2(L), S.rev2(R)
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
                em.emit(f"{label} complete: query {L}{'?'}{R} via {OP_SHOW.get(op, op)} = {v} -> {ans}")
                answers.append((ans, op, mode, dict(asg)))
                return
            return
        # adaptive order: the fact with fewest unknowns next (visible counts)
        i = min(remaining, key=lambda j: len([t for t in dict.fromkeys(fact_syms(*lexs[j]))
                                              if t not in asg]))
        rest2 = [j for j in remaining if j != i]
        inp, out = lexs[i]
        c = inp[2]
        unk = [s for s in dict.fromkeys(fact_syms(inp, out)) if s not in asg]
        free = [d for d in range(10) if d not in asg.values()]
        if len(unk) > 3 or (len(unk) == 3 and len(free) > 6):
            em.emit(f"{label}fact{i}: {len(unk)} unknowns - too wide, drop branch")
            LAST["wide"] = LAST.get("wide", 0) + 1
            return
        kid = 0
        # the same assignment accepted under a second op re-explores nothing
        # when the op pin is inert (char neither queried nor in another fact)
        pin_inert = c != q[2] and all(lexs[j][0][2] != c for j in rest2)
        seen_states = {}
        for op in sorted(ops_pinned.get(c, char_ops[c]), key=OP_PRIOR.index):
            if op in S.CONCAT:
                idxs = S.CONCAT[op]
                if len(out) == 4 and all(out[j] == inp[ii] for j, ii in enumerate(idxs)):
                    np_ = dict(ops_pinned)
                    np_[c] = {op}
                    dfs(rest2, asg, mode, np_, label)
                continue
            acc = alg_block(f"{label}fact{i} ", inp, out, op, mode, asg, unk, free)
            depth = len(order) - len(rest2)
            for a2 in acc:
                key = tuple(sorted(a2.items()))
                if pin_inert and key in seen_states:
                    em.emit(f"{label}same state as {seen_states[key]} - skip")
                    continue
                np_ = dict(ops_pinned)
                np_[c] = {op}
                lab2 = f"{label}{BRANCH_LABELS[min(depth, len(BRANCH_LABELS) - 1)]}{kid}>"
                kid += 1
                if pin_inert:
                    seen_states[key] = lab2[:-1]
                dfs(rest2, a2, mode, np_, lab2)

    rl = len(aout)

    def try_mode(mode, lab):
        """Narrate one mode's candidate list + DFS. Returns a status tag."""
        tag = "standard" if mode == "standard" else "swap (operands+result digit-reversed)"
        cands = anchor_candidates(ainp, aout, anchor_ops, opchars, mode)
        if cands == "OVER":
            em.emit(f"{tag}: pattern {lpat} has over {LIST_CAP} candidates - skip this mode")
            return "OVER"
        if not cands:
            em.emit(f"{tag}: pattern {lpat} -> 0 candidates")
            return "EMPTY"
        em.emit(f"{tag}: pattern {lpat} -> {len(cands)} candidates:")
        for i, (op, asg, av, bv, v) in enumerate(cands):
            em.emit(f"  {av:02d}{op_sym(op)}{bv:02d}{op_suffix(op)}={str(v).zfill(rl)}"
                    f" ->{lab}{i}: " + _asg_str(asg, ren))
        em.emit()
        for ci, (aop, asg, av, bv, v) in enumerate(cands):
            em.emit(f"{lab}{ci}>")
            dfs(list(order), dict(asg), mode, {ainp[2]: {aop}}, f"{lab}{ci}>")
            if em.over():
                _why("governor")
                raise Abort
        return "RAN"

    st1 = try_mode("standard", "R")
    st2 = None
    if not answers:
        if st1 == "RAN":
            em.emit("no standard reading completes - try swap")
        st2 = try_mode("rev_both", "S")

    if not answers:
        if "OVER" in (st1, st2):
            _why("list-over")
        elif st1 == "EMPTY" and st2 == "EMPTY":
            _why("list-empty")
        else:
            _why("dfs-noanswer")
        return None
    uniq = sorted({a for a, _, _, _ in answers})
    LAST["uniq"] = uniq
    if len(uniq) > 1:
        em.emit(f"multiple consistent answers 《{','.join(uniq)}》; priority pick: first complete "
                f"reading in list order")
    ans, op, mode, asg = answers[0]
    em.emit()
    em.emit(f"Query op {OP_SHOW.get(op, op)}; inverse remap: "
            + " ".join(f"{ren[c]}->{c}" for c in dict.fromkeys(ans) if c in ren))
    em.emit("I will now return the answer in \\boxed{}")
    em.emit(f"The answer in \\boxed{{–}} is \\boxed{{{ans}}}")
    if em.over_exact():
        _why("governor-final")
        raise Abort
    return "\n".join(em.lines)
