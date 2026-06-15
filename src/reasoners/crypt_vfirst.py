"""Verification-first narrator for cryptarithm_deduce (audit wf_b996f4a7-4a9).

One fixed skeleton, every problem:
  P0 normalize glyphs -> letters (digits A..J, op chars p/q/r); all reasoning
     in letter space, raw glyphs touched only here and at P8.
  P1 concat gate (literal symbol-rearrangement check; no mapping needed).
  P2 op filter per op char over the 14-op pool via result length/sign
     signatures, one verdict per op citing the killing example.
  P3 lazy hoisted anchor tables: units-column supports per (char,op,mode,ex),
     emitted ONCE on first citation, immutable thereafter.
  P4 combo disposal in fixed priority order: batched table-intersection kills,
     1-line joint-propagation deaths, snapshots for survivors.
  P5 per-survivor resolution: anchored bounded expansions (every candidate
     tuple = one written substitution check), <=2 binary splits with both
     branches written, state snapshot after every commit.
  P6 decision: DETERMINED (all survivors agree) / UNDERDETERMINED with the
     fixed printed priority policy / abstain (return None).
  P7 verify the chosen reading against every fully-pinned example, arithmetic
     written out; ANY mismatch -> hard abort (return None).
  P8 computed answer: decode -> one written 2-digit op -> encode -> map back
     to original glyphs -> box. Gold is never read; the external gold filter
     is the only consumer of problem.answer.

Token governor: per-line cost estimated at chars/1.35 (probe-calibrated on the
real Nemotron tokenizer, 2026-06-11; dense rows run 1.2-1.6 chars/tok) +
fixed overhead; abstain over TOK_CAP. The last-resort bounded-DFS path of the
prototype is NOT rendered (mega-step risk) - those pids abstain.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Optional

from reasoners.store_types import Problem
import reasoners.crypt_vfirst_solver as S

W_EXPAND = 60
MAX_SPLIT_DEPTH = 2
MAX_SPLIT_VALS = 3
TOK_CAP = 7300
CHARS_PER_TOK = 1.35
TOK_OVERHEAD = 120

DIGIT_LETTERS = "ABCDEFGHIJ"
OP_LETTERS = "pqr"

OP_PRIOR = {"add": 0.0, "mul": 0.0, "absdiff": 1.0,
            "add_p1": 1.5, "add_m1": 1.5, "mul_p1": 1.5, "mul_m1": 1.5,
            "sub_signed": 2.0, "rsub_signed": 3.0, "neg_absdiff": 3.0,
            "mod": 3.5, "rmod": 3.8, "gcd": 3.8,
            "concat_fwd": 0.5, "concat_rev": 1.0,
            "concat_fwd_ro": 2.5, "concat_rev_ro": 3.0}

OP_SHOW = {"add": "x+y", "add_p1": "x+y+1", "add_m1": "x+y-1",
           "mul": "x*y", "mul_p1": "x*y+1", "mul_m1": "x*y-1",
           "sub_signed": "x-y (sign-prefixed when negative)",
           "rsub_signed": "y-x (sign-prefixed when negative)",
           "absdiff": "|x-y|", "neg_absdiff": "-|x-y| (sign-prefixed)",
           "mod": "x mod y", "rmod": "y mod x", "gcd": "gcd(x,y)",
           "concat_fwd": "concat x then y", "concat_rev": "concat y then x",
           "concat_fwd_ro": "concat, operands reversed",
           "concat_rev_ro": "concat reversed, operands reversed"}


def combo_prior(ops, mode):
    pen = 0.0 if mode == "standard" else 1.0
    pen += sum(OP_PRIOR.get(o, 4.0) for o in ops.values())
    return (pen, mode, tuple(sorted(ops.items())))


def _dset(d):
    """Domain set -> compact digit-run string, e.g. {0,2,3} -> '023'."""
    return "".join(str(x) for x in sorted(d))


def _expr(op, L, R):
    """Arithmetically exact expression string for 'expr=value' claim lines."""
    if op == 'add':
        return f"{L}+{R}"
    if op == 'add_p1':
        return f"{L}+{R}+1"
    if op == 'add_m1':
        return f"{L}+{R}-1"
    if op == 'mul':
        return f"{L}*{R}"
    if op == 'mul_p1':
        return f"{L}*{R}+1"
    if op == 'mul_m1':
        return f"{L}*{R}-1"
    if op == 'sub_signed':
        return f"{L}-{R}"
    if op == 'rsub_signed':
        return f"{R}-{L}"
    if op == 'absdiff':
        return f"|{L}-{R}|"
    if op == 'neg_absdiff':
        return f"-|{L}-{R}|"
    if op == 'mod':
        return f"{L} mod {R}"
    if op == 'rmod':
        return f"{R} mod {L}"
    if op == 'gcd':
        return f"gcd({L},{R})"
    return f"{L}?{R}"


def _opsym(op):
    if op in S.MUL_FAM:
        return '*'
    if op in S.ADD_FAM:
        return '+'
    if op == 'mod':
        return ' mod '
    if op == 'rmod':
        return ' rmod '
    if op == 'gcd':
        return ' gcd '
    return '-'


def _combo_name(ops, mode, chars):
    return ".".join(ops[c] for c in chars) + ("." + ("std" if mode == "standard" else "rev"))


class Emitter:
    def __init__(self):
        self.lines = []
        self.chars = 0
        self.mute = False

    def emit(self, line=""):
        if self.mute:
            return
        self.lines.append(line)
        self.chars += len(line) + 1

    def est_tok(self):
        return TOK_OVERHEAD + self.chars / CHARS_PER_TOK

    def over(self):
        return (not self.mute) and self.est_tok() > TOK_CAP


class Abort(Exception):
    """Verification failed or budget exceeded -> narrator returns None."""


# ---------------------------------------------------------------- P0
def normalize(prob):
    """Rename digit glyphs -> A..J (first occurrence over examples then query),
    op chars -> p/q/r. Returns (renamed problem dict, glyph maps)."""
    exs = [(e["input_value"], e["output_value"]) for e in prob["examples"]]
    q = prob["question"]
    opchars = []
    for inp, _ in exs:
        if inp[2] not in opchars:
            opchars.append(inp[2])
    if q[2] not in opchars:
        opchars.append(q[2])
    if len(opchars) > len(OP_LETTERS):
        raise Abort
    opmap = {c: OP_LETTERS[i] for i, c in enumerate(opchars)}
    digmap = {}

    def see(c):
        if c in opmap or c in digmap:
            return
        if len(digmap) >= 10:
            raise Abort
        digmap[c] = DIGIT_LETTERS[len(digmap)]

    for inp, out in exs:
        for c in (inp[0], inp[1], inp[3], inp[4]):
            see(c)
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        for c in body:
            see(c)
    for c in (q[0], q[1], q[3], q[4]):
        see(c)
    full = {**opmap, **digmap}

    def ren(s):
        return "".join(full.get(c, c) for c in s)

    rexs = [{"input_value": ren(i), "output_value": ren(o)} for i, o in exs]
    rprob = {"id": prob["id"], "examples": rexs, "question": ren(q),
             "answer": "\x00unused\x00"}
    return rprob, digmap, opmap


# ---------------------------------------------------------------- P3 helpers
def table_supports(inp, out, op, mode, opchars):
    syms = {inp[0], inp[1], inp[3], inp[4]}
    body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
    syms |= set(body)
    syms -= set(opchars)
    domains = {s: set(range(10)) for s in sorted(syms)}
    if op in S.CONCAT:
        idx = S.CONCAT[op]
        if len(out) != 4 or any(out[j] != inp[i] for j, i in enumerate(idx)):
            return None
        return domains
    try:
        S.example_units_tens(inp, out, op, mode, domains, {}, [0])
        return domains
    except S.Dead:
        return None


def ex_unknowns(inp, out, domains):
    body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
    inv = [inp[0], inp[1], inp[3], inp[4]] + list(body)
    uniq = []
    for s in inv:
        if s not in uniq and len(domains[s]) > 1:
            uniq.append(s)
    return uniq


def decode_pair(inp, mode, asg):
    a1, a2, b1, b2 = inp[0], inp[1], inp[3], inp[4]
    L = asg[a1] * 10 + asg[a2]
    R = asg[b1] * 10 + asg[b2]
    if mode == "rev_both":
        L, R = S.rev2(L), S.rev2(R)
    return L, R


def expected_body(inp, out, op, mode, asg):
    """Digit string the output body must equal under asg; None if impossible."""
    L, R = decode_pair(inp, mode, asg)
    v = S.op_value(op, L, R)
    if v is None:
        return None, None
    sign = len(out) > 1 and out[0] == inp[2]
    body = out[1:] if sign else out
    signed = op in ("sub_signed", "rsub_signed", "neg_absdiff")
    if signed and (v < 0) != sign:
        return None, v
    if not signed and v < 0:
        return None, v
    s = str(abs(v)).zfill(len(body))
    if len(s) != len(body):
        return None, v
    if mode == "rev_both":
        s = s[::-1]
    return s, v


class Narrator:
    def __init__(self, prob):
        self.raw = prob
        self.rp, self.digmap, self.opmap = normalize(prob)
        self.em = Emitter()
        self.exs, self.q, opchars, self.syms = S.parse(self.rp)
        self.opchars = sorted(opchars)
        self.tables = {}        # (c, op, mode, k) -> domains or None
        self.table_id = {}      # key -> 'T01'
        self.table_shown = set()
        self.snap_n = 0

    # -------- table identity & lazy emission
    def t_label(self, key):
        if key not in self.table_id:
            self.table_id[key] = f"T{len(self.table_id) + 1:02d}"
        return self.table_id[key]

    def show_table(self, key):
        lab = self.t_label(key)
        if key in self.table_shown:
            return lab
        self.table_shown.add(key)
        c, op, mode, k = key
        t = self.tables[key]
        inp, out = self.exs[k]
        mtag = "std" if mode in ("standard", "NA") else "rev"
        if t is None:
            self.em.emit(f"{lab} {c}:{op}.{mtag}.e{k+1} infeasible (no digit pair fits the units column)")
            return lab
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        nat = body[::-1] if mode == "rev_both" else body
        A = (inp[1], inp[0]) if mode == "rev_both" else (inp[0], inp[1])
        B = (inp[4], inp[3]) if mode == "rev_both" else (inp[3], inp[4])
        narrowed = sorted(((s, d) for s, d in t.items() if len(d) < 10),
                          key=lambda kv: (len(kv[1]), kv[0]))
        sets = " ".join(f"{s}:{_dset(d)}" for s, d in narrowed)
        self.em.emit(f"{lab} {c}:{op}.{mtag}.e{k+1} units {A[1]}{_opsym(op)}{B[1]} ends {nat[-1]}: {sets}")
        return lab

    # -------- P0 (v2: interleaved rename - each glyph=letter pair is written
    # at the moment of use; never a detached table applied across a block.
    # The pilot proved 3B cannot apply a 10-13-entry substitution: full-P0
    # fidelity 2.8%, and rewrites followed the model's own table in 5/106.)
    def _pairs(self, raw):
        full = {**self.opmap, **self.digmap}
        return " ".join(f"{ch}{full[ch]}" for ch in raw)

    def p0(self):
        em = self.em
        full = {**self.opmap, **self.digmap}
        em.emit("Cryptarithm (each glyph = a distinct digit 0-9). Rename glyph by glyph "
                "as we read: digits get A..J in order of first appearance, operators get "
                "p/q/r. Each pair below is glyph then letter; the letter string follows.")
        rexs = [(e["input_value"], e["output_value"]) for e in self.raw["examples"]]
        for k, ((rin, rout), (lin, lout)) in enumerate(zip(rexs, self.exs)):
            em.emit(f"  e{k+1}: {rin} = {rout}")
            em.emit(f"      {self._pairs(rin)}  =  {self._pairs(rout)}  ->  {lin} = {lout}")
        rq = self.raw["question"]
        em.emit(f"  query: {rq}")
        em.emit(f"      {self._pairs(rq)}  ->  {self.q}")
        em.emit("Sign rule: an output starting with its own op letter is a negative result.")
        em.emit()

    # -------- P1
    def p1_concat(self):
        """Concat gate for the QUERY op char only (the validated success path).
        Returns rearranged answer (letter space) or None."""
        qc = self.q[2]
        exl = [(inp, out) for inp, out in self.exs if inp[2] == qc]
        if not exl:
            raise Abort  # query op undemonstrated (not a deduce shape)
        for op, idx in S.CONCAT.items():
            ok = all(len(out) == 4 and all(out[j] == inp[i] for j, i in enumerate(idx))
                     for inp, out in exl)
            if ok:
                self.em.emit(f"Concat check for query op {qc}: every {qc}-example's output is just its "
                             f"input symbols rearranged ({OP_SHOW[op]}) - no digit code needed.")
                for inp, out in exl:
                    self.em.emit(f"  {inp} = {out} ok")
                ans = "".join(self.q[i] for i in idx)
                self.em.emit(f"Apply to query {self.q}: {ans}")
                return ans
        self.em.emit(f"Concat check for query op {qc}: no rearrangement reproduces its examples - "
                     f"the rule is arithmetic, recover digits.")
        return None

    # -------- P2
    def p2_filter(self):
        by_char = defaultdict(list)
        for k, (inp, out) in enumerate(self.exs):
            by_char[inp[2]].append((k, inp, out))
        self.by_char = by_char
        cands = {}
        em = self.em
        em.emit()
        em.emit("Op filter (2-digit operands; result length/sign must fit every example):")
        common = [op for op in S.POOL if op not in S.EXTRA_OPS]
        for c in sorted(by_char):
            exl = [(inp, out) for _, inp, out in by_char[c]]
            keep, kills = [], []
            for op in common:
                verdict = self._stage1_verdict(op, exl, c)
                if verdict is None:
                    keep.append(op)
                else:
                    kills.append((op, verdict))
            rare_note = False
            if not keep:
                # fixed second tier: rare ops considered only when no common
                # op fits this char (same common->rare pattern as eq_num)
                rare_note = True
                for op in S.EXTRA_OPS:
                    verdict = self._stage1_verdict(op, exl, c)
                    if verdict is None:
                        keep.append(op)
                    else:
                        kills.append((op, verdict))
            if not keep:
                em.emit(f"  [{c}] no candidate operation fits - outside the pool.")
                raise Abort
            bygrp = defaultdict(list)
            for op, why in kills:
                bygrp[why].append(op)
            for why, opl in bygrp.items():
                em.emit(f"  [{c}] kill {' '.join(opl)}: {why}")
            if rare_note:
                em.emit(f"  [{c}] no common op fits - extend with the rare ops mod/rmod/gcd")
            em.emit(f"  [{c}] keep: {', '.join(keep)}")
            cands[c] = keep
        return cands

    def _stage1_verdict(self, op, exl, c):
        """None if op survives; else short kill reason citing an example."""
        for i, (inp, out) in enumerate(exl):
            if op in S.CONCAT:
                idx = S.CONCAT[op]
                if len(out) != 4 or any(out[j] != inp[ii] for j, ii in enumerate(idx)):
                    return f"not a symbol rearrangement (e.g. output {out})"
                continue
            sign = len(out) > 1 and out[0] == inp[2]
            body = out[1:] if sign else out
            if any(ch in self.opchars for ch in body):
                return "op letter inside the result body"
            rl = len(body)
            if sign and op not in ("sub_signed", "rsub_signed", "neg_absdiff"):
                return f"result {out} is sign-prefixed but {op} is never negative"
            if op == "neg_absdiff" and not sign and rl != 1:
                return f"unsigned {rl}-digit result; -|x-y| without sign means 0"
            if op in S.ADD_FAM and rl not in (2, 3):
                return f"{rl}-digit result; x+y of 2-digit numbers gives 2-3 digits"
            if op in ("mul", "mul_p1") and rl not in (3, 4):
                return f"{rl}-digit result; 2-digit x 2-digit mul gives 3-4 digits"
            if op == "mul_m1" and rl not in (2, 3, 4):
                return f"{rl}-digit result; mul-1 gives 2-4 digits"
            if op in ("sub_signed", "rsub_signed", "absdiff") and rl not in (1, 2):
                return f"{rl}-digit result; a difference of 2-digit numbers has 1-2 digits"
            if op in S.EXTRA_OPS and rl not in (1, 2):
                return f"{rl}-digit result; mod/gcd of 2-digit numbers has 1-2 digits"
            if op in S.EXTRA_OPS and sign:
                return f"sign-prefixed result but {op} is never negative"
        return None

    # -------- P3+P4
    def p4_dispose(self, cands):
        em = self.em
        chars = sorted(cands)
        self.chars_order = chars
        # build all tables silently; emit lazily on citation
        for c in chars:
            for op in cands[c]:
                modes = ["NA"] if op in S.CONCAT else ["standard", "rev_both"]
                for mode in modes:
                    for k, inp, out in self.by_char[c]:
                        self.tables[(c, op, mode, k)] = table_supports(
                            inp, out, op, "standard" if mode == "NA" else mode, self.opchars)
        combos, seen = [], set()
        for tup in product(*[cands[c] for c in chars]):
            for mode in ("standard", "rev_both"):
                key = (tup, "NA" if all(o in S.CONCAT for o in tup) else mode)
                if key in seen:
                    continue
                seen.add(key)
                combos.append((dict(zip(chars, tup)), mode))
        combos.sort(key=lambda cm: combo_prior(cm[0], cm[1]))
        truncated = False
        if len(combos) > 100:
            combos = combos[:100]
            truncated = True
        em.emit()
        em.emit(f"Hypotheses: {len(combos)} (op,mode) combos in fixed priority "
                f"(std first; exact before +-1; add/mul first):")
        em.emit("  " + " ".join(_combo_name(o, m, chars) for o, m in combos[:8])
                + (" ..." if len(combos) > 8 else ""))
        if truncated:
            em.emit("  (list capped at 100 by priority)")
        # group intersect-kills by kill key
        kill_groups = defaultdict(list)
        weak = []
        order_kept = []
        for ops, mode in combos:
            doms = {s: set(range(10)) for s in self.syms}
            dead = None
            for c in chars:
                o = ops[c]
                bm = "NA" if o in S.CONCAT else mode
                for k, inp, out in self.by_char[c]:
                    t = self.tables[(c, o, bm, k)]
                    if t is None:
                        dead = ("TABLE", c, o, bm, k)
                        break
                    for s, sup in t.items():
                        doms[s] &= sup
                        if not doms[s]:
                            dead = ("SYM", s, c, o, bm, k)
                            break
                    if dead:
                        break
                if dead:
                    break
            if dead:
                kill_groups[dead].append((ops, mode))
                continue
            order_kept.append((ops, mode, doms))
        for key, members in kill_groups.items():
            names = " ".join(_combo_name(o, m, chars) for o, m in members[:6])
            more = f" (+{len(members)-6} more)" if len(members) > 6 else ""
            if key[0] == "TABLE":
                _, c, o, bm, k = key
                lab = self.show_table((c, o, bm, k))
                em.emit(f"kill {names}{more}: {lab} infeasible")
            else:
                _, s, c, o, bm, k = key
                lab = self.show_table((c, o, bm, k))
                em.emit(f"kill {names}{more}: {lab} leaves no digit for {s}")
        # v3: survivors carry their TABLE-INTERSECTION domains - no joint
        # propagation in the narrated path (its outputs were asserted
        # brute-force results the model cannot derive). Truly-inconsistent
        # combos that pass the table check die visibly inside attempts.
        weak = order_kept
        if weak:
            em.emit("Survive the table check: "
                    + " ".join(_combo_name(o, m, chars) for o, m, _ in weak[:10]))
        if len(weak) > 10:
            weak = weak[:10]
            em.emit("  (survivors capped at 10 by priority)")
            truncated = True
        return weak, truncated

    # -------- P5
    def snapshot(self, domains, note=""):
        self.snap_n += 1
        pinned = {s: d for s, d in sorted(domains.items()) if len(d) == 1}
        rest = " ".join(f"{s}:{_dset(d)}" for s, d in sorted(domains.items())
                        if 1 < len(d) <= 5)
        self.em.emit(f"S{self.snap_n}{note}: " +
                     " ".join(f"{s}={_dset(d)}" for s, d in pinned.items()) +
                     (f" | open {rest}" if rest else ""))

    def expand_narrated(self, inp, out, op, mode, domains, unknowns):
        em = self.em
        base = {s: next(iter(domains[s])) for s in domains if len(domains[s]) == 1}
        pinned_digits = set(base.values())
        sup = {s: set() for s in unknowns}
        shown = 0
        found = False
        for tup in product(*[sorted(domains[s]) for s in unknowns]):
            if len(set(tup)) != len(tup) or set(tup) & pinned_digits:
                continue
            asg = dict(base)
            asg.update(zip(unknowns, tup))
            # units pre-filter, silently (it is a 1-digit fact; failures of it
            # are uninformative volume - the written rows are the full checks)
            a1, a2, oc, b1, b2 = inp
            if op not in S.CONCAT:
                signb = len(out) > 1 and out[0] == oc
                body = out[1:] if signb else out
                nat = body[::-1] if mode == "rev_both" else body
                A = (a2, a1) if mode == "rev_both" else (a1, a2)
                B = (b2, b1) if mode == "rev_both" else (b1, b2)
                xs, ys, us = A[1], B[1], nat[-1]
                if xs in asg and ys in asg and us in asg:
                    x, y, u = asg[xs], asg[ys], asg[us]
                    if op in S.ADD_FAM and (x + y + S.ADD_FAM[op]) % 10 != u:
                        continue
                    if op in S.MUL_FAM and (x * y + S.MUL_FAM[op]) % 10 != u:
                        continue
            shown += 1
            exp, v = expected_body(inp, out, op, mode, asg)
            tup_s = "".join(f"{s}{d}" for s, d in zip(unknowns, tup))
            L, R = decode_pair(inp, mode, asg)
            okq = exp is not None and S.check_example_full(inp, out, op, mode, asg)
            em.emit(f"  {tup_s} {_expr(op, L, R)}={v} {'ok' if okq else 'no'}")
            if okq:
                found = True
                for s in unknowns:
                    sup[s].add(asg[s])
            if self.em.over():
                raise Abort
        if not found:
            raise S.Dead
        changed = False
        for s in unknowns:
            nd = domains[s] & sup[s]
            if not nd:
                raise S.Dead
            if nd != domains[s]:
                domains[s] = nd
                changed = True
        return shown, changed

    def nAC_narrated(self, ops, mode, domains, depth=0):
        em = self.em
        while True:
            S.apply_alldiff(domains, [0])
            a = S.try_answer(self.q, ops, mode, domains, self.syms)
            if a is not None:
                return a, True
            widths = []
            for k, (inp, out) in enumerate(self.exs):
                op = ops[inp[2]]
                if op in S.CONCAT:
                    continue
                unk = ex_unknowns(inp, out, domains)
                if not unk:
                    continue
                base = {s: next(iter(domains[s])) for s in domains if len(domains[s]) == 1}
                w = self._anchored_width(inp, out, op, mode, domains, unk)
                widths.append((w, k, unk))
            widths.sort()
            did = False
            for w, k, unk in widths:
                if w > W_EXPAND:
                    break
                inp, out = self.exs[k]
                em.emit(f"Expand e{k+1}: open {''.join(unk)}, {w} candidates after units anchor:")
                _, changed = self.expand_narrated(inp, out, ops[inp[2]], mode, domains, unk)
                if changed:
                    did = True
                    S.apply_alldiff(domains, [0])
                    self.snapshot(domains)
                    break
            if did:
                continue
            if depth >= MAX_SPLIT_DEPTH:
                return None, False
            split_syms = [s for s in sorted(domains)
                          if 2 <= len(domains[s]) <= MAX_SPLIT_VALS]
            if not split_syms:
                return None, False
            s = min(split_syms, key=lambda x: (len(domains[x]), x))
            em.emit(f"Split on {s} (candidates {_dset(domains[s])}):")
            branch_results = []
            for val in sorted(domains[s]):
                d3 = {x: set(v) for x, v in domains.items()}
                d3[s] = {val}
                try:
                    S.apply_alldiff(d3, [0])
                    em.emit(f"  assume {s}={val}:")
                    a2, res2 = self.nAC_narrated(ops, mode, d3, depth + 1)
                    branch_results.append((val, a2, res2, d3))
                except S.Dead:
                    em.emit(f"  assume {s}={val}: contradiction - eliminated")
                if self.em.over():
                    raise Abort
            if not branch_results:
                raise S.Dead
            if len(branch_results) == 1:
                val, a2, res2, d3 = branch_results[0]
                em.emit(f"  only {s}={val} survives")
                domains.clear()
                domains.update(d3)
                if res2:
                    return a2, True
                continue
            answers = {a for _, a, res, _ in branch_results if res}
            if all(res for _, _, res, _ in branch_results) and len(answers) == 1:
                em.emit(f"  every surviving branch yields the same answer")
                return next(iter(answers)), True
            return None, False

    def _anchored_width(self, inp, out, op, mode, domains, unknowns):
        base = {s: next(iter(domains[s])) for s in domains if len(domains[s]) == 1}
        pinned_digits = set(base.values())
        n = 0
        for tup in product(*[sorted(domains[s]) for s in unknowns]):
            if len(set(tup)) != len(tup) or set(tup) & pinned_digits:
                continue
            asg = dict(base)
            asg.update(zip(unknowns, tup))
            a1, a2, oc, b1, b2 = inp
            if op not in S.CONCAT:
                signb = len(out) > 1 and out[0] == oc
                body = out[1:] if signb else out
                nat = body[::-1] if mode == "rev_both" else body
                A = (a2, a1) if mode == "rev_both" else (a1, a2)
                B = (b2, b1) if mode == "rev_both" else (b1, b2)
                xs, ys, us = A[1], B[1], nat[-1]
                if xs in asg and ys in asg and us in asg:
                    x, y, u = asg[xs], asg[ys], asg[us]
                    if op in S.ADD_FAM and (x + y + S.ADD_FAM[op]) % 10 != u:
                        continue
                    if op in S.MUL_FAM and (x * y + S.MUL_FAM[op]) % 10 != u:
                        continue
            n += 1
            if n > W_EXPAND:
                return n
        return n

    def narrate_intersection(self, ops, mode):
        """Visible derivation of an attempt's starting domains: per-symbol
        string intersections of the cited tables. Returns the domains."""
        em = self.em
        doms = {s: set(range(10)) for s in self.syms}
        per_sym = {}
        for c in self.chars_order:
            o = ops[c]
            bm = "NA" if o in S.CONCAT else mode
            for k, inp, out in self.by_char[c]:
                t = self.tables.get((c, o, bm, k))
                if t is None:
                    continue
                lab = self.show_table((c, o, bm, k))
                for s, sup in t.items():
                    if len(sup) < 10:
                        per_sym.setdefault(s, []).append((lab, frozenset(sup)))
        for s in sorted(per_sym):
            parts = per_sym[s]
            inter = set(range(10))
            for _, sup in parts:
                inter &= sup
            doms[s] = inter
            if len(parts) >= 2:
                em.emit(f"  {s}: " + " ^ ".join(f"{lab} {_dset(set(sup))}" for lab, sup in parts)
                        + f" -> {_dset(inter)}")
            else:
                lab, sup = parts[0]
                em.emit(f"  {s}: {lab} {_dset(set(sup))}")
        return doms

    # -------- P7
    def verify(self, ops, mode, domains):
        em = self.em
        em.emit()
        em.emit("Verify the chosen reading against every example:")
        for k, (inp, out) in enumerate(self.exs):
            op = ops[inp[2]]
            if op in S.CONCAT:
                em.emit(f"  e{k+1}: symbol rearrangement, matches by construction")
                continue
            need = {inp[0], inp[1], inp[3], inp[4]}
            body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
            need |= set(body)
            if any(len(domains[s]) != 1 for s in need):
                em.emit(f"  e{k+1}: some symbols stay open; its consistency is the expansion checks above")
                continue
            asg = {s: next(iter(domains[s])) for s in need}
            L, R = decode_pair(inp, mode, asg)
            exp, v = expected_body(inp, out, op, mode, asg)
            got_ok = exp is not None and all(asg[c] == int(d) for c, d in zip(body, exp))
            em.emit(f"  e{k+1}: {_expr(op, L, R)}={v}, result digits {''.join(str(asg[c]) for c in body)} "
                    f"want {exp if exp else '?'} {'ok' if got_ok else 'MISMATCH'}")
            if not got_ok:
                raise Abort  # hard abort: never emit an acknowledged failure

    # -------- P8
    def answer_block(self, ops, mode, domains, letter_answer):
        em = self.em
        q = self.q
        op = ops[q[2]]
        em.emit()
        if op in S.CONCAT:
            em.emit(f"Query {q} is the rearrangement: {letter_answer}")
        else:
            asg = {s: next(iter(domains[s])) for s in (q[0], q[1], q[3], q[4])}
            L, R = decode_pair(q, mode, asg)
            v = S.op_value(op, L, R)
            em.emit(f"Query {q}: {q[0]}{q[1]}={asg[q[0]]}{asg[q[1]]} {q[3]}{q[4]}={asg[q[3]]}{asg[q[4]]}"
                    + (f" (digits reversed: {L} and {R})" if mode == "rev_both" else f" -> {L} and {R}"))
            em.emit(f"  {_expr(op, L, R)} = {v}")
            s = str(abs(v))
            if mode == "rev_both":
                em.emit(f"  write reversed: {s[::-1]}")
                s = s[::-1]
            inv = {}
            for sym2, d in domains.items():
                if len(d) == 1:
                    inv[next(iter(d))] = sym2
            if any(int(ch) not in inv for ch in s):
                raise Abort
            em.emit("  check every result digit has a pinned letter: "
                    + " ".join(f"{ch}={inv[int(ch)]}" for ch in dict.fromkeys(s))
                    + " - all pinned")
            enc = []
            for ch in s:
                enc.append(inv[int(ch)])
            sign = q[2] if (v is not None and v < 0) else ""
            em.emit(f"  encode: " + " ".join(f"{ch}={inv[int(ch)]}" for ch in s)
                    + (f", negative so prefix {q[2]}" if sign else ""))
            letter_answer = sign + "".join(enc)
        # map back to glyphs
        rev_dig = {L: g for g, L in self.digmap.items()}
        rev_op = {L: g for g, L in self.opmap.items()}
        glyph = "".join(rev_dig.get(c, rev_op.get(c, c)) for c in letter_answer)
        if any(c in rev_dig or c in rev_op for c in letter_answer):
            em.emit(f"Back to the original glyphs: "
                    + " ".join(f"{c}={rev_dig.get(c, rev_op.get(c, c))!r}" for c in letter_answer))
        em.emit(f"The answer is \\boxed{{{glyph}}}")
        return glyph


def find_witness(nar, ops, mode, domains, cap_nodes=20000):
    """First full consistent assignment within domains (backtracking), or None."""
    order = sorted(nar.syms, key=lambda s: (len(domains[s]), s))
    exneed = []
    for inp, out in nar.exs:
        need = {inp[0], inp[1], inp[3], inp[4]}
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        need |= set(body)
        exneed.append(need & set(nar.syms))
    pos = {s: i for i, s in enumerate(order)}
    ready = [max((pos[s] for s in need), default=-1) for need in exneed]
    nodes = [0]
    mp = {}

    def bt(i, used):
        if nodes[0] > cap_nodes:
            return None
        if i == len(order):
            return dict(mp)
        s = order[i]
        for d in sorted(domains[s]):
            if d in used:
                continue
            nodes[0] += 1
            mp[s] = d
            ok = True
            for k, (inp, out) in enumerate(nar.exs):
                if ready[k] == i:
                    if not S.check_example_full(inp, out, ops[inp[2]], mode, mp):
                        ok = False
                        break
            if ok:
                r = bt(i + 1, used | {d})
                if r is not None:
                    return r
            del mp[s]
        return None

    return bt(0, set())


def answer_of_witness(nar, ops, mode, witness):
    q = nar.q
    op = ops[q[2]]
    if op in S.CONCAT:
        return "".join(q[i] for i in S.CONCAT[op])
    L, R = decode_pair(q, mode, witness)
    v = S.op_value(op, L, R)
    s = str(abs(v))
    if mode == "rev_both":
        s = s[::-1]
    inv = {d: c for c, d in witness.items()}
    if any(int(c) not in inv for c in s):
        return None
    return (q[2] if v < 0 else "") + "".join(inv[int(c)] for c in s)


def narrate_certificate(nar, name, ops, mode, witness, ans):
    """Rival reading: print the witness assignment + verify every example."""
    em = nar.em
    em.emit(f"Rival {name}: a complete consistent reading exists - check it:")
    em.emit("  " + " ".join(f"{s}={d}" for s, d in sorted(witness.items())))
    for k, (inp, out) in enumerate(nar.exs):
        op = ops[inp[2]]
        if op in S.CONCAT:
            em.emit(f"  e{k+1}: rearrangement ok")
            continue
        L, R = decode_pair(inp, mode, witness)
        v = S.op_value(op, L, R)
        em.emit(f"  e{k+1}: {_expr(op, L, R)}={v} ok")
    em.emit(f"  its query answer: {ans}")


def reasoning_cryptarithm_vfirst(problem: Problem) -> Optional[str]:
    """CoT for cryptarithm_deduce or None (abstain). Never reads problem.answer."""
    prob = {"id": problem.id,
            "examples": [{"input_value": str(e.input_value),
                          "output_value": str(e.output_value)} for e in problem.examples],
            "question": str(problem.question)}
    try:
        return _narrate(prob)
    except (Abort, S.Dead, KeyError, IndexError, ValueError):
        return None


def _narrate(prob):
    for e in prob["examples"]:
        if len(e["input_value"]) != 5:
            return None
    if len(prob["question"]) != 5:
        return None
    nar = Narrator(prob)
    em = nar.em
    nar.p0()
    ans = nar.p1_concat()
    if ans is not None:
        glyph = nar.answer_block({nar.q[2]: next(op for op, idx in S.CONCAT.items()
                                                 if "".join(nar.q[i] for i in idx) == ans)},
                                 "standard", {}, ans)
        return "\n".join(em.lines)
    cands = nar.p2_filter()
    weak, truncated = nar.p4_dispose(cands)
    if not weak:
        return None
    # silent prescan: exact visible-parity (muted v3 nAC over the same
    # intersection domains the narration will derive)
    pres = []
    em.mute = True
    for ops, mode, domains in weak:
        try:
            work = {s: set(v) for s, v in domains.items()}
            S.apply_alldiff(work, [0])
            a, resolved = nar.nAC_narrated(ops, mode, work)
            pres.append((a, resolved, False))
        except S.Dead:
            pres.append((None, False, True))
    em.mute = False
    chosen_i = next((i for i, (a, r, dead) in enumerate(pres) if r and a), None)
    results = []
    em.emit()
    em.emit("Attempt the surviving readings in the order listed; the first that "
            "yields a unique query value is the reading we commit to.")
    for i, (ops, mode, domains) in enumerate(weak):
        name = _combo_name(ops, mode, nar.chars_order)
        pa, pr, pdead = pres[i]
        if pdead:
            em.emit(f"Attempt {name}: resolution runs into a contradiction - eliminated")
            continue
        if chosen_i is not None and i == chosen_i:
            # Explicit step-by-step cipher derivation (every column projection +
            # alldiff strike/pin shown, via the flag-guarded solver trace) instead
            # of asserting the pinned cipher. Falls back to the certificate path
            # only if pure propagation does not pin the query answer.
            S._TRACE.clear(); S._TRACE_ON = True
            try:
                twork, _, _ = S.propagate(nar.exs, set(nar.opchars), nar.syms, ops, mode)
            except S.Dead:
                S._TRACE_ON = False; return None
            S._TRACE_ON = False
            ta = S.try_answer(nar.q, ops, mode, twork, nar.syms)
            if ta is not None:
                em.emit(f"Attempt {name}: derive the cipher one step at a time "
                        f"(each line narrows a symbol's digit set):")
                for line in S._TRACE:
                    em.emit(line)
                fixed = " ".join(f"{s}={next(iter(twork[s]))}"
                                 for s in nar.syms if len(twork[s]) == 1)
                em.emit(f"  cipher fixed by the steps above: {fixed}")
                results.append((ops, mode, ta, True, twork))
                em.emit(f"  -> this reading yields {ta}")
            else:
                em.emit(f"Attempt {name}: derive its starting sets from the tables:")
                work = nar.narrate_intersection(ops, mode)
                try:
                    S.apply_alldiff(work, [0])
                except S.Dead:
                    return None
                nar.snapshot(work, note=f" {name}")
                try:
                    a, resolved = nar.nAC_narrated(ops, mode, work)
                except S.Dead:
                    return None
                results.append((ops, mode, a, resolved, work))
                em.emit(f"  -> this reading yields {a}")
        elif pr and pa:
            wit = find_witness(nar, ops, mode, domains)
            if wit is None:
                em.emit(f"Rival {name}: no complete consistent reading - eliminated")
                continue
            wa = answer_of_witness(nar, ops, mode, wit)
            narrate_certificate(nar, name, ops, mode, wit, wa if wa else "(not encodable)")
            results.append((ops, mode, pa, True, {s: {wit[s]} for s in wit}))
        else:
            # witnessed stall: show the blocking state when it is immediate
            widths = []
            for k, (inp, out) in enumerate(nar.exs):
                op = ops[inp[2]]
                if op in S.CONCAT:
                    continue
                unk = ex_unknowns(inp, out, domains)
                if unk:
                    widths.append((k + 1, nar._anchored_width(inp, out, op, mode, domains, unk)))
            if widths and all(w > W_EXPAND for _, w in widths):
                wtxt = " ".join(f"e{k}:{'>' if w > W_EXPAND else ''}{min(w, W_EXPAND + 1)}" for k, w in widths)
                em.emit(f"Attempt {name}: candidate counts {wtxt} all exceed {W_EXPAND}; "
                        f"no symbol narrow enough to split - stays open")
            else:
                em.emit(f"Attempt {name}: expansions narrow some symbols but no unique "
                        f"query value emerges within bounds - stays open")
            results.append((ops, mode, None, False, domains))
        if em.over():
            raise Abort
    live = [(o, m, a, r, d) for o, m, a, r, d in results if a is not None and r]
    unresolved = [r for r in results if not r[3]]
    answers = {a for _, _, a, r, _ in live if r}
    em.emit()
    if not answers:
        return None
    if len(answers) == 1 and not unresolved and not truncated:
        ops, mode, a, _, dom = live[0]
        em.emit(f"DETERMINED: every surviving reading yields the same answer.")
    elif len(answers) == 1:
        ops, mode, a, _, dom = live[0]
        em.emit(f"All resolved readings agree (some hypotheses stayed open within bounds).")
        em.emit("Priority policy (fixed): standard before reversed; exact op before +-1 variant; "
                "add/mul before the subtraction family. Taking the highest-priority resolved reading.")
    else:
        em.emit(f"UNDERDETERMINED: distinct consistent answers remain: "
                + " ".join(sorted(a for a in answers)) + ".")
        em.emit("Priority policy (fixed): standard before reversed; exact op before +-1 variant; "
                "add/mul before the subtraction family. Taking the highest-priority resolved reading.")
        ops, mode, a, _, dom = live[0]
    nar.verify(ops, mode, dom)
    # re-derive the answer through nAC's pinned domains for P8 decode
    d8 = dom
    if any(len(d8[s]) != 1 for s in (nar.q[0], nar.q[1], nar.q[3], nar.q[4])):
        # pin via bounded enumeration consistent with all examples (silent, small)
        a_chk, res_chk = _silent_resolve(nar, ops, mode, d8)
        if not res_chk or a_chk != a:
            return None
        em.emit(f"(query digits pinned during the expansion above)")
        # find one full consistent assignment to decode from
        ansset, _ = S.enumerate_answers(nar.exs, nar.q, ops, mode, d8, nar.syms,
                                        cap_nodes=4000, cap_ans=2)
        if len(ansset - {"NOSYM"}) != 1:
            return None
    glyph = nar.answer_block(ops, mode, d8, a)
    if em.over():
        raise Abort
    return "\n".join(em.lines)


def _silent_resolve(nar, ops, mode, domains):
    """Non-narrated resolution mirror (for final-domain bookkeeping)."""
    d = {s: set(v) for s, v in domains.items()}
    try:
        import reasoners.crypt_vfirst_solver as SS
        while True:
            SS.apply_alldiff(d, [0])
            a = SS.try_answer(nar.q, ops, mode, d, nar.syms)
            if a is not None:
                for s in d:
                    domains[s] = d[s]
                return a, True
            progressed = False
            for k, (inp, out) in enumerate(nar.exs):
                op = ops[inp[2]]
                if op in SS.CONCAT:
                    continue
                unk = ex_unknowns(inp, out, d)
                if not unk:
                    continue
                w = nar._anchored_width(inp, out, op, mode, d, unk)
                if w > W_EXPAND:
                    continue
                base = {s: next(iter(d[s])) for s in d if len(d[s]) == 1}
                pinned = set(base.values())
                sup = {s: set() for s in unk}
                found = False
                for tup in product(*[sorted(d[s]) for s in unk]):
                    if len(set(tup)) != len(tup) or set(tup) & pinned:
                        continue
                    asg = dict(base)
                    asg.update(zip(unk, tup))
                    if SS.check_example_full(inp, out, op, mode, asg):
                        found = True
                        for s in unk:
                            sup[s].add(asg[s])
                if not found:
                    raise SS.Dead
                for s in unk:
                    nd = d[s] & sup[s]
                    if not nd:
                        raise SS.Dead
                    if nd != d[s]:
                        d[s] = nd
                        progressed = True
                if progressed:
                    d2, _, _ = SS.propagate(nar.exs, set(nar.opchars), nar.syms, ops, mode, init=d)
                    d = d2
                    break
            if not progressed:
                return None, False
    except S.Dead:
        return None, False
