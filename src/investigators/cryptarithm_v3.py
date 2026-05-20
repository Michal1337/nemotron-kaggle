"""Cryptarithm solver v3 — uses the equation_numeric op pool + rev_ops/rev_res.

Reframe (from forum reverse-engineering post + audit of equation_numeric.py):
cryptarithm = encrypted equation_numeric. The cipher is cosmetic. Once symbols
are mapped to digits, the underlying problem is bare equation_numeric, so the
right operator pool is the one [reasoners/equation_numeric.py](../reasoners/equation_numeric.py)
uses — not generic add/xor/gcd. Also: every op is tried under all 4 combos of
(reverse operands, reverse result), exactly like equation_numeric.py.

What v3 changes vs [cryptarithm_v2.py](cryptarithm_v2.py):
  * **Op pool**: replaced generic 30 ops (xor/gcd/etc.) with the ~32 ops from
    equation_numeric.py: add, abs_diff, neg_abs_diff, sub, rev_sub, mul, concat,
    rev_concat, add±1, mul±1, sub±1, rev_sub±1, max_mod_min, int_div, modulo,
    rev_div, rev_modulo, digit_abs_diff, digit_add_mod10, digit_sub_mod10,
    cross_mul, cross_mul_rev, digit_sum_{sum,diff}, digit_prod_{sum,diff},
    determinant, abs_determinant.
  * **rev_ops × rev_res**: every (op, rev_ops, rev_res) is tried; this is what
    equation_numeric.py does and what the post calls "pairings".
  * Multiprocessing + tqdm preserved.
  * Output investigation file format also keeps the global (rev_ops, rev_res).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import signal
import time
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, total=None, **_kw):
        return it


# -----------------------------------------------------------------------------
# Operator pool — matches reasoners/equation_numeric.py exactly.
# Each fn returns int OR None for invalid (e.g. div-by-zero, out-of-range).
# -----------------------------------------------------------------------------

def _safe_int_div(a, b):
    return a // b if b != 0 else None


def _safe_mod(a, b):
    return a % b if b != 0 else None


def _digit_abs_diff(a, b):
    # 2-digit operands only. Returns the 2-char digit-wise abs diff.
    if a > 99 or b > 99 or a < 0 or b < 0:
        return None
    d1, d2 = a // 10, a % 10
    d3, d4 = b // 10, b % 10
    return abs(d1 - d3) * 10 + abs(d2 - d4)


def _digit_add_mod10(a, b):
    if a > 99 or b > 99:
        return None
    d1, d2 = a // 10, a % 10
    d3, d4 = b // 10, b % 10
    return ((d1 + d3) % 10) * 10 + ((d2 + d4) % 10)


def _digit_sub_mod10(a, b):
    if a > 99 or b > 99:
        return None
    d1, d2 = a // 10, a % 10
    d3, d4 = b // 10, b % 10
    return ((d1 - d3) % 10) * 10 + ((d2 - d4) % 10)


def _cross_multiply(a, b):
    if a > 99 or b > 99:
        return None
    d1, d2 = a // 10, a % 10
    d3, d4 = b // 10, b % 10
    return d1 * d3 + d2 * d4


def _cross_multiply_rev(a, b):
    if a > 99 or b > 99:
        return None
    d1, d2 = a // 10, a % 10
    d3, d4 = b // 10, b % 10
    return d1 * d4 + d2 * d3


def _digit_sum_diff(a, b):
    if a > 99 or b > 99:
        return None
    return ((a // 10) + (a % 10)) - ((b // 10) + (b % 10))


def _digit_sum_sum(a, b):
    if a > 99 or b > 99:
        return None
    return ((a // 10) + (a % 10)) + ((b // 10) + (b % 10))


def _digit_prod_diff(a, b):
    if a > 99 or b > 99:
        return None
    return (a // 10) * (a % 10) - (b // 10) * (b % 10)


def _digit_prod_sum(a, b):
    if a > 99 or b > 99:
        return None
    return (a // 10) * (a % 10) + (b // 10) * (b % 10)


def _determinant(a, b):
    if a > 99 or b > 99:
        return None
    return (a // 10) * (b % 10) - (a % 10) * (b // 10)


def _abs_determinant(a, b):
    if a > 99 or b > 99:
        return None
    return abs((a // 10) * (b % 10) - (a % 10) * (b // 10))


OPS_V3: list[tuple[str, callable]] = [
    # --- common ---
    ("addition",            lambda a, b: a + b),
    ("subtraction",         lambda a, b: a - b),                         # can be negative
    ("rev_subtraction",     lambda a, b: b - a),
    ("absolute_diff",       lambda a, b: abs(a - b)),
    ("multiplication",      lambda a, b: a * b),
    ("concat",              lambda a, b: a * 100 + b),
    ("rev_concat",          lambda a, b: b * 100 + a),
    # --- rare ±1 variants ---
    ("add+1",               lambda a, b: a + b + 1),
    ("add-1",               lambda a, b: a + b - 1),
    ("sub+1",               lambda a, b: a - b + 1),
    ("sub-1",               lambda a, b: a - b - 1),
    ("rev_sub+1",           lambda a, b: b - a + 1),
    ("rev_sub-1",           lambda a, b: b - a - 1),
    ("multiply+1",          lambda a, b: a * b + 1),
    ("multiply-1",          lambda a, b: a * b - 1),
    # --- div / mod ---
    ("max_mod_min",         lambda a, b: max(a, b) % min(a, b) if min(a, b) > 0 else None),
    ("int_div",             _safe_int_div),
    ("modulo",              _safe_mod),
    ("rev_div",             lambda a, b: b // a if a != 0 else None),
    ("rev_modulo",          lambda a, b: b % a if a != 0 else None),
    # --- digit-wise (2-digit operands) ---
    ("digit_abs_diff",      _digit_abs_diff),
    ("digit_add_mod10",     _digit_add_mod10),
    ("digit_sub_mod10",     _digit_sub_mod10),
    ("cross_multiply",      _cross_multiply),
    ("cross_multiply_rev",  _cross_multiply_rev),
    ("digit_sum_diff",      _digit_sum_diff),
    ("digit_sum_sum",       _digit_sum_sum),
    ("digit_prod_diff",     _digit_prod_diff),
    ("digit_prod_sum",      _digit_prod_sum),
    ("determinant",         _determinant),
    ("abs_determinant",     _abs_determinant),
]

OP_NAMES = [n for n, _ in OPS_V3]
OP_FNS = [f for _, f in OPS_V3]


def num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return tuple(reversed(out))


def is_concat(ex) -> bool:
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


# -----------------------------------------------------------------------------
# Solver (DFS over symbol->digit). One instance per (rev_ops, rev_res, unique).
# -----------------------------------------------------------------------------

class SolverV3:
    def __init__(self, examples, query, rev_ops=False, rev_res=False,
                 unique=True, max_solutions=400):
        self.examples = examples       # list of (s0, s1, op_sym, s3, s4, rsyms)
        self.query = query             # (qs0, qs1, qop, qs3, qs4)
        self.rev_ops = rev_ops
        self.rev_res = rev_res
        self.unique = unique
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.answers: Counter = Counter()
        self.answer_info: dict[str, tuple[dict, dict]] = {}
        self.max_solutions = max_solutions

    def solve(self) -> tuple[str | None, tuple[dict, dict]]:
        self._process(0)
        if not self.answers:
            return None, ({}, {})
        best, best_count = self.answers.most_common(1)[0]
        total = sum(self.answers.values())
        if not self.unique and total > 1 and best_count < total * 0.3:
            return None, ({}, {})
        return best, self.answer_info.get(best, ({}, {}))

    def _process(self, idx: int) -> None:
        if len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        if op_sym in self.op_assign:
            ops_to_try = [self.op_assign[op_sym]]
        else:
            ops_to_try = list(range(len(OP_FNS)))

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                # Effective left operand
                lv = (d1 * 10 + d0) if self.rev_ops else (d0 * 10 + d1)
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = (d4 * 10 + d3) if self.rev_ops else (d3 * 10 + d4)

                        for op_id in ops_to_try:
                            try:
                                result_val = OP_FNS[op_id](lv, rv)
                            except Exception:
                                continue
                            if result_val is None or result_val < 0:
                                continue
                            rd_raw = list(num_to_digits(result_val))
                            rd = tuple(reversed(rd_raw)) if self.rev_res else tuple(rd_raw)
                            if len(rd) != rlen:
                                continue

                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return
        d0, d1, d3, d4 = (self.mapping[qs0], self.mapping[qs1],
                          self.mapping[qs3], self.mapping[qs4])
        ql = (d1 * 10 + d0) if self.rev_ops else (d0 * 10 + d1)
        qr = (d4 * 10 + d3) if self.rev_ops else (d3 * 10 + d4)

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            d2s.setdefault(d, s)

        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(OP_FNS))

        for op_id in op_candidates:
            try:
                result_val = OP_FNS[op_id](ql, qr)
            except Exception:
                continue
            if result_val is None or result_val < 0:
                continue
            rd_raw = num_to_digits(result_val)
            rd = tuple(reversed(rd_raw)) if self.rev_res else rd_raw
            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue
            ans = "".join(parts)
            self.answers[ans] += 1
            if ans not in self.answer_info:
                op_info = {k: OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = OP_NAMES[op_id]
                self.answer_info[ans] = (dict(self.mapping), op_info)


# -----------------------------------------------------------------------------
# High-level solve_problem: try all 4 (rev_ops, rev_res) combos, optionally
# non-unique fallback. Return best answer + which transform won.
# -----------------------------------------------------------------------------

def solve_problem_v3(data) -> tuple[str | None, tuple[dict, dict, bool, bool]]:
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        if len(inp) < 5:
            return None, ({}, {}, False, False)
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
    q = data["question"]
    if len(q) < 5:
        return None, ({}, {}, False, False)
    query = (q[0], q[1], q[2], q[3], q[4])

    # Concat fast path
    concat_ops, nonconcat_ops = set(), set()
    for ex in examples:
        (concat_ops if is_concat(ex) else nonconcat_ops).add(ex[2])
    q_op = query[2]
    if q_op in concat_ops and q_op not in nonconcat_ops:
        for ex in examples:
            if ex[2] == q_op and is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    return query[0] + query[1] + query[3] + query[4], (
                        {}, {q_op: "concat"}, False, False)
                return query[3] + query[4] + query[0] + query[1], (
                    {}, {q_op: "rev_concat"}, False, False)

    arith = [ex for ex in examples if not is_concat(ex)]
    if not arith:
        return query[0] + query[1] + query[3] + query[4], (
            {}, {q_op: "concat"}, False, False)

    # Aggregate answers across all 4 rev_ops × rev_res combos
    combined: Counter = Counter()
    combined_info: dict[str, tuple[dict, dict, bool, bool]] = {}

    for unique in (True, False):
        for ro in (False, True):
            for rr in (False, True):
                solver = SolverV3(arith, query, rev_ops=ro, rev_res=rr, unique=unique)
                solver._process(0)
                for ans, cnt in solver.answers.items():
                    combined[ans] += cnt
                    if ans not in combined_info:
                        info = solver.answer_info.get(ans, ({}, {}))
                        combined_info[ans] = (info[0], info[1], ro, rr)
        # If unique mode found anything, don't bother with non-unique
        if combined:
            break

    if not combined:
        return None, ({}, {}, False, False)

    best, _ = combined.most_common(1)[0]
    return best, combined_info[best]


# -----------------------------------------------------------------------------
# CLI: multiprocessing + tqdm wrapper, matches v2's interface.
# -----------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError()


def _verify(stored: str, predicted: str) -> bool:
    s, p = (stored or "").strip(), (predicted or "").strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def _worker_init() -> None:
    signal.signal(signal.SIGALRM, _timeout_handler)


def _solve_one(args_tuple: tuple) -> tuple:
    pid, problems_dir, timeout = args_tuple
    prob_file = os.path.join(problems_dir, f"{pid}.jsonl")
    if not os.path.exists(prob_file):
        return (pid, "missing", None, ({}, {}, False, False), 0.0)
    try:
        with open(prob_file) as f:
            data = json.loads(f.readline())
    except Exception:
        return (pid, "error", None, ({}, {}, False, False), 0.0)
    t0 = time.time()
    signal.alarm(timeout)
    try:
        ans, info = solve_problem_v3(data)
    except TimeoutError:
        signal.alarm(0)
        return (pid, "timeout", None, ({}, {}, False, False), time.time() - t0)
    except Exception:
        signal.alarm(0)
        return (pid, "error", None, ({}, {}, False, False), time.time() - t0)
    signal.alarm(0)
    return (pid, "ok", ans, info, time.time() - t0)


def _write_investigation(target_dir: str, pid: str, data: dict, predicted: str,
                          info: tuple[dict, dict, bool, bool], cat: str) -> None:
    mapping, opmap, rev_ops, rev_res = info
    lines = [
        f"problem id: {pid}",
        f"category: {cat}",
        "",
        f"transform: rev_ops={rev_ops}  rev_res={rev_res}",
        "",
        "symbol-to-digit mapping:",
    ]
    for s, d in sorted(mapping.items()):
        lines.append(f"  {s!r} = {d}")
    lines.extend(["", "operator-to-operation mapping:"])
    for s, name in sorted(opmap.items()):
        lines.append(f"  {s!r} = {name}")
    lines.extend(["", "examples:"])
    for e in data["examples"]:
        lines.append(f"  {e['input_value']} = {e['output_value']}")
    lines.extend(["",
                  f"query: {data['question']}",
                  "",
                  f"predicted answer: {predicted}"])
    path = os.path.join(target_dir, f"{pid}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=os.path.dirname(__file__) or ".")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-problem timeout (default 300s — v3 is broader than v2).")
    p.add_argument("--target", choices=["rule_unknown", "all_non_rule_found", "all"],
                   default="rule_unknown")
    p.add_argument("--write-investigations", action="store_true",
                   help="Write investigation files for newly solved problems.")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = p.parse_args()

    base = os.path.abspath(os.path.join(args.repo_root, ".."))
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")
    inv_root = os.path.join(base, "investigations")
    train_csv = os.path.join(base, "train.csv")

    import csv as _csv
    gold: dict[str, str] = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            gold[row["id"]] = row["answer"]

    problems: dict[str, dict] = {}
    with open(problems_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("category", "").startswith("cryptarithm"):
                problems[obj["id"]] = obj

    if args.target == "rule_unknown":
        wanted = {pid for pid, p in problems.items() if p["status"] == "rule_unknown"}
    elif args.target == "all_non_rule_found":
        wanted = {pid for pid, p in problems.items() if p["status"] != "rule_found"}
    else:
        wanted = set(problems.keys())
    pids = sorted(wanted)
    if args.limit:
        pids = pids[: args.limit]

    print(f"Loaded {len(problems)} cryptarithm problems")
    print(f"Targeting {len(pids)} with status={args.target}")
    print(f"Workers: {args.workers}  Timeout: {args.timeout}s")
    print(f"Operator candidates in v3: {len(OPS_V3)} (× 4 rev_ops/rev_res combos)")

    by_cat: Counter = Counter()
    op_counter: Counter = Counter()
    transform_counter: Counter = Counter()
    newly_solved: list = []
    wrong = timed_out = errored = no_answer = missing = 0
    t_start = time.time()

    tasks = [(pid, problems_dir, args.timeout) for pid in pids]

    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        results_iter = pool.imap_unordered(_solve_one, tasks, chunksize=1)
        pbar = tqdm(results_iter, total=len(tasks), desc="solving", smoothing=0.05)
        for pid, status, ans, info, elapsed in pbar:
            if status == "missing":
                missing += 1
                continue
            if status == "timeout":
                timed_out += 1
                continue
            if status == "error":
                errored += 1
                continue
            if ans is None:
                no_answer += 1
                continue
            if _verify(gold.get(pid, ""), ans):
                cat = problems[pid]["category"]
                by_cat[cat] += 1
                _, opmap, ro, rr = info
                for op_name in opmap.values():
                    op_counter[op_name] += 1
                transform_counter[(ro, rr)] += 1
                newly_solved.append((pid, ans, info, elapsed))
                if args.write_investigations:
                    target_dir = os.path.join(inv_root, cat, "correct")
                    os.makedirs(target_dir, exist_ok=True)
                    with open(os.path.join(problems_dir, f"{pid}.jsonl")) as f:
                        data = json.loads(f.readline())
                    _write_investigation(target_dir, pid, data, ans, info, cat)
            else:
                wrong += 1
            pbar.set_postfix(solved=len(newly_solved), wrong=wrong,
                             no_ans=no_answer, timeout=timed_out)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Finished in {total_elapsed:.0f}s")
    print(f"  newly solved (correct):     {len(newly_solved)}")
    print(f"  by category:                {dict(by_cat)}")
    print(f"  returned wrong answer:      {wrong}")
    print(f"  returned None (no answer):  {no_answer}")
    print(f"  timed out (>{args.timeout}s): {timed_out}")
    print(f"  errored:                    {errored}")

    if op_counter:
        print(f"\nOperators used by newly-solved problems:")
        for op, n in op_counter.most_common():
            print(f"  {op:>24s}: {n}")

    if transform_counter:
        print(f"\nTransform (rev_ops, rev_res) usage:")
        for (ro, rr), n in transform_counter.most_common():
            print(f"  rev_ops={ro!s:>5s} rev_res={rr!s:>5s}: {n}")


if __name__ == "__main__":
    main()
