"""Extended cryptarithm solver — broader operator coverage than the v1 investigator.

v1 ([investigators/cryptarithm_deduce.py](cryptarithm_deduce.py)) handles only:
    add, abs_diff, mul, concat, rev_concat

This v2 adds ~20 more candidate operators. The solver structure is the same:
DFS over symbol->digit assignments, accumulating consistent answers across all
examples, then picking the most-common predicted answer for the query.

We don't constrain result length to {1,2,3,4} anymore — any length 1..5 is allowed
because some ops produce more or fewer digits than the original 5 ops.

Each operator is a function ``f(a, b) -> int | None``. Return None on invalid
inputs (e.g. div-by-zero) and the solver will skip that combination.

Usage:
    python investigators/cryptarithm_v2.py [--limit 50] [--timeout 120]

Reads ``problems.jsonl`` (filters to category ``cryptarithm_deduce`` or
``cryptarithm_guess`` with status ``rule_unknown``), runs the v2 solver, and
writes investigation files for newly-solved problems under
``investigations/<category>/correct/`` (matching the new dir layout).
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


def num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    out = []
    while n > 0:
        out.append(n % 10)
        n //= 10
    return tuple(reversed(out))


def _reverse_digits(n: int) -> int:
    """Reverse the digits of n. 12 -> 21, 5 -> 5, 100 -> 1."""
    return int(str(n)[::-1])


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(n))


def _digit_prod(n: int) -> int:
    p = 1
    for c in str(n):
        p *= int(c)
    return p


# Each entry: (name, fn(a, b) -> int | None). Functions return None for invalid args.
OPS_V2: list[tuple[str, callable]] = [
    # --- v1 carry-over (kept by name to share investigation format if needed) ---
    ("add",                lambda a, b: a + b),
    ("abs_diff",           lambda a, b: abs(a - b)),
    ("mul",                lambda a, b: a * b),
    ("concat",             lambda a, b: a * 100 + b),
    ("rev_concat",         lambda a, b: b * 100 + a),
    # --- arithmetic ---
    ("sub_capped",         lambda a, b: a - b if a >= b else None),     # signed sub, only when a >= b (no neg)
    ("sub_neg_to_zero",    lambda a, b: max(a - b, 0)),                 # clip negatives to 0
    ("div_floor",          lambda a, b: a // b if b != 0 else None),
    ("mod",                lambda a, b: a % b if b != 0 else None),
    ("avg_floor",          lambda a, b: (a + b) // 2),
    ("avg_ceil",           lambda a, b: -(-(a + b) // 2)),
    ("min_op",             lambda a, b: min(a, b)),
    ("max_op",             lambda a, b: max(a, b)),
    # --- bitwise (treat 2-digit numbers as small ints) ---
    ("bw_xor",             lambda a, b: a ^ b),
    ("bw_and",             lambda a, b: a & b),
    ("bw_or",              lambda a, b: a | b),
    # --- multiplicative variants ---
    ("mul_mod_100",        lambda a, b: (a * b) % 100),
    ("mul_div_100",        lambda a, b: (a * b) // 100),
    ("sq_a_plus_sq_b",     lambda a, b: a * a + b * b),
    ("sq_a_minus_sq_b",    lambda a, b: a * a - b * b if a * a >= b * b else None),
    # --- digit-level ---
    ("reverse_a",          lambda a, b: _reverse_digits(a)),
    ("reverse_b",          lambda a, b: _reverse_digits(b)),
    ("reverse_sum",        lambda a, b: _reverse_digits(a + b)),
    ("reverse_a_plus_b",   lambda a, b: _reverse_digits(a) + b),
    ("a_plus_reverse_b",   lambda a, b: a + _reverse_digits(b)),
    ("digit_sum_a_b",      lambda a, b: _digit_sum(a) + _digit_sum(b)),
    ("digit_sum_concat",   lambda a, b: _digit_sum(a) * 10 + _digit_sum(b)),
    ("digit_prod_a_b",     lambda a, b: _digit_prod(a) + _digit_prod(b)),
    # --- digit-wise (interleave + per-position) ---
    ("interleave_abab",    lambda a, b: (a // 10) * 1000 + (b // 10) * 100 + (a % 10) * 10 + (b % 10)),
    ("interleave_baba",    lambda a, b: (b // 10) * 1000 + (a // 10) * 100 + (b % 10) * 10 + (a % 10)),
    ("digitwise_sum_nc",   lambda a, b: (((a // 10) + (b // 10)) % 10) * 10 + ((a + b) % 10)),  # no-carry per-digit add
    ("digitwise_diff",     lambda a, b: (abs((a // 10) - (b // 10))) * 10 + abs((a % 10) - (b % 10))),
    # --- number theory ---
    ("gcd",                lambda a, b: math.gcd(a, b) if (a or b) else None),
    ("lcm",                lambda a, b: (a * b // math.gcd(a, b)) if (a and b) else None),
]


class SolverV2:
    """DFS over symbol->digit assignments. Accepts result lengths in [min_len, max_len]."""

    OP_NAMES = [name for name, _ in OPS_V2]
    OP_FNS = [fn for _, fn in OPS_V2]

    def __init__(self, examples, query, unique=True, max_solutions=400, max_result_len=5):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.answers: Counter = Counter()
        self.answer_info: dict[str, tuple[dict, dict]] = {}
        self.max_solutions = max_solutions
        self.max_result_len = max_result_len

    def solve(self) -> tuple[str | None, tuple[dict, dict]]:
        self._process(0)
        if not self.answers:
            return None, ({}, {})
        best, best_count = self.answers.most_common(1)[0]
        total = sum(self.answers.values())
        # In non-unique mode, require a strong consensus to trust the prediction.
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
        if rlen > self.max_result_len:
            return

        if op_sym in self.op_assign:
            ops_to_try = [self.op_assign[op_sym]]
        else:
            ops_to_try = list(range(len(self.OP_FNS)))

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4

                        for op_id in ops_to_try:
                            try:
                                result_val = self.OP_FNS[op_id](lv, rv)
                            except Exception:
                                continue
                            if result_val is None or result_val < 0:
                                continue

                            # Determine output digits.
                            # concat/rev_concat from v1 retain pad-to-4 behavior.
                            name = self.OP_NAMES[op_id]
                            if name in ("concat", "rev_concat",
                                        "interleave_abab", "interleave_baba"):
                                if result_val >= 10000:
                                    continue
                                rd = tuple(int(c) for c in f"{result_val:04d}")
                            elif name == "digitwise_sum_nc" or name == "digitwise_diff":
                                # always 2-digit
                                if result_val >= 100:
                                    continue
                                rd = tuple(int(c) for c in f"{result_val:02d}")
                            else:
                                rd = num_to_digits(result_val)
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
        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(self.OP_FNS))

        d2s: dict[int, str] = {}
        for s, d in self.mapping.items():
            d2s.setdefault(d, s)

        for op_id in op_candidates:
            try:
                result_val = self.OP_FNS[op_id](ql, qr)
            except Exception:
                continue
            if result_val is None or result_val < 0:
                continue
            name = self.OP_NAMES[op_id]
            if name in ("concat", "rev_concat",
                        "interleave_abab", "interleave_baba"):
                if result_val >= 10000:
                    continue
                rd = tuple(int(c) for c in f"{result_val:04d}")
            elif name in ("digitwise_sum_nc", "digitwise_diff"):
                if result_val >= 100:
                    continue
                rd = tuple(int(c) for c in f"{result_val:02d}")
            else:
                rd = num_to_digits(result_val)

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
                op_info = {k: self.OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = self.OP_NAMES[op_id]
                self.answer_info[ans] = (dict(self.mapping), op_info)


def is_concat(ex) -> bool:
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


def solve_problem_v2(data) -> tuple[str | None, tuple[dict, dict]]:
    """High-level entrypoint matching v1's interface."""
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        if len(inp) < 5:
            return None, ({}, {})
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))
    q = data["question"]
    if len(q) < 5:
        return None, ({}, {})
    query = (q[0], q[1], q[2], q[3], q[4])

    # Handle pure-concat fast path (same as v1)
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
                        {}, {q_op: "concat"})
                return query[3] + query[4] + query[0] + query[1], (
                    {}, {q_op: "rev_concat"})

    # Only use non-concat examples for solving
    arith_examples = [ex for ex in examples if not is_concat(ex)]
    if not arith_examples:
        # All examples are concat; default v1 behavior
        return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})

    solver = SolverV2(arith_examples, query, unique=True)
    ans, info = solver.solve()
    if ans is not None:
        return ans, info
    solver2 = SolverV2(arith_examples, query, unique=False)
    return solver2.solve()


# -------------------------------------------------------------------------
# CLI: run on all rule_unknown cryptarithm problems, report new solves.
# -------------------------------------------------------------------------

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
    """Each worker process installs its own SIGALRM handler for per-task timeouts."""
    signal.signal(signal.SIGALRM, _timeout_handler)


def _solve_one(args_tuple: tuple) -> tuple:
    """Worker: load problem, run solver under SIGALRM, return result.

    Returns: (pid, status, predicted, info, elapsed_s)
      status ∈ {"ok", "timeout", "error", "missing"}
    """
    pid, problems_dir, timeout = args_tuple
    prob_file = os.path.join(problems_dir, f"{pid}.jsonl")
    if not os.path.exists(prob_file):
        return (pid, "missing", None, ({}, {}), 0.0)
    try:
        with open(prob_file) as f:
            data = json.loads(f.readline())
    except Exception:
        return (pid, "error", None, ({}, {}), 0.0)
    t0 = time.time()
    signal.alarm(timeout)
    try:
        ans, info = solve_problem_v2(data)
    except TimeoutError:
        signal.alarm(0)
        return (pid, "timeout", None, ({}, {}), time.time() - t0)
    except Exception:
        signal.alarm(0)
        return (pid, "error", None, ({}, {}), time.time() - t0)
    signal.alarm(0)
    return (pid, "ok", ans, info, time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=os.path.dirname(__file__) or ".",
                   help="Default: alongside this script (i.e. nemotron-master/).")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many problems (0=all).")
    p.add_argument("--timeout", type=int, default=120,
                   help="Per-problem timeout in seconds.")
    p.add_argument("--target", choices=["rule_unknown", "all_non_rule_found", "all"],
                   default="rule_unknown",
                   help="rule_unknown = neither reasoner nor v1 investigator solved. "
                        "all_non_rule_found = also try hypothesis_formed.")
    p.add_argument("--write-investigations", action="store_true",
                   help="Write investigation files for newly-solved problems.")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="Number of worker processes (default: min(16, cpu_count)).")
    args = p.parse_args()

    base = os.path.abspath(os.path.join(args.repo_root, ".."))  # parent of investigators/
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")
    inv_root = os.path.join(base, "investigations")
    train_csv = os.path.join(base, "train.csv")

    # Load gold answers
    import csv as _csv
    gold: dict[str, str] = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            gold[row["id"]] = row["answer"]

    # Load problems
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
    print(f"Workers: {args.workers}  Timeout: {args.timeout}s per problem")
    print(f"Operator candidates in v2: {len(OPS_V2)}")

    by_cat: Counter = Counter()
    newly_solved: list = []
    wrong = timed_out = errored = no_answer = missing = 0
    t_start = time.time()

    # Cache problem JSON in workers via filesystem reads (data files are tiny).
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
            # status == "ok"
            if ans is None:
                no_answer += 1
                continue
            if _verify(gold.get(pid, ""), ans):
                cat = problems[pid]["category"]
                by_cat[cat] += 1
                newly_solved.append((pid, ans, info, elapsed))
                if args.write_investigations:
                    target_dir = os.path.join(inv_root, cat, "correct")
                    os.makedirs(target_dir, exist_ok=True)
                    # Re-read problem data here (main process) to avoid passing
                    # the full payload through the worker queue.
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

    if newly_solved:
        op_counter = Counter()
        for pid, ans, info, t in newly_solved:
            symmap, opmap = info
            for op_sym, op_name in opmap.items():
                op_counter[op_name] += 1
        print(f"\nOperators used by newly-solved problems:")
        for op, n in op_counter.most_common():
            print(f"  {op:>24s}: {n}")


def _write_investigation(target_dir: str, pid: str, data: dict,
                          predicted: str, info: tuple[dict, dict], cat: str) -> None:
    mapping, opmap = info
    lines = [
        f"problem id: {pid}",
        f"category: {cat}",
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
        inp = e["input_value"]
        out = e["output_value"]
        lines.append(f"  {inp} = {out}")
    lines.extend(["",
                  f"query: {data['question']}",
                  "",
                  f"predicted answer: {predicted}"])
    path = os.path.join(target_dir, f"{pid}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
