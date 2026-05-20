"""
Solver for cryptarithm_deduce cryptarithmetic problems.

AB op CD = result
- Each symbol maps to a unique digit 0-9
- Operations: addition, absolute difference, multiplication
- Concat: if result == s0s1s3s4, it's just concatenation (trivial)
"""

import json
import os
import signal
import sys
from collections import Counter


OPS = [
    lambda a, b: a + b,  # 0: add
    lambda a, b: abs(a - b),  # 1: abs_diff
    lambda a, b: a * b,  # 2: mul
    lambda a, b: a * 100 + b,  # 3: concat
    lambda a, b: b * 100 + a,  # 4: reverse concat
]


def num_to_digits(n):
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def is_concat(ex):
    """Check if result is concat (s0s1s3s4) or rev_concat (s3s4s0s1)."""
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class Solver:
    def __init__(self, examples, query, unique=True):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping = {}
        self.used = set()
        self.op_assign = {}
        self.answers = Counter()
        self.answer_info = {}  # answer_str -> (mapping, op_assign)
        self.max_solutions = 200

    OP_NAMES = ["add", "abs_diff", "mul", "concat", "rev_concat"]

    def solve(self):
        self._process(0)
        if self.answers:
            best, best_count = self.answers.most_common(1)[0]
            total = sum(self.answers.values())
            # For non-unique mapping, require strong consensus
            if not self.unique and total > 1 and best_count < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def _process(self, idx):
        if len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        # Filter ops by result length feasibility
        # add: 0-198 (1-3 chars), abs_diff: 0-99 (1-2), mul: 0-9801 (1-4)
        # concat: 0-9999 (always 4 padded), rev_concat: 0-9999 (always 4 padded)
        feasible_ops = []
        if rlen <= 3:
            feasible_ops.append(0)  # add
        if rlen <= 2:
            feasible_ops.append(1)  # abs_diff
        if rlen <= 4:
            feasible_ops.append(2)  # mul
        if rlen == 4:
            feasible_ops.extend([3, 4])  # concat, rev_concat

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

                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )

                        for op_id in ops_to_try:
                            result_val = OPS[op_id](lv, rv)
                            # concat/rev_concat: pad to 4 digits
                            if op_id >= 3:
                                if result_val < 0 or result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
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
            # Try all arithmetic operations if query operator was never seen
            # in the examples.
            op_candidates = range(len(self.OP_NAMES))

        # Build reverse mapping from digits we know
        d2s = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_id in op_candidates:
            result_val = OPS[op_id](ql, qr)
            if op_id >= 3:
                if result_val < 0 or result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = num_to_digits(result_val)

            # If any result digit has no symbol, skip this solution
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
                self.answer_info[ans] = (
                    dict(self.mapping),
                    op_info,
                )


def solve_problem(data):
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))

    q = data["question"]
    query = (q[0], q[1], q[2], q[3], q[4])

    # Identify concat operators (result == s0s1s3s4)
    concat_ops = set()
    nonconcat_ops = set()
    for ex in examples:
        if is_concat(ex):
            concat_ops.add(ex[2])
        else:
            nonconcat_ops.add(ex[2])

    q_op = query[2]

    # If query operator is purely concat/rev_concat, answer is trivial
    if q_op in concat_ops and q_op not in nonconcat_ops:
        # Determine which direction from examples
        for ex in examples:
            if ex[2] == q_op and is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    return query[0] + query[1] + query[3] + query[4], (
                        {},
                        {q_op: "concat"},
                    )
                else:
                    return query[3] + query[4] + query[0] + query[1], (
                        {},
                        {q_op: "rev_concat"},
                    )
        # Fallback
        return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})

    # Only use non-concat examples for solving
    arith_examples = [ex for ex in examples if not is_concat(ex)]

    # Try unique mapping first
    solver = Solver(arith_examples, query, unique=True)
    ans, info = solver.solve()
    if ans is not None:
        return ans, info

    # Fallback: non-unique mapping
    solver2 = Solver(arith_examples, query, unique=False)
    return solver2.solve()


def _timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)


def main():
    base = os.path.dirname(__file__) or "."
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")
    investigations_dir = os.path.join(base, "investigations")
    os.makedirs(investigations_dir, exist_ok=True)

    problem_ids = []
    with open(problems_jsonl) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("category") == "cryptarithm_deduce":
                problem_ids.append(obj["id"])

    print(f"Found {len(problem_ids)} cryptarithm_deduce problems")

    correct = wrong = failed = 0

    for i, pid in enumerate(problem_ids):
        with open(os.path.join(problems_dir, f"{pid}.jsonl")) as f:
            data = json.loads(f.readline())

        predicted = None
        mapping_info = {}
        op_info = {}
        try:
            signal.alarm(30)
            predicted, (mapping_info, op_info) = solve_problem(data)
            signal.alarm(0)
        except (TimeoutError, KeyError):
            signal.alarm(0)

        actual = data["answer"]

        if predicted is None:
            failed += 1
        elif predicted == actual:
            correct += 1
        else:
            wrong += 1

        # Only write investigation file when correct
        if predicted == actual:
            inv_path = os.path.join(investigations_dir, f"{pid}.txt")

            lines = [
                f"problem id: {pid}",
                "category: cryptarithm_deduce",
                "",
            ]

            # Symbol mapping
            lines.append("symbol-to-digit mapping:")
            for s, d in sorted(mapping_info.items()):
                lines.append(f"  {s!r} = {d}")
            lines.append("")

            # Operator mapping
            lines.append("operator-to-operation mapping:")
            for s, name in sorted(op_info.items()):
                lines.append(f"  {s!r} = {name}")
            lines.append("")

            # Show each example with numeric values
            op_funcs = {
                "add": lambda a, b: f"{a} + {b} = {a + b}",
                "abs_diff": lambda a, b: f"|{a} - {b}| = {abs(a - b)}",
                "mul": lambda a, b: f"{a} * {b} = {a * b}",
                "concat": lambda a, b: f"concat({a}, {b}) = {a * 100 + b}",
                "rev_concat": lambda a, b: f"rev_concat({a}, {b}) = {b * 100 + a}",
            }
            lines.append("examples:")
            for e in data["examples"]:
                inp = e["input_value"]
                out = e["output_value"]
                op_sym = inp[2]
                op_name = op_info.get(op_sym, "?")
                if mapping_info:
                    lv = mapping_info.get(inp[0], "?")
                    ld = mapping_info.get(inp[1], "?")
                    rv = mapping_info.get(inp[3], "?")
                    rd = mapping_info.get(inp[4], "?")
                    if all(isinstance(x, int) for x in [lv, ld, rv, rd]):
                        left = lv * 10 + ld
                        right = rv * 10 + rd
                        if op_name in op_funcs:
                            calc = op_funcs[op_name](left, right)
                        else:
                            calc = "?"
                        lines.append(f"  {inp} = {out}  =>  {calc}")
                    else:
                        lines.append(f"  {inp} = {out}")
                else:
                    lines.append(f"  {inp} = {out}")
            lines.append("")

            # Show query computation
            q = data["question"]
            op_sym = q[2]
            op_name = op_info.get(op_sym, "?")
            if mapping_info:
                lv = mapping_info.get(q[0], "?")
                ld = mapping_info.get(q[1], "?")
                rv = mapping_info.get(q[3], "?")
                rd = mapping_info.get(q[4], "?")
                if all(isinstance(x, int) for x in [lv, ld, rv, rd]):
                    left = lv * 10 + ld
                    right = rv * 10 + rd
                    if op_name in op_funcs:
                        calc = op_funcs[op_name](left, right)
                    else:
                        calc = "?"
                    lines.append(f"query: {q}  =>  {calc}")
                else:
                    lines.append(f"query: {q}")
            else:
                lines.append(f"query: {q}")
            lines.append("")
            lines.append(f"predicted answer: {predicted}")

            with open(inv_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(problem_ids)}] C={correct} W={wrong} F={failed}")
            sys.stdout.flush()

    print(f"\nFinal: C={correct} W={wrong} F={failed} / {len(problem_ids)}")


if __name__ == "__main__":
    main()
