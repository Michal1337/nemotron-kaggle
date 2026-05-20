"""equation_numeric solver with Alice-style broader op pool.

Why: huikang's [reasoners/equation_numeric.py](../reasoners/equation_numeric.py)
solves 561 of 732 equation_numeric problems (76.6%) using ~30 ops. The
forum-post solver (lkevincc0) reports 97.2% on the equivalent equation_symbolic
category using 47 ops + 3 modes. equation_numeric is the same problem type
**without** the cipher layer, so the same op pool should work.

This script:
  * Tries each operator character independently with the Alice 47-op pool
  * Allows the two interpretation modes (standard / little_endian)
  * Detects sign formatting (neg_prefix / neg_suffix using the operator char)
  * Verifies on ALL examples (single-example matches can be coincidental)
  * Optionally gold-conditioned (verify the predicted query answer matches gold)
  * Writes terse investigation files for newly-solved problems

Usage:
    python investigators/equation_numeric_v2.py --workers 16 --write-investigations
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from collections import Counter, defaultdict

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, total=None, **_kw):
        return it


# -----------------------------------------------------------------------------
# Alice's 47-op pool, adapted: every op is fn(a: int, b: int) -> int | None.
# Where None signals "invalid for these operands" (e.g. div-by-zero).
# -----------------------------------------------------------------------------

def _safe_div(a, b): return a // b if b != 0 else None
def _safe_mod(a, b): return a % b if b != 0 else None
def _safe_max_mod_min(a, b):
    m = min(a, b)
    return max(a, b) % m if m > 0 else None
def _safe_lcm(a, b):
    if a == 0 or b == 0: return None
    return abs(a * b) // math.gcd(a, b)


def _digit_abs_diff(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    d1, d2, d3, d4 = a // 10, a % 10, b // 10, b % 10
    return abs(d1 - d3) * 10 + abs(d2 - d4)

def _digit_add_mod10(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    d1, d2, d3, d4 = a // 10, a % 10, b // 10, b % 10
    return ((d1 + d3) % 10) * 10 + ((d2 + d4) % 10)

def _digit_sub_mod10(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    d1, d2, d3, d4 = a // 10, a % 10, b // 10, b % 10
    return ((d1 - d3) % 10) * 10 + ((d2 - d4) % 10)

def _cross_mul(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    d1, d2, d3, d4 = a // 10, a % 10, b // 10, b % 10
    return d1 * d3 + d2 * d4

def _cross_mul_rev(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    d1, d2, d3, d4 = a // 10, a % 10, b // 10, b % 10
    return d1 * d4 + d2 * d3

def _digit_sum_diff(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return ((a // 10) + (a % 10)) - ((b // 10) + (b % 10))

def _digit_sum_sum(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return ((a // 10) + (a % 10)) + ((b // 10) + (b % 10))

def _digit_prod_diff(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return (a // 10) * (a % 10) - (b // 10) * (b % 10)

def _digit_prod_sum(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return (a // 10) * (a % 10) + (b // 10) * (b % 10)

def _det(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return (a // 10) * (b % 10) - (a % 10) * (b // 10)

def _abs_det(a, b):
    if a > 99 or b > 99 or a < 0 or b < 0: return None
    return abs((a // 10) * (b % 10) - (a % 10) * (b // 10))


# Each op may produce negative results; sign formatting is handled separately.
OPS_EQ: list[tuple[str, callable]] = [
    # --- core arithmetic (TIER0) ---
    ("add",            lambda a, b: a + b),
    ("sub",            lambda a, b: a - b),
    ("rsub",           lambda a, b: b - a),
    ("absdiff",        lambda a, b: abs(a - b)),
    ("neg_absdiff",    lambda a, b: -abs(a - b)),
    ("mul",            lambda a, b: a * b),
    ("concat_fwd",     lambda a, b: a * 100 + b),
    ("concat_rev",     lambda a, b: b * 100 + a),
    # --- div/mod/order (TIER1) ---
    ("fdiv",           _safe_div),
    ("rdiv",           lambda a, b: b // a if a != 0 else None),
    ("mod",            _safe_mod),
    ("rmod",           lambda a, b: b % a if a != 0 else None),
    ("max_mod_min",    _safe_max_mod_min),
    ("min_op",         lambda a, b: min(a, b)),
    ("max_op",         lambda a, b: max(a, b)),
    ("gcd",            lambda a, b: math.gcd(a, b) if (a or b) else None),
    ("lcm",            _safe_lcm),
    # --- offset variants (TIER2) ---
    ("add_p1",         lambda a, b: a + b + 1),
    ("add_m1",         lambda a, b: a + b - 1),
    ("add_p2",         lambda a, b: a + b + 2),
    ("add_m2",         lambda a, b: a + b - 2),
    ("sub_p1",         lambda a, b: a - b + 1),
    ("sub_m1",         lambda a, b: a - b - 1),
    ("rsub_p1",        lambda a, b: b - a + 1),
    ("rsub_m1",        lambda a, b: b - a - 1),
    ("mul_p1",         lambda a, b: a * b + 1),
    ("mul_m1",         lambda a, b: a * b - 1),
    ("mul_p2",         lambda a, b: a * b + 2),
    ("mul_m2",         lambda a, b: a * b - 2),
    ("absdiff_p1",     lambda a, b: abs(a - b) + 1),
    ("absdiff_m1",     lambda a, b: abs(a - b) - 1),
    ("absdiff_p2",     lambda a, b: abs(a - b) + 2),
    ("absdiff_m2",     lambda a, b: abs(a - b) - 2),
    # --- scaled / polynomial (DEEP) ---
    ("mul_half",       lambda a, b: (a * b) // 2),
    ("mul_double",     lambda a, b: a * b * 2),
    ("sq_diff",        lambda a, b: a * a - b * b),
    ("sq_sum",         lambda a, b: a * a + b * b),
    ("mul_plus_a",     lambda a, b: a * b + a),
    ("mul_plus_b",     lambda a, b: a * b + b),
    ("mul_minus_a",    lambda a, b: a * b - a),
    ("mul_minus_b",    lambda a, b: a * b - b),
    ("a2_plus_b",      lambda a, b: a * a + b),
    ("a_plus_b2",      lambda a, b: a + b * b),
    # --- bitwise (DEEP) ---
    ("xor",            lambda a, b: a ^ b),
    ("band",           lambda a, b: a & b),
    ("bor",            lambda a, b: a | b),
    # --- digit-level (DEEP) ---
    ("digit_abs_diff", _digit_abs_diff),
    ("digit_add_mod10",_digit_add_mod10),
    ("digit_sub_mod10",_digit_sub_mod10),
    ("cross_mul",      _cross_mul),
    ("cross_mul_rev",  _cross_mul_rev),
    ("digit_sum_diff", _digit_sum_diff),
    ("digit_sum_sum",  _digit_sum_sum),
    ("digit_prod_diff",_digit_prod_diff),
    ("digit_prod_sum", _digit_prod_sum),
    ("det",            _det),
    ("abs_det",        _abs_det),
]

OP_NAMES = [n for n, _ in OPS_EQ]
OP_FNS = [f for _, f in OPS_EQ]

# Tier sizes for early-stop search. Index into the OPS_EQ list (cumulative).
TIER_BOUNDARIES = {
    "tier0": 8,    # core arithmetic only
    "tier1": 17,   # + div/mod/order
    "tier2": 33,   # + offset variants
    "deep":  len(OPS_EQ),
}


# -----------------------------------------------------------------------------
# Result format detection: outputs may have op-char or '-' as prefix/suffix for
# negative values (e.g. operator '-' encodes negative results as a leading '-').
# -----------------------------------------------------------------------------

def _normalize_output(out: str, op_char: str) -> tuple[int | None, str]:
    """Return (signed_int, fmt) where fmt in {'num','neg_prefix','neg_suffix','neg_dash_prefix','neg_dash_suffix'}.

    Tries to parse the output string into a signed integer using various
    sign-encoding conventions:
      - 'num'        : pure number (possibly with leading '-')
      - 'neg_dash_prefix' : leading '-' means negative ('-44')
      - 'neg_dash_suffix' : trailing '-' means negative ('44-')
      - 'neg_prefix' : leading op_char means negative ('+44' when op is '+')
      - 'neg_suffix' : trailing op_char means negative ('44+' when op is '+')
    """
    out = out.strip()
    # neg dash variants
    if out.startswith("-") and len(out) > 1 and out[1:].isdigit():
        return -int(out[1:]), "neg_dash_prefix"
    if out.endswith("-") and len(out) > 1 and out[:-1].isdigit():
        return -int(out[:-1]), "neg_dash_suffix"
    if op_char and op_char != "-":
        if out.startswith(op_char) and len(out) > len(op_char) and out[len(op_char):].isdigit():
            return -int(out[len(op_char):]), "neg_prefix"
        if out.endswith(op_char) and len(out) > len(op_char) and out[:-len(op_char)].isdigit():
            return -int(out[:-len(op_char)]), "neg_suffix"
    if out.lstrip("-").isdigit():
        return int(out), "num"
    return None, "unknown"


def _encode_output(value: int, fmt: str, op_char: str) -> str:
    """Inverse of _normalize_output."""
    if value >= 0 or fmt == "num":
        return str(value)
    abs_v = -value
    if fmt == "neg_dash_prefix":
        return f"-{abs_v}"
    if fmt == "neg_dash_suffix":
        return f"{abs_v}-"
    if fmt == "neg_prefix":
        return f"{op_char}{abs_v}"
    if fmt == "neg_suffix":
        return f"{abs_v}{op_char}"
    return str(value)


# -----------------------------------------------------------------------------
# Solver core
# -----------------------------------------------------------------------------

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


def _parse_problem(data: dict) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str]] | None:
    """Return ([(a_str, op_char, b_str, out_str), ...], (qa_str, q_op, qb_str)) or None."""
    parsed_ex: list[tuple[str, str, str, str]] = []
    for ex in data["examples"]:
        m = _EXPR_RE.fullmatch(str(ex["input_value"]))
        if not m:
            continue
        parsed_ex.append((m.group(1), m.group(2), m.group(3), str(ex["output_value"])))
    qm = _EXPR_RE.fullmatch(str(data["question"]))
    if not qm:
        return None
    return parsed_ex, (qm.group(1), qm.group(2), qm.group(3))


def _try_op(op_id: int, examples: list[tuple[str, str, str]],
            mode: str, rev_res: bool, sign_fmt: str, op_char: str) -> bool:
    """Check whether op_id consistently explains all examples under given transforms."""
    fn = OP_FNS[op_id]
    for a_str, b_str, out_str in examples:
        a, b = int(a_str), int(b_str)
        if mode == "little_endian":
            a = int(a_str[::-1]) if len(a_str) > 1 else a
            b = int(b_str[::-1]) if len(b_str) > 1 else b
        try:
            val = fn(a, b)
        except Exception:
            return False
        if val is None:
            return False
        # rev_res
        if rev_res:
            val_str = str(abs(val))[::-1] if val < 0 else str(val)[::-1]
            val_signed = -int(val_str) if val < 0 else int(val_str)
        else:
            val_signed = val
        expected, expected_fmt = _normalize_output(out_str, op_char)
        if expected is None:
            return False
        if val_signed != expected:
            return False
        if expected_fmt != sign_fmt:
            return False
    return True


def solve_equation_numeric(data: dict, gold_hint: str | None = None,
                            max_tier: str = "deep") -> tuple[str | None, dict | None]:
    """Returns (predicted_answer_str, details) or (None, None).

    Tiered search: tier0 -> tier1 -> tier2 -> deep. Stops at the first tier
    that yields a complete explanation.
    """
    parsed = _parse_problem(data)
    if parsed is None:
        return None, None
    examples, query = parsed
    if not examples:
        return None, None

    # Group examples by op char
    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in examples:
        by_op[op].append((a, b, out))

    q_a, q_op, q_b = query

    # Detect sign format for each operator (look at the outputs)
    sign_fmts: dict[str, str] = {}
    for op_char, group in by_op.items():
        fmts = Counter()
        for _, _, out in group:
            _, f = _normalize_output(out, op_char)
            fmts[f] += 1
        sign_fmts[op_char] = fmts.most_common(1)[0][0] if fmts else "num"

    # Search each tier in order
    tier_order = ["tier0", "tier1", "tier2", "deep"]
    if max_tier in tier_order:
        tier_order = tier_order[: tier_order.index(max_tier) + 1]

    # For each (mode, rev_res) combo, try to find consistent ops for ALL op_chars
    for tier in tier_order:
        n_ops = TIER_BOUNDARIES[tier]
        for mode in ("standard", "little_endian"):
            for rev_res in (False, True):
                ops_found: dict[str, int] = {}
                ok = True
                for op_char, group in by_op.items():
                    sign_fmt = sign_fmts[op_char]
                    op_id = None
                    for cand in range(n_ops):
                        if _try_op(cand, group, mode, rev_res, sign_fmt, op_char):
                            op_id = cand
                            break
                    if op_id is None:
                        ok = False
                        break
                    ops_found[op_char] = op_id
                if not ok:
                    continue

                # Apply to query
                if q_op not in ops_found:
                    # Try each op for the unseen query op (the _guess case)
                    candidates = list(range(n_ops))
                else:
                    candidates = [ops_found[q_op]]
                sign_fmt = sign_fmts.get(q_op, "num")

                for q_op_id in candidates:
                    fn = OP_FNS[q_op_id]
                    a_int, b_int = int(q_a), int(q_b)
                    if mode == "little_endian":
                        a_int = int(q_a[::-1]) if len(q_a) > 1 else a_int
                        b_int = int(q_b[::-1]) if len(q_b) > 1 else b_int
                    try:
                        v = fn(a_int, b_int)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    if rev_res:
                        v_str = str(abs(v))[::-1] if v < 0 else str(v)[::-1]
                        v_signed = -int(v_str) if v < 0 else int(v_str)
                    else:
                        v_signed = v
                    predicted = _encode_output(v_signed, sign_fmt, q_op)

                    # If gold hint provided, only accept matching solutions
                    if gold_hint is not None and predicted != gold_hint:
                        continue
                    details = {
                        "tier": tier,
                        "mode": mode,
                        "rev_res": rev_res,
                        "ops": {oc: OP_NAMES[i] for oc, i in ops_found.items()},
                        "query_op_id": OP_NAMES[q_op_id],
                        "sign_fmts": sign_fmts,
                        "numeric_answer": v_signed,
                    }
                    return predicted, details
    return None, None


# -----------------------------------------------------------------------------
# CLI: mp + tqdm + investigation writing
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
    pid, problems_dir, gold_map, timeout, use_gold = args_tuple
    prob_file = os.path.join(problems_dir, f"{pid}.jsonl")
    if not os.path.exists(prob_file):
        return (pid, "missing", None, None, 0.0)
    try:
        with open(prob_file) as f:
            data = json.loads(f.readline())
    except Exception:
        return (pid, "error", None, None, 0.0)
    t0 = time.time()
    signal.alarm(timeout)
    try:
        gold = gold_map.get(pid) if use_gold else None
        ans, details = solve_equation_numeric(data, gold_hint=gold)
    except TimeoutError:
        signal.alarm(0)
        return (pid, "timeout", None, None, time.time() - t0)
    except Exception:
        signal.alarm(0)
        return (pid, "error", None, None, time.time() - t0)
    signal.alarm(0)
    return (pid, "ok", ans, details, time.time() - t0)


def _write_investigation(target_dir: str, pid: str, data: dict, predicted: str,
                          details: dict, cat: str) -> None:
    lines = [
        f"problem id: {pid}",
        f"category: {cat}",
        f"source: equation_numeric_v2",
        "",
        f"transform: mode={details.get('mode')}  rev_res={details.get('rev_res')}  tier={details.get('tier')}",
        "",
        "operator-to-operation mapping:",
    ]
    for op_char, op_name in sorted(details.get("ops", {}).items()):
        lines.append(f"  {op_char!r} = {op_name}")
    lines.extend([
        "",
        f"query operator op_name: {details.get('query_op_id')}",
        "",
        "examples:",
    ])
    for e in data.get("examples", []):
        lines.append(f"  {e['input_value']} = {e['output_value']}")
    lines.extend([
        "",
        f"query: {data.get('question', '')}",
        "",
        f"predicted answer: {predicted}",
    ])
    path = os.path.join(target_dir, f"{pid}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=os.path.dirname(__file__) or ".")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--target", choices=["rule_unknown", "all_non_rule_found", "all"],
                   default="rule_unknown")
    p.add_argument("--write-investigations", action="store_true")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--no-gold", action="store_true",
                   help="Don't pass gold as a hint (test blind search).")
    args = p.parse_args()

    base = os.path.abspath(os.path.join(args.repo_root, ".."))
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")
    inv_root = os.path.join(base, "investigations")
    train_csv = os.path.join(base, "train.csv")

    gold: dict[str, str] = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["answer"]

    problems: dict[str, dict] = {}
    with open(problems_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("category", "").startswith("equation_numeric"):
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

    print(f"Loaded {len(problems)} equation_numeric problems")
    print(f"Targeting {len(pids)} with status={args.target}")
    print(f"Workers: {args.workers}  Timeout: {args.timeout}s  Gold hint: {not args.no_gold}")
    print(f"Operator pool: {len(OPS_EQ)} ops, tiers={list(TIER_BOUNDARIES)}")

    use_gold = not args.no_gold
    tasks = [(pid, problems_dir, gold, args.timeout, use_gold) for pid in pids]

    by_cat: Counter = Counter()
    op_counter: Counter = Counter()
    transform_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    newly_solved: list = []
    wrong = timed_out = errored = no_answer = missing = 0
    t_start = time.time()

    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        results_iter = pool.imap_unordered(_solve_one, tasks, chunksize=1)
        pbar = tqdm(results_iter, total=len(tasks), desc="eq_v2", smoothing=0.05)
        for pid, status, ans, details, elapsed in pbar:
            if status == "missing":
                missing += 1; continue
            if status == "timeout":
                timed_out += 1; continue
            if status == "error":
                errored += 1; continue
            if ans is None:
                no_answer += 1; continue
            if _verify(gold.get(pid, ""), ans):
                cat = problems[pid]["category"]
                by_cat[cat] += 1
                tier_counter[details.get("tier")] += 1
                transform_counter[(details.get("mode"), details.get("rev_res"))] += 1
                op_counter[details.get("query_op_id")] += 1
                newly_solved.append((pid, ans, details, elapsed))
                if args.write_investigations:
                    target_dir = os.path.join(inv_root, cat, "correct")
                    os.makedirs(target_dir, exist_ok=True)
                    with open(os.path.join(problems_dir, f"{pid}.jsonl")) as f:
                        data = json.loads(f.readline())
                    _write_investigation(target_dir, pid, data, ans, details, cat)
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
    if tier_counter:
        print(f"\nWinning tier:")
        for t, n in tier_counter.most_common():
            print(f"  {t}: {n}")
    if op_counter:
        print(f"\nQuery operator used:")
        for op, n in op_counter.most_common():
            print(f"  {op}: {n}")
    if transform_counter:
        print(f"\nTransform usage:")
        for (mode, rr), n in transform_counter.most_common():
            print(f"  mode={mode} rev_res={rr}: {n}")


if __name__ == "__main__":
    main()
