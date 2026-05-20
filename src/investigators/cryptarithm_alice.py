"""Wrapper around the `AliceEquationSolver` from lkevincc0/kaggle-nemotron-equation-symbolic.

That solver reports 97.2% on the 823 cryptarithm problems (the comp's
`equation_symbolic` umbrella) using gold-conditioned symbolic search with 47
operators, 3 interpretation modes, and a tiered search to avoid false positives.

We use it for **corpus extension**: feed gold answers from train.csv, get
verified-correct symbolic rationales, write them as investigation files for
later conversion to training reasonings.

The Alice solver has a Rust extension for speed but **also a Python fallback**.
We pass through whichever is available — no compilation needed.

Setup:
    cd <somewhere>
    git clone https://github.com/lkevincc0/kaggle-nemotron-equation-symbolic
    # Don't bother with `maturin develop --release`; the Python fallback works.

Then run from this repo:
    python investigators/cryptarithm_alice.py \\
        --alice-root /path/to/kaggle-nemotron-equation-symbolic \\
        --target rule_unknown \\
        --workers 16 \\
        --timeout 300 \\
        --write-investigations

Output: investigation files at investigations/<category>/correct/<pid>.txt
for every newly-solved problem.
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
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, total=None, **_kw):
        return it


# Worker globals — set by _worker_init so we don't re-import per task.
_ALICE_SOLVER_CLS = None


def _worker_init(alice_src_dir: str) -> None:
    """Each worker imports AliceEquationSolver once + installs SIGALRM."""
    global _ALICE_SOLVER_CLS
    if alice_src_dir not in sys.path:
        sys.path.insert(0, alice_src_dir)
    from solver_eq_symbolic import AliceEquationSolver  # type: ignore[import-not-found]
    _ALICE_SOLVER_CLS = AliceEquationSolver

    def _h(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _h)


def _verify(stored: str, predicted: str) -> bool:
    """Mirror the comp scorer / reasoning.compare_answer."""
    s = (stored or "").strip()
    p = (predicted or "").strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def _solve_one(args_tuple: tuple) -> tuple:
    """Worker entry: load problem, run Alice solver with gold hint, verify.

    Returns: (pid, status, predicted, details_or_none, elapsed_s)
      status ∈ {"ok", "timeout", "error", "missing", "no_prompt"}
    """
    pid, problems_dir, train_csv_gold, timeout = args_tuple
    prob_file = os.path.join(problems_dir, f"{pid}.jsonl")
    if not os.path.exists(prob_file):
        return (pid, "missing", None, None, 0.0)

    try:
        with open(prob_file) as f:
            data = json.loads(f.readline())
    except Exception:
        return (pid, "error", None, None, 0.0)

    # Alice's solver wants the natural-language prompt text. Most problems have
    # it inline, but the field name varies — `prompt` is the comp standard.
    prompt = data.get("prompt") or data.get("question_text") or ""
    if not prompt:
        return (pid, "no_prompt", None, None, 0.0)

    gold = train_csv_gold.get(pid, "")

    t0 = time.time()
    signal.alarm(timeout)
    try:
        solver = _ALICE_SOLVER_CLS(prompt, answer_hint=gold or None)
        ans, details = solver.solve()
    except TimeoutError:
        signal.alarm(0)
        return (pid, "timeout", None, None, time.time() - t0)
    except Exception as exc:
        signal.alarm(0)
        return (pid, "error", str(exc)[:120], None, time.time() - t0)
    signal.alarm(0)

    return (pid, "ok", ans, details, time.time() - t0)


def _details_to_text(details: object) -> str:
    """Serialize the solver's details dict into a readable block."""
    if details is None:
        return "(no details)"
    if isinstance(details, dict):
        lines = []
        for k, v in details.items():
            if isinstance(v, (dict, list)):
                lines.append(f"  {k}: {json.dumps(v, default=str)}")
            else:
                lines.append(f"  {k}: {v!r}")
        return "\n".join(lines)
    return repr(details)


def _write_investigation(target_dir: str, pid: str, data: dict, predicted: str,
                          details: object, cat: str) -> None:
    """Terse investigation format, similar to v1/v3."""
    lines = [
        f"problem id: {pid}",
        f"category: {cat}",
        f"source: alice_eq_solver (gold-conditioned)",
        "",
        "details:",
        _details_to_text(details),
        "",
        "examples:",
    ]
    for e in data.get("examples", []):
        iv = e.get("input_value", "")
        ov = e.get("output_value", "")
        lines.append(f"  {iv} = {ov}")
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
    p.add_argument("--alice-root", required=True,
                   help="Path to the cloned kaggle-nemotron-equation-symbolic repo "
                        "(must contain src/solver_eq_symbolic.py).")
    p.add_argument("--repo-root", default=os.path.dirname(__file__) or ".",
                   help="nemotron-master path; default: alongside this script's parent.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-problem timeout in seconds (default 300).")
    p.add_argument("--target", choices=["rule_unknown", "all_non_rule_found", "all"],
                   default="rule_unknown",
                   help="rule_unknown = problems neither huikang's reasoner nor v1/v3 "
                        "investigator have solved. Default. "
                        "all_non_rule_found also tries hypothesis_formed problems.")
    p.add_argument("--write-investigations", action="store_true",
                   help="Write investigations/<category>/correct/<pid>.txt for newly solved.")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                   help="Default: min(16, cpu_count).")
    p.add_argument("--no-gold", action="store_true",
                   help="Don't pass gold answer as hint. Tests how the solver does blind "
                        "(useful as a sanity check — gold-conditioned will always score higher).")
    args = p.parse_args()

    # Resolve paths
    alice_src = os.path.join(args.alice_root, "src")
    if not os.path.isfile(os.path.join(alice_src, "solver_eq_symbolic.py")):
        sys.exit(f"--alice-root doesn't contain src/solver_eq_symbolic.py: {alice_src}")

    base = os.path.abspath(os.path.join(args.repo_root, ".."))
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")
    inv_root = os.path.join(base, "investigations")
    train_csv = os.path.join(base, "train.csv")

    if not os.path.isfile(problems_jsonl):
        sys.exit(f"problems.jsonl not found at {problems_jsonl}; check --repo-root.")

    # Sanity-import Alice's solver in the main process (for clear errors before mp).
    if alice_src not in sys.path:
        sys.path.insert(0, alice_src)
    try:
        from solver_eq_symbolic import AliceEquationSolver  # noqa: F401
        print("[cryptarithm_alice] Alice solver imported OK.")
    except Exception as exc:
        sys.exit(f"Failed to import Alice solver: {exc}")

    # Gold answers
    gold: dict[str, str] = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["answer"]
    train_gold_for_workers = gold if not args.no_gold else {}

    # Problems
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
    print(f"Gold conditioning: {'OFF (--no-gold)' if args.no_gold else 'ON'}")

    by_cat: Counter = Counter()
    newly_solved: list = []
    wrong = timed_out = errored = no_answer = missing = no_prompt = 0
    t_start = time.time()

    tasks = [(pid, problems_dir, train_gold_for_workers, args.timeout) for pid in pids]

    with mp.Pool(processes=args.workers,
                 initializer=_worker_init, initargs=(alice_src,)) as pool:
        results_iter = pool.imap_unordered(_solve_one, tasks, chunksize=1)
        pbar = tqdm(results_iter, total=len(tasks), desc="alice", smoothing=0.05)
        for pid, status, ans, details, elapsed in pbar:
            if status == "missing":
                missing += 1
                continue
            if status == "no_prompt":
                no_prompt += 1
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
    print(f"\n{'=' * 70}")
    print(f"Finished in {total_elapsed:.0f}s")
    print(f"  newly solved (correct):     {len(newly_solved)}")
    print(f"  by category:                {dict(by_cat)}")
    print(f"  returned wrong answer:      {wrong}")
    print(f"  returned None (no answer):  {no_answer}")
    print(f"  timed out (>{args.timeout}s): {timed_out}")
    print(f"  errored:                    {errored}")
    print(f"  no prompt field:            {no_prompt}")
    print(f"  missing problem file:       {missing}")


if __name__ == "__main__":
    main()
