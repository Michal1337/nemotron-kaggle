"""
bit_manipulation_v2.py

Extends v1's search space with shapes the v1 reasoner can't represent:
- 3-input MAJ(a, b, c) = (a & b) | (a & c) | (b & c)
- 3-input CHO(a, b, c) = (a & b) | (~a & c)              (mux/select)
- 4-input PAR4 = a ^ b ^ c ^ d
- 4-input AOA  = (a & b) | (c & d)
- 4-input OAO  = (a | b) & (c | d)
- 4-input AXA  = (a & b) ^ (c & d)
- 4-input MAJ4 = bitwise (>=3 of 4) majority
- 4-input OXO  = (a | b) ^ (c | d)

v1 already covers AO/OA/AX/OX/XA/XO (3-input one-AND/OR/XOR with one literal)
and PAR3 (triple XOR), so we don't repeat those here.

Vectorised with numpy; multiprocessed via mp.Pool with per-worker timeout.
Writes huikang-compatible investigation files only when the predicted answer
matches the gold answer.
"""

import argparse
import json
import os
import signal
import sys
import time
from itertools import combinations
import multiprocessing as mp

import numpy as np
from tqdm import tqdm


# ----- single-input transform pool (mirrors v1) -----


def _rotl(v, k):
    k %= 8
    return ((v << k) | (v >> (8 - k))) & 0xFF


def _shl(v, k):
    return (v << k) & 0xFF


def _shr(v, k):
    return (v >> k) & 0xFF


def _build_transforms():
    T = [
        ("I", lambda v: v),
        ("NOT", lambda v: v ^ 0xFF),
    ]
    for k in range(1, 8):
        T.append((f"ROT({k})", lambda v, k=k: _rotl(v, k)))
        T.append((f"SHL({k})", lambda v, k=k: _shl(v, k)))
        T.append((f"SHR({k})", lambda v, k=k: _shr(v, k)))
        T.append((f"NOT ROT({k})", lambda v, k=k: _rotl(v, k) ^ 0xFF))
        T.append((f"NOT SHL({k})", lambda v, k=k: _shl(v, k) ^ 0xFF))
        T.append((f"NOT SHR({k})", lambda v, k=k: _shr(v, k) ^ 0xFF))
    return T


TRANSFORMS = _build_transforms()  # 44 entries


# ----- v2 search -----


def _precompute(inputs, query):
    """Return R (nt, n) example results, Q (nt,) query results, names list.

    No dedupe: transforms that agree on example inputs may disagree on the
    query, and we want to enumerate every distinct candidate query answer.
    """
    nt = len(TRANSFORMS)
    n = len(inputs)
    R = np.empty((nt, n), dtype=np.uint8)
    Q = np.empty(nt, dtype=np.uint8)
    for ti, (_, fn) in enumerate(TRANSFORMS):
        Q[ti] = fn(query)
        for ei, inp in enumerate(inputs):
            R[ti, ei] = fn(inp)
    return R, Q, [name for name, _ in TRANSFORMS]


def solve_v2(data, gold_int=None):
    """Search MAJ/CHO/PAR4/AOA/OAO/AXA shapes.

    Returns (binary_str, rule, shape) or (None, None, None).
    If gold_int is provided, only accept candidates whose query answer matches.
    Otherwise the first example-consistent candidate is returned.
    """
    examples = data["examples"]
    inputs = [int(e["input_value"], 2) for e in examples]
    outputs = np.array([int(e["output_value"], 2) for e in examples], dtype=np.uint8)
    query = int(data["question"], 2)

    Rk, Qk, names = _precompute(inputs, query)
    m = len(names)

    def _accept(ans):
        return gold_int is None or (ans & 0xFF) == gold_int

    # --- MAJ(a, b, c) ---
    for i, j, k in combinations(range(m), 3):
        a, b, c = Rk[i], Rk[j], Rk[k]
        res = (a & b) | (a & c) | (b & c)
        if np.array_equal(res, outputs):
            qa, qb, qc = int(Qk[i]), int(Qk[j]), int(Qk[k])
            ans = (qa & qb) | (qa & qc) | (qb & qc)
            if _accept(ans):
                return format(ans & 0xFF, "08b"), f"MAJ({names[i]}, {names[j]}, {names[k]})", "MAJ"

    # --- CHO(a, b, c) = (a & b) | (~a & c) ---
    not_Rk = (~Rk) & 0xFF
    not_Qk = (~Qk) & 0xFF
    for ai in range(m):
        a = Rk[ai]
        na = not_Rk[ai]
        for bi in range(m):
            if bi == ai:
                continue
            ab = a & Rk[bi]
            for ci in range(m):
                if ci == ai or ci == bi:
                    continue
                res = ab | (na & Rk[ci])
                if np.array_equal(res, outputs):
                    ans = (int(Qk[ai]) & int(Qk[bi])) | (int(not_Qk[ai]) & int(Qk[ci]))
                    if _accept(ans):
                        return (
                            format(ans & 0xFF, "08b"),
                            f"CHO({names[ai]}, {names[bi]}, {names[ci]})",
                            "CHO",
                        )

    # --- PAR4 = a ^ b ^ c ^ d ---
    for i, j, k, l in combinations(range(m), 4):
        res = Rk[i] ^ Rk[j] ^ Rk[k] ^ Rk[l]
        if np.array_equal(res, outputs):
            ans = int(Qk[i] ^ Qk[j] ^ Qk[k] ^ Qk[l])
            if _accept(ans):
                return (
                    format(ans & 0xFF, "08b"),
                    f"{names[i]} XOR {names[j]} XOR {names[k]} XOR {names[l]}",
                    "PAR4",
                )

    # Precompute pairwise ANDs / ORs once
    pair_idx = list(combinations(range(m), 2))
    if pair_idx:
        PA = np.stack([Rk[i] & Rk[j] for i, j in pair_idx])
        PO = np.stack([Rk[i] | Rk[j] for i, j in pair_idx])
    else:
        PA = PO = np.empty((0, len(outputs)), dtype=np.uint8)

    # --- AOA = (a & b) | (c & d): each AND-pair must be a sub-mask of outputs ---
    # (sub-mask: PA[p] | outputs == outputs)
    candidates_aoa = []
    for p, (i, j) in enumerate(pair_idx):
        if np.array_equal(PA[p] | outputs, outputs):
            candidates_aoa.append((p, i, j))
    for x in range(len(candidates_aoa)):
        p, i, j = candidates_aoa[x]
        for y in range(x + 1, len(candidates_aoa)):
            q, k, l = candidates_aoa[y]
            if i == k or i == l or j == k or j == l:
                continue
            if np.array_equal(PA[p] | PA[q], outputs):
                ans = (int(Qk[i]) & int(Qk[j])) | (int(Qk[k]) & int(Qk[l]))
                if _accept(ans):
                    return (
                        format(ans & 0xFF, "08b"),
                        f"({names[i]} AND {names[j]}) OR ({names[k]} AND {names[l]})",
                        "AOA",
                    )

    # --- OAO = (a | b) & (c | d): each OR-pair must be a super-mask of outputs ---
    candidates_oao = []
    for p, (i, j) in enumerate(pair_idx):
        if np.array_equal(PO[p] | outputs, PO[p]):
            candidates_oao.append((p, i, j))
    for x in range(len(candidates_oao)):
        p, i, j = candidates_oao[x]
        for y in range(x + 1, len(candidates_oao)):
            q, k, l = candidates_oao[y]
            if i == k or i == l or j == k or j == l:
                continue
            if np.array_equal(PO[p] & PO[q], outputs):
                ans = (int(Qk[i]) | int(Qk[j])) & (int(Qk[k]) | int(Qk[l]))
                if _accept(ans):
                    return (
                        format(ans & 0xFF, "08b"),
                        f"({names[i]} OR {names[j]}) AND ({names[k]} OR {names[l]})",
                        "OAO",
                    )

    # --- AXA = (a & b) ^ (c & d): no clean sub-mask filter, iterate all disjoint pairs ---
    for p in range(len(pair_idx)):
        i, j = pair_idx[p]
        for q in range(p + 1, len(pair_idx)):
            k, l = pair_idx[q]
            if i == k or i == l or j == k or j == l:
                continue
            if np.array_equal(PA[p] ^ PA[q], outputs):
                ans = (int(Qk[i]) & int(Qk[j])) ^ (int(Qk[k]) & int(Qk[l]))
                if _accept(ans):
                    return (
                        format(ans & 0xFF, "08b"),
                        f"({names[i]} AND {names[j]}) XOR ({names[k]} AND {names[l]})",
                        "AXA",
                    )

    # --- MAJ4 = bitwise majority-of-four (>=3 of 4 bits set per position) ---
    # equivalent to (a&b&c) | (a&b&d) | (a&c&d) | (b&c&d)
    for i, j, k, l in combinations(range(m), 4):
        a, b, c, d = Rk[i], Rk[j], Rk[k], Rk[l]
        res = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
        if np.array_equal(res, outputs):
            qa, qb, qc, qd = int(Qk[i]), int(Qk[j]), int(Qk[k]), int(Qk[l])
            ans = (qa & qb & qc) | (qa & qb & qd) | (qa & qc & qd) | (qb & qc & qd)
            if _accept(ans):
                return (
                    format(ans & 0xFF, "08b"),
                    f"MAJ4({names[i]}, {names[j]}, {names[k]}, {names[l]})",
                    "MAJ4",
                )

    # --- OXO = (a | b) ^ (c | d): disjoint pairs of ORs combined via XOR ---
    for p in range(len(pair_idx)):
        i, j = pair_idx[p]
        for q in range(p + 1, len(pair_idx)):
            k, l = pair_idx[q]
            if i == k or i == l or j == k or j == l:
                continue
            if np.array_equal(PO[p] ^ PO[q], outputs):
                ans = (int(Qk[i]) | int(Qk[j])) ^ (int(Qk[k]) | int(Qk[l]))
                if _accept(ans):
                    return (
                        format(ans & 0xFF, "08b"),
                        f"({names[i]} OR {names[j]}) XOR ({names[k]} OR {names[l]})",
                        "OXO",
                    )

    # PAR5 (5-element XOR) was tried and got zero hits on the 94 holdouts
    # while adding ~10x runtime. Left out by default; the remaining unsolved
    # problems likely use shapes outside this combinatorial framework.

    return None, None, None


# ----- multiprocessing harness -----


def _handle_timeout(signum, frame):
    raise TimeoutError("solve_v2 timed out")


def _worker_init(timeout):
    """Each worker installs its own SIGALRM handler with the requested budget."""
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _handle_timeout)
    # store on a module attribute the workers can read
    globals()["_PER_PROBLEM_TIMEOUT"] = timeout


def _solve_one(args):
    pid, problems_dir, gold_map = args
    timeout = globals().get("_PER_PROBLEM_TIMEOUT", 60)
    path = os.path.join(problems_dir, f"{pid}.jsonl")
    try:
        with open(path) as f:
            data = json.loads(f.readline())
    except Exception as e:
        return pid, None, None, None, f"read_error:{e}"

    gold = gold_map.get(pid, data.get("answer"))
    gold_int = None
    if gold and len(gold) == 8 and set(gold) <= {"0", "1"}:
        gold_int = int(gold, 2)

    if hasattr(signal, "SIGALRM"):
        signal.alarm(timeout)
    try:
        try:
            predicted, rule, shape = solve_v2(data, gold_int=gold_int)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
    except TimeoutError:
        return pid, None, None, None, "timeout"
    except Exception as e:
        return pid, None, None, None, f"error:{e}"

    status = "no_match"
    if predicted is None:
        status = "no_rule"
    elif gold and predicted == gold:
        status = "correct"
        # Write investigation eagerly so we don't lose it on crashes
        inv_dir = os.path.join(
            os.path.dirname(problems_dir), "investigations", "bit_manipulation", "correct"
        )
        os.makedirs(inv_dir, exist_ok=True)
        inv_path = os.path.join(inv_dir, f"{pid}.txt")
        # Overwrite even if a file exists: v2 only writes on a gold-verified
        # match, so its rule string is more reliable than any prior LLM-narrated
        # or v1-incorrect investigation that may already be there.
        lines = [
            f"problem id: {pid}",
            "category: bit_manipulation",
            "source: bit_manipulation_v2",
            "",
            f"rule: {rule}",
            "",
            "examples:",
        ]
        for e in data["examples"]:
            lines.append(f"  {e['input_value']} -> {e['output_value']}")
        lines.append("")
        lines.append(f"query: {data['question']}")
        lines.append(f"predicted answer: {predicted}")
        with open(inv_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    return pid, predicted, rule, shape, status


def _load_target_ids(problems_jsonl, target):
    ids = []
    with open(problems_jsonl) as f:
        for line in f:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("category") != "bit_manipulation":
                continue
            status = o.get("status", "unknown")
            if target == "rule_unknown" and status != "rule_unknown":
                continue
            if target == "non_rule_found" and status == "rule_found":
                continue
            if target == "all":
                pass
            ids.append(o["id"])
    return ids


def _load_gold(problems_jsonl, problems_dir, ids):
    gold = {}
    for pid in ids:
        path = os.path.join(problems_dir, f"{pid}.jsonl")
        try:
            with open(path) as f:
                d = json.loads(f.readline())
            if "answer" in d:
                gold[pid] = d["answer"]
        except Exception:
            pass
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        choices=["rule_unknown", "non_rule_found", "all"],
        default="non_rule_found",
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--timeout", type=int, default=60, help="per-problem timeout (s)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (debug)")
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(__file__), os.pardir)
    problems_jsonl = os.path.join(base, "problems.jsonl")
    problems_dir = os.path.join(base, "problems")

    ids = _load_target_ids(problems_jsonl, args.target)
    if args.limit:
        ids = ids[: args.limit]
    print(f"target={args.target}  problems={len(ids)}  workers={args.workers}")

    gold = _load_gold(problems_jsonl, problems_dir, ids)

    work = [(pid, problems_dir, gold) for pid in ids]

    counts = {"correct": 0, "no_match": 0, "no_rule": 0, "timeout": 0}
    shape_counts = {}
    t0 = time.time()

    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.timeout,),
    ) as pool:
        for pid, predicted, rule, shape, status in tqdm(
            pool.imap_unordered(_solve_one, work, chunksize=1),
            total=len(work),
            desc="bit_manip v2",
        ):
            key = "correct" if status == "correct" else ("timeout" if status == "timeout" else status.split(":")[0])
            counts[key] = counts.get(key, 0) + 1
            if status == "correct":
                shape_counts[shape] = shape_counts.get(shape, 0) + 1

    dt = time.time() - t0
    print()
    print(f"done in {dt:.1f}s")
    print(f"counts: {counts}")
    print(f"shapes (correct):")
    for k in sorted(shape_counts.keys()):
        print(f"  {k}: {shape_counts[k]}")


if __name__ == "__main__":
    main()
