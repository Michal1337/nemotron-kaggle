"""
Investigator for bit_manipulation problems.

Strategy: try rotations (7), bit shifts (7 left + 7 right), identity, and NOT
variants of each. Combine up to 3 transforms with XOR, AND, or OR.
"""

import json
import os
import sys


def _rotate_left(val, k):
    """Rotate 8-bit value left by k."""
    k = k % 8
    return ((val << k) | (val >> (8 - k))) & 0xFF


def _shift_left(val, k):
    """Shift 8-bit value left by k, fill with 0."""
    return (val << k) & 0xFF


def _shift_right(val, k):
    """Shift 8-bit value right by k, fill with 0."""
    return (val >> k) & 0xFF


def _build_transforms():
    """Return list of (name, function) for all single transforms."""
    transforms = [
        ("I", lambda v: v),
        ("NOT", lambda v: v ^ 0xFF),
    ]
    for k in range(1, 8):
        transforms.append((f"ROT({k})", lambda v, k=k: _rotate_left(v, k)))
    for k in range(1, 8):
        transforms.append((f"SHL({k})", lambda v, k=k: _shift_left(v, k)))
    for k in range(1, 8):
        transforms.append((f"SHR({k})", lambda v, k=k: _shift_right(v, k)))
    # NOT variants
    for k in range(1, 8):
        transforms.append(
            (f"NOT ROT({k})", lambda v, k=k: _rotate_left(v, k) ^ 0xFF)
        )
    for k in range(1, 8):
        transforms.append(
            (f"NOT SHL({k})", lambda v, k=k: _shift_left(v, k) ^ 0xFF)
        )
    for k in range(1, 8):
        transforms.append(
            (f"NOT SHR({k})", lambda v, k=k: _shift_right(v, k) ^ 0xFF)
        )
    return transforms


TRANSFORMS = _build_transforms()
COMBINERS = [
    ("XOR", lambda a, b: a ^ b),
    ("AND", lambda a, b: a & b),
    ("OR", lambda a, b: a | b),
]


def solve_problem(data):
    """Try to find a rule that maps all examples correctly."""
    examples = data["examples"]
    inputs = [int(e["input_value"], 2) for e in examples]
    outputs = [int(e["output_value"], 2) for e in examples]
    query = int(data["question"], 2)
    n = len(inputs)
    nt = len(TRANSFORMS)

    # Precompute all transform results for each input
    results = []
    for _, fn in TRANSFORMS:
        results.append([fn(inp) for inp in inputs])

    # Try single transforms
    for t in range(nt):
        if all(results[t][i] == outputs[i] for i in range(n)):
            ans = TRANSFORMS[t][1](query)
            return format(ans, "08b"), TRANSFORMS[t][0], 1

    # Try pairs combined with XOR, AND, OR
    for cname, cfn in COMBINERS:
        for t1 in range(nt):
            for t2 in range(t1 + 1, nt):
                if all(
                    cfn(results[t1][i], results[t2][i]) == outputs[i]
                    for i in range(n)
                ):
                    ans = cfn(TRANSFORMS[t1][1](query), TRANSFORMS[t2][1](query))
                    desc = f"{TRANSFORMS[t1][0]} {cname} {TRANSFORMS[t2][0]}"
                    return format(ans, "08b"), desc, 2

    # Try triples combined with XOR
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            # Precompute t1 XOR t2 for all inputs
            pair = [results[t1][i] ^ results[t2][i] for i in range(n)]
            for t3 in range(t2 + 1, nt):
                if all((pair[i] ^ results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query)
                        ^ TRANSFORMS[t2][1](query)
                        ^ TRANSFORMS[t3][1](query)
                    )
                    desc = f"{TRANSFORMS[t1][0]} XOR {TRANSFORMS[t2][0]} XOR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples combined with AND
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] & results[t2][i] for i in range(n)]
            for t3 in range(t2 + 1, nt):
                if all((pair[i] & results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query)
                        & TRANSFORMS[t2][1](query)
                        & TRANSFORMS[t3][1](query)
                    )
                    desc = f"{TRANSFORMS[t1][0]} AND {TRANSFORMS[t2][0]} AND {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples combined with OR
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] | results[t2][i] for i in range(n)]
            for t3 in range(t2 + 1, nt):
                if all((pair[i] | results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query)
                        | TRANSFORMS[t2][1](query)
                        | TRANSFORMS[t3][1](query)
                    )
                    desc = f"{TRANSFORMS[t1][0]} OR {TRANSFORMS[t2][0]} OR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 XOR t2) AND t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] ^ results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] & results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) ^ TRANSFORMS[t2][1](query)
                    ) & TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} XOR {TRANSFORMS[t2][0]}) AND {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 AND t2) XOR t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] & results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] ^ results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) & TRANSFORMS[t2][1](query)
                    ) ^ TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} AND {TRANSFORMS[t2][0]}) XOR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 OR t2) XOR t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] | results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] ^ results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) | TRANSFORMS[t2][1](query)
                    ) ^ TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} OR {TRANSFORMS[t2][0]}) XOR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 XOR t2) OR t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] ^ results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] | results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) ^ TRANSFORMS[t2][1](query)
                    ) | TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} XOR {TRANSFORMS[t2][0]}) OR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 AND t2) OR t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] & results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] | results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) & TRANSFORMS[t2][1](query)
                    ) | TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} AND {TRANSFORMS[t2][0]}) OR {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    # Try triples: (t1 OR t2) AND t3
    for t1 in range(nt):
        for t2 in range(t1 + 1, nt):
            pair = [results[t1][i] | results[t2][i] for i in range(n)]
            for t3 in range(nt):
                if t3 == t1 or t3 == t2:
                    continue
                if all((pair[i] & results[t3][i]) == outputs[i] for i in range(n)):
                    ans = (
                        TRANSFORMS[t1][1](query) | TRANSFORMS[t2][1](query)
                    ) & TRANSFORMS[t3][1](query)
                    desc = f"({TRANSFORMS[t1][0]} OR {TRANSFORMS[t2][0]}) AND {TRANSFORMS[t3][0]}"
                    return format(ans, "08b"), desc, 3

    return None, None, 0


def main():
    base = os.path.join(os.path.dirname(__file__), os.pardir)
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
            if obj.get("category") == "bit_manipulation":
                problem_ids.append(obj["id"])

    print(f"Found {len(problem_ids)} bit_manipulation problems")

    correct = wrong = failed = 0
    transform_counts = {1: 0, 2: 0, 3: 0}
    transform_examples = {1: [], 2: [], 3: []}

    for i, pid in enumerate(problem_ids):
        with open(os.path.join(problems_dir, f"{pid}.jsonl")) as f:
            data = json.loads(f.readline())

        predicted, rule, num_transforms = solve_problem(data)
        actual = data["answer"]

        if predicted is None:
            failed += 1
        elif predicted == actual:
            correct += 1
            transform_counts[num_transforms] += 1
            if len(transform_examples[num_transforms]) < 5:
                transform_examples[num_transforms].append((pid, rule))
        else:
            wrong += 1

        if predicted == actual:
            inv_path = os.path.join(investigations_dir, f"{pid}.txt")
            lines = [
                f"problem id: {pid}",
                "category: bit_manipulation",
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

        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{len(problem_ids)}] C={correct} W={wrong} F={failed}")
            sys.stdout.flush()

    print(f"\nFinal: C={correct} W={wrong} F={failed} / {len(problem_ids)}")
    print(f"\nCorrect by transform count:")
    for k in (1, 2, 3):
        print(f"  {k} transform(s): {transform_counts[k]}")
        for pid, rule in transform_examples[k]:
            print(f"    {pid}: {rule}")


if __name__ == "__main__":
    main()
