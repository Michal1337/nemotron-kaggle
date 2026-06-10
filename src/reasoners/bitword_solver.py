"""Whole-word bit-rule solver (answer-blind) used to extend bit_manipulation coverage.

solve(examples, question) deduces the 8-bit answer from the example pairs ONLY,
by searching one expression of up to 3 ROT/SHL/SHR transforms combined with
AND/OR/XOR and per-operand NOT that reproduces every output bit of every
example. An answer is returned only when every consistent expression agrees on
the query (global agreement) -> deterministic, no guessing, no leakage.

per_bit_ops(examples, question, answer) returns, for each output bit, a
(family, primary, secondary) tuple in the bit_manipulation reasoner's vocabulary
that is consistent with the example column AND yields the given answer bit,
trying families in the reasoner's section order. Used to render the rule in the
existing CoT format.
"""

from __future__ import annotations

N = 8
OPS = ("AND", "OR", "XOR")
# reasoner section order for deterministic per-bit op choice
PAIR_ORDER = ("AND", "OR", "XOR", "AND-NOT", "OR-NOT", "XOR-NOT")


def _bits(s: str):
    return [int(c) for c in s]


def _build_views(inputs_bits, n_ex):
    views = {}
    for kind in ("ROT", "SHL", "SHR"):
        for k in range(N):
            vec = []
            for j in range(N):
                if kind == "ROT":
                    src, valid = (j - k) % N, True
                elif kind == "SHL":
                    src, valid = j + k, (j + k < N)
                else:
                    src, valid = j - k, (j - k >= 0)
                m = 0
                if valid:
                    for e in range(n_ex):
                        if inputs_bits[e][src]:
                            m |= 1 << e
                vec.append(m)
            views[(kind, k)] = vec
    return views


def _q_views(qin):
    qv = {}
    for kind in ("ROT", "SHL", "SHR"):
        for k in range(N):
            row = []
            for j in range(N):
                if kind == "ROT":
                    row.append(qin[(j - k) % N])
                elif kind == "SHL":
                    row.append(qin[j + k] if j + k < N else 0)
                else:
                    row.append(qin[j - k] if j - k >= 0 else 0)
            qv[(kind, k)] = row
    return qv


def solve(examples, question):
    """examples: list of (input_str, output_str). Return 8-char answer or None."""
    if len(question) != N:
        return None
    inputs_bits = [_bits(i) for i, _ in examples]
    n_ex = len(inputs_bits)
    if n_ex == 0 or any(len(i) != N for i, _ in examples) or any(len(o) != N for _, o in examples):
        return None
    full = (1 << n_ex) - 1
    out_targets = []
    for j in range(N):
        m = 0
        for e, (_, o) in enumerate(examples):
            if _bits(o)[j]:
                m |= 1 << e
        out_targets.append(m)

    V = _build_views(inputs_bits, n_ex)
    QV = _q_views(_bits(question))
    keys = list(V.keys())

    def matches(vec):
        return all((vec[j] & full) == out_targets[j] for j in range(N))

    def qstr(qb):
        return "".join(str(b & 1) for b in qb)

    answers = set()

    def add(qb):
        answers.add(qstr(qb))
        return len(answers) > 1

    # variants: each view with optional NOT
    var = []
    for nm in keys:
        v = V[nm]
        q = QV[nm]
        var.append((v, q))
        var.append(([(~v[j]) & full for j in range(N)], [1 - q[j] for j in range(N)]))

    # 1-transform
    for v, q in var:
        if matches(v) and add(q):
            return None

    # 2-transform
    two = []
    for va, qa in var:
        for vb, qb in var:
            for op in OPS:
                if op == "AND":
                    vec = [va[j] & vb[j] for j in range(N)]
                    qrow = [qa[j] & qb[j] for j in range(N)]
                elif op == "OR":
                    vec = [va[j] | vb[j] for j in range(N)]
                    qrow = [qa[j] | qb[j] for j in range(N)]
                else:
                    vec = [(va[j] ^ vb[j]) & full for j in range(N)]
                    qrow = [qa[j] ^ qb[j] for j in range(N)]
                if matches(vec) and add(qrow):
                    return None
                two.append((vec, qrow))

    # 3-transform: (inner) op2 C
    for vinner, qinner in two:
        for vc, qc in var:
            for op2 in OPS:
                if op2 == "AND":
                    vec = [vinner[j] & vc[j] for j in range(N)]
                elif op2 == "OR":
                    vec = [vinner[j] | vc[j] for j in range(N)]
                else:
                    vec = [(vinner[j] ^ vc[j]) & full for j in range(N)]
                if matches(vec):
                    if op2 == "AND":
                        qrow = [qinner[j] & qc[j] for j in range(N)]
                    elif op2 == "OR":
                        qrow = [qinner[j] | qc[j] for j in range(N)]
                    else:
                        qrow = [qinner[j] ^ qc[j] for j in range(N)]
                    if add(qrow):
                        return None

    if len(answers) == 1:
        return next(iter(answers))
    return None


def _eval_pair(fam, a, b):
    if fam == "AND":
        return a & b
    if fam == "OR":
        return a | b
    if fam == "XOR":
        return a ^ b
    if fam == "AND-NOT":
        return a & (1 - b)
    if fam == "OR-NOT":
        return a | (1 - b)
    if fam == "XOR-NOT":
        return a ^ (1 - b)
    return None


def per_bit_ops(examples, question, answer):
    """Return list of 8 (family, primary, secondary) consistent with each example
    column AND yielding answer[j] on the query. Families tried in reasoner order.
    Returns None if any bit cannot be expressed with <=2 inputs (should not happen
    for solve()-accepted problems in the 2-input class)."""
    inputs = [_bits(i) for i, _ in examples]
    outs = [_bits(o) for _, o in examples]
    q = _bits(question)
    A = _bits(answer)
    result = []
    for j in range(N):
        col = [o[j] for o in outs]
        pick = None
        # Identity
        for p in range(N):
            if all(inp[p] == c for inp, c in zip(inputs, col)) and q[p] == A[j]:
                pick = ("I", p, None)
                break
        # NOT
        if pick is None:
            for p in range(N):
                if all((1 - inp[p]) == c for inp, c in zip(inputs, col)) and (1 - q[p]) == A[j]:
                    pick = ("NOT", p, None)
                    break
        # Constant
        if pick is None:
            if all(c == 0 for c in col) and A[j] == 0:
                pick = ("0", None, None)
            elif all(c == 1 for c in col) and A[j] == 1:
                pick = ("1", None, None)
        # 2-input families
        if pick is None:
            for fam in PAIR_ORDER:
                for p in range(N):
                    for r in range(N):
                        if p == r:
                            continue
                        if all(_eval_pair(fam, inp[p], inp[r]) == c for inp, c in zip(inputs, col)) \
                           and _eval_pair(fam, q[p], q[r]) == A[j]:
                            pick = (fam, p, r)
                            break
                    if pick:
                        break
                if pick:
                    break
        if pick is None:
            return None
        result.append(pick)
    return result


# ---------------------------------------------------------------------------
# Sandbox extension: solve_program mirrors solve()'s exact enumeration order
# but also returns the FIRST consistent program for narration. Deterministic.
# Term = (kind, k, neg). Program = (t1,) | (t1, op, t2) | (t1, op, t2, op2, t3)
# with left-assoc semantics ((t1 op t2) op2 t3).
# ---------------------------------------------------------------------------

def solve_program(examples, question):
    if len(question) != N:
        return None, None
    inputs_bits = [_bits(i) for i, _ in examples]
    n_ex = len(inputs_bits)
    if n_ex == 0 or any(len(i) != N for i, _ in examples) or any(len(o) != N for _, o in examples):
        return None, None
    full = (1 << n_ex) - 1
    out_targets = []
    for j in range(N):
        m = 0
        for e, (_, o) in enumerate(examples):
            if _bits(o)[j]:
                m |= 1 << e
        out_targets.append(m)

    V = _build_views(inputs_bits, n_ex)
    QV = _q_views(_bits(question))
    keys = list(V.keys())

    def matches(vec):
        return all((vec[j] & full) == out_targets[j] for j in range(N))

    def qstr(qb):
        return "".join(str(b & 1) for b in qb)

    answers = set()
    first_prog = [None]

    def add(qb, prog):
        answers.add(qstr(qb))
        if first_prog[0] is None:
            first_prog[0] = prog
        return len(answers) > 1

    var = []
    for nm in keys:
        v = V[nm]
        q = QV[nm]
        kind, k = nm
        var.append(((kind, k, False), v, q))
        var.append(((kind, k, True), [(~v[j]) & full for j in range(N)],
                    [1 - q[j] for j in range(N)]))

    for d, v, q in var:
        if matches(v) and add(q, (d,)):
            return None, None

    two = []
    for da, va, qa in var:
        for db, vb, qb in var:
            for op in OPS:
                if op == "AND":
                    vec = [va[j] & vb[j] for j in range(N)]
                    qrow = [qa[j] & qb[j] for j in range(N)]
                elif op == "OR":
                    vec = [va[j] | vb[j] for j in range(N)]
                    qrow = [qa[j] | qb[j] for j in range(N)]
                else:
                    vec = [(va[j] ^ vb[j]) & full for j in range(N)]
                    qrow = [qa[j] ^ qb[j] for j in range(N)]
                if matches(vec) and add(qrow, (da, op, db)):
                    return None, None
                two.append(((da, op, db), vec, qrow))

    for dinner, vinner, qinner in two:
        for dc, vc, qc in var:
            for op2 in OPS:
                if op2 == "AND":
                    vec = [vinner[j] & vc[j] for j in range(N)]
                elif op2 == "OR":
                    vec = [vinner[j] | vc[j] for j in range(N)]
                else:
                    vec = [(vinner[j] ^ vc[j]) & full for j in range(N)]
                if matches(vec):
                    if op2 == "AND":
                        qrow = [qinner[j] & qc[j] for j in range(N)]
                    elif op2 == "OR":
                        qrow = [qinner[j] | qc[j] for j in range(N)]
                    else:
                        qrow = [qinner[j] ^ qc[j] for j in range(N)]
                    if add(qrow, dinner + (op2, dc)):
                        return None, None

    if len(answers) == 1:
        return next(iter(answers)), first_prog[0]
    return None, None


def term_name(term):
    kind, k, neg = term
    return ("NOT " if neg else "") + f"{kind}{k}"


def apply_term(word, term):
    """Apply one (kind, k, neg) view to an 8-char bit word."""
    kind, k, neg = term
    out = []
    for j in range(N):
        if kind == "ROT":
            v = word[(j - k) % N]
        elif kind == "SHL":
            v = word[j + k] if j + k < N else "0"
        else:
            v = word[j - k] if j - k >= 0 else "0"
        out.append(v)
    s = "".join(out)
    if neg:
        s = "".join("1" if c == "0" else "0" for c in s)
    return s


def combine_words(a, b, op):
    if op == "AND":
        return "".join("1" if x == "1" and y == "1" else "0" for x, y in zip(a, b))
    if op == "OR":
        return "".join("1" if x == "1" or y == "1" else "0" for x, y in zip(a, b))
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def eval_program(word, prog):
    """Evaluate a program tuple on an 8-char word; returns (term_words, result)."""
    terms = prog[0::2]
    ops = prog[1::2]
    words = [apply_term(word, t) for t in terms]
    acc = words[0]
    for op, w in zip(ops, words[1:]):
        acc = combine_words(acc, w, op)
    return words, acc


def program_name(prog):
    terms = prog[0::2]
    ops = prog[1::2]
    s = f"({term_name(terms[0])})"
    for op, t in zip(ops, terms[1:]):
        s = f"{s} {op} ({term_name(t)})"
    return s
