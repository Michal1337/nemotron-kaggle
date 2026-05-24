"""Equation numeric reasoning generator."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from reasoners.store_types import Problem

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first."""
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Rare operations tried if common ones don't match."""
    out: list[tuple[str, str]] = []
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    # Extension ops (originally from Alice's solver pool) — added to the END so
    # huikang's existing op-selection order is preserved on previously-known
    # problems.
    import math as _math
    if a != 0 or b != 0:
        out.append(("gcd", str(_math.gcd(abs(a), abs(b)))))
        try:
            lcm_val = _math.lcm(abs(a), abs(b))
            out.append(("lcm", str(lcm_val)))
        except Exception:
            pass
    out.append(("absolute difference - 2", str(abs(a - b) - 2)))
    out.append(("absolute difference + 2", str(abs(a - b) + 2)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare."""
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _expr(name: str, a: str, b: str) -> str:
    """Return the math expression for an operation, e.g. '94 + 48'."""
    if name == "addition":
        return f"{a} + {b}"
    if name == "subtraction (a-b)":
        return f"{a} - {b}"
    if name == "reverse subtraction (b-a)":
        return f"{b} - {a}"
    if name == "multiplication":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b}"
        return f"{a} * {b}"
    if name == "absolute difference":
        return f"|{a} - {b}|"
    if name == "negated absolute difference":
        return f"-|{a} - {b}|"
    if name == "concatenation":
        return f"{a} || {b}"
    if name == "reverse concatenation":
        return f"{b} || {a}"
    if name == "multiply+1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} + 1"
        return f"{a} * {b} + 1"
    if name == "multiply-1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} - 1"
        return f"{a} * {b} - 1"
    if name == "add+1":
        return f"{a} + {b} + 1"
    if name == "add-1":
        return f"{a} + {b} - 1"
    if name == "sub+1":
        return f"{a} - {b} + 1"
    if name == "sub-1":
        return f"{a} - {b} - 1"
    if name == "integer division (a/b)":
        return f"{a} / {b}"
    if name == "modulo (a mod b)":
        return f"{a} mod {b}"
    if name == "reverse division (b/a)":
        return f"{b} / {a}"
    if name == "reverse modulo (b mod a)":
        return f"{b} mod {a}"
    if name == "max mod min":
        big, small = (a, b) if int(a) >= int(b) else (b, a)
        return f"max({a},{b}) mod min({a},{b}) = {big} mod {small}"
    if name == "gcd":
        return f"gcd({a}, {b})"
    if name == "lcm":
        return f"lcm({a}, {b})"
    if name == "absolute difference - 2":
        return f"|{a} - {b}| - 2"
    if name == "absolute difference + 2":
        return f"|{a} - {b}| + 2"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = a[0], a[1], b[0], b[1]
        if name == "digit absolute diff":
            return f"|{d1}-{d3}| || |{d2}-{d4}|"
        if name == "digit add mod10":
            return f"({d1}+{d3})%10 || ({d2}+{d4})%10"
        if name == "digit sub mod10":
            return f"({d1}-{d3})%10 || ({d2}-{d4})%10"
        if name == "cross multiply":
            return f"{d1}*{d3} + {d2}*{d4}"
        if name == "cross multiply rev":
            return f"{d1}*{d4} + {d2}*{d3}"
        if name == "digit multiply":
            return f"{d1}*{d3} || {d2}*{d4}"
        if name == "digit multiply rev":
            return f"{d1}*{d4} || {d2}*{d3}"
        if name == "digit sum diff":
            return f"({d1}+{d2}) - ({d3}+{d4})"
        if name == "digit sum sum":
            return f"({d1}+{d2}) + ({d3}+{d4})"
        if name == "digit product diff":
            return f"{d1}*{d2} - {d3}*{d4}"
        if name == "digit product sum":
            return f"{d1}*{d2} + {d3}*{d4}"
        if name == "determinant":
            return f"{d1}*{d4} - {d2}*{d3}"
        if name == "abs determinant":
            return f"|{d1}*{d4} - {d2}*{d3}|"
    return ""


def _expr_intermediate(name: str, a: str, b: str) -> str:
    """Return intermediate evaluated form for operations with multiplications, else ''."""
    ia, ib = int(a), int(b)
    if name in ("multiply+1", "multiply-1", "multiplication") and len(a) >= 2:
        # Decompose a by place value: 70 → [70, 0], 73 → [70, 3]
        places = [int(d) * (10 ** (len(a) - 1 - i)) for i, d in enumerate(a)]
        decomp = " + ".join(f"{p} * {b}" for p in places)
        evald = " + ".join(str(p * ib) for p in places)
        product_sum = sum(p * ib for p in places)
        if name == "multiply+1":
            return f"{decomp} + 1 = {evald} + 1 = {product_sum} + 1"
        if name == "multiply-1":
            return f"{decomp} - 1 = {evald} - 1 = {product_sum} - 1"
        return f"{decomp} = {evald}"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        if name == "cross multiply":
            return f"{d1 * d3} + {d2 * d4}"
        if name == "cross multiply rev":
            return f"{d1 * d4} + {d2 * d3}"
        if name == "digit multiply":
            return f"{d1 * d3} || {d2 * d4}"
        if name == "digit multiply rev":
            return f"{d1 * d4} || {d2 * d3}"
        if name == "digit product diff":
            return f"{d1 * d2} - {d3 * d4}"
        if name == "digit product sum":
            return f"{d1 * d2} + {d3 * d4}"
        if name == "determinant":
            return f"{d1 * d4} - {d2 * d3}"
        if name == "abs determinant":
            return f"|{d1 * d4} - {d2 * d3}|"
    return ""


def _rev(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


@dataclass
class FoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str


def _apply_op(found: FoundOp, a_str: str, b_str: str) -> tuple[str, list[str]]:
    """Apply the found operation and return (result, explanation_lines)."""
    steps: list[str] = []
    ta = a_str[::-1] if found.rev_ops else a_str
    tb = b_str[::-1] if found.rev_ops else b_str

    # Header line always present
    if found.rev_ops and found.rev_res:
        steps.append(
            f"reversed operands [{a_str}->{ta}, {b_str}->{tb}] and reversed result"
        )
    elif found.rev_ops:
        steps.append(f"reversed operands [{a_str}->{ta}, {b_str}->{tb}]")
    elif found.rev_res:
        steps.append("reversed result")
    else:
        steps.append("identity")

    # Find the matching candidate
    raw_result = ""
    for name, res in _all_candidates(int(ta), int(tb), ta, tb):
        if name == found.op_name:
            raw_result = res
            break

    final = _rev(raw_result) if found.rev_res else raw_result

    expr = _expr(found.op_name, ta, tb)
    inter = _expr_intermediate(found.op_name, ta, tb)
    if expr and inter:
        detail = f" {expr} = {inter} ="
    elif expr:
        detail = f" {expr} ="
    else:
        detail = ""
    val = f"{raw_result} -rev-> {final}" if found.rev_res else final
    steps.append(f"{found.op_name} f({ta}, {tb}) ={detail} {val}")

    if found.fmt == "pre":
        final = found.op_char + final
        steps.append(f"Prefix operator: {final}")
    elif found.fmt == "neg_suffix":
        if final.startswith("-"):
            old = final
            final = final[1:] + found.op_char
            steps.append(
                f"Result is negative - we add back the operator suffix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no suffix needed: 【{final}】")
    elif found.fmt == "neg_prefix":
        if final.startswith("-"):
            old = final
            final = found.op_char + final[1:]
            steps.append(
                f"Result is negative - we add back the operator prefix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no prefix needed: 【{final}】")

    return final, steps


def _detect_fmt(op_char: str, group: list[tuple[str, str, str]]) -> tuple[str, list[tuple[str, str, str]]]:
    """Detect prefix/suffix sigil pattern for an operator group.

    Returns ``(fmt, transformed_group)`` where ``fmt`` is one of:
      - ``"num"``         no sigil detected
      - ``"neg_prefix"``  outputs start with a sign-marker char (either ``-``
                          or the operator char itself); transformed_group
                          replaces the marker with a leading ``-``.
      - ``"neg_suffix"``  outputs end with a sign-marker char; transformed
                          group rewrites it as a leading ``-``.

    Logic isolated from the narrator so unit tests can pin the detector
    behavior without parsing CoT text.
    """
    any_neg_suffixed = op_char != "-" and any(
        out.endswith("-") and len(out) > 1 for _, _, out in group
    )
    any_neg_prefixed = op_char != "-" and any(
        out.startswith("-") and len(out) > 1 for _, _, out in group
    )
    any_suffixed = any(
        out.endswith(op_char) and len(out) > 1 for _, _, out in group
    )
    any_prefixed = any(
        out.startswith(op_char) and len(out) > 1 for _, _, out in group
    )
    if any_neg_suffixed:
        return "neg_suffix", [
            (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
            for a, b, out in group
        ]
    if any_neg_prefixed:
        return "neg_prefix", list(group)
    if any_suffixed:
        return "neg_suffix", [
            (a, b, "-" + out[: -len(op_char)] if out.endswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
    if any_prefixed:
        return "neg_prefix", [
            (a, b, "-" + out[len(op_char):] if out.startswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
    return "num", list(group)


def _sigil_reasoning_block(
    op_char: str,
    fmt: str,
    group: list[tuple[str, str, str]],
    transformed_group: list[tuple[str, str, str]],
) -> list[str]:
    """Emit the per-operator sigil reasoning paragraph.

    Returns an empty list when ``fmt == "num"`` or when the transform is a
    no-op (e.g. ``op_char == "-"`` and the output already starts with ``-``,
    so stripping the sigil and re-attaching ``-`` yields the same string).
    Keeping the no-op suppression here means callers never emit the dead
    "We now consider the outputs to be ..." line that just echoes the prior
    "outputs are ..." line.
    """
    if fmt == "num":
        return []

    is_prefix = fmt == "neg_prefix"
    position_word = "begin" if is_prefix else "end"

    # Identify the marker character actually appearing on the outputs:
    #   - For op_char == "-" with leading "-" on outputs: the marker IS "-"
    #     (the natural minus sign, which happens to coincide with the op).
    #   - For non-"-" op_char appearing as prefix/suffix: the marker is the
    #     op_char itself acting as a sigil.
    if is_prefix:
        sigil_examples = [(a, b, out) for (a, b, out) in group if len(out) > 1 and (out.startswith(op_char) or out.startswith("-"))]
    else:
        sigil_examples = [(a, b, out) for (a, b, out) in group if len(out) > 1 and out.endswith(op_char)]
    if not sigil_examples:
        return []
    marker_char = sigil_examples[0][2][0] if is_prefix else sigil_examples[0][2][-1]

    samples_str = "; ".join(f"{a}{op_char}{b} = {out}" for a, b, out in sigil_examples[:3])

    # Pair each sigil example with its decoded (transformed) form for the
    # "decoded as" enumeration. When the transform is identity (the op-char
    # IS "-"), we still emit a paragraph but skip the redundant decoding
    # list — there's nothing to decode.
    tmap = {(a, b, out): t for (a, b, out), (_, _, t) in zip(group, transformed_group)}
    decoded_pairs = [
        (out, tmap.get((a, b, out), out))
        for a, b, out in sigil_examples
        if tmap.get((a, b, out), out) != out
    ]

    lines: list[str] = []
    if op_char == "-" and marker_char == "-":
        # Natural minus sign case — explain WHY no decoding step is needed.
        first_neg = sigil_examples[0][2]  # e.g. "-91"
        bare_digits = first_neg[1:]       # "91"
        # If the digits have a leading zero (e.g. "-06"), call that out
        # explicitly, since it signals fixed-width zero padding.
        if bare_digits.startswith("0") and len(bare_digits) > 1:
            pad_note = " (with leading-zero padding)"
        else:
            pad_note = ""
        lines.append(
            f"Observation: for operator 【{op_char}】, some outputs {position_word} with "
            f"`-` ({samples_str}). Since `-` is the operator AND the standard "
            f"arithmetic minus sign, the leading `-` indicates the operation "
            f"can produce signed (negative) results. No decoding step is "
            f"needed — we read `{first_neg}` directly as the value -{int(bare_digits)}{pad_note}, "
            f"and we expect to find a rule for 【{op_char}】 that yields negative "
            f"outputs (e.g. negated absolute difference or signed subtraction)."
        )
    elif is_prefix:
        decoded_str = ", ".join(f"`{out}` -> `{t}`" for out, t in decoded_pairs[:3])
        lines.append(
            f"Observation: for operator 【{op_char}】, some outputs begin with "
            f"the character `{op_char}` ({samples_str}). `{op_char}` is not a "
            f"decimal digit (0-9), so it cannot be part of the numeric "
            f"output. The most plausible reading is that `{op_char}` is a "
            f"sign marker: an output `{op_char}X` encodes the negative value "
            f"-X. We decode these as: {decoded_str}. After we infer the rule, "
            f"if our answer is negative we will re-attach the `{op_char}` "
            f"prefix to encode the sign."
        )
    else:
        decoded_str = ", ".join(f"`{out}` -> `{t}`" for out, t in decoded_pairs[:3])
        lines.append(
            f"Observation: for operator 【{op_char}】, some outputs end with "
            f"the character `{op_char}` ({samples_str}). `{op_char}` is not a "
            f"decimal digit (0-9), so it cannot be part of the numeric "
            f"output. The most plausible reading is that `{op_char}` is a "
            f"sign marker: an output `X{op_char}` encodes the negative value "
            f"-X. We decode these as: {decoded_str}. After we infer the rule, "
            f"if our answer is negative we will re-attach the `{op_char}` "
            f"suffix to encode the sign."
        )
    return lines


_DEFAULT_TRANSFORM_ORDER = ((True, True), (False, False), (True, False), (False, True))


def _resolve_transform_order(preferred_mode: str | None) -> tuple[tuple[bool, bool], ...]:
    """Reorder the (rev_ops, rev_res) search to put preferred_mode first.

    Lets a caller (e.g. the cryptarithm narrator that decoded an Alice-solved
    problem) nudge huikang to pick a particular interpretation when the example
    set is satisfiable under multiple transforms. The full 4-combo space is
    still explored, just in a different order — so the first match (which the
    reasoner commits to) aligns with the caller's known-correct interpretation.
    """
    if preferred_mode is None:
        return _DEFAULT_TRANSFORM_ORDER
    if preferred_mode in ("standard", "identity", "none"):
        head = (False, False)
    elif preferred_mode in ("little_endian", "alice", "rev_ops_res"):
        head = (True, True)
    elif preferred_mode == "rev_ops":
        head = (True, False)
    elif preferred_mode == "rev_res":
        head = (False, True)
    else:
        return _DEFAULT_TRANSFORM_ORDER
    rest = tuple(combo for combo in _DEFAULT_TRANSFORM_ORDER if combo != head)
    return (head,) + rest


def _common_candidates_v2(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """v2 candidate order: signed ops (sub_signed, rsub_signed) tried before
    absdiff/neg_absdiff.

    Rationale: when example outputs are all non-negative AND ``a >= b`` in
    every example, both ``abs(a-b)`` and ``sub_signed(a-b)`` produce the
    same example values; the legacy order picks absdiff (unsigned) which
    then mispredicts the QUERY when ``a < b``. The signed op is strictly
    more general (recovers absdiff when the example happens to have
    ``a >= b``), so preferring it is safe on cases where the rule is
    genuinely absdiff (those have examples with both ``a > b`` and ``a <
    b`` — sub_signed won't fit, narrator falls through to absdiff).
    """
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    # signed ops first — narrator picks them when ambiguous on examples
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("multiplication", str(a * b)))
    return out


def _apply_op_v2(found: FoundOp, a_str: str, b_str: str) -> tuple[str, list[str]]:
    """Cleaner-format variant of ``_apply_op`` for the v2 deduce narrator.

    Same logic as ``_apply_op`` but drops the redundant 【】 around result
    numbers (only the operator char keeps brackets).
    """
    steps: list[str] = []
    ta = a_str[::-1] if found.rev_ops else a_str
    tb = b_str[::-1] if found.rev_ops else b_str

    if found.rev_ops and found.rev_res:
        steps.append(f"reversed operands [{a_str}->{ta}, {b_str}->{tb}] and reversed result")
    elif found.rev_ops:
        steps.append(f"reversed operands [{a_str}->{ta}, {b_str}->{tb}]")
    elif found.rev_res:
        steps.append("reversed result")
    else:
        steps.append("identity")

    raw_result = ""
    for name, res in _all_candidates(int(ta), int(tb), ta, tb):
        if name == found.op_name:
            raw_result = res
            break
    final = _rev(raw_result) if found.rev_res else raw_result

    expr = _expr(found.op_name, ta, tb)
    inter = _expr_intermediate(found.op_name, ta, tb)
    if expr and inter:
        detail = f" {expr} = {inter} ="
    elif expr:
        detail = f" {expr} ="
    else:
        detail = ""
    val = f"{raw_result} -rev-> {final}" if found.rev_res else final
    steps.append(f"{found.op_name} f({ta}, {tb}) ={detail} {val}")

    if found.fmt == "neg_suffix":
        if final.startswith("-"):
            old = final
            final = final[1:] + found.op_char
            steps.append(
                f"Result is negative, re-attaching the 【{found.op_char}】 suffix: {old} -> {final}"
            )
        else:
            steps.append(f"Result is non-negative, no suffix needed: {final}")
    elif found.fmt == "neg_prefix":
        if final.startswith("-"):
            old = final
            final = found.op_char + final[1:]
            steps.append(
                f"Result is negative, re-attaching the 【{found.op_char}】 prefix: {old} -> {final}"
            )
        else:
            steps.append(f"Result is non-negative, no prefix needed: {final}")
    else:
        # fmt == "num" — no sigil was detected in examples. When the
        # signed rule produces a negative for THIS query, keep the natural
        # `-` prefix (don't strip) so the answer encodes the sign correctly.
        # This addresses the under-attach failure mode where no example
        # happened to be negative, so fmt stayed "num", but the actual rule
        # is sign-producing and the query falls on the negative side.
        if final.startswith("-"):
            steps.append(f"Result is negative: {final}")

    return final, steps


def _build_deduce_v2(
    problem: Problem,
    preferred_mode: str | None,
    query_op_override: tuple[str, bool, bool] | None,
) -> str | None:
    """V2 narrator for ``equation_numeric_deduce``.

    Differences from the legacy narrator:
      1. Question + query operator stated up front (model sees the target
         before the example analysis).
      2. Per-operator sigil reasoning paragraph that explicitly distinguishes
         "natural minus sign" vs "op-char as sigil marker", instead of an
         awkward "We now consider the outputs to be ..." line that sometimes
         echoes the prior line verbatim.
      3. 【】 reserved for operator/sigil characters; numeric results are
         emitted bare.
      4. Same underlying search and ``_apply_op_v2`` logic so the boxed
         answer matches the legacy narrator on cases that worked before.
    """
    q_match = _EXPR_RE.fullmatch(str(problem.question))
    if q_match is None:
        return None
    q_op = q_match.group(2)
    qa, qb = q_match.group(1), q_match.group(3)

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if m is None:
            continue
        parsed.append((m.group(1), m.group(2), m.group(3), str(ex.output_value)))

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    if q_op not in by_op:
        # deduce by definition always has the query op in examples; if it
        # somehow doesn't, hand off to the legacy narrator (handles guess).
        return None

    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    for op_char, group in by_op.items():
        fmt, tgroup = _detect_fmt(op_char, group)
        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = tgroup

    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append(f"The question is: {problem.question}")
    lines.append(f"The query operator is 【{q_op}】. We need to determine the value of {qa}{q_op}{qb}.")
    lines.append("")
    lines.append("Examples:")
    for a, op, b, out in parsed:
        lines.append(f"  {a}{op}{b} = {out}")
    lines.append("")

    op_list = list(by_op.keys())
    lines.append(f"The example operators are: {', '.join(op_list)}.")
    lines.append(f"The query operator 【{q_op}】 appears in the examples, so we can infer its rule directly.")

    # Emit sigil reasoning for the query op only (other ops aren't used).
    sig_lines = _sigil_reasoning_block(
        q_op, detected_fmts[q_op], by_op[q_op], transformed_groups[q_op]
    )
    if sig_lines:
        lines.append("")
        lines.extend(sig_lines)

    # Search for the rule on the query operator only.
    group = transformed_groups[q_op]
    examples_str = ", ".join(f"{a}{q_op}{b} = {out}" for a, b, out in group)
    lines.append("")
    lines.append(f"Looking at operator 【{q_op}】 [{examples_str}]:")

    found: FoundOp | None = None
    n_ex = len(group)
    transform_order = _resolve_transform_order(preferred_mode)
    candidate_sets = [("common", _common_candidates), ("rare", _rare_candidates)]
    cycled = list(group)

    for set_name, cand_fn in candidate_sets:
        for rev_ops, rev_res in transform_order:
            # Operand pair display — tuple notation so pairs are visually
            # distinct (legacy "64 61, 21 05" looks like 4 loose numbers).
            label = f"{set_name} operations"
            if rev_ops:
                rev_parts = ", ".join(f"({ax},{bx})->({ax[::-1]},{bx[::-1]})" for ax, bx, _ in cycled)
                if rev_res:
                    label += f" reversed operands [{rev_parts}] and reversed result"
                else:
                    label += f" reversed operands [{rev_parts}]"
            elif rev_res:
                id_parts = ", ".join(f"({ax},{bx})" for ax, bx, _ in cycled)
                label += f" identity operands [{id_parts}] reversed result"
            else:
                id_parts = ", ".join(f"({ax},{bx})" for ax, bx, _ in cycled)
                label += f" on identity [{id_parts}]"
            # The "target" we show in the header must be what the candidate
            # f(a,b) itself should evaluate to — i.e. the value that, after
            # any rev_res reversal, equals the example's gold output. When
            # rev_res is on, that's _rev(exp), not exp.
            def _disp_target(exp: str) -> str:
                return _rev(exp) if rev_res else exp
            if rev_ops:
                all_expected = ", ".join(
                    f"({ax[::-1]},{bx[::-1]})->{_disp_target(exp)}" for ax, bx, exp in cycled
                )
            else:
                all_expected = ", ".join(
                    f"({ax},{bx})->{_disp_target(exp)}" for ax, bx, exp in cycled
                )
            # Now the displayed target IS what f(...) must produce directly.
            # The per-candidate "-rev->" annotation then makes the reversal
            # step explicit. No misleading "expected" label needed.
            lines.append(f"  Trying {label} [expected f(a,b) {all_expected}]:")

            ca_str, cb_str = cycled[0][0], cycled[0][1]
            cta = ca_str[::-1] if rev_ops else ca_str
            ctb = cb_str[::-1] if rev_ops else cb_str
            candidates = cand_fn(int(cta), int(ctb), cta, ctb)
            cand_idx = 0
            for cand_name, _ in candidates:
                rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                cand_idx += 1
                parts: list[str] = []
                all_pass = True
                for i, (ax, bx, exp_x) in enumerate(rotated):
                    rax = ax[::-1] if rev_ops else ax
                    rbx = bx[::-1] if rev_ops else bx
                    raw = next(
                        r for n, r in _all_candidates(int(rax), int(rbx), rax, rbx) if n == cand_name
                    )
                    expr_x = _expr(cand_name, rax, rbx)
                    inter_x = _expr_intermediate(cand_name, rax, rbx)
                    if expr_x and inter_x:
                        detail_x = f" {expr_x} = {inter_x} ="
                    elif expr_x:
                        detail_x = f" {expr_x} ="
                    else:
                        detail_x = ""
                    fin = _rev(raw) if rev_res else raw
                    status = "match" if fin == exp_x else "wrong"
                    if fin != exp_x:
                        all_pass = False
                    # Put the status next to the value it's comparing
                    # (the header's advertised target = raw when rev_res,
                    # fin when not). For matches, also show the post-rev
                    # value as a side-channel confirming the example's
                    # gold string; for wrongs, suppress the rev tail —
                    # it would just be noise.
                    if rev_res and status == "match":
                        val = f"{raw} {status} -rev-> {fin}"
                    elif rev_res:
                        val = f"{raw} {status}"
                    else:
                        val = f"{fin} {status}"
                    arrow = i > 0
                    if arrow:
                        parts.append(f"f({rax},{rbx}) ->{detail_x} {val}")
                    else:
                        parts.append(f"f({rax}, {rbx}) ={detail_x} {val}")
                    if fin != exp_x:
                        break
                if all_pass:
                    if found is not None:
                        # Multiple rules fit the examples. We selection-
                        # commit to the FIRST one found and keep listing
                        # later matches for completeness, but they don't
                        # change the result we apply to the query.
                        parts.append(
                            f"correct, but already committed to {found.op_name}"
                            " (first rule that fit all examples) — keeping that choice"
                        )
                    else:
                        summary: list[str] = []
                        if rev_ops:
                            summary.append("reversed operands")
                        if rev_res:
                            summary.append("reversed result")
                        summary.append(cand_name)
                        parts.append(
                            "correct, this is the first rule that fits all "
                            "examples — committing to it. actions: "
                            + ", ".join(summary)
                        )
                lines.append(f"    {cand_name} " + ", ".join(parts))
                if all_pass and found is None:
                    found = FoundOp(
                        op_name=cand_name,
                        rev_ops=rev_ops,
                        rev_res=rev_res,
                        fmt=detected_fmts[q_op],
                        op_char=q_op,
                    )

    if found is None:
        return None

    # Apply to query
    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    result_val, steps = _apply_op_v2(found, qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: {result_val}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)


# ---------------- v2 guess builder ----------------

# R1: example-op signature → query rule (highest priority). Mined from
# the dataset by majority-vote across all guess problems sharing the same
# sorted tuple of inferred example op rules. Only signatures with >=2
# pids are kept; we use the simplest fitting rule by majority count.
# Pids whose chosen rule doesn't produce gold under R1 fall through to
# R2/R3/R4.
_GUESS_R1 = {
    ('absolute difference', 'addition'): 'multiply-1',
    ('addition', 'multiply-1'): 'absolute difference',
    ('addition', 'negated absolute difference'): 'concatenation',
    ('multiplication', 'negated absolute difference'): 'addition',
    ('absolute difference', 'multiplication'): 'add+1',
    ('concatenation', 'multiplication'): 'subtraction (a-b)',
    ('max mod min', 'multiplication'): 'addition',
    ('multiplication',): 'addition',
    ('addition',): 'multiplication',
    ('addition', 'multiplication'): 'subtraction (a-b)',
    ('addition', 'multiply+1'): 'subtraction (a-b)',
    ('subtraction (a-b)',): 'concatenation',
    ('absolute difference',): 'addition',
    ('absolute difference', 'add-1'): 'multiplication',
    ('addition', 'concatenation'): 'subtraction (a-b)',
    ('concatenation', 'multiply+1'): 'subtraction (a-b)',
    ('concatenation', 'negated absolute difference'): 'addition',
    ('multiply+1', 'negated absolute difference'): 'addition',
    ('negated absolute difference',): 'multiplication',
    ('absolute difference', 'multiply-1'): 'addition',
    ('absolute difference', 'reverse concatenation'): 'addition',
    ('add+1', 'multiplication'): 'subtraction (a-b)',
    ('add+1', 'negated absolute difference'): 'multiplication',
    ('add+1', 'subtraction (a-b)'): 'concatenation',
    ('add-1',): 'multiplication',
    ('add-1', 'multiplication'): 'subtraction (a-b)',
    ('concatenation', 'subtraction (a-b)'): 'concatenation',
    ('max mod min',): 'multiplication',
    ('multiplication', 'reverse concatenation'): 'absolute difference',
    ('multiply+1',): 'subtraction (a-b)',
    ('negated absolute difference', 'reverse concatenation'): 'addition',
}

# R2: natural arithmetic chars — canonical real-world meaning.
_GUESS_R2 = {
    '+': 'addition',
    '-': 'subtraction (a-b)',
    '*': 'multiplication',
}

# R3: per-char dataset conventions (strong/hardcoded mapping).
_GUESS_R3 = {
    '@': 'addition',
    ')': 'multiply-1',
    '#': 'addition',
}

# R4: per-char majority for chars not in R2/R3. Mined ONLY over pids that
# would actually reach R4 (i.e., pids whose example-op signature isn't in
# _GUESS_R1, so R1 doesn't catch them). This gives the correct per-char
# majority for the R4-eligible subset, rather than counting R1-covered
# pids that R4 never sees. Chars where every pid is R1-covered are not
# listed (R4 would never fire for them).
_GUESS_R4 = {
    '!': 'multiplication',
    '%': 'multiplication',
    '&': 'add-1',
    '(': 'multiplication',
    '/': 'multiplication',
    ':': 'add+1',
    '<': 'multiplication',
    '?': 'concatenation',
    '[': 'addition',
    '\\': 'multiply+1',
    '^': 'concatenation',
    '`': 'multiplication',
    '{': 'concatenation',
    '|': 'addition',
    '}': 'concatenation',
}

# Backwards-compat aliases (some imports/tests may still use the old names).
_GUESS_NATURAL = {k: [v] for k, v in _GUESS_R2.items()}
_GUESS_PER_CHAR = {k: [v] for k, v in _GUESS_R3.items()}
_GUESS_PER_CHAR_EXT = _GUESS_R4


def _candidate_rule_names_for(qa: str, qb: str, rev_ops: bool) -> list[str]:
    """Return the full list of candidate op_names (in priority order) we'd
    consider for the query operands under a given transform."""
    qta = qa[::-1] if rev_ops else qa
    qtb = qb[::-1] if rev_ops else qb
    return [n for n, _ in _all_candidates(int(qta), int(qtb), qta, qtb)]


def _apply_rule_with_transform(
    rule_name: str, qa: str, qb: str, rev_ops: bool, rev_res: bool
) -> str | None:
    qta = qa[::-1] if rev_ops else qa
    qtb = qb[::-1] if rev_ops else qb
    raw = next((r for n, r in _all_candidates(int(qta), int(qtb), qta, qtb)
                if n == rule_name), None)
    if raw is None:
        return None
    return _rev(raw) if rev_res else raw


def _pick_guess_rule(
    q_op: str,
    qa: str,
    qb: str,
    gold: str,
    example_op_results: dict[str, FoundOp],
) -> tuple[FoundOp, str] | None:
    """Pick a rule for the unseen query operator, using only example-derived
    heuristics. The chosen rule must produce gold; otherwise return None.
    Returns (FoundOp, reasoning_text) or None.

    Transform pick: deterministic, based on example operators only (no
    gold inspection). The transform with the most example-op occurrences
    wins; ties resolve by _DEFAULT_TRANSFORM_ORDER.
    """
    # Pick the transform deterministically: majority across example ops,
    # tiebreak by _DEFAULT_TRANSFORM_ORDER (identity-first).
    from collections import Counter as _Counter
    transform_counts = _Counter((r.rev_ops, r.rev_res) for r in example_op_results.values())
    seen = set(transform_counts)

    def _transform_priority_key(t):
        # Higher count first; then earlier position in _DEFAULT_TRANSFORM_ORDER.
        try:
            idx = _DEFAULT_TRANSFORM_ORDER.index(t)
        except ValueError:
            idx = len(_DEFAULT_TRANSFORM_ORDER)
        return (-transform_counts.get(t, 0), idx)

    picked_transform = sorted(seen, key=_transform_priority_key)[0]

    # Find rules that fit gold under the picked transform.
    rev_ops, rev_res = picked_transform
    fitting: list[tuple[str, bool, bool]] = []
    for rule_name in _candidate_rule_names_for(qa, qb, rev_ops):
        pred = _apply_rule_with_transform(rule_name, qa, qb, rev_ops, rev_res)
        if pred == gold:
            fitting.append((rule_name, rev_ops, rev_res))
    if not fitting:
        return None

    def _transform_label(rev_ops: bool, rev_res: bool) -> str:
        if not rev_ops and not rev_res:
            return "identity (no reversal)"
        parts = []
        if rev_ops: parts.append("reversed operands")
        if rev_res: parts.append("reversed result")
        return " + ".join(parts)

    def _transform_explanation(rev_ops: bool, rev_res: bool) -> str:
        label = _transform_label(rev_ops, rev_res)
        if len(seen) == 1:
            return (
                f"All example operators apply [{label}]. "
                f"We follow the same convention for the unseen operator."
            )
        # Mixed transforms — pick majority across example ops.
        majority_count = transform_counts[(rev_ops, rev_res)]
        total = sum(transform_counts.values())
        others = sorted(seen - {(rev_ops, rev_res)})
        other_labels = ", ".join(f"[{_transform_label(o, r)}]" for o, r in others)
        return (
            f"Example operators use multiple transforms (also: {other_labels}); "
            f"[{label}] is the most common ({majority_count}/{total} example ops), "
            f"so we apply it to the unseen operator."
        )

    def _make(rule_name, rev_ops, rev_res, rule_reason):
        # Two-step explanation: (1) why this rule, (2) why this transform.
        transform_text = _transform_explanation(rev_ops, rev_res)
        full = f"{rule_reason}\n  Transform: {transform_text}"
        return FoundOp(op_name=rule_name, rev_ops=rev_ops, rev_res=rev_res,
                       fmt='num', op_char=q_op), full

    # Determine which SINGLE rule fires based on preconditions only (NO
    # fallthrough). Priority: R1 (sig) > R2 (qop=+/-/*) > R3 (qop=@/#/))
    # > R4 (qop in R4 map) > R5 (default concat). The chosen rule's
    # prediction must fit gold; otherwise the pid is dropped. This gives
    # consistent training (same precondition always picks same rule).
    ex_sig = tuple(sorted(r.op_name for r in example_op_results.values()))
    r1_rule = _GUESS_R1.get(ex_sig)

    chosen_rule = None
    reasoning = None
    if r1_rule is not None:
        chosen_rule = r1_rule
        ex_sig_str = ", ".join(ex_sig)
        reasoning = (
            f"R1 applies: the example operators' inferred rules are "
            f"({ex_sig_str}); for this signature, the unseen operator "
            f"follows the rule `{r1_rule}`."
        )
    elif q_op in _GUESS_R2:
        chosen_rule = _GUESS_R2[q_op]
        reasoning = (
            f"R2 applies: the character `{q_op}` is a standard arithmetic "
            f"operator symbol; it represents `{chosen_rule}` in real-world "
            f"arithmetic. Applying that interpretation."
        )
    elif q_op in _GUESS_R3:
        chosen_rule = _GUESS_R3[q_op]
        reasoning = (
            f"R3 applies: the character `{q_op}` is not a standard "
            f"arithmetic symbol, but the dataset convention is that "
            f"`{q_op}` represents `{chosen_rule}`. Applying that convention."
        )
    elif q_op in _GUESS_R4:
        chosen_rule = _GUESS_R4[q_op]
        reasoning = (
            f"R4 applies: across this dataset's guess problems, the "
            f"query character `{q_op}` most commonly represents "
            f"`{chosen_rule}`. Applying that convention."
        )
    else:
        chosen_rule = 'concatenation'
        reasoning = (
            f"R5 applies (default fallback): no specific rule maps `{q_op}` "
            f"to a known operation; defaulting to `concatenation`."
        )

    for rule_name, rev_ops, rev_res in fitting:
        if rule_name == chosen_rule:
            return _make(rule_name, rev_ops, rev_res, reasoning)

    # Chosen rule's prediction doesn't fit gold — drop.
    return None


def _guess_common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Reduced common-op pool for the GUESS narrator's per-example-op search.
    Only includes ops that appear in the guessing ruleset (R1-R3 families
    + concatenation). Digit-level, determinant, cross-multiply etc. are
    intentionally omitted — they're not in our guess decision tree, and
    they tend to coincidentally fit examples in ways that confuse the
    pattern-match step."""
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    return out


def _guess_rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Reduced rare-op pool for guess. Keeps the offset variants and the
    mod-family ops (R3 references the mod family). Drops gcd/lcm/digit/
    cross/det ops which are never the right answer in our guess pool AND
    confuse the search by coincidentally fitting examples."""
    out: list[tuple[str, str]] = []
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse modulo (b mod a)", str(b % a)))
    return out


def _format_brief_rule(found: FoundOp) -> str:
    """Compact human-readable description of a FoundOp: rule + transform."""
    parts = [found.op_name]
    if found.rev_ops and found.rev_res:
        parts.append("under reversed operands + reversed result")
    elif found.rev_ops:
        parts.append("under reversed operands")
    elif found.rev_res:
        parts.append("under reversed result")
    return " ".join(parts)


def _emit_op_search(
    lines: list[str],
    op_char: str,
    group: list[tuple[str, str, str]],
    fmt: str,
    preferred_mode: str | None,
    candidate_sets: list[tuple[str, callable]] | None = None,
) -> FoundOp | None:
    """Run the per-op search and emit the trace into ``lines``. Returns the
    first-found rule or None. Used by the guess narrator (deduce has its
    own inline search). Pass ``candidate_sets`` to restrict the op pool;
    default is the full (common + rare) pool."""
    n_ex = len(group)
    transform_order = _resolve_transform_order(preferred_mode)
    if candidate_sets is None:
        candidate_sets = [("common", _common_candidates), ("rare", _rare_candidates)]
    cycled = list(group)
    found: FoundOp | None = None
    for set_name, cand_fn in candidate_sets:
        for rev_ops, rev_res in transform_order:
            label = f"{set_name} operations"
            if rev_ops:
                rev_parts = ", ".join(f"({ax},{bx})->({ax[::-1]},{bx[::-1]})" for ax, bx, _ in cycled)
                if rev_res:
                    label += f" reversed operands [{rev_parts}] and reversed result"
                else:
                    label += f" reversed operands [{rev_parts}]"
            elif rev_res:
                id_parts = ", ".join(f"({ax},{bx})" for ax, bx, _ in cycled)
                label += f" identity operands [{id_parts}] reversed result"
            else:
                id_parts = ", ".join(f"({ax},{bx})" for ax, bx, _ in cycled)
                label += f" on identity [{id_parts}]"
            def _disp_target(exp: str) -> str:
                return _rev(exp) if rev_res else exp
            if rev_ops:
                all_expected = ", ".join(
                    f"({ax[::-1]},{bx[::-1]})->{_disp_target(exp)}" for ax, bx, exp in cycled
                )
            else:
                all_expected = ", ".join(
                    f"({ax},{bx})->{_disp_target(exp)}" for ax, bx, exp in cycled
                )
            lines.append(f"  Trying {label} [expected f(a,b) {all_expected}]:")
            ca_str, cb_str = cycled[0][0], cycled[0][1]
            cta = ca_str[::-1] if rev_ops else ca_str
            ctb = cb_str[::-1] if rev_ops else cb_str
            candidates = cand_fn(int(cta), int(ctb), cta, ctb)
            cand_idx = 0
            for cand_name, _ in candidates:
                rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                cand_idx += 1
                parts: list[str] = []
                all_pass = True
                for i, (ax, bx, exp_x) in enumerate(rotated):
                    rax = ax[::-1] if rev_ops else ax
                    rbx = bx[::-1] if rev_ops else bx
                    raw = next(r for n, r in _all_candidates(int(rax), int(rbx), rax, rbx) if n == cand_name)
                    expr_x = _expr(cand_name, rax, rbx)
                    inter_x = _expr_intermediate(cand_name, rax, rbx)
                    if expr_x and inter_x:
                        detail_x = f" {expr_x} = {inter_x} ="
                    elif expr_x:
                        detail_x = f" {expr_x} ="
                    else:
                        detail_x = ""
                    fin = _rev(raw) if rev_res else raw
                    status = "match" if fin == exp_x else "wrong"
                    if fin != exp_x:
                        all_pass = False
                    if rev_res and status == "match":
                        val = f"{raw} {status} -rev-> {fin}"
                    elif rev_res:
                        val = f"{raw} {status}"
                    else:
                        val = f"{fin} {status}"
                    if i > 0:
                        parts.append(f"f({rax},{rbx}) ->{detail_x} {val}")
                    else:
                        parts.append(f"f({rax}, {rbx}) ={detail_x} {val}")
                    if fin != exp_x:
                        break
                if all_pass:
                    if found is not None:
                        parts.append(
                            f"correct, but already committed to {found.op_name}"
                            " (first rule that fit all examples) — keeping that choice"
                        )
                    else:
                        summary: list[str] = []
                        if rev_ops:
                            summary.append("reversed operands")
                        if rev_res:
                            summary.append("reversed result")
                        summary.append(cand_name)
                        parts.append(
                            "correct, this is the first rule that fits all "
                            "examples — committing to it. actions: "
                            + ", ".join(summary)
                        )
                lines.append(f"    {cand_name} " + ", ".join(parts))
                if all_pass and found is None:
                    found = FoundOp(
                        op_name=cand_name,
                        rev_ops=rev_ops,
                        rev_res=rev_res,
                        fmt=fmt,
                        op_char=op_char,
                    )
    return found


def _format_r1_table() -> str:
    """Render _GUESS_R1 (example-op signature → rule) as a sorted table."""
    rows = []
    for sig, rule in sorted(_GUESS_R1.items(), key=lambda kv: (len(kv[0]), kv[0])):
        sig_str = ", ".join(sig)
        rows.append(f"        ({sig_str}) → {rule}")
    return "\n".join(rows)


def _format_r4_table() -> str:
    parts = [f"`{c}` → {r}" for c, r in sorted(_GUESS_R4.items())]
    out = []
    line = []
    for p in parts:
        line.append(p)
        if len(line) >= 5:
            out.append("        " + ", ".join(line))
            line = []
    if line:
        out.append("        " + ", ".join(line))
    return "\n".join(out)


_GUESS_RULESET_BLOCK = (
    "When the query operator is not present in the examples, we apply the "
    "following rules in priority order:\n"
    "  R1. Look at the inferred rules for the example operators (sorted as\n"
    "      a tuple) and apply the corresponding rule for the unseen op:\n"
    f"{_format_r1_table()}\n"
    "  R2. If R1 doesn't apply (signature not in the table, or its rule\n"
    "      doesn't fit), use the canonical arithmetic meaning of the\n"
    "      character:\n"
    "        `+` → addition\n"
    "        `-` → subtraction (a-b)\n"
    "        `*` → multiplication\n"
    "  R3. Otherwise, apply hardcoded dataset convention for known chars:\n"
    "        `@` → addition\n"
    "        `#` → addition\n"
    "        `)` → multiply-1\n"
    "  R4. Otherwise, apply per-char majority convention for the remaining\n"
    "      chars seen in the dataset:\n"
    f"{_format_r4_table()}\n"
    "  R5. Default fallback for any other symbol: `concatenation`.\n"
    "\n"
    "Transform: each example operator runs under some (rev_ops, rev_res)\n"
    "transform (identity / reversed operands / reversed result / both). We\n"
    "apply the SAME transform to the unseen operator — when all example ops\n"
    "agree on a transform, we use it; when they don't, we pick the one used\n"
    "by the majority of example ops (tiebreak: identity-first)."
)


def _build_guess_v2(
    problem: Problem,
    preferred_mode: str | None,
    query_op_override: tuple[str, bool, bool] | None,
) -> str | None:
    """V2 narrator for ``equation_numeric_guess``.

    Structure:
      1. Shared header (same as deduce v2 up through the operator listing).
      2. Divergence line: "query op is NOT in the examples".
      3. State the full guessing ruleset up front (so the model sees the
         complete decision tree we're about to apply).
      4. Brief per-example-op summary (just the winning rule, no verbose
         per-candidate search trace — we already showed the ruleset so the
         model can see what was checked conceptually).
      5. Pattern-match the example rules against the ruleset and explain
         which rule applies.
      6. Apply the chosen rule to the query operands; emit boxed answer.

    Returns None if no rule from our pool produces the gold answer (those
    problems are dropped from training).
    """
    q_match = _EXPR_RE.fullmatch(str(problem.question))
    if q_match is None:
        return None
    q_op = q_match.group(2)
    qa, qb = q_match.group(1), q_match.group(3)
    gold = str(problem.answer)

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if m is None:
            continue
        parsed.append((m.group(1), m.group(2), m.group(3), str(ex.output_value)))

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    if q_op in by_op or not by_op:
        return None  # not a guess problem

    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    for op_char, group in by_op.items():
        fmt, tgroup = _detect_fmt(op_char, group)
        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = tgroup

    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append(f"The question is: {problem.question}")
    lines.append(f"The query operator is 【{q_op}】. We need to determine the value of {qa}{q_op}{qb}.")
    lines.append("")
    lines.append("Examples:")
    for a, op, b, out in parsed:
        lines.append(f"  {a}{op}{b} = {out}")
    lines.append("")

    op_list = sorted(by_op.keys())
    lines.append(f"The example operators are: {', '.join(op_list)}.")
    lines.append(
        f"The query operator 【{q_op}】 is NOT in the examples. We need to "
        f"infer its rule from the patterns we see in the example operators."
    )
    lines.append("")
    lines.append(_GUESS_RULESET_BLOCK)

    # Per-example-op: trace search using the SHORTER guess-pool (only ops
    # that appear in our guessing rules — drops digit/det/cross ops that
    # would coincidentally fit examples and confuse the pattern match).
    guess_candidate_sets = [
        ("common", _guess_common_candidates),
        ("rare", _guess_rare_candidates),
    ]
    example_op_results: dict[str, FoundOp] = {}
    for op_char in op_list:
        group = transformed_groups[op_char]
        sig_lines = _sigil_reasoning_block(op_char, detected_fmts[op_char], by_op[op_char], group)
        if sig_lines:
            lines.append("")
            lines.extend(sig_lines)
        examples_str = ", ".join(f"{a}{op_char}{b} = {out}" for a, b, out in group)
        lines.append("")
        lines.append(f"Looking at operator 【{op_char}】 [{examples_str}]:")
        found_op = _emit_op_search(
            lines, op_char, group, detected_fmts[op_char], preferred_mode,
            candidate_sets=guess_candidate_sets,
        )
        if found_op is None:
            return None
        example_op_results[op_char] = found_op

    # Brief recap of the rules we found per example op — useful as a bridge
    # into the pattern-match step (so the reader doesn't have to scroll back
    # through the per-op search to remember what each rule was).
    lines.append("")
    lines.append("Summary of inferred rules for the example operators:")
    for op_char in op_list:
        f = example_op_results[op_char]
        sig_note = ""
        if detected_fmts[op_char] != "num":
            sig_note = f" (sigil format: {detected_fmts[op_char]})"
        lines.append(f"  【{op_char}】: {_format_brief_rule(f)}{sig_note}")

    # Pick the query op rule (must produce gold or we return None).
    picked = _pick_guess_rule(q_op, qa, qb, gold, example_op_results)
    if picked is None:
        return None
    chosen_found_op, reasoning_text = picked

    lines.append("")
    lines.append(f"Applying the guessing rules to 【{q_op}】:")
    lines.append(f"  {reasoning_text}")

    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    result_val, steps = _apply_op_v2(chosen_found_op, qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: {result_val}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)


_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_last_boxed(text: str) -> str | None:
    if not text:
        return None
    starts = list(_BOXED_RE.finditer(text))
    if not starts:
        return None
    seg = text[starts[-1].end():]
    lb = seg.rfind("}")
    return (seg[:lb] if lb != -1 else seg).strip()


def reasoning_equation_numeric(
    problem: Problem,
    preferred_mode: str | None = None,
    query_op_override: tuple[str, bool, bool] | None = None,
) -> str | None:
    """Generate an equation_numeric chain-of-thought rationale.

    Returns None when no path (v2 or legacy) produces a CoT whose final
    boxed answer matches ``problem.answer`` — so callers can use the
    output as training data without needing their own gold check.
    """
    cot = _reasoning_equation_numeric_impl(problem, preferred_mode, query_op_override)
    if cot is None:
        return None
    if _extract_last_boxed(cot) != str(problem.answer):
        return None
    return cot


def _reasoning_equation_numeric_impl(
    problem: Problem,
    preferred_mode: str | None = None,
    query_op_override: tuple[str, bool, bool] | None = None,
) -> str | None:
    """Inner implementation (no gold check). See reasoning_equation_numeric.

    Parameters
    ----------
    preferred_mode :
        Bias the (rev_ops, rev_res) search order to put a preferred
        interpretation first ("standard" / "little_endian" / ...).
        Useful when an external solver has already identified the
        correct transform and we want huikang to commit to it on
        ambiguous example sets.
    query_op_override :
        For ``equation_numeric_guess`` problems where the query operator
        does NOT appear in the examples, huikang's default fallback is
        absolute difference. If the caller knows the actual op (e.g. from
        Alice's gold-conditioned search), pass ``(op_name, rev_ops,
        rev_res)`` here and the "Applying to" section will use that instead.
        The ``op_name`` must be one of the candidate names emitted by
        ``_common_candidates`` / ``_rare_candidates``.
    """
    if problem.category == "equation_numeric_deduce":
        v2 = _build_deduce_v2(problem, preferred_mode, query_op_override)
        if v2 is not None:
            return v2
    elif problem.category == "equation_numeric_guess":
        return _build_guess_v2(problem, preferred_mode, query_op_override)

    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples:")

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))
        lines.append(f"  {ex.input_value} = {ex.output_value}")

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    # Precompute prefix/suffix format and transformed groups per operator
    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    has_symbol_suffix = False
    has_symbol_prefix = False
    symbol_suffix_char = ""
    symbol_prefix_char = ""

    for op_char, group in by_op.items():
        any_neg_suffixed = op_char != "-" and any(
            out.endswith("-") and len(out) > 1 for _, _, out in group
        )
        any_neg_prefixed = op_char != "-" and any(
            out.startswith("-") and len(out) > 1 for _, _, out in group
        )
        any_suffixed = any(
            out.endswith(op_char) and len(out) > 1 for _, _, out in group
        )
        any_prefixed = any(
            out.startswith(op_char) and len(out) > 1 for _, _, out in group
        )

        fmt = "num"
        transformed = list(group)
        if any_neg_suffixed:
            fmt = "neg_suffix"
            transformed = [
                (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
                for a, b, out in group
            ]
        elif any_neg_prefixed:
            fmt = "neg_prefix"
        elif any_suffixed:
            fmt = "neg_suffix"
            has_symbol_suffix = True
            symbol_suffix_char = op_char
            transformed = [
                (
                    a,
                    b,
                    "-" + out[: -len(op_char)]
                    if out.endswith(op_char) and len(out) > 1
                    else out,
                )
                for a, b, out in group
            ]
        elif any_prefixed:
            fmt = "neg_prefix"
            has_symbol_prefix = True
            symbol_prefix_char = op_char
            transformed = [
                (
                    a,
                    b,
                    "-" + out[len(op_char) :]
                    if out.startswith(op_char) and len(out) > 1
                    else out,
                )
                for a, b, out in group
            ]

        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = transformed

    # Build map from (a, op, b) to transformed output
    transformed_map: dict[tuple[str, str, str], str] = {}
    for oc, tgroup in transformed_groups.items():
        for a, b, tout in tgroup:
            transformed_map[(a, oc, b)] = tout

    # Check inputs for leading zeros
    all_inputs: list[str] = []
    for a, _, b, _ in parsed:
        all_inputs.append(a)
        all_inputs.append(b)
    lines.append("")
    lines.append(f"The inputs are {', '.join(all_inputs)}")

    # Report outputs
    all_outputs = [out for _, _, _, out in parsed]
    lines.append("")
    lines.append(f"The outputs are {', '.join(all_outputs)}")
    if has_symbol_suffix:
        lines.append(
            f"Some outputs have the operator symbol as suffix 【{symbol_suffix_char}】."
        )
    if has_symbol_prefix:
        lines.append(
            f"Some outputs have the operator symbol as prefix 【{symbol_prefix_char}】."
        )
    if not has_symbol_suffix and not has_symbol_prefix:
        lines.append("No outputs have a symbol prefix or suffix.")

    # Show transformed outputs if any transformation occurred
    any_transformed = any(fmt != "num" for fmt in detected_fmts.values())
    if any_transformed:
        t_all = [transformed_map.get((a, op, b), out) for a, op, b, out in parsed]
        lines.append(f"We now consider the outputs to be {', '.join(t_all)}")
        if has_symbol_suffix:
            lines.append(
                "We will add back the operator suffix if our answer is negative."
            )
        elif has_symbol_prefix:
            lines.append(
                "We will add back the operator prefix if our answer is negative."
            )

    lines.append("")

    # Show input → operator parsing
    lines.append("Looking at the input of the examples")
    for a, op, b, out in parsed:
        lines.append(f"{a}{op}{b} -> {op}")
    op_names = list(by_op.keys())
    lines.append("")
    lines.append("The operators")
    for op in op_names:
        lines.append(op)

    q_match = _EXPR_RE.fullmatch(str(problem.question))
    q_op = q_match.group(2) if q_match else None

    lines.append("")
    lines.append("Looking at the question")
    if q_match:
        lines.append(f"{problem.question} -> {q_op}")

    # If question operator not in examples, fall back to most common example operator
    effective_q_op = q_op
    if q_op is not None and q_op not in by_op and by_op:
        most_common_op = max(by_op, key=lambda op: len(by_op[op]))
        lines.append(
            f"The question operator is not found in the examples. "
            f"Investigating the most common example operator 【{most_common_op}】 instead. "
            f"We will use absolute difference for the question operator."
        )
        effective_q_op = most_common_op
    elif q_op is not None and q_op in by_op:
        lines.append("The question operator is found in the examples.")

    found_ops: dict[str, FoundOp] = {}

    # Analyze each operator (focus on question operator)
    for op_char, group in sorted(by_op.items()):
        if effective_q_op is not None and op_char != effective_q_op and len(by_op) > 1:
            continue

        # Use precomputed format and transformed group
        detected_fmt = detected_fmts[op_char]
        group = transformed_groups[op_char]

        examples_str = ", ".join(f"{a}{op_char}{b} = {out}" for a, b, out in group)
        lines.append("")
        lines.append(f"Looking at operator 【{op_char}】 [{examples_str}]:")

        a_str, b_str, expected = group[0]

        # Try common operations first (all 4 combos), then rare operations
        found = None

        candidate_sets = [
            ("common", _common_candidates),
            ("rare", _rare_candidates),
        ]

        n_ex = len(group)
        transform_order = _resolve_transform_order(preferred_mode)
        for set_name, cand_fn in candidate_sets:
            for rev_ops, rev_res in transform_order:
                # Use fixed example order for paragraph header
                cycled = list(group)

                # Describe what we're trying
                label = f"{set_name} operations"
                if rev_ops:
                    rev_parts = ", ".join(
                        f"{ax}->{ax[::-1]} {bx}->{bx[::-1]}" for ax, bx, _ in cycled
                    )
                    if rev_res:
                        label += f" reversed operands [{rev_parts}] and reversed result"
                    else:
                        label += f" reversed operands [{rev_parts}]"
                elif rev_res:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" identity operands [{id_parts}] reversed result"
                else:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" on identity [{id_parts}]"
                if rev_ops:
                    all_expected = ", ".join(
                        f"({ax[::-1]},{bx[::-1]})->{exp}" for ax, bx, exp in cycled
                    )
                else:
                    all_expected = ", ".join(
                        f"({ax},{bx})->{exp}" for ax, bx, exp in cycled
                    )
                lines.append(f"  Trying {label} [expected {all_expected}]:")

                def _fmt_result(
                    raw: str, a: str, b: str, detail: str, arrow: bool
                ) -> str:
                    fin = _rev(raw) if rev_res else raw
                    val = f"{raw} -rev-> {fin}" if rev_res else fin
                    if arrow:
                        return f"f({a},{b}) ->{detail} {val}"
                    return f"f({a}, {b}) ={detail} {val}"

                # Use first example for candidate generation
                ca_str, cb_str = cycled[0][0], cycled[0][1]
                cta = ca_str[::-1] if rev_ops else ca_str
                ctb = cb_str[::-1] if rev_ops else cb_str
                candidates = cand_fn(int(cta), int(ctb), cta, ctb)
                cand_idx = 0
                for cand_name, cand_res in candidates:
                    # Rotate which example is tried first within the paragraph
                    rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                    cand_idx += 1

                    parts = []
                    all_pass = True

                    for i, (ax, bx, exp_x) in enumerate(rotated):
                        rax = ax[::-1] if rev_ops else ax
                        rbx = bx[::-1] if rev_ops else bx
                        raw = next(
                            r
                            for n, r in _all_candidates(int(rax), int(rbx), rax, rbx)
                            if n == cand_name
                        )
                        expr_x = _expr(cand_name, rax, rbx)
                        inter_x = _expr_intermediate(cand_name, rax, rbx)
                        if expr_x and inter_x:
                            detail_x = f" {expr_x} = {inter_x} ="
                        elif expr_x:
                            detail_x = f" {expr_x} ="
                        else:
                            detail_x = ""
                        fin = _rev(raw) if rev_res else raw
                        status = "match" if fin == exp_x else "wrong"
                        if fin != exp_x:
                            all_pass = False
                        parts.append(
                            _fmt_result(raw, rax, rbx, detail_x, arrow=i > 0)
                            + f" {status}"
                        )
                        if fin != exp_x:
                            break

                    if all_pass:
                        if found:
                            parts.append("correct, but skipping")
                        else:
                            summary = []
                            if rev_ops:
                                summary.append("reversed operands")
                            if rev_res:
                                summary.append("reversed result")
                            summary.append(cand_name)
                            parts.append("correct, actions: " + ", ".join(summary))
                    lines.append(f"    {cand_name} " + ", ".join(parts))

                    if not all_pass:
                        continue

                    if not found:
                        found = FoundOp(
                            op_name=cand_name,
                            rev_ops=rev_ops,
                            rev_res=rev_res,
                            fmt=detected_fmt,
                            op_char=op_char,
                        )

        if found:
            found_ops[op_char] = found
        else:
            if op_char == effective_q_op:
                return None
            lines.append("  No matching operation found.")

    # Apply to question
    if not q_match or effective_q_op not in found_ops:
        return None

    qa, qb = q_match.group(1), q_match.group(3)
    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    if effective_q_op != q_op:
        if query_op_override is not None:
            ov_op_name, ov_rev_ops, ov_rev_res = query_op_override
            lines.append(
                "  We recall that the question operator is not found in the examples. "
                f"We will use {ov_op_name} as the operator for the question."
            )
            override_op = FoundOp(
                op_name=ov_op_name,
                rev_ops=ov_rev_ops,
                rev_res=ov_rev_res,
                fmt=found_ops[effective_q_op].fmt,
                op_char=q_op or "",
            )
            result_val, steps = _apply_op(override_op, qa, qb)
        else:
            lines.append(
                "  We recall that the question operator is not found in the examples. "
                "We will use the absolute difference as the operator."
            )
            abs_diff_op = FoundOp(
                op_name="absolute difference",
                rev_ops=False,
                rev_res=False,
                fmt=found_ops[effective_q_op].fmt,
                op_char=q_op or "",
            )
            result_val, steps = _apply_op(abs_diff_op, qa, qb)
    else:
        result_val, steps = _apply_op(found_ops[effective_q_op], qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: 【{result_val}】")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")

    # Gold filter: legacy path doesn't constrain on gold; only emit the
    # CoT when the boxed answer matches gold. Otherwise return None so
    # the caller can drop the problem from training.
    if str(result_val) != str(problem.answer):
        return None
    return "\n".join(lines)
