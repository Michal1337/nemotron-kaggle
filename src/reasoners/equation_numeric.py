"""Equation numeric reasoning generator."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

from reasoners.store_types import Problem

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first.

    Order: concat, rconcat, add, sub, absdiff, negabsdiff, rsub, mul.
    Asymmetric placement of sub vs rsub: `sub` is high-priority (catches
    signed-result cases when query has a<b and examples have a>=b), but
    `rsub` is demoted past absdiff/negabsdiff (catches the inverse: when
    examples have a<b and rsub fits, the unsigned absdiff usually matches
    the gold's intent better). Empirically +8 deduce on 596 vs symmetric
    sub-then-rsub placement (550 → 558).
    """
    out: list[tuple[str, str]] = []
    out.append(("concat", sa + sb))
    out.append(("rconcat", sb + sa))
    out.append(("add", str(a + b)))
    out.append(("sub", str(a - b)))
    out.append(("absdiff", str(abs(a - b))))
    out.append(("negabsdiff", str(-abs(a - b))))
    out.append(("rsub", str(b - a)))
    out.append(("mul", str(a * b)))
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Trimmed rare pool — keeps only the 5 ops that actually fire as first-fit
    in our 596-pid deduce universe (verified by op-usage audit). The 23 ops
    we dropped (sub±1, gcd, lcm, absdiff±2, div, mod, rdiv, rmod, all digit_*,
    cross_*, det_*) were never picked in any solved pid in either deduce or
    guess, so removing them costs zero coverage and shaves ~92 trial lines
    (23 ops × 4 transforms) per deduce CoT."""
    out: list[tuple[str, str]] = []
    out.append(("mul+1", str(a * b + 1)))
    out.append(("mul-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare."""
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _expr(name: str, a: str, b: str) -> str:
    """Return the math expression for an operation, e.g. '94 + 48'."""
    if name == "add":
        return f"{a} + {b}"
    if name == "sub":
        return f"{a} - {b}"
    if name == "rsub":
        return f"{b} - {a}"
    if name == "mul":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b}"
        return f"{a} * {b}"
    if name == "absdiff":
        return f"|{a} - {b}|"
    if name == "negabsdiff":
        return f"-|{a} - {b}|"
    if name == "concat":
        return f"{a} || {b}"
    if name == "rconcat":
        return f"{b} || {a}"
    if name == "mul+1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} + 1"
        return f"{a} * {b} + 1"
    if name == "mul-1":
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
    if name == "div":
        return f"{a} / {b}"
    if name == "mod":
        return f"{a} mod {b}"
    if name == "rdiv":
        return f"{b} / {a}"
    if name == "rmod":
        return f"{b} mod {a}"
    if name == "max mod min":
        big, small = (a, b) if int(a) >= int(b) else (b, a)
        return f"max({a},{b}) mod min({a},{b}) = {big} mod {small}"
    if name == "gcd":
        return f"gcd({a}, {b})"
    if name == "lcm":
        return f"lcm({a}, {b})"
    if name == "absdiff-2":
        return f"|{a} - {b}| - 2"
    if name == "absdiff+2":
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
    if name in ("mul+1", "mul-1", "mul") and len(a) >= 2:
        # Decompose a by place value: 70 → [70, 0], 73 → [70, 3]
        places = [int(d) * (10 ** (len(a) - 1 - i)) for i, d in enumerate(a)]
        decomp = " + ".join(f"{p} * {b}" for p in places)
        evald = " + ".join(str(p * ib) for p in places)
        product_sum = sum(p * ib for p in places)
        if name == "mul+1":
            return f"{decomp} + 1 = {evald} + 1 = {product_sum} + 1"
        if name == "mul-1":
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


# legacy _apply_op deleted (audit 2026-06-10): no callers — the live deduce/
# guess paths use _apply_op_v2.


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


def _sigil_check_summary(
    op_chars: list[str],
    detected_fmts: dict[str, str],
    by_op: dict[str, list[tuple[str, str, str]]],
    transformed_groups: dict[str, list[tuple[str, str, str]]],
) -> list[str]:
    """Always-on sigil check listing every example for every operator.

    Emitted directly under "The example operators are: ..." so the model learns
    a consistent "check then proceed" routine instead of only seeing sigil
    reasoning on the ~30% of guess problems where a sigil actually fires.
    Each operator block enumerates every example explicitly — for sigil ops
    the decoded form is shown side-by-side; for plain ops the examples are
    listed verbatim as a no-op confirmation.
    """
    out = ["Sign-marker check:"]
    for op in op_chars:
        fmt = detected_fmts.get(op, "num")
        orig = by_op.get(op, [])
        tgrp = transformed_groups.get(op, orig)
        if fmt == "neg_prefix":
            if op == "-":
                # natural minus: outputs starting with '-' are ordinary negative
                # numbers, not a sigil encoding (audit 2026-06-10: the generic
                # wording here contradicted its own per-row "no sigil" notes)
                out.append(
                    f"  【{op}】: the operator is the natural minus sign — negative "
                    f"outputs simply start with `-`, nothing to decode.")
                for (a, b, raw) in orig:
                    out.append(f"    {a}{op}{b} = {raw}")
            else:
                out.append(f"  【{op}】: sign sigil at output PREFIX — decode `{op}X` → `-X`.")
                for (a, b, raw), (_, _, dec) in zip(orig, tgrp):
                    if raw == dec:
                        out.append(f"    {a}{op}{b} = {raw}  (no sigil, kept as-is)")
                    else:
                        out.append(f"    {a}{op}{b} = {raw}  →  {a}{op}{b} = {dec}")
        elif fmt == "neg_suffix":
            out.append(f"  【{op}】: sign sigil at output SUFFIX — decode `X{op}` → `-X`.")
            for (a, b, raw), (_, _, dec) in zip(orig, tgrp):
                if raw == dec:
                    out.append(f"    {a}{op}{b} = {raw}  (no sigil, kept as-is)")
                else:
                    out.append(f"    {a}{op}{b} = {raw}  →  {a}{op}{b} = {dec}")
        else:
            out.append(f"  【{op}】: no sign sigil — outputs are plain digits.")
            for (a, b, raw) in orig:
                out.append(f"    {a}{op}{b} = {raw}")
    return out


def _sigil_reasoning_block(
    op_char: str,
    fmt: str,
    group: list[tuple[str, str, str]],
    transformed_group: list[tuple[str, str, str]],
) -> list[str]:
    """Deprecated — sigil reasoning is now emitted once at the top of the CoT
    by `_sigil_check_summary`. Kept as a no-op for backward compatibility with
    existing call sites.
    """
    return []


# Only id (no reversal) and rev_both (little-endian) are ever the answer in the
# Alice equation_numeric generator — the mixed rev_ops-only / rev_res-only modes
# never win and only shadow correct combos. Dropping them: deduce 557->558,
# guess 83->83 (no regression), ~30% shorter CoTs. (audit: _shuffle_order.py)
_DEFAULT_TRANSFORM_ORDER = ((True, True), (False, False))


# _resolve_transform_order and _common_candidates_v2 deleted (audit 2026-06-10):
# both dead in production. _common_candidates_v2's order (signed ops before
# absdiff) was measured net -7 on the 596 real deduce rows — do not re-add.


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
    signed_raw = final  # pre-sigil value (leading '-' iff the true result is negative)

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

    # Apply the sign sigil to the boxed value. The sign is EXPLAINED uniformly by
    # the consolidated _sign_resolution_line emitted just before the box (positive
    # and negative alike); here we only do the mechanical rewrite.
    #   - neg_suffix: a negative result carries the operator glyph as a suffix.
    #   - neg_prefix / num: a non-'-' operator glyph stands in for the minus sign
    #     as a PREFIX; a '-' operator is the natural minus and is left as '-N'.
    if found.fmt == "neg_suffix":
        if final.startswith("-"):
            final = final[1:] + found.op_char
    elif found.op_char != "-" and final.startswith("-"):
        final = found.op_char + final[1:]

    return final, steps, signed_raw


def _sign_resolution_line(found: "FoundOp", signed_raw: str, final: str) -> str:
    """The single, ALWAYS-emitted final result statement. One uniform shape, but
    the negative branch is HONEST about how the prefix/suffix position is known:
      - positive       -> "result is positive, RES -> RES"  (no sigil)
      - natural '-'    -> the result keeps its own leading '-'.
      - neg, OBSERVED  -> position came from a negative example in the sign-marker
                          check (fmt neg_prefix / neg_suffix): "the examples write
                          【q】's negatives as a PREFIX/SUFFIX, so ...".
      - neg, DEFAULT   -> no negative example for this op (fmt num, incl. every
                          guess query op): say so, and that PREFIX is the dataset
                          convention default — do NOT pretend it was observed.
    Datamine basis (eq_sign_convention memory): the sigil is ALWAYS the operator's
    own glyph; position is per-problem and only knowable from a negative example.
    """
    q = found.op_char
    if not signed_raw.startswith("-"):
        return f"Query op 【{q}】: result is positive, {signed_raw} -> {final}"
    if q == "-":
        return (f"Query op 【-】: result is negative; 【-】 is the natural minus "
                f"sign, {signed_raw} -> {final}")
    if found.fmt == "neg_suffix":
        return (f"Query op 【{q}】: result is negative; the examples write 【{q}】's "
                f"negatives with the operator glyph as a SUFFIX (X【{q}】), so "
                f"{signed_raw} -> {final}")
    if found.fmt == "neg_prefix":
        return (f"Query op 【{q}】: result is negative; the examples write 【{q}】's "
                f"negatives with the operator glyph as a PREFIX (【{q}】X), so "
                f"{signed_raw} -> {final}")
    # fmt == "num": no negative example for 【q】 was observed (always the case for
    # a guess query op) — prefix is the convention default, stated as such.
    return (f"Query op 【{q}】: result is negative; no negative example fixes 【{q}】's "
            f"position, so by convention a negative is written with the operator "
            f"glyph as a PREFIX (【{q}】X), {signed_raw} -> {final}")


def _mode_tag(rev_ops: bool, rev_res: bool) -> str:
    if rev_ops and rev_res:
        return "rev_both"
    if rev_ops:
        return "rev_ops"
    if rev_res:
        return "rev_res"
    return "id"


def _emit_operator_search(
    lines: list[str],
    op_char: str,
    group: list[tuple[str, str, str]],
    fmt: str,
    rev_ops: bool,
    rev_res: bool,
) -> "FoundOp | None":
    """Emit the full-derivation candidate search for ONE operator under ONE
    orientation; return the first matching rule (or None if no candidate fits).

    Lines keep the legacy per-candidate format with the complete arithmetic for
    every number (``sub f(45, 21) = 45 - 21 = 24 M``); the search stops at the
    first match. ``rconcat`` is excluded from the pool — it is redundant with
    ``concat`` under reversal and only created spurious orientation ties.
    """
    n_ex = len(group)
    cycled = list(group)
    if rev_ops:
        segs = []
        for a, b, o in cycled:
            seg = f"{a}{op_char}{b}={o} (reverse operands -> {a[::-1]},{b[::-1]}"
            if rev_res:
                seg += f", reversed target -> {_rev(o)}"
            seg += ")"
            segs.append(seg)
        ex_disp = ", ".join(segs)
    else:
        ex_disp = ", ".join(f"{a}{op_char}{b}={o}" for a, b, o in cycled)
    lines.append(f"  Operator 【{op_char}】 [{ex_disp}]:")
    found: "FoundOp | None" = None
    for cand_fn in (_common_candidates, _rare_candidates):
        ca_str, cb_str = cycled[0][0], cycled[0][1]
        cta = ca_str[::-1] if rev_ops else ca_str
        ctb = cb_str[::-1] if rev_ops else cb_str
        # rconcat WAS excluded (redundant with concat under reversal), but that
        # dropped every deduce problem containing an rconcat operator (the
        # orientation got rejected because that operator couldn't resolve).
        # Keep it: ties are already broken first-fit below (concat is tried
        # first in _common_candidates, so genuine concat problems are unaffected).
        candidates = [(n, r) for n, r in cand_fn(int(cta), int(ctb), cta, ctb)]
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
                status = "M" if fin == exp_x else "W"
                if fin != exp_x:
                    all_pass = False
                if rev_res and status == "M":
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
                if found is None:
                    parts.append(f"MATCH -> 【{op_char}】 = {cand_name}")
                    found = FoundOp(op_name=cand_name, rev_ops=rev_ops,
                                    rev_res=rev_res, fmt=fmt, op_char=op_char)
                else:
                    parts.append(
                        f"also matches, but 【{op_char}】 = {found.op_name} "
                        f"already chosen (first fit)")
            lines.append(f"    {cand_name} " + ", ".join(parts))
    return found


def _build_deduce_v2(problem: Problem) -> str | None:
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
    lines.append(
        f"The query operator 【{q_op}】 is one of them — it appears in the "
        f"examples, so we can infer its rule directly from those examples.")
    lines.append("")
    lines.extend(_sigil_check_summary(op_list, detected_fmts, by_op, transformed_groups))
    lines.append("")
    # ---- Step 1: determine the orientation from ALL operators ----
    # The whole problem uses one orientation; the correct one is the only one
    # under which EVERY example operator has a consistent rule. We always check
    # both orientations and narrate the full search, so the choice is derived,
    # not asserted.
    lines.append(
        "Step 1 - find the orientation. The whole problem uses one orientation "
        "(identity, or reversed operands + reversed result), so the correct one "
        "must give every operator a consistent rule. We check both.")

    # rev_both is the more common orientation (370:177 in the solved set), so
    # when both orientations explain the examples we prefer it as the tiebreak.
    # Fixed order — audit 2026-06-10: the preferred_mode flip here was a leak
    # door (callers could pass a gold-conditioned mode and the CoT would narrate
    # the flipped pick as "convention"). Orientation priority is a global
    # convention, never caller-biased.
    orient_specs = [
        ((True, True), "reversed operands and reversed result"),
        ((False, False), "identity (operands and result unchanged)"),
    ]

    consistent: list[tuple[tuple[bool, bool], dict[str, FoundOp]]] = []
    for (rev_ops, rev_res), desc in orient_specs:
        lines.append("")
        lines.append(f"Orientation [{_mode_tag(rev_ops, rev_res)}] - {desc}:")
        op_rules: dict[str, FoundOp] = {}
        for op_char in op_list:
            fo = _emit_operator_search(
                lines, op_char, transformed_groups[op_char],
                detected_fmts[op_char], rev_ops, rev_res)
            if fo is None:
                lines.append(
                    f"  -> 【{op_char}】 has no consistent rule under "
                    f"[{_mode_tag(rev_ops, rev_res)}].")
            else:
                op_rules[op_char] = fo
        if len(op_rules) == len(op_list):
            lines.append(
                f"  -> every operator has a rule under "
                f"[{_mode_tag(rev_ops, rev_res)}]; this orientation is consistent.")
            consistent.append(((rev_ops, rev_res), op_rules))
        else:
            lines.append(
                f"  -> not every operator has a rule under "
                f"[{_mode_tag(rev_ops, rev_res)}]; this orientation is rejected.")

    if not consistent:
        return None

    chosen_mode, chosen_rules = consistent[0]
    lines.append("")
    if len(consistent) > 1:
        # Examples-only check (no gold): do the consistent orientations agree on
        # the query answer? If yes the choice is harmless; if not, we fall back
        # to the more common orientation (rev_both, tried first) as a stated
        # convention — we do NOT use the gold answer to choose.
        q_answers = {_apply_op_v2(r[q_op], qa, qb)[0] for _, r in consistent}
        if len(q_answers) == 1:
            lines.append(
                "More than one orientation explains every operator, and they "
                f"yield the same answer, so we use [{_mode_tag(*chosen_mode)}].")
        else:
            lines.append(
                "More than one orientation explains every operator and the "
                "examples do not distinguish them; by convention we use the more "
                f"common orientation, [{_mode_tag(*chosen_mode)}].")
    else:
        lines.append(
            f"Only [{_mode_tag(*chosen_mode)}] explains every operator, so the "
            f"orientation is [{_mode_tag(*chosen_mode)}].")

    found = chosen_rules[q_op]

    # ---- Step 2: apply the query operator under the chosen orientation ----
    lines.append("")
    lines.append(
        f"Step 2 - apply the query operator 【{q_op}】 = {found.op_name} under "
        f"[{_mode_tag(*chosen_mode)}] (found in Step 1).")
    lines.append(f"Applying to {problem.question}:")
    result_val, steps, signed_raw = _apply_op_v2(found, qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  {_sign_resolution_line(found, signed_raw, result_val)}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)


# ---------------- v2 guess builder ----------------

# R1: (example-op signature, picked transform) → query rule. Mined from
# the dataset by majority-vote across all guess problems sharing the same
# (sorted rule tuple, picked transform) cell. Picked transform is computed
# the same way the cascade does — majority across example ops, tiebreak
# by _DEFAULT_TRANSFORM_ORDER. This split table lifts honest catches from
# 63 to 82 on the 136 real eq_guess pids vs sig-only R1 (+19 gain).
# Pids whose chosen rule doesn't produce gold under R1 fall through to
# R2 (per-char), then R3 (default 'concat').
_GUESS_R1 = {
    (('absdiff',), 'rev_both'): 'add',
    (('absdiff',), 'id'): 'add',
    (('absdiff', 'add'), 'rev_both'): 'mul-1',
    (('absdiff', 'add'), 'id'): 'mul-1',
    (('absdiff', 'add-1'), 'rev_both'): 'mul',
    (('absdiff', 'add-1'), 'id'): 'mul',
    (('absdiff', 'concat'), 'rev_both'): 'mul',
    (('absdiff', 'mul'), 'rev_both'): 'add+1',
    (('absdiff', 'mul'), 'id'): 'add-1',
    (('absdiff', 'mul-1'), 'rev_both'): 'add',
    (('absdiff', 'rconcat'), 'rev_both'): 'add',
    (('add',), 'rev_both'): 'mul-1',
    (('add', 'concat'), 'rev_both'): 'max mod min',
    (('add', 'max mod min'), 'rev_both'): 'mul+1',
    (('add', 'mul'), 'rev_both'): 'absdiff',
    (('add', 'mul+1'), 'id'): 'absdiff',
    (('add', 'mul-1'), 'rev_both'): 'negabsdiff',  # re-mined 2026-06-03: 2/2 -> negabsdiff
    (('add', 'mul-1'), 'id'): 'absdiff',
    (('add', 'negabsdiff'), 'rev_both'): 'mul',
    (('add', 'negabsdiff'), 'id'): 'concat',
    (('add', 'sub'), 'rev_both'): 'mul-1',
    (('add+1',), 'id'): 'absdiff',
    (('add+1', 'max mod min'), 'id'): 'mul',
    (('add+1', 'mul'), 'id'): 'max mod min',
    (('add+1', 'negabsdiff'), 'rev_both'): 'mul+1',
    (('add+1', 'negabsdiff'), 'id'): 'mul',
    (('add+1', 'rsub'), 'rev_both'): 'mul',
    (('add+1', 'sub'), 'rev_both'): 'concat',
    (('add-1',), 'rev_both'): 'absdiff',
    (('add-1',), 'id'): 'mul',
    (('add-1', 'mul'), 'rev_both'): 'negabsdiff',  # re-mined 2026-06-03: 2/2 -> negabsdiff
    (('add-1', 'max mod min'), 'id'): 'mul',
    (('add-1', 'mul-1'), 'id'): 'absdiff',
    (('add-1', 'sub'), 'rev_both'): 'mul-1',
    (('concat',), 'rev_both'): 'add',
    (('concat', 'mul'), 'rev_both'): 'absdiff',
    (('concat', 'mul+1'), 'rev_both'): 'sub',  # re-mined 2026-06-03: 3/3 -> sub
    (('concat', 'negabsdiff'), 'rev_both'): 'concat',
    (('concat', 'sub'), 'rev_both'): 'concat',
    # (('gcd', 'mul'), 'id'): 'add+1',  # dropped: narrator's guess pool excludes gcd, only 1 real pid
    (('max mod min',), 'rev_both'): 'mul',
    (('max mod min',), 'id'): 'concat',
    (('max mod min', 'mul'), 'rev_both'): 'add',
    (('max mod min', 'mul'), 'id'): 'concat',
    (('max mod min', 'mul+1'), 'rev_both'): 'add',
    (('mul',), 'rev_both'): 'max mod min',
    (('mul', 'negabsdiff'), 'rev_both'): 'add',
    (('mul', 'negabsdiff'), 'id'): 'concat',
    (('mul', 'rconcat'), 'rev_both'): 'absdiff',
    (('mul+1',), 'id'): 'absdiff',
    (('mul+1', 'negabsdiff'), 'rev_both'): 'add',
    (('mul+1', 'negabsdiff'), 'id'): 'add',
    (('mul+1', 'sub'), 'rev_both'): 'add+1',
    (('mul-1',), 'rev_both'): 'add+1',
    (('mul-1', 'negabsdiff'), 'rev_both'): 'add',
    (('mul-1', 'sub'), 'id'): 'concat',
    (('negabsdiff',), 'rev_both'): 'mul',
    (('negabsdiff',), 'id'): 'mul',
    (('negabsdiff', 'rconcat'), 'rev_both'): 'add',
    (('rconcat',), 'rev_both'): 'absdiff',
    (('sub',), 'rev_both'): 'add+1',
    (('sub',), 'id'): 'concat',
}

# R2: data-mined per-char rule for q_ops where R1 either has no signature
# or R1's rule didn't fit gold. Mined from R1-miss residual on real data.
# This is the same "+/-/*" channel as before but with data-derived rules
# (not canonical). Catches ~11 of the 53 R1-miss real pids.
_GUESS_R2 = {
    '+': 'add-1',
    '-': 'negabsdiff',
    '*': 'concat',
}

# R3: per-char fallback after R2. Mined from R1+R2-miss residual.
_GUESS_R3 = {
    '#': 'add',
    ')': 'add',
}

# R4: per-char majority for residual after R1+R2+R3. Mined from real R1+R2+R3-miss.
# Only chars with at least one real firing are kept.
_GUESS_R4 = {
    '!': 'digit add mod10',
    '%': 'mul',
    '&': 'add-1',
    "'": 'add',
    '(': 'mul',
    '/': 'add',
    ':': 'add',
    '<': 'mul',
    '[': 'mul',
    '\\': 'mul+1',
    '^': 'mul-1',
    '|': 'add-1',
    '}': 'concat',
}

# R5: default fallback when nothing matches.
_GUESS_R5_DEFAULT = 'concat'

_GUESS_NATURAL = {k: [v] for k, v in _GUESS_R2.items()}
_GUESS_PER_CHAR = {k: [v] for k, v in _GUESS_R3.items()}
_GUESS_PER_CHAR_EXT = _GUESS_R4


# _candidate_rule_names_for / _apply_rule_with_transform / _pick_guess_rule
# deleted (audit 2026-06-10): gold-conditioned rule selection (pred==gold
# filter mid-pick) — the prohibited selection leak. Dead in production
# (test-only callers). Do not re-add.


def _guess_common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Reduced common-op pool for the GUESS narrator's per-example-op search.
    Only includes ops that appear in the guessing ruleset (R1-R3 families
    + concatenation). Digit-level, determinant, cross-multiply etc. are
    intentionally omitted — they're not in our guess decision tree, and
    they tend to coincidentally fit examples in ways that confuse the
    pattern-match step."""
    out: list[tuple[str, str]] = []
    out.append(("concat", sa + sb))
    out.append(("rconcat", sb + sa))
    out.append(("add", str(a + b)))
    out.append(("absdiff", str(abs(a - b))))
    out.append(("negabsdiff", str(-abs(a - b))))
    out.append(("sub", str(a - b)))
    out.append(("rsub", str(b - a)))
    out.append(("mul", str(a * b)))
    return out


def _guess_rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Reduced rare-op pool for guess. Audit of the real data showed sub+1/
    sub-1/mod/rmod are 0-1 uses each — dropped to shrink search trace.
    Keeps mul±1/add±1 (offset variants used ~100 times each) and
    `max mod min` (50 uses)."""
    out: list[tuple[str, str]] = []
    out.append(("mul+1", str(a * b + 1)))
    out.append(("mul-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    return out


def _format_brief_rule(found: FoundOp) -> str:
    """Compact human-readable description of a FoundOp: rule + transform."""
    if found.rev_ops and found.rev_res:
        t = "rev_both"
    elif found.rev_ops:
        t = "rev_ops"
    elif found.rev_res:
        t = "rev_res"
    else:
        t = "id"
    return f"{found.op_name} [{t}]"


# _emit_op_search deleted (audit 2026-06-10): dead in production (the live
# guess path uses _emit_operator_search + the family table). It was the only
# reader of _resolve_transform_order. Do not re-add.


def _format_r1_table() -> str:
    """Render _GUESS_R1 ((sig, xf) → rule) as a sorted table, grouping by
    signature with [xf]=rule pairs inline for compactness."""
    by_sig: dict[tuple, dict[str, str]] = {}
    for (sig, xf), rule in _GUESS_R1.items():
        by_sig.setdefault(sig, {})[xf] = rule
    rows = []
    for sig in sorted(by_sig.keys(), key=lambda s: (len(s), s)):
        sig_str = ", ".join(sig)
        xf_map = by_sig[sig]
        xf_parts = ", ".join(f"[{xf}]={r}" for xf, r in sorted(xf_map.items()))
        rows.append(f"        ({sig_str}) {xf_parts}")
    return "\n".join(rows)


def _format_r1_table_no_xf() -> str:
    """OLD-style render: `(sig) → rule` per signature. Used when every xf-tag
    for a sig maps to the same rule (OLD rulesets had no transform split)."""
    by_sig: dict[tuple, set] = {}
    for (sig, xf), rule in _GUESS_R1.items():
        by_sig.setdefault(sig, set()).add(rule)
    rows = []
    for sig in sorted(by_sig.keys(), key=lambda s: (len(s), s)):
        sig_str = ", ".join(sig)
        rules = by_sig[sig]
        if len(rules) == 1:
            rows.append(f"        ({sig_str}) → {next(iter(rules))}")
        else:
            # Fallback to xf-breakdown if rules differ per xf
            pairs = {}
            for (s, xf), r in _GUESS_R1.items():
                if s == sig:
                    pairs[xf] = r
            xf_parts = ", ".join(f"[{xf}]={r}" for xf, r in sorted(pairs.items()))
            rows.append(f"        ({sig_str}) {xf_parts}")
    return "\n".join(rows)


def _format_char_table(d, cols=5) -> str:
    """Render a {char: rule} mapping as comma-separated `c` → r columns."""
    parts = [f"`{c}` → {r if not isinstance(r, list) else r[0]}"
             for c, r in sorted(d.items())]
    out = []
    line = []
    for p in parts:
        line.append(p)
        if len(line) >= cols:
            out.append("        " + ", ".join(line))
            line = []
    if line:
        out.append("        " + ", ".join(line))
    return "\n".join(out)


def _format_r4_table() -> str:
    return _format_char_table(_GUESS_R4)


# Preamble display style. "compact" (NEW default): R1 table + R2 +/-/* + R3
# default = concat (R3/R4 dicts hidden from the preamble; still applied
# internally). "full" (OLD style): explicit R1-R5 with R3/R4 dicts displayed.
_PREAMBLE_VARIANT = "compact"


def _build_ruleset_block_compact() -> str:
    """NEW-style 3-tier preamble (R3/R4 dicts hidden)."""
    parts = [
        "When the query op is not in the examples, apply in priority:",
        "  R1. Match (example-ops signature, picked transform) to this table:",
        _format_r1_table(),
        "  R2. If (signature, transform) not in R1, per-char fallback for `+`/`-`/`*`:",
        _format_char_table(_GUESS_R2),
        "  R3. Default fallback for any other char: `concat`.",
        "",
        "Transform: each example op uses one of [id, rev_ops, rev_res, rev_both]. "
        "Picked transform for unseen op = majority across example ops; tiebreak id-first.",
    ]
    return "\n".join(parts)


def _build_ruleset_block_full() -> str:
    """OLD-style 5-tier preamble (R3 hardcoded chars + R4 per-char majority + R5 default)."""
    parts = [
        "When the query op is not in the examples, apply in priority:",
        "  R1. Look at the inferred rules for the example operators (sorted as a tuple) "
        "and apply the corresponding rule for the unseen op:",
        _format_r1_table_no_xf(),
        "  R2. If R1 doesn't apply, per-char fallback for `+`/`-`/`*`:",
        _format_char_table(_GUESS_R2),
        "  R3. Otherwise, apply hardcoded dataset convention for known chars:",
        _format_char_table(_GUESS_R3),
        "  R4. Otherwise, apply per-char majority convention for the remaining chars:",
        _format_r4_table(),
        "  R5. Default fallback for any other symbol: `concat`.",
        "",
        "Transform: each example operator runs under some (rev_ops, rev_res) "
        "transform (identity / reversed operands / reversed result / both). We "
        "apply the SAME transform to the unseen operator — when all example ops "
        "agree on a transform, we use it; when they don't, we pick the one used "
        "by the majority of example ops (tiebreak: identity-first).",
    ]
    return "\n".join(parts)


def _build_ruleset_block() -> str:
    if _PREAMBLE_VARIANT == "full":
        return _build_ruleset_block_full()
    return _build_ruleset_block_compact()


_GUESS_RULESET_BLOCK = _build_ruleset_block()


def _build_ruleset_block_exact_2mode() -> str:
    """Compact 3-tier exact-op preamble restricted to id/rev_both modes
    (EQ_GUESS_TABLE=exact). R1 exact-op signature table -> R2 +/-/* -> R3 concat."""
    by_sig: dict[tuple, dict] = {}
    for (sig, xf), rule in _GUESS_R1.items():
        if xf in ("id", "rev_both"):
            by_sig.setdefault(sig, {})[xf] = rule
    rows = []
    for sig in sorted(by_sig.keys(), key=lambda s: (len(s), s)):
        xf_parts = ", ".join(f"[{xf}]={r}" for xf, r in sorted(by_sig[sig].items()))
        rows.append(f"        ({', '.join(sig)}) {xf_parts}")
    parts = [
        "When the query op is not in the examples, apply in priority:",
        "  R1. Match (example-ops signature, picked transform) to this table:",
        "\n".join(rows),
        "  R2. If (signature, transform) not in R1, per-char fallback for `+`/`-`/`*`:",
        _format_char_table(_GUESS_R2),
        "  R3. Default fallback for any other char: `concat`.",
        "",
        "Transform: each example op uses one of [id, rev_both]. Picked transform "
        "for the unseen op = majority across example ops; tiebreak id-first.",
    ]
    return "\n".join(parts)


_GUESS_RULESET_BLOCK_EXACT = _build_ruleset_block_exact_2mode()


# ---------------- guess v2: family-signature cascade ----------------
# The unseen query op is guessed from (orientation, FAMILIES of the example-op
# rules). Families collapse the random ±1 variants (mul/mul±1 -> MUL, etc.),
# which is what makes the table generalize (LOO 61 vs the old per-signature
# table's 35). Cascade: R1 family table -> R2 canonical +/-/* -> R3 concat.
# (audit: _eq_guess_best2.py / _mine_fam_table.py)
_GUESS_FAM = {
    "add": "ADD", "add+1": "ADD", "add-1": "ADD", "sub": "SUB",
    "mul": "MUL", "mul+1": "MUL", "mul-1": "MUL",
    "absdiff": "ABS", "negabsdiff": "ABS", "concat": "CON",
    "max mod min": "MMM", "rsub": "RSUB",
}
_GUESS_CANON = {"+": "add", "-": "sub", "*": "mul"}
# Re-mined 2026-06-05 under the current narrator (rconcat in the candidate pool +
# rev_both-first orientation), so the table key matches what Step-1 now computes.
# In-sample R1-cell accuracy 74 -> 85 of 135; full guess coverage 82 -> 85 of 136.
_GUESS_FAM_R1 = {
    ('id', ('ABS',)): 'add',
    ('id', ('ABS', 'ADD')): 'mul+1',
    ('id', ('ABS', 'CON')): 'mul',
    ('id', ('ABS', 'MUL')): 'add-1',
    ('id', ('ADD',)): 'mul',
    ('id', ('ADD', 'CON')): 'sub',
    ('id', ('ADD', 'MMM')): 'mul',
    ('id', ('ADD', 'MUL')): 'sub',
    ('id', ('ADD', 'SUB')): 'mul',
    ('id', ('CON', 'MUL')): 'sub',
    ('id', ('MMM',)): 'concat',
    ('id', ('MMM', 'MUL')): 'add-1',
    ('id', ('MUL',)): 'add-1',
    ('id', ('MUL', 'SUB')): 'add+1',
    ('id', ('SUB',)): 'concat',
    ('rev_both', ('ABS',)): 'add',
    ('rev_both', ('ABS', 'ADD')): 'mul',
    ('rev_both', ('ABS', 'CON')): 'add',
    ('rev_both', ('ABS', 'MUL')): 'add+1',
    ('rev_both', ('ABS', 'rconcat')): 'add',
    ('rev_both', ('ADD',)): 'absdiff',
    ('rev_both', ('ADD', 'CON')): 'max mod min',
    ('rev_both', ('ADD', 'MMM')): 'mul+1',
    ('rev_both', ('ADD', 'MUL')): 'sub',
    ('rev_both', ('ADD', 'RSUB')): 'mul',
    ('rev_both', ('ADD', 'SUB')): 'mul-1',
    ('rev_both', ('CON',)): 'add',
    ('rev_both', ('CON', 'MUL')): 'sub',
    ('rev_both', ('CON', 'SUB')): 'concat',
    ('rev_both', ('MMM',)): 'mul',
    ('rev_both', ('MMM', 'MUL')): 'add',
    ('rev_both', ('MUL',)): 'max mod min',
    ('rev_both', ('MUL', 'SUB')): 'add',
    ('rev_both', ('MUL', 'rconcat')): 'absdiff',
    ('rev_both', ('SUB',)): 'add',
    ('rev_both', ('SUB', 'rconcat')): 'mul',
    ('rev_both', ('rconcat',)): 'sub',
}


def _format_fam_ruleset() -> str:
    out = [
        "When the query operator is NOT in the examples, guess its rule in priority:",
        "  R1. From the orientation and the FAMILIES of the example-operator rules",
        "      (ADD={add,add±1}, MUL={mul,mul±1}, ABS={absdiff,negabsdiff}, SUB,",
        "       CON=concat, MMM=max mod min, RSUB), look up the unseen operator:",
    ]
    for (mode, fsig), op in sorted(_GUESS_FAM_R1.items()):
        out.append(f"        [{mode}] ({', '.join(fsig)}) -> {op}")
    out.append("  R2. Else use the canonical meaning of the symbol: + -> add, - -> sub, * -> mul.")
    out.append("  R3. Else default to concat.")
    return "\n".join(out)


_GUESS_FAM_RULESET_BLOCK = _format_fam_ruleset()


def _build_guess_v2(problem: Problem) -> str | None:
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
        f"The query operator 【{q_op}】 is NOT one of them — it does not appear in "
        f"the examples, so we cannot read its rule off directly; we must infer it.")
    lines.append("")
    lines.extend(_sigil_check_summary(op_list, detected_fmts, by_op, transformed_groups))
    lines.append("")
    lines.append(
        f"We first pin the orientation from the example operators, then guess the "
        f"unseen operator 【{q_op}】's rule."
    )

    # ---- Step 1: pin the orientation from the example operators ----
    lines.append("")
    lines.append(
        "Step 1 - find the orientation. The whole problem uses one orientation, so "
        "the correct one gives every example operator a consistent rule. Check both.")
    # rev_both first (more common orientation), consistent with deduce. The R1
    # family table is re-mined under this same orientation order.
    orient_specs = [
        ((True, True), "reversed operands and reversed result"),
        ((False, False), "identity (operands and result unchanged)"),
    ]
    consistent: list[tuple[tuple[bool, bool], dict[str, FoundOp]]] = []
    for (rev_ops, rev_res), desc in orient_specs:
        lines.append("")
        lines.append(f"Orientation [{_mode_tag(rev_ops, rev_res)}] - {desc}:")
        op_rules: dict[str, FoundOp] = {}
        for op_char in op_list:
            fo = _emit_operator_search(
                lines, op_char, transformed_groups[op_char],
                detected_fmts[op_char], rev_ops, rev_res)
            if fo is None:
                lines.append(
                    f"  -> 【{op_char}】 has no consistent rule under "
                    f"[{_mode_tag(rev_ops, rev_res)}].")
            else:
                op_rules[op_char] = fo
        if len(op_rules) == len(op_list):
            lines.append(
                f"  -> every example operator has a rule under "
                f"[{_mode_tag(rev_ops, rev_res)}]; this orientation is consistent.")
            consistent.append(((rev_ops, rev_res), op_rules))
        else:
            lines.append(
                f"  -> not every operator has a rule under "
                f"[{_mode_tag(rev_ops, rev_res)}]; this orientation is rejected.")
    if not consistent:
        return None
    chosen_mode, chosen_rules = consistent[0]
    mtag = _mode_tag(*chosen_mode)
    lines.append("")
    if len(consistent) > 1:
        lines.append(
            f"More than one orientation explains the example operators; we use [{mtag}].")
    else:
        lines.append(
            f"Only [{mtag}] explains every example operator, so the orientation is [{mtag}].")

    # ---- Step 2: guess the unseen query operator ----
    if os.environ.get("EQ_GUESS_TABLE", "family") == "exact":
        # exact-operator 3-tier cascade: R1 (signature, transform) -> R2 +/-/* -> R3 concat
        ex_sig = tuple(sorted(fo.op_name for fo in chosen_rules.values()))
        lines.append("")
        lines.append(f"Step 2 - guess the rule for the unseen operator 【{q_op}】.")
        resolved_note = ", ".join(
            f"【{oc}】={chosen_rules[oc].op_name}" for oc in op_list)
        lines.append(f"  Example operators resolve to: {resolved_note}.")
        lines.append(f"  Signature (sorted): ({', '.join(ex_sig)}).")
        lines.append("")
        lines.append(_GUESS_RULESET_BLOCK_EXACT)
        lines.append("")
        if (ex_sig, mtag) in _GUESS_R1:
            guess_op = _GUESS_R1[(ex_sig, mtag)]
            lines.append(
                f"  R1 fires: ({', '.join(ex_sig)}) [{mtag}] -> {guess_op}. "
                f"So 【{q_op}】 = {guess_op}.")
        elif q_op in _GUESS_R2:
            guess_op = _GUESS_R2[q_op]
            if isinstance(guess_op, list):
                guess_op = guess_op[0]
            lines.append(
                f"  No R1 entry for ({', '.join(ex_sig)}) [{mtag}]. R2: 【{q_op}】 -> {guess_op}.")
        else:
            guess_op = "concat"
            lines.append(f"  No R1 or R2 match. R3 default: 【{q_op}】 = concat.")
    else:
        # ---- family-signature cascade (default) ----
        fams = tuple(sorted({_GUESS_FAM.get(fo.op_name, fo.op_name) for fo in chosen_rules.values()}))
        lines.append("")
        lines.append(f"Step 2 - guess the rule for the unseen operator 【{q_op}】.")
        resolved_note = ", ".join(
            f"【{oc}】={chosen_rules[oc].op_name} "
            f"[{_GUESS_FAM.get(chosen_rules[oc].op_name, chosen_rules[oc].op_name)}]"
            for oc in op_list)
        lines.append(f"  Example operators resolve to: {resolved_note}.")
        lines.append(f"  Their families (sorted): ({', '.join(fams)}).")
        lines.append("")
        lines.append(_GUESS_FAM_RULESET_BLOCK)
        lines.append("")
        if (mtag, fams) in _GUESS_FAM_R1:
            guess_op = _GUESS_FAM_R1[(mtag, fams)]
            lines.append(
                f"  R1 fires: [{mtag}] ({', '.join(fams)}) -> {guess_op}. So 【{q_op}】 = {guess_op}.")
        elif q_op in _GUESS_CANON:
            guess_op = _GUESS_CANON[q_op]
            lines.append(
                f"  No R1 entry for this signature. R2: 【{q_op}】 is a canonical arithmetic "
                f"symbol -> {guess_op}.")
        else:
            guess_op = "concat"
            lines.append(f"  No R1 or R2 match. R3 default: 【{q_op}】 = concat.")

    found = FoundOp(op_name=guess_op, rev_ops=chosen_mode[0], rev_res=chosen_mode[1],
                    fmt="num", op_char=q_op)
    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    result_val, steps, signed_raw = _apply_op_v2(found, qa, qb)
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  {_sign_resolution_line(found, signed_raw, result_val)}")
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

    Returns None when no path produces a CoT whose final boxed answer
    matches ``problem.answer`` — so callers can use the output as
    training data without needing their own gold check. Gold is consumed
    ONLY by that final keep/drop filter; generation never reads it.

    ``preferred_mode`` and ``query_op_override`` are accepted for caller
    compatibility but IGNORED (audit 2026-06-10): both were caller-side
    doors for gold-conditioned hints to bias the derivation, which the
    decoy-gold test showed would be a mid-CoT selection leak. The
    derivation is a single fixed procedure for every problem.
    """
    cot = _reasoning_equation_numeric_impl(problem)
    if cot is None:
        return None
    if _extract_last_boxed(cot) != str(problem.answer):
        return None
    return cot


def _reasoning_equation_numeric_impl(problem: Problem) -> str | None:
    """Inner implementation (no gold check). See reasoning_equation_numeric."""
    if problem.category == "equation_numeric_deduce":
        return _build_deduce_v2(problem)
    if problem.category == "equation_numeric_guess":
        return _build_guess_v2(problem)
    return None
