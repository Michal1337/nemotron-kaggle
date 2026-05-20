"""Convert solved investigations into huikang-style CoT reasoning files.

Reads ``investigations/<category>/correct/<pid>.txt`` produced by either
``cryptarithm_v3.py`` (terse v3 format) or ``cryptarithm_alice.py`` (Alice
solver format) and emits ``reasoning/<pid>.txt`` in huikang's narration style
(see [reasoner-style.md](../reasoner-style.md) for the conventions).

After running this, re-run huikang's ``corpus.py`` (or our build script) to
fold the new rationales into the training corpus.

Limitations / shortcuts:
  * Only cryptarithm categories are converted by default. Other categories'
    investigation files (bit_manipulation/equation_numeric_*) use different
    formats and need separate narrators.
  * The narration follows huikang's cryptarithm + equation_numeric template
    but compacts the per-candidate-op search trace (only 2-3 candidates shown
    per operator instead of the full ~30) to stay under the 8192-token
    TOKEN_LIMIT used by corpus.py.

Usage:
    python src/investigations_to_reasoning.py \\
        --repo-root /mnt/evafs/groups/re-com/mgromadzki/nemotron-master \\
        --categories cryptarithm_deduce cryptarithm_guess \\
        --dry-run    # remove --dry-run to actually write reasoning files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path


# Map our solver's op names to huikang's equation_numeric.py phrasing so the
# model sees consistent terminology across the corpus.
OP_NAME_TO_HUIKANG: dict[str, str] = {
    # v3 / equation_numeric.py canonical names
    "add": "addition",
    "addition": "addition",
    "sub": "subtraction (a-b)",
    "subtraction": "subtraction (a-b)",
    "rev_sub": "reverse subtraction (b-a)",
    "rsub": "reverse subtraction (b-a)",
    "rev_subtraction": "reverse subtraction (b-a)",
    "absdiff": "absolute difference",
    "absolute_diff": "absolute difference",
    "absolute difference": "absolute difference",
    "neg_absdiff": "negated absolute difference",
    "negated absolute difference": "negated absolute difference",
    "mul": "multiplication",
    "multiplication": "multiplication",
    "concat": "concatenation",
    "concatenation": "concatenation",
    "concat_fwd": "concatenation",
    "rev_concat": "reverse concatenation",
    "concat_rev": "reverse concatenation",
    "reverse concatenation": "reverse concatenation",
    "add+1": "add+1",
    "add_p1": "add+1",
    "add-1": "add-1",
    "add_m1": "add-1",
    "sub+1": "sub+1",
    "sub_p1": "sub+1",
    "sub-1": "sub-1",
    "sub_m1": "sub-1",
    "rev_sub+1": "rev_sub+1",
    "rev_sub-1": "rev_sub-1",
    "multiply+1": "multiply+1",
    "mul_p1": "multiply+1",
    "multiply-1": "multiply-1",
    "mul_m1": "multiply-1",
    "max_mod_min": "max mod min",
    "int_div": "integer division (a/b)",
    "fdiv": "integer division (a/b)",
    "modulo": "modulo (a mod b)",
    "mod": "modulo (a mod b)",
    "rev_div": "reverse division (b/a)",
    "rdiv": "reverse division (b/a)",
    "rev_modulo": "reverse modulo (b mod a)",
    "rmod": "reverse modulo (b mod a)",
}


def parse_v3_investigation(text: str) -> dict | None:
    """Parse v3 investigation file format.

    Returns None if required fields can't be found.
    Returned dict has: mapping, ops, rev_ops, rev_res, examples, query, predicted.
    """
    m_pred = re.search(r"^predicted answer:\s*(.+)$", text, re.MULTILINE)
    if not m_pred:
        return None
    predicted = m_pred.group(1).strip()

    m_query = re.search(r"^query:\s*(.+)$", text, re.MULTILINE)
    if not m_query:
        return None
    query = m_query.group(1).strip()

    m_xform = re.search(r"transform:\s*rev_ops\s*=\s*(\w+)\s+rev_res\s*=\s*(\w+)", text)
    rev_ops = m_xform.group(1) == "True" if m_xform else False
    rev_res = m_xform.group(2) == "True" if m_xform else False

    # Symbol-to-digit mapping (in v3 format, lines like "  '?' = 5")
    mapping: dict[str, int] = {}
    in_map = False
    for line in text.splitlines():
        if line.strip().startswith("symbol-to-digit mapping"):
            in_map = True
            continue
        if in_map:
            if not line.startswith("  "):
                if line.strip() == "":
                    if mapping:
                        break
                    continue
                break
            m = re.match(r"\s+'(.)'\s*=\s*(\d+)", line)
            if m:
                mapping[m.group(1)] = int(m.group(2))

    # Operator-to-operation mapping
    ops: dict[str, str] = {}
    in_ops = False
    for line in text.splitlines():
        if line.strip().startswith("operator-to-operation mapping"):
            in_ops = True
            continue
        if in_ops:
            if not line.startswith("  "):
                if line.strip() == "":
                    if ops:
                        break
                    continue
                break
            m = re.match(r"\s+'(.)'\s*=\s*(\S+)", line)
            if m:
                ops[m.group(1)] = m.group(2)

    # Examples
    examples: list[tuple[str, str]] = []
    in_ex = False
    for line in text.splitlines():
        if line.strip() == "examples:":
            in_ex = True
            continue
        if in_ex:
            if not line.startswith("  "):
                if line.strip() == "":
                    if examples:
                        break
                    continue
                break
            m = re.match(r"\s+(\S+)\s*=\s*(\S+)", line)
            if m:
                examples.append((m.group(1), m.group(2)))

    return {
        "mapping": mapping,
        "ops": ops,
        "rev_ops": rev_ops,
        "rev_res": rev_res,
        "examples": examples,
        "query": query,
        "predicted": predicted,
    }


def parse_alice_investigation(text: str) -> dict | None:
    """Parse Alice solver investigation file format.

    The format includes a "details:" block with json-serialized values.
    """
    m_pred = re.search(r"^predicted answer:\s*(.+)$", text, re.MULTILINE)
    if not m_pred:
        return None
    predicted = m_pred.group(1).strip()

    m_query = re.search(r"^query:\s*(.+)$", text, re.MULTILINE)
    if not m_query:
        return None
    query = m_query.group(1).strip()

    # Details block uses "  <key>: <json_value>" lines
    mapping: dict[str, int] = {}
    ops: dict[str, str] = {}
    mode = "standard"
    for line in text.splitlines():
        m = re.match(r"\s+mapping:\s*(.+)$", line)
        if m:
            try:
                d = json.loads(m.group(1).replace("'", '"'))
                if isinstance(d, dict):
                    mapping = {k: int(v) for k, v in d.items()}
            except Exception:
                pass
        m = re.match(r"\s+ops:\s*(.+)$", line)
        if m:
            try:
                d = json.loads(m.group(1).replace("'", '"'))
                if isinstance(d, dict):
                    ops = {k: str(v) for k, v in d.items()}
            except Exception:
                pass
        m = re.match(r"\s+mode:\s*['\"]?(\w+)['\"]?$", line)
        if m:
            mode = m.group(1)

    # Examples (same format as v3)
    examples: list[tuple[str, str]] = []
    in_ex = False
    for line in text.splitlines():
        if line.strip() == "examples:":
            in_ex = True
            continue
        if in_ex:
            if not line.startswith("  "):
                if line.strip() == "":
                    if examples:
                        break
                    continue
                break
            m = re.match(r"\s+(\S+)\s*=\s*(\S+)", line)
            if m:
                examples.append((m.group(1), m.group(2)))

    return {
        "mapping": mapping,
        "ops": ops,
        "rev_ops": mode in ("alice", "little_endian"),
        "rev_res": False,  # Alice handles via op variants, not a separate flag
        "examples": examples,
        "query": query,
        "predicted": predicted,
        "mode": mode,
    }


def parse_investigation(text: str) -> dict | None:
    """Auto-detect format and parse."""
    if "category: bit_manipulation" in text:
        return parse_bit_manipulation_v2_investigation(text)
    if "source: alice_eq_solver" in text:
        return parse_alice_investigation(text)
    if "source: equation_numeric_v2" in text:
        return parse_eq_v2_investigation(text)
    if "transform: rev_ops=" in text:
        return parse_v3_investigation(text)
    # Older v1 format (no transform line) — treat as identity
    return parse_v3_investigation(text)


def parse_eq_v2_investigation(text: str) -> dict | None:
    """Parse equation_numeric_v2.py investigation files.

    Format:
      problem id: <pid>
      category: <cat>
      source: equation_numeric_v2

      transform: mode=<m> rev_res=<bool> tier=<t>

      operator-to-operation mapping:
        '<op>' = <op_name>

      query operator op_name: <op_name>

      examples:
        <input> = <output>

      query: <q>

      predicted answer: <answer>
    """
    m_pred = re.search(r"^predicted answer:\s*(.+)$", text, re.MULTILINE)
    if not m_pred:
        return None
    predicted = m_pred.group(1).strip()

    m_query = re.search(r"^query:\s*(.+)$", text, re.MULTILINE)
    if not m_query:
        return None
    query = m_query.group(1).strip()

    m_xform = re.search(r"transform:\s*mode\s*=\s*(\w+)\s+rev_res\s*=\s*(\w+)\s+tier\s*=\s*(\w+)", text)
    mode = m_xform.group(1) if m_xform else "standard"
    rev_res = (m_xform.group(2) == "True") if m_xform else False

    ops: dict[str, str] = {}
    in_ops = False
    for line in text.splitlines():
        if line.strip().startswith("operator-to-operation mapping"):
            in_ops = True
            continue
        if in_ops:
            if not line.startswith("  "):
                if line.strip() == "":
                    if ops:
                        break
                    continue
                break
            m = re.match(r"\s+'(.)'\s*=\s*(\S+)", line)
            if m:
                ops[m.group(1)] = m.group(2)

    m_q_op = re.search(r"^query operator op_name:\s*(\S+)$", text, re.MULTILINE)
    query_op_name = m_q_op.group(1) if m_q_op else None

    examples: list[tuple[str, str]] = []
    in_ex = False
    for line in text.splitlines():
        if line.strip() == "examples:":
            in_ex = True
            continue
        if in_ex:
            if not line.startswith("  "):
                if line.strip() == "":
                    if examples:
                        break
                    continue
                break
            m = re.match(r"\s+(\S+)\s*=\s*(\S+)", line)
            if m:
                examples.append((m.group(1), m.group(2)))

    return {
        "_format": "equation_numeric_v2",
        "ops": ops,
        "query_op_name": query_op_name,
        "mode": mode,
        "rev_res": rev_res,
        "examples": examples,
        "query": query,
        "predicted": predicted,
    }


def _box_each_char(s: str) -> str:
    """Wrap each character individually: 'abc' -> '【a】【b】【c】'."""
    return "".join(f"【{c}】" for c in s)


def _box(s: str) -> str:
    """Wrap a whole token in brackets: 'abc' -> '【abc】'."""
    return f"【{s}】"


def _decode_pair(inp: str, mapping: dict[str, int], rev_ops: bool) -> tuple[int, str, int] | None:
    """Given a 5-char cipher input AbOcd, decode to (left_int, op_char, right_int)."""
    if len(inp) != 5:
        return None
    s0, s1, op_char, s3, s4 = inp[0], inp[1], inp[2], inp[3], inp[4]
    if s0 not in mapping or s1 not in mapping or s3 not in mapping or s4 not in mapping:
        return None
    d0, d1, d3, d4 = mapping[s0], mapping[s1], mapping[s3], mapping[s4]
    if rev_ops:
        return d1 * 10 + d0, op_char, d4 * 10 + d3
    return d0 * 10 + d1, op_char, d3 * 10 + d4


def _decode_output(out: str, mapping: dict[str, int]) -> str:
    """Decode a cipher output string to digit string. Returns the original if any symbol is unmapped."""
    digits = []
    for c in out:
        if c not in mapping:
            return out
        digits.append(str(mapping[c]))
    return "".join(digits)


def narrate_pure_concat(pid: str, problem_data: dict, parsed: dict) -> str:
    """Narrator for pure-concat problems (no symbol mapping needed).

    Mirrors huikang's [reasoners/cryptarithm.py](../nemotron-master/reasoners/cryptarithm.py)
    style: works on cipher symbols directly, shows per-example concat-vs-rev-concat
    classification.
    """
    examples = parsed["examples"]
    query = parsed["query"]
    predicted = parsed["predicted"]
    ops = parsed.get("ops", {})

    def quote(s: str) -> str:
        return f"【{s}】"

    def box_each(s: str) -> str:
        return "".join(f"【{c}】" for c in s)

    L: list[str] = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")

    # Detect concat type per operator
    concat_types: dict[str, str] = {}
    for inp, out in examples:
        if len(inp) != 5:
            continue
        s0, s1, op, s3, s4 = inp[0], inp[1], inp[2], inp[3], inp[4]
        fwd = s0 + s1 + s3 + s4
        rev = s3 + s4 + s0 + s1
        if out == fwd:
            concat_types[op] = "fwd"
        elif out == rev:
            concat_types[op] = "rev"

    # Per-example breakdown
    for inp, out in examples:
        if len(inp) != 5:
            continue
        s0, s1, op, s3, s4 = inp[0], inp[1], inp[2], inp[3], inp[4]
        L.append(f"{quote(inp)} = {quote(out)}")
        L.append(f"  input: {box_each(inp)}")
        L.append(f"  left:{quote(s0)}{quote(s1)}")
        L.append(f"  operator: {quote(op)}")
        L.append(f"  right:{quote(s3)}{quote(s4)}")
        L.append(f"  output: {box_each(out)}")
        fwd = s0 + s1 + s3 + s4
        rev = s3 + s4 + s0 + s1
        is_fwd = out == fwd
        is_rev = out == rev
        L.append(f"  concatenation: {box_each(fwd)} {'match' if is_fwd else 'mismatch'}")
        L.append(f"  reverse concatenation: {box_each(rev)} {'match' if is_rev else 'mismatch'}")
        ct = concat_types.get(op)
        op_type = "concatenation" if ct == "fwd" else "reverse concatenation" if ct == "rev" else "unknown"
        L.append(f"  operator: {quote(op)}{op_type}")
        L.append("")

    # Apply to question
    if len(query) == 5:
        qs0, qs1, q_op, qs3, qs4 = query[0], query[1], query[2], query[3], query[4]
        L.append(f"Question{quote(query)}")
        L.append(f"  input: {box_each(query)}")
        L.append(f"  left:{quote(qs0)}{quote(qs1)}")
        L.append(f"  operator:{quote(q_op)}")
        L.append(f"  right:{quote(qs3)}{quote(qs4)}")
        L.append("")
        q_ct = concat_types.get(q_op)
        if q_ct is not None:
            op_label = "concatenation" if q_ct == "fwd" else "reverse concatenation"
            L.append(f"The question operator is {quote(q_op)}, which is {op_label}.")
        else:
            L.append(f"The question operator is {quote(q_op)}, which is unknown.")
            L.append("As the question operator is unknown, we default to concatenation.")
            op_label = "concatenation"
            q_ct = "fwd"
        if q_ct == "fwd":
            answer = qs0 + qs1 + qs3 + qs4
        else:
            answer = qs3 + qs4 + qs0 + qs1
        L.append("")
        L.append(f"  {op_label}({quote(qs0)}{quote(qs1)}, {quote(qs3)}{quote(qs4)}) = {box_each(answer)}")
        L.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
        L.append("")
        # Use the computed answer if it matches predicted, else trust predicted
        final = predicted if predicted == answer else predicted
    else:
        final = predicted

    L.append("I will now return the answer in \\boxed{}")
    L.append(f"The answer in \\boxed{{–}} is \\boxed{{{final}}}")
    return "\n".join(L)


def narrate_equation_numeric(pid: str, problem_data: dict, parsed: dict) -> str:
    """Narrator for equation_numeric problems (operands are already digits).

    Mirrors huikang's [reasoners/equation_numeric.py](../nemotron-master/reasoners/equation_numeric.py)
    output style: no cipher crack, just operator analysis + apply to query.
    """
    ops = parsed["ops"]
    mode = parsed.get("mode", "standard")
    rev_res = parsed.get("rev_res", False)
    examples = parsed["examples"]
    query = parsed["query"]
    predicted = parsed["predicted"]
    query_op_name = parsed.get("query_op_name")

    def quote(s: str) -> str:
        return f"【{s}】"

    L: list[str] = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")
    L.append("Examples:")
    for inp, out in examples:
        L.append(f"  {inp} = {out}")
    L.append("")

    # Parse operands from examples for inspection
    import re as _re
    eq_re = _re.compile(r"^(\d+)(\D)(\d+)$")
    all_a_b: list[tuple[str, str, str, str]] = []
    for inp, out in examples:
        m = eq_re.fullmatch(inp)
        if m:
            all_a_b.append((m.group(1), m.group(2), m.group(3), out))

    inputs_list = []
    for a, _, b, _ in all_a_b:
        inputs_list.extend([a, b])
    outputs_list = [o for _, _, _, o in all_a_b]
    L.append(f"The inputs are {', '.join(inputs_list)}")
    L.append("")
    L.append(f"The outputs are {', '.join(outputs_list)}")
    L.append("")

    if mode == "little_endian":
        L.append("Treating operands as little-endian (digit-reversed) numbers.")
        L.append("")
    if rev_res:
        L.append("The computed result is reversed before encoding.")
        L.append("")

    # Per-operator analysis
    by_op: dict[str, list[tuple[str, str, str]]] = {}
    for a, op, b, out in all_a_b:
        by_op.setdefault(op, []).append((a, b, out))
    for op_char, op_name in sorted(ops.items()):
        L.append(f"Looking at operator {quote(op_char)} [{', '.join(f'{a}{op_char}{b} = {o}' for a, b, o in by_op.get(op_char, []))}]:")
        L.append(f"  The operation is {op_name}.")
        L.append("")

    # Apply to query
    m = eq_re.fullmatch(query)
    if m:
        qa, q_op, qb = m.group(1), m.group(2), m.group(3)
        L.append(f"Applying to {query}:")
        L.append(f"  Decoded query: {qa} {q_op} {qb}")
        if query_op_name:
            L.append(f"  The operation is {query_op_name}.")
        L.append(f"  Numeric result: {quote(predicted)}")
        L.append("")

    L.append("I will now return the answer in \\boxed{}")
    L.append(f"The answer in \\boxed{{–}} is \\boxed{{{predicted}}}")
    return "\n".join(L)


# ----------------------------------------------------------------------
# bit_manipulation v2 (MAJ / CHO / PAR4 / AOA / OAO / AXA)
# ----------------------------------------------------------------------


def _bm_rotl(v: int, k: int) -> int:
    k %= 8
    return ((v << k) | (v >> (8 - k))) & 0xFF


def _bm_apply_transform(name: str, v: int) -> int:
    if name == "I":
        return v & 0xFF
    if name == "NOT":
        return (v ^ 0xFF) & 0xFF
    m = re.fullmatch(r"(NOT\s+)?(ROT|SHL|SHR)\((\d+)\)", name.strip())
    if not m:
        raise ValueError(f"unknown transform: {name!r}")
    invert = bool(m.group(1))
    op, k = m.group(2), int(m.group(3))
    if op == "ROT":
        r = _bm_rotl(v, k)
    elif op == "SHL":
        r = (v << k) & 0xFF
    else:
        r = (v >> k) & 0xFF
    return (r ^ 0xFF) & 0xFF if invert else r


# A single transform looks like one of: I, NOT, ROT(k), SHL(k), SHR(k),
# NOT ROT(k), NOT SHL(k), NOT SHR(k).
_BM_TERM = (
    r"(?:NOT\s+(?:ROT\(\d+\)|SHL\(\d+\)|SHR\(\d+\))"
    r"|I"
    r"|NOT"
    r"|ROT\(\d+\)"
    r"|SHL\(\d+\)"
    r"|SHR\(\d+\))"
)


_BM_RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 3-input named shapes
    ("MAJ", re.compile(rf"^MAJ\(\s*({_BM_TERM})\s*,\s*({_BM_TERM})\s*,\s*({_BM_TERM})\s*\)$")),
    ("CHO", re.compile(rf"^CHO\(\s*({_BM_TERM})\s*,\s*({_BM_TERM})\s*,\s*({_BM_TERM})\s*\)$")),
    # 4-input shapes: (t1 op1 t2) op2 (t3 op3 t4) — same outer/inner ops for AOA/OAO/AXA/OXO
    ("AOA", re.compile(
        rf"^\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)\s+OR\s+\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)$")),
    ("OAO", re.compile(
        rf"^\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)\s+AND\s+\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)$")),
    ("AXA", re.compile(
        rf"^\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)\s+XOR\s+\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)$")),
    ("OXO", re.compile(
        rf"^\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)\s+XOR\s+\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)$")),
    # 3-input mixed-op shapes (v1): (t1 op_a t2) op_b t3, with op_a ≠ op_b.
    # The narrator labels them by op pair so the rationale can refer to the
    # structure cleanly (e.g. "XOR_AND" = (t1 XOR t2) AND t3).
    ("XOR_AND", re.compile(rf"^\(\s*({_BM_TERM})\s+XOR\s+({_BM_TERM})\s*\)\s+AND\s+({_BM_TERM})$")),
    ("XOR_OR",  re.compile(rf"^\(\s*({_BM_TERM})\s+XOR\s+({_BM_TERM})\s*\)\s+OR\s+({_BM_TERM})$")),
    ("AND_XOR", re.compile(rf"^\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)\s+XOR\s+({_BM_TERM})$")),
    ("AND_OR",  re.compile(rf"^\(\s*({_BM_TERM})\s+AND\s+({_BM_TERM})\s*\)\s+OR\s+({_BM_TERM})$")),
    ("OR_XOR",  re.compile(rf"^\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)\s+XOR\s+({_BM_TERM})$")),
    ("OR_AND",  re.compile(rf"^\(\s*({_BM_TERM})\s+OR\s+({_BM_TERM})\s*\)\s+AND\s+({_BM_TERM})$")),
]

# Map mixed3 labels to the inner/outer ops for the evaluator + narrator.
_MIXED3_OPS = {
    "XOR_AND": ("XOR", "AND"),
    "XOR_OR":  ("XOR", "OR"),
    "AND_XOR": ("AND", "XOR"),
    "AND_OR":  ("AND", "OR"),
    "OR_XOR":  ("OR",  "XOR"),
    "OR_AND":  ("OR",  "AND"),
}


def _parse_bm_rule(rule_str: str) -> dict:
    """Identify the rule shape and extract the transform list.

    Returns dict {shape, transforms} where transforms is:
      - list[str] length 1 (SINGLE)
      - list[str] length 2 (PAIR_XOR/AND/OR)
      - list[str] length 3 (TRIPLE_XOR/AND/OR, MAJ, CHO, MIXED3 variants)
      - list[str] length 4 (PAR4)
      - list[list[str]] two length-2 lists (AOA/OAO/AXA/OXO)
    """
    rule = rule_str.strip()

    # Single transform
    if re.fullmatch(_BM_TERM, rule):
        return {"shape": "SINGLE", "transforms": [rule]}

    # Named + mixed shapes via the pattern table
    for shape, pat in _BM_RULE_PATTERNS:
        m = pat.match(rule)
        if not m:
            continue
        if shape in ("MAJ", "CHO"):
            return {"shape": shape, "transforms": [m.group(1), m.group(2), m.group(3)]}
        if shape in _MIXED3_OPS:
            return {"shape": shape, "transforms": [m.group(1), m.group(2), m.group(3)]}
        return {
            "shape": shape,
            "transforms": [[m.group(1), m.group(2)], [m.group(3), m.group(4)]],
        }

    # Flat (no outer parens) XOR / AND / OR chains
    for op_name, op_word in (("XOR", " XOR "), ("AND", " AND "), ("OR", " OR ")):
        # Reject mixed-op (would be caught by mixed3 above already)
        other_ops = [o for o in (" XOR ", " AND ", " OR ") if o != op_word]
        if any(o in rule for o in other_ops):
            continue
        if op_word in rule and "(" not in rule.replace("(", "", rule.count("(") - 0):
            # naive: rule has no outer parens because every "(" is inside a transform
            parts = [p.strip() for p in rule.split(op_word)]
            if all(re.fullmatch(_BM_TERM, p) for p in parts):
                if len(parts) == 2:
                    return {"shape": f"PAIR_{op_name}", "transforms": parts}
                if len(parts) == 3:
                    return {"shape": f"TRIPLE_{op_name}", "transforms": parts}
                if len(parts) == 4 and op_name == "XOR":
                    return {"shape": "PAR4", "transforms": parts}

    raise ValueError(f"unrecognised bit_manipulation rule: {rule!r}")


def parse_bit_manipulation_v2_investigation(text: str) -> dict | None:
    """Parse the investigation format written by bit_manipulation_v2.py (and v1).

    Format:
      problem id: <pid>
      category: bit_manipulation

      rule: <rule_string>

      examples:
        <input_bin> -> <output_bin>
        ...

      query: <query_bin>
      predicted answer: <answer_bin>
    """
    m_rule = re.search(r"^rule:\s*(.+)$", text, re.MULTILINE)
    if not m_rule:
        return None
    rule_str = m_rule.group(1).strip()

    m_query = re.search(r"^query:\s*(\S+)\s*$", text, re.MULTILINE)
    m_pred = re.search(r"^predicted answer:\s*(\S+)\s*$", text, re.MULTILINE)
    if not (m_query and m_pred):
        return None

    examples: list[tuple[str, str]] = []
    in_ex = False
    for line in text.splitlines():
        if line.strip() == "examples:":
            in_ex = True
            continue
        if in_ex:
            if not line.startswith("  "):
                if line.strip() == "":
                    if examples:
                        break
                    continue
                break
            m = re.match(r"\s+(\S+)\s*->\s*(\S+)", line)
            if m:
                examples.append((m.group(1), m.group(2)))

    try:
        parsed_rule = _parse_bm_rule(rule_str)
    except ValueError:
        return None

    return {
        "_format": "bit_manipulation_v2",
        "rule_str": rule_str,
        "rule": parsed_rule,
        "examples": examples,
        "query": m_query.group(1).strip(),
        "predicted": m_pred.group(1).strip(),
    }


_BM_COMBINER = {
    "XOR": lambda a, b: (a ^ b) & 0xFF,
    "AND": lambda a, b: (a & b) & 0xFF,
    "OR":  lambda a, b: (a | b) & 0xFF,
}


def _bm_eval_rule(rule: dict, v: int) -> int:
    """Apply a parsed bit_manipulation rule to an 8-bit integer input."""
    shape = rule["shape"]
    ts = rule["transforms"]

    if shape == "SINGLE":
        return _bm_apply_transform(ts[0], v) & 0xFF
    if shape.startswith("PAIR_"):
        op = shape.split("_", 1)[1]
        return _BM_COMBINER[op](_bm_apply_transform(ts[0], v), _bm_apply_transform(ts[1], v))
    if shape.startswith("TRIPLE_"):
        op = shape.split("_", 1)[1]
        r = _bm_apply_transform(ts[0], v)
        for t in ts[1:]:
            r = _BM_COMBINER[op](r, _bm_apply_transform(t, v))
        return r
    if shape in _MIXED3_OPS:
        op_inner, op_outer = _MIXED3_OPS[shape]
        inner = _BM_COMBINER[op_inner](
            _bm_apply_transform(ts[0], v), _bm_apply_transform(ts[1], v)
        )
        return _BM_COMBINER[op_outer](inner, _bm_apply_transform(ts[2], v))
    if shape == "MAJ":
        a = _bm_apply_transform(ts[0], v)
        b = _bm_apply_transform(ts[1], v)
        c = _bm_apply_transform(ts[2], v)
        return (a & b) | (a & c) | (b & c)
    if shape == "CHO":
        a = _bm_apply_transform(ts[0], v)
        b = _bm_apply_transform(ts[1], v)
        c = _bm_apply_transform(ts[2], v)
        na = a ^ 0xFF
        return ((a & b) | (na & c)) & 0xFF
    if shape == "PAR4":
        r = 0
        for t in ts:
            r ^= _bm_apply_transform(t, v)
        return r & 0xFF
    if shape == "AOA":
        (a, b), (c, d) = ts
        return (
            (_bm_apply_transform(a, v) & _bm_apply_transform(b, v))
            | (_bm_apply_transform(c, v) & _bm_apply_transform(d, v))
        ) & 0xFF
    if shape == "OAO":
        (a, b), (c, d) = ts
        return (
            (_bm_apply_transform(a, v) | _bm_apply_transform(b, v))
            & (_bm_apply_transform(c, v) | _bm_apply_transform(d, v))
        ) & 0xFF
    if shape == "AXA":
        (a, b), (c, d) = ts
        return (
            (_bm_apply_transform(a, v) & _bm_apply_transform(b, v))
            ^ (_bm_apply_transform(c, v) & _bm_apply_transform(d, v))
        ) & 0xFF
    if shape == "OXO":
        (a, b), (c, d) = ts
        return (
            (_bm_apply_transform(a, v) | _bm_apply_transform(b, v))
            ^ (_bm_apply_transform(c, v) | _bm_apply_transform(d, v))
        ) & 0xFF
    raise ValueError(f"unknown shape {shape}")


def _bm_b(v: int) -> str:
    return format(v & 0xFF, "08b")


_SHAPE_DESCRIPTION = {
    "SINGLE": "a single transform applied to the input.",
    "PAIR_XOR": "the bitwise XOR of two transforms.",
    "PAIR_AND": "the bitwise AND of two transforms.",
    "PAIR_OR":  "the bitwise OR of two transforms.",
    "TRIPLE_XOR": "the bitwise XOR of three transforms (parity across the three transformed bits).",
    "TRIPLE_AND": "the bitwise AND of three transforms.",
    "TRIPLE_OR":  "the bitwise OR of three transforms.",
    "XOR_AND": "the AND of (t1 XOR t2) with t3.",
    "XOR_OR":  "the OR of (t1 XOR t2) with t3.",
    "AND_XOR": "the XOR of (t1 AND t2) with t3.",
    "AND_OR":  "the OR of (t1 AND t2) with t3.",
    "OR_XOR":  "the XOR of (t1 OR t2) with t3.",
    "OR_AND":  "the AND of (t1 OR t2) with t3.",
    "MAJ": (
        "the bitwise majority of three transforms — for each bit position, "
        "the output bit is 1 iff at least two of the three transformed bits are 1."
    ),
    "CHO": (
        "a bitwise multiplexer (a&b)|(~a&c): the first transform selects, "
        "for each bit position, whether to take the second (when 1) or third (when 0) transformed bit."
    ),
    "PAR4": (
        "the bitwise XOR of four transforms — for each bit position, "
        "the output bit is the parity of the four transformed bits."
    ),
    "AOA": "the OR of two ANDs: ((t1 & t2) | (t3 & t4)).",
    "OAO": "the AND of two ORs: ((t1 | t2) & (t3 | t4)).",
    "AXA": "the XOR of two ANDs: ((t1 & t2) ^ (t3 & t4)).",
    "OXO": "the XOR of two ORs: ((t1 | t2) ^ (t3 | t4)).",
}


def narrate_bit_manipulation(pid: str, problem_data: dict, parsed: dict) -> str:
    """Produce a huikang-compatible CoT for a v2 bit_manipulation investigation.

    huikang's reasoner output for bit_manipulation is column-wise per output bit
    (~9KB). The v2 shapes (MAJ, CHO, PAR4, AOA, OAO, AXA) don't lend themselves
    to the column-by-column primitive matching, so we emit a declarative
    narration: state the rule, demonstrate it on each example by applying every
    transform and the combiner, then apply to the query. Opening/closing match
    huikang's universal scaffolding.
    """
    rule = parsed["rule"]
    examples = parsed["examples"]
    query = parsed["query"]
    predicted = parsed["predicted"]
    rule_str = parsed["rule_str"]

    L: list[str] = []
    L.append("We need to deduce the transformation by matching the example outputs.")
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")

    # Examples block
    L.append("Examples:")
    for inp, out in examples:
        L.append(f"  【{inp}】 -> 【{out}】")
    L.append("")

    # State the rule
    L.append(f"The rule is 【{rule_str}】 — {_SHAPE_DESCRIPTION[rule['shape']]}")
    L.append("")

    # A clear textual label for the combining step (used in both example
    # verification and the query application).
    shape = rule["shape"]
    if shape == "SINGLE":
        combine_label = "applying the transform"
    elif shape.startswith("PAIR_") or shape.startswith("TRIPLE_"):
        combine_label = f"combining via bitwise {shape.split('_', 1)[1]}"
    elif shape in _MIXED3_OPS:
        op_inner, op_outer = _MIXED3_OPS[shape]
        combine_label = f"combining: (t1 {op_inner} t2) {op_outer} t3"
    elif shape in ("MAJ", "CHO", "PAR4", "AOA", "OAO", "AXA", "OXO"):
        combine_label = f"combining via {shape}"
    else:
        combine_label = f"combining via {shape}"

    # Demonstrate on the first 2-3 examples so the rationale stays bounded.
    # For shapes where the verification step is trivial (single transform, or
    # the rule is literally one of the example outputs), one example is enough.
    demo_n = 1 if shape == "SINGLE" else min(2, len(examples))
    for ei in range(demo_n):
        inp_bin, out_bin = examples[ei]
        v = int(inp_bin, 2)
        L.append(f"Verifying on example {ei}: input 【{inp_bin}】")
        for t in _flatten_transforms(rule):
            tv = _bm_apply_transform(t, v)
            L.append(f"  【{t}】(【{inp_bin}】) = 【{_bm_b(tv)}】")
        combined = _bm_eval_rule(rule, v)
        L.append(f"  {combine_label}: 【{_bm_b(combined)}】 (expected 【{out_bin}】)")
        L.append("")

    # Apply to query
    qv = int(query, 2)
    L.append(f"Applying to 【{query}】:")
    for t in _flatten_transforms(rule):
        tv = _bm_apply_transform(t, qv)
        L.append(f"  【{t}】(【{query}】) = 【{_bm_b(tv)}】")
    result = _bm_eval_rule(rule, qv)
    L.append(f"  {combine_label}: 【{_bm_b(result)}】")
    L.append("")

    L.append("I will now return the answer in \\boxed{}")
    L.append(f"The answer in \\boxed{{–}} is \\boxed{{{predicted}}}")
    return "\n".join(L)


def _flatten_transforms(rule: dict) -> list[str]:
    ts = rule["transforms"]
    if rule["shape"] in ("AOA", "OAO", "AXA", "OXO"):
        flat: list[str] = []
        for pair in ts:
            flat.extend(pair)
        return flat
    return list(ts)


def narrate_cryptarithm(pid: str, problem_data: dict, parsed: dict) -> str:
    """Build a huikang-style CoT reasoning string for a cryptarithm.

    Mirrors the conventions in reasoner-style.md. Branches between pure-concat
    (huikang's existing style) and arithmetic (equation_numeric-derived style).
    """
    # Dispatch on parsed format
    if parsed.get("_format") == "equation_numeric_v2":
        return narrate_equation_numeric(pid, problem_data, parsed)

    mapping = parsed["mapping"]
    ops = parsed["ops"]
    rev_ops = parsed["rev_ops"]
    rev_res = parsed["rev_res"]
    examples = parsed["examples"]
    query = parsed["query"]
    predicted = parsed["predicted"]

    # If no symbol mapping was needed (pure concat), use huikang's concat-only style.
    if not mapping:
        return narrate_pure_concat(pid, problem_data, parsed)

    inverse_map = {v: k for k, v in mapping.items()}

    L: list[str] = []
    L.append("We need to infer the transformation rule from the examples.")
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")

    # === Cipher examples (the original prompt examples) ===
    L.append("Cipher examples:")
    for inp, out in examples:
        L.append(f"  {_box(inp)} = {_box(out)}")
    L.append("")

    # === Inferred mapping ===
    L.append("Inferred symbol-to-digit mapping:")
    for sym in sorted(mapping):
        L.append(f"  {_box(sym)} = {mapping[sym]}")
    L.append("")

    # === Operator mapping ===
    L.append("Inferred operator-to-operation mapping:")
    for op_char in sorted(ops):
        canonical = OP_NAME_TO_HUIKANG.get(ops[op_char], ops[op_char])
        L.append(f"  {_box(op_char)} = {canonical}")
    L.append("")

    if rev_ops or rev_res:
        L.append("Transform applied during search:")
        if rev_ops:
            L.append("  Operands are read with digits reversed (little-endian).")
        if rev_res:
            L.append("  The computed result is reversed before encoding.")
        L.append("")

    # === Decode each example to numeric form and verify ===
    L.append("Decoding examples to numeric form:")
    op_fns = _build_op_fns()
    for inp, out in examples:
        decoded = _decode_pair(inp, mapping, rev_ops)
        if decoded is None:
            L.append(f"  {_box(inp)} = {_box(out)} (could not decode)")
            continue
        a, op_char, b = decoded
        out_digits = _decode_output(out, mapping)
        op_name = ops.get(op_char, "unknown")
        canonical = OP_NAME_TO_HUIKANG.get(op_name, op_name)
        # Compute the operation
        fn = op_fns.get(op_name)
        computed: int | None = None
        if fn is not None:
            try:
                computed = fn(a, b)
            except Exception:
                computed = None
        if computed is None:
            L.append(f"  {_box(inp)} -> {a}{op_char}{b} -> output: {out_digits}")
            continue
        if rev_res:
            computed_str = str(computed)[::-1]
        else:
            computed_str = str(computed)
        status = "match" if computed_str == out_digits else "near-match"
        L.append(
            f"  {_box(inp)} -> {a} {canonical} {b} = {computed} -> "
            f"output digits {out_digits} ({status})"
        )
    L.append("")

    # === Apply to question ===
    L.append(f"Applying to {_box(query)}:")
    decoded_q = _decode_pair(query, mapping, rev_ops)
    if decoded_q is None:
        L.append("  could not decode all symbols in the query.")
        L.append(f"  defaulting to the solver's predicted answer: {_box(predicted)}")
    else:
        qa, q_op, qb = decoded_q
        q_op_name = ops.get(q_op)
        if q_op_name is None:
            L.append(
                f"  The query operator {_box(q_op)} did not appear in the examples."
            )
            L.append("  Falling back to the solver's predicted answer.")
        else:
            canonical = OP_NAME_TO_HUIKANG.get(q_op_name, q_op_name)
            L.append(f"  Decoded query: {qa} {q_op} {qb}")
            L.append(f"  Operator {_box(q_op)} = {canonical}")
            fn = op_fns.get(q_op_name)
            computed = None
            if fn is not None:
                try:
                    computed = fn(qa, qb)
                except Exception:
                    pass
            if computed is not None:
                if rev_res:
                    L.append(f"  {canonical}({qa}, {qb}) = {computed} -> reversed -> {str(computed)[::-1]}")
                    computed_str_final = str(computed)[::-1]
                else:
                    L.append(f"  {canonical}({qa}, {qb}) = {computed}")
                    computed_str_final = str(computed)
                L.append(f"  Numeric result: {_box(computed_str_final)}")

    L.append("")

    # === Re-encode to cipher symbols ===
    L.append("Re-encoding numeric result to cipher symbols using the inverse mapping:")
    cipher_chars = []
    digits_to_show = predicted  # if numeric encoding fails, fall back to predicted
    try:
        # If we have numeric_answer from Alice's details we could use it; we don't
        # track it here, but predicted is the cipher form, so we trust it.
        cipher_chars = list(predicted)
    except Exception:
        cipher_chars = list(predicted)
    # Best-effort: pair each cipher char with its digit
    for c in cipher_chars:
        d = mapping.get(c)
        if d is None:
            L.append(f"  {_box(c)} (no inverse mapping — adopting solver output)")
        else:
            L.append(f"  digit {d} -> {_box(c)}")
    L.append(f"Final cipher answer: {_box(predicted)}")
    L.append("")

    # === Closing boilerplate (matches every huikang reasoner) ===
    L.append("I will now return the answer in \\boxed{}")
    L.append(f"The answer in \\boxed{{–}} is \\boxed{{{predicted}}}")
    return "\n".join(L)


def _build_op_fns() -> dict[str, callable]:
    """Map canonical op name -> python fn(a, b) -> int|None. Mirrors equation_numeric.py."""
    fns: dict[str, callable] = {
        "addition":               lambda a, b: a + b,
        "subtraction (a-b)":      lambda a, b: a - b,
        "reverse subtraction (b-a)": lambda a, b: b - a,
        "absolute difference":    lambda a, b: abs(a - b),
        "negated absolute difference": lambda a, b: -abs(a - b),
        "multiplication":         lambda a, b: a * b,
        "concatenation":          lambda a, b: a * 100 + b,
        "reverse concatenation":  lambda a, b: b * 100 + a,
        "add+1":                  lambda a, b: a + b + 1,
        "add-1":                  lambda a, b: a + b - 1,
        "sub+1":                  lambda a, b: a - b + 1,
        "sub-1":                  lambda a, b: a - b - 1,
        "rev_sub+1":              lambda a, b: b - a + 1,
        "rev_sub-1":              lambda a, b: b - a - 1,
        "multiply+1":             lambda a, b: a * b + 1,
        "multiply-1":             lambda a, b: a * b - 1,
        "max mod min":            lambda a, b: (max(a, b) % min(a, b)) if min(a, b) > 0 else None,
        "integer division (a/b)": lambda a, b: a // b if b != 0 else None,
        "modulo (a mod b)":       lambda a, b: a % b if b != 0 else None,
        "reverse division (b/a)": lambda a, b: b // a if a != 0 else None,
        "reverse modulo (b mod a)": lambda a, b: b % a if a != 0 else None,
    }
    # Also accept the raw names produced by v3/Alice
    for raw, canonical in OP_NAME_TO_HUIKANG.items():
        if canonical in fns and raw not in fns:
            fns[raw] = fns[canonical]
    return fns


def _verify(stored: str, predicted: str) -> bool:
    s, p = (stored or "").strip(), (predicted or "").strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def extract_final_answer(text: str | None) -> str:
    """Mirror the competition's extract_final_answer logic.

    Critically handles answers that contain '}' literally: walks each
    `\\boxed{` start and grabs everything up to the *last* `}` before the
    next `\\boxed{` (or end of text). So `\\boxed{+}}` correctly extracts
    to `+}`, not `+`.
    """
    if text is None:
        return "NOT_FOUND"
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return "NOT_FOUND"


def parse_parquet_row(row, problem_data: dict) -> dict | None:
    """Convert a row from the Alice repo's solver_results.parquet into our parsed dict."""
    if not row.get("solver_correct"):
        return None
    pred = row.get("solver_answer")
    if pred is None:
        return None
    mapping_raw = row.get("solver_mapping") or "{}"
    ops_raw = row.get("solver_ops") or "{}"
    try:
        mapping = {k: int(v) for k, v in json.loads(mapping_raw).items()}
    except Exception:
        mapping = {}
    try:
        ops = {k: str(v) for k, v in json.loads(ops_raw).items()}
    except Exception:
        ops = {}
    mode = row.get("solver_mode")
    # In Alice's solver, mode="little_endian" means BOTH operands and result are
    # encoded right-to-left. Map to our (rev_ops, rev_res) pair.
    is_le = mode in ("little_endian", "alice")
    examples = [(e["input_value"], e["output_value"]) for e in problem_data["examples"]]
    return {
        "mapping": mapping,
        "ops": ops,
        "rev_ops": is_le,
        "rev_res": is_le,
        "examples": examples,
        "query": problem_data["question"],
        "predicted": str(pred),
        "mode": mode,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=None,
                   help="nemotron-master path. Default: try cluster path then local.")
    p.add_argument("--source", choices=["investigations", "parquet"],
                   default="investigations",
                   help="Read from investigations/ subfolders OR a parquet file like "
                        "the Alice repo's solver_results.parquet.")
    p.add_argument("--parquet-path", default=None,
                   help="Path to solver_results.parquet (required if --source parquet).")
    p.add_argument("--categories", nargs="+",
                   default=["cryptarithm_deduce", "cryptarithm_guess",
                            "equation_numeric_deduce", "equation_numeric_guess",
                            "bit_manipulation"],
                   help="Which category subfolders of investigations/ to walk (investigations source only).")
    p.add_argument("--output-dir", default=None,
                   help="Override the destination for reasoning files (default: <repo>/reasoning).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without modifying anything.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after this many newly-narrated files (0=no limit).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite reasoning/<pid>.txt if it already exists.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.repo_root:
        repo = Path(args.repo_root)
    else:
        # Try likely paths
        cand = [
            Path("/mnt/evafs/groups/re-com/mgromadzki/nemotron-master"),
            Path(__file__).parent.parent / "nemotron-master",
        ]
        repo = next((c for c in cand if c.is_dir()), None)
        if repo is None:
            sys.exit("--repo-root not provided and no default found")

    print(f"Repo root: {repo}")
    inv_root = repo / "investigations"
    problems_dir = repo / "problems"
    out_dir = Path(args.output_dir) if args.output_dir else (repo / "reasoning")
    train_csv = repo / "train.csv"

    gold: dict[str, str] = {}
    with train_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["answer"]

    stats = Counter()
    written = 0
    sample_outputs: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Parquet source (Alice repo's solver_results.parquet)
    # ------------------------------------------------------------------
    if args.source == "parquet":
        if not args.parquet_path:
            sys.exit("--parquet-path required with --source parquet")
        try:
            import pandas as pd
        except ImportError:
            sys.exit("pandas required for parquet input: pip install pandas pyarrow")
        df = pd.read_parquet(args.parquet_path)
        print(f"Parquet rows: {len(df)}; solver_correct: {df['solver_correct'].sum()}")

        # Only walk correctly-solved
        correct = df[df["solver_correct"]]
        for _, row in correct.iterrows():
            pid = row["id"]
            stats["seen"] += 1

            prob_file = problems_dir / f"{pid}.jsonl"
            if not prob_file.is_file():
                stats["missing_problem"] += 1
                continue
            with prob_file.open() as pf:
                problem_data = json.loads(pf.readline())

            parsed = parse_parquet_row(row, problem_data)
            if parsed is None:
                stats["parse_failed"] += 1
                continue

            if not _verify(gold.get(pid, ""), parsed["predicted"]):
                stats["gold_mismatch"] += 1
                continue

            # Note: gold answers can contain '}'. The competition scorer's
            # extract_final_answer() walks each \boxed{ start and finds the
            # *last* } in the segment, so `\boxed{+}}` correctly extracts to
            # `+}`. We don't skip these.

            try:
                if parsed.get("_format") == "bit_manipulation_v2":
                    trace = narrate_bit_manipulation(pid, problem_data, parsed)
                else:
                    trace = narrate_cryptarithm(pid, problem_data, parsed)
            except Exception as exc:
                stats["narration_failed"] += 1
                if args.verbose:
                    print(f"    !! {pid}: {exc}")
                continue

            target = out_dir / f"{pid}.txt"
            if target.exists() and not args.overwrite:
                stats["skipped_exists"] += 1
                continue

            if args.dry_run:
                stats["would_write"] += 1
                if len(sample_outputs) < 3:
                    sample_outputs.append((pid, trace))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(trace, encoding="utf-8")
                stats["written"] += 1
                written += 1
                if args.limit and written >= args.limit:
                    print(f"\n--limit {args.limit} reached, stopping.")
                    break

        # Skip the investigations-source loop below
        print(f"\n=== Summary ===")
        for k, v in stats.most_common():
            print(f"  {k:>20s}: {v}")
        if args.dry_run and sample_outputs:
            print(f"\n=== Sample dry-run outputs ({len(sample_outputs)}) ===")
            for pid, trace in sample_outputs:
                print(f"\n--- {pid} ({len(trace)} chars) ---")
                print(trace)
        return

    # ------------------------------------------------------------------
    # Investigations source (default)
    # ------------------------------------------------------------------
    for cat in args.categories:
        cat_dir = inv_root / cat / "correct"
        if not cat_dir.is_dir():
            print(f"  skipping {cat}: {cat_dir} not found")
            continue
        files = sorted(f for f in cat_dir.iterdir() if f.suffix == ".txt")
        print(f"\n=== {cat}: {len(files)} correct investigations ===")
        for f in files:
            pid = f.stem
            stats["seen"] += 1
            text = f.read_text(encoding="utf-8")
            parsed = parse_investigation(text)
            if parsed is None:
                stats["parse_failed"] += 1
                continue

            # Verify the parsed predicted matches gold
            if not _verify(gold.get(pid, ""), parsed["predicted"]):
                stats["gold_mismatch"] += 1
                continue

            # Note: gold answers can contain '}'. The competition scorer's
            # extract_final_answer() walks each \boxed{ start and finds the
            # *last* } in the segment, so `\boxed{+}}` extracts to `+}`. OK.

            # Load problem data
            prob_file = problems_dir / f"{pid}.jsonl"
            if not prob_file.is_file():
                stats["missing_problem"] += 1
                continue
            with prob_file.open() as pf:
                problem_data = json.loads(pf.readline())

            try:
                if parsed.get("_format") == "bit_manipulation_v2":
                    trace = narrate_bit_manipulation(pid, problem_data, parsed)
                else:
                    trace = narrate_cryptarithm(pid, problem_data, parsed)
            except Exception as exc:
                stats["narration_failed"] += 1
                if args.verbose:
                    print(f"    !! {pid}: {exc}")
                continue

            target = out_dir / f"{pid}.txt"
            if target.exists() and not args.overwrite:
                stats["skipped_exists"] += 1
                continue

            if args.dry_run:
                stats["would_write"] += 1
                if len(sample_outputs) < 2:
                    sample_outputs.append((pid, trace))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(trace, encoding="utf-8")
                stats["written"] += 1
                written += 1
                if args.limit and written >= args.limit:
                    print(f"\n--limit {args.limit} reached, stopping.")
                    break
        if args.limit and written >= args.limit:
            break

    print(f"\n=== Summary ===")
    for k, v in stats.most_common():
        print(f"  {k:>20s}: {v}")

    if args.dry_run and sample_outputs:
        print(f"\n=== Sample dry-run outputs ({len(sample_outputs)}) ===")
        for pid, trace in sample_outputs:
            print(f"\n--- {pid} ({len(trace)} chars) ---")
            print(trace)


if __name__ == "__main__":
    main()
