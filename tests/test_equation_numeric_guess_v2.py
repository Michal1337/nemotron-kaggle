"""Tests for the v2 equation_numeric_guess narrator.

Run from repo root:  pytest tests/test_equation_numeric_guess_v2.py -v
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reasoners.equation_numeric import (
    _build_guess_v2,
    _pick_guess_rule,
    _GUESS_R1,
    _GUESS_R2,
    _GUESS_R3,
    _GUESS_R4,
    reasoning_equation_numeric,
    FoundOp,
)
from reasoners.store_types import Example, Problem


PROBLEMS_DIR = ROOT / "nemotron-master" / "problems"
BOX_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    if not text:
        return None
    starts = list(BOX_RE.finditer(text))
    if not starts:
        return None
    matches = []
    for i, m in enumerate(starts):
        start = m.end()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[start:end]
        lb = seg.rfind("}")
        matches.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in matches if m.strip()]
    return ne[-1] if ne else matches[-1].strip()


def load_problem(pid):
    with (PROBLEMS_DIR / f"{pid}.jsonl").open(encoding="utf-8") as f:
        d = json.loads(f.read())
    return Problem(
        id=d["id"], category=d["category"],
        examples=[Example(ex["input_value"], ex["output_value"]) for ex in d["examples"]],
        question=d["question"], answer=d["answer"],
    )


# ---------------- Unit: _pick_guess_rule heuristic priority ----------------

def _fake_ex_results(ops_to_rules):
    return {op: FoundOp(op_name=rule, rev_ops=False, rev_res=False, fmt="num", op_char=op)
            for op, rule in ops_to_rules.items()}


class TestPickGuessRule:
    def test_returns_none_when_no_rule_produces_gold(self):
        # Bogus gold that no op produces from (5, 3). Use ex_ops with sig
        # NOT in R3 so we exercise the no-fitting-rule path properly.
        out = _pick_guess_rule(
            "@", "5", "3", "99999",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is None

    def test_r1_exact_op_sig_lookup(self):
        # ex_ops have signature ('addition', 'multiplication') → R1 = 'subtraction (a-b)'.
        # qop = '?' (non-canonical), 5-3 = 2.
        out = _pick_guess_rule(
            "?", "5", "3", "2",
            _fake_ex_results({"a": "addition", "b": "multiplication"}),
        )
        assert out is not None
        fop, reason = out
        assert fop.op_name == "subtraction (a-b)"
        assert "R1" in reason

    def test_r2_natural_arithmetic_plus(self):
        # 5+3 = 8 (addition). qop = '+', canonical = addition.
        # Use ex_op with sig NOT in R1 so R2 wins cleanly.
        out = _pick_guess_rule(
            "+", "5", "3", "8",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is not None
        fop, reason = out
        assert fop.op_name == "addition"
        assert "R2" in reason and "standard arithmetic" in reason

    def test_r2_natural_arithmetic_minus_signed(self):
        # 5-9 = -4 (subtraction). qop = '-', canonical = subtraction.
        out = _pick_guess_rule(
            "-", "5", "9", "-4",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is not None
        fop, _ = out
        assert fop.op_name == "subtraction (a-b)"

    def test_r3_per_char_at_means_addition(self):
        # qop = '@', dataset convention = addition. Use ex_op with sig NOT in R1.
        out = _pick_guess_rule(
            "@", "5", "3", "8",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is not None
        fop, reason = out
        assert fop.op_name == "addition"
        assert "R3" in reason and "dataset convention" in reason

    def test_r1_drops_when_rule_doesnt_fit_no_fallthrough(self):
        # qop = '+'. ex_ops sig ('subtraction (a-b)',) → R1 = 'concatenation'.
        # gold = 8 = 5+3 = addition. R1 says concat (53) — doesn't fit gold.
        # NO fallthrough: pid is dropped even though R2 would have caught it.
        # This guarantees training consistency (same sig always picks R1).
        out = _pick_guess_rule(
            "+", "5", "3", "8",
            _fake_ex_results({"a": "subtraction (a-b)"}),
        )
        assert out is None

    def test_no_rule_when_nothing_applies(self):
        # qop is a rare char with no per-char rule. ex_ops sig NOT in R1.
        # gold = 15. Even though multiplication fits, no rule's prediction
        # matches → drop.
        out = _pick_guess_rule(
            "~", "5", "3", "15",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is None

    def test_r4_per_char_majority(self):
        # qop = '/' which has R4 = multiplication. 5*3 = 15. ex_ops sig
        # NOT in R1 so R4 gets to fire.
        out = _pick_guess_rule(
            "/", "5", "3", "15",
            _fake_ex_results({"a": "reverse subtraction (b-a)"}),
        )
        assert out is not None
        fop, reason = out
        assert fop.op_name == "multiplication"
        assert "R4" in reason


# ---------------- Format invariants on the produced CoT ----------------

def _format_invariants(cot):
    assert cot is not None
    # Shared header
    assert "The question is:" in cot
    # DIVERGENCE marker
    assert "NOT in the examples" in cot
    # Up-front ruleset (the key structural change in v2-guess)
    assert all(f"R{i}." in cot for i in (1, 2, 3, 4, 5))
    # Per-example-op search trace (verbose, "Looking at operator" lines)
    assert "Looking at operator" in cot
    # Brief recap of inferred rules per example op
    assert "Summary of inferred rules for the example operators:" in cot
    # The pattern-match application
    assert "Applying the guessing rules to" in cot
    # Final apply + boxed
    assert "Applying to" in cot
    assert "\\boxed{" in cot


class TestRealGuessPids:
    @pytest.mark.parametrize("pid", ["1b019515", "260f20c1"])
    def test_real_guess_pid_format(self, pid):
        if not (PROBLEMS_DIR / f"{pid}.jsonl").exists():
            pytest.skip(f"{pid} not present")
        p = load_problem(pid)
        if p.category != "equation_numeric_guess":
            pytest.skip(f"{pid} is {p.category}")
        cot = reasoning_equation_numeric(p)
        if cot is None:
            pytest.skip(f"v2 returned None for {pid} (unsolvable)")
        _format_invariants(cot)


# ---------------- Coverage integration: all 136 guess pids ----------------

def _all_guess_pids():
    out = []
    for fn in sorted(os.listdir(PROBLEMS_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        pid = fn[:-6]
        with (PROBLEMS_DIR / fn).open(encoding="utf-8") as f:
            d = json.loads(f.read())
        if d.get("category") == "equation_numeric_guess":
            out.append(pid)
    return out


@pytest.mark.slow
def test_all_136_guess_coverage_floor_and_correctness():
    """Run guess v2 narrator on every guess pid. Assert:
      - At least 100 produce a non-None CoT (the ~105 solvable expected).
      - Every non-None CoT has the boxed answer == gold.
      - Every non-None CoT passes format invariants.
    """
    pids = _all_guess_pids()
    assert len(pids) == 136, f"expected 136 guess pids, got {len(pids)}"

    coverage = 0
    correct = 0
    invariant_violations = []
    none_pids = []
    for pid in pids:
        p = load_problem(pid)
        cot = reasoning_equation_numeric(p)
        if cot is None:
            none_pids.append(pid)
            continue
        coverage += 1
        try:
            _format_invariants(cot)
        except AssertionError as e:
            invariant_violations.append((pid, str(e)[:120]))
        if extract_boxed(cot) == p.answer:
            correct += 1

    # Hard invariant: every produced CoT must be format-valid AND correct.
    assert not invariant_violations, (
        f"format violations: {len(invariant_violations)} pids; "
        f"first 3: {invariant_violations[:3]}"
    )
    # If we produced a CoT, it MUST equal gold (we filter on gold-matching).
    assert correct == coverage, (
        f"coverage produced {coverage} CoTs but only {correct} match gold; "
        f"narrator should never emit a wrong-answer CoT"
    )

    # Strict no-fallthrough coverage: ~69 expected (consistent rule per
    # precondition; pid dropped if chosen rule doesn't fit gold).
    assert coverage >= 60, (
        f"coverage {coverage} below floor 60; strict-no-fallthrough expects ~69 pids"
    )
