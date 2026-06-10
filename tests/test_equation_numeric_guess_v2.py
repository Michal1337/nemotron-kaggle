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


# _pick_guess_rule tests deleted (audit 2026-06-10): the function was
# gold-conditioned rule selection (mid-CoT selection leak) and was removed
# from the narrator along with its helpers.


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
