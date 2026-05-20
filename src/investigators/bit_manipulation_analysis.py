"""Analyze bit_manipulation accuracy by number of rule-vector sections.

A "section" is a maximal contiguous run of rules sharing the same family.
E.g. if all 8 bits use XOR that's 1 section; if bits 0-3 are XOR and 4-7 are
AND, that's 2 sections.

Usage:
    uv run investigators/bit_manipulation_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Problem
from narrators.reasoning import compare_answer, extract_answer

PROBLEMS_INDEX = Path("problems.jsonl")


@dataclass
class ParsedRule:
    family: str
    primary: int | None
    secondary: int | None


def parse_selected_rules(reasoning_text: str) -> list[ParsedRule] | None:
    """Extract the rules from the 'Selected' block of reasoning output."""
    lines = reasoning_text.split("\n")
    in_selected = False
    rules: list[ParsedRule] = []
    for line in lines:
        if line.strip() == "Selected":
            in_selected = True
            continue
        if in_selected:
            if line.strip() == "":
                if rules:
                    break
                continue
            # Lines look like: "0 I3" or "3 XOR25" or "5 C0" or "7 default 1"
            m = re.match(r"^\d+\s+(.+)$", line.strip())
            if m:
                expr = m.group(1)
                family, primary, secondary = _parse_expr(expr)
                rules.append(ParsedRule(family, primary, secondary))
            else:
                break
    return rules if len(rules) == 8 else None


def _parse_expr(expr: str) -> tuple[str, int | None, int | None]:
    """Parse an expr like 'AND25' into ('AND', 2, 5) or 'I3' into ('I', 3, None)."""
    if expr.startswith("default"):
        return ("DEFAULT", None, None)
    if expr.startswith("C"):
        return ("Constant", None, None)

    # Order matters: check longer prefixes first
    for prefix in ("XOR-NOT", "OR-NOT", "AND-NOT", "XOR", "OR", "AND", "NOT", "I"):
        if expr.startswith(prefix):
            digits = expr[len(prefix):]
            if len(digits) == 2 and digits[0].isdigit() and digits[1].isdigit():
                return (prefix, int(digits[0]), int(digits[1]))
            if len(digits) == 1 and digits[0].isdigit():
                return (prefix, int(digits[0]), None)
            return (prefix, None, None)
    return (expr, None, None)


def _stride_consistent(prev: ParsedRule, curr: ParsedRule) -> bool:
    """Check if two consecutive rules are stride-consistent (same family, operands +1 mod 8)."""
    if prev.family != curr.family:
        return False
    # Constants and defaults are always consistent with themselves
    if prev.family in ("Constant", "DEFAULT"):
        return True
    # Check primary stride
    if prev.primary is not None and curr.primary is not None:
        if (prev.primary + 1) % 8 != curr.primary:
            return False
    # Check secondary stride
    if prev.secondary is not None and curr.secondary is not None:
        if (prev.secondary + 1) % 8 != curr.secondary:
            return False
    return True


def count_sections(rules: list[ParsedRule]) -> int:
    """Count contiguous segments that are stride-consistent."""
    if not rules:
        return 0
    sections = 1
    for i in range(1, len(rules)):
        if not _stride_consistent(rules[i - 1], rules[i]):
            sections += 1
    return sections


def main() -> None:
    # Load bit_manipulation problem IDs
    bm_ids: list[str] = []
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["category"] == "bit_manipulation":
                bm_ids.append(entry["id"])

    print(f"Found {len(bm_ids)} bit_manipulation problems\n")

    # Stats: sections -> (total, correct, wrong_ids, correct_ids)
    stats: dict[int, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "wrong": [], "right": []}
    )
    skipped = 0

    for pid in bm_ids:
        problem = Problem.load_from_json(pid)
        reasoning_text = reasoning_bit_manipulation(problem)
        if reasoning_text is None:
            skipped += 1
            continue

        rules = parse_selected_rules(reasoning_text)
        if rules is None:
            skipped += 1
            continue

        sections = count_sections(rules)
        predicted = extract_answer(reasoning_text)
        is_correct = compare_answer(problem.answer, predicted)

        bucket = stats[sections]
        bucket["total"] += 1
        if is_correct:
            bucket["correct"] += 1
            if len(bucket["right"]) < 3:
                bucket["right"].append(pid)
        else:
            if len(bucket["wrong"]) < 3:
                bucket["wrong"].append(pid)

    if skipped:
        print(f"Skipped {skipped} problems (no reasoning generated)\n")

    # Print table
    print(
        f"{'sections':>8} | {'total':>6} | {'correct':>7} | {'accuracy':>8} | {'sample correct':30} | {'sample wrong':30}"
    )
    print("-" * 130)
    for sec in sorted(stats.keys()):
        b = stats[sec]
        acc = b["correct"] / b["total"] * 100 if b["total"] else 0
        right_str = ", ".join(b["right"]) if b["right"] else "-"
        wrong_str = ", ".join(b["wrong"]) if b["wrong"] else "-"
        print(
            f"{sec:>8} | {b['total']:>6} | {b['correct']:>7} | {acc:>7.1f}% | {right_str:30} | {wrong_str:30}"
        )

    total = sum(b["total"] for b in stats.values())
    correct = sum(b["correct"] for b in stats.values())
    acc = correct / total * 100 if total else 0
    print("-" * 130)
    print(f"{'TOTAL':>8} | {total:>6} | {correct:>7} | {acc:>7.1f}%")


if __name__ == "__main__":
    main()
