"""Generate augmented data with secret processing rules on tokenizer tokens.

Each problem:
1. The "secret rule" is breaking text into individual characters
2. Shows demo pairs in sample input/output sections
3. Has 200 test lines in "your input"

Output format: –s–e–x–v–e–x– (leading/trailing –, all chars merged across tokens)

Every bare and space-prefixed token appears at least once across all problems.
The smaller pool is reused to fill remaining slots.

Usage: uv run python3 -m investigators.augment_data
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer.json"
OUTPUT_PATH = Path(__file__).parent.parent / "augmented.csv"

EN_DASH = "\u2013"
LINES_PER_PROBLEM = 50


def load_tokens() -> tuple[list[str], list[str]]:
    """Load bare and space-prefixed lowercase alphabetic tokens from the tokenizer."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab = tok.get_vocab()

    bare: list[str] = []
    spaced: list[str] = []
    for token in vocab:
        if token.startswith("\u0120") and len(token) > 1:
            text = token[1:]
            if text.isascii() and text.isalpha() and text.islower() and 2 <= len(text) <= 8:
                spaced.append(text)
        elif (
            token.isascii()
            and token.isalpha()
            and token.islower()
            and 2 <= len(token) <= 8
        ):
            bare.append(token)

    return sorted(bare), sorted(spaced)


def spell_out(text: str) -> str:
    """Break text into individual characters, strip spaces, wrap with en-dashes."""
    chars = [c for c in text if c != " "]
    return EN_DASH + EN_DASH.join(chars) + EN_DASH


def main() -> None:
    bare_tokens, spaced_tokens = load_tokens()
    print(f"Loaded {len(bare_tokens)} bare tokens, {len(spaced_tokens)} spaced tokens")

    rng = random.Random(42)

    # Shuffle both pools
    bare_shuffled = bare_tokens[:]
    rng.shuffle(bare_shuffled)
    spaced_shuffled = spaced_tokens[:]
    rng.shuffle(spaced_shuffled)

    # Each line: 1 bare + 2 spaced tokens
    # bare needed per problem: 200, spaced needed per problem: 400
    n_problems = max(
        math.ceil(len(bare_shuffled) / LINES_PER_PROBLEM),
        math.ceil(len(spaced_shuffled) / (LINES_PER_PROBLEM * 2)),
    )
    print(f"Generating {n_problems} problems")

    bare_idx = 0
    spaced_idx = 0
    problems = []

    for i in range(n_problems):
        # Demo lines for sample input section (random tokens, not from pools)
        sample_input_lines = []
        for _ in range(3):
            b = rng.choice(bare_tokens)
            s1, s2 = rng.choice(spaced_tokens), rng.choice(spaced_tokens)
            inp = f"{b} {s1} {s2}"
            sample_input_lines.append(f"{inp} -> {spell_out(inp)}")

        # Demo lines for sample output section
        sample_output_lines = []
        for _ in range(3):
            b = rng.choice(bare_tokens)
            s1, s2 = rng.choice(spaced_tokens), rng.choice(spaced_tokens)
            inp = f"{b} {s1} {s2}"
            sample_output_lines.append(f"{inp} -> {spell_out(inp)}")

        # 200 test lines, consuming tokens from the pools
        test_inputs = []
        test_answers = []
        for _ in range(LINES_PER_PROBLEM):
            b = bare_shuffled[bare_idx % len(bare_shuffled)]
            bare_idx += 1
            s1 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1
            s2 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1

            inp = f"{b} {s1} {s2}"
            test_inputs.append(inp)
            test_answers.append(f"{inp} -> {spell_out(inp)}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )

        answer = "\n".join(test_answers)
        pid = hashlib.sha256(f"spelling_{i}".encode()).hexdigest()[:8]
        problems.append({"id": pid, "prompt": prompt, "answer": answer, "category": "spelling"})

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "category"])
        writer.writeheader()
        writer.writerows(problems)

    # Verify coverage
    bare_covered = min(bare_idx, len(bare_shuffled))
    spaced_covered = min(spaced_idx, len(spaced_shuffled))
    print(f"Wrote {len(problems)} problems to {OUTPUT_PATH}")
    print(f"Bare tokens used: {bare_idx} (pool size {len(bare_shuffled)}, covered all: {bare_idx >= len(bare_shuffled)})")
    print(f"Spaced tokens used: {spaced_idx} (pool size {len(spaced_shuffled)}, covered all: {spaced_idx >= len(spaced_shuffled)})")


if __name__ == "__main__":
    main()
