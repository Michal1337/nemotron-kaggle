"""Find synthetic.json files in a corpus whose tokens field is malformed.

Usage:
    python src/scan_bad_synthetic.py <corpus_root>

where <corpus_root> contains tokens/<pid>/synthetic.json files.
Prints one pid per line on stdout for each broken file, plus a summary to stderr.
"""

import json
import os
import sys
from pathlib import Path


def scan(root: Path) -> None:
    tokens_dir = root / "tokens"
    if not tokens_dir.is_dir():
        sys.exit(f"no tokens/ under {root}")
    bad = []
    ok = 0
    for child in tokens_dir.iterdir():
        synth = child / "synthetic.json"
        if not synth.is_file():
            continue
        try:
            with synth.open() as f:
                rec = json.load(f)
        except Exception as e:
            bad.append((child.name, f"json_error:{e}"))
            continue
        toks = rec.get("tokens")
        mask = rec.get("mask")
        if not isinstance(toks, list):
            bad.append((child.name, f"tokens_type:{type(toks).__name__}"))
            continue
        if toks and not isinstance(toks[0], int):
            head_repr = repr(toks[:5])
            bad.append((child.name, f"tokens_elem:{type(toks[0]).__name__} head={head_repr}"))
            continue
        if not isinstance(mask, list):
            bad.append((child.name, f"mask_type:{type(mask).__name__}"))
            continue
        ok += 1
    for pid, reason in bad:
        print(pid)
        print(f"  {pid}: {reason}", file=sys.stderr)
    print(f"\nscanned: ok={ok} bad={len(bad)}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/scan_bad_synthetic.py <corpus_root>")
    scan(Path(sys.argv[1]))
