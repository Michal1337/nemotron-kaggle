"""Summarize a train_logprobs.jsonl (min-logprob metric): the SFT objective is
to lift the MINIMUM per-token logprob; tokens below -ln2 (p<0.5) are unlikely
to reproduce at sampling. Reports the distribution of min_logprob and the
worst-token contexts, split real vs synth, so we can see which traces (and
which token positions) the model still can't reproduce.

Usage: python analyze_logprobs.py <train_logprobs.jsonl> [--worst N]
"""
import json
import sys
from collections import Counter

LN2 = 0.6931471805599453


def main():
    path = sys.argv[1]
    show = int(sys.argv[sys.argv.index('--worst') + 1]) if '--worst' in sys.argv else 20
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    real = [r for r in rows if not r['pid'].startswith('syn')]
    synth = [r for r in rows if r['pid'].startswith('syn')]
    print(f"traces {len(rows)}  (real {len(real)}, synth {len(synth)})")
    for name, grp in (('ALL', rows), ('real', real), ('synth', synth)):
        if not grp:
            continue
        mins = sorted(r['min_logprob'] for r in grp)
        n = len(mins)
        below = [r for r in grp if r['min_logprob'] < -LN2]
        frac_tok = sum(r['n_below_ln2'] for r in grp) / max(1, sum(r['num_loss_tokens'] for r in grp))
        print(f"\n[{name}] min_logprob p10={mins[n//10]:.3f} p50={mins[n//2]:.3f} "
              f"p90={mins[int(n*.9)]:.3f} worst={mins[0]:.3f}")
        print(f"  traces with a token below -ln2: {len(below)}/{n} ({len(below)/n:.1%})")
        print(f"  fraction of ALL loss tokens below -ln2: {frac_tok:.4%}")
    # worst traces overall
    print(f"\n=== {show} worst traces (lowest min_logprob) ===")
    for r in sorted(rows, key=lambda r: r['min_logprob'])[:show]:
        w = r['worst'][0]
        print(f"  {r['pid']} min={r['min_logprob']:.2f} below-ln2={r['n_below_ln2']}/{r['num_loss_tokens']}"
              f"  worst tok={w['tok']!r} @ {w['ctx']!r}")
    # which token strings are worst, aggregated
    print(f"\n=== most common worst-tokens (top of each trace's worst list) ===")
    c = Counter(r['worst'][0]['tok'] for r in rows if r['worst'])
    for tok, n in c.most_common(15):
        print(f"  {n:5d}  {tok!r}")


if __name__ == '__main__':
    main()
