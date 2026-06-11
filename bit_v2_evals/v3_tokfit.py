"""Exact-token budget audit of v3 CoTs: how many of the 1454 narrator-ok reals
(and a synth sample) exceed the 7680 completion cap, under three check-line
variants:
  full    - current render (term words on every example-check line)
  compact - check lines drop term words: "inp -> res vs out yes"
  first6  - full term words on the first 6 example checks, the rest compact
Synth from synth_bit_zyx.jsonl problems (re-narrated).
"""
import json
import re
import sys
import multiprocessing as mp

CHECK_RE = re.compile(r"^([01]{8}) (?:\S+=[01]{8} ?)+-> ([01]{8} vs [01]{8} (?:yes|no))$")


def variants(cot):
    lines = cot.split("\n")
    comp, f6 = [], []
    seen_checks = 0
    in_check = False
    for ln in lines:
        if ln == "Check on examples":
            in_check = True
            seen_checks = 0
            comp.append(ln)
            f6.append(ln)
            continue
        m = CHECK_RE.match(ln) if in_check else None
        if m:
            seen_checks += 1
            short = f"{m.group(1)} -> {m.group(2)}"
            comp.append(short)
            f6.append(ln if seen_checks <= 6 else short)
            continue
        if in_check and ln.startswith("Apply to"):
            in_check = False
        comp.append(ln)
        f6.append(ln)
    return "\n".join(comp), "\n".join(f6)


def _init():
    sys.path.insert(0, 'src')


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.bit_manipulation_zyx as Z
    from reasoners.store_types import Problem, Example
    from tokenizers import Tokenizer
    global _TOK
    try:
        _TOK
    except NameError:
        _TOK = Tokenizer.from_file('runs/baseline/tokenizer.json')
    p = Problem(id=r['id'], category='bit_manipulation',
                examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                question=r['question'], answer=r['answer'])
    try:
        cot = Z.reasoning_bit_manipulation(p)
    except Exception:
        return None
    if cot is None:
        return None
    boxed = cot.rsplit('boxed{', 1)[-1].split('}')[0]
    if boxed != r['answer']:
        return None
    comp, f6 = variants(cot)
    out = []
    for tag, c in (('full', cot), ('compact', comp), ('first6', f6)):
        completion = f"{c.rstrip()}\n</think>\n\\boxed{{{r['answer']}}}<|im_end|>"
        n = len(_TOK.encode(completion).ids) + 4
        out.append((tag, n))
    return (r['id'], r.get('_src', 'real'), out)


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('bit_real_all.jsonl', encoding='utf-8')]
    synth = [json.loads(l) for l in open('synth_bit_zyx_local.jsonl', encoding='utf-8')] \
        if False else []
    from collections import Counter, defaultdict
    over = Counter()
    dist = defaultdict(list)
    n_ok = 0
    with mp.Pool(10, initializer=_init) as pool:
        for r in pool.imap_unordered(work, rows, chunksize=16):
            if r is None:
                continue
            n_ok += 1
            for tag, n in r[2]:
                dist[tag].append(n)
                if n > 7680:
                    over[tag] += 1
    print('narrator-ok reals:', n_ok)
    for tag in ('full', 'compact', 'first6'):
        d = sorted(dist[tag])
        m = len(d)
        print(f"{tag:8s} over-cap {over.get(tag, 0):4d}   tok p50={d[m//2]} p90={d[int(m*.9)]} p99={d[int(m*.99)]} max={d[-1]}")
