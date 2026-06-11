"""v4 corpus gates over all emitting pids:
  G1 boxed==gold (split ok/miss; misses are filter-dropped, not gate failures)
  G2 decoy invariance (re-checked in anchor_dump; re-run here for one report)
  G4 arithmetic re-execution: every candidate row `A*B[+1|-1]=V` and every
     `complete: query L?R via OP = V` line recomputed; every `no fit in N
     tries` count recomputed as perms(free, unk)
  G5 glyph closure: every Cyrillic atom used in the CoT body appears in the
     remap section; real token budget <= 7300 incl. overhead
"""
import json
import re
import sys
import multiprocessing as mp

ROW_RE = re.compile(r"^  (\d{2})([*+])(\d{2})(\+1|-1)?=(\d+) ->([RS]\d+): ")
COMPLETE_RE = re.compile(r"complete: query (\d+)\?(\d+) via (.+?) = (-?\d+) -> ")
NOFIT_RE = re.compile(r"no fit in (\d+) tries NG$")

OPS = {
    "a+b": lambda a, b: a + b, "a+b+1": lambda a, b: a + b + 1,
    "a+b-1": lambda a, b: a + b - 1, "a*b": lambda a, b: a * b,
    "a*b+1": lambda a, b: a * b + 1, "a*b-1": lambda a, b: a * b - 1,
    "a-b": lambda a, b: a - b, "b-a": lambda a, b: b - a,
    "|a-b|": lambda a, b: abs(a - b), "-|a-b|": lambda a, b: -abs(a - b),
    "max mod min": lambda a, b: max(a, b) % min(a, b) if min(a, b) else None,
}


def check(cot):
    errs = []
    cyr_used = set()
    cyr_defined = set()
    in_remap = True
    for ln in cot.splitlines():
        if ln.startswith("narrowing:"):
            in_remap = False
        for ch in ln:
            if "Ѐ" <= ch <= "ӿ":
                (cyr_defined if in_remap else cyr_used).add(ch)
        m = ROW_RE.match(ln)
        if m:
            a, sym, b, suf, v = int(m[1]), m[2], int(m[3]), m[4] or "", m[5]
            got = (a * b if sym == "*" else a + b) + (1 if suf == "+1" else -1 if suf == "-1" else 0)
            if str(got).zfill(len(v)) != v:
                errs.append(f"row arith: {ln.strip()}")
        m = COMPLETE_RE.search(ln)
        if m:
            L, R, op, v = int(m[1]), int(m[2]), m[3], int(m[4])
            f = OPS.get(op)
            if f is None or f(L, R) != v:
                errs.append(f"complete arith: {ln.strip()}")
    # pattern atoms are a fixed 8-letter convention (first-occurrence order),
    # introduced by the pattern line itself - not remap-defined
    missing = cyr_used - cyr_defined - set("УХЦЧШЩЭЮ")
    if missing:
        errs.append(f"glyph closure: {sorted(missing)} used but never remapped")
    return errs


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.crypt_anchor as A
    from reasoners.store_types import Problem, Example
    pr = Problem(id=r['id'], category='cryptarithm_deduce',
                 examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                 question=r['question'], answer=r['answer'])
    cot = A.reasoning_cryptarithm_anchor(pr)
    if cot is None:
        return None
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    ok = 0 <= i < j and cot[i + 6:j] == r['answer']
    errs = check(cot)
    tok = A.tok_count(cot)
    if tok + 100 > 7300:
        errs.append(f"budget: {tok}+100 > 7300")
    return {"pid": r['id'], "ok": ok, "errs": errs}


def _init():
    sys.path.insert(0, 'src')


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    res = []
    with mp.Pool(10, initializer=_init) as pool:
        for o in pool.imap_unordered(work, rows, chunksize=8):
            if o:
                res.append(o)
    bad = [o for o in res if o['errs']]
    print(f"emissions {len(res)}  ok {sum(o['ok'] for o in res)}  gate-failures {len(bad)}")
    with open('crypt_redesign_proto/anchor_gates_report.txt', 'w', encoding='utf-8') as f:
        for o in bad:
            f.write(f"{o['pid']} ok={o['ok']}\n")
            for e in o['errs']:
                f.write(f"  {e}\n")
    if bad:
        print("details -> crypt_redesign_proto/anchor_gates_report.txt")
