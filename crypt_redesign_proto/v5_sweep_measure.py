"""v5 render verification + candidate-list design measurement (eden, short).

Part 1 - sweep all 659 deduce pids with the v5 narrator:
  ok/miss/abstain, decoy invariance (full double sweep), real-token stats,
  verdict shapes. Expect ok == the same 160 as v4 (render-only changes).
Part 2 - enumeration feasibility for every arithmetic-ok pid's anchor fact:
  k_res        distinct result symbols
  n_res_assign P(10, k_res)  (raw result-assignment scan size)
  op_free      distinct operand symbols NOT pinned by the result assignment
  scan_checks  n_res_assign * 10^op_free  (derivable-scan visit count)
  list_size    solver candidate rows
Part 3 - pattern coverage: canonical anchor pattern per real arith pid vs the
  2200-row synth file (exposure histogram) - the memorize-at-scale input.
"""
import json
import sys
import multiprocessing as mp
from collections import Counter, defaultdict

B = '/mnt/evafs/groups/re-com/mgromadzki'


def _init():
    sys.path.insert(0, 'src')


def canon_pattern(inp, out, opchars):
    pat = {}
    atoms = "УХЦЧШЩЭЮ"
    for c in list(inp[0] + inp[1] + inp[3] + inp[4]) + list(out):
        if c not in opchars and c not in pat:
            pat[c] = atoms[len(pat)] if len(pat) < len(atoms) else '?'
    return ("".join(pat.get(c, c) for c in (inp[0], inp[1])) + "?" +
            "".join(pat.get(c, c) for c in (inp[3], inp[4])) + "=" +
            "".join(pat[c] for c in out))


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.crypt_anchor as A
    from reasoners.store_types import Problem, Example

    def mk(ans):
        return Problem(id=r['id'], category='cryptarithm_deduce',
                       examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                       question=r['question'], answer=ans)
    try:
        cot = A.reasoning_cryptarithm_anchor(mk(r['answer']))
        cot_d = A.reasoning_cryptarithm_anchor(mk('DECOY'))
    except Exception as e:
        return {'pid': r['id'], 'st': 'ERR:' + type(e).__name__}
    if cot is None:
        return {'pid': r['id'], 'st': 'abstain', 'decoy_same': cot_d is None}
    # rfind both ways: gold answers may contain literal { } glyphs
    i = cot.rfind('boxed{')
    j = cot.rfind('}')
    boxed = cot[i + 6:j] if 0 <= i < j else ''
    st = 'ok' if boxed == r['answer'] else 'miss'
    out = {'pid': r['id'], 'st': st, 'decoy_same': cot == cot_d,
           'tok': A.tok_count(cot),
           'tail': ('concat' if 'concat check' in cot and 'no digit code needed' in cot
                    else 'swap' if 'try swap' in cot else 'std')}
    # part 2/3 for arithmetic pids: anchor-fact geometry
    if out['tail'] != 'concat' and st == 'ok':
        exs = [(e['input_value'], e['output_value']) for e in r['examples']]
        opchars = {i[2] for i, _ in exs} | {r['question'][2]}
        # recompute anchor selection like the narrator
        char_ops = {}
        bych = defaultdict(list)
        for inp, o in exs:
            bych[inp[2]].append((inp, o))
        for c, facts in bych.items():
            cand = None
            for inp, o in facts:
                fo = set(A.fact_ops(inp, o, opchars))
                cand = fo if cand is None else cand & fo
            char_ops[c] = cand or set()
        best = None
        for k, (inp, o) in enumerate(exs):
            ops = char_ops.get(inp[2], set())
            body = o[1:] if (len(o) > 1 and o[0] == inp[2]) else o
            mul_ok = any(x.startswith('mul') for x in ops)
            score = 3 if (mul_ok and len(body) == 4) else 2 if (mul_ok and len(body) == 3) else None
            if score is not None and (best is None or score > best[0]):
                best = (score, k)
        if best:
            ainp, aout = exs[best[1]]
            res_syms = list(dict.fromkeys(aout))
            op_syms = list(dict.fromkeys(ainp[0] + ainp[1] + ainp[3] + ainp[4]))
            k_res = len(res_syms)
            op_free = len([s for s in op_syms if s not in res_syms])
            n_assign = 1
            for i in range(k_res):
                n_assign *= (10 - i)
            out.update({'k_res': k_res, 'op_free': op_free,
                        'n_res_assign': n_assign,
                        'scan_checks': n_assign * (10 ** op_free),
                        'pattern': canon_pattern(ainp, aout, opchars)})
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl', encoding='utf-8')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    res = []
    with mp.Pool(16, initializer=_init) as pool:
        for o in pool.imap_unordered(work, rows, chunksize=8):
            res.append(o)
    st = Counter(o['st'] for o in res)
    tails = Counter(o.get('tail') for o in res if o['st'] == 'ok')
    bad_decoy = [o['pid'] for o in res if o.get('decoy_same') is False]
    print('STATUS', dict(st), 'tails(ok)', dict(tails))
    print('decoy-divergent:', len(bad_decoy), bad_decoy[:8])
    toks = sorted(o['tok'] for o in res if o['st'] == 'ok')
    if toks:
        m = len(toks)
        print(f'ok real-tok p50={toks[m//2]} p90={toks[int(m*.9)]} max={toks[-1]}')

    arith = [o for o in res if o['st'] == 'ok' and 'k_res' in o]
    print(f'\n=== enumeration feasibility ({len(arith)} arithmetic-ok anchors) ===')
    for key in ('k_res', 'op_free'):
        print(key, dict(sorted(Counter(o[key] for o in arith).items())))
    sc = sorted(o['scan_checks'] for o in arith)
    m = len(sc)
    print(f'scan_checks p10={sc[m//10]} p50={sc[m//2]} p90={sc[int(m*.9)]} max={sc[-1]}')
    for cap in (100, 200, 500, 1000, 5000):
        print(f'  anchors with scan_checks <= {cap}: {sum(1 for x in sc if x <= cap)}/{m}')

    pats = Counter(o['pattern'] for o in arith)
    print(f'\n=== pattern coverage ===')
    print(f'distinct real anchor patterns: {len(pats)} over {len(arith)} pids; '
          f'top: {pats.most_common(5)}')
    syn_pats = Counter()
    try:
        for l in open(B + '/synth_crypt_anchor.jsonl', encoding='utf-8'):
            s = json.loads(l)
            exs = [(e['input_value'], e['output_value']) for e in s['examples']]
            opchars = {i[2] for i, _ in exs} | {s['question'][2]}
            ops = s['_meta']['ops']
            anchor_oc = next(oc for oc, op in ops.items() if op.startswith('mul'))
            af = next(((i, o) for i, o in exs if i[2] == anchor_oc and len(o) == 4), None)
            if af:
                syn_pats[canon_pattern(af[0], af[1], opchars)] += 1
    except FileNotFoundError:
        print('synth file missing - skip')
    if syn_pats:
        hit = sum(1 for p in pats if p in syn_pats)
        exposure = [syn_pats.get(p, 0) for p in pats]
        exposure.sort()
        print(f'real patterns covered by synth: {hit}/{len(pats)}; '
              f'per-real-pattern synth exposure p50={exposure[len(exposure)//2]} '
              f'max={exposure[-1]}; zero-exposure={sum(1 for e in exposure if e == 0)}')
    json.dump(res, open('crypt_redesign_proto/v5_measure.json', 'w'))
    print('V5_MEASURE_DONE')
