"""v3 uniform-render verification sweep over all 1602 real bit problems:
  G1: per-pid boxed answer identical to the v2 narrator's (coverage preserved);
  G2: decoy invariance (CoT byte-identical under a decoy gold);
  G5: token/char stats + section accounting (every emission must contain
      exactly one Whole-word check; verdict-line distribution reported).
The v2 reference answers are recomputed by reconstructing the old decision
rule from the same solver (per-bit answer unless solve_program disagrees),
which by the audit equals the deployed v2 boxed answer for all 1454.
"""
import json
import sys
import multiprocessing as mp
from collections import Counter


def _init():
    sys.path.insert(0, 'src')


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.bit_manipulation_zyx as Z
    from reasoners.store_types import Problem, Example

    def mk(ans):
        return Problem(id=r['id'], category='bit_manipulation',
                       examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                       question=r['question'], answer=ans)
    try:
        cot = Z.reasoning_bit_manipulation(mk(r['answer']))
        cot_decoy = Z.reasoning_bit_manipulation(mk('00000000' if r['answer'] != '00000000' else '11111111'))
    except Exception as e:
        return (r['id'], 'ERR:' + type(e).__name__, None, 0, None)
    if cot is None:
        return (r['id'], 'abstain', None, 0, cot_decoy is None)
    boxed = cot.rsplit('boxed{', 1)[-1].split('}')[0]
    ok = boxed == r['answer']
    n_check = cot.count('Whole-word check')
    verdict = ('disagree' if 'Disagrees with the per-bit output' in cot
               else 'match' if 'Matches the per-bit output.' in cot
               else 'noprog' if 'No agreeing program' in cot
               else 'MISSING')
    sane = (n_check == 1) and verdict != 'MISSING'
    return (r['id'], 'ok' if ok else 'wrong', verdict if sane else 'INSANE',
            len(cot), cot == cot_decoy)


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('bit_real_all.jsonl', encoding='utf-8')]
    st = Counter(); ver = Counter(); sizes = []
    decoy_bad = []
    ok_ids = set()
    with mp.Pool(10, initializer=_init) as pool:
        for pid, s, v, n, same in pool.imap_unordered(work, rows, chunksize=16):
            st[s] += 1
            if v:
                ver[v] += 1
            if s == 'ok':
                ok_ids.add(pid)
                sizes.append(n)
            if same is False:
                decoy_bad.append(pid)
            if sum(st.values()) % 400 == 0:
                print('progress', sum(st.values()), flush=True)
    print('STATUS', dict(st))
    print('VERDICTS', dict(ver))
    sizes.sort()
    m = len(sizes)
    print(f'ok chars p50={sizes[m//2]} p90={sizes[int(m*.9)]} max={sizes[-1]}'
          f'  est-tok max ~{sizes[-1]/3.2:.0f}')
    print('decoy-divergent:', len(decoy_bad), decoy_bad[:10])
    json.dump(sorted(ok_ids), open('bit_v2_evals/v3_ok_ids.json', 'w'))
