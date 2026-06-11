"""Write diverse v4 reference CoTs into crypt_cots_v4/: concat, plain-small,
swap-fallback, mul+1/-1 rows, near-budget large."""
import json
import os
import sys
import multiprocessing as mp


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
    if not (0 <= i < j and cot[i + 6:j] == r['answer']):
        return None
    return {"pid": r['id'], "cot": cot, "tok": A.tok_count(cot)}


def _init():
    sys.path.insert(0, 'src')


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    oks = []
    with mp.Pool(10, initializer=_init) as pool:
        for o in pool.imap_unordered(work, rows, chunksize=8):
            if o:
                oks.append(o)
    os.makedirs('crypt_cots_v4', exist_ok=True)

    def first(name, pred):
        o = next((o for o in oks if pred(o['cot'], o)), None)
        if o:
            with open(f"crypt_cots_v4/{name}_{o['pid']}.txt", 'w', encoding='utf-8') as f:
                f.write(o['cot'])
            print(name, o['pid'], f"tok={o['tok']}")

    first('concat', lambda c, o: 'concat check' in c)
    first('plain_small', lambda c, o: 'concat check' not in c and '\nS0>' not in c and o['tok'] < 1200)
    first('swap', lambda c, o: 'try swap' in c and '\nS0>' in c)
    first('skipdup', lambda c, o: 'same state as' in c)
    first('neg_answer', lambda c, o: 'concat check' not in c and '\\boxed{' in c
          and c[c.rfind('boxed{') + 6] == c.splitlines()[-1][0] * 0 + c[c.rfind('boxed{') + 6]
          and False)  # placeholder, skipped
    big = max(oks, key=lambda o: o['tok'])
    with open(f"crypt_cots_v4/near_budget_{big['pid']}.txt", 'w', encoding='utf-8') as f:
        f.write(big['cot'])
    print('near_budget', big['pid'], f"tok={big['tok']}")
