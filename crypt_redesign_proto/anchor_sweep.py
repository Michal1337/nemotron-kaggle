import json, sys
import multiprocessing as mp

def _init():
    sys.path.insert(0, 'src')

def work(r):
    import sys
    sys.path.insert(0, 'src')
    from reasoners.crypt_anchor import reasoning_cryptarithm_anchor as narrate
    from reasoners.store_types import Problem, Example
    pr = Problem(id=r['id'], category='cryptarithm_deduce',
                 examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                 question=r['question'], answer=r['answer'])
    try:
        c = narrate(pr)
    except Exception as e:
        return (r['id'], 'ERR:' + type(e).__name__, 0)
    if c is None:
        return (r['id'], 'abstain', 0)
    i = c.rfind('boxed{'); j = c.rfind('}')
    good = 0 <= i < j and c[i+6:j] == r['answer']
    from reasoners.crypt_anchor import tok_count
    return (r['id'], 'ok' if good else 'miss', tok_count(c))

if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    from collections import Counter
    out = Counter(); sizes = []
    with mp.Pool(10, initializer=_init) as pool:
        for pid, st, n in pool.imap_unordered(work, rows, chunksize=8):
            out[st] += 1
            if st == 'ok':
                sizes.append(n)
            if sum(out.values()) % 100 == 0:
                print('progress', sum(out.values()), flush=True)
    print('FINAL', dict(out))
    sizes.sort()
    if sizes:
        n = len(sizes)
        print(f"ok real-tok(+100 overhead) p50={100+sizes[n//2]} p90={100+sizes[int(n*.9)]} max={100+sizes[-1]}")
