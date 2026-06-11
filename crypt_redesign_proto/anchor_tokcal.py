"""Token calibration: regenerate all v4 CoTs with the governor lifted, tokenize
with the real Nemotron tokenizer (runs/baseline), report the true chars/tok
ratio and which governor-abstained pids actually fit the budget.
Also records, per pid, the answer set (uniq) for the pick-rule comparison.
"""
import json
import sys
import multiprocessing as mp


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.crypt_anchor as A
    from reasoners.store_types import Problem, Example
    A.TOK_CAP = 10 ** 9  # lift governor for measurement
    pr = Problem(id=r['id'], category='cryptarithm_deduce',
                 examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                 question=r['question'], answer=r['answer'])
    cot = A.reasoning_cryptarithm_anchor(pr)
    if cot is None:
        return None
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    boxed = cot[i + 6:j] if 0 <= i < j else ""
    return {"pid": r['id'], "gold": r['answer'], "boxed": boxed,
            "uniq": A.LAST.get("uniq"), "chars": len(cot), "cot": cot}


def _init():
    sys.path.insert(0, 'src')


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    res = []
    with mp.Pool(10, initializer=_init) as pool:
        for out in pool.imap_unordered(work, rows, chunksize=8):
            if out:
                res.append(out)
            if len(res) % 50 == 0:
                print('emitted', len(res), flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file('runs/baseline/tokenizer.json')
    ratios = []
    fit = over = 0
    for o in res:
        n = len(tok.encode(o['cot']).ids)
        o['tok'] = n
        del o['cot']
        ratios.append(o['chars'] / n)
        if n + 100 <= 7300:
            fit += 1
        else:
            over += 1
    ratios.sort()
    m = len(ratios)
    print(f"\nemissions {len(res)}  fit(real tok+100<=7300) {fit}  over {over}")
    print(f"chars/tok p10={ratios[m//10]:.3f} p50={ratios[m//2]:.3f} p90={ratios[int(m*.9)]:.3f} min={ratios[0]:.3f}")
    toks = sorted(o['tok'] for o in res)
    print(f"real tok p50={toks[m//2]} p90={toks[int(m*.9)]} max={toks[-1]}")
    json.dump(res, open('crypt_redesign_proto/anchor_tokcal.json', 'w'))
    # pick-rule comparison over multi-answer pids
    multi = [o for o in res if o.get('uniq') and len(o['uniq']) > 1]
    dfs_first = sum(1 for o in multi if o['boxed'] == o['gold'])
    lex = sum(1 for o in multi if sorted(o['uniq'])[0] == o['gold'])
    print(f"\nmulti-answer pids {len(multi)}: dfs-first correct {dfs_first}, lex-smallest correct {lex}")
