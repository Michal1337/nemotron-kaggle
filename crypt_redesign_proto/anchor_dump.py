"""Dump sample v4 CoTs: one swap-mode (S-branch) solve, one with add-anchor
rows, one plain standard mul. Also decoy invariance over all emitting pids."""
import json
import sys
import multiprocessing as mp


def work(r):
    sys.path.insert(0, 'src')
    import reasoners.crypt_anchor as A
    from reasoners.store_types import Problem, Example

    def mk(ans):
        return Problem(id=r['id'], category='cryptarithm_deduce',
                       examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                       question=r['question'], answer=ans)
    c1 = A.reasoning_cryptarithm_anchor(mk(r['answer']))
    c2 = A.reasoning_cryptarithm_anchor(mk('DECOY'))
    if c1 is None:
        return None
    i = c1.rfind('boxed{'); j = c1.rfind('}')
    ok = 0 <= i < j and c1[i+6:j] == r['answer']
    return {"pid": r['id'], "ok": ok, "decoy_same": c1 == c2,
            "swap": "\nS0>" in c1 or ": pattern" in c1 and "swap (operands" in c1,
            "addrow": ("+" in line.split("->")[0] if False else any(
                ln.lstrip().startswith(tuple("0123456789")) and "+" in ln.split("->")[0]
                for ln in c1.splitlines() if "->R" in ln or "->S" in ln)),
            "cot": c1}


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
    bad_decoy = [o['pid'] for o in res if not o['decoy_same']]
    print(f"emissions {len(res)}  decoy-identical {len(res) - len(bad_decoy)}/{len(res)}")
    if bad_decoy:
        print("DECOY-SENSITIVE:", bad_decoy[:20])
    oks = [o for o in res if o['ok']]
    swap = next((o for o in oks if o['swap']), None)
    add = next((o for o in oks if o['addrow']), None)
    plain = next((o for o in oks if not o['swap'] and not o['addrow']), None)
    for name, o in (("SWAP", swap), ("ADDROW", add), ("PLAIN", plain)):
        if o:
            open(f'crypt_redesign_proto/sample_v4_{name.lower()}_{o["pid"]}.txt',
                 'w', encoding='utf-8').write(o['cot'])
            print(f"{name}: {o['pid']} chars={len(o['cot'])} -> sample file written")
