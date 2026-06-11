"""Gap diagnosis: pids the measure solver (anchor_measure) solves but the
narrator (crypt_anchor) does not. Classifies each by the narrator's abstain
reason + sub-checks for the 4 known behavioral divergences:
  - combined-OVER: both-modes-in-one-list blows LIST_CAP where per-mode fits
  - zfill: measure admits zero-padded results, narrator rejects them
  - wide-drop: narrator drops >3-unknown branches the measure enumerates
  - pick: multi-answer pick rule (lex-smallest vs first-in-DFS-order)
"""
import json
import sys
import multiprocessing as mp
from collections import Counter


def _init():
    sys.path.insert(0, 'src')
    sys.path.insert(0, 'crypt_redesign_proto')


def work(r):
    sys.path.insert(0, 'src')
    sys.path.insert(0, 'crypt_redesign_proto')
    import anchor_measure as M
    import reasoners.crypt_anchor as A
    from reasoners.store_types import Problem, Example

    mres = M.work(dict(r))
    mc = bool(mres.get("correct"))

    pr = Problem(id=r['id'], category='cryptarithm_deduce',
                 examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                 question=r['question'], answer=r['answer'])
    cot = A.reasoning_cryptarithm_anchor(pr)
    if cot is None:
        nst = 'abstain'
    else:
        i = cot.rfind('boxed{'); j = cot.rfind('}')
        nst = 'ok' if (0 <= i < j and cot[i+6:j] == r['answer']) else 'miss'

    if not mc or nst == 'ok':
        return (r['id'], mc, nst, None)

    # gap pid: classify
    reason = A.LAST.get("reason") or ("miss" if nst == 'miss' else "ret-none")
    detail = ""
    if A.LAST.get("wide"):
        detail += f" wide={A.LAST['wide']}"
    if reason == "list-over":
        # per-mode sizes under a huge cap
        prob = {"id": r["id"],
                "examples": [{"input_value": e["input_value"],
                              "output_value": e["output_value"]} for e in r["examples"]],
                "question": r["question"]}
        exs = [(e["input_value"], e["output_value"]) for e in prob["examples"]]
        opchars = {i2[2] for i2, _ in exs} | {r["question"][2]}
        # recompute char_ops + anchor like the narrator
        from collections import defaultdict
        by_char = defaultdict(list)
        for k, (inp, out) in enumerate(exs):
            by_char[inp[2]].append((k, inp, out))
        char_ops = {}
        for c in by_char:
            cand = None
            for _, inp, out in by_char[c]:
                fo = set(A.fact_ops(inp, out, opchars))
                cand = fo if cand is None else cand & fo
            char_ops[c] = cand or set()
        best = None
        for k, (inp, out) in enumerate(exs):
            ops = char_ops.get(inp[2], set())
            body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
            mul_ok = any(o.startswith("mul") for o in ops)
            score = 3 if (mul_ok and len(body) == 4) else 2 if (mul_ok and len(body) == 3) else None
            if score is not None and (best is None or score > best[0]):
                best = (score, k)
        if best:
            ainp, aout = exs[best[1]]
            aops = sorted((o for o in char_ops[ainp[2]] if o.startswith("mul")),
                          key=A.OP_PRIOR.index)
            old = A.LIST_CAP
            A.LIST_CAP = 10000
            full = A.anchor_candidates(ainp, aout, aops, opchars)
            A.LIST_CAP = old
            if isinstance(full, list):
                ns = sum(1 for row in full if row[0] == "standard")
                nr = len(full) - ns
                detail += f" std={ns} rev={nr} solvedmode={mres.get('mode')}"
    if reason == "miss":
        detail += f" boxed!={r['answer']} measure={mres.get('answer')} anchor={mres.get('anchor')}"
    if reason == "dfs-noanswer":
        detail += f" m-anchor={mres.get('anchor')} m-nodes={mres.get('nodes')}"
    return (r['id'], mc, nst, reason + "|" + detail.strip())


if __name__ == '__main__':
    rows = [json.loads(l) for l in open('crypt_real_all.jsonl')
            if json.loads(l)['category'] == 'cryptarithm_deduce']
    gap = []
    stat = Counter()
    with mp.Pool(10, initializer=_init) as pool:
        for pid, mc, nst, info in pool.imap_unordered(work, rows, chunksize=8):
            stat[(mc, nst)] += 1
            if info:
                gap.append((pid, nst, info))
    print("status (measure-correct, narrator):", dict(stat))
    print(f"\nGAP {len(gap)} pids:")
    rc = Counter(i.split('|')[0] for _, _, i in gap)
    print("by reason:", dict(rc))
    for pid, nst, info in sorted(gap, key=lambda t: t[2]):
        print(f"  {pid} [{nst}] {info}")
