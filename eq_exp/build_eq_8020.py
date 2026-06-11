"""eq-only dataset: equation_numeric_deduce + equation_numeric_guess, real
problems narrated by the production eq narrator, 80/20 train/val split.

Narrator: reasoners.equation_numeric.reasoning_equation_numeric(prob, None,
None) - self-filtering (returns None unless its boxed answer == gold; gold is
consumed only by that keep/drop filter). Honest (decoy 732/732, audit
2026-06-10); coverage ~653/732.

Split: md5(pid) %% 5 == 0 -> val (20%); applied per category so both deduce and
guess are ~80/20. \\boxed escape correct; deterministic ordering.

Output: crypt-style layout tokens/<pid>/synthetic.json + logprobs/index.jsonl
+ val.jsonl (with category + source for source-separable --val-jsonl eval).
"""
import json, csv, os, sys, hashlib
import multiprocessing as mp
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
OUT = B + '/nemotron-master/training/sft/eq-deduce-guess-8020'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
CATS = {'equation_numeric_deduce', 'equation_numeric_guess'}
_tok = _nar = None


def _init():
    global _tok, _nar
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.equation_numeric as EQ
    _nar = EQ.reasoning_equation_numeric


def _tok_cot(prompt, cot, ans):
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion), "control char in completion"
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return None
    return {'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids),
            'nloss': len(cids)}


def work(task):
    from reasoners.store_types import Problem, Example
    pid, cat, prompt, exs, ans, q = task
    prob = Problem(id=pid, category=cat,
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _nar(prob, None, None)
    except Exception:
        return None
    if cot is None:
        return None
    b = cot.rsplit('boxed{', 1)[-1].split('}')[0]
    if b != ans:
        return None
    tk = _tok_cot(prompt, cot, ans)
    if tk is None:
        return ('toolong', pid)
    return (pid, cat, prompt, ans, tk)


def is_val(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 5 == 0


if __name__ == '__main__':
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    pids = sorted(p for p, c in meta.items() if c in CATS and p in trcsv)
    tasks = []
    for pid in pids:
        d = json.load(open(f'{PB}/{pid}.jsonl'))
        prompt, ans = trcsv[pid]
        tasks.append((pid, meta[pid], prompt, d['examples'], ans, str(d['question'])))

    os.makedirs(OUT + '/tokens', exist_ok=True)
    os.makedirs(OUT + '/logprobs', exist_ok=True)
    train_rows, val, tl = [], [], 0
    from collections import Counter
    kept = Counter()
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            if r[0] == 'toolong':
                tl += 1
                continue
            pid, cat, prompt, ans, tk = r
            if is_val(pid):
                val.append({'id': pid, 'category': cat, 'source': 'real',
                            'prompt': prompt, 'answer': ans})
                kept[f'val:{cat}'] += 1
                continue
            train_rows.append((pid, cat, tk))
            kept[f'train:{cat}'] += 1

    train_rows.sort(key=lambda t: t[0])
    val.sort(key=lambda v: v['id'])
    recs = []
    for pid, cat, tk in train_rows:
        os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
        json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                  open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
        recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                     'category': cat, 'num_loss_tokens': tk['nloss'],
                     'total_loss': 0.0, 'min_logprob': 0.0})
    import random
    random.Random(0).shuffle(recs)
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')

    # post-build tail check
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d = json.load(open(f"{OUT}/tokens/{recs[0]['problem_id']}/synthetic.json"))
    tail = tok.decode(d['tokens'][-40:])
    assert '</think>' in tail and '\\boxed{' in tail.split('</think>')[-1], repr(tail)
    print('tail check OK:', repr(tail[-60:]))
    print('kept:', dict(kept))
    print(f'train={len(recs)} val={len(val)} toolong={tl}')
    print('OUT', OUT)
    print('EQ_BUILD_DONE')
