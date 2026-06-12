"""Standalone cryptarithm_guess dataset (eden, SLURM short).
All narratable real (crypt_guess_fam narrator) + crypt_guess_synth, source-
separable val: real 80/20 (md5%5) + 200 held-out synth (new ciphers, id-disjoint).
The family table is embedded per CoT, so held-out synth = cipher-recovery
generalization (the only thing that must transfer to the private set).
"""
import json, csv, os, sys, hashlib
import multiprocessing as mp
from collections import Counter
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
OUT = B + '/nemotron-master/training/sft/crypt-guess-standalone'
SYNTH = B + '/crypt_guess_synth.jsonl'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
VAL_SYNTH = 200
_tok = _nar = None


def _init():
    global _tok, _nar
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.crypt_guess_fam as CGF
    _nar = CGF.reasoning_cryptarithm_guess_fam


def work(task):
    from reasoners.store_types import Problem, Example
    src, pid, prompt, exs, ans, q = task
    prob = Problem(id=pid, category='cryptarithm_guess',
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _nar(prob)
    except Exception:
        return None
    if cot is None:
        return None
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    if not (0 <= i < j) or cot[i + 6:j] != ans:
        return None
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion)
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return ('toolong', src, pid)
    return (src, pid, prompt, ans, {'tokens': pids_ + cids,
            'mask': [0] * len(pids_) + [1] * len(cids), 'nloss': len(cids)})


def is_val(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 5 == 0


if __name__ == '__main__':
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    reals = sorted(p for p, c in meta.items() if c == 'cryptarithm_guess' and p in trcsv)
    tasks = []
    for pid in reals:
        d = json.load(open(f'{PB}/{pid}.jsonl'))
        prompt, ans = trcsv[pid]
        tasks.append(('real', pid, prompt, d['examples'], ans, str(d['question'])))
    synth = sorted((json.loads(l) for l in open(SYNTH, encoding='utf-8')), key=lambda s: s['id'])
    val_synth = synth[:VAL_SYNTH]
    for s in synth[VAL_SYNTH:]:
        tasks.append(('synth', s['id'], s['prompt'], s['examples'], s['answer'], s['question']))

    os.makedirs(OUT + '/tokens', exist_ok=True)
    os.makedirs(OUT + '/logprobs', exist_ok=True)
    train_rows, val, tl = [], [], 0
    kept = Counter()
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            if r[0] == 'toolong':
                tl += 1
                continue
            src, pid, prompt, ans, tk = r
            if src == 'real' and is_val(pid):
                val.append({'id': pid, 'category': 'cryptarithm_guess', 'source': 'real',
                            'prompt': prompt, 'answer': ans})
                kept['real:val'] += 1
                continue
            train_rows.append((pid, tk))
            kept[f'{src}:train'] += 1
    # held-out synth -> val (problem rows only, eval narrates)
    for s in val_synth:
        val.append({'id': s['id'], 'category': 'cryptarithm_guess', 'source': 'synth',
                    'prompt': s['prompt'], 'answer': s['answer']})

    train_rows.sort(key=lambda t: t[0])
    val.sort(key=lambda v: (v['source'], v['id']))
    recs = []
    for pid, tk in train_rows:
        os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
        json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                  open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
        recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                     'category': 'cryptarithm_guess', 'num_loss_tokens': tk['nloss'],
                     'total_loss': 0.0, 'min_logprob': 0.0})
    import random
    random.Random(0).shuffle(recs)
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')
    assert train_rows and val
    print('kept:', dict(kept))
    print(f'train={len(recs)} val={len(val)} toolong={tl}')
    vc = Counter(v['source'] for v in val)
    print('val by source:', dict(vc))
    print('OUT', OUT)
    print('CRYPT_GUESS_BUILD_DONE')
