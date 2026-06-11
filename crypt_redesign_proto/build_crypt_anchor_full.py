"""crypt-anchor-v5-full dataset build (eden, SLURM short).

v5 narrator over ALL 659 real cryptarithm_deduce + the v5 synth file
(pattern-targeted + diverse tail). Val = the SAME md5%2 pids as the pilot
(comparability with the 6ep pilot eval is the whole point); train = remaining
narrator-ok reals + every synth row that re-narrates gold-correct.
Self-checks as in the pilot build; rfind boxed extraction ({}-bearing golds);
deterministic selection; exact-tokenizer governor enforced via env assert.
"""
import json, csv, os, sys, hashlib
import multiprocessing as mp
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
OUT = B + '/nemotron-master/training/sft/crypt-anchor-v5-full'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
SYNTH = B + '/synth_crypt_anchor_v5.jsonl'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
_tok = _nar = None


def _init():
    global _tok, _nar
    os.environ.setdefault('NEMOTRON_TOKENIZER_JSON', MODEL + '/tokenizer.json')
    assert os.path.isfile(os.environ['NEMOTRON_TOKENIZER_JSON']), \
        os.environ['NEMOTRON_TOKENIZER_JSON']
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.crypt_anchor as A
    _nar = A.reasoning_cryptarithm_anchor


def work(task):
    from reasoners.store_types import Problem, Example
    src, pid, prompt, exs, ans, q = task
    prob = Problem(id=pid, category='cryptarithm_deduce',
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _nar(prob)
    except Exception:
        return None
    if cot is None:
        return None
    i = cot.rfind('boxed{')
    j = cot.rfind('}')
    if not (0 <= i < j) or cot[i + 6:j] != ans:
        return None
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion), f"control char {pid}"
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return ('toolong', src, pid)
    tail = ('concat' if 'no digit code needed' in cot
            else 'swap' if 'try swap' in cot else 'std')
    return (src, pid, prompt, ans,
            {'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids),
             'nloss': len(cids)}, tail)


def is_val(pid):  # SAME split as the anchor pilot (md5 % 2)
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2 == 0


if __name__ == '__main__':
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    real = sorted(p for p, c in meta.items()
                  if c == 'cryptarithm_deduce' and p in trcsv)
    tasks = []
    for pid in real:
        d = json.load(open(B + f'/nemotron-master/problems/{pid}.jsonl'))
        prompt, ans = trcsv[pid]
        tasks.append(('real', pid, prompt, d['examples'], ans, str(d['question'])))
    for s in (json.loads(l) for l in open(SYNTH, encoding='utf-8')):
        kind = 'synthpat' if s['id'].startswith('synp_') else 'synth'
        tasks.append((kind, s['id'], s['prompt'], s['examples'], s['answer'], s['question']))

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
            src, pid, prompt, ans, tk, tail = r
            if src == 'real' and is_val(pid):
                val.append({'id': pid, 'category': 'cryptarithm_deduce', 'source': 'real',
                            'prompt': prompt, 'answer': ans})
                continue
            kept[f'{src}:{tail}'] += 1
            train_rows.append((pid, tk))

    train_rows.sort(key=lambda t: t[0])
    val.sort(key=lambda v: v['id'])
    recs = []
    for pid, tk in train_rows:
        os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
        json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                  open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
        recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                     'category': 'cryptarithm_deduce', 'num_loss_tokens': tk['nloss'],
                     'total_loss': 0.0, 'min_logprob': 0.0})
    import random
    random.Random(0).shuffle(recs)
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')

    assert train_rows and val, f"empty build: train={len(train_rows)} val={len(val)}"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d = json.load(open(f"{OUT}/tokens/{recs[0]['problem_id']}/synthetic.json"))
    tail = tok.decode(d['tokens'][-60:])
    assert '</think>' in tail and '\\boxed{' in tail.split('</think>')[-1], repr(tail)
    print('tail check OK:', repr(tail[-60:]))
    print('kept:', dict(kept))
    print(f'train={len(recs)} val={len(val)} toolong={tl}')
    print('OUT', OUT)
    print('FULL_BUILD_DONE')
